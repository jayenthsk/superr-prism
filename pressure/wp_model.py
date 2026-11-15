"""Win probability model for both innings, out-of-fold by match and calibrated."""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from common.config import cfg
from common.io import log_run, read_parquet, stage_dir, write_parquet, write_table

COMP_TYPES = ["IPL", "T20I_full", "T20I_associate", "Emerging"]
# Sign of each feature effect on batting side WP, zero means unconstrained
FEATURES = {
    1: {"balls_left": 1, "wkts_before": -1, "score_vs_par": 1, "comp_code": 0, "impact": 0},
    2: {"balls_left": 1, "wkts_before": -1, "runs_needed": -1, "rrr": -1, "comp_code": 0, "impact": 0},
}
EPS = 1e-4


def design(df: pd.DataFrame, inn: int) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in FEATURES[inn]:
        if col == "comp_code":
            X[col] = pd.Categorical(df["comp_type"], categories=COMP_TYPES).codes
        elif col == "impact":
            X[col] = df["ipl_impact_era"].astype(int)
        elif col == "score_vs_par":
            # Raw venue index is constant within a match and lets trees memorise results,
            # so venue enters as a par that scales the score instead
            X[col] = df["score_before"] / df["venue_scoring_index"]
        else:
            X[col] = df[col].astype(float)
    if inn == 2:
        # Chase rate beyond 36 an over is lost anyway, cap so trees do not chase outliers
        X["rrr"] = X["rrr"].clip(upper=36)
    return X


class CentredIsotonic:
    """Isotonic calibration joined through block centres so WP never sits on a flat step."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> CentredIsotonic:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y)
        fitted = iso.predict(p)
        blocks = pd.DataFrame({"p": p, "f": fitted}).groupby("f")["p"].mean()
        self.x_, self.y_ = blocks.values, blocks.index.values
        order = np.argsort(self.x_)
        self.x_, self.y_ = self.x_[order], np.maximum.accumulate(self.y_[order])
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.interp(p, self.x_, self.y_)


class _Booster:
    """LightGBM with monotone constraints so WP always moves the cricket way."""

    def __init__(self, inn: int) -> None:
        w = {**cfg()["wp"], **(cfg()["wp"].get("inn1_overrides", {}) if inn == 1 else {})}
        self.inn = inn
        self.params = dict(objective="binary", learning_rate=w["learning_rate"], num_leaves=w["max_leaf_nodes"],
                           min_child_samples=w["min_samples_leaf"], verbose=-1, seed=cfg()["seed"],
                           deterministic=True, force_row_wise=True, num_threads=w.get("n_jobs", 4),
                           monotone_constraints=list(FEATURES[inn].values()),
                           monotone_constraints_method="advanced")
        self.rounds = w["max_iter"]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _Booster:
        ds = lgb.Dataset(X, y, categorical_feature=["comp_code"], free_raw_data=True)
        self.booster = lgb.train(self.params, ds, num_boost_round=self.rounds)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.booster.predict(X)
        return np.column_stack([1 - p, p])


class Platt:
    """Two-parameter logistic recalibration, robust when balls cluster within matches."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> Platt:
        self.lr_ = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class Identity:
    def fit(self, p: np.ndarray, y: np.ndarray) -> Identity:
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p, EPS, 1 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


CALIBRATORS = {"isotonic": CentredIsotonic, "platt": Platt, "none": Identity}


def make_calibrator(inn: int):
    method = cfg()["wp"]["calibrate"]
    method = method.get(inn, method.get(str(inn))) if isinstance(method, dict) else method
    return CALIBRATORS[method]()


def _hgb(inn: int) -> _Booster:
    return _Booster(inn)


COMBOS = [("IPL", 0), ("IPL", 1), ("T20I_full", 0), ("T20I_associate", 0), ("Emerging", 0)]


def _combo(df: pd.DataFrame) -> np.ndarray:
    imp = df["ipl_impact_era"].astype(bool).to_numpy() & (df["comp_type"] == "IPL").to_numpy()
    key = {c: i for i, c in enumerate(COMBOS)}
    return np.array([key.get((c, int(m)), 0) for c, m in zip(df["comp_type"], imp, strict=True)])


def _third_axis(inn: int) -> np.ndarray:
    g = cfg()["wp"]["grid"]
    return np.arange(0, g["svp_max"] + 1) if inn == 1 else np.arange(1, g["runs_max"] + 1)


def build_surface(booster: _Booster, inn: int) -> np.ndarray:
    """Dense WP surface over wickets, balls left and runs, smoothed along balls and runs.

    Trees are step functions, so a single run can jump a split and fake a big swing. A small
    Gaussian kernel removes those steps and, being positive, keeps every monotone direction.
    """
    third = _third_axis(inn)
    wk, bl, th = np.meshgrid(np.arange(10), np.arange(1, 121), third, indexing="ij")
    grids = []
    for comp, imp in COMBOS:
        X = pd.DataFrame({"balls_left": bl.ravel().astype(float), "wkts_before": wk.ravel().astype(float)})
        if inn == 1:
            X["score_vs_par"] = th.ravel().astype(float)
        else:
            X["runs_needed"] = th.ravel().astype(float)
            X["rrr"] = (X["runs_needed"] * 6 / X["balls_left"]).clip(upper=36)
        X["comp_code"] = COMP_TYPES.index(comp)
        X["impact"] = imp
        X = X[list(FEATURES[inn])]
        g = booster.predict_proba(X)[:, 1].reshape(wk.shape)
        sm = cfg()["wp"]["smooth"]
        grids.append(gaussian_filter(g, sigma=(0, sm["balls"], sm["runs"]), mode="nearest"))
    return np.stack(grids).astype(np.float32)


def lookup(surface: np.ndarray, df: pd.DataFrame, inn: int) -> np.ndarray:
    third = _third_axis(inn)
    c = _combo(df)
    w = df["wkts_before"].clip(0, 9).astype(int).to_numpy()
    b = df["balls_left"].clip(1, 120).astype(int).to_numpy() - 1
    if inn == 1:
        x = (df["score_before"] / df["venue_scoring_index"]).to_numpy(float)
    else:
        x = df["runs_needed"].to_numpy(float)
    x = np.clip(x, third[0], third[-1]) - third[0]
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(third) - 1)
    f = x - lo
    return (1 - f) * surface[c, w, b, lo] + f * surface[c, w, b, hi]


@dataclass
class WPModel:
    """Per-fold WP surfaces for out-of-fold lookups plus one full-data surface for new states."""

    fold_models: dict[int, dict] = field(default_factory=dict)
    full: dict[int, tuple] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame, inn: int, use_folds: bool = True) -> np.ndarray:
        if not use_folds:
            surf, cal = self.full[inn]
            return cal.predict(lookup(surf, df, inn))
        out = np.empty(len(df))
        folds = df["fold"].to_numpy()
        for f, (surf, cal) in self.fold_models[inn].items():
            idx = folds == f
            if idx.any():
                out[idx] = cal.predict(lookup(surf, df[idx], inn))
        return out


def train_set(balls: pd.DataFrame, inn: int) -> pd.DataFrame:
    """Decided non-DLS matches only, ties have no clean winner so they sit out."""
    return balls[balls["in_scope"] & (balls["innings"] == inn) & ~balls["is_tie"]]


def fit_innings(balls: pd.DataFrame, inn: int) -> tuple[dict, tuple, pd.DataFrame]:
    d = train_set(balls, inn)
    X, y, folds = design(d, inn), d["is_bat_team_winner"].astype(int).to_numpy(), d["fold"].to_numpy()
    raw = np.empty(len(d))
    surfaces = {}
    for f in np.unique(folds):
        tr = folds != f
        surfaces[f] = build_surface(_hgb(inn).fit(X[tr], y[tr]), inn)
        raw[~tr] = lookup(surfaces[f], d[~tr], inn)
    # Calibrator for a fold never sees that fold, so OOF stays clean
    fold_models, cal_oof = {}, np.empty(len(d))
    for f in np.unique(folds):
        cal = make_calibrator(inn).fit(raw[folds != f], y[folds != f])
        fold_models[f] = (surfaces[f], cal)
        cal_oof[folds == f] = cal.predict(raw[folds == f])
    full = (build_surface(_hgb(inn).fit(X, y), inn), make_calibrator(inn).fit(raw, y))
    oof = pd.DataFrame({"y": y, "raw": raw, "wp": cal_oof, "fold": folds}, index=d.index)
    return fold_models, full, oof


def calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def metrics(oof: pd.DataFrame, inn: int) -> dict:
    slope, icpt = calibration_slope(oof["y"].to_numpy(), oof["wp"].to_numpy())
    return {"innings": inn, "balls": len(oof), "auc": roc_auc_score(oof["y"], oof["wp"]),
            "brier": brier_score_loss(oof["y"], oof["wp"]), "log_loss": log_loss(oof["y"], oof["wp"]),
            "auc_raw": roc_auc_score(oof["y"], oof["raw"]), "cal_slope": slope, "cal_intercept": icpt}


def calibration_table(oof: pd.DataFrame, inn: int, bins: int = 20) -> pd.DataFrame:
    q = pd.cut(oof["wp"], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = oof.groupby(q, observed=True).agg(pred=("wp", "mean"), obs=("y", "mean"), n=("y", "size"))
    return t.reset_index(drop=True).assign(innings=inn)


def inn2_start_state(df: pd.DataFrame) -> pd.DataFrame:
    s = df[["match_id", "fold", "comp_type", "ipl_impact_era", "venue_scoring_index"]].copy()
    s["balls_left"], s["wkts_before"] = 120, 0
    s["runs_needed"] = df["inn1_final"] + 1
    s["rrr"] = s["runs_needed"] * 6 / 120
    return s


def add_transitions(balls: pd.DataFrame, model: WPModel) -> pd.DataFrame:
    """WP before every in-scope ball, then WP after from the next state, and the swing.

    End of innings 1 hands over to one minus the chase start WP, end of the chase is the result.
    """
    b = balls.copy()
    b["wp_pre"] = np.nan
    for inn in (1, 2):
        m = b["in_scope"] & (b["innings"] == inn)
        b.loc[m, "wp_pre"] = model.predict(b[m], inn)
    s = b[b["in_scope"]]
    nxt = s.groupby(["match_id", "innings"])["wp_pre"].shift(-1)
    last = nxt.isna()
    i1_last = last & (s["innings"] == 1)
    i1 = s[i1_last].assign(inn1_final=s.loc[i1_last, "inn1_total"])
    nxt.loc[i1_last] = 1 - model.predict(inn2_start_state(i1), 2)
    i2_last = last & (s["innings"] == 2)
    res = np.where(s.loc[i2_last, "is_tie"], 0.5, s.loc[i2_last, "is_bat_team_winner"].astype(float))
    nxt.loc[i2_last] = res
    b["wp_post"] = nxt
    b["dwp"] = b["wp_post"] - b["wp_pre"]
    return b


def fit_all(balls: pd.DataFrame) -> tuple[WPModel, pd.DataFrame, pd.DataFrame, dict]:
    model = WPModel()
    mets, cals, oofs = [], [], {}
    for inn in (1, 2):
        fm, full, oof = fit_innings(balls, inn)
        model.fold_models[inn], model.full[inn] = fm, full
        mets.append(metrics(oof, inn))
        oofs[inn] = oof
        cals.append(calibration_table(oof, inn))
    return model, pd.DataFrame(mets), pd.concat(cals, ignore_index=True), oofs


def known_situations(model: WPModel) -> pd.DataFrame:
    rows = [(12, 6, 30), (12, 2, 30), (6, 5, 12), (6, 5, 20), (30, 5, 50), (60, 3, 90),
            (60, 7, 90), (120, 10, 180), (120, 10, 230), (24, 4, 40), (1, 5, 2), (18, 4, 25)]
    df = pd.DataFrame(rows, columns=["balls_left", "wkts_in_hand", "runs_needed"])
    df["wkts_before"] = 10 - df["wkts_in_hand"]
    df["rrr"] = df["runs_needed"] * 6 / df["balls_left"]
    df["ipl_impact_era"] = True
    out = []
    for ct in ["IPL", "T20I_full"]:
        d = df.assign(comp_type=ct)
        d["wp_chasing"] = model.predict(d, 2, use_folds=False)
        out.append(d)
    return pd.concat(out)[["comp_type", "balls_left", "wkts_in_hand", "runs_needed", "rrr", "wp_chasing"]]


def monotonic_checks(model: WPModel) -> pd.DataFrame:
    g = pd.DataFrame([(bl, w, rn) for bl in range(6, 121, 6) for w in range(0, 10) for rn in range(5, 200, 5)],
                     columns=["balls_left", "wkts_before", "runs_needed"])
    g["rrr"] = g["runs_needed"] * 6 / g["balls_left"]
    g["comp_type"], g["ipl_impact_era"] = "IPL", True
    g["wp"] = model.predict(g, 2, use_folds=False)
    rows = []
    for col, sign, keys in [("runs_needed", -1, ["balls_left", "wkts_before"]),
                            ("wkts_before", -1, ["balls_left", "runs_needed"])]:
        d = g.sort_values(keys + [col]).groupby(keys)["wp"].diff()
        rows.append({"feature": col, "violations": float((d * sign < -1e-9).mean())})
    g2 = g.sort_values(["wkts_before", "runs_needed", "balls_left"])
    d = g2.groupby(["wkts_before", "runs_needed"])["wp"].diff()
    rows.append({"feature": "balls_left at fixed need", "violations": float((d < -1e-9).mean())})
    return pd.DataFrame(rows)


def calibrator_comparison(oofs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for inn, oof in oofs.items():
        y, raw, f = oof["y"].to_numpy(), oof["raw"].to_numpy(), oof["fold"].to_numpy()
        for name, cls in CALIBRATORS.items():
            cal = np.empty(len(y))
            for k in np.unique(f):
                cal[f == k] = cls().fit(raw[f != k], y[f != k]).predict(raw[f == k])
            slope, _ = calibration_slope(y, cal)
            rows.append({"innings": inn, "method": name, "auc": roc_auc_score(y, cal),
                         "brier": brier_score_loss(y, cal), "cal_slope": slope})
    return pd.DataFrame(rows)


def inn1_ablation(balls: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold AUC for innings 1 feature sets, shows why raw venue index was dropped."""
    d = train_set(balls, 1)
    y, folds = d["is_bat_team_winner"].astype(int).to_numpy(), d["fold"].to_numpy()
    X = design(d, 1)
    X["score_before"], X["venue_scoring_index"], X["crr"] = d["score_before"], d["venue_scoring_index"], d["crr"]
    sets = {"score, wickets, balls, crr": (["balls_left", "wkts_before", "score_before", "crr"], [1, -1, 1, 1]),
            "plus raw venue index": (["balls_left", "wkts_before", "score_before", "crr", "venue_scoring_index"],
                                     [1, -1, 1, 1, 0]),
            "score scaled by venue par": (["balls_left", "wkts_before", "score_vs_par", "crr"], [1, -1, 1, 1]),
            "par score without crr, used": (["balls_left", "wkts_before", "score_vs_par"], [1, -1, 1])}
    rows = []
    for name, (cols, mono) in sets.items():
        oof = np.empty(len(d))
        for f in np.unique(folds):
            tr = folds != f
            prm = {**_Booster(1).params, "monotone_constraints": mono}
            bst = lgb.train(prm, lgb.Dataset(X.loc[tr, cols], y[tr]), num_boost_round=cfg()["wp"]["max_iter"])
            oof[~tr] = bst.predict(X.loc[~tr, cols])
        rows.append({"feature_set": name, "oof_auc": roc_auc_score(y, oof)})
    return pd.DataFrame(rows)


def run() -> None:
    balls = read_parquet("data_prep", "balls_clean.parquet")
    model, mets, cals, oofs = fit_all(balls)
    b = add_transitions(balls, model)
    joblib.dump(model, stage_dir("pressure") / "wp_model.joblib")
    write_parquet(b, "pressure", "balls_wp.parquet")
    write_table(mets, "pressure", "wp_metrics.csv")
    write_table(cals, "pressure", "wp_calibration.csv")
    write_table(known_situations(model), "pressure", "wp_known_situations.csv")
    write_table(monotonic_checks(model), "pressure", "wp_monotonic_checks.csv")
    write_table(inn1_ablation(balls), "pressure", "wp_inn1_ablation.csv")
    write_table(calibrator_comparison(oofs), "pressure", "wp_calibrator_comparison.csv")
    log_run("wp_model", {"metrics": mets.to_dict("records"), "config": cfg()["wp"]})


def load_model() -> WPModel:
    return joblib.load(stage_dir("pressure") / "wp_model.joblib")


if __name__ == "__main__":
    # Import by package name so pickled classes resolve outside this script
    from pressure import wp_model

    wp_model.run()
