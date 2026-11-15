"""Expected outcome of every ball for an average player in the identical situation."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.config import cfg

COMPONENTS = ["boundary", "dot", "out", "runs"]
CAT_COLS = ["comp_type", "pace_or_spin", "batting_type", "phase"]
NUM_COLS = ["innings", "over_no", "wkts_before", "balls_since_wkt", "batter_balls_faced_capped", "season",
            "venue_scoring_index", "impact", "order_bucket", "rrr_bucket", "wp_bucket"]
RRR_EDGES = [-np.inf, 6, 8, 10, 12, 15, np.inf]


def batter_rows(b: pd.DataFrame) -> pd.DataFrame:
    """Balls faced, wides are not faced so they stay out."""
    d = b[~b["is_wide"]].copy()
    d["y_boundary"] = (d["runs"] >= 4).astype(int)
    d["y_dot"] = (d["is_legal"] & (d["total_runs"] == 0)).astype(int)
    d["y_out"] = d["is_batter_out"].astype(int)
    d["y_runs"] = d["bat_runs_credited"].astype(float)
    d["player_id"], d["opp_id"] = d["striker_id"], d["bowler_id"]
    return d


def bowler_rows(b: pd.DataFrame) -> pd.DataFrame:
    """Every delivery, run-outs and byes are not down to the bowler."""
    d = b.copy()
    d["y_boundary"] = (d["runs"] >= 4).astype(int)
    d["y_dot"] = (d["is_legal"] & (d["total_runs"] == 0)).astype(int)
    d["y_out"] = d["is_bowler_wicket"].astype(int)
    d["y_runs"] = d["bowler_runs_conceded"].astype(float)
    d["player_id"], d["opp_id"] = d["bowler_id"], d["striker_id"]
    return d


def context_matrix(d: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """Situation only, player identity never enters the baseline."""
    X = pd.DataFrame(index=d.index)
    for col in CAT_COLS:
        X[col] = d[col].astype("category")
    X["innings"], X["over_no"], X["wkts_before"] = d["innings"], d["over_no"], d["wkts_before"]
    X["balls_since_wkt"] = d["balls_since_wkt"]
    X["batter_balls_faced_capped"] = d["batter_balls_faced_capped"]
    X["season"], X["venue_scoring_index"] = d["season"], d["venue_scoring_index"]
    X["impact"] = d["ipl_impact_era"].astype(int)
    order = d["batting_order"].fillna(6)
    X["order_bucket"] = np.select([order <= 2, order <= 4, order <= 7], [0, 1, 2], 3)
    X["rrr_bucket"] = np.where(d["innings"] == 2, pd.cut(d["rrr"], RRR_EDGES, labels=False), -1)
    X["wp_bucket"] = np.floor(d["wp_pre"].clip(0, 0.999) * 10)
    for col in extra or []:
        X[col] = d[col]
    return X


def _params(objective: str) -> dict:
    c = cfg()["baseline"]
    return dict(objective=objective, learning_rate=c["learning_rate"], num_leaves=c["max_leaf_nodes"],
                min_child_samples=c["min_samples_leaf"], verbose=-1, seed=cfg()["seed"], deterministic=True,
                force_row_wise=True, num_threads=cfg()["wp"].get("n_jobs", 4))


def fit_oof(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, objective: str) -> np.ndarray:
    """Out-of-fold expectation by match fold, so no ball sees its own result."""
    out = np.empty(len(y))
    rounds = cfg()["baseline"]["max_iter"]
    for f in np.unique(folds):
        tr = folds != f
        bst = lgb.train(_params(objective), lgb.Dataset(X[tr], y[tr]), num_boost_round=rounds)
        out[~tr] = bst.predict(X[~tr])
    return out


def add_expected(d: pd.DataFrame, extra_by_comp: dict[str, list[str]] | None = None) -> pd.DataFrame:
    d = d.copy()
    folds = d["fold"].to_numpy()
    for comp in COMPONENTS:
        X = context_matrix(d, (extra_by_comp or {}).get(comp))
        obj = "poisson" if comp == "runs" else "binary"
        d[f"x_{comp}"] = fit_oof(X, d[f"y_{comp}"].to_numpy(), folds, obj)
        d[f"r_{comp}"] = d[f"y_{comp}"] - d[f"x_{comp}"]
    return d


def diagnostics(d: pd.DataFrame, role: str) -> pd.DataFrame:
    rows = []
    for comp in COMPONENTS:
        for ph, g in [("ALL", d)] + list(d.groupby("phase")):
            rows.append({"role": role, "component": comp, "phase": ph, "actual": g[f"y_{comp}"].mean(),
                         "expected": g[f"x_{comp}"].mean(), "mean_resid": g[f"r_{comp}"].mean(), "n": len(g)})
    return pd.DataFrame(rows)
