"""PRISM rating, partial pooling of normal skill, shrunk clutch effect and win probability value."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import cfg
from common.io import log_run, read_parquet, read_raw, write_parquet, write_table
from player_ratings import eligibility
from player_ratings.baseline import COMPONENTS, add_expected, batter_rows, bowler_rows, diagnostics
from player_ratings.opposition import rank_changes, shrink_k

ROLES = ("bat", "bowl")
BALL_COLS = ["match_id", "fold", "season", "comp_type", "innings", "phase", "over_no", "player_id", "opp_id",
             "is_legal", "is_pressure", "pressure_tier", "li", "li_w", "dwp", "wp_pre", "is_knockout",
             "has_associate", "is_revealed_pressure", "batter_balls_faced_before", "wkts_before", "runs",
             "total_runs", "bat_runs_credited", "bowler_runs_conceded", "is_wicket", "is_batter_out",
             "is_bowler_wicket", "is_wide", "is_noball", "pace_or_spin", "delivery_style", "stroke",
             "batting_type", "striker_id", "non_striker_id", "bowler_id", "innings_ball_num", "legal_before",
             "after_wicket", "gap_s", "tempo_z", "dismissed_player_id", "is_nonstriker_out", "is_tie",
             "is_hard_entry", "is_entry_window", "is_hard_entry_ball", "dot_streak_before", "balls_since_wkt",
             "is_dot", "striker_name", "bowler_name", "batting_team_id", "bowling_team_id", "over_ball_num"]


def side_dwp(d: pd.DataFrame, role: str) -> pd.Series:
    return 100 * (d["dwp"] if role == "bat" else -d["dwp"])


def component_values(d: pd.DataFrame, role: str) -> pd.DataFrame:
    """WP value of each outcome on pressure balls, by innings and phase.

    Least squares of side WP swing on boundary, dot, out and runs with an intercept,
    so each slope is the swing an outcome causes relative to an average ball.
    """
    rows = []
    p = d[d["is_pressure"]].assign(y=side_dwp(d[d["is_pressure"]], role))
    for (inn, ph), g in p.groupby(["innings", "phase"]):
        X = np.column_stack([np.ones(len(g))] + [g[f"y_{c}"].to_numpy(float) for c in COMPONENTS])
        coef, *_ = np.linalg.lstsq(X, g["y"].to_numpy(), rcond=None)
        rows.append({"innings": inn, "phase": ph, "n": len(g), **{f"v_{c}": coef[i + 1]
                                                                  for i, c in enumerate(COMPONENTS)}})
    return pd.DataFrame(rows)


def ball_weights(d: pd.DataFrame, opts: dict) -> tuple[np.ndarray, np.ndarray]:
    bw = np.ones(len(d))
    hl = opts.get("recency_half_life")
    if hl:
        bw = 0.5 ** ((d["season"].max() - d["season"]).to_numpy() / hl)
    press = d[opts.get("pressure_col", "is_pressure")].to_numpy()
    li = d["li"].to_numpy() if opts.get("li_cap", "cfg") is None else \
        d["li"].clip(upper=cfg()["li"]["cap"] if opts.get("li_cap", "cfg") == "cfg" else opts["li_cap"]).to_numpy()
    if opts.get("pressure_col") == "is_revealed_pressure":
        li = np.ones(len(d))
    w = np.where(press, li, 0.0) * bw
    ko = opts.get("knockout", cfg()["modifiers"]["knockout"])
    if ko != 1.0:
        w = w * np.where(d["is_knockout"], ko, 1.0)
    he = opts.get("hard_entry_weight", 1.0)
    if he != 1.0 and "is_hard_entry_ball" in d:
        w = w * np.where(d["is_hard_entry_ball"], he, 1.0)
    return np.where(press, 0.0, bw), w


def player_match_stats(d: pd.DataFrame, v: pd.DataFrame, opts: dict | None = None,
                       resid_prefix: str = "r_") -> pd.DataFrame:
    """Sufficient sums per player and match, the bootstrap only reweights these."""
    opts = opts or {}
    bw, w = ball_weights(d, opts)
    vv = d[["innings", "phase"]].merge(v, on=["innings", "phase"], how="left").fillna(0)
    cols = {"n_norm": bw, "sw": w, "sw2": w**2, "n_press": (w > 0).astype(float)}
    for c in COMPONENTS:
        r = d[f"{resid_prefix}{c}"].to_numpy()
        cols[f"s_{c}"] = bw * r
        cols[f"q_{c}"] = bw * r**2
        cols[f"swr_{c}"] = w * r
        cols[f"swv_{c}"] = w * vv[f"v_{c}"].to_numpy()
    df = pd.DataFrame(cols, index=d.index)
    df["player_id"], df["match_id"] = d["player_id"].to_numpy(), d["match_id"].to_numpy()
    return df.groupby(["player_id", "match_id"]).sum()


def estimate_k(ps: pd.DataFrame, min_n: int) -> dict[str, float]:
    rc = cfg()["ratings"]
    k = {}
    for c in COMPONENTS:
        kn = shrink_k(ps[f"s_{c}"], ps[f"q_{c}"], ps["n_norm"], min_n, rc["k_floor_tau2"])
        k[f"kN_{c}"] = kn
        ok = (ps["n_norm"] >= min_n) & (ps["sw"] > 0)
        n, s, q = ps.loc[ok, "n_norm"], ps.loc[ok, f"s_{c}"], ps.loc[ok, f"q_{c}"]
        sigma2 = float((q - s**2 / n).sum() / (n - 1).sum())
        n_eff = ps.loc[ok, "sw"] ** 2 / ps.loc[ok, "sw2"]
        ok2 = n_eff >= min_n / 3
        # Raw pressure mean minus raw normal mean, noise from both sides comes off the spread
        delta = (ps.loc[ok, f"swr_{c}"] / ps.loc[ok, "sw"] - s / n)[ok2]
        # Player level noise, power hitters swing more than a pooled variance would admit
        sig_p = (q / n - (s / n) ** 2).clip(lower=1e-9)[ok2]
        tau2_d = float(delta.var() - np.mean(sig_p / n_eff[ok2] + sig_p / n[ok2]))
        k[f"tau2_delta_{c}"] = tau2_d
        # Clutch effect barely repeats, so when its true spread is not positive pull it hard to normal
        bad = not np.isfinite(tau2_d) or tau2_d <= 0
        k[f"kD_{c}"] = rc["k_delta_max"] if bad else min(sigma2 / tau2_d, rc["k_delta_max"])
    return k


def rate(ps: pd.DataFrame, k: dict[str, float]) -> pd.DataFrame:
    """Normal level, shrunk clutch delta and WP value, from player level sums."""
    out = pd.DataFrame(index=ps.index)
    sw = ps["sw"].replace(0, np.nan)
    n_eff = (ps["sw"] ** 2 / ps["sw2"].replace(0, np.nan)).fillna(0)
    out["n_norm"], out["n_press"], out["n_eff"] = ps["n_norm"], ps["n_press"], n_eff
    prism = np.zeros(len(ps))
    for c in COMPONENTS:
        theta_n = ps[f"s_{c}"] / (ps["n_norm"] + k[f"kN_{c}"])
        ybar_p = (ps[f"swr_{c}"] / sw).fillna(theta_n)
        delta = n_eff * (ybar_p - theta_n) / (n_eff + k[f"kD_{c}"])
        theta_p = theta_n + delta
        vbar = (ps[f"swv_{c}"] / sw).fillna(0)
        out[f"thetaN_{c}"], out[f"delta_{c}"], out[f"thetaP_{c}"] = theta_n, delta, theta_p
        out[f"rawP_{c}"] = ybar_p
        out[f"value_{c}"] = 100 * vbar * theta_p
        prism += out[f"value_{c}"].to_numpy()
    out["prism"] = prism
    return out


def compute(d: pd.DataFrame, v: pd.DataFrame, opts: dict | None = None, k: dict | None = None,
            resid_prefix: str = "r_") -> tuple[pd.DataFrame, dict]:
    pm = player_match_stats(d, v, opts, resid_prefix)
    ps = pm.groupby(level="player_id").sum()
    k = k or estimate_k(ps, cfg()["eligibility"]["min_pressure_balls"])
    return rate(ps, k), k


def raw_stats(d: pd.DataFrame, role: str) -> pd.DataFrame:
    rows = {}
    for tag, m in [("P", d["is_pressure"]), ("N", ~d["is_pressure"])]:
        g = d[m].groupby("player_id")
        legal = d[m & d["is_legal"]].groupby("player_id").size()
        balls = g.size()
        if role == "bat":
            runs, outs = g["bat_runs_credited"].sum(), g["is_batter_out"].sum()
            rows[f"sr_{tag}"] = 100 * runs / balls
            rows[f"avg_{tag}"] = runs / outs.replace(0, np.nan)
        else:
            runs, outs = g["bowler_runs_conceded"].sum(), g["is_bowler_wicket"].sum()
            rows[f"econ_{tag}"] = 6 * runs / legal
            rows[f"avg_{tag}"] = runs / outs.replace(0, np.nan)
            rows[f"bsr_{tag}"] = legal / outs.replace(0, np.nan)
        rows[f"dot_{tag}"] = g["y_dot"].mean()
        rows[f"bnd_{tag}"] = g["y_boundary"].mean()
    return pd.DataFrame(rows)


def player_names(role: str) -> pd.DataFrame:
    b = read_parquet("data_prep", "balls_clean.parquet")
    id_col, name_col, team_col = ("striker_id", "striker_name", "batting_team_id") if role == "bat" else \
        ("bowler_id", "bowler_name", "bowling_team_id")
    names = b.groupby(id_col)[name_col].agg(lambda s: s.mode().iat[0])
    teams = read_raw("teams.csv").drop_duplicates("team_id").set_index("team_id")["team_name"]
    tl = b.assign(team=b[team_col].map(teams)).groupby(id_col)["team"].agg(
        lambda s: ", ".join(s.value_counts().index))
    comp = b.groupby(id_col)["comp_type"].agg(lambda s: ", ".join(sorted(s.unique())))
    out = pd.DataFrame({"player_name_raw": names, "teams": tl, "comp_types_played": comp})
    # IDs are never merged, a shared name gets the main team appended
    all_names = pd.concat([b.groupby("striker_id")["striker_name"].first(),
                           b.groupby("bowler_id")["bowler_name"].first()])
    all_names = all_names[~all_names.index.duplicated()]
    shared = set(all_names[all_names.duplicated(keep=False)].values)
    main = out["teams"].str.split(", ").str[0]
    out["player_name"] = np.where(out["player_name_raw"].isin(shared),
                                  out["player_name_raw"] + " (" + main + ")", out["player_name_raw"])
    out.index.name = "player_id"
    return out


def build_residuals(opposition: bool) -> dict[str, pd.DataFrame]:
    """Two-pass baseline, pass one plain context, pass two adds out-of-fold opponent skill."""
    from player_ratings.opposition import oof_opponent_skill

    b = read_parquet("pressure", "balls_pressure.parquet")
    b = b[b["in_scope"]]
    tabs = {"bat": batter_rows(b), "bowl": bowler_rows(b)}
    tabs = {r: add_expected(t) for r, t in tabs.items()}
    for t in tabs.values():
        for c in COMPONENTS:
            t[f"r1_{c}"] = t[f"r_{c}"]
    if opposition:
        rc, e = cfg()["ratings"], cfg()["eligibility"]
        opp = {"bat": oof_opponent_skill(tabs["bowl"], tabs["bat"], e["min_pressure_balls"], rc["k_floor_tau2"]),
               "bowl": oof_opponent_skill(tabs["bat"], tabs["bowl"], e["min_pressure_balls"], rc["k_floor_tau2"])}
        extra = {c: [f"opp_{x}" for x in COMPONENTS] for c in COMPONENTS}
        for r in ROLES:
            t = pd.concat([tabs[r], opp[r]], axis=1)
            tabs[r] = add_expected(t, extra)
    return tabs


def assemble(role: str, d: pd.DataFrame, v: pd.DataFrame, names: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rt, k = compute(d, v)
    sizes = eligibility.sample_sizes(d)
    out = names.join(sizes, how="right").join(rt).join(raw_stats(d, role))
    out["eligible"] = eligibility.apply_rules(sizes)
    if "opp_runs" in d:
        press = d[d["is_pressure"]]
        out["opp_strength_pressure"] = press.groupby("player_id")[[f"opp_{c}" for c in COMPONENTS]].mean() \
            .pipe(lambda x: x["opp_runs"])
    out["rank_all"] = out.loc[out["eligible"], "prism"].rank(ascending=False, method="first")
    return out.reset_index().sort_values("prism", ascending=False), k


def run() -> None:
    rc = cfg()["ratings"]
    tabs = build_residuals(rc["opposition_adjust"])
    diag, ks, values, results = [], [], {}, {}
    for role in ROLES:
        d = tabs[role]
        v = component_values(d, role)
        values[role] = v
        names = player_names(role)
        res, k = assemble(role, d, v, names)
        results[role] = res
        ks.append({"role": role, **k})
        diag.append(diagnostics(d, role))
        keep = [c for c in BALL_COLS if c in d] + [f"{p}_{c}" for p in ("y", "x", "r", "r1") for c in COMPONENTS] \
            + [c for c in d if c.startswith("opp_") and c != "opp_id"]
        write_parquet(d[keep], "player_ratings", f"balls_{role}.parquet")
        write_parquet(res, "player_ratings", f"player_ratings_{role}.parquet")
        write_table(v, "player_ratings", f"component_values_{role}.csv")
        cols = ["player_id", "player_name", "teams", "total_balls", "pressure_balls", "matches_with_pressure",
                "eligible", "prism"] + [f"{p}_{c}" for p in ("thetaN", "delta", "thetaP") for c in COMPONENTS]
        write_table(res[cols], "player_ratings", f"ratings_{role}.csv")
        if rc["opposition_adjust"]:
            before, _ = compute(d, v, k=k, resid_prefix="r1_")
            el = res[res["eligible"]].set_index("player_id")
            b4 = before.loc[el.index].assign(player_name=el["player_name"]).reset_index()
            write_table(rank_changes(b4, el.reset_index()), "player_ratings", f"opposition_rank_changes_{role}.csv")
    sizes = {r: eligibility.sample_sizes(tabs[r]) for r in ROLES}
    write_table(eligibility.pool_sizes(sizes), "player_ratings", "eligibility_pools.csv")
    write_table(pd.DataFrame(ks), "player_ratings", "shrinkage_constants.csv")
    write_table(pd.concat(diag), "player_ratings", "baseline_diagnostics.csv")
    log_run("ratings", {"k": ks, "eligibility": cfg()["eligibility"], "opposition_adjust": rc["opposition_adjust"]})


if __name__ == "__main__":
    run()
