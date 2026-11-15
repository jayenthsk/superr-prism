"""Bowler execution under pressure and bounce-back after a boundary."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import cfg
from common.io import log_run, read_parquet, write_table

GROUPS = ["yorker", "full_toss", "slower", "bouncer"]


def tag_deliveries(d: pd.DataFrame) -> pd.DataFrame:
    """Delivery style groups from the analyst tags, beamers count as missed full balls."""
    dg = cfg()["delivery_groups"]
    out = d.copy()
    for g in GROUPS:
        out[g] = out["delivery_style"].isin(dg[g])
    out["wide"] = out["is_wide"]
    return out


def style_coverage(d: pd.DataFrame) -> pd.DataFrame:
    t = tag_deliveries(d[d["pace_or_spin"] == "Pace"])
    return t.groupby("comp_type")[GROUPS + ["wide"]].mean().assign(
        missing_style=t.groupby("comp_type")["delivery_style"].apply(lambda s: s.isna().mean())).reset_index()


def league_execution(d: pd.DataFrame) -> pd.DataFrame:
    t = tag_deliveries(d[(d["pace_or_spin"] == "Pace") & (d["phase"] == "DEATH")])
    g = t.groupby("is_pressure")
    out = g[GROUPS + ["wide"]].mean().T
    out.columns = ["normal" if not c else "pressure" for c in out.columns]
    runs = {grp: t[t[grp]].groupby("is_pressure")["y_runs"].mean() for grp in GROUPS}
    out["runs_normal"] = [runs[grp].get(False, np.nan) for grp in GROUPS] + [np.nan]
    out["runs_pressure"] = [runs[grp].get(True, np.nan) for grp in GROUPS] + [np.nan]
    return out.reset_index(names="delivery")


def pace_execution(d: pd.DataFrame, k: float) -> pd.DataFrame:
    """Yorker minus full toss share in pressure against normal death balls, shrunk by sample."""
    t = tag_deliveries(d[d["pace_or_spin"] == "Pace"])
    t = t[(t["phase"] == "DEATH") | t["is_pressure"]]
    t["net"] = t["yorker"].astype(int) - t["full_toss"].astype(int)
    p, n = t[t["is_pressure"]], t[~t["is_pressure"] & (t["phase"] == "DEATH")]
    gp, gn = p.groupby("player_id"), n.groupby("player_id")
    out = pd.DataFrame({"press_balls": gp.size(), "normal_death_balls": gn.size(),
                        "yorker_P": gp["yorker"].mean(), "fulltoss_P": gp["full_toss"].mean(),
                        "slower_P": gp["slower"].mean(), "wide_P": gp["wide"].mean(),
                        "yorker_N": gn["yorker"].mean(), "fulltoss_N": gn["full_toss"].mean(),
                        "net_P": gp["net"].mean(), "net_N": gn["net"].mean(),
                        "runs_per_yorker_P": p[p["yorker"]].groupby("player_id")["y_runs"].mean(),
                        "runs_per_slower_P": p[p["slower"]].groupby("player_id")["y_runs"].mean()})
    raw = out["net_P"] - out["net_N"]
    out["execution_raw"] = raw
    out["execution_score"] = raw * out["press_balls"] / (out["press_balls"] + k)
    return out.reset_index()


def spin_execution(d: pd.DataFrame) -> pd.DataFrame:
    t = tag_deliveries(d[d["pace_or_spin"] == "Spin"])
    g = t.groupby(["player_id", "is_pressure"]).agg(balls=("full_toss", "size"), fulltoss=("full_toss", "mean"),
                                                    wide=("wide", "mean"), bnd_resid=("r_boundary", "mean"))
    w = g.unstack("is_pressure")
    w.columns = [f"{a}_{'P' if b else 'N'}" for a, b in w.columns]
    # Bowler residuals are actual minus expected, so fewer boundaries than context is negative
    w["bnd_resid_P_better"] = -w["bnd_resid_P"]
    return w.reset_index()


def bounce_back(d: pd.DataFrame, k: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Next legal ball by the same bowler after he concedes a boundary, raw and against the baseline."""
    t = d[d["is_legal"]].sort_values(["match_id", "innings", "innings_ball_num"])
    t = t.assign(prev_bnd=t.groupby(["match_id", "player_id"])["y_boundary"].shift(1))
    t = t[t["prev_bnd"].notna()]
    t["after_bnd"] = t["prev_bnd"] == 1
    league = t.groupby("after_bnd").agg(balls=("y_boundary", "size"), boundary_rate=("y_boundary", "mean"),
                                        expected_boundary=("x_boundary", "mean"), runs=("y_runs", "mean"),
                                        expected_runs=("x_runs", "mean")).reset_index()
    a = t[t["after_bnd"]]
    g = a.groupby("player_id").agg(n=("r_runs", "size"), sr=("r_runs", "sum"), sb=("r_boundary", "sum"))
    # Shrunk toward zero, positive means he tightens up after being hit
    g["bounce_runs"] = -g["sr"] / (g["n"] + k)
    g["bounce_boundary"] = -g["sb"] / (g["n"] + k)
    return league, g.reset_index()


def run() -> None:
    k = cfg()["extras"]["min_attempts_k"]
    d = read_parquet("player_ratings", "balls_bowl.parquet")
    elig = read_parquet("player_ratings", "player_ratings_bowl.parquet")[["player_id", "player_name", "eligible"]]
    write_table(style_coverage(d), "player_insights", "e1_style_coverage.csv")
    write_table(league_execution(d), "player_insights", "e1_league_execution.csv")
    write_table(pace_execution(d, k).merge(elig, on="player_id"), "player_insights", "e1_pace_execution.csv")
    write_table(spin_execution(d).merge(elig, on="player_id"), "player_insights", "e1_spin_execution.csv")
    league, bb = bounce_back(d, k)
    write_table(league, "player_insights", "e1_bounce_back_league.csv")
    write_table(bb.merge(elig, on="player_id"), "player_insights", "e1_bounce_back.csv")
    log_run("e1", {"k": k})


if __name__ == "__main__":
    run()
