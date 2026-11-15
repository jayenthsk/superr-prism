"""Batter shot-mix drift between pressure and normal balls."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from common.config import cfg
from common.io import log_run, read_parquet, write_table

GROUPS = ["aggressive", "orthodox", "defensive"]


def stroke_group(stroke: pd.Series) -> pd.Series:
    sg = cfg()["stroke_groups"]
    out = pd.Series("untagged", index=stroke.index)
    for g in GROUPS:
        out[stroke.isin(sg[g])] = g
    return out


def _mix(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby(["player_id", "phase"])["sgroup"].value_counts().unstack(fill_value=0) \
        .reindex(columns=GROUPS, fill_value=0)


def drift_table(d: pd.DataFrame) -> pd.DataFrame:
    t = d[d["stroke"].notna()].assign(sgroup=stroke_group(d["stroke"]))
    t = t[t["sgroup"] != "untagged"]
    P, N = t[t["is_pressure"]], t[~t["is_pressure"]]
    league = {tag: x.groupby("phase")["sgroup"].value_counts(normalize=True).unstack().reindex(columns=GROUPS)
              for tag, x in [("P", P), ("N", N)]}
    mp, mn = _mix(P), _mix(N)
    rows = []
    for pid in mp.index.get_level_values(0).unique().intersection(mn.index.get_level_values(0).unique()):
        cp, cn = mp.loc[pid], mn.loc[pid]
        own_p, own_n = cp.sum() / cp.values.sum(), cn.sum() / cn.values.sum()
        wp, wn = cp.sum(axis=1) / cp.values.sum(), cn.sum(axis=1) / cn.values.sum()
        # League drift for his own phase mix, so a finisher is judged against finishers
        lp = (league["P"].reindex(wp.index).mul(wp, axis=0)).sum()
        ln = (league["N"].reindex(wn.index).mul(wn, axis=0)).sum()
        rows.append({"player_id": pid, "press_strokes": int(cp.values.sum()), "normal_strokes": int(cn.values.sum()),
                     "drift": jensenshannon(own_p, own_n, base=2) ** 2,
                     "league_drift": jensenshannon(lp, ln, base=2) ** 2,
                     "aggr_P": own_p["aggressive"], "aggr_N": own_n["aggressive"],
                     "def_P": own_p["defensive"], "def_N": own_n["defensive"]})
    out = pd.DataFrame(rows)
    out["excess_drift"] = out["drift"] - out["league_drift"]
    return out


def aggressive_productivity(d: pd.DataFrame) -> pd.DataFrame:
    t = d[stroke_group(d["stroke"]) == "aggressive"]
    g = t.groupby(["player_id", "is_pressure"]).agg(n=("y_runs", "size"), rpb=("y_runs", "mean"),
                                                    out=("y_out", "mean")).unstack("is_pressure")
    g.columns = [f"{a}_{'P' if b else 'N'}" for a, b in g.columns]
    return g.reset_index()


def quadrants(drift: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    j = drift.merge(ratings[["player_id", "player_name", "eligible", "rawP_runs", "thetaN_runs"]], on="player_id")
    j["press_prod"] = j["rawP_runs"] - j["thetaN_runs"]
    hi = j["excess_drift"] > j.loc[j["eligible"], "excess_drift"].median()
    pos = j["press_prod"] > 0
    j["quadrant"] = np.select([~hi & pos, hi & pos, hi & ~pos], ["Composed", "Adaptive", "Forced"], "Frozen")
    return j


def league_shift(d: pd.DataFrame) -> pd.DataFrame:
    t = d[d["stroke"].notna()].assign(sgroup=stroke_group(d["stroke"]))
    return t.groupby(["phase", "is_pressure"])["sgroup"].value_counts(normalize=True).unstack().reset_index()


def tag_coverage(d: pd.DataFrame) -> pd.DataFrame:
    t = d.assign(sgroup=stroke_group(d["stroke"]))
    return t.groupby("comp_type")["sgroup"].value_counts(normalize=True).unstack().reset_index()


def run() -> None:
    d = read_parquet("player_ratings", "balls_bat.parquet")
    r = read_parquet("player_ratings", "player_ratings_bat.parquet")
    q = quadrants(drift_table(d), r).merge(aggressive_productivity(d), on="player_id", how="left")
    write_table(q, "player_insights", "e2_shot_mix.csv")
    write_table(league_shift(d), "player_insights", "e2_league_shift.csv")
    write_table(tag_coverage(d), "player_insights", "e2_tag_coverage.csv")
    log_run("e2", {"players": len(q)})


if __name__ == "__main__":
    run()
