"""Folklore tests, dot ball streaks and tight overs against wicket rates, context controlled."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import cfg
from common.io import read_parquet, write_table


def dot_streak_test(bowl: pd.DataFrame) -> pd.DataFrame:
    """Wicket rate by dots in a row before the ball, within phase, actual against baseline."""
    cap = cfg()["myths"]["streak_cap"]
    d = bowl[bowl["is_legal"]].copy()
    d["streak"] = d["dot_streak_before"].clip(upper=cap)
    t = d.groupby(["phase", "streak"]).agg(balls=("y_out", "size"), wicket_rate=("y_out", "mean"),
                                           expected=("x_out", "mean")).reset_index()
    t["excess"] = t["wicket_rate"] - t["expected"]
    return t


def dot_streak_slope(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ph, g in t.groupby("phase"):
        w = g["balls"]
        x, y = g["streak"], g["excess"]
        xm, ym = np.average(x, weights=w), np.average(y, weights=w)
        slope = np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
        se = np.sqrt(np.average((y - ym) ** 2, weights=w) / np.sum(w * (x - xm) ** 2) * w.mean())
        rows.append({"phase": ph, "slope_per_dot": slope, "approx_se": se})
    return pd.DataFrame(rows)


def tight_over_test(bowl: pd.DataFrame) -> pd.DataFrame:
    """Wicket rate in the next over after a tight one, same innings, within phase."""
    tight = cfg()["myths"]["tight_over_runs"]
    ov = bowl.groupby(["match_id", "innings", "over_no"]).agg(
        runs=("y_runs", "sum"), phase=("phase", "first")).reset_index()
    nxt = bowl[bowl["is_legal"]].groupby(["match_id", "innings", "over_no"]).agg(
        wk=("y_out", "sum"), xwk=("x_out", "sum"), balls=("y_out", "size")).reset_index()
    nxt["over_no"] -= 1
    j = ov.merge(nxt, on=["match_id", "innings", "over_no"])
    j["tight"] = j["runs"] <= tight
    t = j.groupby(["phase", "tight"]).agg(overs=("wk", "size"), wkts_next=("wk", "sum"),
                                          expected_next=("xwk", "sum"), balls=("balls", "sum")).reset_index()
    t["wicket_rate_next"] = t["wkts_next"] / t["balls"]
    t["excess_per_100"] = 100 * (t["wkts_next"] - t["expected_next"]) / t["balls"]
    return t


def run() -> None:
    bowl = read_parquet("player_ratings", "balls_bowl.parquet")
    t = dot_streak_test(bowl)
    write_table(t, "reliability", "myth_dot_streak.csv")
    write_table(dot_streak_slope(t), "reliability", "myth_dot_streak_slope.csv")
    write_table(tight_over_test(bowl), "reliability", "myth_tight_over.csv")


if __name__ == "__main__":
    run()
