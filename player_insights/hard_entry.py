"""Hardest entries, the first ten balls when a batter walks into a crisis."""
from __future__ import annotations

import pandas as pd

from common.config import cfg
from common.io import log_run, read_parquet, write_table

COMPS = ["runs", "boundary", "out"]


def entry_effects(d: pd.DataFrame, k: float) -> pd.DataFrame:
    """Residuals in hard entries against his own normal entries, first ten balls, shrunk."""
    w = d[d["is_entry_window"]]
    g = w.groupby(["player_id", "is_hard_entry"]).agg(
        n=("r_runs", "size"), innings=("match_id", "nunique"),
        **{f"r_{c}": (f"r_{c}", "mean") for c in COMPS}).unstack("is_hard_entry")
    g.columns = [f"{a}_{'hard' if b else 'normal'}" for a, b in g.columns]
    g = g.dropna(subset=["n_hard", "n_normal"])
    for c in COMPS:
        diff = g[f"r_{c}_hard"] - g[f"r_{c}_normal"]
        # Shrink by hard entry balls, few batters face enough of them to trust the raw gap
        g[f"hard_gap_{c}"] = diff * g["n_hard"] / (g["n_hard"] + k)
    return g.reset_index()


def league_entry(d: pd.DataFrame) -> pd.DataFrame:
    w = d[d["is_entry_window"]]
    return w.groupby("is_hard_entry").agg(balls=("y_runs", "size"), runs_pb=("y_runs", "mean"),
                                          x_runs_pb=("x_runs", "mean"), out_rate=("y_out", "mean"),
                                          x_out_rate=("x_out", "mean")).reset_index()


def run() -> None:
    k = cfg()["extras"]["min_attempts_k"]
    d = read_parquet("player_ratings", "balls_bat.parquet")
    r = read_parquet("player_ratings", "player_ratings_bat.parquet")[["player_id", "player_name", "eligible"]]
    write_table(entry_effects(d, k).merge(r, on="player_id"), "player_insights", "e3_hard_entry.csv")
    write_table(league_entry(d), "player_insights", "e3_league_entry.csv")
    share = d.drop_duplicates(["match_id", "innings", "player_id"])["is_hard_entry"].mean()
    log_run("e3", {"k": k, "hard_entry_share_of_innings": float(share)})


if __name__ == "__main__":
    run()
