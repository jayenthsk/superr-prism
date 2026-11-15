"""Ball-state features, phases, streaks and timestamp gaps."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import cfg
from common.io import log_run, read_parquet, write_parquet

INN = ["match_id", "innings"]


def assign_folds(match_ids: pd.Series, n_folds: int, seed: int) -> pd.Series:
    """One fixed match-level fold map shared by every out-of-fold model."""
    uniq = np.sort(match_ids.unique())
    rng = np.random.default_rng(seed)
    folds = pd.Series(rng.permutation(len(uniq)) % n_folds, index=uniq)
    return match_ids.map(folds).astype(int)


def add_state(b: pd.DataFrame) -> pd.DataFrame:
    c = cfg()
    g = b.groupby(INN, sort=False)
    b["legal_before"] = g["is_legal"].cumsum() - b["is_legal"].astype(int)
    b["legal_ball_no"] = (b["legal_before"] + 1).clip(upper=120)
    b["over_no"] = b["overid"].astype(int)
    b["phase"] = "MID"
    for name, (lo, hi) in c["phases"].items():
        b.loc[b["over_no"].between(lo, hi), "phase"] = name
    b["score_before"] = g["total_runs"].cumsum() - b["total_runs"]
    b["wkts_before"] = (g["is_wicket"].cumsum() - b["is_wicket"].astype(int)).clip(upper=9)
    b["balls_left"] = (120 - b["legal_before"]).clip(lower=1)
    # Run rate on a tiny sample is meaningless, so floor the denominator at one over
    b["crr"] = b["score_before"] * 6 / b["legal_before"].clip(lower=6)
    inn1 = b[b["innings"] == 1].groupby("match_id")["total_runs"].sum()
    b["inn1_total"] = b["match_id"].map(inn1)
    b["target"] = np.where(b["innings"] == 2, b["inn1_total"] + 1, np.nan)
    b["runs_needed"] = b["target"] - b["score_before"]
    b["rrr"] = b["runs_needed"] * 6 / b["balls_left"]
    b["is_bat_team_winner"] = b["batting_team_id"] == b["win_team_id"]
    return b


def add_streaks(b: pd.DataFrame) -> pd.DataFrame:
    c = cfg()["caps"]
    b["is_dot"] = b["is_legal"] & (b["total_runs"] == 0)
    # Streak counts legal dots in a row by the batting side, any scoring delivery resets it
    breaker = (~b["is_dot"] & (b["is_legal"] | (b["total_runs"] > 0))).astype(int)
    run_id = breaker.groupby([b["match_id"], b["innings"]]).cumsum()
    streak_after = b["is_dot"].astype(int).groupby([b["match_id"], b["innings"], run_id]).cumsum()
    b["dot_streak_before"] = streak_after.groupby([b["match_id"], b["innings"]]).shift(1).fillna(0)
    b["dot_streak_before"] = b["dot_streak_before"].astype(int)
    wkt_id = b.groupby(INN)["is_wicket"].cumsum() - b["is_wicket"].astype(int)
    legal_since = b["is_legal"].astype(int).groupby([b["match_id"], b["innings"], wkt_id]).cumsum() \
        - b["is_legal"].astype(int)
    b["partnership_balls"] = legal_since
    b["balls_since_wkt"] = legal_since.clip(upper=c["balls_since_wkt"])
    faced = (~b["is_wide"]).astype(int)
    by_batter = [b["match_id"], b["innings"], b["striker_id"]]
    b["batter_balls_faced_before"] = faced.groupby(by_batter).cumsum() - faced
    b["batter_balls_faced_capped"] = b["batter_balls_faced_before"].clip(upper=c["batter_balls_faced"])
    return b


def add_gaps(b: pd.DataFrame) -> pd.DataFrame:
    """Time since the previous delivery, only trusted inside an over."""
    t = cfg()["tempo"]
    gap = b.groupby(INN)["ts"].diff().dt.total_seconds()
    first_of_over = b["over_no"] != b.groupby(INN)["over_no"].shift(1)
    midnight = b["ts"].dt.strftime("%H:%M:%S") == "00:00:00"
    prev_midnight = midnight.groupby([b["match_id"], b["innings"]]).shift(1).fillna(True).astype(bool)
    ok = gap.between(t["gap_min"], t["gap_max"]) & ~first_of_over & ~midnight & ~prev_midnight
    b["gap_s"] = gap.where(ok)
    b["after_wicket"] = b.groupby(INN)["is_wicket"].shift(1).fillna(False).astype(bool)
    return b


def add_venue_index(b: pd.DataFrame, n_folds: int, k: float) -> pd.DataFrame:
    """Shrunk venue par from other folds only, so a match never sees its own total."""
    inn1 = (b[(b["innings"] == 1) & ~b["exclude"]]
            .groupby("match_id").agg(venue_id=("venue_id", "first"), total=("total_runs", "sum"),
                                     fold=("fold", "first")))
    mean_all = inn1["total"].mean()
    idx = {}
    for f in range(n_folds):
        tr = inn1[inn1["fold"] != f]
        v = tr.groupby("venue_id")["total"].agg(["sum", "count"])
        mu = tr["total"].mean()
        shrunk = (v["sum"] + k * mu) / (v["count"] + k)
        idx[f] = (shrunk / mu).to_dict()
    b["venue_scoring_index"] = [idx[f].get(v, 1.0) for f, v in zip(b["fold"], b["venue_id"], strict=True)]
    b.attrs["venue_mean_total"] = mean_all
    return b


def build_features(balls: pd.DataFrame) -> pd.DataFrame:
    c = cfg()
    b = balls.copy()
    b["fold"] = assign_folds(b["match_id"], c["wp"]["folds"], c["seed"])
    reg = b["innings"] <= 2
    feats = add_gaps(add_streaks(add_state(b[reg].copy())))
    feats = add_venue_index(feats, c["wp"]["folds"], c["wp"]["venue_shrink_k"])
    feats["in_scope"] = ~feats["exclude"]
    out = pd.concat([feats, b[~reg].assign(in_scope=False)], ignore_index=True)
    return out.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)


def run() -> None:
    balls = read_parquet("data_prep", "balls_clean_raw.parquet")
    out = build_features(balls)
    assert len(out) == len(balls), "ball rows lost in feature build"
    write_parquet(out, "data_prep", "balls_clean.parquet")
    log_run("features", {"rows": len(out), "in_scope_rows": int(out["in_scope"].sum()),
                         "in_scope_matches": int(out.loc[out["in_scope"], "match_id"].nunique()),
                         "valid_gap_share": float(out["gap_s"].notna().mean())})


if __name__ == "__main__":
    run()
