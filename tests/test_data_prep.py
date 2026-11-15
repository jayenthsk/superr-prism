"""Data validation checks."""
from __future__ import annotations

import pandas as pd
import pytest

from common.io import read_parquet, read_raw


@pytest.fixture(scope="module")
def balls() -> pd.DataFrame:
    return read_parquet("data_prep", "balls_clean.parquet")


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return read_parquet("data_prep", "matches_clean.parquet")


@pytest.fixture(scope="module")
def innings(balls: pd.DataFrame) -> pd.DataFrame:
    b = balls[balls["innings"] <= 2]
    return b.groupby(["match_id", "innings"]).agg(
        legal=("is_legal", "sum"), wkts=("is_wicket", "sum"), runs=("total_runs", "sum"),
        target=("target", "first"), winner=("win_team_id", "first"), bat=("batting_team_id", "first"),
        excl=("exclude", "first"), tie=("is_tie", "first")).reset_index()


def test_no_rows_lost(balls: pd.DataFrame) -> None:
    assert len(balls) == len(read_raw("ball_by_ball.csv"))


def test_legal_balls_capped(innings: pd.DataFrame) -> None:
    assert innings["legal"].max() <= 120


def test_all_out_means_ten(innings: pd.DataFrame) -> None:
    short = innings[(innings["legal"] < 120) & (innings["wkts"] >= 10)]
    assert (short["wkts"] == 10).all()


def test_chasing_winner_reached_target(innings: pd.DataFrame) -> None:
    i2 = innings[(innings["innings"] == 2) & ~innings["excl"] & ~innings["tie"]]
    won = i2[i2["winner"] == i2["bat"]]
    assert (won["runs"] >= won["target"]).all()


def test_runs_margin_matches(innings: pd.DataFrame, matches: pd.DataFrame) -> None:
    m = matches[~matches["exclude"]].copy()
    m["margin"] = m["result"].str.extract(r"[Bb]y (\d+) [Rr]un", expand=False).astype(float)
    m = m.dropna(subset=["margin"])
    tot = innings.pivot_table(index="match_id", columns="innings", values="runs")
    m = m.join(tot, on="match_id").dropna(subset=[1, 2])
    assert ((m[1] - m[2]) == m["margin"]).mean() > 0.98


def test_one_match_row_per_ball_match(balls: pd.DataFrame, matches: pd.DataFrame) -> None:
    counts = matches["match_id"].value_counts()
    assert (counts == 1).all()
    assert balls["match_id"].isin(matches["match_id"]).all()


def test_super_overs_out_of_scope(balls: pd.DataFrame) -> None:
    assert not balls.loc[balls["innings"] > 2, "in_scope"].any()


def test_bowler_runs_exclude_byes(balls: pd.DataFrame) -> None:
    byes = balls["extras_type"].isin(["byes", "legbyes"])
    assert (balls.loc[byes, "bowler_runs_conceded"] == balls.loc[byes, "runs"]).all()
