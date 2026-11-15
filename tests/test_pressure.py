"""Checks on win probability, transitions and leverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.config import cfg
from common.io import read_parquet, read_table


@pytest.fixture(scope="module")
def balls() -> pd.DataFrame:
    return read_parquet("pressure", "balls_pressure.parquet")


def test_wp_in_unit_interval(balls: pd.DataFrame) -> None:
    assert balls["wp_pre"].between(0, 1).all()
    assert balls["wp_post"].between(0, 1).all()


def test_innings_handover_consistent(balls: pd.DataFrame) -> None:
    """End of innings 1 WP equals one minus the chase start WP."""
    last1 = balls[balls["innings"] == 1].groupby("match_id").tail(1).set_index("match_id")["wp_post"]
    first2 = balls[balls["innings"] == 2].groupby("match_id").head(1).set_index("match_id")["wp_pre"]
    j = pd.concat([last1, first2], axis=1, keys=["a", "b"]).dropna()
    assert np.allclose(j["a"], 1 - j["b"], atol=1e-6)


def test_chase_ends_on_result(balls: pd.DataFrame) -> None:
    last2 = balls[balls["innings"] == 2].groupby("match_id").tail(1)
    assert last2["wp_post"].isin([0.0, 0.5, 1.0]).all()


def test_li_normalised(balls: pd.DataFrame) -> None:
    assert abs(balls["li"].mean() - 1) < 1e-9
    assert balls["li_w"].max() <= cfg()["li"]["cap"]


def test_tier_shares(balls: pd.DataFrame) -> None:
    share = balls["is_pressure"].mean()
    assert abs(share - (1 - cfg()["li"]["tiers"]["high"])) < 0.01


def test_wp_targets_inn2() -> None:
    m = read_table("pressure", "wp_metrics.csv").set_index("innings")
    t = cfg()["wp"]["targets"]
    assert m.loc[2, "auc"] >= t["inn2_auc"]
    assert m.loc[2, "brier"] <= t["inn2_brier"]
    assert t["slope_lo"] <= m.loc[2, "cal_slope"] <= t["slope_hi"]


def test_monotonic() -> None:
    assert (read_table("pressure", "wp_monotonic_checks.csv")["violations"] == 0).all()
