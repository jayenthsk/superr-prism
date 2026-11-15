"""Checks on baseline residuals, shrinkage, eligibility and the final lists."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.config import ROOT, cfg
from common.io import read_parquet, read_table
from player_ratings import ratings as R


@pytest.fixture(scope="module", params=["bat", "bowl"])
def role(request) -> str:
    return request.param


def test_residuals_centred(role: str) -> None:
    d = read_parquet("player_ratings", f"balls_{role}.parquet")
    for c in ("boundary", "dot", "runs"):
        assert abs(d[f"r_{c}"].mean()) < 0.01 * max(1.0, d[f"y_{c}"].mean())


def test_shrinkage_pulls_small_samples(role: str) -> None:
    """A player with five balls of perfect output must sit close to zero."""
    ps = pd.DataFrame({c: [0.0] for c in ["n_norm", "sw", "sw2", "n_press"]})
    ps["n_norm"], ps["sw"], ps["sw2"], ps["n_press"] = 5.0, 5.0, 5.0, 5.0
    k = {}
    for c in R.COMPONENTS:
        ps[f"s_{c}"], ps[f"swr_{c}"], ps[f"swv_{c}"] = 5.0, 5.0, 5.0
        k[f"kN_{c}"], k[f"kD_{c}"] = 300.0, 2000.0
    out = R.rate(ps, k)
    assert (out[[f"thetaN_{c}" for c in R.COMPONENTS]].abs() < 0.02).all().all()


def test_k_delta_capped(role: str) -> None:
    k = read_table("player_ratings", "shrinkage_constants.csv").set_index("role")
    cols = [c for c in k.columns if c.startswith("kD_")]
    assert (k.loc[role, cols] <= cfg()["ratings"]["k_delta_max"]).all()


def test_eligible_meet_thresholds(role: str) -> None:
    r = read_parquet("player_ratings", f"player_ratings_{role}.parquet")
    e = cfg()["eligibility"]
    el = r[r["eligible"]]
    assert (el["total_balls"] >= e["min_balls"]).all()
    assert (el["pressure_balls"] >= e["min_pressure_balls"]).all()
    assert (el["matches_with_pressure"] >= e["min_pressure_matches"]).all()


@pytest.mark.parametrize("name", ["batters", "bowlers"])
def test_top10_lists(name: str) -> None:
    t = pd.read_csv(ROOT / "final_rankings" / f"top10_{name}.csv")
    full = pd.read_csv(ROOT / "final_rankings" / f"full_{name}.csv")
    assert len(t) == 10 and t["player_id"].is_unique
    assert list(t["player_id"]) == list(full["player_id"][:10])
    assert np.all(np.diff(t["median_rank"]) >= 0)


@pytest.mark.parametrize("name", ["batters", "bowlers"])
def test_dashboard_matches_submission(name: str) -> None:
    """The app shows exactly the submitted lists."""
    a = pd.read_csv(ROOT / "final_rankings" / f"top10_{name}.csv")
    b = pd.read_csv(ROOT / "dashboard" / "data" / f"top10_{name}.csv")
    assert list(a["player_id"]) == list(b["player_id"])
