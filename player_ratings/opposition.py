"""Opponent strength adjustment, second pass of the baseline with opponent skill added."""
from __future__ import annotations

import numpy as np
import pandas as pd

from player_ratings.baseline import COMPONENTS


def shrink_k(sums: pd.Series, sq: pd.Series, n: pd.Series, min_n: int, tau2_floor: float) -> float:
    """Method of moments shrinkage constant, within-player noise over true spread."""
    ok = n >= min_n
    means = sums[ok] / n[ok]
    sigma2 = float(((sq[ok] - n[ok] * means**2).sum()) / max((n[ok] - 1).sum(), 1))
    tau2 = max(float(means.var() - np.mean(sigma2 / n[ok])), tau2_floor)
    return sigma2 / tau2


def oof_opponent_skill(source: pd.DataFrame, target: pd.DataFrame, min_n: int, tau2_floor: float) -> pd.DataFrame:
    """Shrunk normal-ball residual of each opponent from other folds only, so no leakage.

    Source holds the opponents own balls, bowler skill for batter balls and batter skill for bowler balls.
    """
    out = pd.DataFrame(index=target.index)
    normal = source[~source["is_pressure"]]
    key = pd.MultiIndex.from_arrays([target["opp_id"], target["fold"]])
    for comp in COMPONENTS:
        r = f"r_{comp}"
        g = normal.assign(sq=normal[r] ** 2).groupby(["player_id", "fold"]).agg(
            sum=(r, "sum"), size=(r, "size"), sq=("sq", "sum"))
        tot = g.groupby(level="player_id").sum()
        k = shrink_k(tot["sum"], tot["sq"], tot["size"], min_n, tau2_floor)
        full = tot.reindex(target["opp_id"]).fillna(0).to_numpy()
        own = g.reindex(key).fillna(0).to_numpy()
        # Opponent totals minus the fold this ball sits in
        s = full[:, 0] - own[:, 0]
        n = full[:, 1] - own[:, 1]
        out[f"opp_{comp}"] = s / (n + k)
    return out


def rank_changes(before: pd.DataFrame, after: pd.DataFrame, key: str = "prism") -> pd.DataFrame:
    a = before[["player_id", "player_name", key]].rename(columns={key: "before"})
    b = after[["player_id", key]].rename(columns={key: "after"})
    j = a.merge(b, on="player_id")
    j["rank_before"] = j["before"].rank(ascending=False, method="min")
    j["rank_after"] = j["after"].rank(ascending=False, method="min")
    j["rank_change"] = j["rank_before"] - j["rank_after"]
    return j.sort_values("rank_after")
