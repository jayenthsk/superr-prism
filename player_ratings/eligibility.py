"""Sample size rules, every player is rated but only qualifiers can make a top 10."""
from __future__ import annotations

import pandas as pd

from common.config import cfg


def sample_sizes(d: pd.DataFrame) -> pd.DataFrame:
    legal = d[d["is_legal"]]
    out = pd.DataFrame({
        "total_balls": legal.groupby("player_id").size(),
        "pressure_balls": d[d["is_pressure"]].groupby("player_id").size(),
        "matches": d.groupby("player_id")["match_id"].nunique(),
        "matches_with_pressure": d[d["is_pressure"]].groupby("player_id")["match_id"].nunique(),
        "last_season": d.groupby("player_id")["season"].max(),
    }).fillna(0)
    return out.astype({"total_balls": int, "pressure_balls": int, "matches_with_pressure": int})


def apply_rules(sizes: pd.DataFrame, min_pressure_balls: int | None = None,
                active_since: int | None | bool = None) -> pd.Series:
    e = cfg()["eligibility"]
    mpb = e["min_pressure_balls"] if min_pressure_balls is None else min_pressure_balls
    act = e["active_since"] if active_since is None else active_since
    ok = (sizes["total_balls"] >= e["min_balls"]) & (sizes["pressure_balls"] >= mpb) \
        & (sizes["matches_with_pressure"] >= e["min_pressure_matches"])
    if act:
        ok &= sizes["last_season"] >= act
    return ok


def pool_sizes(sizes_by_role: dict[str, pd.DataFrame]) -> pd.DataFrame:
    e = cfg()["eligibility"]
    rows = []
    for role, s in sizes_by_role.items():
        m1 = s["total_balls"] >= e["min_balls"]
        m2 = m1 & (s["pressure_balls"] >= e["min_pressure_balls"])
        m3 = m2 & (s["matches_with_pressure"] >= e["min_pressure_matches"])
        m4 = m3 & (s["last_season"] >= e["active_since"])
        rows.append({"role": role, "all_players": len(s), "min_balls": int(m1.sum()),
                     "plus_pressure_balls": int(m2.sum()), "plus_pressure_matches": int(m3.sum()),
                     "plus_active": int(m4.sum()), "with_500_balls": int((s["total_balls"] >= 500).sum())})
    return pd.DataFrame(rows)
