"""Submission lists, eligibility applied and ordered by median bootstrap rank."""
from __future__ import annotations

import pandas as pd

from common.config import cfg
from common.io import log_run, stage_dir
from player_ratings.ratings import ROLES
from reliability.robustness import robust_share
from reliability.uncertainty import final_order

COLS = ["rank", "player_id", "player_name", "teams", "comp_types_played", "total_balls", "pressure_balls",
        "matches_with_pressure", "prism", "prism_p10", "prism_p90", "median_rank", "rank_lo80", "rank_hi80",
        "p_top10", "consistency_index", "robust_share", "robust_badge"]
ROLE_NAME = {"bat": "batters", "bowl": "bowlers"}


def build(role: str) -> pd.DataFrame:
    j = final_order(role)
    j["robust_share"] = j["player_id"].map(robust_share(role)).fillna(0)
    j["robust_badge"] = j["robust_share"] >= cfg()["robustness"]["badge_share"]
    return j[COLS]


def run() -> None:
    out = stage_dir("final_rankings")
    n_res = cfg()["reserves"]
    summary = {}
    for role in ROLES:
        full = build(role)
        name = ROLE_NAME[role]
        top = full.head(10)
        assert top["player_id"].is_unique and len(top) == 10
        top.round(4).to_csv(out / f"top10_{name}.csv", index=False)
        full.iloc[10:10 + n_res].round(4).to_csv(out / f"reserves_{name}.csv", index=False)
        full.round(4).to_csv(out / f"full_{name}.csv", index=False)
        summary[name] = top["player_name"].tolist()
    log_run("final_rankings", summary)


if __name__ == "__main__":
    run()
