"""Match-level consistency, downside floor and season stability for eligible players."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import cfg
from common.io import log_run, read_parquet, read_table, write_table
from player_ratings import ratings as R
from player_ratings.baseline import COMPONENTS


def match_contributions(d: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """LI weighted WP value of player residuals in each match, pressure balls only."""
    p = d[d["is_pressure"]].merge(v, on=["innings", "phase"], how="left")
    val = sum(p[f"v_{c}"].fillna(0) * p[f"r_{c}"] for c in COMPONENTS)
    p = p.assign(contrib=p["li_w"] * val)
    return p.groupby(["player_id", "match_id"]).agg(contrib=("contrib", "sum"), balls=("contrib", "size"),
                                                    season=("season", "first")).reset_index()


def consistency_table(mc: pd.DataFrame) -> pd.DataFrame:
    c = cfg()["consistency"]
    q = mc[mc["balls"] >= c["min_pressure_balls_per_match"]]

    def floor(x: pd.Series) -> float:
        n = max(1, int(np.ceil(len(x) * c["floor_share"])))
        return float(np.sort(x.to_numpy())[:n].mean())

    g = q.groupby("player_id")["contrib"]
    return pd.DataFrame({"qual_matches": g.size(), "consistency_index": g.apply(lambda x: (x > 0).mean()),
                         "floor": g.apply(floor), "median_match": g.median()}).reset_index()


def season_prism(d: pd.DataFrame, v: pd.DataFrame, k: dict) -> pd.DataFrame:
    rows = []
    for s, g in d.groupby("season"):
        rt, _ = R.compute(g, v, k=k)
        rows.append(rt[["prism", "n_press"]].assign(season=s))
    return pd.concat(rows).reset_index()


def run() -> None:
    for role in R.ROLES:
        d = read_parquet("player_ratings", f"balls_{role}.parquet")
        v = read_table("player_ratings", f"component_values_{role}.csv")
        res = read_parquet("player_ratings", f"player_ratings_{role}.parquet")
        elig = set(res.loc[res["eligible"], "player_id"])
        _, k = R.compute(d, v)
        mc = match_contributions(d, v)
        mc = mc[mc["player_id"].isin(elig)]
        write_table(mc, "reliability", f"match_contributions_{role}.csv")
        write_table(consistency_table(mc), "reliability", f"consistency_{role}.csv")
        sp = season_prism(d[d["player_id"].isin(elig)], v, k)
        write_table(sp, "reliability", f"season_prism_{role}.csv")
    log_run("consistency", cfg()["consistency"])


if __name__ == "__main__":
    run()
