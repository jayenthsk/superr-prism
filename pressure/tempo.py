"""Tempo signal, fielding sides slow the game down when the stakes rise."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.config import cfg
from common.io import log_run, read_parquet, write_parquet, write_table


def tempo_z(b: pd.DataFrame) -> pd.Series:
    """Gap z score within match innings, removes broadcast and venue pace differences."""
    g = b.groupby(["match_id", "innings"])["gap_s"]
    return (b["gap_s"] - g.transform("mean")) / g.transform("std")


def tempo_vs_li(b: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    d = b[b["tempo_z"].notna()]
    q = pd.qcut(d["li"], bins, duplicates="drop")
    return d.groupby(q, observed=True).agg(li_mid=("li", "median"), tempo_z=("tempo_z", "mean"),
                                           gap_median=("gap_s", "median"), n=("li", "size")).reset_index(drop=True)


def tempo_summary(b: pd.DataFrame) -> pd.DataFrame:
    d = b[b["tempo_z"].notna()]
    rho = spearmanr(d["li"], d["tempo_z"])
    by_tier = d.groupby("pressure_tier")["gap_s"].median()
    return pd.DataFrame([{"spearman_li_tempo": rho.statistic, "p_value": rho.pvalue, "balls": len(d),
                          **{f"median_gap_{k}": v for k, v in by_tier.items()}}])


def revealed_pressure(b: pd.DataFrame) -> pd.Series:
    """Alternative pressure flag from slow tempo, first ball after a wicket left out."""
    z = cfg()["tempo"]["revealed_z"]
    return (b["tempo_z"] >= z) & ~b["after_wicket"]


def run() -> None:
    b = read_parquet("pressure", "balls_pressure.parquet")
    b["tempo_z"] = tempo_z(b)
    b["is_revealed_pressure"] = revealed_pressure(b)
    write_parquet(b, "pressure", "balls_pressure.parquet")
    write_table(tempo_vs_li(b), "pressure", "tempo_vs_li.csv")
    s = tempo_summary(b)
    s["revealed_share"] = float(b["is_revealed_pressure"].mean())
    s["revealed_overlap_with_high"] = float(b.loc[b["is_revealed_pressure"], "is_pressure"].mean())
    write_table(s, "pressure", "tempo_summary.csv")
    log_run("tempo", s.replace({np.nan: None}).to_dict("records")[0])


if __name__ == "__main__":
    run()
