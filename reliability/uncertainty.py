"""Bootstrap uncertainty, rank intervals and split-half reliability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common.config import cfg
from common.io import log_run, read_parquet, read_table, write_parquet, write_table
from player_ratings import ratings as R
from player_ratings.baseline import COMPONENTS

RAW = {"boundary": "y_boundary", "dot": "y_dot", "out": "y_out", "runs": "y_runs"}


def _half_stats(d: pd.DataFrame, use: str) -> pd.DataFrame:
    cols = {}
    w = np.where(d["is_pressure"], d["li_w"], 0.0)
    for c in COMPONENTS:
        col = f"r_{c}" if use == "resid" else RAW[c]
        x = d[col].to_numpy(float)
        g = pd.DataFrame({"p": d["player_id"].to_numpy(), "n": (~d["is_pressure"]).to_numpy(float),
                          "sx": np.where(d["is_pressure"], 0, x), "w": w, "wx": w * x}).groupby("p").sum()
        cols[f"level_{c}"] = g["sx"] / g["n"]
        cols[f"delta_{c}"] = g["wx"] / g["w"] - g["sx"] / g["n"]
        cols["n"], cols["np"] = g["n"], (d[d["is_pressure"]].groupby("player_id").size())
    return pd.DataFrame(cols)


def split_half(d: pd.DataFrame, n_splits: int, min_half: int, seed: int, use: str = "raw") -> pd.DataFrame:
    """Random match halves, correlation of each metric with itself across players."""
    rng = np.random.default_rng(seed)
    matches = d["match_id"].unique()
    rows = []
    for s in range(n_splits):
        a = set(rng.choice(matches, len(matches) // 2, replace=False))
        in_a = d["match_id"].isin(a)
        A, B = _half_stats(d[in_a], use), _half_stats(d[~in_a], use)
        j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j["n_a"] >= min_half) & (j["n_b"] >= min_half) & (j["np_a"] >= min_half / 3)
              & (j["np_b"] >= min_half / 3)]
        for c in COMPONENTS:
            for kind in ("level", "delta"):
                rows.append({"split": s, "component": c, "kind": kind, "players": len(j),
                             "r": j[f"{kind}_{c}_a"].corr(j[f"{kind}_{c}_b"])})
    return pd.DataFrame(rows)


def split_half_summary(sh: pd.DataFrame) -> pd.DataFrame:
    return sh.groupby(["role", "metric_basis", "component", "kind"]).agg(
        r_mean=("r", "mean"), r_lo=("r", "min"), r_hi=("r", "max"), players=("players", "mean")).reset_index()


def _boot_once(pm: pd.DataFrame, matches: np.ndarray, k: dict, seed: int, players: pd.Index) -> pd.DataFrame:
    """One match-level resample, players rated from reweighted sufficient sums."""
    rng = np.random.default_rng(seed)
    draw = rng.choice(matches, len(matches), replace=True)
    mult = pd.Series(draw).value_counts()
    m = pm.index.get_level_values("match_id").map(mult).fillna(0).to_numpy()
    ps = pm.mul(m, axis=0).groupby(level="player_id").sum().reindex(players).fillna(0)
    if cfg()["bootstrap"]["refit_k"]:
        k = R.estimate_k(ps, cfg()["eligibility"]["min_pressure_balls"])
    rt = R.rate(ps, k)
    return rt[["prism"]].assign(rank=rt["prism"].rank(ascending=False, method="first"))


def bootstrap(role: str, B: int, n_jobs: int, seed: int) -> pd.DataFrame:
    """Prism and rank for every eligible player across B match resamples, k held fixed by default."""
    d = read_parquet("player_ratings", f"balls_{role}.parquet")
    res = read_parquet("player_ratings", f"player_ratings_{role}.parquet")
    v = read_table("player_ratings", f"component_values_{role}.csv")
    pm = R.player_match_stats(d, v)
    ps = pm.groupby(level="player_id").sum()
    k = R.estimate_k(ps, cfg()["eligibility"]["min_pressure_balls"])
    elig = pd.Index(res.loc[res["eligible"], "player_id"])
    pm = pm[pm.index.get_level_values("player_id").isin(elig)]
    matches = d["match_id"].unique()
    seeds = np.random.default_rng(seed).integers(0, 2**31, B)
    # Threads, not processes, process workers fail without a console on Windows
    outs = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_boot_once)(pm, matches, k, int(s), elig) for s in seeds)
    return pd.concat([o.assign(b=i) for i, o in enumerate(outs)]).reset_index()


def summarise(boot: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    g = boot.groupby("player_id")
    return pd.DataFrame({
        "prism_mean": g["prism"].mean(), "prism_se": g["prism"].std(),
        "prism_p10": g["prism"].quantile(0.10), "prism_p90": g["prism"].quantile(0.90),
        "median_rank": g["rank"].median(), "rank_lo80": g["rank"].quantile(0.10),
        "rank_hi80": g["rank"].quantile(0.90), "p_top10": g["rank"].apply(lambda r: (r <= top).mean()),
    }).reset_index()


def run(dev: bool = False) -> None:
    c = cfg()
    bc, sc = c["bootstrap"], c["split_half"]
    B = bc["dev_B"] if dev else bc["B"]
    sh_all = []
    for role in R.ROLES:
        d = read_parquet("player_ratings", f"balls_{role}.parquet")
        for basis in ("raw", "resid"):
            sh = split_half(d, sc["n_splits"], sc["min_balls_each_half"], c["seed"], basis)
            sh_all.append(sh.assign(role=role, metric_basis=basis))
        boot = bootstrap(role, B, bc["n_jobs"], c["seed"])
        write_parquet(boot, "reliability", f"bootstrap_{role}.parquet")
        write_table(summarise(boot), "reliability", f"bootstrap_summary_{role}.csv")
    sh = pd.concat(sh_all)
    write_table(sh, "reliability", "split_half_all.csv")
    write_table(split_half_summary(sh), "reliability", "split_half_summary.csv")
    log_run("uncertainty", {"B": B, "refit_k": bc["refit_k"], "split_half": sc})


if __name__ == "__main__":
    run()


def final_order(role: str) -> pd.DataFrame:
    """Median bootstrap rank first, then P10 of PRISM, then consistency."""
    res = read_parquet("player_ratings", f"player_ratings_{role}.parquet")
    s = read_table("reliability", f"bootstrap_summary_{role}.csv")
    con = read_table("reliability", f"consistency_{role}.csv")
    j = res[res["eligible"]].merge(s, on="player_id").merge(con, on="player_id", how="left")
    j = j.sort_values(["median_rank", "prism_p10", "consistency_index"], ascending=[True, False, False])
    j["rank"] = np.arange(1, len(j) + 1)
    return j.reset_index(drop=True)
