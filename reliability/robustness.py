"""Sensitivity variants, top 10 overlap and rank agreement against the main ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.config import cfg
from common.io import log_run, read_parquet, read_table, write_table
from player_ratings import eligibility
from player_ratings import ratings as R
from reliability.uncertainty import final_order


def _retier(d: pd.DataFrame, q: float, li_all: pd.Series) -> pd.DataFrame:
    cut = li_all.quantile(q)
    return d.assign(is_pressure=d["li"] >= cut)


def variants() -> list[dict]:
    c = cfg()
    rb = c["robustness"]
    out = [{"name": f"Tier p{int(q * 100)}", "tier": q} for q in rb["tiers"] if q != c["li"]["tiers"]["high"]]
    out += [{"name": f"LI cap {cap if cap else 'none'}", "opts": {"li_cap": cap}}
            for cap in rb["li_caps"] if cap != c["li"]["cap"]]
    out += [
        {"name": "Knockout x1.25", "opts": {"knockout": 1.25}},
        {"name": "No associate T20Is", "filter": lambda d: ~d["has_associate"]},
        {"name": "IPL only", "filter": lambda d: d["comp_type"] == "IPL"},
        {"name": "T20I only", "filter": lambda d: d["comp_type"].str.startswith("T20I")},
        {"name": "No opposition adjustment", "resid": "r1_"},
        {"name": "Recency half-life 2", "opts": {"recency_half_life": 2}},
        {"name": "Tempo pressure", "tempo": True},
        {"name": "Hard entry x1.2", "opts": {"hard_entry_weight": c["extras"]["hard_entry_weight"]}, "role": "bat"},
        {"name": "Not requiring 2025 activity", "active": False},
    ]
    out += [{"name": f"Min pressure balls {m}", "min_pb": m} for m in c["eligibility_grid"]
            if m != c["eligibility"]["min_pressure_balls"]]
    return out


def run_variant(d: pd.DataFrame, v: pd.DataFrame, var: dict, li_all: pd.Series) -> pd.DataFrame:
    if "filter" in var:
        d = d[var["filter"](d)]
    if "tier" in var:
        d = _retier(d, var["tier"], li_all)
    min_pb = var.get("min_pb")
    if var.get("tempo"):
        d = d.assign(is_pressure=d["is_revealed_pressure"].fillna(False).astype(bool))
        # Slow tempo flags far fewer balls, so scale the pressure ball minimum to the same share
        share = d["is_pressure"].mean() / (1 - cfg()["li"]["tiers"]["high"])
        min_pb = int(round(cfg()["eligibility"]["min_pressure_balls"] * share))
    rt, _ = R.compute(d, v, opts=var.get("opts"), resid_prefix=var.get("resid", "r_"))
    sizes = eligibility.sample_sizes(d)
    ok = eligibility.apply_rules(sizes, min_pb, var.get("active"))
    return rt.join(ok.rename("eligible"))


def compare(main: pd.DataFrame, var_rt: pd.DataFrame, top: int = 10) -> dict:
    el = var_rt[var_rt["eligible"]].sort_values("prism", ascending=False)
    vtop = set(el.index[:top])
    mtop = set(main["player_id"][:top])
    both = main[main["player_id"].isin(el.index)]
    rho = spearmanr(both["rank"], -el.loc[both["player_id"], "prism"]).statistic if len(both) > 3 else np.nan
    return {"overlap": len(vtop & mtop), "spearman": rho, "pool": len(el), "top": vtop}


def method_agreement(role: str, main: pd.DataFrame, d: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """Agreement of the final order with simpler or narrower rankings."""
    raw_col, asc = ("sr_P", False) if role == "bat" else ("econ_P", True)
    alts = {"Raw pressure " + ("strike rate" if role == "bat" else "economy"):
            main.set_index("player_id")[raw_col] * (1 if not asc else -1)}
    for name, var in [("No opposition adjustment", {"resid": "r1_"}),
                      ("IPL only", {"filter": lambda x: x["comp_type"] == "IPL"})]:
        alts[name] = run_variant(d, v, var, d["li"])["prism"]
    rows = []
    for name, score in alts.items():
        s = score.reindex(main["player_id"])
        ok = s.notna().to_numpy()
        rows.append({"role": role, "alternative": name,
                     "spearman": spearmanr(main["rank"][ok], -s[ok]).statistic, "players": int(ok.sum())})
    return pd.DataFrame(rows)


def run() -> None:
    li_all = read_parquet("pressure", "balls_pressure.parquet")["li"]
    rows, members, agree = [], [], []
    for role in R.ROLES:
        d = read_parquet("player_ratings", f"balls_{role}.parquet")
        v = read_table("player_ratings", f"component_values_{role}.csv")
        main = final_order(role)
        for var in variants():
            if var.get("role", role) != role:
                continue
            res = compare(main, run_variant(d, v, var, li_all))
            rows.append({"role": role, "variant": var["name"], "top10_overlap": res["overlap"],
                         "spearman": res["spearman"], "eligible_pool": res["pool"]})
            members += [{"role": role, "variant": var["name"], "player_id": p} for p in res["top"]]
        agree.append(method_agreement(role, main, d, v))
    rob = pd.DataFrame(rows)
    mem = pd.DataFrame(members)
    write_table(rob, "reliability", "robustness.csv")
    write_table(mem, "reliability", "robustness_membership.csv")
    write_table(pd.concat(agree), "reliability", "method_agreement.csv")
    log_run("robustness", {"variants": rob["variant"].unique().tolist()})


def robust_share(role: str) -> pd.Series:
    mem = read_table("reliability", "robustness_membership.csv")
    rob = read_table("reliability", "robustness.csv")
    n = rob.loc[rob["role"] == role, "variant"].nunique()
    return mem[mem["role"] == role].groupby("player_id").size() / n


if __name__ == "__main__":
    run()
