"""Leverage index for every ball and the pressure tiers built on it."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from common.config import cfg
from common.io import log_run, read_parquet, write_parquet, write_table
from pressure.wp_model import WPModel, inn2_start_state, load_model

OUTCOMES = {"dot": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "wicket": 0, "extra": 1}


def wkt_bucket(w: pd.Series) -> pd.Series:
    edges = cfg()["li"]["wkt_buckets"]
    out = pd.Series("", index=w.index)
    for lo, hi in edges:
        out[w.between(lo, hi)] = f"{lo}-{hi}"
    return out


def outcome_label(b: pd.DataFrame) -> pd.Series:
    r = b["total_runs"].clip(upper=6).replace({5: 4})
    lab = r.astype(int).astype(str).replace({"0": "dot"})
    lab[b["is_wicket"]] = "wicket"
    lab[b["is_wide"] | b["is_noball"]] = "extra"
    return lab


def outcome_rates(b: pd.DataFrame) -> pd.DataFrame:
    d = b.assign(outcome=outcome_label(b), wband=wkt_bucket(b["wkts_before"]))
    t = pd.crosstab([d["comp_type"], d["phase"], d["wband"]], d["outcome"], normalize="index")
    return t.reindex(columns=list(OUTCOMES), fill_value=0.0)


def post_state_wp(b: pd.DataFrame, outcome: str, model: WPModel, use_folds: bool = True) -> np.ndarray:
    """Batting side WP after a hypothetical outcome, terminal states handled by cricket rules.

    A wicket at nine down or the last legal ball ends the innings. Innings 1 then hands over
    to one minus the chase start WP, a finished chase is won, lost or tied outright.
    """
    runs = OUTCOMES[outcome]
    legal = outcome != "extra"
    s = b.copy()
    s["score_before"] = s["score_before"] + runs
    s["wkts_before"] = s["wkts_before"] + (outcome == "wicket")
    s["legal_before"] = s["legal_before"] + legal
    s["balls_left"] = 120 - s["legal_before"]
    s["crr"] = s["score_before"] * 6 / s["legal_before"].clip(lower=6)
    ends = (s["wkts_before"] >= 10) | (s["balls_left"] <= 0)
    s["wkts_before"] = s["wkts_before"].clip(upper=9)
    s["balls_left"] = s["balls_left"].clip(lower=1)
    out = np.empty(len(s))
    i1 = (s["innings"] == 1).to_numpy()
    if i1.any():
        d = s[i1]
        wp = model.predict(d, 1, use_folds)
        e = ends[i1].to_numpy()
        if e.any():
            start = inn2_start_state(d[e].assign(inn1_final=d.loc[e, "score_before"]))
            wp[e] = 1 - model.predict(start, 2, use_folds)
        out[i1] = wp
    if (~i1).any():
        d = s[~i1].copy()
        d["runs_needed"] = d["target"] - d["score_before"]
        d["rrr"] = d["runs_needed"] * 6 / d["balls_left"]
        wp = model.predict(d, 2, use_folds)
        won = (d["runs_needed"] <= 0).to_numpy()
        e = ends[~i1].to_numpy()
        tie = e & (d["runs_needed"] == 1).to_numpy()
        wp[e] = 0.0
        wp[tie] = 0.5
        wp[won] = 1.0
        out[~i1] = wp
    return out


def compute_leverage(b: pd.DataFrame, model: WPModel) -> pd.DataFrame:
    """Expected win probability swing from this state, scaled so the average ball is 1."""
    c = cfg()["li"]
    rates = outcome_rates(b)
    key = pd.MultiIndex.from_arrays([b["comp_type"], b["phase"], wkt_bucket(b["wkts_before"])])
    p = rates.reindex(key).fillna(0).to_numpy()
    raw = np.zeros(len(b))
    for j, o in enumerate(OUTCOMES):
        raw += p[:, j] * np.abs(post_state_wp(b, o, model) - b["wp_pre"].to_numpy())
    out = b.copy()
    out["raw_li"] = raw
    out["li"] = raw / raw.mean()
    out["li_w"] = out["li"].clip(upper=c["cap"]) if c["cap"] else out["li"]
    q = out["li"].quantile([c["tiers"]["low"], c["tiers"]["high"], c["tiers"]["extreme"]]).to_numpy()
    out["pressure_tier"] = pd.cut(out["li"], [-np.inf, q[0], q[1], q[2], np.inf], right=False,
                                  labels=["Low", "Normal", "High", "Extreme"]).astype(str)
    out["is_pressure"] = out["pressure_tier"].isin(["High", "Extreme"])
    out.attrs["li_cuts"] = dict(zip(["p_low", "p_high", "p_extreme"], q.tolist(), strict=True))
    return out


def threshold_comparison(b: pd.DataFrame) -> pd.DataFrame:
    """How a scoreboard threshold mislabels pressure compared with leverage."""
    i2 = b[b["innings"] == 2]
    rrr10 = i2[i2["rrr"] >= 10]
    hi = b[b["is_pressure"]]
    rows = [
        ("Chase balls with RRR 10 plus at WP under 0.1 or over 0.9",
         ((rrr10["wp_pre"] < 0.1) | (rrr10["wp_pre"] > 0.9)).mean()),
        ("Chase balls with RRR 10 plus that are Low or Normal leverage", (~rrr10["is_pressure"]).mean()),
        ("High leverage balls in overs 1 to 15", (hi["over_no"] <= 15).mean()),
        ("High leverage balls in innings 1", (hi["innings"] == 1).mean()),
        ("Death over balls that are High leverage", b.loc[b["phase"] == "DEATH", "is_pressure"].mean()),
    ]
    return pd.DataFrame(rows, columns=["check", "share"])


def separation_test(b: pd.DataFrame) -> pd.DataFrame:
    """Actual WP swing is larger on High balls, KS test against Normal."""
    a = b.loc[b["is_pressure"], "dwp"].abs().dropna()
    n = b.loc[b["pressure_tier"] == "Normal", "dwp"].abs().dropna()
    ks = ks_2samp(a, n)
    return pd.DataFrame([{"mean_abs_dwp_high": a.mean(), "mean_abs_dwp_normal": n.mean(),
                          "ratio": a.mean() / n.mean(), "ks_stat": ks.statistic, "ks_p": ks.pvalue}])


def _stratified_diff(b: pd.DataFrame, value: pd.Series, cells: list[str]) -> tuple[float, float, float]:
    """Pressure minus Normal within matching cells, weighted by the pressure ball mix."""
    d = b.assign(v=value.astype(float))
    d = d[d["pressure_tier"].isin(["Normal", "High", "Extreme"])]
    g = d.groupby(cells + ["is_pressure"])["v"].agg(["mean", "size"]).unstack("is_pressure").dropna()
    w = g[("size", True)]
    hi = np.average(g[("mean", True)], weights=w)
    lo = np.average(g[("mean", False)], weights=w)
    return float(lo), float(hi), float(hi - lo)


def behaviour_checks(b: pd.DataFrame) -> pd.DataFrame:
    """Independent signs that players feel pressure where leverage says it is."""
    c = cfg()
    dg, sg = c["delivery_groups"], c["stroke_groups"]
    death_pace = b[(b["phase"] == "DEATH") & (b["pace_or_spin"] == "Pace")]
    death = b[b["phase"] == "DEATH"]
    cells = ["innings", "over_no"]
    rows = []
    for label, d, val in [
        ("Gap between balls, seconds", b[b["gap_s"].notna()], b.loc[b["gap_s"].notna(), "gap_s"]),
        ("Pace death full toss share", death_pace, death_pace["delivery_style"].isin(dg["full_toss"])),
        ("Pace death yorker share", death_pace, death_pace["delivery_style"].isin(dg["yorker"])),
        ("Death aggressive stroke share", death, death["stroke"].isin(sg["aggressive"])),
    ]:
        lo, hi, diff = _stratified_diff(d, val, cells)
        raw_lo = val[~d["is_pressure"]].mean()
        raw_hi = val[d["is_pressure"]].mean()
        rows.append({"signal": label, "normal_raw": raw_lo, "pressure_raw": raw_hi,
                     "normal_matched": lo, "pressure_matched": hi, "matched_diff": diff})
    return pd.DataFrame(rows)


def pressure_tax(b: pd.DataFrame) -> pd.DataFrame:
    """League-wide change in outcome rates on pressure balls, matched on innings, over and wickets."""
    legal = b[b["is_legal"]]
    cells = ["innings", "over_no", "wband"]
    legal = legal.assign(wband=wkt_bucket(legal["wkts_before"]))
    rows = []
    for name, val in [("runs per ball", legal["total_runs"]),
                      ("boundary rate", legal["runs"] >= 4),
                      ("dot rate", legal["total_runs"] == 0),
                      ("wicket rate", legal["is_wicket"])]:
        lo, hi, diff = _stratified_diff(legal, val, cells)
        rows.append({"metric": name, "normal": lo, "pressure": hi, "diff": diff})
    return pd.DataFrame(rows)


def li_by_over(b: pd.DataFrame) -> pd.DataFrame:
    return b.groupby(["innings", "over_no"]).agg(mean_li=("li", "mean"),
                                                 high_share=("is_pressure", "mean")).reset_index()


def _grid_li(g: pd.DataFrame, model: WPModel, rates: pd.DataFrame) -> pd.DataFrame:
    ph = cfg()["phases"]
    g["over_no"] = (g["legal_before"] // 6 + 1).clip(upper=20)
    g["phase"] = np.select([g["over_no"] <= ph["PP"][1], g["over_no"] <= ph["MID"][1]], ["PP", "MID"], "DEATH")
    g["venue_scoring_index"], g["fold"], g["ipl_impact_era"], g["match_id"] = 1.0, 0, True, 0
    g["crr"] = g["score_before"] * 6 / g["legal_before"].clip(lower=6)
    inn = int(g["innings"].iloc[0])
    g["wp_pre"] = model.predict(g, inn, use_folds=False)
    p = rates.reindex(pd.MultiIndex.from_arrays([g["comp_type"], g["phase"], wkt_bucket(g["wkts_before"])]))
    raw = np.zeros(len(g))
    for j, o in enumerate(OUTCOMES):
        raw += p.iloc[:, j].to_numpy() * np.abs(post_state_wp(g, o, model, use_folds=False) - g["wp_pre"].to_numpy())
    g["raw_li"] = raw
    return g


def li_grid(model: WPModel, rates: pd.DataFrame, comp_type: str = "IPL", inn: int = 2) -> pd.DataFrame:
    if inn == 2:
        rows = [(bl, w, rn) for bl in range(6, 121, 6) for w in range(10) for rn in range(6, 241, 6)]
        g = pd.DataFrame(rows, columns=["balls_left", "wkts_before", "runs_needed"])
        g = g[g["runs_needed"] <= g["balls_left"] * 3].copy()
        g["target"] = 200
        g["score_before"] = g["target"] - g["runs_needed"]
        g["rrr"] = g["runs_needed"] * 6 / g["balls_left"]
    else:
        rows = [(bl, w, sc) for bl in range(6, 121, 6) for w in range(10) for sc in range(0, 241, 6)]
        g = pd.DataFrame(rows, columns=["balls_left", "wkts_before", "score_before"])
        g = g[g["score_before"] <= (120 - g["balls_left"]) * 3].copy()
        g["target"] = np.nan
    g["innings"], g["comp_type"] = inn, comp_type
    g["legal_before"] = 120 - g["balls_left"]
    return _grid_li(g, model, rates)


def hard_entry_flags(b: pd.DataFrame) -> pd.DataFrame:
    """Hard entry, the batter walks in to a high leverage ball or just after a wicket in a live game.

    Flags his first ten balls of that innings, the window where a new batter is most exposed.
    """
    e = cfg()["extras"]
    lo, hi = e["hard_entry_wp"]
    first = b[(b["batter_balls_faced_before"] == 0) & ~b["is_wide"]]
    hard = first["is_pressure"] | ((first["balls_since_wkt"] <= 2) & first["wp_pre"].between(lo, hi))
    key = first.assign(hard=hard).drop_duplicates(["match_id", "innings", "striker_id"])
    key = key.set_index(["match_id", "innings", "striker_id"])["hard"]
    idx = pd.MultiIndex.from_arrays([b["match_id"], b["innings"], b["striker_id"]])
    out = b.copy()
    out["is_hard_entry"] = key.reindex(idx).fillna(False).to_numpy().astype(bool)
    out["is_entry_window"] = out["batter_balls_faced_before"] < e["hard_entry_balls"]
    out["is_hard_entry_ball"] = out["is_hard_entry"] & out["is_entry_window"]
    return out


def run() -> None:
    balls = read_parquet("pressure", "balls_wp.parquet")
    model = load_model()
    scope = balls[balls["in_scope"]]
    lev = hard_entry_flags(compute_leverage(scope, model))
    mean_raw = float(lev["raw_li"].mean())
    rates = outcome_rates(scope)
    grid = pd.concat([li_grid(model, rates, inn=i) for i in (1, 2)], ignore_index=True)
    grid["li"] = grid["raw_li"] / mean_raw
    write_parquet(lev, "pressure", "balls_pressure.parquet")
    write_table(rates.reset_index(), "pressure", "li_outcome_rates.csv")
    write_table(threshold_comparison(lev), "pressure", "li_threshold_comparison.csv")
    write_table(separation_test(lev), "pressure", "li_separation.csv")
    write_table(behaviour_checks(lev), "pressure", "li_behaviour_checks.csv")
    write_table(pressure_tax(lev), "pressure", "pressure_tax.csv")
    write_table(li_by_over(lev), "pressure", "li_by_over.csv")
    write_table(grid[grid["innings"] == 2], "pressure", "li_grid_chase.csv")
    write_table(grid, "pressure", "li_grid_all.csv")
    tiers = lev["pressure_tier"].value_counts(normalize=True).rename("share").reset_index()
    write_table(tiers, "pressure", "li_tier_shares.csv")
    cuts = {**lev.attrs["li_cuts"], "mean_raw_li": mean_raw, "cap": cfg()["li"]["cap"]}
    write_table(pd.DataFrame([cuts]), "pressure", "li_cuts.csv")
    log_run("leverage", cuts)


if __name__ == "__main__":
    run()
