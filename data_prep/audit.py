"""Audit and match-context tables, every data quirk checked with evidence."""
from __future__ import annotations

import pandas as pd

from common.config import cfg
from common.io import read_parquet, write_table
from data_prep.clean import load_raw


def inventory(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([{"file": f"{k}.csv", "rows": len(v), "columns": v.shape[1],
                          "missing_cells_pct": 100 * v.isna().mean().mean()} for k, v in raw.items()])


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_pct": 100 * df.isna().mean(),
                         "unique": df.nunique()}).reset_index(names="column")


def quirk_checks(raw: dict[str, pd.DataFrame], balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per data quirk with the number that confirms it."""
    b, c = raw["ball_by_ball"], raw["competitions"]
    reg = balls[balls["innings"] <= 2]
    legal_max = reg[reg["is_legal"]].groupby(["match_id", "innings"]).size().max()
    no_other = reg[~reg["extras_type"].isin(["wide", "Overstep"])].groupby(["match_id", "innings"]).size()
    names = pd.concat([b[["striker_id", "striker_name"]].set_axis(["id", "name"], axis=1),
                       b[["bowler_id", "bowler_name"]].set_axis(["id", "name"], axis=1)]).drop_duplicates()
    dup = names.groupby("name")["id"].nunique()
    dls_text = matches["result"].str.contains("DLS|D/L", case=False, na=False)
    no_revised = matches[["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]].isna().all(axis=1)
    gaps = balls["gap_s"]
    rows = [
        ("Q1", "T20 label competitions", int((c["match_format"] == "T20").sum()),
         "IPL 2022 to 2026 plus two emerging events"),
        ("Q2", "Bat runs only in runs column, extras separate", float(b.loc[b["extras_type"] == "wide", "runs"].sum()),
         "zero bat runs on wides"),
        ("Q3", "Max legal balls per innings with other treated as no-ball", int(legal_max), "120 expected"),
        ("Q3", "Innings above 120 legal balls if other were legal", int((no_other > 120).sum()), "other is a no-ball"),
        ("Q3", "Mean bat runs on other extras", float(b.loc[b["extras_type"] == "other", "runs"].mean()),
         "batter scores off it like a no-ball"),
        ("Q4", "Super over balls", int((b["innings"] > 2).sum()), "kept apart"),
        ("Q5", "Obstructing typo rows", int((b["dismissal_type"] == "Obstracting the field").sum()), "normalised"),
        ("Q6", "Run-outs of the non-striker", int(balls["is_nonstriker_out"].sum()), "charged by dismissed id"),
        ("Q7", "No result or abandoned matches", int(matches["is_no_result"].sum()), "dropped"),
        ("Q7", "Decided matches with revised overs or DLS", int((matches["is_dls"] & ~matches["is_no_result"]).sum()),
         "excluded in v1, revised overs columns are held in balls"),
        ("Q7", "DLS results with no revised fields", int((dls_text & no_revised).sum()),
         "caught by result text"),
        ("Q8", "Distinct result wordings for wickets or runs", int(matches["result"].str.extract(
            r"(Wkts|wickets|Wickets|Runs|runs|run)")[0].nunique()), "use win team id"),
        ("Q9", "Ball gaps valid for tempo, share", float(gaps.notna().mean()), "5 to 240 seconds within an over"),
        ("Q9", "Median valid gap seconds", float(gaps.median()), "about 37 expected"),
        ("Q10", "Stroke filled share", float(b["stroke"].notna().mean()), "about 0.97"),
        ("Q10", "Bowl speed non zero share", float((b["bowl_speed"] > 0).mean()), "about 0.61"),
        ("Q11", "Names mapped to more than one ID", int((dup > 1).sum()), ", ".join(dup[dup > 1].index)),
        ("Q12", "Ball batting order filled share", float(b["batting_order"].notna().mean()), "slot baselines"),
        ("Q13", "IPL impact era matches", int(matches["ipl_impact_era"].sum()), "IPL from 2023"),
        ("Q14", "Associate bowling side share of T20I balls",
         float(balls.loc[balls["comp_class"] == "T20I"].pipe(
             lambda x: x["bowling_team_id"].isin(_associate_ids(raw))).mean()), "about 0.07"),
    ]
    return pd.DataFrame(rows, columns=["quirk", "check", "value", "note"])


def _associate_ids(raw: dict[str, pd.DataFrame]) -> set:
    assoc = {t.lower() for t in cfg()["associate_teams"]}
    t = raw["teams"]
    return set(t.loc[t["team_name"].str.lower().isin(assoc), "team_id"])


def coverage_by_comp(balls: pd.DataFrame) -> pd.DataFrame:
    """Tagged field coverage by comp type, providers differ."""
    return balls.groupby("comp_type").agg(
        balls=("match_id", "size"), stroke=("stroke", lambda s: s.notna().mean()),
        delivery_style=("delivery_style", lambda s: s.notna().mean()),
        speed=("bowl_speed", lambda s: s.notna().mean()), gap=("gap_s", lambda s: s.notna().mean())).reset_index()


def matches_by_season(matches: pd.DataFrame) -> pd.DataFrame:
    return matches.pivot_table(index="season", columns="comp_type", values="match_id", aggfunc="count",
                               fill_value=0).reset_index()


def scoring_by_phase(balls: pd.DataFrame) -> pd.DataFrame:
    s = balls[balls["in_scope"]]
    return s.groupby(["comp_type", "phase"]).agg(runs_per_over=("total_runs", lambda x: 6 * x.sum()),
                                                 legal=("is_legal", "sum"),
                                                 wkts=("is_wicket", "sum")).assign(
        runs_per_over=lambda x: x["runs_per_over"] / x["legal"],
        balls_per_wkt=lambda x: x["legal"] / x["wkts"]).reset_index()


def impact_era(balls: pd.DataFrame) -> pd.DataFrame:
    s = balls[balls["in_scope"] & (balls["comp_type"] == "IPL")]
    t = s.groupby(["season", "innings"]).agg(runs=("total_runs", "sum"), legal=("is_legal", "sum"))
    return t.assign(run_rate=6 * t["runs"] / t["legal"]).reset_index()


def innings_totals(balls: pd.DataFrame) -> pd.DataFrame:
    s = balls[balls["in_scope"]]
    return s.groupby(["match_id", "innings", "comp_type"])["total_runs"].sum().reset_index()


def chase_success(balls: pd.DataFrame) -> pd.DataFrame:
    s = balls[balls["in_scope"] & (balls["innings"] == 2) & ~balls["is_tie"]]
    m = s.groupby("match_id").agg(target=("target", "first"), won=("is_bat_team_winner", "first"),
                                  comp_type=("comp_type", "first"))
    m["band"] = pd.cut(m["target"], [0, 140, 160, 180, 200, 220, 400],
                       labels=["up to 140", "141-160", "161-180", "181-200", "201-220", "221 plus"])
    return m.groupby("band", observed=True).agg(matches=("won", "size"),
                                                chase_win=("won", lambda x: x.astype(float).mean())).reset_index()


def venue_index(balls: pd.DataFrame) -> pd.DataFrame:
    s = balls[balls["in_scope"] & (balls["innings"] == 1)]
    m = s.groupby("match_id").agg(venue_id=("venue_id", "first"), total=("total_runs", "sum"))
    v = m.groupby("venue_id")["total"].agg(["mean", "count"])
    return v[v["count"] >= 5].sort_values("mean", ascending=False).reset_index()


def slot_balls(balls: pd.DataFrame) -> pd.DataFrame:
    s = balls[balls["in_scope"] & ~balls["is_wide"]]
    return s.groupby(["batting_order", "phase"]).size().unstack(fill_value=0).reset_index()


def pace_spin_by_phase(balls: pd.DataFrame) -> pd.DataFrame:
    s = balls[balls["in_scope"]]
    return pd.crosstab([s["comp_type"], s["phase"]], s["pace_or_spin"], normalize="index").reset_index()


def run() -> None:
    raw = load_raw()
    balls = read_parquet("data_prep", "balls_clean.parquet")
    matches = read_parquet("data_prep", "matches_clean.parquet")
    write_table(inventory(raw), "data_prep", "inventory.csv")
    write_table(quirk_checks(raw, balls, matches), "data_prep", "quirk_checks.csv")
    write_table(coverage_by_comp(balls), "data_prep", "coverage_by_comp.csv")
    write_table(matches_by_season(matches), "data_prep", "matches_by_season.csv")
    write_table(scoring_by_phase(balls), "data_prep", "scoring_by_phase.csv")
    write_table(impact_era(balls), "data_prep", "impact_era.csv")
    write_table(chase_success(balls), "data_prep", "chase_success.csv")
    write_table(pace_spin_by_phase(balls), "data_prep", "pace_spin_by_phase.csv")


if __name__ == "__main__":
    run()
