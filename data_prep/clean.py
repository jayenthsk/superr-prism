"""Cleaning rules for the raw data pack, raw files are never modified."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import cfg
from common.io import log_run, read_raw, write_parquet, write_table

BOOL_COLS = ["is_four", "is_six", "is_wicket"]


def load_raw() -> dict[str, pd.DataFrame]:
    names = ["ball_by_ball", "matches", "competitions", "teams", "playing_eleven",
             "competition_team_player_mapping"]
    return {n: read_raw(f"{n}.csv") for n in names}


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def comp_types(comps: pd.DataFrame) -> pd.DataFrame:
    """IPL and Emerging come from the T20 label, everything else is T20I."""
    c = cfg()
    out = comps.copy()
    is_emerging = out["comp_name"].isin(c["emerging_comp_names"])
    is_ipl = (out["match_format"] == "T20") & ~is_emerging
    out["comp_class"] = np.select([is_ipl, is_emerging], ["IPL", "Emerging"], "T20I")
    return out


def clean_matches(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    c = cfg()
    comps = comp_types(raw["competitions"])
    m = raw["matches"].merge(comps[["comp_id", "comp_name", "match_format", "season", "comp_class"]],
                             on="comp_id", how="left")
    m = m.rename(columns={"season": "comp_season"})
    m["match_date"] = pd.to_datetime(m["match_date"], errors="coerce")
    # Split seasons such as 2022-23 exist, so calendar year of the match is the season
    m["season"] = m["match_date"].dt.year.astype(int)
    assoc = {t.lower() for t in c["associate_teams"]}
    m["has_associate"] = m["home_team"].str.lower().isin(assoc) | m["away_team"].str.lower().isin(assoc)
    # Associate tag only splits full-member T20Is, IPL and Emerging keep their own class
    m["comp_type"] = np.where((m["comp_class"] == "T20I") & m["has_associate"], "T20I_associate",
                              np.where(m["comp_class"] == "T20I", "T20I_full", m["comp_class"]))
    m["ipl_impact_era"] = (m["comp_type"] == "IPL") & (m["season"] >= c["impact_era_start"])
    m["era"] = np.where(m["comp_type"] == "IPL",
                        np.where(m["ipl_impact_era"], "IPL_impact", "IPL_pre_impact"), "non_IPL")
    m["is_knockout"] = pd.to_numeric(m["knock_out"], errors="coerce").fillna(0).astype(int) == 1
    res = m["result"].fillna("").str.lower()
    # No results have no winner, outcome comes from win team id not the text
    m["is_tie"] = res.str.contains("tie")
    m["is_no_result"] = (m["win_team_id"].isna() & ~m["is_tie"]) | res.str.contains("no result|abandon")
    # Two DLS results carry no revised fields, so the result text is a second signal
    revised = m[["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]].notna().any(axis=1)
    m["is_dls"] = revised | res.str.contains("d/l|dls|duckworth", regex=True)
    m["exclude"] = (m["is_no_result"] & c["exclude"]["no_result"]) | (m["is_dls"] & c["exclude"]["dls"])
    return m


def _wides_noballs(extras: pd.Series) -> pd.Series:
    return extras.isin(cfg()["wide_noball_extras"])


def clean_balls(raw: dict[str, pd.DataFrame], matches: pd.DataFrame) -> pd.DataFrame:
    c = cfg()
    b = raw["ball_by_ball"].copy()
    for col in BOOL_COLS:
        b[col] = _to_bool(b[col])
    b["extras_type"] = b["extras_type"].replace({"None": np.nan, "": np.nan})
    for col in ["runs", "extras1_runs", "extras_2_runs"]:
        b[col] = pd.to_numeric(b[col], errors="coerce").fillna(0).astype(int)
    # Runs column is bat runs only
    b["extras_runs"] = b["extras1_runs"] + b["extras_2_runs"]
    b["total_runs"] = b["runs"] + b["extras_runs"]
    # Wides and no-balls do not count toward the over
    b["is_legal"] = ~b["extras_type"].isin(c["legal_exclude_extras"])
    b["is_wide"] = b["extras_type"] == "wide"
    b["is_noball"] = b["extras_type"].isin(["Overstep", "other"])
    # Byes and leg byes are neither batter runs nor the bowler at fault
    b["bat_runs_credited"] = b["runs"]
    b["bowler_runs_conceded"] = b["runs"] + np.where(_wides_noballs(b["extras_type"]), b["extras_runs"], 0)
    # Typo fix, then only bowler-credited modes count to the bowler
    b["dismissal_type"] = b["dismissal_type"].replace(c["dismissal_normalise"])
    b["is_bowler_wicket"] = b["is_wicket"] & b["dismissal_type"].isin(c["bowler_wicket_types"])
    # Q6 run-outs can remove the non-striker, so key on dismissed player
    b["dismissed_player_id"] = pd.to_numeric(b["dismissed_player_id"], errors="coerce")
    b["is_batter_out"] = b["is_wicket"] & (b["dismissed_player_id"] == b["striker_id"])
    b["is_nonstriker_out"] = b["is_wicket"] & (b["dismissed_player_id"] == b["non_striker_id"])
    # Zero speed means not captured
    b["bowl_speed"] = pd.to_numeric(b["bowl_speed"], errors="coerce").replace(0, np.nan)
    b["ts"] = pd.to_datetime(b["delivery_ist_time"].str.replace(" UTC", "", regex=False), errors="coerce")
    keep = ["match_id", "comp_id", "comp_type", "comp_class", "season", "era", "ipl_impact_era",
            "is_knockout", "venue_id", "match_date", "win_team_id", "is_tie", "exclude",
            "is_no_result", "is_dls", "has_associate"]
    b = b.merge(matches[keep], on="match_id", how="left", validate="many_to_one")
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    return b


def exclusions_table(matches: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """Shortened and no-result matches dropped, with match and ball counts."""
    rows = []
    # Revised overs columns are held in balls, and some no-results carry them too
    dls_decided = matches["is_dls"] & ~matches["is_no_result"]
    for label, mask in [("No result or abandoned", matches["is_no_result"]),
                        ("DLS or revised overs, decided", dls_decided)]:
        ids = set(matches.loc[mask, "match_id"])
        rows.append({"exclusion": label, "matches": len(ids),
                     "balls": int(balls["match_id"].isin(ids).sum())})
    so = balls["innings"] > 2
    rows.append({"exclusion": "Super over balls", "matches": balls.loc[so, "match_id"].nunique(),
                 "balls": int(so.sum())})
    keep = ~balls["exclude"] & ~so
    rows.append({"exclusion": "Kept for ratings and WP", "matches": balls.loc[keep, "match_id"].nunique(),
                 "balls": int(keep.sum())})
    return pd.DataFrame(rows)


def run() -> None:
    raw = load_raw()
    matches = clean_matches(raw)
    balls = clean_balls(raw, matches)
    assert len(balls) == len(raw["ball_by_ball"]), "ball rows lost in cleaning"
    super_overs = balls[balls["innings"] > 2]
    excl = exclusions_table(matches, balls)
    write_parquet(matches, "data_prep", "matches_clean.parquet")
    write_parquet(balls, "data_prep", "balls_clean_raw.parquet")
    write_parquet(super_overs, "data_prep", "super_overs.parquet")
    write_table(excl, "data_prep", "exclusions.csv")
    log_run("cleaning", {"raw_ball_rows": len(raw["ball_by_ball"]), "clean_ball_rows": len(balls),
                         "matches": len(matches), "exclusions": excl.to_dict("records")})


if __name__ == "__main__":
    run()
