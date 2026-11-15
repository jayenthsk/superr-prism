"""Copies the small files the app needs into dashboard data, the app never trains anything."""
from __future__ import annotations

import json
import shutil

import pandas as pd

from common.config import ROOT, cfg
from common.io import git_hash, read_parquet, stage_dir, table_path

TABLES = {
    "pressure": ["li_grid_all.csv", "li_threshold_comparison.csv", "li_separation.csv", "li_behaviour_checks.csv",
                 "pressure_tax.csv", "li_by_over.csv", "tempo_vs_li.csv", "tempo_summary.csv", "wp_metrics.csv",
                 "li_cuts.csv", "wp_known_situations.csv"],
    "player_ratings": ["shrinkage_constants.csv", "eligibility_pools.csv", "component_values_bat.csv",
                       "component_values_bowl.csv"],
    "reliability": ["split_half_summary.csv", "robustness.csv", "robustness_membership.csv",
                    "method_agreement.csv", "myth_dot_streak.csv", "myth_dot_streak_slope.csv", "myth_tight_over.csv",
                    "consistency_bat.csv", "consistency_bowl.csv", "season_prism_bat.csv", "season_prism_bowl.csv",
                    "match_contributions_bat.csv", "match_contributions_bowl.csv", "bootstrap_summary_bat.csv",
                    "bootstrap_summary_bowl.csv"],
    "player_insights": ["e1_pace_execution.csv", "e1_spin_execution.csv", "e1_bounce_back.csv",
                        "e1_bounce_back_league.csv", "e1_league_execution.csv", "e2_shot_mix.csv",
                        "e2_league_shift.csv", "e3_hard_entry.csv", "e3_league_entry.csv"],
    "data_prep": ["exclusions.csv", "quirk_checks.csv"],
}
RATING_COLS = ["player_id", "player_name", "teams", "comp_types_played", "total_balls", "pressure_balls", "matches",
               "matches_with_pressure", "last_season", "eligible", "prism", "n_eff", "opp_strength_pressure"]


def ratings_slim(role: str) -> pd.DataFrame:
    r = read_parquet("player_ratings", f"player_ratings_{role}.parquet")
    keep = [c for c in r.columns if c in RATING_COLS or c.split("_")[0] in
            ("thetaN", "thetaP", "delta", "rawP", "value", "sr", "avg", "econ", "bsr", "dot", "bnd")]
    return r[keep]


def match_worms() -> pd.DataFrame:
    """Per-ball WP and leverage for the match picker, close matches only to keep it small."""
    b = read_parquet("pressure", "balls_pressure.parquet")
    cols = ["match_id", "innings", "innings_ball_num", "over_no", "batting_team_id", "wp_pre", "li", "pressure_tier",
            "comp_type", "season", "striker_name", "bowler_name", "total_runs", "is_wicket"]
    return b[cols].astype({"wp_pre": "float32", "li": "float32"})


def run() -> None:
    out = stage_dir("dashboard_data")
    for stage, names in TABLES.items():
        for n in names:
            src = table_path(stage, n)
            if src.exists():
                shutil.copy(src, out / n)
    for role in ("bat", "bowl"):
        ratings_slim(role).to_parquet(out / f"ratings_{role}.parquet", index=False)
        boot = read_parquet("reliability", f"bootstrap_{role}.parquet")
        boot.astype({"prism": "float32", "rank": "float32"}).to_parquet(out / f"bootstrap_{role}.parquet", index=False)
    for name in ("top10_batters", "top10_bowlers", "full_batters", "full_bowlers", "reserves_batters",
                 "reserves_bowlers"):
        shutil.copy(ROOT / "final_rankings" / f"{name}.csv", out / f"{name}.csv")
    match_worms().to_parquet(out / "worms.parquet", index=False)
    teams = pd.read_csv(ROOT / cfg()["paths"]["raw"] / "teams.csv").drop_duplicates("team_id")
    teams[["team_id", "team_name"]].to_csv(out / "teams.csv", index=False)
    meta = {"version": cfg()["version"], "freeze_date": cfg()["freeze_date"], "git_hash": git_hash(),
            "li_cap": cfg()["li"]["cap"], "eligibility": cfg()["eligibility"]}
    (out / "version.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    assert size < 50, f"dashboard data is {size:.1f} MB, cap is 50"
    print(f"dashboard data {size:.1f} MB")


if __name__ == "__main__":
    run()
