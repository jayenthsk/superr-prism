"""Runs every PRISM stage in order from the raw CSVs, Windows friendly."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NOTEBOOKS = [
    "data_prep/01_data_audit.ipynb",
    "data_prep/02_data_cleaning.ipynb",
    "data_prep/03_match_context_eda.ipynb",
    "pressure/04_win_probability_model.ipynb",
    "pressure/05_pressure_definition.ipynb",
    "player_ratings/06_player_ratings.ipynb",
    "reliability/07_reliability_and_myth_busting.ipynb",
    "player_insights/08_player_insights.ipynb",
    "final_rankings/09_results_review.ipynb",
]


def _step(name: str, fn) -> None:
    t0 = time.time()
    print(f"[run_all] {name} ...", flush=True)
    fn()
    print(f"[run_all] {name} done in {time.time() - t0:.0f}s", flush=True)


def run_pipeline(dev: bool = False, start: str | None = None) -> None:
    """Stage order matters, each stage reads only earlier outputs."""
    from data_prep import audit, clean, features
    from data_prep import plots as p1
    from final_rankings import plots as p6
    from final_rankings import select_top10
    from player_insights import bowler_execution, hard_entry, shot_mix
    from player_insights import plots as p5
    from player_ratings import plots as p3
    from player_ratings import ratings
    from pressure import leverage, tempo, wp_model
    from pressure import plots as p2
    from reliability import consistency, myths, robustness, uncertainty
    from reliability import plots as p4

    steps = [
        ("data_prep clean", clean.run), ("data_prep features", features.run), ("data_prep audit", audit.run),
        ("data_prep figures", p1.run),
        ("pressure wp model", wp_model.run), ("pressure leverage", leverage.run), ("pressure tempo", tempo.run),
        ("pressure figures", p2.run),
        ("player_ratings", ratings.run), ("player_ratings figures", p3.run),
        ("reliability uncertainty", lambda: uncertainty.run(dev=dev)), ("reliability consistency", consistency.run),
        ("reliability myths", myths.run), ("reliability robustness", robustness.run),
        ("player_insights bowler execution", bowler_execution.run), ("player_insights shot mix", shot_mix.run),
        ("player_insights hard entry", hard_entry.run), ("player_insights figures", p5.run),
        ("final rankings", select_top10.run), ("reliability figures", p4.run), ("final rankings figures", p6.run),
    ]
    names = [n for n, _ in steps]
    first = next((i for i, n in enumerate(names) if start and n.startswith(start)), 0)
    for name, fn in steps[first:]:
        _step(name, fn)


def run_downstream() -> None:
    """Dashboard data comes last, after the rankings are frozen."""
    from dashboard import export_app_data

    _step("dashboard export", export_app_data.run)
    if (ROOT / "report" / "build_report.py").exists():
        from report import build_report

        _step("report", build_report.run)


def run_notebooks() -> None:
    for nb in NOTEBOOKS:
        path = ROOT / nb
        if not path.exists():
            print(f"[run_all] skip missing {nb}")
            continue
        _step(f"notebook {nb}", lambda p=path: subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook",
             "--inplace", "--ExecutePreprocessor.timeout=900", str(p)],
            cwd=p.parent, check=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="PRISM pipeline")
    ap.add_argument("--dev", action="store_true", help="smaller bootstrap for development")
    ap.add_argument("--notebooks", action="store_true", help="also execute notebooks in order")
    ap.add_argument("--skip-pipeline", action="store_true", help="only run notebooks or downstream")
    ap.add_argument("--from", dest="start", default=None, help="resume from the step whose name starts with this")
    ap.add_argument("--no-downstream", action="store_true", help="skip dashboard export and report")
    args = ap.parse_args()
    if not args.skip_pipeline:
        run_pipeline(dev=args.dev, start=args.start)
    if not args.no_downstream:
        run_downstream()
    if args.notebooks:
        run_notebooks()


if __name__ == "__main__":
    main()
