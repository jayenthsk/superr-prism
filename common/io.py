"""Stage path helpers so every stage reads and writes in the same places."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common.config import ROOT, cfg


def stage_dir(stage: str) -> Path:
    path = ROOT / cfg()["paths"][stage]
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_path(name: str) -> Path:
    return ROOT / cfg()["paths"]["raw"] / name


def fig_path(stage: str, name: str) -> Path:
    path = stage_dir(stage) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def table_path(stage: str, name: str) -> Path:
    path = stage_dir(stage) / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(raw_path(name), low_memory=False)


def write_parquet(df: pd.DataFrame, stage: str, name: str) -> Path:
    path = stage_dir(stage) / name
    df.to_parquet(path, index=False)
    return path


def read_parquet(stage: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(stage_dir(stage) / name)


def write_table(df: pd.DataFrame, stage: str, name: str, index: bool = False) -> Path:
    path = table_path(stage, name)
    df.to_csv(path, index=index)
    return path


def read_table(stage: str, name: str) -> pd.DataFrame:
    return pd.read_csv(table_path(stage, name))


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "nogit"


def log_run(section: str, payload: dict[str, Any]) -> None:
    path = stage_dir("final_rankings") / "run_log.json"
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    log[section] = {**payload, "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "git_hash": git_hash()}
    path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
