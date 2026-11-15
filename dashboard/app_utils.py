"""Shared loaders, colours and page furniture for the PRISM app."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
BAT, BOWL, ACCENT, GREEN, PURPLE, NEUTRAL = "#E4007C", "#254AA5", "#C98A00", "#2E7D32", "#8E3BBF", "#7A7A7A"
PALETTE = [BAT, BOWL, ACCENT, GREEN, PURPLE, NEUTRAL]
ROLE_COL = {"bat": BAT, "bowl": BOWL}
ROLE_FILE = {"bat": "batters", "bowl": "bowlers"}
TIER_COLOURS = {"Low": "#F6DFC4", "Normal": "#E9AE6B", "High": "#C9661C", "Extreme": "#8A3500"}
REPO_URL = "https://github.com/jayenthsk/superr-prism"
AUTHOR = "Jayenth S"


@st.cache_data
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data
def pq(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / name)


@st.cache_data
def meta() -> dict:
    return json.loads((DATA / "version.json").read_text(encoding="utf-8"))


def page(title: str) -> None:
    st.set_page_config(page_title=f"PRISM, {title}", page_icon="🏏", layout="wide")
    m = meta()
    st.caption(f"Submission version v{m['version']}, frozen {m['freeze_date']}, commit {m['git_hash']}")
    st.title(title)


def footer() -> None:
    st.divider()
    st.caption(f"PRISM by {AUTHOR}.")


def style_fig(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(title=title, template="simple_white", height=height, colorway=PALETTE,
                      margin=dict(l=40, r=20, t=60, b=40), font=dict(size=13), hoverlabel=dict(font_size=12))
    return fig


def ranking_table(role: str) -> pd.DataFrame:
    t = csv(f"full_{ROLE_FILE[role]}.csv")
    t["80% rank interval"] = t["rank_lo80"].round(0).astype(int).astype(str) + " to " + \
        t["rank_hi80"].round(0).astype(int).astype(str)
    t["Robust"] = t["robust_badge"].map({True: "Yes", False: ""})
    return t


def top10_display(role: str) -> pd.DataFrame:
    t = ranking_table(role).head(10)
    return t[["rank", "player_name", "teams", "prism", "80% rank interval", "p_top10", "Robust"]].rename(columns={
        "rank": "Rank", "player_name": "Player", "teams": "Teams", "prism": "PRISM", "p_top10": "P top 10"})


def player_options(role: str, eligible_only: bool = True) -> pd.DataFrame:
    r = pq(f"ratings_{role}.parquet")
    if eligible_only:
        r = r[r["eligible"]]
    return r.sort_values("prism", ascending=False)
