"""Folklore against the data."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_utils import BOWL, NEUTRAL, PALETTE, csv, footer, page, style_fig

page("Myth-busting insights")
c1, c2 = st.columns(2)
with c1:
    t = csv("myth_dot_streak.csv")
    fig = go.Figure()
    for i, ph in enumerate(["PP", "MID", "DEATH"]):
        g = t[t["phase"] == ph]
        fig.add_scatter(x=g["streak"], y=100 * g["excess"], mode="lines+markers", name=ph,
                        line=dict(color=PALETTE[i], width=2), customdata=g["balls"],
                        hovertemplate="Streak %{x}<br>%{y:.2f} per 100<br>%{customdata} balls<extra></extra>")
    fig.add_hline(y=0, line_color="black")
    fig.update_xaxes(title="Dots in a row before the ball, 5 means 5 plus")
    fig.update_yaxes(title="Wickets per 100 balls above context")
    st.plotly_chart(style_fig(fig, "Do dot-ball streaks bring wickets?", 380), width="stretch")
    st.caption("No. Within each phase the excess wicket rate barely moves with streak length.")
with c2:
    t = csv("myth_tight_over.csv")
    fig = go.Figure()
    for tight, col, name in [(False, NEUTRAL, "After any other over"), (True, BOWL, "After a tight over")]:
        g = t[t["tight"] == tight].set_index("phase").reindex(["PP", "MID", "DEATH"])
        fig.add_bar(x=g.index, y=g["excess_per_100"], name=name, marker_color=col)
    fig.add_hline(y=0, line_color="black")
    fig.update_yaxes(title="Next over wickets per 100 balls above context")
    st.plotly_chart(style_fig(fig, "Does a tight over bring a wicket next over?", 380), width="stretch")
    st.caption("No consistent effect across phases.")

c1, c2 = st.columns(2)
with c1:
    t = csv("e1_league_execution.csv")
    fig = go.Figure()
    fig.add_bar(x=t["delivery"], y=100 * t["normal"], name="Normal death balls", marker_color=NEUTRAL)
    fig.add_bar(x=t["delivery"], y=100 * t["pressure"], name="Pressure death balls", marker_color=BOWL)
    fig.update_yaxes(title="Share of pace deliveries, percent")
    st.plotly_chart(style_fig(fig, "Pace bowlers at the death under pressure", 380), width="stretch")
with c2:
    t = csv("e2_league_shift.csv")
    t = t[t["phase"] == "DEATH"]
    fig = go.Figure()
    for i, g in enumerate(["aggressive", "orthodox", "defensive"]):
        fig.add_bar(x=["Normal", "Pressure"], y=100 * t.sort_values("is_pressure")[g], name=g, marker_color=PALETTE[i])
    fig.update_yaxes(title="Share of strokes, percent")
    st.plotly_chart(style_fig(fig, "Death-over stroke mix, normal against pressure", 380), width="stretch")

t = csv("tempo_vs_li.csv")
fig = go.Figure(go.Scatter(x=t["li_mid"], y=t["gap_median"], mode="lines+markers", line=dict(color=BOWL, width=2),
                           hovertemplate="LI %{x:.2f}<br>%{y:.0f} seconds<extra></extra>"))
fig.update_xaxes(type="log", title="Leverage decile median")
fig.update_yaxes(title="Median seconds between balls")
st.plotly_chart(style_fig(fig, "The game slows down when leverage is high", 340), width="stretch")
bb = csv("e1_bounce_back_league.csv")
st.write(f"Boundary on the next ball after a boundary: {bb.loc[1, 'boundary_rate']:.1%} against "
         f"{bb.loc[0, 'boundary_rate']:.1%} otherwise. Context expects {bb.loc[1, 'expected_boundary']:.1%} and "
         f"{bb.loc[0, 'expected_boundary']:.1%}, so most of the rise is the situation, not the bowler.")
footer()
