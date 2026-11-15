"""Leverage explorer, threshold comparison and match worms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_utils import BOWL, TIER_COLOURS, csv, footer, page, pq, style_fig

page("What is pressure?")
st.markdown("Leverage is the expected absolute swing in win probability from a ball, given the match state. "
            "It is scaled so the average ball is 1. The top 20 percent of balls by leverage are **pressure balls**, "
            "and the top 5 percent are **Extreme**.")

st.subheader("Leverage explorer")
g = csv("li_grid_all.csv")
c1, c2 = st.columns([1, 3])
with c1:
    inn = st.radio("Innings", [1, 2], index=1, format_func=lambda x: "First innings" if x == 1 else "Chase")
    wk = st.slider("Wickets down", 0, 9, 4)
    bl = st.select_slider("Balls left", options=sorted(g["balls_left"].unique()), value=30)
with c2:
    s = g[(g["innings"] == inn) & (g["wkts_before"] == wk)]
    xcol = "runs_needed" if inn == 2 else "score_before"
    p = s.pivot_table(index=xcol, columns="balls_left", values="li")
    fig = go.Figure(go.Heatmap(z=p.values, x=p.columns, y=p.index, colorscale="Oranges", zmin=0,
                               zmax=float(g["li"].quantile(0.98)), colorbar=dict(title="LI"),
                               hovertemplate="Balls left %{x}<br>" + ("Runs needed" if inn == 2 else "Score")
                               + " %{y}<br>LI %{z:.2f}<extra></extra>"))
    fig.update_xaxes(autorange="reversed", title="Balls left")
    fig.update_yaxes(title="Runs needed" if inn == 2 else "Score so far")
    st.plotly_chart(style_fig(fig, f"Leverage, {wk} wickets down, IPL", 420), width="stretch")
line = s[s["balls_left"] == bl].sort_values(xcol)
fig = go.Figure(go.Scatter(x=line[xcol], y=line["li"], mode="lines+markers", line=dict(color=BOWL, width=2),
                           hovertemplate="%{x}<br>LI %{y:.2f}<extra></extra>"))
fig.add_hline(y=1, line_dash="dash", line_color="grey")
fig.update_xaxes(title="Runs needed" if inn == 2 else "Score so far")
fig.update_yaxes(title="Leverage index")
st.plotly_chart(style_fig(fig, f"Leverage at {bl} balls left", 320), width="stretch")

st.subheader("Why not a required-rate threshold?")
th = csv("li_threshold_comparison.csv")
fig = go.Figure(go.Bar(x=th["share"], y=th["check"], orientation="h", marker_color=BOWL,
                       text=[f"{v:.0%}" for v in th["share"]], textposition="outside",
                       hovertemplate="%{y}<br>%{x:.1%}<extra></extra>"))
fig.update_xaxes(range=[0, 1.1], tickformat=".0%")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style_fig(fig, "Scoreboard thresholds against leverage", 360), width="stretch")

st.subheader("Match worm with leverage shading")
w = pq("worms.parquet")
teams = csv("teams.csv").set_index("team_id")["team_name"]
first = w[w["innings"] == 1].groupby("match_id").agg(bat=("batting_team_id", "first"), season=("season", "first"),
                                                       comp=("comp_type", "first"))
second = w[w["innings"] == 2].groupby("match_id")["batting_team_id"].first()
first["label"] = first["season"].astype(str) + " " + first["comp"] + ", " + first["bat"].map(teams) + " v " + \
    second.reindex(first.index).map(teams).fillna("") + ", id " + first.index.astype(str)
mid = st.selectbox("Match", first.sort_values("season", ascending=False).index,
                   format_func=lambda x: first.loc[x, "label"])
m = w[w["match_id"] == mid].sort_values(["innings", "innings_ball_num"]).reset_index(drop=True)
wp1 = np.where(m["innings"] == 1, m["wp_pre"], 1 - m["wp_pre"])
fig = go.Figure()
for tier in ["High", "Extreme"]:
    sel = m["pressure_tier"] == tier
    fig.add_bar(x=m.index[sel], y=[1] * sel.sum(), marker_color=TIER_COLOURS[tier], opacity=0.35, name=tier,
                hoverinfo="skip", width=1)
fig.add_scatter(x=m.index, y=wp1, mode="lines", line=dict(color=BOWL, width=2), name="Win probability",
                customdata=np.stack([m["innings"], m["over_no"], m["striker_name"], m["bowler_name"], m["li"]], 1),
                hovertemplate="Inn %{customdata[0]} over %{customdata[1]}<br>%{customdata[2]} v %{customdata[3]}"
                              "<br>WP %{y:.2f}, LI %{customdata[4]:.2f}<extra></extra>")
fig.add_vline(x=int((m["innings"] == 1).sum()), line_dash="dash", line_color="grey")
fig.update_yaxes(range=[0, 1], title=f"WP {teams.get(first.loc[mid, 'bat'], '')}")
fig.update_xaxes(title="Ball of match")
fig.update_layout(barmode="overlay", bargap=0)
st.plotly_chart(style_fig(fig, "Win probability for the side batting first", 380), width="stretch")

st.subheader("League-wide pressure tax, matched on innings, over and wickets")
st.dataframe(csv("pressure_tax.csv").round(4), hide_index=True)
footer()
