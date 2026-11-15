"""One player, pressure against normal, components, consistency and extras."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_utils import NEUTRAL, ROLE_COL, csv, footer, page, player_options, style_fig

COMPS = ["boundary", "dot", "out", "runs"]
page("Player profile")
c1, c2 = st.columns([1, 3])
role = c1.radio("Role", ["bat", "bowl"], format_func=lambda r: "Batter" if r == "bat" else "Bowler")
only = c1.checkbox("Eligible players only", value=True)
opts = player_options(role, only)
pid = c2.selectbox("Player", opts["player_id"], format_func=lambda p: opts.set_index("player_id").loc[p, "player_name"])
r = opts.set_index("player_id").loc[pid]
full = csv(f"full_{'batters' if role == 'bat' else 'bowlers'}.csv").set_index("player_id")

m = st.columns(5)
m[0].metric("PRISM", f"{r['prism']:.1f}")
if pid in full.index:
    f = full.loc[pid]
    m[1].metric("Final rank", int(f["rank"]))
    m[2].metric("80% rank interval", f"{f['rank_lo80']:.0f} to {f['rank_hi80']:.0f}")
    m[3].metric("P top 10", f"{f['p_top10']:.0%}")
    m[4].metric("Consistency", f"{f['consistency_index']:.2f}" if pd.notna(f["consistency_index"]) else "n a")
else:
    m[1].metric("Final rank", "not eligible")
st.caption(f"{r['teams']}. {int(r['total_balls'])} legal balls, {int(r['pressure_balls'])} pressure balls, "
           f"{int(r['matches_with_pressure'])} matches with pressure.")

st.subheader("Raw numbers, pressure against normal")
if role == "bat":
    raw = pd.DataFrame({"Strike rate": [r["sr_N"], r["sr_P"]], "Average": [r["avg_N"], r["avg_P"]],
                        "Dot percent": [100 * r["dot_N"], 100 * r["dot_P"]],
                        "Boundary percent": [100 * r["bnd_N"], 100 * r["bnd_P"]]}, index=["Normal", "Pressure"])
else:
    raw = pd.DataFrame({"Economy": [r["econ_N"], r["econ_P"]], "Average": [r["avg_N"], r["avg_P"]],
                        "Strike rate": [r["bsr_N"], r["bsr_P"]], "Dot percent": [100 * r["dot_N"], 100 * r["dot_P"]],
                        "Boundary percent": [100 * r["bnd_N"], 100 * r["bnd_P"]]}, index=["Normal", "Pressure"])
st.dataframe(raw.round(1))

st.subheader("Skill against context, normal level and pressure estimate")
sign = 1 if role == "bat" else -1
fig = go.Figure()
fig.add_bar(x=COMPS, y=[sign * r[f"thetaN_{c}"] for c in COMPS], name="Normal level", marker_color=NEUTRAL)
fig.add_bar(x=COMPS, y=[sign * r[f"thetaP_{c}"] for c in COMPS], name="Pressure estimate",
            marker_color=ROLE_COL[role])
fig.add_hline(y=0, line_color="black")
fig.update_yaxes(title="Per ball above context" + (", bowler sign so higher is fewer" if role == "bowl" else ""))
st.plotly_chart(style_fig(fig, "Shrunk residual per ball by component", 360), width="stretch")
st.caption("For bowlers, boundary, runs and out are shown with the sign flipped where needed so that higher "
           "means fewer boundaries and runs conceded. Dot and out keep their natural direction for batters.")

c1, c2 = st.columns(2)
mc = csv(f"match_contributions_{role}.csv")
mc = mc[(mc["player_id"] == pid) & (mc["balls"] >= 6)]
with c1:
    fig = go.Figure(go.Box(y=mc["contrib"], boxpoints="all", jitter=0.5, pointpos=0, marker_color=ROLE_COL[role],
                           line_color=NEUTRAL, name="", hovertemplate="%{y:.1f}<extra></extra>"))
    fig.add_hline(y=0, line_color="black")
    fig.update_yaxes(title="WP points from pressure balls in the match")
    st.plotly_chart(style_fig(fig, f"Match contributions, {len(mc)} qualifying matches", 380),
                    width="stretch")
with c2:
    sp = csv(f"season_prism_{role}.csv")
    sp = sp[sp["player_id"] == pid]
    fig = go.Figure(go.Scatter(x=sp["season"], y=sp["prism"], mode="lines+markers",
                               line=dict(color=ROLE_COL[role], width=2), customdata=sp["n_press"],
                               hovertemplate="%{x}<br>PRISM %{y:.1f}<br>%{customdata:.0f} pressure balls"
                               "<extra></extra>"))
    fig.add_hline(y=0, line_color="black")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(style_fig(fig, "PRISM by calendar year", 380), width="stretch")

st.subheader("Diagnostics")
if role == "bowl":
    pe = csv("e1_pace_execution.csv")
    bb = csv("e1_bounce_back.csv")
    if pid in set(pe["player_id"]):
        st.write("Pace execution at the death and on pressure balls")
        st.dataframe(pe[pe["player_id"] == pid].drop(columns=["player_id", "eligible"]).round(3), hide_index=True)
    if pid in set(bb["player_id"]):
        st.write("Bounce-back after conceding a boundary, positive means tighter than context")
        st.dataframe(bb[bb["player_id"] == pid][["n", "bounce_runs", "bounce_boundary"]].round(3), hide_index=True)
else:
    sm = csv("e2_shot_mix.csv")
    he = csv("e3_hard_entry.csv")
    if pid in set(sm["player_id"]):
        row = sm[sm["player_id"] == pid].iloc[0]
        st.write(f"Shot-mix quadrant **{row['quadrant']}**. Aggressive strokes {row['aggr_N']:.0%} normal, "
                 f"{row['aggr_P']:.0%} under pressure. Drift above league {row['excess_drift']:.3f}.")
    if pid in set(he["player_id"]):
        st.write("First ten balls after a hard entry, against his own normal entries, shrunk")
        st.dataframe(he[he["player_id"] == pid][["n_hard", "n_normal", "hard_gap_runs", "hard_gap_boundary",
                                                  "hard_gap_out"]].round(3), hide_index=True)
if "opp_strength_pressure" in r and pd.notna(r["opp_strength_pressure"]):
    st.write(f"Opposition faced on pressure balls, mean opponent runs skill per ball {r['opp_strength_pressure']:+.3f}"
             f", league spread is about {np.nanstd(opts['opp_strength_pressure']):.3f}.")
footer()
