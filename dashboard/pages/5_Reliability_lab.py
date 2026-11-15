"""Why the clutch gap is shrunk, split-half reliability and bootstrap ranks."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_utils import NEUTRAL, ROLE_COL, csv, footer, page, player_options, pq, style_fig

page("Reliability lab")
role = st.radio("Role", ["bat", "bowl"], horizontal=True, format_func=lambda r: "Batters" if r == "bat" else "Bowlers")
comps = ["boundary", "dot", "out", "runs"]

sh = csv("split_half_summary.csv")
basis = st.radio("Metric basis", ["resid", "raw"], horizontal=True,
                 format_func=lambda b: "Against context" if b == "resid" else "Raw rates")
s = sh[(sh["role"] == role) & (sh["metric_basis"] == basis)].set_index(["component", "kind"])
fig = go.Figure()
for kind, col, name in [("level", ROLE_COL[role], "Normal level"), ("delta", NEUTRAL, "Pressure minus normal")]:
    fig.add_bar(x=comps, y=[s.loc[(c, kind), "r_mean"] for c in comps], name=name, marker_color=col,
                error_y=dict(type="data", symmetric=False,
                             array=[s.loc[(c, kind), "r_hi"] - s.loc[(c, kind), "r_mean"] for c in comps],
                             arrayminus=[s.loc[(c, kind), "r_mean"] - s.loc[(c, kind), "r_lo"] for c in comps]),
                text=[f"{s.loc[(c, kind), 'r_mean']:.2f}" for c in comps], textposition="outside")
fig.update_yaxes(title="Correlation between random match halves", range=[-0.4, 1])
st.plotly_chart(style_fig(fig, "Split-half reliability over 20 random match splits", 420), width="stretch")
st.caption("Bars show the mean over splits, whiskers the lowest and highest split.")

k = csv("shrinkage_constants.csv").set_index("role").loc[role]
st.subheader("Shrinkage constants")
st.dataframe({"component": comps, "k normal level, balls": [round(k[f"kN_{c}"]) for c in comps],
              "k clutch, pressure balls": [round(k[f"kD_{c}"]) for c in comps]}, hide_index=True)
st.caption("k is the number of balls at which a player's own evidence counts for half. The clutch k is capped at "
           "2000 when the estimated true spread of clutch effects is not positive.")

r = pq(f"ratings_{role}.parquet")
r = r[r["n_eff"] > 0]
comp = st.selectbox("Component for the shrinkage picture", comps, index=3)
raw = r[f"rawP_{comp}"] - r[f"thetaN_{comp}"]
fig = go.Figure()
fig.add_scatter(x=r["n_eff"], y=raw, mode="markers", name="Raw pressure gap", marker=dict(color=NEUTRAL, size=6),
                text=r["player_name"], hovertemplate="%{text}<br>n eff %{x:.0f}<br>%{y:.3f}<extra></extra>")
fig.add_scatter(x=r["n_eff"], y=r[f"delta_{comp}"], mode="markers", name="Shrunk clutch effect",
                marker=dict(color=ROLE_COL[role], size=6), text=r["player_name"],
                hovertemplate="%{text}<br>n eff %{x:.0f}<br>%{y:.3f}<extra></extra>")
fig.update_xaxes(type="log", title="Effective pressure balls")
fig.update_yaxes(title="Residual per ball")
st.plotly_chart(style_fig(fig, "Raw against shrunk clutch effect", 420), width="stretch")

st.subheader("Bootstrap rank distribution")
opts = player_options(role)
pid = st.selectbox("Player", opts["player_id"], format_func=lambda p: opts.set_index("player_id").loc[p, "player_name"])
b = pq(f"bootstrap_{role}.parquet")
b = b[b["player_id"] == pid]
fig = go.Figure(go.Histogram(x=b["rank"], xbins=dict(size=1), marker_color=ROLE_COL[role]))
fig.add_vline(x=10.5, line_dash="dash", line_color="grey")
fig.update_xaxes(title="Rank in resample")
fig.update_yaxes(title="Resamples")
st.plotly_chart(style_fig(fig, f"Rank across {len(b)} match resamples", 360), width="stretch")
footer()
