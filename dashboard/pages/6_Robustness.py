"""Top 10 membership across sensitivity variants."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_utils import ROLE_FILE, csv, footer, page, style_fig

page("Robustness")
role = st.radio("Role", ["bat", "bowl"], horizontal=True, format_func=lambda r: "Batters" if r == "bat" else "Bowlers")
rob = csv("robustness.csv")
rob = rob[rob["role"] == role]
mem = csv("robustness_membership.csv")
mem = mem[mem["role"] == role]
full = csv(f"full_{ROLE_FILE[role]}.csv").head(20)
m = pd.DataFrame(0, index=full["player_id"], columns=rob["variant"])
for _, r in mem.iterrows():
    if r["player_id"] in m.index:
        m.loc[r["player_id"], r["variant"]] = 1
fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=full["player_name"], colorscale=[[0, "#F2F2F2"],
                           [1, "#254AA5" if role == "bowl" else "#E4007C"]], showscale=False,
                           hovertemplate="%{y}<br>%{x}<br>%{z}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style_fig(fig, "Top 10 membership by variant, filled means in, main top 20 shown", 620),
                width="stretch")
st.dataframe(rob.drop(columns="role").round(3), hide_index=True, width="stretch")
st.subheader("Agreement with simpler rankings")
st.dataframe(csv("method_agreement.csv").query("role == @role").drop(columns="role").round(3), hide_index=True)
footer()
