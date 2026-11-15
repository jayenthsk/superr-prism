"""Shared layout for the batting and bowling ranking pages."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from app_utils import PALETTE, ROLE_COL, footer, page, pq, ranking_table, style_fig

COMPS = ["boundary", "dot", "out", "runs"]


def render(role: str) -> None:
    title = "Batting rankings" if role == "bat" else "Bowling rankings"
    page(title)
    t = ranking_table(role)
    comp_opts = sorted({c for s in t["comp_types_played"] for c in s.split(", ")})
    pick = st.multiselect("Show players who played in", comp_opts, default=comp_opts)
    view = t[t["comp_types_played"].apply(lambda s: any(c in s for c in pick))]
    cols = ["rank", "player_name", "teams", "total_balls", "pressure_balls", "prism", "prism_p10", "prism_p90",
            "median_rank", "80% rank interval", "p_top10", "consistency_index", "Robust"]
    st.dataframe(view[cols], hide_index=True, width="stretch", height=380,
                 column_config={"prism": st.column_config.NumberColumn("PRISM", format="%.1f"),
                                "prism_p10": st.column_config.NumberColumn("P10", format="%.1f"),
                                "prism_p90": st.column_config.NumberColumn("P90", format="%.1f"),
                                "p_top10": st.column_config.ProgressColumn("P top 10", min_value=0, max_value=1),
                                "consistency_index": st.column_config.NumberColumn("Consistency", format="%.2f")})

    top = t.head(25)
    col = [ROLE_COL[role] if r <= 10 else "#9E9E9E" for r in top["rank"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=top["median_rank"], y=top["player_name"], mode="markers", marker=dict(color=col, size=9),
        error_x=dict(type="data", symmetric=False, array=top["rank_hi80"] - top["median_rank"],
                     arrayminus=top["median_rank"] - top["rank_lo80"], color="#9E9E9E", thickness=2),
        customdata=top[["rank_lo80", "rank_hi80", "p_top10"]],
        hovertemplate="%{y}<br>Median rank %{x}<br>80 percent %{customdata[0]:.0f} to %{customdata[1]:.0f}"
                      "<br>P top 10 %{customdata[2]:.0%}<extra></extra>"))
    fig.add_vline(x=10.5, line_dash="dash", line_color="grey")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Rank among eligible players")
    st.plotly_chart(style_fig(fig, "Bootstrap rank with 80 percent interval, top 25", 640), width="stretch")

    r = pq(f"ratings_{role}.parquet").set_index("player_id").loc[top["player_id"]]
    fig = go.Figure()
    for i, c in enumerate(COMPS):
        fig.add_bar(y=top["player_name"], x=r[f"value_{c}"], orientation="h", name=c, marker_color=PALETTE[i],
                    hovertemplate="%{y}<br>" + c + " %{x:.1f}<extra></extra>")
    fig.update_layout(barmode="relative")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="WP points per 100 pressure balls")
    st.plotly_chart(style_fig(fig, "What each rating is made of", 640), width="stretch")
    footer()
