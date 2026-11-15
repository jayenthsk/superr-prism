"""PRISM dashboard home, the thesis and both top 10 lists."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_utils import AUTHOR, REPO_URL, csv, footer, page, top10_display

page("PRISM, who delivers when it matters")

st.markdown(
    "**PRISM** stands for Pressure-Rated Index of Skill in Moments. Pressure here means the stakes of the ball, "
    "not the scoreboard. Every delivery gets a **leverage index**: how far its result is expected to move the "
    "batting side's win probability from that exact match state. A player's rating is the win probability he adds "
    "per 100 pressure balls above an average player in identical situations. It is built from the parts of "
    "performance that repeat, with the clutch part shrunk according to how reliable it actually is. The final "
    "order uses the **median rank across 500 match resamples**, so a player ranks high only if he stays high."
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Top 10 batters under pressure")
    st.dataframe(top10_display("bat"), hide_index=True, width="stretch",
                 column_config={"PRISM": st.column_config.NumberColumn(format="%.1f"),
                                "P top 10": st.column_config.ProgressColumn(min_value=0, max_value=1)})
with c2:
    st.subheader("Top 10 bowlers under pressure")
    st.dataframe(top10_display("bowl"), hide_index=True, width="stretch",
                 column_config={"PRISM": st.column_config.NumberColumn(format="%.1f"),
                                "P top 10": st.column_config.ProgressColumn(min_value=0, max_value=1)})
st.caption("PRISM is WP percentage points added per 100 pressure balls above an average player in identical "
           "situations. Robust means top 10 in at least 80 percent of the sensitivity variants.")

th = csv("li_threshold_comparison.csv").set_index("check")["share"]
sh = csv("split_half_summary.csv")
sh = sh[(sh["metric_basis"] == "resid") & (sh["role"] == "bat")].set_index(["component", "kind"])["r_mean"]
st.subheader("Three headline insights")
i1, i2, i3 = st.columns(3)
i1.metric("RRR 10 plus chase balls that are dead games",
          f"{th['Chase balls with RRR 10 plus at WP under 0.1 or over 0.9']:.0%}")
i1.caption("Scoreboard thresholds flag balls whose result barely matters.")
i2.metric("High leverage balls before the death overs", f"{th['High leverage balls in overs 1 to 15']:.0%}")
i2.caption("Pressure arrives in the middle overs far more often than folklore says.")
i3.metric("Split-half repeatability, batter boundary skill vs clutch gap",
          f"{sh[('boundary', 'level')]:.2f} vs {sh[('boundary', 'delta')]:.2f}")
i3.caption("Skill repeats, the pressure gap barely does, so it is shrunk hard.")

st.markdown(f"Built by {AUTHOR} · [Code and notebooks on GitHub]({REPO_URL})")
footer()
