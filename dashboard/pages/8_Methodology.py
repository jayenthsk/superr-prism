"""Method summary, glossary, exclusions, limitations and reproduction."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_utils import csv, footer, meta, page

page("Methodology and data")
m = meta()
st.markdown(f"""
**1. Win probability.** One model per innings, LightGBM with monotone constraints, trained out of fold by match.
Each fold model is turned into a smooth surface over wickets, balls left and runs, so a single run cannot jump a
tree split. Chase calibration uses a two-parameter logistic fit.

**2. Leverage.** For every ball, the expected absolute win probability swing over the eight possible outcomes, with
outcome rates by competition type, phase and wicket band. Scaled so the average ball is 1 and capped at
{m['li_cap']} for weighting. The top 20 percent of balls are pressure balls.

**3. Context baseline.** For every ball, the outcome an average player would produce in the same state: innings,
over, wickets, balls since a wicket, balls faced, competition type, era, season, venue, pace or spin, batter hand,
batting slot, required rate and win probability. Out of fold by match. A second pass adds out-of-fold opponent skill.

**4. PRISM.** Normal-level skill per component is shrunk toward the average by empirical Bayes. The pressure effect
adds a clutch adjustment shrunk by how much clutch effects really vary. Each component is valued in win probability
on pressure balls by innings and phase. Units are WP points per 100 pressure balls above an average player.

**5. Final order.** Match bootstrap, the median rank decides, then the 10th percentile of PRISM, then consistency.
Eligibility: {m['eligibility']['min_balls']} legal balls, {m['eligibility']['min_pressure_balls']} pressure balls,
{m['eligibility']['min_pressure_matches']} matches with pressure, active in {m['eligibility']['active_since']} or later.
""")
st.subheader("Exclusions")
st.dataframe(csv("exclusions.csv"), hide_index=True)
st.subheader("Data quirks checked")
st.dataframe(csv("quirk_checks.csv"), hide_index=True, width="stretch")
st.subheader("Glossary")
st.markdown("""
- **LI** leverage index, average ball 1.
- **Pressure ball** LI in the top 20 percent of all rated balls. Extreme is the top 5 percent.
- **Residual** actual outcome minus the context baseline.
- **theta N, theta P** shrunk residual per ball on normal and pressure balls.
- **k** balls of evidence at which a player's own data counts for half.
- **P top 10** share of bootstrap resamples in which the player ranks in the top 10.
""")
st.subheader("Limitations")
st.markdown("""
- First innings win probability is hard to predict, AUC about 0.68, so first innings leverage is compressed.
- DLS and revised-overs matches are excluded in this version.
- Stroke and delivery tags come from analysts and differ a little between providers.
- Clutch effects barely repeat, so the ranking mostly rewards skill shown in high-stakes states.
""")
st.subheader("Reproduce")
st.code("pip install -r requirements.txt\npip install -e .\npython run_all.py\npython run_all.py --notebooks "
        "--skip-pipeline --no-downstream", language="bash")
footer()
