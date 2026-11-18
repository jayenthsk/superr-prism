# PRISM: Pressure-Rated Index of Skill in Moments

**SupeRR Selector Season 2 (Rajasthan Royals), Round 1: identify the most reliable players under pressure.**

PRISM ranks IPL and T20I batters and bowlers (2022–2026) by how much win probability they add in high-stakes moments. It compares each player with an average player in the identical situation, and orders the final lists by how consistently a player stays near the top across 500 resamples of matches.

**Author:** Jayenth S

- **Live dashboard:** https://superr-prism.streamlit.app

## The idea in four lines
1. **Pressure is the stakes of the ball, not the scoreboard.** Every delivery gets a leverage index (LI): the expected absolute swing in win probability from that exact state. The top 20% of balls by LI are pressure balls.
2. **"Clutch" is mostly noise at this sample size.** Across random splits of matches, skill repeats (batter boundary rate r ≈ 0.6), but the pressure-minus-normal gap does not (r ≈ 0). So the clutch term is shrunk according to how reliable it really is.
3. **Use the stable parts of performance and value them in win probability.** Boundary, dot, dismissal and run rates are estimated with partial pooling against a context baseline, then converted into WP points.
4. **Rank reliability, not a point estimate.** The final order is the median rank across 500 match bootstraps, with ties broken by the 10th percentile of PRISM and then by match-level consistency.

## Results
### Top 10 batters
| Rank | Player | PRISM | 80% rank interval | P(top 10) | Robust |
|---|---|---|---|---|---|
| 1 | Tim David | 70.4 | 1–9 | 92% | ✓ |
| 2 | Vaibhav Sooryavanshi | 68.3 | 1–8 | 93% |  |
| 3 | Abhishek Sharma | 53.0 | 2–14 | 81% | ✓ |
| 4 | Phil Salt | 53.2 | 2–19 | 75% | ✓ |
| 5 | Shashank Singh | 48.3 | 2–28 | 66% |  |
| 6 | Romario Shepherd | 44.4 | 3–26 | 56% | ✓ |
| 7 | Glenn Maxwell | 40.4 | 2–72 | 42% | ✓ |
| 8 | Suryakumar Yadav | 36.2 | 7–33 | 28% | ✓ |
| 9 | Rinku Singh | 36.8 | 5–35 | 30% |  |
| 10 | Heinrich Klaasen | 34.6 | 6–36 | 27% |  |

### Top 10 bowlers
| Rank | Player | PRISM | 80% rank interval | P(top 10) | Robust |
|---|---|---|---|---|---|
| 1 | Sunil Narine | 43.0 | 1–7 | 95% | ✓ |
| 2 | Josh Hazlewood | 35.0 | 1–22 | 76% | ✓ |
| 3 | Jasprit Bumrah | 30.1 | 2–17 | 74% | ✓ |
| 4 | Rashid Khan | 28.3 | 2–26 | 61% | ✓ |
| 5 | Nathan Ellis | 24.0 | 3–29 | 50% |  |
| 6 | Blessing Muzarabani | 25.6 | 3–30 | 51% | ✓ |
| 7 | Krunal Pandya | 24.3 | 2–36 | 52% |  |
| 8 | Matheesha Pathirana | 22.5 | 3–38 | 46% | ✓ |
| 9 | Varun Chakravarthy | 21.8 | 4–30 | 41% | ✓ |
| 10 | Wanindu Hasaranga | 18.4 | 6–34 | 29% |  |

PRISM is the win probability, in percentage points, added per 100 pressure balls above an average player in identical situations. Robust means the player is in the top 10 in at least 80% of the sensitivity variants. The full eligible lists and three reserves per role are in [final_rankings/](final_rankings/).

## How to reproduce (Windows-first, Python 3.12)
The raw data pack is confidential and not in this repo. Place its six CSVs in `data_prep/raw/` before running the pipeline. The dashboard runs without them.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python run_all.py              # every stage from the raw CSVs, then the dashboard data
python run_all.py --notebooks  # also re-executes the nine notebooks in place
pytest -q                      # data, pressure and rating checks
```
- `python run_all.py --dev` uses B = 100 bootstrap draws instead of 500, for development.
- `python run_all.py --from "reliability"` resumes from the named stage.
- Seeds and every threshold live in [config.yaml](config.yaml). Run logs, with counts, parameters and the git hash, are written to [final_rankings/run_log.json](final_rankings/run_log.json).

## Folder map
Folders follow the pipeline stages. Each one holds its own code, notebooks and outputs, and reads only from earlier stages.

| Stage | Folder | What it does | Notebook |
|---|---|---|---|
| 1 | [data_prep/](data_prep/) | Audit, cleaning rules Q1–Q14, ball-state features, match-context EDA | 01, 02, 03 |
| 2 | [pressure/](pressure/) | Win-probability model, smoothed WP surface, leverage index, pressure tiers, tempo | 04, 05 |
| 3 | [player_ratings/](player_ratings/) | Context baseline, opposition adjustment, eligibility, PRISM | 06 |
| 4 | [reliability/](reliability/) | Bootstrap, split-half reliability, consistency, robustness, myth tests | 07 |
| 5 | [player_insights/](player_insights/) | Bowler execution, shot-mix drift, hard entries | 08 |
| 6 | [final_rankings/](final_rankings/) | Submission CSVs and results review | 09 |
| 7 | [dashboard/](dashboard/) | Streamlit app, reads only `dashboard/data/` | |
| shared | [common/](common/) | Config loader, path helpers, chart style | |

## Limitations
- First-innings win probability is hard to predict (AUC ≈ 0.68), so first-innings leverage is compressed. Only about 17% of pressure balls fall in the first innings.
- Matches shortened by DLS or revised overs are excluded in this version.
- Stroke and delivery-style tags come from analysts and differ a little between data providers. The execution and shot-mix extras are diagnostic only.
- Clutch effects barely repeat, so the ranking mostly rewards skill shown in high-stakes states. The robustness table shows how the lists move under 14–15 alternative choices.
- Players with different IDs are never merged, even when they share a name; shared names carry a team tag.
