"""Stage 1 figures, audit and match context."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import style
from common.io import fig_path, read_parquet, read_table

S = "data_prep"


def missingness(balls) -> plt.Figure:
    cols = ["stroke", "delivery_style", "bowl_speed", "gap_s", "dismissal_type", "batting_order", "ts"]
    m = balls[balls["innings"] <= 2].groupby("comp_type")[cols].apply(lambda x: x.isna().mean())
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(m.values, cmap=style.SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(m)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.5 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Share missing")
    ax.set_title("Missing share by field and competition type")
    style.finish(fig, fig_path(S, "01_missingness.png"))
    return fig


def matches_by_season() -> plt.Figure:
    t = read_table(S, "matches_by_season.csv").set_index("season")
    fig, ax = plt.subplots()
    bottom = np.zeros(len(t))
    for col in [c for c in style.COMP_COLOURS if c in t]:
        ax.bar(t.index, t[col], bottom=bottom, color=style.COMP_COLOURS[col], label=col, edgecolor="white", width=0.7)
        bottom += t[col].to_numpy()
    ax.set(title="Matches by season and competition type", xlabel="Season", ylabel="Matches")
    ax.legend(ncol=4, loc="upper left")
    style.finish(fig, fig_path(S, "01_matches_by_season.png"))
    return fig


def scoring_by_phase() -> plt.Figure:
    t = read_table(S, "scoring_by_phase.csv")
    phases = ["PP", "MID", "DEATH"]
    fig, ax = plt.subplots()
    w = 0.2
    for i, (ct, col) in enumerate(style.COMP_COLOURS.items()):
        g = t[t["comp_type"] == ct].set_index("phase").reindex(phases)
        ax.bar(np.arange(3) + (i - 1.5) * w, g["runs_per_over"], w, color=col, label=ct, edgecolor="white")
    ax.set_xticks(range(3), ["Powerplay 1-6", "Middle 7-15", "Death 16-20"])
    ax.set(title="Run rate by phase and competition type", ylabel="Runs per over")
    ax.legend(ncol=4, loc="upper left")
    style.finish(fig, fig_path(S, "03_scoring_by_phase.png"))
    return fig


def impact_era() -> plt.Figure:
    t = read_table(S, "impact_era.csv")
    fig, ax = plt.subplots()
    for inn, col in [(1, style.BAT), (2, style.BOWL)]:
        g = t[t["innings"] == inn]
        ax.plot(g["season"], g["run_rate"], marker="o", lw=2, color=col, label=f"Innings {inn}")
    ax.axvline(2022.5, color=style.NEUTRAL, ls="--", lw=1)
    ax.text(2022.55, ax.get_ylim()[1] * 0.99, "Impact Player rule", va="top", fontsize=8, color=style.NEUTRAL)
    ax.set(title="IPL run rate by season", xlabel="Season", ylabel="Runs per over")
    ax.legend()
    style.finish(fig, fig_path(S, "03_impact_era.png"))
    return fig


def innings_totals(balls) -> plt.Figure:
    from data_prep.audit import innings_totals as it
    t = it(balls)
    t = t[t["innings"] == 1]
    fig, ax = plt.subplots()
    for ct, col in style.COMP_COLOURS.items():
        x = t.loc[t["comp_type"] == ct, "total_runs"]
        if len(x):
            ax.hist(x, bins=np.arange(40, 300, 10), histtype="step", lw=2, color=col, density=True, label=ct)
    ax.set(title="First innings totals", xlabel="Runs", ylabel="Density")
    ax.legend()
    style.finish(fig, fig_path(S, "03_innings_totals.png"))
    return fig


def chase_success() -> plt.Figure:
    t = read_table(S, "chase_success.csv")
    fig, ax = plt.subplots()
    ax.bar(t["band"], t["chase_win"], color=style.BOWL, width=0.6)
    for i, (v, n) in enumerate(zip(t["chase_win"], t["matches"], strict=True)):
        ax.text(i, v + 0.01, f"{v:.0%}\nn {n}", ha="center", va="bottom", fontsize=8)
    ax.set(title="Chase success by target", xlabel="Target", ylabel="Chasing side win share", ylim=(0, 1.05))
    style.finish(fig, fig_path(S, "03_chase_success.png"))
    return fig


def pace_spin() -> plt.Figure:
    t = read_table(S, "pace_spin_by_phase.csv")
    t = t[t["comp_type"].isin(["IPL", "T20I_full"])]
    fig, ax = plt.subplots()
    labels = [f"{c}\n{p}" for c, p in zip(t["comp_type"], t["phase"], strict=True)]
    ax.bar(labels, t["Spin"], color=style.BOWL, width=0.6)
    ax.set(title="Spin share of deliveries by phase", ylabel="Share of balls bowled by spin")
    style.finish(fig, fig_path(S, "03_pace_spin.png"))
    return fig


def run() -> None:
    balls = read_parquet(S, "balls_clean.parquet")
    for f in (lambda: missingness(balls), matches_by_season, scoring_by_phase, impact_era,
              lambda: innings_totals(balls), chase_success, pace_spin):
        plt.close(f())


if __name__ == "__main__":
    run()
