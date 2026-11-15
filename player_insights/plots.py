"""Stage 5 figures, execution, bounce-back, shot mix and hard entry."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import style
from common.io import fig_path, read_table

S = "player_insights"


def execution() -> plt.Figure:
    t = read_table(S, "e1_league_execution.csv")
    fig, ax = plt.subplots()
    x = np.arange(len(t))
    for i, (col, c) in enumerate([("normal", style.NEUTRAL), ("pressure", style.BOWL)]):
        ax.bar(x + (i - 0.5) * 0.38, 100 * t[col], 0.36, color=c, label=f"{col.title()} death balls")
        for xi, v in zip(x, t[col], strict=True):
            ax.text(xi + (i - 0.5) * 0.38, 100 * v + 0.3, f"{100 * v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, t["delivery"].str.replace("_", " "))
    ax.set(title="Pace bowlers at the death, delivery mix on pressure balls", ylabel="Share of deliveries, percent")
    ax.legend()
    style.finish(fig, fig_path(S, "08_execution.png"))
    return fig


def bounce_back() -> plt.Figure:
    t = read_table(S, "e1_bounce_back_league.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    lab = ["Any other ball", "Ball after a boundary"]
    x = np.arange(2)
    ax.bar(x - 0.2, 100 * t["boundary_rate"], 0.38, color=style.BOWL, label="Actual")
    ax.bar(x + 0.2, 100 * t["expected_boundary"], 0.38, color=style.NEUTRAL, label="Expected from context")
    for xi, a, e in zip(x, t["boundary_rate"], t["expected_boundary"], strict=True):
        ax.text(xi - 0.2, 100 * a + 0.3, f"{100 * a:.1f}", ha="center", fontsize=9)
        ax.text(xi + 0.2, 100 * e + 0.3, f"{100 * e:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, lab)
    ax.set(title="Boundary rate on a bowler's next ball after being hit", ylabel="Boundary percent")
    ax.legend()
    style.finish(fig, fig_path(S, "08_bounce_back.png"))
    return fig


def shot_mix() -> plt.Figure:
    t = read_table(S, "e2_shot_mix.csv")
    t = t[t["eligible"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = {"Composed": style.BOWL, "Adaptive": style.PALETTE[3], "Forced": style.ACCENT, "Frozen": style.BAT}
    for q, c in colours.items():
        g = t[t["quadrant"] == q]
        ax.scatter(g["excess_drift"], g["press_prod"], s=18, color=c, label=f"{q} {len(g)}")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(t["excess_drift"].median(), color=style.NEUTRAL, ls="--", lw=1)
    for _, r in t.nlargest(4, "press_prod").iterrows():
        ax.annotate(r["player_name"], (r["excess_drift"], r["press_prod"]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set(title="Shot-mix drift under pressure against pressure productivity",
           xlabel="Drift above league for the same phase mix, JS divergence",
           ylabel="Pressure runs residual minus normal, per ball")
    ax.legend(fontsize=8)
    style.finish(fig, fig_path(S, "08_shot_mix_quadrants.png"))
    return fig


def hard_entry() -> plt.Figure:
    t = read_table(S, "e3_hard_entry.csv")
    t = t[t["eligible"]].sort_values("hard_gap_runs")
    t = t.iloc[list(range(8)) + list(range(len(t) - 8, len(t)))]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cols = [style.BAT if v > 0 else style.NEUTRAL for v in t["hard_gap_runs"]]
    ax.barh(t["player_name"], t["hard_gap_runs"], color=cols)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="First ten balls after a hard entry, against own normal entries",
           xlabel="Runs per ball above context, shrunk")
    ax.tick_params(axis="y", labelsize=8)
    style.finish(fig, fig_path(S, "08_hard_entry.png"))
    return fig


def run() -> None:
    for f in (execution, bounce_back, shot_mix, hard_entry):
        plt.close(f())


if __name__ == "__main__":
    run()
