"""Stage 4 figures, reliability, myths, intervals and robustness."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import style
from common.io import fig_path, read_table
from reliability.uncertainty import final_order

S = "reliability"
ROLE_COL = {"bat": style.BAT, "bowl": style.BOWL}
ROLE_NAME = {"bat": "Batters", "bowl": "Bowlers"}


def split_half(basis: str = "resid") -> plt.Figure:
    t = read_table(S, "split_half_summary.csv")
    t = t[t["metric_basis"] == basis]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    comps = ["boundary", "dot", "out", "runs"]
    for ax, role in zip(axes, ("bat", "bowl"), strict=True):
        g = t[t["role"] == role].set_index(["component", "kind"])
        x = np.arange(len(comps))
        for i, (kind, col) in enumerate([("level", ROLE_COL[role]), ("delta", style.NEUTRAL)]):
            vals = [g.loc[(c, kind), "r_mean"] for c in comps]
            lo = [g.loc[(c, kind), "r_mean"] - g.loc[(c, kind), "r_lo"] for c in comps]
            hi = [g.loc[(c, kind), "r_hi"] - g.loc[(c, kind), "r_mean"] for c in comps]
            ax.bar(x + (i - 0.5) * 0.38, vals, 0.36, yerr=[lo, hi], color=col, capsize=3,
                   label="Normal level" if kind == "level" else "Pressure minus normal")
            for xi, v in zip(x, vals, strict=True):
                ax.text(xi + (i - 0.5) * 0.38, max(v, 0) + 0.03, f"{v:.2f}", ha="center", fontsize=8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, comps)
        ax.set(title=ROLE_NAME[role])
    axes[0].set_ylabel("Split-half correlation, 20 random match splits")
    axes[0].legend(loc="upper right")
    fig.suptitle("Skill repeats, the clutch gap barely does", fontweight="bold")
    style.finish(fig, fig_path(S, "07_split_half.png"))
    return fig


def dot_streak() -> plt.Figure:
    t = read_table(S, "myth_dot_streak.csv")
    fig, ax = plt.subplots()
    for i, ph in enumerate(["PP", "MID", "DEATH"]):
        g = t[t["phase"] == ph]
        ax.plot(g["streak"], 100 * g["excess"], marker="o", lw=2, color=style.PALETTE[i], label=ph)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Wicket rate after dot-ball streaks, against context",
           xlabel="Dots in a row before the ball, 5 means 5 plus", ylabel="Wickets per 100 balls above expected")
    ax.legend()
    style.finish(fig, fig_path(S, "07_dot_streak.png"))
    return fig


def tight_over() -> plt.Figure:
    t = read_table(S, "myth_tight_over.csv")
    fig, ax = plt.subplots()
    phases = ["PP", "MID", "DEATH"]
    x = np.arange(3)
    for i, (tight, col) in enumerate([(False, style.NEUTRAL), (True, style.BOWL)]):
        g = t[t["tight"] == tight].set_index("phase").reindex(phases)
        ax.bar(x + (i - 0.5) * 0.38, g["excess_per_100"], 0.36, color=col,
               label="After a tight over, 4 runs or fewer" if tight else "After any other over")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, phases)
    ax.set(title="Next over wickets after a tight over, against context", ylabel="Wickets per 100 balls above expected")
    ax.legend()
    style.finish(fig, fig_path(S, "07_tight_over.png"))
    return fig


def rank_forest(role: str, n: int = 25) -> plt.Figure:
    t = final_order(role).head(n)
    fig, ax = plt.subplots(figsize=(7, 0.28 * n + 1.2))
    y = np.arange(len(t))
    cols = [ROLE_COL[role] if r <= 10 else style.NEUTRAL for r in t["rank"]]
    ax.hlines(y, t["rank_lo80"], t["rank_hi80"], color=cols, lw=2)
    ax.scatter(t["median_rank"], y, color=cols, s=24, zorder=3)
    ax.set_yticks(y, t["player_name"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(10.5, color=style.NEUTRAL, ls="--", lw=1)
    ax.set(title=f"{ROLE_NAME[role]}, bootstrap rank with 80 percent interval", xlabel="Rank among eligible players")
    style.finish(fig, fig_path(S, f"07_rank_forest_{role}.png"))
    return fig


def robustness_heatmap(role: str) -> plt.Figure:
    mem = read_table(S, "robustness_membership.csv")
    mem = mem[mem["role"] == role]
    top = final_order(role).head(15)
    variants = read_table(S, "robustness.csv").query("role == @role")["variant"].tolist()
    m = pd.DataFrame(0, index=top["player_id"], columns=variants)
    for _, r in mem.iterrows():
        if r["player_id"] in m.index:
            m.loc[r["player_id"], r["variant"]] = 1
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.imshow(m.values, cmap="Blues" if role == "bowl" else "RdPu", vmin=0, vmax=1.3, aspect="auto")
    ax.set_yticks(range(len(top)), top["player_name"], fontsize=8)
    ax.set_xticks(range(len(variants)), variants, rotation=45, ha="right", fontsize=8)
    ax.grid(False)
    share = m.mean(axis=1)
    for i, s in enumerate(share):
        ax.text(len(variants) - 0.4, i, f"{s:.0%}", va="center", fontsize=8)
    ax.set_title(f"{ROLE_NAME[role]}, top 10 membership across robustness variants, shaded means in")
    style.finish(fig, fig_path(S, f"07_robustness_{role}.png"))
    return fig


def run() -> None:
    plt.close(split_half())
    plt.close(dot_streak())
    plt.close(tight_over())
    for role in ("bat", "bowl"):
        plt.close(rank_forest(role))
        plt.close(robustness_heatmap(role))


if __name__ == "__main__":
    run()
