"""Stage 3 figures, baseline, shrinkage and opposition adjustment."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import style
from common.io import fig_path, read_parquet, read_table

S = "player_ratings"
ROLE_COL = {"bat": style.BAT, "bowl": style.BOWL}
ROLE_NAME = {"bat": "Batters", "bowl": "Bowlers"}


def shrinkage(role: str = "bat", comp: str = "runs") -> plt.Figure:
    r = read_parquet(S, f"player_ratings_{role}.parquet")
    r = r[r["n_eff"] > 0]
    raw = r[f"rawP_{comp}"] - r[f"thetaN_{comp}"]
    fig, ax = plt.subplots()
    ax.scatter(r["n_eff"], raw, s=10, color=style.NEUTRAL, alpha=0.5, label="Raw pressure gap")
    ax.scatter(r["n_eff"], r[f"delta_{comp}"], s=10, color=ROLE_COL[role], label="Shrunk clutch effect")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set(title=f"{ROLE_NAME[role]}, clutch effect on {comp} per ball before and after shrinkage",
           xlabel="Effective pressure balls, log scale", ylabel="Residual per ball")
    ax.legend()
    style.finish(fig, fig_path(S, f"06_shrinkage_{role}.png"))
    return fig


def component_values(role: str) -> plt.Figure:
    v = read_table(S, f"component_values_{role}.csv")
    v["bucket"] = "Inn " + v["innings"].astype(str) + " " + v["phase"]
    comps = ["boundary", "dot", "out", "runs"]
    fig, ax = plt.subplots()
    w = 0.2
    x = np.arange(len(v))
    for i, c in enumerate(comps):
        ax.bar(x + (i - 1.5) * w, v[f"v_{c}"], w, color=style.PALETTE[i], label=c)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, v["bucket"], rotation=20)
    side = "batting" if role == "bat" else "bowling"
    ax.set(title=f"WP value of each outcome on pressure balls, {side} side view",
           ylabel="WP points per outcome")
    ax.legend(ncol=4)
    style.finish(fig, fig_path(S, f"06_component_values_{role}.png"))
    return fig


def opposition(role: str) -> plt.Figure:
    t = read_table(S, f"opposition_rank_changes_{role}.csv")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(t["before"], t["after"], s=14, color=ROLE_COL[role])
    lim = [min(t["before"].min(), t["after"].min()), max(t["before"].max(), t["after"].max())]
    ax.plot(lim, lim, color=style.NEUTRAL, ls="--", lw=1)
    for _, r in t.nsmallest(5, "rank_after").iterrows():
        ax.annotate(r["player_name"], (r["before"], r["after"]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set(title=f"{ROLE_NAME[role]}, PRISM before and after opposition adjustment",
           xlabel="Without opponent strength", ylabel="With opponent strength")
    style.finish(fig, fig_path(S, f"06_opposition_{role}.png"))
    return fig


def run() -> None:
    for role in ("bat", "bowl"):
        for f in (shrinkage, component_values, opposition):
            plt.close(f(role))


if __name__ == "__main__":
    run()
