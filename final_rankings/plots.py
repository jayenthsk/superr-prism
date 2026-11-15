"""Stage 6 figures, what each top 10 rating is made of."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import style
from common.config import ROOT
from common.io import read_parquet

ROLE_FILE = {"bat": "batters", "bowl": "bowlers"}
COMPS = ["boundary", "dot", "out", "runs"]


def fig_dir():
    p = ROOT / "final_rankings" / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def components(role: str) -> plt.Figure:
    top = pd.read_csv(ROOT / "final_rankings" / f"top10_{ROLE_FILE[role]}.csv")
    r = read_parquet("player_ratings", f"player_ratings_{role}.parquet").set_index("player_id")
    v = r.loc[top["player_id"], [f"value_{c}" for c in COMPS]]
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(top))
    pos, neg = np.zeros(len(top)), np.zeros(len(top))
    for i, c in enumerate(COMPS):
        vals = v[f"value_{c}"].to_numpy()
        left = np.where(vals >= 0, pos, neg)
        ax.barh(y, vals, left=left, color=style.PALETTE[i], label=c, edgecolor="white", height=0.7)
        pos += np.clip(vals, 0, None)
        neg += np.clip(vals, None, 0)
    ax.scatter(top["prism"], y, color="black", s=16, zorder=3, label="PRISM")
    ax.set_yticks(y, top["player_name"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    title = "Batters" if role == "bat" else "Bowlers"
    ax.set(title=f"Top 10 {title.lower()}, PRISM split by component",
           xlabel="WP points added per 100 pressure balls above an average player")
    ax.legend(ncol=5, loc="lower right", fontsize=8)
    style.finish(fig, fig_dir() / f"09_components_{role}.png")
    return fig


def run() -> None:
    for role in ("bat", "bowl"):
        plt.close(components(role))


if __name__ == "__main__":
    run()
