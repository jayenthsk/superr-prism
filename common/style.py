"""One chart style for notebooks and dashboard."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAT = "#E4007C"
BOWL = "#254AA5"
NEUTRAL = "#7A7A7A"
ACCENT = "#C98A00"
# Tiers are ordinal, so one hue from light to dark
TIER_COLOURS = {"Low": "#F6DFC4", "Normal": "#E9AE6B", "High": "#C9661C", "Extreme": "#8A3500"}
SEQ = "Oranges"
DIVERGING = "RdBu"
# Validated for colour blind separation in this order, never cycled
PALETTE = [BAT, BOWL, ACCENT, "#2E7D32", "#8E3BBF", NEUTRAL]
COMP_COLOURS = {"IPL": BAT, "T20I_full": BOWL, "T20I_associate": ACCENT, "Emerging": "#2E7D32"}
DPI = 200


def apply() -> None:
    plt.rcParams.update({
        "figure.figsize": (8, 4.5), "figure.dpi": 100, "savefig.dpi": DPI,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.3, "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.labelsize": 10, "font.size": 10, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })


def finish(fig, path=None) -> None:
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=DPI, bbox_inches="tight")


def plotly_layout(title: str = "") -> dict:
    return dict(title=title, template="simple_white", font=dict(size=13),
                colorway=PALETTE, margin=dict(l=40, r=20, t=50, b=40))


apply()
