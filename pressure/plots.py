"""Stage 2 figures, win probability and the pressure definition."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import style
from common.io import fig_path, read_parquet, read_table

S = "pressure"


def calibration() -> plt.Figure:
    t = read_table(S, "wp_calibration.csv")
    m = read_table(S, "wp_metrics.csv").set_index("innings")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, inn in zip(axes, (1, 2), strict=True):
        g = t[t["innings"] == inn]
        ax.plot([0, 1], [0, 1], color=style.NEUTRAL, ls="--", lw=1)
        ax.plot(g["pred"], g["obs"], marker="o", lw=2, color=style.BAT if inn == 1 else style.BOWL)
        r = m.loc[inn]
        ax.set(title=f"Innings {inn}, AUC {r['auc']:.3f}, Brier {r['brier']:.3f}", xlabel="Predicted win probability")
        ax.text(0.03, 0.92, f"Calibration slope {r['cal_slope']:.2f}", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel("Observed win share")
    fig.suptitle("Win probability calibration, out of fold by match", fontweight="bold")
    style.finish(fig, fig_path(S, "04_wp_calibration.png"))
    return fig


def wp_curves() -> plt.Figure:
    from pressure.wp_model import load_model
    model = load_model()
    fig, ax = plt.subplots()
    bl = np.arange(6, 121)
    for need, col in [(20, style.PALETTE[3]), (40, style.ACCENT), (60, style.BOWL), (90, style.BAT)]:
        d = pd.DataFrame({"balls_left": bl, "wkts_before": 4, "runs_needed": need})
        d["rrr"] = d["runs_needed"] * 6 / d["balls_left"]
        d["comp_type"], d["ipl_impact_era"] = "IPL", True
        ax.plot(bl, model.predict(d, 2, use_folds=False), lw=2, color=col, label=f"Need {need}")
    ax.invert_xaxis()
    ax.set(title="Chase win probability, 6 wickets in hand, IPL", xlabel="Balls left", ylabel="Win probability")
    ax.legend()
    style.finish(fig, fig_path(S, "04_wp_curves.png"))
    return fig


def li_heatmap() -> plt.Figure:
    g = read_table(S, "li_grid_chase.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    vmax = g["li"].quantile(0.98)
    for ax, w in zip(axes, (2, 5, 8), strict=True):
        p = g[g["wkts_before"] == w].pivot_table(index="runs_needed", columns="balls_left", values="li")
        im = ax.imshow(p.values, origin="lower", aspect="auto", cmap=style.SEQ, vmin=0, vmax=vmax,
                       extent=[p.columns.min(), p.columns.max(), p.index.min(), p.index.max()])
        ax.set(title=f"{w} wickets down", xlabel="Balls left")
        ax.invert_xaxis()
        ax.grid(False)
    axes[0].set_ylabel("Runs needed")
    fig.colorbar(im, ax=axes, label="Leverage index, average ball 1")
    fig.suptitle("Chase leverage by balls left and runs needed", fontweight="bold")
    fig.text(0.99, 0.005, style.SOURCE, ha="right", va="bottom", fontsize=7, color=style.NEUTRAL)
    fig.savefig(fig_path(S, "05_li_heatmap.png"), dpi=style.DPI, bbox_inches="tight")
    return fig


def li_by_over() -> plt.Figure:
    t = read_table(S, "li_by_over.csv")
    fig, ax = plt.subplots()
    for inn, col in [(1, style.BAT), (2, style.BOWL)]:
        g = t[t["innings"] == inn]
        ax.plot(g["over_no"], g["mean_li"], marker="o", lw=2, color=col, label=f"Innings {inn}")
    ax.axhline(1, color=style.NEUTRAL, ls="--", lw=1)
    ax.set(title="Average leverage by over", xlabel="Over", ylabel="Mean leverage index", xticks=range(1, 21))
    ax.legend()
    style.finish(fig, fig_path(S, "05_li_by_over.png"))
    return fig


def threshold_vs_li() -> plt.Figure:
    t = read_table(S, "li_threshold_comparison.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(t["check"], t["share"], color=style.BOWL, height=0.6)
    for i, v in enumerate(t["share"]):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set(title="Why scoreboard thresholds miss pressure", xlabel="Share", xlim=(0, 1.1))
    style.finish(fig, fig_path(S, "05_threshold_vs_li.png"))
    return fig


def dwp_by_tier(b: pd.DataFrame) -> plt.Figure:
    order = ["Low", "Normal", "High", "Extreme"]
    m = b.groupby("pressure_tier")["dwp"].apply(lambda x: x.abs().mean()).reindex(order)
    fig, ax = plt.subplots()
    ax.bar(order, 100 * m.values, color=[style.TIER_COLOURS[o] for o in order], width=0.6)
    for i, v in enumerate(m.values):
        ax.text(i, 100 * v, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set(title="Actual win probability swing per ball by pressure tier", ylabel="Mean absolute swing, WP points")
    style.finish(fig, fig_path(S, "05_dwp_by_tier.png"))
    return fig


def pick_worm_matches(b: pd.DataFrame, n: int = 3) -> list[int]:
    s = b[b["innings"] == 2].groupby("match_id").agg(ext=("pressure_tier", lambda x: (x == "Extreme").sum()),
                                                      comp=("comp_type", "first"))
    return s[s["comp"] == "IPL"].sort_values("ext", ascending=False).index[:n].tolist()


def worms(b: pd.DataFrame, match_ids: list[int] | None = None) -> plt.Figure:
    match_ids = match_ids or pick_worm_matches(b)
    teams = pd.read_csv(fig_path("data_prep", "x").parents[2] / "raw" / "teams.csv").drop_duplicates("team_id") \
        .set_index("team_id")["team_name"]
    fig, axes = plt.subplots(len(match_ids), 1, figsize=(9, 2.6 * len(match_ids)))
    for ax, mid in zip(np.atleast_1d(axes), match_ids, strict=True):
        m = b[b["match_id"] == mid].sort_values(["innings", "innings_ball_num"]).reset_index(drop=True)
        first_bat = m.loc[m["innings"] == 1, "batting_team_id"].iloc[0]
        wp1 = np.where(m["innings"] == 1, m["wp_pre"], 1 - m["wp_pre"])
        x = np.arange(len(m))
        for tier in ["High", "Extreme"]:
            sel = (m["pressure_tier"] == tier).to_numpy()
            ax.bar(x[sel], 1, width=1, color=style.TIER_COLOURS[tier], alpha=0.35, lw=0)
        ax.plot(x, wp1, color=style.BOWL, lw=2)
        ax.axvline((m["innings"] == 1).sum(), color=style.NEUTRAL, ls="--", lw=1)
        ax.set(ylim=(0, 1), ylabel=f"WP {teams.get(first_bat, first_bat)}"[:28],
               title=f"Match {mid}, shading marks High and Extreme leverage balls")
    np.atleast_1d(axes)[-1].set_xlabel("Ball of match")
    style.finish(fig, fig_path(S, "05_worms.png"))
    return fig


def tempo() -> plt.Figure:
    t = read_table(S, "tempo_vs_li.csv")
    fig, ax = plt.subplots()
    ax.plot(t["li_mid"], t["gap_median"], marker="o", lw=2, color=style.BOWL)
    ax.set_xscale("log")
    ax.set(title="Time between balls rises with leverage", xlabel="Leverage decile median, log scale",
           ylabel="Median gap between balls, seconds")
    style.finish(fig, fig_path(S, "05_tempo_vs_li.png"))
    return fig


def run() -> None:
    b = read_parquet(S, "balls_pressure.parquet")
    for f in (calibration, wp_curves, li_heatmap, li_by_over, threshold_vs_li, lambda: dwp_by_tier(b),
              lambda: worms(b), tempo):
        plt.close(f())


if __name__ == "__main__":
    run()
