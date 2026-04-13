# generate_mw_figure.py
#
# 2x2 figure for MutexWatershed-Transform on synthetic networks.
# Each panel shows all benchmark metrics as lines + a #communities inset.
#
# Layout
# ──────
#   col 0 : color-node   col 1 : color-full
#   row 0 : sweep same_color_p (0..1), fixed p_sensitive=0.5
#   row 1 : sweep p_sensitive,  fixed same_color_p=0.5

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

LOGS_DIR = Path("../logs")

CSV_SWEEP_P = {
    "color-node": LOGS_DIR / "benchmark_color-node_1000_r01_K2_c05.csv",
    "color-full": LOGS_DIR / "benchmark_color-full_1000_r01_K2_c05.csv",
}

GLOB_SWEEP_PSENS = {
    "color-node": "benchmark_color-node_1000_r01_K2_c*.csv",
    "color-full": "benchmark_color-full_1000_r01_K2_c*.csv",
}

FIXED_TRANSFORM_P          = 0.5
FIXED_TRANSFORM_P_STRATEGY = f"same_color_p={FIXED_TRANSFORM_P}"

OUTPUT_FILE = "mw_sweep_figure.pdf"

# (mean_col, std_col, label, color, linestyle)
METRICS = [
    ("modularity_mean",  "modularity_std",  "Modularity",                  "#1f77b4", "-"),
    ("unfairness_mean",  "unfairness_std",  "1 - |Unfairness| (fairness)", "#d62728", "-"),
]

# Columns to extract from CSV (all metric means + stds + n_communities)
METRIC_COLS = [c for pair in [(m[0], m[1]) for m in METRICS] for c in pair]

COL_LABELS = {
    "color-node": "Individual-node colouring\n(color-node)",
    "color-full": "Full-clique colouring\n(color-full)",
}

ROW_LABELS = [
    r"Sweep $p_{\rm sc}$, fixed $p_{\rm sens}=0.5$",
    r"Sweep $p_{\rm sens}$, fixed $p_{\rm sc}=0.5$",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_p_sensitive(path: Path) -> float:
    """_c05 -> 0.5, _c01 -> 0.1, _c10 -> 1.0  (digits are tenths)."""
    m = re.search(r"_c(\d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot parse p_sensitive from: {path.name}")
    return int(m.group(1)) / 10.0


def _parse_same_color_p(s):
    m = re.search(r"same_color_p=([0-9.]+)", str(s))
    return float(m.group(1)) if m else float("nan")


def load_mw_transform(csv_path: Path, strategy_filter: str) -> pd.DataFrame:
    """Load all MutexWatershed-Transform rows, optionally filtered by strategy."""
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    mask = df["algorithm"] == "MutexWatershed-Transform"
    if strategy_filter:
        mask &= df["strategy"].astype(str) == strategy_filter
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["same_color_p"] = sub["strategy"].apply(_parse_same_color_p)
    return sub.sort_values("same_color_p").reset_index(drop=True)


def load_sweep_p_sensitive(glob_pattern: str, fixed_strategy: str) -> pd.DataFrame:
    """One row per _cXX CSV: extract p_sensitive + fixed-strategy metrics."""
    records = []
    for csv_path in sorted(LOGS_DIR.glob(glob_pattern)):
        try:
            p_sens = parse_p_sensitive(csv_path)
        except ValueError as e:
            print(f"WARNING: {e}")
            continue
        sub = load_mw_transform(csv_path, fixed_strategy)
        if sub.empty:
            continue
        row = sub.iloc[0]
        rec = {"p_sensitive": p_sens}
        for col in METRIC_COLS:
            rec[col] = row[col] if col in row.index else float("nan")
        if "n_communities_mean" in row.index:
            rec["n_communities_mean"] = row["n_communities_mean"]
            rec["n_communities_std"]  = row.get("n_communities_std", float("nan"))
        records.append(rec)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("p_sensitive").reset_index(drop=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_panel(ax, data: pd.DataFrame, x_col: str, xlabel: str):
    """Plot all metric lines with SD shading; add #communities inset."""
    if data.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=9)
        return

    xvals = data[x_col].values

    for mean_col, std_col, label, color, ls in METRICS:
        if mean_col not in data.columns:
            continue
        y    = data[mean_col].values
        yerr = data[std_col].fillna(0).values if std_col in data.columns else np.zeros_like(y)
        if mean_col == "unfairness_mean":
            y    = np.clip(1 - np.abs(y), 0, 1)
            yerr = np.clip(yerr, 0, 1)
        ax.plot(xvals, y, marker="o", color=color, label=label,
                linewidth=1.5, linestyle=ls, markersize=4)
        ax.fill_between(xvals, y - yerr, y + yerr, color=color, alpha=0.10)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=6.5, loc="upper left", ncol=2,
              framealpha=0.7, borderpad=0.4, labelspacing=0.3)

    # ── #communities inset (top-right) ──────────────────────────────────────
    if "n_communities_mean" in data.columns and not data["n_communities_mean"].isna().all():
        inset = ax.inset_axes([0.60, 0.60, 0.38, 0.36])
        yn    = data["n_communities_mean"].values
        ynerr = data["n_communities_std"].fillna(0).values if "n_communities_std" in data.columns else np.zeros_like(yn)
        inset.plot(xvals, yn, marker="s", color="#7f7f7f", linewidth=1.4, markersize=4)
        inset.fill_between(xvals, yn - ynerr, yn + ynerr, color="#7f7f7f", alpha=0.20)
        inset.set_ylabel("#comm", fontsize=7)
        inset.tick_params(labelsize=6.5)
        inset.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        inset.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
        inset.patch.set_facecolor("white")
        inset.patch.set_alpha(0.85)
        for spine in inset.spines.values():
            spine.set_linewidth(0.8)


def build_figure():
    net_types = ["color-node", "color-full"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(hspace=0.45, wspace=0.32)

    for col, net in enumerate(net_types):
        axes[0, col].set_title(COL_LABELS[net], fontsize=10, fontweight="bold", pad=8)

    # Row 0: sweep 1-same_color_p (left = max repulsion)
    for col, net in enumerate(net_types):
        ax   = axes[0, col]
        data = load_mw_transform(CSV_SWEEP_P[net], strategy_filter="")
        if not data.empty:
            data = data.copy()
            data["repulsion_p"] = 1 - data["same_color_p"]
            data = data.sort_values("repulsion_p").reset_index(drop=True)
        plot_panel(ax, data, x_col="repulsion_p",
                   xlabel=r"$1-p_{\rm sc}$ (same-color repulsion prob.)")
        ax.set_ylabel("Metric value", fontsize=9)

    # Row 1: sweep p_sensitive
    for col, net in enumerate(net_types):
        ax   = axes[1, col]
        data = load_sweep_p_sensitive(GLOB_SWEEP_PSENS[net], FIXED_TRANSFORM_P_STRATEGY)
        plot_panel(ax, data, x_col="p_sensitive",
                   xlabel=r"$p_{\rm sens}$ (minority proportion)")
        ax.set_ylabel("Metric value", fontsize=9)

    # Row labels
    for row, label in enumerate(ROW_LABELS):
        axes[row, 0].annotate(
            label, xy=(0, 0.5), xytext=(-axes[row, 0].yaxis.labelpad - 28, 0),
            xycoords="axes fraction", textcoords="offset points",
            fontsize=8.5, ha="right", va="center", rotation=90,
        )

    fig.suptitle(
        "MutexWatershed-Transform: sensitivity to $p_{\\rm sc}$ and $p_{\\rm sens}$\n"
        "on synthetic networks",
        fontsize=11, fontweight="bold", y=1.01,
    )

    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    print(f"Figure saved to {OUTPUT_FILE}")
    return fig


if __name__ == "__main__":
    build_figure()