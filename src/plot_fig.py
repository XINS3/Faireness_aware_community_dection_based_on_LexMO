# generate_mw_figure.py
#
# Produces a 2x2 figure analysing MutexWatershed-Transform on synthetic networks.
#
# Layout
# ──────
#   col 0: color-node networks  (individual-node colouring)
#   col 1: color-full networks  (full-clique colouring)
#
#   row 0: sweep transform_p (same_color_p = 0.0 … 1.0), fixed p_sensitive = 0.5
#          x-axis = same_color_p
#          CSV source: benchmark_color-{node,full}_1000_r01_K2_c05.csv
#
#   row 1: sweep p_sensitive, fixed transform_p = 0.5
#          x-axis = p_sensitive  (parsed from filename suffix _cXX)
#          CSV source: benchmark_color-{node,full}_1000_r01_K2_cXX.csv  (glob)
#
# Each panel plots two lines: modularity_mean and fexp_mean (with ±1 SD shading).
#
# Configuration
# ─────────────
# Edit the paths/patterns below to match your logs directory.

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

LOGS_DIR = Path("../logs")

# CSVs for row 0 (fixed p_sensitive=0.5, sweep transform_p)
CSV_SWEEP_P = {
    "color-node": LOGS_DIR / "benchmark_color-node_1000_r01_K2_c05.csv",
    "color-full": LOGS_DIR / "benchmark_color-full_1000_r01_K2_c05.csv",
}

# Glob patterns for row 1 (sweep p_sensitive, fixed transform_p=0.5)
# Files are expected to be named: benchmark_color-{node,full}_1000_r01_K2_cXX.csv
# where XX encodes p_sensitive*100 (e.g. c05 -> 0.05, c50 -> 0.50).
GLOB_SWEEP_PSENS = {
    "color-node": "benchmark_color-node_1000_r01_K2_c*.csv",
    "color-full": "benchmark_color-full_1000_r01_K2_c*.csv",
}

# Which transform_p to fix for row 1
FIXED_TRANSFORM_P = 0.5
FIXED_TRANSFORM_P_STRATEGY = f"same_color_p={FIXED_TRANSFORM_P}"

OUTPUT_FILE = "mw_sweep_figure.pdf"

# Metrics to plot in each panel (name in CSV, display label, colour)
METRICS = [
    ("modularity_mean", "modularity_std", "Modularity",        "#1f77b4"),
    ("fexp_mean",       "fexp_std",       "Prop. fairness",    "#ff7f0e"),
]

COL_LABELS = {
    "color-node": "Individual-node colouring\n(color-node)",
    "color-full": "Full-clique colouring\n(color-full)",
}

ROW_LABELS = [
    r"Sweep $p_{\rm sc}$ (same-color attract. prob.), fixed $p_{\rm sens}=0.5$",
    r"Sweep $p_{\rm sens}$ (minority proportion), fixed $p_{\rm sc}=0.5$",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_p_sensitive(path: Path) -> float:
    """Extract p_sensitive from filename suffix _cXX.
    Convention: _c05 -> 0.5, _c01 -> 0.1, _c25 -> 2.5 (i.e. digits are tenths).
    """
    m = re.search(r"_c(\d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot parse p_sensitive from filename: {path.name}")
    digits = m.group(1)
    # digits are tenths: c05 -> 0.5, c01 -> 0.1, c10 -> 1.0
    return int(digits) / 10.0


def load_mw_transform(csv_path: Path, strategy_filter: str) -> pd.DataFrame:
    """
    Load MutexWatershed-Transform rows matching strategy_filter from a CSV.
    Returns a DataFrame with columns: same_color_p, modularity_mean, modularity_std,
    fexp_mean, fexp_std.
    """
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    mask = (df["algorithm"] == "MutexWatershed-Transform")
    if strategy_filter:
        mask &= df["strategy"].astype(str) == strategy_filter
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame()
    # Parse same_color_p from the strategy column ("same_color_p=0.5" -> 0.5)
    def _parse_p(s):
        m = re.search(r"same_color_p=([0-9.]+)", str(s))
        return float(m.group(1)) if m else float("nan")
    sub["same_color_p"] = sub["strategy"].apply(_parse_p)
    return sub.sort_values("same_color_p").reset_index(drop=True)


def load_sweep_p_sensitive(glob_pattern: str, fixed_strategy: str) -> pd.DataFrame:
    """
    For each CSV matching glob_pattern, extract p_sensitive from filename and
    load the MutexWatershed-Transform row with fixed_strategy.
    Returns DataFrame with columns: p_sensitive, modularity_mean, modularity_std,
    fexp_mean, fexp_std.
    """
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
        records.append({
            "p_sensitive":    p_sens,
            "modularity_mean": row["modularity_mean"],
            "modularity_std":  row["modularity_std"],
            "fexp_mean":       row["fexp_mean"],
            "fexp_std":        row["fexp_std"],
        })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("p_sensitive").reset_index(drop=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_panel(ax, x, data: pd.DataFrame, x_col: str, xlabel: str):
    """Plot modularity and fexp lines with shaded SD bands on ax."""
    if data.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        return

    for mean_col, std_col, label, color in METRICS:
        if mean_col not in data.columns:
            continue
        y     = data[mean_col].values
        yerr  = data[std_col].fillna(0).values
        xvals = data[x_col].values
        ax.plot(xvals, y, marker="o", color=color, label=label, linewidth=1.8)
        ax.fill_between(xvals, y - yerr, y + yerr, color=color, alpha=0.15)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8, loc="best")


def build_figure():
    net_types = ["color-node", "color-full"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.42, wspace=0.30)

    # ── Column headers ────────────────────────────────────────────────────────
    for col, net in enumerate(net_types):
        axes[0, col].set_title(COL_LABELS[net], fontsize=10, fontweight="bold", pad=8)

    # ── Row 0: sweep transform_p, fixed p_sensitive=0.5 ──────────────────────
    for col, net in enumerate(net_types):
        ax = axes[0, col]
        csv = CSV_SWEEP_P[net]
        # Load all transform_p rows (no strategy filter -> all same_color_p values)
        data = load_mw_transform(csv, strategy_filter="")
        plot_panel(ax, None, data, x_col="same_color_p",
                   xlabel=r"$p_{\rm sc}$ (same-color attraction prob.)")
        ax.set_ylabel("Metric value", fontsize=9)

    # ── Row 1: sweep p_sensitive, fixed transform_p=0.5 ──────────────────────
    for col, net in enumerate(net_types):
        ax = axes[1, col]
        data = load_sweep_p_sensitive(GLOB_SWEEP_PSENS[net], FIXED_TRANSFORM_P_STRATEGY)
        plot_panel(ax, None, data, x_col="p_sensitive",
                   xlabel=r"$p_{\rm sens}$ (minority proportion)")
        ax.set_ylabel("Metric value", fontsize=9)

    # ── Row labels on the left ────────────────────────────────────────────────
    for row, label in enumerate(ROW_LABELS):
        axes[row, 0].annotate(
            label, xy=(0, 0.5), xytext=(-axes[row, 0].yaxis.labelpad - 30, 0),
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_figure()