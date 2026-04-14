# generate_mw_figure.py
#
# 2x2 figure for MutexWatershed-Transform on synthetic networks.
# Row 0: sweep 1-same_color_p (repulsion prob.), fixed p_sensitive=0.5
# Row 1: sweep p_sensitive, fixed same_color_p=0.5
# Each panel: modularity, prop. fairness, modularity fairness (1-|unfairness|)
# Inset: number of communities

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
LOG_SWEEP_P = {
    "color-node": LOGS_DIR / "benchmark_color-node_1000_r01_K2_c05.txt",
    "color-full": LOGS_DIR / "benchmark_color-full_1000_r01_K2_c05.txt",
}

GLOB_SWEEP_PSENS = {
    "color-node": "benchmark_color-node_1000_r01_K2_c*.csv",
    "color-full": "benchmark_color-full_1000_r01_K2_c*.csv",
}
GLOB_LOG_PSENS = {
    "color-node": "benchmark_color-node_1000_r01_K2_c*.txt",
    "color-full": "benchmark_color-full_1000_r01_K2_c*.txt",
}

FIXED_TRANSFORM_P          = 0.5
FIXED_TRANSFORM_P_STRATEGY = f"same_color_p={FIXED_TRANSFORM_P}"

OUTPUT_FILE = "mw_sweep_figure.pdf"

# (mean_col, std_col, label, color, linestyle)
METRICS = [
    ("modularity_mean",  "modularity_std",  "Modularity",            "#1f77b4", "-"),
    ("fexp_mean",        "fexp_std",        "Prop. fairness",        "#ff7f0e", "-"),
    ("unfairness_mean",  "unfairness_std",  "Modularity fairness",   "#d62728", "-"),
]

COL_LABELS = {
    "color-node": "Individual-node colouring\n(color-node)",
    "color-full": "Full-clique colouring\n(color-full)",
}

ROW_LABELS = [
    r"Sweep $1-p_{\rm sc}$, fixed $p_{\rm sens}=0.5$",
    r"Sweep $p_{\rm sens}$, fixed $p_{\rm sc}=0.5$",
]

# ── Log parsing: extract #communities per (algorithm, strategy) ──────────────

ALG_RE  = re.compile(r"▶ (\S+)\s+alpha=\S+\s+strategy=(\S+)")
COMM_RE = re.compile(r"Computing metrics over (\d+) communities")

def parse_ncomm_from_log(log_path: Path) -> dict:
    """
    Returns dict: strategy_str -> (mean, std) of #communities
    for MutexWatershed-Transform runs only.
    """
    if not log_path.exists():
        return {}
    lines = log_path.read_text(encoding="utf-8").splitlines()
    records = {}   # strategy -> [counts]
    current = None
    for line in lines:
        m = ALG_RE.search(line)
        if m:
            alg, strat = m.group(1), m.group(2)
            current = strat if alg == "MutexWatershed-Transform" else None
            if current and current not in records:
                records[current] = []
            continue
        m = COMM_RE.search(line)
        if m and current:
            records[current].append(int(m.group(1)))
    result = {}
    for strat, counts in records.items():
        if counts:
            result[strat] = (np.mean(counts),
                             np.std(counts, ddof=1) if len(counts) > 1 else 0.0)
    return result


# ── CSV loading ───────────────────────────────────────────────────────────────

def _parse_same_color_p(s):
    m = re.search(r"same_color_p=([0-9.]+)", str(s))
    return float(m.group(1)) if m else float("nan")


def parse_p_sensitive(path: Path) -> float:
    """_c05 -> 0.5, _c01 -> 0.1, _c10 -> 1.0  (digits are tenths)."""
    m = re.search(r"_c(\d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot parse p_sensitive from: {path.name}")
    return int(m.group(1)) / 10.0


def load_mw_transform(csv_path: Path, log_path: Path, strategy_filter: str) -> pd.DataFrame:
    """Load MutexWatershed-Transform rows and merge in #communities from log."""
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

    # Merge #communities from log
    ncomm = parse_ncomm_from_log(log_path)
    sub["n_communities_mean"] = sub["strategy"].map(
        lambda s: ncomm[s][0] if s in ncomm else float("nan"))
    sub["n_communities_std"]  = sub["strategy"].map(
        lambda s: ncomm[s][1] if s in ncomm else float("nan"))

    return sub.sort_values("same_color_p").reset_index(drop=True)


def load_sweep_p_sensitive(csv_glob: str, log_glob: str, fixed_strategy: str) -> pd.DataFrame:
    """One point per _cXX network: fixed strategy, x = p_sensitive."""
    csv_paths = {parse_p_sensitive(p): p
                 for p in LOGS_DIR.glob(csv_glob)
                 if re.search(r"_c(\d+)", p.stem)}
    log_paths = {parse_p_sensitive(p): p
                 for p in LOGS_DIR.glob(log_glob)
                 if re.search(r"_c(\d+)", p.stem)}
    print(f"  [sweep p_sens] found {len(csv_paths)} CSV(s): {sorted(csv_paths.keys())}")
    print(f"  [sweep p_sens] found {len(log_paths)} log(s): {sorted(log_paths.keys())}")
    records = []
    for p_sens in sorted(csv_paths):
        csv_path = csv_paths[p_sens]
        log_path = log_paths.get(p_sens, Path("nonexistent"))
        sub = load_mw_transform(csv_path, log_path, fixed_strategy)
        if sub.empty:
            continue
        row = sub.iloc[0]
        rec = {"p_sensitive": p_sens}
        for col in list(sub.columns):
            rec[col] = row[col]
        records.append(rec)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("p_sensitive").reset_index(drop=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_panel(ax, data: pd.DataFrame, x_col: str, xlabel: str):
    """Lines + SD shading for all metrics; #communities inset bottom-left."""
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

    # ── #communities inset (bottom-left) ─────────────────────────────────────
    ncomm_valid = (data["n_communities_mean"].notna() if "n_communities_mean" in data.columns
                    else pd.Series(dtype=float))
    if len(ncomm_valid.dropna()) >= 1:
        inset = ax.inset_axes([0.28, 0.18, 0.36, 0.34])
        yn    = data["n_communities_mean"].values
        ynerr = (data["n_communities_std"].fillna(0).values
                 if "n_communities_std" in data.columns else np.zeros_like(yn))
        inset.plot(xvals, yn, marker="s", color="#7f7f7f", linewidth=1.4, markersize=4)
        inset.fill_between(xvals, yn - ynerr, yn + ynerr, color="#7f7f7f", alpha=0.20)
        inset.set_ylabel("#comm", fontsize=7)
        inset.tick_params(labelsize=6.5)
        inset.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        inset.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
        inset.patch.set_facecolor("white")
        inset.patch.set_alpha(0.9)
        for spine in inset.spines.values():
            spine.set_linewidth(0.8)


def build_figure():
    net_types = ["color-node", "color-full"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(hspace=0.45, wspace=0.32, top=0.88)

    # ── Shared legend at the top ──────────────────────────────────────────────
    dummy_ax = axes[0, 0]
    handles = []
    for _, _, label, color, ls in METRICS:
        h, = dummy_ax.plot([], [], color=color, linestyle=ls,
                           linewidth=1.8, marker="o", markersize=5, label=label)
        handles.append(h)
    fig.legend(handles=handles, loc="upper center", ncol=len(METRICS),
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.97),
               bbox_transform=fig.transFigure)

    # ── Column headers ────────────────────────────────────────────────────────
    for col, net in enumerate(net_types):
        axes[0, col].set_title(COL_LABELS[net], fontsize=10, fontweight="bold", pad=8)

    # ── Row 0: sweep 1-same_color_p ───────────────────────────────────────────
    for col, net in enumerate(net_types):
        ax   = axes[0, col]
        data = load_mw_transform(CSV_SWEEP_P[net], LOG_SWEEP_P[net], strategy_filter="")
        if not data.empty:
            data = data.copy()
            data["repulsion_p"] = 1 - data["same_color_p"]
            data = data.sort_values("repulsion_p").reset_index(drop=True)
        plot_panel(ax, data, x_col="repulsion_p",
                   xlabel=r"$1-p_{\rm sc}$ (same-color repulsion prob.)")
        ax.set_ylabel("Metric value", fontsize=9)

    # ── Row 1: sweep p_sensitive ──────────────────────────────────────────────
    for col, net in enumerate(net_types):
        ax   = axes[1, col]
        data = load_sweep_p_sensitive(
            GLOB_SWEEP_PSENS[net], GLOB_LOG_PSENS[net], FIXED_TRANSFORM_P_STRATEGY)
        plot_panel(ax, data, x_col="p_sensitive",
                   xlabel=r"$p_{\rm sens}$ (minority proportion)")
        ax.set_ylabel("Metric value", fontsize=9)

    # ── Row labels ────────────────────────────────────────────────────────────
    for row, label in enumerate(ROW_LABELS):
        axes[row, 0].annotate(
            label, xy=(0, 0.5), xytext=(-axes[row, 0].yaxis.labelpad - 28, 0),
            xycoords="axes fraction", textcoords="offset points",
            fontsize=8.5, ha="right", va="center", rotation=90,
        )



    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    print(f"Figure saved to {OUTPUT_FILE}")
    return fig


if __name__ == "__main__":
    build_figure()