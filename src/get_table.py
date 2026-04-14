# generate_benchmark_table.py
# Preamble needs: \usepackage{booktabs}, \usepackage{caption}, \usepackage{graphicx}

import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

LOG_FILES = {
    "facebook": "../logs/benchmark_facebook.txt",
}

CSV_FILES = {
    "facebook": "../logs/benchmark_facebook.csv",
}

NETWORK_ORDER = ["facebook"]

NET_LABELS = {
    "facebook": "Facebook",
}

SFAIRSC_MAP = {
    "facebook": ("sFairSC-k13", "sFairSC-k15"),
}

OUTPUT_FILE = "benchmark_table.tex"

# ── Row definitions ───────────────────────────────────────────────────────────
# (latex_label, algorithm, alpha, sfairsc_role, strategy)

ROWS = [
    # --- sFairSC ---
    (r"sFairSC, $k=\#c_{\text{M}(\alpha=0.5)}$", None,                   None,  "mouflon", None),
    (r"sFairSC, $k=\#c_{\text{Louvain}}$",        None,                   None,  "louvain", None),
    # --- FAL ---
    (r"FAL (Red Mod.)",    "FAL-RedModularity",   None,  None, None),
    (r"FAL (Blue Mod.)",   "FAL-BlueModularity",  None,  None, None),
    (r"FAL (Diversity)",   "FAL-Diversity",        None,  None, None),
    (r"FAL (L-Red Mod.)",  "FAL-LRedModularity",  None,  None, None),
    (r"FAL (L-Blue Mod.)", "FAL-LBlueModularity", None,  None, None),
    (r"FAL (L-Diversity)", "FAL-LDiversity",       None,  None, None),
    # --- Fair-mod (MOUFLON-base) ---
    (r"Fair-mod ($\alpha=0.25$)", "MOUFLON-base",  0.25, None, None),
    (r"Fair-mod ($\alpha=0.5$)",  "MOUFLON-base",  0.50, None, None),
    (r"Fair-mod ($\alpha=0.75$)", "MOUFLON-base",  0.75, None, None),
    # --- MOUFLON (MOUFLON-hybrid) ---
    (r"MOUFLON ($\alpha=0.25$)", "MOUFLON-hybrid", 0.25, None, None),
    (r"MOUFLON ($\alpha=0.5$)",  "MOUFLON-hybrid", 0.50, None, None),
    (r"MOUFLON ($\alpha=0.75$)", "MOUFLON-hybrid", 0.75, None, None),
    # --- Louvain ---
    (r"Louvain", "Louvain", None, None, None),
    # --- Mutex Watershed ---
    (r"MutexWatershed (Standard)",             "MutexWatershed-Standard",  None, None, "connect_all=False"),
    (r"MutexWatershed (ConnectAll)",           "MutexWatershed-ConnectAll", None, None, "connect_all=True"),
    (r"MutexWatershed-T ($p_{\rm sc}=0.0$)",  "MutexWatershed-Transform",  None, None, "same_color_p=0.0"),
    (r"MutexWatershed-T ($p_{\rm sc}=0.25$)", "MutexWatershed-Transform",  None, None, "same_color_p=0.25"),
    (r"MutexWatershed-T ($p_{\rm sc}=0.5$)",  "MutexWatershed-Transform",  None, None, "same_color_p=0.5"),
    (r"MutexWatershed-T ($p_{\rm sc}=0.75$)", "MutexWatershed-Transform",  None, None, "same_color_p=0.75"),
    (r"MutexWatershed-T ($p_{\rm sc}=1.0$)",  "MutexWatershed-Transform",  None, None, "same_color_p=1.0"),
]

MIDRULE_AFTER = {
    r"sFairSC, $k=\#c_{\text{Louvain}}$",
    r"FAL (L-Diversity)",
    r"Fair-mod ($\alpha=0.75$)",
    r"MOUFLON ($\alpha=0.75$)",
    r"Louvain",
}

# ── Parsing ───────────────────────────────────────────────────────────────────

ALG_RE  = re.compile(r"\u25b6 (\S+)\s+alpha=(\S+)\s+strategy=(\S+)")
COMM_RE = re.compile(r"Computing metrics over (\d+) communities")


def parse_logs(log_files):
    comm_rows = []
    for network, path in log_files.items():
        p = Path(path)
        if not p.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        current_key = None
        records = {}
        for line in lines:
            m = ALG_RE.search(line)
            if m:
                key = (m.group(1), m.group(2), m.group(3))
                current_key = key
                records.setdefault(key, [])
                continue
            m = COMM_RE.search(line)
            if m and current_key is not None:
                records[current_key].append(int(m.group(1)))
        for (alg, alpha, strategy), counts in records.items():
            if not counts:
                comm_rows.append(dict(network=network, algorithm=alg, alpha=alpha,
                                      strategy=strategy,
                                      n_runs=0, n_communities_mean=float("nan"),
                                      n_communities_std=float("nan")))
            else:
                comm_rows.append(dict(network=network, algorithm=alg, alpha=alpha,
                                      strategy=strategy,
                                      n_runs=len(counts),
                                      n_communities_mean=np.mean(counts),
                                      n_communities_std=np.std(counts, ddof=1) if len(counts) > 1 else 0.0))
    return pd.DataFrame(comm_rows)


def load_bench(csv_files):
    return {k: pd.read_csv(v) for k, v in csv_files.items() if Path(v).exists()}


# ── Lookup ────────────────────────────────────────────────────────────────────

def get_bench_row(bench, network, alg, alpha=None, strategy=None):
    df = bench.get(network)
    if df is None:
        return None
    mask = df["algorithm"] == alg
    if alpha is not None:
        mask &= df["alpha"].round(2) == round(alpha, 2)
    if strategy is not None:
        mask &= df["strategy"].astype(str) == str(strategy)
    rows = df[mask]
    return rows.iloc[0] if not rows.empty else None


def get_comm_row(comm, network, alg, alpha=None, strategy=None):
    if comm.empty:
        return None
    mask = (comm["network"] == network) & (comm["algorithm"] == alg)
    if alpha is not None:
        mask &= comm["alpha"].astype(str).str.startswith(str(alpha))
    if strategy is not None:
        mask &= comm["strategy"].astype(str) == str(strategy)
    rows = comm[mask]
    return rows.iloc[0] if not rows.empty else None


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt(mean, std, d=3):
    if pd.isna(mean) or pd.isna(std):
        return r"--"
    return f"{mean:.{d}f} ({std:.{d}f})"


def fmt_comm(mean, std):
    if pd.isna(mean) or pd.isna(std):
        return r"--"
    return f"{mean:.1f} ({std:.1f})"


# ── Sub-table builder ─────────────────────────────────────────────────────────
# Columns: Method | Mod. | Prop. fairness | Unfairness | #Comm | Time

def build_subtable(comm, bench, network, net_label, sfairsc_map, rows,
                   midrule_after, is_first):
    L = []
    if not is_first:
        L.append(r"\bigskip")
    L.append(r"\noindent\textbf{" + net_label + r"}")
    L.append(r"\vspace{2pt}")
    L.append(r"\resizebox{\linewidth}{!}{%")
    L.append(r"\begin{tabular}{lrrrrr}")
    L.append(r"\toprule")
    L.append(r"\textbf{Method} & \textbf{Modularity} & \textbf{Prop.\ fairness}"
             r" & \textbf{Unfairness} & \textbf{Num.\ comm.} & \textbf{Time (s)} \\")
    L.append(r"\midrule")

    for label, alg, alpha, sfairsc_role, strategy in rows:
        if sfairsc_role is not None:
            k_mouflon, k_louvain = sfairsc_map[network]
            actual_alg = k_mouflon if sfairsc_role == "mouflon" else k_louvain
            b = get_bench_row(bench, network, actual_alg)
            c = get_comm_row(comm, network, actual_alg)
        else:
            b = get_bench_row(bench, network, alg, alpha, strategy)
            c = get_comm_row(comm, network, alg,
                             str(alpha) if alpha is not None else None,
                             strategy)

        mod    = fmt(b["modularity_mean"],  b["modularity_std"])  if b is not None else r"--"
        fexp   = fmt(b["fexp_mean"],        b["fexp_std"])        if b is not None else r"--"
        unfair = fmt(b["unfairness_mean"],  b["unfairness_std"])  if b is not None else r"--"
        rt     = fmt(b["runtime_mean"],     b["runtime_std"])     if b is not None else r"--"
        ncomm  = (fmt_comm(c["n_communities_mean"], c["n_communities_std"])
                  if c is not None and not pd.isna(c["n_communities_mean"]) else r"--")

        L.append(f"{label} & {mod} & {fexp} & {unfair} & {ncomm} & {rt} \\\\")
        if label in midrule_after:
            L.append(r"\midrule")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    return L


CAPTION = (
    r"Results for the Facebook network. "
    r"Columns show modularity, proportional fairness, unfairness, "
    r"number of communities, and runtime. "
    r"Values reported as mean (SD) over 10 runs. "
    r"MutexWatershed-T denotes the Transform variant; "
    r"$p_{\rm sc}$ is the same-color edge attraction probability. "
    r"Dashes indicate runs skipped due to memory or timeout."
)
LABEL = "tab:benchmark_facebook"


def build_table(comm, bench, network_order, net_labels, sfairsc_map, rows, midrule_after):
    L = []
    L.append(r"\begin{table}[tp]")
    L.append(r"\centering")
    L.append(r"\caption{" + CAPTION + r"}")
    L.append(r"\label{" + LABEL + r"}")
    for idx, network in enumerate(network_order):
        sub = build_subtable(comm, bench, network, net_labels[network],
                             sfairsc_map, rows, midrule_after, is_first=(idx == 0))
        L.extend(sub)
    L.append(r"\end{table}")
    return "\n".join(L)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    comm  = parse_logs(LOG_FILES)
    bench = load_bench(CSV_FILES)
    tex   = build_table(comm, bench, NETWORK_ORDER, NET_LABELS,
                        SFAIRSC_MAP, ROWS, MIDRULE_AFTER)
    Path(OUTPUT_FILE).write_text(tex, encoding="utf-8")
    print(f"Written to {OUTPUT_FILE}")