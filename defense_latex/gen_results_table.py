#!/usr/bin/env python3
"""gen_results_table.py — turn the consolidated sweep CSV into a \\input-able
LaTeX table for the dissertation (eval plan Step 1a/1c/4).

Reads defense_latex/results/sweep_all.csv (schema = Config.header()+Profiler.header()),
normalizes to the canonical config superset, and emits a tabular where, within each
(graph, hardware, model, depth) cell, the fastest system row (min per-epoch time) is
highlighted in red via \\textcolor{red}.

Usage:
    python3 gen_results_table.py --csv results/sweep_all.csv --out tables/sweep_all.tex
    python3 gen_results_table.py --self-test      # write+render a synthetic CSV, no real data
"""
import argparse
import os
import sys
import tempfile

import pandas as pd

# per-epoch wall time: lower is better -> defines the winning system per cell.
PRIMARY_METRIC = "duration (s)"

# canonical config key (plan contract) <- raw CSV column, with fallbacks applied below
CONFIG_RENAME = {
    "graph_name": "graph",
    "num_redundant_layers": "k",
    "fanouts": "fanout",
}

METRIC_COLS = [
    ("duration (s)", "epoch(s)"),
    ("sampling (s)", "sample"),
    ("feature (s)", "feat"),
    ("forward (s)", "fwd"),
    ("backward (s)", "bwd"),
    ("allocated (MB)", "mem(MB)"),
    ("test accuracy %", "acc%"),
]


def _depth(fanout):
    try:
        s = str(fanout).strip().strip("[]")
        return len([x for x in s.replace(" ", "").split(",") if x != ""])
    except Exception:
        return -1


def _num(series):
    # coerce to numeric; "oom"/"" -> NaN so they never win a min()
    return pd.to_numeric(series, errors="coerce")


def _sys_label(row):
    """Human/LaTeX-safe system label that distinguishes k and cache_frac variants."""
    sysname = str(row.get("system", "?"))
    k = row.get("k", 0)
    frac = row.get("cache_frac", -1)
    try:
        k = int(float(k))
    except Exception:
        k = 0
    if sysname == "split" and k > 0:
        return f"split(k={k})"
    if sysname == "gsplit\\_pipepar" or sysname == "gsplit_pipepar":
        return "pipeline-split"
    try:
        if sysname == "quiver" and float(frac) > 0:
            return f"quiver({int(round(float(frac)*100))}\\%)"
    except Exception:
        pass
    return sysname


def _esc(x):
    return str(x).replace("_", "\\_").replace("%", "\\%")


def normalize(df):
    df = df.rename(columns=CONFIG_RENAME).copy()
    # hardware fallback: derive from machine_name suffix if the column is absent
    if "hardware" not in df.columns:
        if "machine_name" in df.columns:
            df["hardware"] = df["machine_name"].astype(str).apply(
                lambda m: "nvlink" if m.endswith("nvlink") else ("pcie_emu" if m.endswith("pcie") else "unknown"))
        else:
            df["hardware"] = "unknown"
    for col, default in (("k", 0), ("cache_frac", -1.0), ("cache_size", ""),
                         ("hid_size", -1), ("batch_size", -1), ("world_size", -1),
                         ("sample_mode", "uva")):
        if col not in df.columns:
            df[col] = default
    df["depth"] = df["fanout"].apply(_depth)
    # numeric view of the primary metric for winner detection; keep the raw column
    # intact so 'oom'/missing still render as text.
    df["__metric_num"] = _num(df[PRIMARY_METRIC])
    return df


def _slug(*parts):
    s = "-".join(str(p) for p in parts)
    for a, b in (("_", "-"), (" ", ""), ("%", ""), ("/", "-")):
        s = s.replace(a, b)
    return s.lower()


def _cell_rows(grp, win_idx):
    """The system rows for one (depth,hidden) cell, fastest first, winner in red."""
    out = []
    grp = grp.sort_values("__metric_num", na_position="last")
    for idx, row in grp.iterrows():
        syslabel = _sys_label(row)
        cache = _esc(str(row.get("cache_size", "")))
        try:
            frac = float(row.get("cache_frac", -1))
        except Exception:
            frac = -1.0
        if frac > 0:
            cache = f"{int(round(frac * 100))}\\%"   # already LaTeX-escaped
        cells = [syslabel, cache]
        for col, _ in METRIC_COLS:
            val = row.get(col, "")
            if isinstance(val, str) and val.strip().lower() in ("oom", "timeout"):
                cells.append(val.strip().upper() if val.strip().lower() == "oom" else "t/o")
                continue
            num = _num(pd.Series([val]))[0]
            cells.append("--" if pd.isna(num) else f"{num:.2f}")
        rowtex = " & ".join(cells) + " \\\\"
        if idx == win_idx:
            rowtex = "\\rowcolor{red!15} " + rowtex   # fastest system in this cell
        out.append(rowtex)
    return out


def emit_tex(df, base_label="tab:sweep_all"):
    """Emit ONE table float per (graph, hardware, model) setting, stacked vertically
    so each fits a page and the set page-breaks between tables. Cells are keyed by
    (depth, hidden); the fastest system per cell is highlighted."""
    df = normalize(df)
    cell_keys = ["graph", "hardware", "model", "depth", "hid_size"]
    winners = {}
    for key, grp in df.groupby(cell_keys, dropna=False):
        valid = grp.dropna(subset=["__metric_num"])
        winners[key] = valid["__metric_num"].idxmin() if len(valid) else None

    ncol = 2 + len(METRIC_COLS)                     # system, cache, metrics...
    colfmt = ("l l " + "r " * len(METRIC_COLS)).strip()
    head = ["system", "cache"] + [h for _, h in METRIC_COLS]

    out = ["% auto-generated by gen_results_table.py -- do not edit by hand",
           "% one table per (graph, hardware, model) setting, stacked vertically."]
    first = True
    for skey, sgrp in df.groupby(["graph", "hardware", "model"], dropna=False):
        graph, hardware, model = skey
        caption = ("\\textbf{%s, %s, %s}: per-epoch time (s) and breakdown by "
                   "depth$\\times$hidden; fastest system per cell in red. "
                   "\\texttt{t/o}=900\\,s cap, \\textsc{oom}=out of memory." % (
                       _esc(graph), _esc(hardware.replace("_emu", "")), _esc(model).upper()))
        label = "tab:sweep-" + _slug(graph, hardware, model)
        out.append("\\begin{table}[t]")
        out.append("\\centering\\small")
        out.append("\\begin{tabular}{" + colfmt + "}")
        out.append("\\toprule")
        out.append(" & ".join(_esc(h) for h in head) + " \\\\")
        out.append("\\midrule")
        for (depth, hid), cgrp in sgrp.groupby(["depth", "hid_size"], dropna=False):
            try:
                hid_s = str(int(float(hid)))
            except Exception:
                hid_s = _esc(str(hid))
            out.append("\\multicolumn{%d}{l}{\\textbf{$L{=}%s$, $h{=}%s$}} \\\\" % (
                ncol, _esc(depth), hid_s))
            out += _cell_rows(cgrp, winners.get((graph, hardware, model, depth, hid)))
            out.append("\\midrule")
        out[-1] = "\\bottomrule"
        out.append("\\end{tabular}")
        out.append("\\caption{%s}" % caption)
        if first:                                   # keep legacy \ref{tab:sweep_all} resolving
            out.append("\\label{%s}" % base_label)
            first = False
        out.append("\\label{%s}" % label)
        out.append("\\end{table}")
        out.append("")
    return "\n".join(out) + "\n"


def _synthetic_csv(path):
    rows = [
        # graph, hardware, model, fanout, system, k, cache_size, cache_frac, dur, samp, feat, fwd, bwd, mem, acc
        ("products", "nvlink", "sage", "15,15,15", "split", 0, "10G", -1, 1.20, 0.3, 0.2, 0.4, 0.3, 8000, 78.1),
        ("products", "nvlink", "sage", "15,15,15", "split", 1, "10G", -1, 1.05, 0.3, 0.2, 0.3, 0.25, 8200, 78.0),
        ("products", "nvlink", "sage", "15,15,15", "dgl", 0, "10G", -1, 0.98, 0.3, 0.2, 0.25, 0.23, 9000, 78.2),
        ("products", "nvlink", "sage", "15,15,15", "quiver", 0, "", 0.25, 1.40, 0.4, 0.3, 0.4, 0.3, 7000, 78.1),
        ("products", "pcie_emu", "sage", "15,15,15", "split", 0, "10G", -1, 2.80, 0.3, 0.2, 0.4, 0.3, 8000, 78.1),
        ("products", "pcie_emu", "sage", "15,15,15", "dgl", 0, "10G", -1, 1.60, 0.3, 0.2, 0.25, 0.23, 9000, 78.2),
        ("products", "pcie_emu", "sage", "15,15,15", "quiver", 0, "", 0.10, "oom", "oom", "oom", "oom", "oom", "oom", "oom"),
    ]
    cols = ["graph_name", "hardware", "model", "fanouts", "system",
            "num_redundant_layers", "cache_size", "cache_frac",
            "duration (s)", "sampling (s)", "feature (s)", "forward (s)",
            "backward (s)", "allocated (MB)", "test accuracy %"]
    assert all(len(r) == len(cols) for r in rows), "synthetic row/col length mismatch"
    recs = [dict(zip(cols, r)) for r in rows]
    df = pd.DataFrame(recs)
    df.to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.path.join(here, "results", "sweep_all.csv"))
    ap.add_argument("--out", default=os.path.join(here, "tables", "sweep_all.tex"))
    ap.add_argument("--caption", default="gsplit vs. baselines: per-epoch time by (graph, hardware, model). Fastest system per cell in red.")
    ap.add_argument("--self-test", action="store_true",
                    help="generate a synthetic CSV, render it, print the .tex, and exit")
    args = ap.parse_args()

    if args.self_test:
        tmp = os.path.join(tempfile.gettempdir(), "sweep_synth.csv")
        _synthetic_csv(tmp)
        df = pd.read_csv(tmp)
        tex = emit_tex(df)
        assert "\\begin{tabular}" in tex and "\\rowcolor{red!15}" in tex, "self-test: no highlighted winner"
        print(tex)
        print("# self-test OK: valid tabular with a red-highlighted winner per cell", file=sys.stderr)
        return

    if not os.path.isfile(args.csv):
        sys.exit(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)
    tex = emit_tex(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
