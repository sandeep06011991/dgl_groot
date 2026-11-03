#!/usr/bin/env python3
"""gen_sec1_tables.py -- generate the two Section-1 tables (one per graph) from the
consolidated sweep CSV.

Section 1 compares **gsplit (``split``, k=0) vs DGL, both at a 0% feature cache**.
Each table has GAT and SAGE horizontal sections; within each, a hidden sweep (at L=4)
and a depth sweep (at h=1024), with (L=4, h=1024) the shared pivot. Columns are
PCIe (emulated) and NVLink, each showing DGL, gsplit, and Delta. Every system cell is
``epoch (train)`` seconds where ``train = forward + backward`` (the compute that survives
a perfect feature cache). Delta is gsplit's improvement over DGL, ``+% (+%)`` for
``epoch (train)``, **positive = gsplit faster**.

Quiver rows are filtered OUT of the render (they stay in the CSV; see prompt_monday.md
"remove the mention of quiver"). Missing cells render as \\placeholder; oom/timeout
sentinels render as OOM / t/o with Delta ``--``.

Usage:
    python3 gen_sec1_tables.py                                  # default csv -> tables/
    python3 gen_sec1_tables.py --csv results/sweep_all.csv --outdir tables
    python3 gen_sec1_tables.py --self-test                      # synthetic CSV, no real data
"""
import argparse
import os
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CSV = os.path.join(HERE, "results", "sweep_all.csv")
DEFAULT_OUTDIR = os.path.join(HERE, "tables")

# --- grid (keep in sync with experiment/script/gen_settings.py) ---------------
GRAPHS = ["products", "papers100M"]
MODELS = ["sage", "gat"]
HID_SWEEP = [256, 1024, 4096]
DEPTH_SWEEP = [2, 4, 6]
DEPTH_ANCHOR = 4       # hidden sweep runs at this depth
HID_ANCHOR = 1024      # depth sweep runs at this hidden
HARDWARES = ["pcie_emu", "nvlink"]   # table column order: PCIe (emul.) then NVLink

GRAPH_LABEL = {"products": ("products", "tab:sec1-products"),
               "papers100M": ("papers100M", "tab:sec1-papers")}
SENTINELS = {"oom": "OOM", "timeout": "t/o"}


def _depth_from_fanouts(s):
    s = str(s).strip().strip("[]")
    return len([x for x in s.replace(" ", "").split(",") if x != ""]) if s else 0


def _cache_gb(s):
    s = str(s).strip().upper().rstrip("B").rstrip("G")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _num(x):
    """float, or None for sentinel/blank."""
    if isinstance(x, str) and x.strip().lower() in SENTINELS:
        return None
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def load_lookup(csv_path):
    """(graph, hardware, model, depth, hid, system) -> dict(epoch, train, sentinel).

    Keeps only 0%-cache rows of dgl/split (the Section-1 condition); on duplicates the
    last-written row wins (CSV is append-ordered)."""
    lut = {}
    if not os.path.isfile(csv_path):
        return lut
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        system = str(r.get("system"))
        if system not in ("dgl", "split"):
            continue
        if _cache_gb(r.get("cache_size", "0")) != 0.0:
            continue
        try:
            if float(r.get("cache_frac", -1.0)) > 0:
                continue
        except (TypeError, ValueError):
            pass
        key = (str(r.get("graph_name")), str(r.get("hardware")), str(r.get("model")),
               _depth_from_fanouts(r.get("fanouts")), int(r.get("hid_size")), system)
        dur_raw = r.get("duration (s)")
        sentinel = None
        if isinstance(dur_raw, str) and dur_raw.strip().lower() in SENTINELS:
            sentinel = dur_raw.strip().lower()
        epoch = _num(dur_raw)
        fwd, bwd = _num(r.get("forward (s)")), _num(r.get("backward (s)"))
        train = (fwd + bwd) if (fwd is not None and bwd is not None) else None
        lut[key] = {"epoch": epoch, "train": train, "sentinel": sentinel}
    return lut


def _cell(lut, graph, hw, model, depth, hid, system):
    """LaTeX for one system cell: 'epoch (train)', OOM/t/o, or \\placeholder."""
    rec = lut.get((graph, hw, model, depth, hid, system))
    if rec is None:
        return r"\placeholder", None, None
    if rec["sentinel"]:
        return r"\textsc{%s}" % SENTINELS[rec["sentinel"]], None, None
    ep, tr = rec["epoch"], rec["train"]
    if ep is None:
        return r"\placeholder", None, None
    tr_s = f"{tr:.2f}" if tr is not None else "--"
    return f"{ep:.2f} ({tr_s})", ep, tr


def _delta(dgl_v, split_v):
    """gsplit improvement over DGL in %, positive = gsplit faster; None if not computable."""
    if dgl_v is None or split_v is None or dgl_v == 0:
        return None
    return 100.0 * (dgl_v - split_v) / dgl_v


def _delta_cell(dgl_ep, split_ep, dgl_tr, split_tr):
    de, dt = _delta(dgl_ep, split_ep), _delta(dgl_tr, split_tr)
    if de is None:
        return "--"
    dt_s = f"{dt:+.0f}\\%" if dt is not None else "--"
    return f"${de:+.0f}\\%\\,({dt_s})$"


def _row(lut, graph, model, depth, hid):
    """One tabular row: config label + 6 cells (PCIe DGL/gsplit/Delta, NVLink ...)."""
    if depth == DEPTH_ANCHOR and hid != HID_ANCHOR:
        label = f"$h{{=}}{hid}$"
    elif hid == HID_ANCHOR and depth != DEPTH_ANCHOR:
        label = f"$L{{=}}{depth}$"
    else:  # the shared pivot
        label = f"$L{{=}}{depth},h{{=}}{hid}$"
    cells = [label]
    for hw in HARDWARES:
        dgl_s, dgl_ep, dgl_tr = _cell(lut, graph, hw, model, depth, hid, "dgl")
        spl_s, spl_ep, spl_tr = _cell(lut, graph, hw, model, depth, hid, "split")
        cells += [dgl_s, spl_s, _delta_cell(dgl_ep, spl_ep, dgl_tr, spl_tr)]
    return " & ".join(cells) + r" \\"


def emit_table(lut, graph):
    name, label = GRAPH_LABEL[graph]
    L = []
    L.append(r"% AUTO-GENERATED by defense_latex/gen_sec1_tables.py -- do not edit by hand.")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{\textbf{%s.} Per-epoch time (s); \emph{training-only} time" % name)
    L.append(r"(forward+backward) in parentheses --- the compute that survives a perfect feature")
    L.append(r"cache. $\Delta$ is gsplit's improvement over DGL (\textbf{positive $=$ gsplit faster}),")
    L.append(r"shown as per-epoch $(\text{training})$. gsplit runs with a $0\%$ feature cache; PCIe is")
    L.append(r"emulated via \texttt{NCCL\_P2P\_DISABLE=1}. Features are random (perf sweep).}")
    L.append(r"\label{%s}" % label)
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\begin{tabular}{@{}l rrr @{\hskip 10pt} rrr@{}}")
    L.append(r"\toprule")
    L.append(r" & \multicolumn{3}{c}{\textbf{PCIe (emul.)}} & \multicolumn{3}{c}{\textbf{NVLink}} \\")
    L.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    L.append(r"Config & DGL & gsplit & $\Delta$ & DGL & gsplit & $\Delta$ \\")
    L.append(r"\midrule")
    for mi, model in enumerate(MODELS):
        up = model.upper()
        L.append(r"\multicolumn{7}{@{}l}{\textbf{%s} --- hidden sweep ($L{=}%d$)}\\" % (up, DEPTH_ANCHOR))
        for hid in HID_SWEEP:
            L.append(_row(lut, graph, model, DEPTH_ANCHOR, hid))
        L.append(r"\multicolumn{7}{@{}l}{\textbf{%s} --- depth sweep ($h{=}%d$)}\\" % (up, HID_ANCHOR))
        for depth in DEPTH_SWEEP:
            L.append(_row(lut, graph, model, depth, HID_ANCHOR))
        if mi == 0:
            L.append(r"\midrule")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def write_tables(csv_path, outdir):
    lut = load_lookup(csv_path)
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for graph in GRAPHS:
        slug = "products" if graph == "products" else "papers100M"
        path = os.path.join(outdir, f"sec1_{slug}.tex")
        with open(path, "w") as f:
            f.write(emit_table(lut, graph))
        out[graph] = path
    return out


def _synthetic_csv(path):
    cols = ["graph_name", "hardware", "system", "model", "fanouts", "hid_size",
            "cache_size", "cache_frac", "duration (s)", "forward (s)", "backward (s)"]
    rows = []
    for graph in GRAPHS:
        for hw in HARDWARES:
            for model in MODELS:
                for (depth, hid) in [(DEPTH_ANCHOR, h) for h in HID_SWEEP] + \
                                    [(d, HID_ANCHOR) for d in DEPTH_SWEEP]:
                    fan = "[" + ", ".join(["15"] * depth) + "]"
                    base = 0.5 + 0.0003 * hid + 0.1 * depth
                    for system, mult in (("dgl", 1.0), ("split", 0.9)):
                        rows.append([graph, hw, system, model, fan, hid, "0G", -1.0,
                                     round(base * mult, 3), round(base * mult * 0.4, 3),
                                     round(base * mult * 0.2, 3)])
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="generate the two Section-1 tables")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--self-test", action="store_true", help="synthetic CSV -> tables, then print")
    args = ap.parse_args()

    if args.self_test:
        d = tempfile.mkdtemp(prefix="sec1_selftest_")
        csv = os.path.join(d, "synthetic.csv")
        _synthetic_csv(csv)
        out = write_tables(csv, d)
        for g, p in out.items():
            print(f"--- {g}: {p} ---")
            print(open(p).read())
        return

    out = write_tables(args.csv, args.outdir)
    for g, p in out.items():
        print(f"wrote {g:11s} -> {p}")


if __name__ == "__main__":
    main()
