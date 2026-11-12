#!/usr/bin/env python3
"""gen_hybrid_table.py -- lay out the gsplit-hybrid ablation as a PER-GRAPH table.

For every Section-1 ablation cell (graph x model x interconnect x (L,h)) we compare the
baseline full-split gsplit (system=split, k=0) against the two hybrid families and report,
per cell, the SINGLE FASTEST variant found in each family:

  baseline gsplit : system=split, k=0                       (from the sec1 rows)
  best DP         : min over system=split, k = 1 .. L-1     (k leading data-parallel layers)
  best pipe       : system=gsplit_pipepar, k = 0 ONLY (pure, fully non-redundant pipeline).
                    k>0 pipeline rows mix pipeline with the DP hybrid, so they are NOT
                    candidates for this column; they stay in hybrid_ablation.csv.
  Delta           : improvement of the fastest hybrid over baseline gsplit
                    (POSITIVE = hybrid faster), as per-epoch (training).

"Best" here is purely mechanical (the minimum measured per-epoch time) -- it is a readout
of the data, NOT an analytical claim. Every individual variant (all k) is also written to
results/hybrid_ablation.csv so nothing is hidden and missing points can be spotted by hand.

Times are per-epoch seconds with training-only (forward+backward) in parentheses. A family
whose every variant hit a sentinel is shown as that sentinel (e.g. OOM) -- the sentinel is
kept, never dropped.

Layout (matches sec1 per-graph style): one table per graph, Model multirow, the (L,h)
ablation cells with L/h multirow over the two interconnect (PCIe / NVLink) rows.

Preliminary observations (user-supplied, flagged "to verify") are emitted as notes under
each graph's table; edit OBSERVATIONS below.

Outputs:
  results/hybrid_ablation.csv        long format, EVERY variant (one row per graph,model,L,h,hw,system,k)
  tables/hybrid_<graph>.tex          per-graph LaTeX table (only graphs that have hybrid rows)

Usage: python3 gen_hybrid_table.py [--csv results/sweep_all.csv]
"""
import argparse
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CSV = os.path.join(HERE, "results", "sweep_all.csv")
ABLATION_CELLS = [(3, 256), (3, 1024), (3, 4096), (2, 1024), (4, 1024)]  # sec1 order
HW_ORDER = ["pcie_emu", "nvlink"]
HW_LABEL = {"pcie_emu": "PCIe (emul.)", "nvlink": "NVLink"}
SENTINELS = {"timeout": "t/o", "oom": "OOM"}

# gsplit_pipepar rows logged at or before this timestamp were run with an UNHALVED
# micro-batch: each of the two micro-batches held a full batch_size (2x the seeds per
# optimizer step of the split baseline, ~2x live memory, half the optimizer steps).
# pipeline_split_trainer.py now uses batch_size//2 per micro-batch. These legacy rows are
# KEPT and shown flagged with LEGACY_MARK; in a cell that has any re-run (post-fix) pipeline
# row, the best pipe is taken from the re-run rows only (legacy rows stay in the CSVs).
LEGACY_PIPE_CUTOFF = "2026-11-04 09:20:34.862848"   # last row of the 2026-11-04 sweep
LEGACY_MARK = r"$^{\dagger}$"

# Post-fix (2026-11-12) pipeline re-run: run_settings.py records ANY nonzero exit as an "oom"
# sentinel, but the run log (results/hybrid/pipefix_run_log.txt) shows every failed re-run cell
# died with "CUDA error: an illegal memory access" and the log has 0 out-of-memory messages.
# So post-fix gsplit_pipepar "oom" rows are displayed as a crash, not OOM. The raw rows in
# sweep_all.csv keep the logged "oom" value; only the label shown here changes.
CRASH_LABEL = "crash"
CRASH_MARK = r"$^{\ddagger}$"

# Preliminary observations to print under each graph's table. User will verify these.
# Keyed by graph name; value is a list of (short) LaTeX lines.
OBSERVATIONS = {
    "papers100M": [
        r"\textbf{DP hybrid (\texttt{split}, $k>0$) pays off most consistently} --- it helps "
        r"both SAGE and GAT, and the speedup \emph{grows with} $k$: the largest interior "
        r"$k$ ($k{=}L{-}1$) is the fastest hybrid in 13 of 18 cells. Largest gains are on "
        r"\textbf{PCIe + deep/wide} (SAGE PCIe $L{=}4$: $+34\%$).",
        r"\textbf{Pipeline column = 2026-11-12 re-run} (pure pipeline, two half-size "
        r"micro-batches). All 10 GAT pipeline cells crashed with a CUDA illegal memory access "
        r"(crash$^{\ddagger}$; under investigation). Legacy unhalved rows stay in the CSVs.",
        r"\textbf{Gains track communication pressure} --- PCIe $>$ NVLink and "
        r"deeper/wider $>$ shallow, i.e. the hybrids help most exactly where the "
        r"shuffle/all-to-all dominates. (Preliminary; to verify.)",
    ],
    "products": [
        r"\textbf{Same pattern as papers100M, smaller graph.} A hybrid beats baseline "
        r"gsplit in 18 of 20 cells; the DP hybrid is the fastest in 15 of them and its "
        r"gain again grows with $k$, peaking on \textbf{PCIe + deep/wide} (SAGE PCIe "
        r"$L{=}3,h{=}4096$: $+39\%$; $L{=}4$: $+33\%$).",
        r"\textbf{The two non-wins are small GAT on NVLink} ($L{=}2$ and $L{=}4,h{=}1024$), "
        r"where the cheap NVLink all-to-all leaves little shuffle for a hybrid to recover "
        r"and baseline split is already near-best --- consistent with the "
        r"interconnect argument of \S\ref{sec:hybrid-tradeoff}.",
        r"\textbf{Pipeline column = 2026-11-12 re-run} (pure pipeline, two half-size "
        r"micro-batches). 6 of 10 GAT pipeline cells crashed with a CUDA illegal memory "
        r"access (crash$^{\ddagger}$; under investigation). Legacy unhalved rows stay in the CSVs. "
        r"(Win counts in the items above predate the re-run; to re-verify.)",
    ],
}


def depth_of(s):
    s = str(s).strip().strip("[]")
    return len([x for x in s.replace(" ", "").split(",") if x != ""]) if s else 0


def num(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in SENTINELS or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def cache_gb(s):
    s = str(s).strip().upper().rstrip("B").rstrip("G")
    try:
        return float(s)
    except ValueError:
        return 0.0


def is_zero_cache(r):
    if cache_gb(r.get("cache_size", "0")) != 0.0:
        return False
    try:
        if float(r.get("cache_frac", -1) or -1) > 0:
            return False
    except ValueError:
        pass
    return True


def hybrid_label(system, k):
    if system == "split":
        return "gsplit" if k == 0 else f"DP k{k}"
    if system == "gsplit_pipepar":
        return "pipe" if k == 0 else f"pipe+DP k{k}"
    return f"{system} k{k}"


def load_rows(csv_path):
    """Return list of per-variant dicts for split/gsplit_pipepar at 0% cache in the grid."""
    out = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("system") not in ("split", "gsplit_pipepar"):
                continue
            if not is_zero_cache(r):
                continue
            L = depth_of(r.get("fanouts"))
            try:
                h = int(r.get("hid_size"))
            except (TypeError, ValueError):
                continue
            if (L, h) not in ABLATION_CELLS:
                continue
            try:
                k = int(r.get("num_redundant_layers", 0) or 0)
            except ValueError:
                k = 0
            dur = r.get("duration (s)")
            fwd, bwd = num(r.get("forward (s)")), num(r.get("backward (s)"))
            train = (fwd + bwd) if (fwd is not None and bwd is not None) else None
            sent = next((v for kk, v in SENTINELS.items()
                         if str(dur).strip().lower() == kk), None)
            legacy = (r.get("system") == "gsplit_pipepar"
                      and str(r.get("timestamp", "")) <= LEGACY_PIPE_CUTOFF)
            if (r.get("system") == "gsplit_pipepar" and not legacy
                    and sent == SENTINELS["oom"]):
                sent = CRASH_LABEL + CRASH_MARK
            out.append(dict(graph=r.get("graph_name"), model=r.get("model"),
                            hw=r.get("hardware"), L=L, h=h, system=r.get("system"), k=k,
                            hybrid=hybrid_label(r.get("system"), k),
                            epoch=num(dur), train=train, sentinel=sent, legacy=legacy))
    return out


def et(epoch, train):
    """format 'epoch (train)'."""
    e = f"{epoch:.2f}" if epoch is not None else "--"
    t = f"{train:.2f}" if train is not None else "--"
    return f"{e} ({t})"


def pct(new, base):
    return 100.0 * (base - new) / base if (new is not None and base) else None


def best_of(variants):
    """pick the fastest (min epoch) non-sentinel variant; else report the sentinel."""
    ok = [v for v in variants if v["epoch"] is not None]
    if ok:
        return min(ok, key=lambda v: v["epoch"]), None
    if variants:
        # all sentinel -> report the (first) sentinel label
        s = next((v["sentinel"] for v in variants if v["sentinel"]), None)
        return None, (s or "--")
    return None, None  # family absent entirely


# ------------------------------------------------------------------ CSV (all variants)
def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graph", "model", "L", "h", "interconnect", "system",
                    "k", "hybrid", "epoch_s", "train_s", "status", "note"])
        for r in sorted(rows, key=lambda x: (x["graph"], x["model"], x["L"], x["h"],
                                             x["hw"], x["system"], x["k"])):
            w.writerow([r["graph"], r["model"], r["L"], r["h"], r["hw"], r["system"],
                        r["k"], r["hybrid"],
                        "" if r["epoch"] is None else f"{r['epoch']:.3f}",
                        "" if r["train"] is None else f"{r['train']:.3f}",
                        (CRASH_LABEL if (r["sentinel"] or "").startswith(CRASH_LABEL)
                         else r["sentinel"]) or "ok",
                        "unhalved_microbatch" if r["legacy"] else
                        ("illegal_memory_access (logged as oom)"
                         if (r["sentinel"] or "").startswith(CRASH_LABEL) else "")])
    print(f"wrote {path}")


# ------------------------------------------------------------------ per-graph LaTeX
def dp_cell(rows_bykhw):
    """rows for one (model,L,h,hw) family 'split k>=1' -> cell text with winning k."""
    v, sent = best_of(rows_bykhw)
    if v is not None:
        return f"{et(v['epoch'], v['train'])}~{{\\footnotesize $k{v['k']}$}}", v
    return (sent or "--"), None


def pipe_cell(rows_bykhw):
    """Prefer re-run (post-fix) pipeline rows; fall back to legacy rows, flagged."""
    fixed = [v for v in rows_bykhw if not v["legacy"]]
    use = fixed if fixed else rows_bykhw
    mark = LEGACY_MARK if (use and not fixed) else ""
    v, sent = best_of(use)
    if v is not None:
        return f"{et(v['epoch'], v['train'])}{mark}", v
    return ((sent or "--") + mark if sent else "--"), None


def write_graph_tex(graph, rows, path):
    # index: model -> (L,h) -> hw -> system -> [variants]
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in rows:
        idx[r["model"]][(r["L"], r["h"])][r["hw"]][r["system"]].append(r)

    models = sorted(idx)  # gat, sage
    L = [f"% AUTO-GENERATED by defense_latex/gen_hybrid_table.py -- do not edit by hand.",
         r"\begin{table}[t]", r"\centering",
         (r"\caption{\textbf{__GRAPH__ --- gsplit hybrids.} Per-epoch time (s), "
          r"\emph{training-only} (fwd+bwd) in parentheses. \textbf{best DP}: fastest "
          r"\texttt{split} with $k$ leading data-parallel layers ($k{=}1..L{-}1$); "
          r"\textbf{best pipe}: pure \texttt{gsplit\_pipepar} pipeline ($k{=}0$, no "
          r"redundant layers); the winning DP $k$ is shown. $\Delta$ is the fastest hybrid's improvement over baseline gsplit "
          r"(\textbf{positive $=$ hybrid faster}), per-epoch $(\text{training})$. "
          r"\texttt{OOM} $=$ every variant in that column ran out of memory (kept in "
          r"place). $^{\dagger}$ $=$ pipeline run with an \emph{unhalved} micro-batch (each "
          r"of the two micro-batches held a full batch, i.e.\ $2\times$ the seeds per "
          r"optimizer step of baseline gsplit): not comparable; shown only where no re-run row exists. "
          r"\texttt{crash}$^{\ddagger}$ $=$ re-run pipeline cell died with a CUDA illegal "
          r"memory access (logged by the runner as \texttt{oom}; the run log has no "
          r"out-of-memory error). $0\%$ cache, 3 epochs, random features; PCIe emulated via "
          r"\texttt{NCCL\_P2P\_DISABLE=1}.}").replace("__GRAPH__", graph),
         r"\label{tab:hybrid-" + graph.lower() + "}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{|l|c|c|c|r|r|r|r|}",
         r"\hline",
         r"\textbf{Model} & $\mathbf{L}$ & $\mathbf{h}$ & \textbf{Interconnect} & "
         r"\textbf{baseline gsplit} & \textbf{best DP} & \textbf{best pipe} & $\mathbf{\Delta}$ \\",
         r"\hline"]

    for mi, model in enumerate(models):
        cells = [c for c in ABLATION_CELLS if c in idx[model]]
        nrows = sum(len(HW_ORDER) for _ in cells)
        first_model = True
        for (Lv, hv) in cells:
            hw_present = [hw for hw in HW_ORDER if hw in idx[model][(Lv, hv)]]
            for hi, hw in enumerate(hw_present):
                fam = idx[model][(Lv, hv)][hw]
                # baseline gsplit = split k=0
                base_list = [v for v in fam.get("split", []) if v["k"] == 0]
                base = base_list[0] if base_list else None
                dp_list = [v for v in fam.get("split", []) if v["k"] >= 1]
                # pure pipeline only: k>0 would conflate pipeline with the DP hybrid
                pipe_list = [v for v in fam.get("gsplit_pipepar", []) if v["k"] == 0]
                dptxt, dpv = dp_cell(dp_list)
                pptxt, ppv = pipe_cell(pipe_list)
                basetxt = et(base["epoch"], base["train"]) if base else "--"
                # delta: fastest hybrid (min epoch among dpv,ppv) vs baseline
                cands = [v for v in (dpv, ppv) if v is not None]
                if base and cands:
                    win = min(cands, key=lambda v: v["epoch"])
                    de = pct(win["epoch"], base["epoch"])
                    dt = pct(win["train"], base["train"]) if (win["train"] and base["train"]) else None
                    tag = "DP" if win is dpv else "pipe"
                    dtxt = (f"$+{de:.1f}\\%$" if de >= 0 else f"${de:.1f}\\%$")
                    if dt is not None:
                        dtxt += f"\\,({'+' if dt>=0 else ''}{dt:.0f}\\%)"
                    if tag == "pipe" and win["legacy"]:
                        tag += LEGACY_MARK
                    dtxt += f"~{{\\footnotesize {tag}}}"
                else:
                    dtxt = "--"

                # row prefix: model multirow on first row; L,h multirow on first hw row
                pre = ""
                if first_model:
                    pre += f"\\multirow{{{nrows}}}{{*}}{{\\textbf{{{model.upper()}}}}} "
                    first_model = False
                pre += "& "
                if hi == 0:
                    pre += f"\\multirow{{{len(hw_present)}}}{{*}}{{${Lv}$}} & " \
                           f"\\multirow{{{len(hw_present)}}}{{*}}{{${hv}$}} & "
                else:
                    pre += "& & "
                L.append(f"{pre}{HW_LABEL[hw]} & {basetxt} & {dptxt} & {pptxt} & {dtxt} \\\\")
            L.append(r"\cline{2-8}")
        L[-1] = r"\hline"  # close model block with full rule
    L += [r"\end{tabular}", r"}"]

    # observations note
    obs = OBSERVATIONS.get(graph)
    if obs:
        L += [r"\par\vspace{2pt}\footnotesize\textit{Preliminary observations (to verify):}",
              r"\begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}"]
        L += [f"  \\item {o}" for o in obs]
        L += [r"\end{itemize}"]
    L += [r"\end{table}"]

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    rows = load_rows(args.csv)
    print(f"loaded {len(rows)} split/gsplit_pipepar rows from {args.csv}")
    write_csv(rows, os.path.join(args.outdir, "results", "hybrid_ablation.csv"))
    graphs = sorted({r["graph"] for r in rows
                     if any(not (x["system"] == "split" and x["k"] == 0)
                            for x in rows if x["graph"] == r["graph"])})
    for g in graphs:
        grows = [r for r in rows if r["graph"] == g]
        has_hyb = any(not (r["system"] == "split" and r["k"] == 0) for r in grows)
        if not has_hyb:
            print(f"  skip {g}: no hybrid rows yet")
            continue
        write_graph_tex(g, grows, os.path.join(args.outdir, "tables", f"hybrid_{g}.tex"))


if __name__ == "__main__":
    main()
