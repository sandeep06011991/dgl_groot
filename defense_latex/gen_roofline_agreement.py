#!/usr/bin/env python3
"""Roofline pick vs measured winner, for Sec. 5 (tab:final).

For every counts CSV (experiment/roofline/sample_counts.py), every hidden size measured
in results/sweep_all.csv, and both interconnects:
  modelled training time per strategy, summed over layers (same counts and rates as
  gen_roofline_points.py):
      dgl          sum_l T_cmp
      split k      sum_l (T_cmp + T_net)            gsplit (k=0) and hybrids, serialised
      pipeline k   2 * sum_l max(T_cmp, T_net)      one micro-batch of B/2, two identical halves
  roofline pick  = fastest modelled strategy among those MEASURED for that setting
  measured best  = fastest measured forward+backward time among the same strategies
  regret         = t_measured(pick) / t_measured(best) - 1
Two candidate sets are evaluated: without the pipeline (DGL, gsplit, hybrids; the table rows)
and with it (summary line). Under (R1) the pipeline overlaps perfectly, which flatters it.
Mismatches are kept: the roofline is an approximate model.

Outputs: results/roofline/agreement.csv, tables/roofline_agreement.tex
Light CPU work; safe to run locally.
"""
import glob
import os

import numpy as np
import pandas as pd

import gen_roofline_points as rp

HERE = rp.HERE
COUNTS_GLOB = os.path.join(HERE, "results", "roofline", "counts", "*.csv")
# gsplit_pipepar rows at or before this timestamp used unhalved micro-batches (see
# gen_hybrid_table.py); they are not comparable and are left out here.
LEGACY_PIPE_CUTOFF = "2026-11-04 09:20:34.862848"
LABEL = {"dgl": "DGL", "split": "split", "pipeline": "pipe"}

# GAT calibration: its shuffle moves narrow z-rows whose small-message all-to-all runs below the
# measured peak bandwidth, so we scale GAT's modelled shuffle time. Calibrated on this sweep; set to
# 1.0 to recover the uncalibrated baseline (results/roofline/agreement_cn1_baseline.csv).
GAT_NET_SCALE = 2.0


def strategy_name(system, k, L):
    if system == "dgl":
        return "DGL"
    if system == "split":
        return "gsplit" if k == 0 else f"hybrid k={k}"
    return "pipeline" if k == 0 else f"pipe k={k}"


def measured(sweep, r0, L, h, ic, column="train"):
    """Measured min time per (system, k) for one setting: column="train" is forward +
    backward (what the roofline models), column="epoch" is the whole epoch."""
    fan = "[" + ", ".join(str(r0.fanouts).split(",")) + "]"
    s = sweep[(sweep.graph_name == r0.graph_name) & (sweep.model == r0.model)
              & (sweep.fanouts == fan) & (sweep.hid_size == h) & (sweep.hardware == ic)
              & (sweep.batch_size == r0.batch_size) & (sweep.world_size == r0.world_size)
              & (sweep.partition_type == r0.partition_type)]
    s = s[~((s.system == "gsplit_pipepar") & (s.timestamp.astype(str) <= LEGACY_PIPE_CUTOFF))]
    if column == "train":
        t = (pd.to_numeric(s["forward (s)"], errors="coerce")
             + pd.to_numeric(s["backward (s)"], errors="coerce"))
    else:
        t = pd.to_numeric(s["duration (s)"], errors="coerce")
    s = s.assign(t=t).dropna(subset=["t"])
    out = {}
    for row in s.itertuples():
        k = int(row.num_redundant_layers)
        key = {"dgl": ("dgl", L), "split": ("split", k),
               "gsplit_pipepar": ("pipeline", k)}.get(row.system)
        if key is not None:
            out[key] = min(out.get(key, np.inf), row.t)
    return out


def modelled(counts, model, h, rates, ic, L, f_in):
    """Modelled training time per (system, k), summed over layers (and minibatches)."""
    t = rp.busiest_gpu_counts(counts, model, h, rates, ic, L, f_in)
    out = {}
    net_scale = GAT_NET_SCALE if model == "gat" else 1.0
    for (system, k), g in t.groupby(["system", "k"]):
        Tc, Tn = g.T_cmp, g.T_net * net_scale
        if system == "pipeline":
            out[(system, k)] = 2 * np.maximum(Tc, Tn).sum()
        else:
            out[(system, k)] = (Tc + Tn).sum()
    return out


def main():
    sweep = pd.read_csv(os.path.join(HERE, "results", "sweep_all.csv"))
    rows = []
    files = sorted(glob.glob(COUNTS_GLOB))
    if not files:
        raise SystemExit(f"no counts CSVs under {COUNTS_GLOB}")
    for path in files:
        counts = pd.read_csv(path)
        r0 = counts.iloc[0]
        model, L = r0.model, int(counts.layer.max()) + 1
        f_in = rp.FEAT_DIM[r0.graph_name]
        fan = "[" + ", ".join(str(r0.fanouts).split(",")) + "]"
        hs = sorted(sweep[(sweep.graph_name == r0.graph_name) & (sweep.model == model)
                          & (sweep.fanouts == fan)].hid_size.unique())
        for h in hs:
            try:
                rates = rp.load_rates(int(h), "csr_spmm", ["pcie_emu", "nvlink"])
            except SystemExit:
                continue
            for ic in ["pcie_emu", "nvlink"]:
                meas = measured(sweep, r0, L, h, ic)
                epoch = measured(sweep, r0, L, h, ic, column="epoch")
                mod = modelled(counts, model, int(h), rates, ic, L, f_in)
                for with_pipe in (False, True):
                    # pipeline only at k = 0: a data-parallel prefix has no shuffle to overlap
                    common = [s for s in meas if s in mod
                              and (s[0] != "pipeline" or (with_pipe and s[1] == 0))]
                    if len(common) < 2:
                        continue
                    best = min(common, key=meas.get)
                    pick = min(common, key=mod.get)
                    # simple baselines for comparison
                    dgl = ("dgl", L) if ("dgl", L) in common else None
                    by_link = ("dgl", L) if ic == "pcie_emu" else ("split", 0)
                    by_link = by_link if by_link in common else None
                    epoch_common = [x for x in common if x in epoch]
                    epoch_best = min(epoch_common, key=epoch.get) if epoch_common else None
                    rows.append(dict(graph=r0.graph_name, model=model, L=L, h=int(h),
                                     interconnect=ic, with_pipeline=with_pipe,
                                     n_strategies=len(common),
                                     measured_best=strategy_name(*best, L),
                                     roofline_pick=strategy_name(*pick, L),
                                     match=best == pick,
                                     t_best=meas[best], t_pick=meas[pick],
                                     regret=meas[pick] / meas[best] - 1,
                                     base_dgl_match=dgl == best,
                                     base_link_match=by_link == best,
                                     epoch_best=strategy_name(*epoch_best, L) if epoch_best else "",
                                     epoch_winner_differs=(epoch_best is not None
                                                           and epoch_best != best)))
    allrows = pd.DataFrame(rows).sort_values(["graph", "model", "L", "h", "interconnect"])
    df, dfp = allrows[~allrows.with_pipeline], allrows[allrows.with_pipeline]
    os.makedirs(os.path.join(HERE, "results", "roofline"), exist_ok=True)
    allrows.to_csv(os.path.join(HERE, "results", "roofline", "agreement.csv"), index=False)
    print(df.to_string(index=False))
    for name, d in [("without pipeline", df), ("with pipeline", dfp)]:
        miss = d[~d.match]
        print(f"{name}: agreement {d.match.mean():.0%} of {len(d)}, "
              f"mean regret {d.regret.mean():.1%} (over misses {miss.regret.mean():.1%}), "
              f"max regret {d.regret.max():.1%}")
    print(f"baselines (without pipeline): always DGL {df.base_dgl_match.mean():.0%}, "
          f"PCIe->DGL/NVLink->gsplit {df.base_link_match.mean():.0%}; "
          f"winner differs under epoch time in {df.epoch_winner_differs.sum()} of {len(df)}")

    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    pct = lambda x: f"{100 * x:.0f}\\%"

    # full per-setting table (longtable, breaks across pages)
    rows_tex = []
    for r in df.itertuples():
        pick = r.roofline_pick if r.match else r"\textbf{" + r.roofline_pick + "}"
        rows_tex.append(f"{r.graph} & {r.model.upper()} & {r.L} & {r.h} & "
                        f"{rp.NET_LABEL[r.interconnect]} & {r.measured_best} & {pick} & "
                        f"{pct(r.regret)} \\\\")
    head = ("Graph & Model & $L$ & $h$ & Interconnect & Measured best & Roofline pick & Regret \\\\\n")
    with open(os.path.join(HERE, "tables", "roofline_agreement.tex"), "w") as f:
        f.write("% generated by gen_roofline_agreement.py -- do not edit\n"
                "\\begin{longtable}{l l c r l l l r}\n"
                "\\caption{Roofline pick vs.\\ measured best, every setting (DGL, gsplit, and the "
                "hybrids; forward plus backward time). Mismatches in bold.}\\label{tab:final-full}\\\\\n"
                "\\toprule\n" + head + "\\midrule\n\\endfirsthead\n"
                "\\toprule\n" + head + "\\midrule\n\\endhead\n"
                "\\bottomrule\n\\endlastfoot\n" + "\n".join(rows_tex) + "\n\\end{longtable}\n")

    # summary table by graph and model
    groups = ([(f"{g}, {m.upper()}", x) for (g, m), x in df.groupby(["graph", "model"])]
              + [(f"all, {m.upper()}", x) for m, x in df.groupby("model")]
              + [("all", df)])
    summ = []
    for name, x in groups:
        xp = dfp.loc[x.index.intersection(dfp.index)] if False else dfp[
            dfp.set_index(["graph", "model", "L", "h", "interconnect"]).index.isin(
                x.set_index(["graph", "model", "L", "h", "interconnect"]).index)]
        miss = x[~x.match]
        rule = r"\midrule" + "\n" if name.startswith("all") and name.endswith("GAT") else ""
        summ.append(rule + f"{name} & {len(x)} & {pct(x.match.mean())} & {pct(x.regret.mean())} & "
                    f"{pct(miss.regret.mean()) if len(miss) else '--'} & "
                    f"{pct(x.base_dgl_match.mean())} & {pct(x.base_link_match.mean())} & "
                    f"{pct(xp.match.mean()) if len(xp) else '--'} \\\\")
    with open(os.path.join(HERE, "tables", "roofline_agreement_summary.tex"), "w") as f:
        f.write("% generated by gen_roofline_agreement.py -- do not edit\n"
                "\\begin{tabular}{l r r r r r r r}\n\\toprule\n"
                " & & \\multicolumn{3}{c}{Roofline} & \\multicolumn{2}{c}{Baselines: agreement} & "
                "With pipeline \\\\\n"
                "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
                "Settings & $n$ & agreement & regret & regret (misses) & always DGL & "
                "link rule & agreement \\\\\n\\midrule\n"
                + "\n".join(summ) + "\n\\bottomrule\n\\end{tabular}\n")
    print("wrote results/roofline/agreement.csv, tables/roofline_agreement{,_summary}.tex")


if __name__ == "__main__":
    main()
