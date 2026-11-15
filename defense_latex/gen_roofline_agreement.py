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


def strategy_name(system, k, L):
    if system == "dgl":
        return "DGL"
    if system == "split":
        return "gsplit" if k == 0 else f"hybrid k={k}"
    return "pipeline" if k == 0 else f"pipe k={k}"


def measured(sweep, r0, L, h, ic):
    """Measured min fwd+bwd per (system, k) for one setting."""
    fan = "[" + ", ".join(str(r0.fanouts).split(",")) + "]"
    s = sweep[(sweep.graph_name == r0.graph_name) & (sweep.model == r0.model)
              & (sweep.fanouts == fan) & (sweep.hid_size == h) & (sweep.hardware == ic)
              & (sweep.batch_size == r0.batch_size) & (sweep.world_size == r0.world_size)
              & (sweep.partition_type == r0.partition_type)]
    s = s[~((s.system == "gsplit_pipepar") & (s.timestamp.astype(str) <= LEGACY_PIPE_CUTOFF))]
    s = s.assign(t=pd.to_numeric(s["forward (s)"], errors="coerce")
                 + pd.to_numeric(s["backward (s)"], errors="coerce")).dropna(subset=["t"])
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
    for (system, k), g in t.groupby(["system", "k"]):
        if system == "pipeline":
            out[(system, k)] = 2 * np.maximum(g.T_cmp, g.T_net).sum()
        else:
            out[(system, k)] = (g.T_cmp + g.T_net).sum()
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
                mod = modelled(counts, model, int(h), rates, ic, L, f_in)
                common = [s for s in meas if s in mod]
                if len(common) < 2:
                    continue
                best = min(common, key=meas.get)
                pick = min(common, key=mod.get)
                rows.append(dict(graph=r0.graph_name, model=model, L=L, h=int(h),
                                 interconnect=ic, n_strategies=len(common),
                                 measured_best=strategy_name(*best, L),
                                 roofline_pick=strategy_name(*pick, L),
                                 match=best == pick,
                                 t_best=meas[best], t_pick=meas[pick],
                                 regret=meas[pick] / meas[best] - 1))
    df = pd.DataFrame(rows).sort_values(["graph", "model", "L", "h", "interconnect"])
    os.makedirs(os.path.join(HERE, "results", "roofline"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", "roofline", "agreement.csv"), index=False)
    print(df.to_string(index=False))
    print(f"\nagreement {df.match.mean():.0%} of {len(df)}, "
          f"mean regret {df.regret.mean():.1%}, max regret {df.regret.max():.1%}")

    lines = []
    for r in df.itertuples():
        pick = r.roofline_pick if r.match else r"\textbf{" + r.roofline_pick + "}"
        lines.append(f"{r.graph} & {r.model.upper()} & {r.L} & {r.h} & "
                     f"{rp.NET_LABEL[r.interconnect]} & {r.measured_best} & {pick} & "
                     f"{100 * r.regret:.0f}\\% \\\\")
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{5}}{{l}}{{Agreement / mean regret ({len(df)} settings)}} & "
                 rf"\multicolumn{{2}}{{c}}{{{100 * df.match.mean():.0f}\%}} & "
                 rf"{100 * df.regret.mean():.0f}\% \\")
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    with open(os.path.join(HERE, "tables", "roofline_agreement.tex"), "w") as f:
        f.write("% generated by gen_roofline_agreement.py -- do not edit\n"
                "\\begin{tabular}{l l c r l l l r}\n\\toprule\n"
                "Graph & Model & $L$ & $h$ & Interconnect & Measured best & Roofline pick & Regret \\\\\n"
                "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    print("wrote results/roofline/agreement.csv, tables/roofline_agreement.tex")


if __name__ == "__main__":
    main()
