#!/usr/bin/env python3
"""Sweep GAT-only calibration constants and measure agreement.

Two independent GAT knobs, applied only to model=='gat':
  cc : multiply GAT compute time T_cmp  (attention work in/out of the GEMM count)
  cn : multiply GAT shuffle time T_net  (real all-to-all cost vs peak bandwidth)
For each (cc, cn) we recompute the floorline pick for every setting and report GAT
agreement (and overall). Reads the same inputs as gen_roofline_agreement; writes only
results/roofline/gat_tuning.csv -- it does not touch agreement.csv.
"""
import glob, os
import numpy as np, pandas as pd
import gen_roofline_points as rp
import gen_roofline_agreement as ag

HERE = rp.HERE


def modelled_scaled(counts, model, h, rates, ic, L, f_in, cc, cn):
    t = rp.busiest_gpu_counts(counts, model, h, rates, ic, L, f_in)
    out = {}
    for (system, k), g in t.groupby(["system", "k"]):
        Tc, Tn = g.T_cmp.copy(), g.T_net.copy()
        if model == "gat":
            Tc, Tn = Tc * cc, Tn * cn
        out[(system, k)] = (2 * np.maximum(Tc, Tn).sum() if system == "pipeline"
                            else (Tc + Tn).sum())
    return out


def agreement(cc, cn):
    sweep = pd.read_csv(os.path.join(HERE, "results", "sweep_all.csv"))
    rows = []
    for path in sorted(glob.glob(ag.COUNTS_GLOB)):
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
                meas = ag.measured(sweep, r0, L, h, ic)
                mod = modelled_scaled(counts, model, int(h), rates, ic, L, f_in, cc, cn)
                common = [s for s in meas if s in mod and s[0] != "pipeline"]
                if len(common) < 2:
                    continue
                best = min(common, key=meas.get)
                pick = min(common, key=mod.get)
                rows.append(dict(graph=r0.graph_name, model=model, match=best == pick))
    d = pd.DataFrame(rows)
    return d


def main():
    ccs = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
    cns = [1.0, 1.5, 2.0, 3.0, 4.0]
    print("GAT agreement (overall in parens), by cc (rows) x cn (cols):")
    print("cc\\cn  " + "  ".join(f"{cn:>10.1f}" for cn in cns))
    recs = []
    for cc in ccs:
        cells = []
        for cn in cns:
            d = agreement(cc, cn)
            gat = d[d.model == "gat"].match.mean()
            allm = d.match.mean()
            cells.append(f"{gat:.0%}({allm:.0%})")
            recs.append(dict(cc=cc, cn=cn, gat_agree=gat, all_agree=allm,
                             n_gat=(d.model == "gat").sum()))
        print(f"{cc:<5.1f}  " + "  ".join(f"{c:>10s}" for c in cells))
    out = os.path.join(HERE, "results", "roofline", "gat_tuning.csv")
    pd.DataFrame(recs).to_csv(out, index=False)
    print(f"\nbaseline (cc=1, cn=1) GAT agreement = {recs[0]['gat_agree']:.0%}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
