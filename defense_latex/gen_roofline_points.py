#!/usr/bin/env python3
"""Place sampled layers on the classic roofline of Sec. 3 (roofline_classic.tex).

Inputs
  --counts   CSV from experiment/roofline/sample_counts.py: one row per (system, k,
             minibatch, layer) with per-GPU max/sum of n, e, m (shuffled nodes) and u
  rates      results/microbench/rates_compute.csv, rates_net.csv (tab:cm-microbench)
  constants  MODEL_CONSTANTS below (tab:cm-constants) -- PROVISIONAL, edit here

Outputs (under --out_dir, default defense_latex/; tag = --tag)
  figures/roofline_<tag>.pdf/.png      one panel per interconnect, same points
  results/roofline/<tag>_points.csv    one row per (interconnect, system, k, layer)
  results/roofline/<tag>_rules.csv     per layer: Eq. (cm-rule) verdicts and winner
Light CPU work (pandas + matplotlib); safe to run locally.

Example:
  python3 defense_latex/gen_roofline_points.py \
      --counts defense_latex/results/roofline/products_sage_4x15_b1024.csv \
      --hid_size 512 --tag products_sage_4x15_h512
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BYTES_PER_VALUE = 4  # fp32
MICROBENCH_GPUS = 4  # rates_net.csv was measured with a 4-GPU all-to-all

# Eq. (dense-cost)/(sparse-cost) per destination node with e in-edges:
#   F(e) = 2[(be*e + bn) h^2 + (ge*e + gn) h]   FLOPs,   S(e) = kappa * e * s_H  bytes.
# Both are linear in e, so summing them over nodes = plugging in summed n and e.
# PROVISIONAL forward-pass values (tab:cm-constants is still ??):
#  SAGE: 2 GEMMs per dst (W_self, W_neigh after the mean); 1 row gathered per edge.
#  GAT : z = W h charged per edge (M2 over-count; the self loop is one of the edges,
#        so no extra per-node GEMM); el per edge, er per node, alpha*z per edge;
#        1 row gathered per edge.
MODEL_CONSTANTS = {
    "sage": dict(kappa=1, be=0, bn=2, ge=0, gn=0),
    "gat":  dict(kappa=1, be=1, bn=0, ge=2, gn=1),
}
# input-feature width F_in of the outermost layer (utils/dataloading.py get_feat_dim)
FEAT_DIM = {"products": 100, "papers100M": 128, "orkut": 512, "friendster": 128}
# training step = forward + backward (assumption R5): dense work x3 (x2 for the input
# layer, no input gradient), shuffled bytes x2 (rows forward, gradients back)
PHI_CMP, PHI_CMP_INPUT, PHI_NET = 3.0, 2.0, 2.0
NET_KERNEL = {"pcie_emu": "nccl_a2a_pcie_emu", "nvlink": "nccl_a2a_nvlink"}
NET_LABEL = {"pcie_emu": "PCIe-emu", "nvlink": "NVLink"}


def parse_args():
    p = argparse.ArgumentParser(description="Plot sampled layers on the classic roofline.")
    p.add_argument("--counts", required=True, help="CSV written by sample_counts.py")
    p.add_argument("--hid_size", type=int, required=True, help="hidden size h (must be in the rates CSV)")
    p.add_argument("--interconnects", default="pcie_emu,nvlink",
                   help="comma list of panels: pcie_emu, nvlink")
    p.add_argument("--dp_system", default="dgl", choices=["dgl", "split"],
                   help="pure-DP line from the DGL baseline (default) or from the split "
                        "loader with k = L (same seeds as gsplit; assumption M4)")
    p.add_argument("--gather_kernel", default="csr_spmm",
                   choices=["csr_spmm", "index_select_sum"],
                   help="which measured B_mem row defines the gather roof")
    p.add_argument("--tag", required=True, help="name for the output files")
    p.add_argument("--title_note", default="", help="extra text appended to the figure title")
    p.add_argument("--out_dir", default=HERE,
                   help="root for figures/ and results/roofline/ (default: defense_latex)")
    p.add_argument("--sweep", default=os.path.join(HERE, "results", "sweep_all.csv"),
                   help="measured epoch times annotated on each panel ('' to skip)")
    return p.parse_args()


def load_rates(h, gather_kernel, interconnects):
    """Measured P_hat, B_mem, B_net at hidden size h (tab:cm-microbench), in FLOP/s and B/s."""
    rc = pd.read_csv(os.path.join(HERE, "results", "microbench", "rates_compute.csv"))
    rn = pd.read_csv(os.path.join(HERE, "results", "microbench", "rates_net.csv"))

    def one(df, kernel):
        row = df[(df.kernel == kernel) & (df.h == h)]
        if row.empty:
            raise SystemExit(f"no measured '{kernel}' rate at h={h}; have h={sorted(df.h.unique())}")
        return row.value.iloc[0]

    rates = dict(P=one(rc, "gemm") * 1e12, Bmem=one(rc, gather_kernel) * 1e9, net={})
    for ic in interconnects:
        if ic not in NET_KERNEL:
            raise SystemExit(f"unknown interconnect '{ic}'; choose from {list(NET_KERNEL)}")
        rates["net"][ic] = one(rn, NET_KERNEL[ic]) * 1e9
    return rates


def busiest_gpu_counts(mb, model, h, rates, ic, L, f_in):
    """Per (system, k, layer): the busiest GPU's work, taken PER MINIBATCH (the shuffle
    syncs GPUs every minibatch) and then summed over minibatches.
    Per minibatch the busiest GPU is represented by the per-GPU maxima n_max, e_max,
    m_max, each maximised on its own (assumption M3; an upper bound when the maxima
    sit on different GPUs); its time is max(T_cmp, T_net) under (R1)."""
    c = MODEL_CONSTANTS[model]
    a = mb.copy()
    outermost = a.layer == L - 1
    d_in = np.where(outermost, f_in, h)                              # input row width d_l (M1)
    a["row_bytes"] = d_in * BYTES_PER_VALUE                           # s_l: shuffled row size
    gathered = a.row_bytes if model == "sage" else h * BYTES_PER_VALUE  # SAGE: input row; GAT: z
    phi_cmp = np.where(outermost, PHI_CMP_INPUT, PHI_CMP)
    # training-step n*F(e_bar) and n*S(e_bar) of the busiest GPU (Eqs. dense-cost, sparse-cost, R5)
    a["flops"] = phi_cmp * 2 * ((c["be"] * a.e_max + c["bn"] * a.n_max) * d_in * h
                                + (c["ge"] * a.e_max + c["gn"] * a.n_max) * h)
    a["bytes_gathered"] = phi_cmp * c["kappa"] * a.e_max * gathered
    a["net_bytes"] = PHI_NET * a.m_max * a.row_bytes
    g = a.bytes_gathered / a.flops                                   # bytes gathered per FLOP
    a["C"] = np.minimum(rates["P"], rates["Bmem"] / g)                # Eq. roof-cmp
    a["T_cmp"] = a.flops / a.C                                        # Eq. cm-times
    a["T_net"] = a.net_bytes / rates["net"][ic]
    sums = a.groupby(["system", "k", "layer", "mode"], as_index=False)[
        ["n_max", "e_max", "m_max", "u", "flops", "bytes_gathered", "net_bytes",
         "T_cmp", "T_net"]].sum()
    return sums.rename(columns={"n_max": "n", "e_max": "e", "m_max": "m"})


def roofline_points(sums, rates, ic, G):
    """Eqs. cm-x, cm-redundancy, cm-pipe-point, cm-gsplit-point, cm-dp-point, ridge."""
    B_net = rates["net"][ic]
    pts = sums.copy()
    pts["C"] = pts.flops / pts.T_cmp                     # FLOP-weighted compute ceiling
    with np.errstate(divide="ignore"):
        pts["I"] = pts.flops / pts.net_bytes             # inf for DP layers (m = 0)
    pts["delta"] = G * pts.n / pts.u
    interconnect_roof = B_net * pts.I
    pts["y_pipe"] = np.minimum(pts.C, interconnect_roof) / pts.delta
    pts["y_gsplit"] = 1.0 / (pts.delta * (1 / pts.C + 1 / interconnect_roof))   # 1/inf = 0
    pts["y_dp"] = pts.C / pts.delta
    pts["I_star"] = pts.C / B_net
    pts["bound"] = np.where(pts.C <= interconnect_roof, "compute", "interconnect")
    g = pts.bytes_gathered / pts.flops
    pts["compute_roof"] = np.where(rates["P"] <= rates["Bmem"] / g, "dense", "gather")
    pts["interconnect"] = ic
    pts["B_net"] = B_net
    return pts


def pick(pts, system, k):
    return pts[(pts.system == system) & (pts.k == k)].set_index("layer")


def rule_table(pts, L, dp_system):
    """Per layer l: pure-DP line vs gsplit (split, k=0, serialised) and pipeline
    (pipeline, k=0, its own counts, overlapped) -- Eq. cm-rule plus the actual heights.
    The rule assumes one C for both; C_spread reports how far that is from true."""
    dp, split, pipe = pick(pts, dp_system, L), pick(pts, "split", 0), pick(pts, "pipeline", 0)
    rows = []
    for l in sorted(split.index):
        Delta = dp.delta[l] / split.delta[l]
        dist = split.I_star[l] / split.I[l]                       # I*/I, distance from ridge
        heights = {"dp": dp.y_dp[l], "pipe": pipe.y_pipe[l], "gsplit": split.y_gsplit[l]}
        winner = max(heights, key=heights.get)
        rows.append(dict(layer=l, Delta=Delta, Istar_over_I=dist,
                         dp_beats_gsplit=Delta < 1 + dist,
                         dp_beats_pipe=Delta < max(1.0, dist),
                         C_spread=abs(dp.C[l] / split.C[l] - 1),
                         winner=winner))
    return pd.DataFrame(rows)


def measured_times(sweep, counts, h, ic):
    """Measured min epoch times for this setting from sweep_all.csv, as a short string."""
    if not sweep or not os.path.exists(sweep):
        return ""
    s = pd.read_csv(sweep)
    r0 = counts.iloc[0]
    fan = "[" + ", ".join(str(r0.fanouts).split(",")) + "]"
    s = s[(s.graph_name == r0.graph_name) & (s.model == r0.model) & (s.fanouts == fan)
          & (s.hid_size == h) & (s.hardware == ic) & (s.batch_size == r0.batch_size)
          & (s.world_size == r0.world_size) & (s.partition_type == r0.partition_type)]
    # training time only (what the roofline models): forward + backward, no sampling/features
    s = s.assign(t=pd.to_numeric(s["forward (s)"], errors="coerce")
                 + pd.to_numeric(s["backward (s)"], errors="coerce")).dropna(subset=["t"])
    if s.empty:
        return "measured: none"
    parts = []
    for label, sel in [("DGL", s.system == "dgl"),
                       ("gsplit", (s.system == "split") & (s.num_redundant_layers == 0)),
                       ("pipeline", (s.system == "gsplit_pipepar") & (s.num_redundant_layers == 0))]:
        t = s[sel].t.min()
        if not np.isnan(t):
            parts.append(f"{label} {t:.2f}s")
    hybrid = s[(s.system == "split") & (s.num_redundant_layers > 0)]
    if not hybrid.empty:
        r = hybrid.loc[hybrid.t.idxmin()]
        parts.append(f"hybrid k={int(r.num_redundant_layers)} {r.t:.2f}s")
    return "measured fwd+bwd: " + ", ".join(parts)


def stagger(ys, min_ratio=1.07):
    """Spread label heights apart (log scale) so right-edge labels never overlap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        prev, cur = order[i - 1], order[i]
        out[cur] = max(out[cur], out[prev] * min_ratio)
    return out


def plot(all_pts, rules, L, args, model, counts):
    """One panel per interconnect. Per layer (one colour): gsplit x, pipeline dot, and the
    pure-DP dashed line; the highest mark of a colour wins that layer."""
    ics = list(all_pts)
    fig, axes = plt.subplots(1, len(ics), figsize=(5.6 * len(ics), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, L))
    TF = 1e12   # plot in TFLOP/s

    finite = np.concatenate([b.I[np.isfinite(b.I)].values for b in all_pts.values()]
                            + [b.I_star.values for b in all_pts.values()])
    xlo, xhi = finite.min() / 3, finite.max() * 3
    xx = np.logspace(np.log10(xlo), np.log10(xhi), 200)
    ys = np.concatenate([np.r_[b.y_gsplit.values, b.y_pipe.values, b.y_dp.values]
                         for b in all_pts.values()]) / TF
    C_max = max(b.C.max() for b in all_pts.values()) / TF
    ylo, yhi = ys.min() / 1.5, C_max * 1.4

    for ax, ic in zip(axes, ics):
        b = all_pts[ic]
        dp, split, pipe = (b[(b.system == args.dp_system) & (b.k == L)],
                           b[(b.system == "split") & (b.k == 0)],
                           b[(b.system == "pipeline") & (b.k == 0)])
        B_net = b.B_net.iloc[0]
        # roof: one line when the compute ceiling is the same for all layers, else one per layer
        Cs = split.C.values / TF
        if Cs.max() / Cs.min() < 1.05:
            ax.plot(xx, np.minimum(np.median(Cs), B_net * xx / TF), color="k", lw=1.4)
        else:
            for l, C_l in zip(split.layer, Cs):
                ax.plot(xx, np.minimum(C_l, B_net * xx / TF), color=colors[l], lw=0.9, alpha=0.7)
        I_star = split.I_star.median()
        ax.axvline(I_star, color="0.45", ls=":", lw=1)
        ax.text(I_star, ylo * 1.08, r" ridge $I^*$", color="0.35", fontsize=8, va="bottom")
        dp_y = []
        for l in range(L):
            col = colors[l]
            s, p, d = split[split.layer == l], pipe[pipe.layer == l], dp[dp.layer == l]
            if not d.empty:
                dp_y.append((l, d.y_dp.iloc[0] / TF))
                ax.axhline(dp_y[-1][1], color=col, ls="--", lw=1.1)
            if s.empty or not np.isfinite(s.I.iloc[0]):
                continue
            xs, yg = s.I.iloc[0], s.y_gsplit.iloc[0] / TF
            if not p.empty and np.isfinite(p.I.iloc[0]):
                xp, yp = p.I.iloc[0], p.y_pipe.iloc[0] / TF
                ax.plot([xs, xp], [yg, yp], color=col, lw=0.8)
                ax.plot(xp, yp, "o", color=col, ms=6)
            ax.plot(xs, yg, "x", color=col, ms=7, mew=1.8)
        if dp_y:
            labels_y = stagger([y for _, y in dp_y])
            for (l, _), y in zip(dp_y, labels_y):
                ax.text(xhi, y, f" ℓ={l}", color=colors[l], va="center", fontsize=8, clip_on=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("FLOPs per interconnect byte $I$")
        ax.set_title(f"{NET_LABEL[ic]} ({B_net/1e9:.0f} GB/s)\n"
                     + measured_times(args.sweep, counts, args.hid_size, ic), fontsize=9)
        ax.grid(True, which="major", alpha=0.25)
    axes[0].set_ylabel("useful TFLOP/s per GPU")
    handles = [plt.Line2D([], [], color=colors[l], lw=4) for l in range(L)]
    labels = [f"ℓ={l}" for l in range(L)]
    handles += [plt.Line2D([], [], color="0.3", marker="x", ls="", mew=1.8),
                plt.Line2D([], [], color="0.3", marker="o", ls=""),
                plt.Line2D([], [], color="0.3", ls="--"),
                plt.Line2D([], [], color="k", lw=1.4)]
    labels += ["gsplit", "pipeline", "DGL" if args.dp_system == "dgl" else "pure DP", "roof"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8,
               frameon=False, handlelength=1.6, columnspacing=1.2)
    r0 = counts.iloc[0]
    fig.suptitle(f"{r0.graph_name}, {model.upper()}, L={L}, h={args.hid_size} "
                 f"{args.title_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 0.98, 1))
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    out = os.path.join(args.out_dir, "figures", f"roofline_{args.tag}")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=150)
    print("wrote", out + ".pdf/.png")


def main():
    args = parse_args()
    counts = pd.read_csv(args.counts)
    if "system" not in counts.columns:
        raise SystemExit("old-format counts (no 'system' column); re-run sample_counts.py")
    model = counts.model.iloc[0]
    G = int(counts.world_size.iloc[0])
    L = int(counts.layer.max()) + 1
    if G != MICROBENCH_GPUS:
        print(f"WARNING: counts use {G} GPUs but B_net was measured over {MICROBENCH_GPUS}")
    have = set(zip(counts.system, counts.k))
    for need in [("split", 0), ("pipeline", 0), (args.dp_system, L)]:
        if need not in have:
            raise SystemExit(f"counts need (system, k)={need}; have {sorted(have)}")
    f_in = FEAT_DIM[counts.graph_name.iloc[0]]
    interconnects = args.interconnects.split(",")
    rates = load_rates(args.hid_size, args.gather_kernel, interconnects)

    all_pts, rules, tables, rule_rows = {}, {}, [], []
    for ic in interconnects:
        sums = busiest_gpu_counts(counts, model, args.hid_size, rates, ic, L, f_in)
        pts = roofline_points(sums, rates, ic, G)
        r = rule_table(pts, L, args.dp_system)
        all_pts[ic], rules[ic] = pts, r
        tables.append(pts)
        rule_rows.append(r.assign(interconnect=ic))
        print(f"\n== {NET_LABEL[ic]}  B_net={rates['net'][ic]/1e9:.1f} GB/s  "
              f"P={rates['P']/1e12:.1f} TFLOP/s  B_mem={rates['Bmem']/1e9:.0f} GB/s")
        print(pts[["system", "k", "layer", "mode", "n", "m", "u", "I", "delta", "C", "I_star",
                   "bound", "y_pipe", "y_gsplit", "y_dp"]].to_string(index=False))
        print(r.to_string(index=False))

    res_dir = os.path.join(args.out_dir, "results", "roofline")
    os.makedirs(res_dir, exist_ok=True)
    pd.concat(tables).to_csv(os.path.join(res_dir, f"{args.tag}_points.csv"), index=False)
    pd.concat(rule_rows).to_csv(os.path.join(res_dir, f"{args.tag}_rules.csv"), index=False)
    print("wrote", os.path.join(res_dir, f"{args.tag}_points.csv / _rules.csv"))
    plot(all_pts, rules, L, args, model, counts)


if __name__ == "__main__":
    main()
