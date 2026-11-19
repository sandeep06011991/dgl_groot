#!/usr/bin/env python3
"""Floorline pick vs measured winner as a colored grid, for Sec. 5.

Reads results/roofline/agreement.csv (produced by gen_roofline_agreement.py) and emits two grids,
one cell per setting showing the floorline's predicted strategy, green when it matches the
measured winner and red when it does not (a miss cell names the measured winner in parentheses):
  tables/roofline_grid.tex      the Section 2 grid, in the layout of tab:sec1-* (multirow
                                Model/h/L, graph x interconnect columns); green = exact or one layer off, red = further.
  tables/roofline_grid_all.tex  every setting with sampled counts (the earlier grid); darker red
                                for regret >= 10%; empty cell = not run or timed out.

Reads the agreement CSV only; it does not recompute the model and does not overwrite it.
Light CPU work; safe to run locally.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# short labels for the grid cells
ABBR = {"DGL": "D", "gsplit": "G", "hybrid k=1": "h1", "hybrid k=2": "h2", "hybrid k=3": "h3"}
NET = {"nvlink": "NVLink", "pcie_emu": "PCIe"}
GRAPHS = ["products", "papers100M"]
ICS = ["nvlink", "pcie_emu"]


def split_point(name, L):
    """Number of leading data-parallel layers k: gsplit = 0, hybrid k = k, DGL = L."""
    if name == "DGL":
        return L
    if name == "gsplit":
        return 0
    return int(name.split("=")[1])


def cell(r, binary=False):
    pick = ABBR.get(r.roofline_pick, r.roofline_pick)
    if r.match:
        return f"\\cellcolor{{{'green!45' if binary else 'green!22'}}}{pick}"
    best = ABBR.get(r.measured_best, r.measured_best)
    if binary:
        # Section 2 grid: a pick whose split point is one layer from the measured winner's
        # counts as agreement (lighter green, winner still named); beyond one layer is a miss.
        near = abs(split_point(r.roofline_pick, r.L) - split_point(r.measured_best, r.L)) <= 1
        shade = "green!15" if near else "red!30"
    else:
        # all-settings grid keeps the regret-graded shade.
        shade = "red!45" if r.regret >= 0.10 else "red!18"
    return f"\\cellcolor{{{shade}}}{pick}\\,({best})"


def all_settings_grid(d, idx):
    """Every setting with sampled counts, one row per (model, L, h); empty cell = no usable
    measurement (not run, or timed out). Written to tables/roofline_grid_all.tex."""
    lines = []
    colspec = "l c c " + "c " * (len(GRAPHS) * len(ICS))
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    gh = " & ".join(f"\\multicolumn{{{len(ICS)}}}{{c}}{{{g}}}" for g in GRAPHS)
    lines.append(f"Model & $L$ & $h$ & {gh} \\\\")
    start = 4
    cmids = []
    for _ in GRAPHS:
        cmids.append(f"\\cmidrule(lr){{{start}-{start+len(ICS)-1}}}")
        start += len(ICS)
    lines.append("".join(cmids))
    ich = " & ".join(NET[ic] for _ in GRAPHS for ic in ICS)
    lines.append(f" & & & {ich} \\\\")
    lines.append("\\midrule")
    for model in ["sage", "gat"]:
        dm = d[d.model == model]
        first = True
        for (L, h), _ in dm.groupby(["L", "h"]):
            cells = []
            for g in GRAPHS:
                for ic in ICS:
                    key = (g, model, L, h, ic)
                    cells.append(cell(idx.loc[key]) if key in idx.index else "")
            label = f"\\textbf{{{model.upper()}}}" if first else ""
            first = False
            lines.append(f"{label} & {L} & {h} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return lines


# The Section 1 / Section 2 grid: two single-axis ablations crossing at the default
# (L=3, h=1024) -- a hidden sweep at L=3, then a depth sweep at h=1024 (as gen_sec1_tables.py).
SEC2_ICS = ["pcie_emu", "nvlink"]
SEC2_NET = {"pcie_emu": "PCIe (emul.)", "nvlink": "NVLink"}
HIDDEN_SWEEP = [(3, 256), (3, 1024), (3, 4096)]   # (L, h)
DEPTH_SWEEP = [(2, 1024), (3, 1024), (4, 1024)]


def sec2_grid(idx):
    """Section 2 grid in the layout of tab:sec1-*: Model | h | L with multirow spans, graph x
    interconnect column groups, binary green/red cells."""
    n = len(SEC2_ICS)
    ncol = 3 + n * len(GRAPHS)
    L_ = []
    L_.append("\\begin{tabular}{|l|c|c|" + "c|" * (n * len(GRAPHS)) + "}")
    L_.append("\\hline")
    gh = " & ".join(f"\\multicolumn{{{n}}}{{c|}}{{\\textbf{{{g}}}}}" for g in GRAPHS)
    L_.append(f"\\multicolumn{{3}}{{|c|}}{{}} & {gh} \\\\")
    L_.append(f"\\cline{{4-{ncol}}}")
    ich = " & ".join(SEC2_NET[ic] for _ in GRAPHS for ic in SEC2_ICS)
    L_.append(f"\\textbf{{Model}} & $\\mathbf{{h}}$ & $\\mathbf{{L}}$ & {ich} \\\\")
    L_.append("\\hline")

    def cells(model, L, h):
        out = []
        for g in GRAPHS:
            for ic in SEC2_ICS:
                key = (g, model, L, h, ic)
                out.append(cell(idx.loc[key], binary=True) if key in idx.index else "--")
        return " & ".join(out)

    for model in ["sage", "gat"]:
        nrow = len(HIDDEN_SWEEP) + len(DEPTH_SWEEP)
        for i, (L, h) in enumerate(HIDDEN_SWEEP):
            lead = f"\\multirow{{{nrow}}}{{*}}{{\\textbf{{{model.upper()}}}}}" if i == 0 else ""
            span = f"\\multirow{{{len(HIDDEN_SWEEP)}}}{{*}}{{${L}$}}" if i == 0 else ""
            L_.append(f"{lead} & ${h}$ & {span} & {cells(model, L, h)} \\\\")
        L_.append(f"\\cline{{2-{ncol}}}")
        for i, (L, h) in enumerate(DEPTH_SWEEP):
            span = f"\\multirow{{{len(DEPTH_SWEEP)}}}{{*}}{{${h}$}}" if i == 0 else ""
            L_.append(f" & {span} & ${L}$ & {cells(model, L, h)} \\\\")
        L_.append("\\hline")
    L_.append("\\end{tabular}")
    return L_


def main():
    d = pd.read_csv(os.path.join(HERE, "results", "roofline", "agreement.csv"))
    d = d[d.with_pipeline == False].copy()
    idx = d.set_index(["graph", "model", "L", "h", "interconnect"], drop=False)

    for name, lines in [("roofline_grid.tex", sec2_grid(idx)),
                        ("roofline_grid_all.tex", all_settings_grid(d, idx))]:
        out = os.path.join(HERE, "tables", name)
        with open(out, "w") as f:
            f.write("% generated by gen_roofline_grid.py -- do not edit\n" + "\n".join(lines) + "\n")
        print(f"wrote {out}")

    # console summary: all settings, then the Section 2 grid (distinct settings)
    print(f"all settings: {len(d)}, agreement {d.match.mean():.0%}")
    for (g, m), x in d.groupby(["graph", "model"]):
        print(f"  {g:11s} {m.upper():4s}  {x.match.mean():.0%}  ({len(x)})")
    keys = set(HIDDEN_SWEEP + DEPTH_SWEEP)
    s = d[[(L, h) in keys for L, h in zip(d.L, d.h)]]
    print(f"Section 2 grid: {len(s)} settings, {int(s.match.sum())} match")
    print(s.groupby(["graph", "model", "interconnect"]).match.agg(["sum", "count"]).to_string())
    dk = [split_point(a, L) - split_point(b, L)
          for a, b, L in zip(s.roofline_pick, s.measured_best, s.L)]
    s = s.assign(dk=dk)
    print(f"  exact {int((s.dk == 0).sum())}, one layer off {int((s.dk.abs() == 1).sum())}, "
          f"further {int((s.dk.abs() > 1).sum())}   (dk = k_pick - k_best)")
    print(s[s.dk.abs() > 1][["graph", "model", "L", "h", "interconnect", "roofline_pick",
                             "measured_best", "dk"]].to_string(index=False))
    miss = s[~s.match]
    print(miss.groupby(["roofline_pick", "measured_best"]).size().to_string())


if __name__ == "__main__":
    main()
