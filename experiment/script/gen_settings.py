#!/usr/bin/env python3
"""gen_settings.py -- enumerate the Section-1 run grid and diff it against the
consolidated results CSV, emitting ONLY the runs that are not already present.

Section 1 of the dissertation compares gsplit (``--system=split``, k=0) against DGL,
both at a 0% feature cache, on two graphs. The grid is two single-axis ablations per
(graph, model, interconnect, system):

  * hidden sweep: h in {256, 1024, 4096} at depth L=3
  * depth sweep:  L in {2, 3, 4}         at hidden h=1024
  (shared pivot: L=3, h=1024)

Rather than a bash for-loop (which re-runs cells already measured), we enumerate the
grid here, read the master CSV, and write a settings file (one JSON object per line)
holding only the MISSING cells. ``run_settings.py`` then iterates that file.

A cell counts as "present" when the CSV already has a row with the same
(graph, hardware, system, model, depth, hid_size) AT A 0% CACHE -- including sentinel
``timeout``/``oom`` rows, so a cell that was attempted and capped/OOM'd is not retried.

Usage:
    python3 gen_settings.py                       # writes sec1_settings.jsonl next to this file
    python3 gen_settings.py --csv ../../defense_latex/results/sweep_all.csv
    python3 gen_settings.py --dry-run             # print the summary + missing cells, write nothing
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CSV = os.path.normpath(os.path.join(HERE, "..", "..", "defense_latex", "results", "sweep_all.csv"))
DEFAULT_OUT = os.path.join(HERE, "sec1_settings.jsonl")

# --- the Section-1 grid (prompt_monday.md) -----------------------------------
GRAPHS = ["papers100M", "products"]          # papers100M first (products already collected)
MODELS = ["sage", "gat"]
INTERCONNECTS = ["nvlink", "pcie"]          # pcie is emulated (NCCL_P2P_DISABLE=1)
SYSTEMS = ["dgl", "split"]                   # gsplit@0 vs DGL@0 only -- no quiver, no k>0
HID_SWEEP = [256, 1024, 4096]                # at DEPTH_ANCHOR (coarsened grid)
DEPTH_SWEEP = [2, 3, 4]                       # at HID_ANCHOR
DEPTH_ANCHOR = 3                              # depth at which the hidden sweep runs
HID_ANCHOR = 1024                             # hidden at which the depth sweep runs
                                             # -> shared pivot is (L=3, h=1024)
FPL = 15                                     # fanout per layer

# fixed run knobs (match the Section-1 protocol: 3 epochs, 0% cache, random feats)
NUM_EPOCH = 3
BATCH_SIZE = 1024
WORLD_SIZE = 4
RANDOM_FEAT = True


def hw_label(interconnect):
    """settings 'nvlink'/'pcie' -> the value recorded in the CSV 'hardware' column."""
    return "nvlink" if interconnect == "nvlink" else "pcie_emu"


def fanout_of(depth):
    return ",".join([str(FPL)] * depth)


def grid_cells():
    """Yield (depth, hid) for the union of the two single-axis sweeps (anchor once)."""
    seen = set()
    for h in HID_SWEEP:
        key = (DEPTH_ANCHOR, h)
        if key not in seen:
            seen.add(key)
            yield key
    for d in DEPTH_SWEEP:
        key = (d, HID_ANCHOR)
        if key not in seen:
            seen.add(key)
            yield key


def enumerate_grid():
    """Full planned grid as a list of settings dicts."""
    out = []
    for graph in GRAPHS:
        for model in MODELS:
            for interconnect in INTERCONNECTS:
                for system in SYSTEMS:
                    for depth, hid in grid_cells():
                        out.append({
                            "graph": graph,
                            "interconnect": interconnect,
                            "hardware": hw_label(interconnect),
                            "system": system,
                            "model": model,
                            "depth": depth,
                            "hid_size": hid,
                            "fanout": fanout_of(depth),
                            "cache_size": "0G",
                            "cache_frac": -1.0,
                            "num_redundant_layer": 0,
                            "num_epoch": NUM_EPOCH,
                            "batch_size": BATCH_SIZE,
                            "world_size": WORLD_SIZE,
                            "random_feat": RANDOM_FEAT,
                        })
    return out


def _depth_from_fanouts(s):
    """'[15, 15, 15]' or '15,15,15' -> 3."""
    s = str(s).strip().strip("[]")
    if not s:
        return 0
    return len([x for x in s.replace(" ", "").split(",") if x != ""])


def _cache_gb(s):
    """'10G'/'0GB'/'0G'/'' -> float GB (used only to decide 0% vs cached)."""
    s = str(s).strip().upper().rstrip("B").rstrip("G")
    try:
        return float(s)
    except ValueError:
        return 0.0


def present_identities(csv_path):
    """Set of (graph, hardware, system, model, depth, hid) that already have a
    0%-cache row in the CSV (any row, including timeout/oom sentinels)."""
    if not os.path.isfile(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    ids = set()
    for _, r in df.iterrows():
        # only 0%-cache rows satisfy the Section-1 condition
        if _cache_gb(r.get("cache_size", "0")) != 0.0:
            continue
        frac = r.get("cache_frac", -1.0)
        try:
            if float(frac) > 0:
                continue
        except (TypeError, ValueError):
            pass
        ids.add((
            str(r.get("graph_name")),
            str(r.get("hardware")),
            str(r.get("system")),
            str(r.get("model")),
            _depth_from_fanouts(r.get("fanouts")),
            int(r.get("hid_size")),
        ))
    return ids


def identity_of(cell):
    return (cell["graph"], cell["hardware"], cell["system"], cell["model"],
            int(cell["depth"]), int(cell["hid_size"]))


def main():
    ap = argparse.ArgumentParser(description="enumerate Section-1 grid, emit missing runs")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="consolidated results CSV to diff against")
    ap.add_argument("--out", default=DEFAULT_OUT, help="settings file to write (JSONL)")
    ap.add_argument("--dry-run", action="store_true", help="print summary + missing cells, write nothing")
    args = ap.parse_args()

    planned = enumerate_grid()
    present = present_identities(args.csv)
    missing = [c for c in planned if identity_of(c) not in present]

    print(f"CSV:      {args.csv}" + ("" if os.path.isfile(args.csv) else "  (absent -> all planned are missing)"))
    print(f"planned:  {len(planned)} cells")
    print(f"present:  {len(planned) - len(missing)} already in CSV (0%-cache rows, incl. timeout/oom)")
    print(f"to run:   {len(missing)} missing cells")
    if missing:
        print("--- missing (graph/hw/system/model  L=depth h=hid) ---")
        for c in missing:
            print(f"  {c['graph']:11s} {c['hardware']:9s} {c['system']:6s} {c['model']:4s}  "
                  f"L={c['depth']} h={c['hid_size']}")

    if args.dry_run:
        print("\n[dry-run] no settings file written.")
        return

    with open(args.out, "w") as f:
        for c in missing:
            f.write(json.dumps(c) + "\n")
    print(f"\nwrote {len(missing)} settings -> {args.out}")


if __name__ == "__main__":
    main()
