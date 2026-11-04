#!/usr/bin/env python3
"""gen_hybrid_settings.py -- enumerate the gsplit HYBRID ablation grid across the
Section-1 ablation cells and write a settings JSONL for run_settings.py.

For each Section-1 ablation cell (graph x model x interconnect x (L,h)) we run the gsplit
hybrid family:
  * DP hybrid       (system=split, k leading data-parallel layers): k = 1 .. L-1
      (interior only -- k=0 is full gsplit, already measured in sec1; k=L is pure DP)
  * pipeline hybrid (system=gsplit_pipepar, overlaps shuffle w/ compute): k = 0 .. L-1
      (k=0 is pure pipeline, the headline pipeline hybrid; k=L degenerates to pure DP)

The (L,h) ablation cells are exactly the sec1 single-axis grid (gen_settings.grid_cells):
  {(2,1024),(3,256),(3,1024),(3,4096),(4,1024)}.

run_settings.py runs each under the per-cell cap and records timeout/oom sentinels.
gen_settings.identity_of now includes k, so k-variants of one cell are distinct runs.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # experiment/
import gen_settings as gs  # noqa: E402  (reuse grid/fanout/constants)

GRAPHS = ["papers100M", "products"]
MODELS = ["sage", "gat"]
INTERCONNECTS = ["nvlink", "pcie"]
DP_SYSTEM = "split"
PIPE_SYSTEM = "gsplit_pipepar"
OUT = os.path.join(HERE, "sec1_hybrid_settings.jsonl")


def row(graph, model, interconnect, depth, hid, system, k):
    return {
        "graph": graph, "interconnect": interconnect, "hardware": gs.hw_label(interconnect),
        "system": system, "model": model, "depth": depth, "hid_size": hid,
        "fanout": gs.fanout_of(depth), "num_redundant_layer": k,
        "cache_size": "0G", "cache_frac": -1.0,
        "num_epoch": gs.NUM_EPOCH, "batch_size": gs.BATCH_SIZE,
        "world_size": gs.WORLD_SIZE, "random_feat": gs.RANDOM_FEAT,
    }


def main():
    out = []
    for graph in GRAPHS:
        for model in MODELS:
            for interconnect in INTERCONNECTS:
                for (depth, hid) in gs.grid_cells():
                    for k in range(1, depth):       # DP hybrid, interior k=1..L-1
                        out.append(row(graph, model, interconnect, depth, hid, DP_SYSTEM, k))
                    for k in range(0, depth):       # pipeline hybrid, k=0..L-1
                        out.append(row(graph, model, interconnect, depth, hid, PIPE_SYSTEM, k))
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} hybrid cells -> {OUT}")
    by_sys = Counter(r["system"] for r in out)
    print("  by system:", dict(by_sys))
    by_graph = Counter(r["graph"] for r in out)
    print("  by graph:", dict(by_graph))


if __name__ == "__main__":
    main()
