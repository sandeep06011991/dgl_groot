# Plan — Section 1: Where gsplit Underperforms

**Key argument:** gsplit is already built and decent, but ablations show its gain over the
baseline collapses at (a) deeper layers and (b) larger hidden sizes, both via rising
shuffle cost. Conclusion: settings exist where gsplit loses to DGL.

**Structure (done in `limitations.tex`):**
1. The two bad situations explained (depth → more shuffles; hidden → bigger shuffle volume).
2. **Two tables, one per graph** (`tab:abl-products`, `tab:abl-papers`). Each: DGL/Quiver/gsplit
   columns, SAGE+GAT rows, a layer sweep (L=3,4,5 at hidden 256) and a hidden sweep
   (h=64,128,256,512 at L=3). arxiv dropped (toy graph).
3. Expected conclusion (gsplit falls behind DGL as depth/hidden grow; crossover earlier on
   papers100M) + section conclusion.

**Settings:** default is **4×A100-SXM** (was 2×A100 in the old measured runs — all gsplit
cells are placeholders to re-measure at 4 GPUs, not ported).

**Source material / TODO to fill the placeholders:**
- gsplit numbers: re-run at 4×A100 (old 2-GPU numbers no longer apply).
- DGL numbers: run `experiment/nodepred/dgl_trainer.py` (`--system=dgl`) same configs.
- Quiver numbers: run `experiment/nodepred/quiver_trainer.py` (`--system=quiver`).
- Decide whether to report total epoch time or the sample/load/train breakdown per system.
- Confirm the default hidden (256) for the layer sweep and default depth (L=3) for the hidden sweep.
