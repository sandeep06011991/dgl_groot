# Plan — Section 2: Alternate Hybrids

**Key argument:** Two hybrids can relieve the Section 1 settings — (2A) data-parallel hybrid
(DP in top layers, split in bottom; split-point k), (2B) pipeline hybrid (overlap one
stream's shuffle with another's compute). Both add config parameters, so a brute-force
sweep is too costly → motivates the simulator (Section 3).

**Kept deliberately brief** per the user — will gain detail once experimental results land.

**Structure (done in `hybrids.tex`):**
- Intro, 2A subsection, 2B subsection, conclusion linking to the simulator.

**Source material:**
- `plans/01_dp_plus_split.md` — DP-then-split, split-point k, `num_redundant_layer`, micro-tests.
- `plans/02_pipeline_plus_split.md` — two staggered micro-batch streams.
- `plans/README.md` — n-clique + toy-op micro-test methodology.

**TODO when expanding:** add a small schematic per hybrid (layer timeline showing where the
shuffle fires / overlaps); state the exact config parameters each introduces.
