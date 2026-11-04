# Plan — Section 3: The Simulator and Its Fidelity

> **2026-11-03 — superseded in the PDF by `roofline.tex`.** The chapter now presents an
> *analytical roofline model* (per-node pull-vs-recompute, nested memory/network rooflines,
> SAGE/GAT factors in units of |E| and |V|, Algorithm 1) instead of the brute-force cost-model
> simulator. `simulator.tex` is kept on disk but commented out of `main.tex`; no simulator
> code is referenced in the LaTeX. Numbers/figure come from `gen_roofline.py` →
> `tables/roofline_instances.tex`, `figures/roofline.tex` (figure is a standalone tikzpicture
> for slides). Parameters to refine in the script: `RHO`, `B_NET`, `HS`. Open items: measured
> `r`, `δ` per hop; `η`, `β`; the h=4096 ladder as the test of the "wider ⇒ more compute-bound"
> prediction.
>
> _Original plan below, unchanged._

**Key argument:** A cost-model simulator predicts each hybrid configuration's performance
cheaply (≈1 GPU, no full training), turning the intractable config sweep into a few
predictions. Simulator + fidelity are ONE section (per user). Fidelity is demonstrated on
**papers100M, 4×A100, SAGE & GAT, L=3**; a prediction **within 10%** of measured is good.

**Structure (done in `simulator.tex`):**
1. Cheap-prediction philosophy (per-layer compute timed once; comm/load = volume ÷ BW).
2. `sec:sim-baseline` — validate against DGL baseline first (note + placeholder).
3. `sec:sim-fidelity` — **papers100M / 4GPU / SAGE+GAT / L=3**, `tab:sim-fidelity` with
   |Err| and a within-10% pass column.
4. `sec:microprofile` — nsys component cross-check (REAL data, kept).
5. Conclusion → leads into Section 4 (predicted cases).

**Reorg note:** the old 2×A100 products/arxiv `sim-vs-measured` table was removed — fidelity
is now redefined to papers100M/4GPU/SAGE+GAT/L=3 (within 10%). The old numbers remain in git
history if ever needed. The nsys microprofile (also products, component-level) is kept as
supporting evidence that the cost-model components are sound.

**Source material / TODO:**
- `experiment/simulation/hybrid_dp_sim.py`, `get_best_strategy.py`.
- Fill `tab:sim-fidelity` (papers100M 4GPU SAGE/GAT L=3 sim vs measured + err%).
- Fill the DGL-baseline validation placeholder.
- Pending cost-model fixes: add backward to compute term; model the feature cache in the
  load term (biggest fix); average small graphs over epochs.
