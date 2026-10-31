# Plan — Section 4: Simulator-Predicted Cases

**Key argument:** With the simulator validated (Section 3), run it on the Section 1 bad
cases (deep-layer, large-hidden; products + papers100M) to predict — before building
anything — the best Split/hybrid configuration and its standing vs. Quiver and DGL.

**Structure (done in `predicted_cases.tex`):**
- Intro: sweep the hybrid config space (split-point k, pipeline schedule) per case.
- `tab:predicted`: DGL / Quiver / Best Split / k* per case (scratch).
- Expected: a hybrid with 0 < k* < L beats both full-split gsplit and Quiver; k* grows
  with depth/hidden. Risk: if the sweep keeps picking k*=0, the hybrids offer no benefit.
- Conclusion: these predicted configs are what Section 5 builds and measures.

**Source material / TODO:**
- `experiment/simulation/hybrid_dp_sim.py::simulate_optimal_k`, `get_best_strategy.py`.
- Fill `tab:predicted` by running the simulator (single GPU) over k for each case.
- Confirm what "Quiver prediction" means in the simulator (does it model Quiver, or is
  Quiver measured? — Section 1 has measured Quiver; here it may be predicted or carried over).
