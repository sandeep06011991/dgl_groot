# Chapter Outline — Hybrid Parallelism for gsplit

**Chapter premise.** gsplit (Spara) is *already built* and earlier chapters show it performs
decently in many settings. But our ablation studies reveal settings where its performance
gain over the baseline **drops significantly** — specifically at **deeper layers** (shuffle
cost grows) and at **larger hidden sizes** (shuffle volume grows). This final chapter
explores **hybrid parallelism** schemes to handle those settings better, and builds a
**simulator** to pick the right hybrid configuration cheaply instead of sweeping everything.

**Overall narrative flow:**
1. Show the *problem* with gsplit (where it loses to the baseline).
2. Propose *two hybrids* that could fix it (but which add configuration parameters).
3. Build and validate a *simulator* that predicts configurations cheaply (≈1 GPU, no full training).
4. Run the simulator on the problem cases to *predict the best Quiver/Split configuration*.
5. Show the *final results* from the real hybrid runs.

Each section lives in `sections/<NN_name>/` with two files:
- `claude_plan.md` — working plan: the argument, source material, tables/figures to fill, TODOs. (Not in the PDF.)
- `<name>.tex` — the LaTeX source, `\input`-ed by top-level `main.tex`.

**Writing conventions (global rules):**
- Every LaTeX paragraph opens with a one-line lead question/point in **blue** via `\paraq{...}`.
- Every section **ends with a conclusion** paragraph.
- Use **scratch tables with placeholder cells** (e.g. `\placeholder`) we can fill as results land.
- State an **expected conclusion** as a placeholder, so once real results arrive we can see where the expectation was wrong.
- Mark planned figures in *italic* as `\emph{[Expected diagram: ...]}`.

---

## Section 1 — Where gsplit underperforms (`01_gsplit_limitations`)
- **Key question:** In which settings does gsplit's performance gain over the baseline collapse?
- **Covers:** Briefly explain the two bad situations — (a) **deeper layers** → more per-layer
  shuffles → shuffle cost rises; (b) **larger hidden size** → larger feature volume per shuffle
  → shuffle cost rises. Then an **ablation** comparing **DGL, Quiver, and gsplit**, as **two
  tables (one per graph: products, papers100M)**, each sweeping a layer ablation and a hidden
  ablation for SAGE+GAT. Default **4×A100**; arxiv omitted as a toy graph.
- **Expectation:** gsplit does **worse than DGL** in these cases.
- **Conclusion:** there exist experimental settings where gsplit's performance is not good.
- **Source:** measured breakdown in `defense_latex/main.tex` (gsplit-only today — DGL/Quiver columns are scratch to fill); baselines in `experiment/nodepred/{dgl,quiver}_trainer.py`.

## Section 2 — Alternate hybrids (`02_alternate_hybrids`)
- **Key question:** What two hybrid schemes could recover performance in those bad settings?
- **Kept deliberately brief** — will gain clarity once more experimental results land.
- **2A — Hybrid data parallelism:** top layers run traditional **data parallelism**, bottom
  layers run **gsplit-style** split parallelism (e.g. DP in the top ~3 layers, split in the
  last ~2). Trigger the shuffle only from layer *k* onward.
- **2B — Hybrid pipeline parallelism:** two micro-batch streams run in parallel so one
  layer's **shuffle** overlaps with the other stream's **compute**.
- **Conclusion:** both hybrids can *potentially* fix the Section 1 problems, but they
  introduce **extra configuration parameters** (e.g. the split-point *k*, stream count).
  A brute-force sweep over all configs is too expensive — which motivates the simulator.
- **Source:** `plans/01_dp_plus_split.md`, `plans/02_pipeline_plus_split.md`, `plans/README.md`.

## Section 3 — The simulator and its fidelity (`03_simulator`)
- **Key question:** Can we predict each hybrid configuration's performance cheaply, without running full training?
- **Covers:** A simulator that predicts expected performance using **few resources (≈1 GPU)**
  and **without finishing full training**. Simulator + fidelity are **one section**. First a
  **DGL-baseline validation** note; then **fidelity demonstrated on papers100M, 4×A100, SAGE
  & GAT, L=3**, with a **within-10%** pass criterion; plus the nsys microprofile component
  cross-check (real data).
- **Source:** `defense_latex` §Simulator Fidelity + §Microprofile; `experiment/simulation/hybrid_dp_sim.py`.

## Section 4 — Simulator-predicted cases (`04_simulator_predicted_cases`)
- **Key question:** What does the validated simulator predict for the Section 1 bad cases?
- **Covers:** Run the simulator on the deep-layer / large-hidden settings (products, papers100M),
  sweeping the hybrid config space, to find the **best Quiver/Split performance** and the chosen
  split-point k* per case — the targets the real runs must hit.
- **Source:** `experiment/simulation/hybrid_dp_sim.py::simulate_optimal_k`, `plans/03_configurations_and_gpus.md`.

## Section 5 — Final results (`05_final_results`)
- **Key question:** With the simulator-chosen configs, do the hybrids actually beat gsplit/baseline at runtime?
- **Covers:** Real runtime results showing that **pipeline** and **data-parallel hybrid**,
  at the chosen configurations, recover performance on the Section 1 bad cases.
- **Status caveat:** both hybrid models **may not be built yet**; a **separate build loop**
  will implement them and fill these results. Tables are scratch/placeholder for now.
- **Note:** this section may later use a **bar chart** instead of a large table — not a concern yet.

---

## Top-level assembly
This chapter lives under `defense_latex/`. The **existing `defense_latex/main.tex`** is the
top-level document and is what builds — not a new file. The reorganization:
- `defense_latex/main.tex` — keep its preamble (geometry, booktabs, graphicx, amsmath,
  siunitx, hyperref), **add** `xcolor` + the `\paraq`/`\placeholder` macros, and replace the
  current inline chapter body with `\input{sections/01_gsplit_limitations/limitations}` …
  `\input{sections/05_final_results/final_results}` in order.
- `defense_latex/build.sh` — already present; builds `main.tex` with latexmk. No new build script.
- All source-material paths below are relative to the repo root (`/home/sandeep/dojo/dgl_groot`).
