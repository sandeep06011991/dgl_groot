# Chapter Outline — Hybrid Parallelism for gsplit

**Chapter premise.** gsplit (Spara) is *already built* and earlier chapters show it performs
decently in many settings. But our ablation studies reveal settings where its performance
gain over the baseline **drops significantly** — specifically at **deeper layers** (shuffle
cost grows) and at **larger hidden sizes** (shuffle volume grows). This final chapter
explores **hybrid parallelism** schemes to handle those settings better, and introduces a
**communication-centric roofline** that identifies the bottleneck of each layer under each
strategy, as the basis for choosing a configuration without sweeping everything.

**Overall narrative flow:**
1. Show the *problem* with gsplit (where it loses to the baseline).
2. Propose *two hybrids* that could fix it (but which add configuration parameters).
3. Introduce a *communication-centric roofline*: every layer of every strategy is one point;
   the region it falls in names its bottleneck (explanatory, not a timing model).
4. Draw the roofline for 3 representative settings of the sweep; explore presentations;
   design an algorithm from the patterns (to be filled).
5. Check *how close* the roofline's choice comes to the measured winners (to be filled).

Each section lives in `sections/<NN_name>/` with its LaTeX source `<name>.tex`, `\input`-ed by
top-level `main.tex` (Sections 1-2 also keep a `claude_plan.md` working plan, not in the PDF).

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

## Section 3 — A communication-centric roofline (`03_simulator/roofline.tex`)
- **Key question:** What bounds each layer under each strategy, and why does one strategy beat another at that layer?
- **Covers:** composition rules (concurrent -> max / min over roofs; sequential -> sum);
  per-node cost (dense F, gather S; constants α, β, γ; measured rates); axes
  (x = FLOPs per useful byte, y = useful bytes/s per GPU); three roofs (dense, gather — parallel
  slope −1 — and flat B_net) and the ridge x*(r); a layer as a point from busiest-GPU sampled
  counts (n, e, m, u); where DP / gsplit / pipeline / DP-hybrid land; **regions** (I comm-bound,
  II ridge band -> pipeline, III compute-bound gather/FLOP); assumptions list.
- **Not** a timing model and **no** measured check here.

## Section 4 — Roofline of representative settings (`04_simulator_predicted_cases/predicted_cases.tex`)
- **Key question:** What do the per-layer points look like for real settings?
- **Covers:** 3 settings (gsplit wins / interior-k hybrid wins / DGL wins) from the sweep;
  sampling-only per-layer counts; one roofline per setting with DP, gsplit, best-hybrid points
  (pipeline derived from gsplit); presentation variants; observations; algorithm (to be filled).
- Candidate settings are in a LaTeX comment in the .tex, to confirm.

## Section 5 — How close is the roofline? (`05_final_results/final_results.tex`)
- **Key question:** Does the roofline's choice match the measured winner, and what does a miss cost?
- **Covers:** agreement, regret = t(selected)/t(best) − 1, per-layer region consistency; table skeleton.

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
