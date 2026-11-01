# Long-running job — dissertation evaluation plan

Goal of the whole job: show **where gsplit (the `split` system) underperforms**,
explain **why** (shuffle cost vs. redundant compute), show our **simulator can
predict the right configuration**, and then **prove** that the predicted
configuration wins. Four sections, split into steps (Section 1 → **1a/1b/1c**),
each run by a **slave agent**, coordinated by one **master agent** (below). The
default order is 1a → 1b → 1c → 3 → 4; Section 2 is independent and may run any
time. This is **not a single linear job** — the master branches on results.

---

## Orchestration — master / slave flow

One long-lived **master agent** owns the plan; each step runs in a fresh
**slave agent** the master spawns. The master is the only writer of shared state
(serializes `sweep_all.csv` commits so slaves never conflict) and the only one
that decides what runs next.

**Master loop:**
1. Spawn the next step's slave with a tight brief (its section below + the
   artifacts it depends on) and wait for its committed output + handoff note.
2. **Inspect the result against expectation**, then branch:
   - *On track* → spawn the next step in the default order.
   - **Anomaly** (measured ≠ simulator estimate, a baseline unexpectedly wins/
     loses, a non-monotonic `k` curve, OOM where none expected, quiver-PCIe hang)
     → **spawn a focused analysis slave** to explain it *before* proceeding
     (e.g. re-profile that one cell, check the partition/cache, inspect the
     simulator term that disagrees). Record the explanation (see the results-
     explanation rule), then decide: accept + annotate, re-run, or adjust scope.
   - *Blocked* (e.g. quiver replicated-cache won't stabilize on PCIe) → dispatch
     a "drop + document the caveat" task rather than stalling the whole run.
3. Keep a short running **ledger** (`defense_latex/results/run_ledger.md`):
   per step — slave id, what it produced, verdict (on-track / anomaly+action /
   blocked), and the next decision. This is the master's memory across the run.

**Why master/slave:** Section 3→4 is inherently conditional on Section 1's
results, and anomalies need side-quests that shouldn't derail the main chain.
Slaves start cold (cheap, isolated); the master holds continuity.

---

## Execution contract (read first — applies to every section)

**Environment.** RunPod only; never build/run/train on this WSL box. Every
pod command runs inside the conda env (`conda activate /home/groot_env`). Logs
are backed up to B2 (`gsplit-data/groot/logs`) because pods are ephemeral.

**Vocabulary → code (so agents don't re-derive it):**
- **gsplit / split** = `--system=split`.
- **data-split** = `--system=split --num_redundant_layer=k` with `k>0` (the new
  configurable DP-then-split; `k` = leading data-parallel layers, `k=0` is full
  split, `k=len(fanouts)` is pure DP). **`k` is the "additional config variable."**
- **pipeline-split** = `--system=gsplit_pipepar`. **Already landed on
  `final_dissertation`** (commit `0cd49acb`; the `hybrid-pipeline-parallelism`
  branch commit is a byte-identical duplicate). No merge needed — the master just
  verifies `0cd49acb` is in `git log` before starting.
- **baselines** = `--system=dgl`, `--system=quiver`, `--system=p3`.
- **NVLink setting** = run as-is on the A100-SXM pod.
- **PCIe setting** = *same pod*, rerun with `NCCL_P2P_DISABLE=1` (forces comm onto
  the PCIe path). This is the agreed emulation — no second pod. **Caution:**
  changing NVLink availability may require a **C++ rebuild** (`build.sh`), since
  topology/cache assumptions can be baked in at build time — budget one rebuild
  between the NVLink and PCIe passes, not just an env-var flip.

**Graphs:** `products` and `papers100M`, both with **real OGB features**
(predownloaded — provisioning is not the bottleneck). `feat.npy` is produced by
`get_npgraph` from the OGB download; enable real features for papers100M with a
one-line add to the real-feature list in `load_data`
(`experiment/utils/dataloading.py`) — today only `products` loads real feat.
Accuracy is therefore meaningful for both graphs. Size the pod for papers100M's
~57 GB feature array (disk + host RAM for UVA pinning); see Risk 6.

**Partition maps — B2-first, generate only if absent.** Provision with
`RESTORE_B2=1` (`experiment/script/provision_data.sh`) so maps and topology come
from B2 when present — this is the common, fast, deterministic path and avoids
METIS's nondeterministic segfaults (esp. papers100M, whose maps are already in
B2). Generation (`get_partition.py`/METIS, retry-on-segfault) is the **fallback
only for an absent map**; a freshly built map is pushed back to B2 (`PUSH_MAPS=1`,
small) so the next pod reuses it. Never repartition a map that B2 already has.

**Shared results file (the single source of truth):**
`defense_latex/results/sweep_all.csv` — one row per run, holding **all** of
Section 1 (and the Section 4 confirmation rows). Commit protocol so parallel/
sequential agents never conflict:
1. `pandas.read_csv` → append new rows → `sort_values` on the full Config key →
   `to_csv` → `git add` → commit. Sorting before *and* after keeps diffs clean.
2. Required Config columns (normalize every per-run CSV into this superset —
   this is point 2 of the old plan, "headers may change"):
   `hardware` (`nvlink|pcie_emu`), `graph`, `system`, `model`, `depth`,
   `fanout`, `k` (=`num_redundant_layer`; 0 for non-split), `cache_size`,
   `cache_frac`, `batch_size`, `world_size`, `hid_size`, `sample_mode`.
   Keep every existing metric column (epoch/sample/load/forward/backward/shuffle
   times, test_acc, peak_mem, …).

**Explain every result (cross-cutting).** No number stands alone. For each
result or table cell, the owning slave writes a **one-line mechanistic reason**
for the winner (e.g. "split loses here: shuffle on PCIe > the redundant compute
it saves at hidden=512"). For any **anomaly** — measured deviates from the
simulator estimate beyond a stated tolerance (e.g. >15%), a counter-intuitive
ordering, a non-monotonic `k` curve, or an unexpected OOM/hang — the slave must
give a **reasoned explanation** (root cause, not just "noise"), and flag it to
the master, which may spawn a dedicated analysis slave (see Orchestration). These
explanations land in the section prose and in `run_ledger.md`; they are what make
the chapter defensible, not just the raw table.

**Per-section rule:** each agent ends by committing its artifacts with a clear
message, and leaves a one-paragraph handoff note (what it produced, where, and
what the next section should read).

---

## Reviewability — which sections are too big for one agent/review

Natural fault line: **code-plumbing** (review the diff) ≠ **long compute runs**
(review the produced CSV, not a diff) ≠ **writing** (review the prose). A section
that does all three at once is hard to review as one unit.

- **Section 1 — split into 1a/1b/1c (done below).** It changes several code
  files, triggers a rebuild, runs a multi-hour sweep on two pods, *and* builds the
  table + prose — so it is broken along the diff/data/prose fault line:
  **1a** plumbing (code diff), **1b** the sweep (review the CSV), **1c** table +
  "where gsplit loses" writeup (prose).
- **Section 2 — one agent** (small, self-contained, no code).
- **Section 3 — one agent**, but if simulator-code changes become necessary (the
  read-only step turns into a code diff), split that fix off as **3a** so the
  writeup/validation (3b) stays reviewable separately.
- **Section 4 — one agent** (short compute + table regen + prose).

---

## Risks (high) — what the master watches throughout

1. **Simulator can't predict accurately (the #1 risk).** Mitigation: the nsys /
   CUDA-event gap investigation + recalibration, **Step 3a** below. (The earlier
   constant inconsistency — `hybrid_pipe_sim.py` shadowing `PCIE_BW=16e9` vs
   `common.py`'s `25e9`, and the "1ms"/1µs `BARRIER_COST` comment — is **already
   fixed**: the hybrid sims now inherit the single source of truth in
   `simulation/common.py`. 3a recalibrates that unified value against nsys.)
2. **No predicted config wins anywhere — acceptable, not a failure.** If across
   all cells no predicted gsplit config beats the baselines, the honest conclusion
   is *"the baseline is best in this regime,"* a legitimate result to report — not
   a reason to stall or retune. The master still surfaces it after Section 3
   (before spending Section 4 compute on cells that can't win), but as
   information, not an alarm.
3. **PCIe emulation may be unfaithful.** `NCCL_P2P_DISABLE=1` might not reproduce
   true PCIe bandwidth; if so the *measurement* is wrong and the simulator gets
   blamed. **Preflight (cheap, do it):** measure effective bus BW under
   P2P-disable (nsys or an `all2all` microbench) and confirm it ≈ the `PCIE_BW`
   constant; if not, fix the constant, not the sim's structure.
4. **Quiver replicated-cache on PCIe** — may hang / need a rebuild. Fallback:
   drop quiver-PCIe with a documented caveat (already in 1a/1b).
5. **pipeline-split is under-characterized — measure before concluding.** It has
   only been run on small graphs (arxiv); the earlier "no net speedup on
   A100-SXM" note is from that setting, not a large graph. Do **not** pre-judge
   that overlap-overhead > benefit. Run it across the full sweep (both graphs,
   both hardware) first, then conclude from the data — whether it helps at scale
   is a thing to *find out*, possibly a result in itself.
6. **papers100M pod sizing** — real features (below) mean the pod must hold the
   ~57 GB `feat.npy` (+ the OGB predownload during regen) on disk, with enough
   host RAM to pin it for UVA. Not a blocker given a right-sized pod; size it
   before 1b and confirm `get_npgraph` produced `feat.npy` and the B2 maps restore.

Risks 1 and 3 decide whether the simulator story stands; 2 is a reporting stance
(baseline-best is fine); 4–6 are containable with the notes above.

---

## Section 1 — Performance sweep: where does gsplit underperform?

Run as **three steps in order** (1a → 1b → 1c), each its own agent, split along
the diff / data / prose fault line so each is reviewable on its own.

**Hypothesis (shared by all three steps):** gsplit loses to dgl/quiver in some
regimes, and the two hardware settings (NVLink vs PCIe) move the crossover
because they have different shuffle cost.

### Step 1a — Plumbing (code only, no long runs)

**Owner:** agent 1a. **Depends on:** `0cd49acb` present in `git log` (pipeline-
split already landed); nothing else.

**Steps:**
1. **CSV schema:** add `hardware` and `cache_frac` to the row written by
   `write_to_csv` (`experiment/utils/utils.py`), the `Config` class, and
   `experiment/utils/args.py` (`k` already exists as `num_redundant_layer`).
2. **Sweep script:** edit `experiment/script/main.sh` → or a new
   `experiment/script/sweep.sh` (reuse, don't rewrite) covering:
   - systems: `split` (k grid `k∈{0,1,2}`), `dgl`, `quiver`, `p3`, `gsplit_pipepar`.
   - `quiver` at **25% and 10%** cache (record `cache_frac`); on PCIe the cache
     must be **replicated** (not partitioned) or it hangs — that pass needs a
     rebuild. (Ties into the contract's rebuild note.)
   - models `sage`, `gat`; fanout `15,15,15`; `world_size` per pod GPUs.
   - each run writes `hardware` and `k`.
3. **Table generator:** `defense_latex/gen_results_table.py` — pandas reads the
   CSV, emits a `\input{}`-able `.tex`, **highlighting the superior system per
   (graph, hardware, config) in red** (`\textcolor{red}`/`\cellcolor`). Test it
   against a tiny **synthetic** CSV — no real data yet.

**Files touched:** `experiment/utils/utils.py`, the `Config` class,
`experiment/utils/args.py`, `experiment/script/sweep.sh` (or `main.sh`),
`experiment/nodepred/quiver_trainer.py` (replicated-cache path),
`experiment/utils/dataloading.py` (one-line: add `papers100M` to the real-feature
list in `load_data`), `defense_latex/gen_results_table.py` (new).

**Done when:** schema + script + generator pass review as a plain code diff, and
the generator emits a valid `.tex` from the synthetic CSV. **No sweep run yet.**

### Step 1b — The sweep (compute)

**Owner:** agent 1b. **Depends on:** 1a merged.

**Steps:**
1. **Pods:** two A100-SXM pods in parallel (one per graph: `products`,
   `papers100M`) to stay near the ~2 h budget. Each pod runs NVLink, then
   `NCCL_P2P_DISABLE=1` (PCIe) — expect a **rebuild** between the two hardware
   passes and for quiver's replicated cache.
2. **Run** 1a's sweep on both settings, both graphs; back logs up to B2.
3. **Consolidate** all per-run CSVs → `defense_latex/results/sweep_all.csv`
   (schema in the contract), pandas-sorted (sort before *and* after append).
4. **Commit** `sweep_all.csv` (+ the sweep script if it was tweaked on the pod).

**Files touched:** `defense_latex/results/sweep_all.csv` (new data). No code diff
expected if 1a was complete.

**Done when:** `sweep_all.csv` has NVLink+PCIe × both graphs × all systems; review
= sanity-check the CSV (row count, no unexpected OOM/empty rows), not a diff.

### Step 1c — Table + "where gsplit loses" writeup

**Owner:** agent 1c. **Depends on:** `sweep_all.csv` from 1b.

**Steps:**
1. Run `gen_results_table.py` → `defense_latex/tables/sweep_all.tex` (whole
   table; fine if large — the chapter shows a curated subset later).
2. **Write** the experimental-setup prose (hardware, systems, cache settings,
   graphs; state the quiver replicated-cache-on-PCIe caveat).
3. **Answer the hypothesis:** list the concrete (graph, hardware, config) cells
   where gsplit loses — the typed input Section 3 consumes — **each with its
   one-line mechanistic reason** (per the explain-every-result rule), and flag any
   anomaly to the master.
4. **Commit** the `.tex` table, the prose, and the losing-cells list.

**Files touched:** `defense_latex/tables/sweep_all.tex` (new), the prose section
`defense_latex/sections/01_gsplit_limitations/limitations.tex` (**already
scaffolded**), a `losing_cells` list (in the prose or a small
`defense_latex/results/losing_cells.csv`).

**Done when:** the `.tex` renders in the PDF and the losing-cells list exists for
Section 3 to act on.

---

## Section 2 — Diagrams + the cost-tradeoff explanation (independent)

**Owner:** agent 2. **Depends on:** nothing (may run any time).

**Steps:**
1. **Two diagrams** (TikZ or SVG→PDF, consistent with the chapter style):
   - **data-split (DP-then-split):** first `k` layers data-parallel (no
     cross-GPU shuffle), remaining layers partition-aware split.
   - **pipeline-split:** 2 micro-batches on 2 CUDA streams, one stream's split
     shuffle overlapping the other's compute.
2. **Explanation (prose, into the chapter):** why **shuffling cost** and
   **redundant-compute cost** trade off against each other; how **different
   machines (NVLink vs PCIe) and different hidden sizes** change the shuffle cost
   and therefore the best strategy; and the central **configuration challenge** —
   which strategy to pick where, plus the extra knob of **how many DP layers
   `k`** to use.
3. **Commit** diagrams + prose.

**Files likely touched (all under `defense_latex/`):**
- two figures, e.g. `defense_latex/figures/data_split.tex` and
  `figures/pipeline_split.tex` (TikZ) or `.svg`→`.pdf` assets.
- prose into `defense_latex/sections/02_alternate_hybrids/hybrids.tex`
  (**already scaffolded**); draw on `plans/01_dp_plus_split.md` and
  `plans/02_pipeline_plus_split.md`.
- no code files.

**Done when:** both diagrams compile into the PDF and the tradeoff section reads
cleanly and sets up Section 3's "the simulator picks the config."

---

## Section 3 — Simulator fidelity and prediction

**Owner:** agent 3. **Depends on:** Section 1's `sweep_all.csv` and its list of
gsplit-losing cells.

**Steps:**
1. **Write up the simulator (2–3 paragraphs):** we estimate all costs from a
   **single GPU**. State the assumptions honestly — compute cost approximated as
   a **linear function** (drop this claim if the code doesn't actually do it —
   verify first), data-movement cost model, and that we **don't hold
   gradients/optimizer state in memory** when extrapolating. Explain how, despite
   these approximations, the simulator lets us make the **right configuration
   choice** without running every setting.
   **Constants are already unified** to `simulation/common.py` (the hybrid sims
   inherit it; the old `PCIE_BW` 16/25 mismatch is fixed) — just verify before
   validating.
2. **Validate** the simulator against measured gsplit from Section 1: run
   `experiment/script/run_simulation.sh` (uses `get_best_strategy.py`,
   `--nvlink` toggles NVLink vs PCIe) for **2–5 main settings per graph** (not
   all) and compare predicted vs. measured. Report the agreement, and for any
   cell outside tolerance (e.g. >15%) give the **reasoned cause** and flag it to
   the master — this is the fidelity story, state it honestly.
3. **If the gap exceeds tolerance → Step 3a** (nsys investigation + recalibrate)
   *before* predicting anything. Prediction on an uncalibrated simulator is worse
   than no prediction.
4. **Predict fixes:** for each (graph, hardware, config) cell where Section 1
   showed gsplit losing, run the simulator to **predict the configuration that
   should win** — the right `k` (data-split) and/or whether pipeline-split
   overlap should be enabled.
5. **Commit** the simulator writeup, the validation table/numbers, and a small
   `predicted_configs` artifact (CSV/JSON: for each losing cell → the predicted
   winning config) that Section 4 consumes.

### Step 3a — nsys gap investigation & recalibration (triggered on fidelity gap)

**Owner:** a dedicated slave (per Reviewability: simulator *code* changes split
off here so the writeup/validation stays reviewable). **Depends on:** Section 3
step 2 flagging a cell outside tolerance.

**Procedure — find the term that's wrong, reset it, re-validate:**
1. **Decompose** the failing cell into the simulator's stages (sample /
   feature-load / compute fwd+bwd / shuffle-comm) and compare each predicted
   stage to measured. The gap lives in one or two stages — find which.
2. **Measure ground truth with nsys / CUDA events:** run
   `experiment/script/run_profile.sh <graph> <depth> "<hiddens>" "<models>"`
   (drives `profile_model.py`; emits per-stage CUDA-event times + an nsys
   kernel breakdown: gemm/aggregate/reduce). For the comm stage, profile the
   `all2all` under both NVLink and `NCCL_P2P_DISABLE=1` to get the *real*
   effective bandwidth (this also settles Risk 3).
3. **Reset the constant/model** in `simulation/common.py` from the measurement:
   - comm: refit `NVLINK_BW` / `PCIE_BW` to the measured effective bandwidth.
   - compute: check the linear-cost assumption against the nsys gemm/aggregate
     scaling vs hidden size; refit the slope/intercept (or replace the linear
     term if it doesn't hold — say so in the writeup).
   - memory/`BARRIER_COST`: fix the 1 µs-vs-1 ms discrepancy if it matters.
   Change the model **structurally only if** a constant refit can't close the gap
   — and document any structural change as a limitation.
4. **Re-run Section 3 step 2** on the same validation cells; confirm the gap is
   within tolerance. If a refit for one hardware breaks another, that's itself a
   finding — report it, don't silently tune to one point.
5. **Commit** the constant/model change (code) **separately** from the prose, and
   record in `run_ledger.md`: which stage was off, by how much, the old→new
   value, and the post-fix agreement.

**Files touched:** `experiment/simulation/common.py` (+ `hybrid_dp_sim.py` /
`hybrid_pipe_sim.py` if constants are still duplicated there), possibly
`experiment/profile_model.py` (minor). Code diff — review separately from 3's prose.

**Done when:** every validation cell is within tolerance with the recalibrated
constants, or any residual gap is explained as a documented model limitation.

**Files likely touched:**
- `defense_latex/results/predicted_configs.{csv,json}` (new) and a validation
  numbers file, e.g. `defense_latex/results/sim_vs_measured.csv` (new).
- prose into `defense_latex/sections/03_simulator/simulator.tex` (fidelity) and
  `defense_latex/sections/04_simulator_predicted_cases/predicted_cases.tex`
  (predicted fixes) — **both already scaffolded**.
- `experiment/script/run_simulation.sh` — minor edits at most.
- **Ideally read-only** on `experiment/get_best_strategy.py` /
  `experiment/simulation/hybrid_dp_sim.py`; touch them only if validation exposes
  a bug (flag it loudly if so — it changes Section 1's story).

**Done when:** simulator-vs-measured agreement is reported for the validation
settings, and `predicted_configs` names a concrete config for every losing cell.

---

## Section 4 — Final evaluation in the predicted configurations

**Owner:** agent 4. **Depends on:** Section 3's `predicted_configs`.

**Steps:**
1. For each losing cell, **measure** the simulator-predicted config (correct `k`
   and/or pipeline-split) on the matching hardware setting; append rows to
   `sweep_all.csv` (`system`/`k` reflecting the predicted config).
2. **Compare** predicted-config result vs. the baselines that beat plain gsplit
   in Section 1. **A predicted gsplit config winning in ≥1 position is the
   hoped-for result** — but a cell where no gsplit config wins is a *legitimate
   outcome* ("baseline is best here"), reported honestly, not a failure to
   engineer around (Risk 2).
3. **Explain each outcome + fallback (make this explicit, don't silently pass):**
   for every cell, state *why* the predicted config won or lost (which cost term
   moved). If no predicted config wins a cell, report the closest result, the
   measured-vs-predicted gap, and the root cause — and flag it to the master,
   which may spawn an analysis slave before the chapter over- or under-claims.
4. **Regenerate** the `.tex` table (now including the predicted-config rows) and
   write the results prose. **Commit.**

**Files likely touched:**
- appends to `defense_latex/results/sweep_all.csv` (data only).
- re-runs `defense_latex/gen_results_table.py` → regenerated
  `defense_latex/tables/sweep_all.tex` (no code change if Section 1's generator
  is general).
- prose into `defense_latex/sections/05_final_results/final_results.tex`
  (**already scaffolded**).
- optional `experiment/script/run_predicted.sh` (new) to drive the predicted runs.

**Done when:** every losing cell has a measured predicted-config result with its
mechanistic reason, the table reflects it, and both outcomes are reported
honestly — a predicted win is the hoped-for result; a baseline-best cell is an
acceptable one (Risk 2).
