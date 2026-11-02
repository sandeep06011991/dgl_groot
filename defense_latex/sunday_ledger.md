# Sunday ledger — finish Section 1 (where does gsplit underperform?)

**Status:** PLANNING → code pushed, pod not yet up. Master/slave run. The
**master** owns this file (sole writer) and `sweep_all.csv`; each phase runs in a
fresh **slave**.

Supersedes the broad Section-1 scope in `long_running.md` for this job. Goal is
narrow and defensible: a minimal 4-config comparison on **training time** that
pinpoints the regime where gsplit (`--system=split`, k=0) loses — **on both
graphs, stressed to OOM**, because a conclusion is only valid once the system is
sufficiently stressed. The problem is already known from the **V100** results;
this reproduces and characterizes it on a **larger (A100) GPU**.

---

## 1. Objective

Characterize where **gsplit does not perform**, on **both papers100M and products**,
on **both hardware settings (NVLink, then PCIe)**, comparing **training time only**
(per-epoch `duration (s)`, lower = better). Both graphs are first-class.

Per graph, per hardware, two axes each climbed **until OOM** (or the 900 s/cell cap):
1. **Hidden size** — exponentially increasing (`256→512→1024→2048→…`) until OOM.
2. **Depth** — at `hidden = H*/2` (H* = the OOM-ceiling hidden from axis 1, per
   hardware/model), increasing (`2→3→4→…`) until OOM.

Both axes run both models, **gat first, then sage** (gat's attention memory
OOMs earlier → finds the ceiling fast; sage confirms the cheaper regime).

**Hardware (user, 2026-11-02):** do **NVLink first, then PCIe** (`NCCL_P2P_DISABLE=1`
emulation, drops `--nvlink`) for each graph. PCIe is the expensive-shuffle regime
where gsplit is expected to lose hardest. No C++ rebuild needed between passes (split
comm is runtime NCCL; `--nvlink` only flips quiver's cache policy + the label).

**Order (user):** finish papers100M completely (NVLink + PCIe, both axes) and
**write its results** before repeating the whole thing for products.

**Stop = OOM (or 900 s cap), not "gsplit underperforms."** We keep climbing past
the crossover and record it along the way — a conclusion needs the full stressed
curve, not just the first losing rung.

---

## 2. The four situations (the only columns in every table)

| # | system | cache | pairing role |
|---|--------|-------|--------------|
| 1 | `dgl`    | **0%** (`--cache_size=0G`) | uncached baseline |
| 2 | `gsplit` (`split`, k=0) | **0%** (`--cache_size=0G`) | uncached gsplit |
| 3 | `quiver` | **25%** (`--cache_frac=0.25`) | cached baseline |
| 4 | `gsplit` (`split`, k=0) | **25%** (needs plumbing — Phase 0) | cached gsplit |

**Two head-to-heads:**
- **uncached:** (2) vs (1) — isolates gsplit's sampling+shuffle vs dgl's replicated graph, no cache noise.
- **cached:** (4) vs (3) — gsplit vs quiver, both holding the whole graph (see below).

**Why 25% is the *fair* cached setting (user, 2026-11-02):** on a 4-GPU run,
quiver's 25%-per-GPU cache is **partitioned (disjoint) across the clique**, so
`4 × 25% = 100%` of the feature tensor is resident across the GPUs and reached over
NVLink p2p — i.e. **the whole graph is partitioned across the GPUs**, exactly like
gsplit, where each GPU owns and caches its ~25% partition. So (4 vs 3) is
apples-to-apples: both hold the full feature set distributed. **10% is a later
*reduced-cache* stress point** (each GPU caches 10% → ~40% aggregate residency →
more misses), not a fairness correction.

**"gsplit underperforms"** ≡ (2) slower than (1), OR (4) slower than (3), by
**≥ 5%** — a milestone we **record** on the climb, **not** a stop condition.

---

## 3. Decisions / assumptions (master — correct me if wrong)

- **gsplit 0% cache** = `--cache_size=0G` (yields 0 cached nodes; note: the arg
  default `0GB` crashes `int(...)` — must be `0G`).
- **gsplit 25% cache** = add `cache_frac` support to `split_trainer.py`, mirroring
  quiver: `num_ids_cached = min(cache_frac * total_nodes, local_ids)`. Same
  fraction-of-full denominator as quiver so the two match (§2). (Phase-0 plumbing.)
- **Metric** = per-epoch training time, column `duration (s)`. Accuracy not compared
  (perf sweep → `--random_feat` to skip the 57 GB feat load; it still allocates a
  full-size pinned tensor so feature-movement + cache sizing stay realistic — P7).
- **Hidden ladder** `256 → 512 → 1024 → 2048 → 4096 → …` (gat hidden stays
  divisible by `num_heads=4`; the ladder is).
- **Depth ladder** `2 → 3 → 4 → 5 → …`, fanout `15` per layer.
- **Per-cell cap = 900 s (15 min)** — records a timeout row; a cell that hits it is
  an effective ceiling (impractically large).
- **Stop a ladder** (don't climb further) when the gsplit cell **OOMs** or **hits the
  900 s cap**. Larger/deeper is monotonically worse, so that ceiling ends the climb.
  Underperform crossovers are recorded, never a stop — we stress to OOM.
- **H\*** = the OOM/time ceiling hidden on axis 1. Depth axis (axis 2) uses `H*/2`
  so the depth ladder has headroom to reach its own OOM.
- Order: **papers100M {NVLink then PCIe, both axes} → write papers100M results →
  products {NVLink then PCIe, both axes} → write products results.** NVLink before
  PCIe within each graph; each hardware stressed to OOM before concluding.

---

## 4. Phases

### Phase 0 — prereqs + plumbing (slave 0)  ▢

**Pod (live):** `mbkd71iymth7sa` / `groot-sunday` — 4×A100-SXM-80GB, 500 GB disk,
**$7.16/hr**, created 2026-11-02 08:58 UTC. SSH `root@154.54.102.40 -p 18294`
(key `~/.runpod/ssh/runpodctl-ssh-key`). NVLink confirmed (NV12 all pairs). B2 creds:
`~/.config/groot/b2.env` → `rclone config create gsplit-b2 b2 account $B2_KEY_ID key $B2_APP_KEY`.

- [x] **Push `final_dissertation` to GitHub** (user decision) so a fresh pod bootstraps with the 1a code + the quiver/torchmetrics env-setup fixes. (torchmetrics is already in `environment_setup.sh:136-139` — no change needed.)
- [x] Acquire 4×A100-SXM pod, 500 GB disk — `mbkd71iymth7sa` (see Pod box). NVLink confirmed.
- [x] Bootstrap DONE (09:25 UTC, ~27 min): torch 2.4.0+cu124, CUDA ok, dgl built release. B2 `gsplit-b2` remote created; `dgl.dev` (Spara layer) imports under `PYTHONPATH=repo/python`. Pod RAM ~2.1 TB (ample for papers100M's 57 GB random feat).
- [~] Provisioning both graphs detached (pid 24041, log `/root/overnight_logs/provision.log`, watcher `bezhb4pyu`). papers100M from B2 (no `feat.npy` needed — `--random_feat`, `get_feat_dim` hardcodes 128); products from OGB+METIS. Need only topology + idx + `_w4_ndst_efreq_xbal` map per graph.
- [x] **Plumbing:** `cache_frac` branch added to `split_trainer.py` (config 4), pushed (`454bb565`).
- [x] Write `experiment/script/sunday_sweep.sh` (§5) — pushed (`454bb565`).
- [x] **Smoke test PASSED** (papers100M/NVLink/gat/h256/d2, 2ep): 4 distinct rows, cache_frac −1.0 vs 0.25, split@25% alloc 2060 MB > split@0% 1077 MB (cache_frac path works), GPUs cleaned to 0 MiB. First data point: gsplit loses to both baselines even here (split 19.4 vs dgl 15.1; split@25% 20.1 vs quiver@25% 17.4).
- **Done when:** ✅ 4-config smoke produced 4 rows with sane `duration (s)`; both graphs provisioned. **PHASE 0 COMPLETE.**

### Phase 1 — papers100M, **NVLink** (hidden then depth) (slave 1)  🔄 depth RUNNING
- [x] **hidden ladder COMPLETE** (log `phase1_papers_nvlink.log`, CSV `sunday_papers100M_nvlink_hidden.csv`, 44 data rows incl. 8 `timeout` sentinels). Both ceilings were the **900 s cap** (not hard OOM): **gat H\*=4096, sage H\*=8192.** Per-epoch `duration (s)`, cols = split@0 / dgl@0 / quiver@25% / split@25%:
  - **gat** @d3: 256→27.5/26.1/32.7/25.1 · 512→27.4/25.9/34.0/26.0 · 1024→35.0/38.3/45.3/32.0 · 2048→60.3/75.5/83.6/59.2 · **4096→timeout(all)**.
  - **sage** @d3: 256→16.0/12.6/25.0/13.7 · 512→15.5/14.1/24.8/13.9 · 1024→18.5/15.0/25.7/17.7 · 2048→29.3/25.3/35.3/27.9 · 4096→62.2/61.3/71.8/61.4 · **8192→timeout(all)**.
  - **Finding (clean):** on **sage**, `dgl@0` beats `split@0` at *every* hidden (12.6<16.0 … 61.3<62.2) → gsplit's uncached partition path is slower than replicated DGL. gsplit only wins the *cached* pair (split@25% < quiver@25% everywhere). On **gat** the uncached gap closes/reverses at large hidden (split 60.3 < dgl 75.5 @2048) — attention compute dominates, hiding the shuffle cost.
- [x] **Fixed `record_timeout.py`**: `Profiler.content()`/`avg()` crashed on string sentinels (`"timeout"`/`"oom"`) → *no ceiling rows were written* during the run. Patched `profiler.py` (avg passes non-numeric through), synced, backfilled the 8 hidden-ceiling rows.
- [~] **depth ladder RUNNING** (pid 101015, `run_depth_nvlink.sh`, log `phase1b_papers_nvlink_depth.log`, watcher `bhf19nb0h`): two chained invocations — gat@`FIXED_HID=2048`, sage@`FIXED_HID=4096` (=H\*/2), `DEPTH_LADDER="3 4 5 6 7 8"`, early-break on gsplit timeout. CSV `sunday_papers100M_nvlink_depth.csv`.
- **NOTE:** launches must `export WORKSPACE_DIR=/root` (env.sh defaults to the missing Unity `/scratch3` path; data lives at `/root/graph`). First bad depth launch (pid 100493) crashed on the dead path, killed, relaunched clean.
- **Done when:** papers100M NVLink stressed to the cap on both axes, both models; crossover + ceiling recorded with one-line reasons.

### Phase 2 — papers100M, **PCIe** (hidden then depth) (slave 2)  ▢
- [ ] Same as Phase 1 with `HARDWARE=pcie` (`NCCL_P2P_DISABLE=1`, drops `--nvlink`; no rebuild). Expensive-shuffle regime → expect gsplit to lose hardest here.
- **Done when:** papers100M PCIe stressed to OOM on both axes, both models.

### Phase 3 — **write papers100M results** (writeup slave)  ▢
- [ ] Consolidate papers100M rows (NVLink+PCIe) into `sweep_all.csv` (schema preserved, pandas sort before+after); back logs to B2.
- [ ] Regenerate the table via `gen_results_table.py`; **update** `sections/01_gsplit_limitations/limitations.tex` with the papers100M 4-config training-time story, the hidden/depth stress-to-OOM curves, the losing cells **each with its one-line mechanistic reason**, and the V100→A100 framing. Refresh `losing_cells.csv`.
- [ ] Compile the PDF (`latexmk` on WSL — CPU-only, allowed).
- **Done when:** papers100M results are in the table + prose and the PDF builds. *Efficiency:* this writeup slave (WSL, no pod) can run **in parallel with Phase 4** so the pod never idles.

### Phase 4 — products, **NVLink** (hidden then depth) (slave 3)  ▢
- [ ] Repeat Phase 1 for **products** (different graph → its own crossover/ceiling).

### Phase 5 — products, **PCIe** (hidden then depth) (slave 4)  ▢
- [ ] Repeat Phase 2 for **products**.

### Phase 6 — **write products results** + finalize (writeup slave)  ▢
- [ ] Consolidate products rows; regenerate the table; extend `limitations.tex` with products; refresh `losing_cells.csv`; rebuild the PDF.
- **Done when:** both graphs × both hardware are in the table + prose, the PDF builds, and the losing-cells list is complete. Writeup runs as its **own agent** (prose review ≠ compute review).

---

## 5. The script — `experiment/script/sunday_sweep.sh` (write in Phase 0)

A focused driver (do **not** rewrite `sweep.sh`; new file). Contract:
- Runs exactly the **4 configs** of §2 per (model, hidden, depth) cell.
- `MODE=hidden` → sweep `HID_LADDER` at fixed `DEPTH`. `MODE=depth` → sweep
  `DEPTH_LADDER` at fixed `HID`.
- Ascending order; per-cell `timeout -k 30 900`; record timeout rows (reuse `record_timeout.py`).
- **Early-break** the ladder when the `split` (gsplit) cell OOMs (nonzero rc) or
  times out (124/137) — larger is only worse. (Underperform is logged, not a break.)
- **pkill on EVERY cell exit, including error-out** (user, P3): `pkill -9 -f
  '[t]rain_main.py'; pkill -9 -f '[m]ultiprocessing.spawn'; sleep 3` then an
  `nvidia-smi` free-mem check before the next cell — a crashed/OOM'd cell's workers
  otherwise poison the next.
- `HARDWARE=nvlink` (default, `--nvlink`) or `HARDWARE=pcie` (`NCCL_P2P_DISABLE=1`,
  drops `--nvlink`). `--random_feat`, `batch_size=1024`, `world_size=4`,
  `num_redundant_layer=0`, fanout `15`/layer.
- Writes `duration (s)` etc. to a per-pass CSV the master consolidates.

---

## 6. Master/slave orchestration

- **Master** = this session: owns `sunday_ledger.md` + `sweep_all.csv`, spawns one
  slave per phase with a tight brief, inspects each result, decides H* / ceilings,
  branches. Phase 4 (LaTeX) is its own writeup slave — separate from the compute slaves.
- **Slave** = fresh agent per phase (cold, cheap, isolated); runs its ladder on the
  pod, commits its per-pass CSV + logs, leaves a one-paragraph handoff.
- **Anomaly** (non-monotonic curve, gsplit wins where V100 said it should lose, OOM
  where none expected) → master spawns a focused analysis slave before proceeding.
- **If a phase hangs** (pod unresponsive / workers stuck past the 900 s cap + cleanup):
  **reboot the RunPod instance** — re-provision from B2, resume at the last unrecorded
  rung. The CSV + this ledger are the durable trail.

**Blocker + cost protocol (user, 2026-11-02):**
- On a **sudden blocker** needing a decision, **email** sandeep06011991@gmail.com
  (Gmail MCP tools *are* loaded this session — the master can send). Clear subject;
  state the blocker, what's paused, and the options.
- **No reply within 30 min on a blocker → stop/kill the pod** (`runpodctl pod stop`/
  terminate) to save money; record state here to resume cleanly. **Never hold an
  active (billing) pod idle > 30 min** — hard cap, mirrored into global config.

---

## 7. Potential problems (pre-mortem — read before Phase 0)

Ranked by time-waste risk. Last session's killers were **provisioning friction +
scope creep**, not compute — so Phase 0 absorbs provisioning, and scope is now
fixed at 4 configs / 2 axes / 2 graphs.

- **P1 — provisioning is the real time-sink (highest).** papers100M needs a **~500 GB
  container disk** (feat ~57 GB, prepared npy ~70 GB — a default 60 GB pod can't hold
  it, [[papers100m-run-plan]]). Last run lost time to **B2 auth failing silently** and
  **quiver not being built**. *Mitigation:* `final_dissertation` is now pushed (quiver
  + torchmetrics in `environment_setup.sh`); stage B2 creds first; Phase 0 gate = a
  green 4-row smoke run before any ladder.
- **P2 — "25%" cache is the *fair* setting, not a bias (resolved — user).** `4 × 25%`
  = the whole graph partitioned across the clique, matching gsplit's partition
  residency (§2). 10% is a later reduced-cache stress point, not a correction. No
  action beyond reporting both cache levels.
- **P3 — OOM poisons the next cell.** `sweep.sh` only cleans workers on *timeout*, not
  on an OOM crash (trainers `os._exit` → rc≠0, no row). The ladders are *built* to hit
  OOM, so this will bite. *Mitigation:* `sunday_sweep.sh` pkills + `nvidia-smi`-checks
  after **every** cell including error-out (§5).
- **P4 — the loss is expected (V100 → A100 reproduction).** We already saw gsplit lose
  on V100; this reproduces it on a larger GPU under enough stress. So push to OOM to
  find *where* on A100 it crosses — don't under-stress and conclude "no problem."
  (A non-monotonic or no-crossover curve is still worth an analysis slave, not a fudge.)
- **P5 — distinguish OOM-ceiling from underperform-crossover.** A rung where *all four*
  systems OOM is a hardware ceiling, not a gsplit result. Only "gsplit slower than its
  pair while the baseline still runs" is the finding. GAT OOMing first is expected.
- **P6 — gotchas ([[groot-experiment-code-gotchas]]):** torchmetrics is now installed
  by `environment_setup.sh` (was the old failure mode). `get_weight_fast.py` always
  exits non-zero — don't gate on it. `os._exit` means an OOM records no row → the
  ladder treats a missing/nonzero gsplit cell as the ceiling.
- **P7 — random_feat is safe (not a problem).** `gen_rand_feat` makes a full-size
  pinned tensor → realistic UVA gather + cache sizing; only accuracy is meaningless,
  which we don't compare.

## 8. Progress log (master appends; newest last)

- 2026-11-02 — Plan drafted. Confirmed cache plumbing gap (split has no `cache_frac`),
  `sweep.sh` 900 s cap, 0-cache must be `0G`.
- 2026-11-02 — User round 2: gat before sage; email on blocker + 30-min idle-pod cost
  cap (added to global config). Pre-mortem added; papers100M-NVLink confirmed unrun.
- 2026-11-02 — User round 3: (1) push `final_dissertation` — DONE; (2) quiver 25% =
  whole graph partitioned across 4 GPUs → 25% is the *fair* cached setting, P2
  downgraded; (3) pkill on error-out in the script; (4) **both graphs** must be stressed
  **to OOM** — stop = OOM, not underperform; (5) torchmetrics already in env-setup; (6)
  LaTeX update is a **dedicated writeup slave** (Phase 4). Ready for the go to acquire
  the pod (Phase 0 starts billing).
- 2026-11-02 — User: Go Ahead. Phase 0 started. Pushed branch (`454bb565`) with the
  cache_frac plumbing + `sunday_sweep.sh`. Acquiring 4×A100-SXM / 500 GB pod
  (`groot-sunday`, bg). Existing EXITED pods unfit (2-GPU / small disk). Plumbing +
  script done; smoke test + provisioning pending pod allocation.
- 2026-11-02 — User: add PCIe after NVLink per graph; write papers100M results before
  products. Added `HARDWARE=nvlink|pcie` knob to `sunday_sweep.sh`; restructured phases
  to 1–6 (papers NVLink → papers PCIe → write papers → products NVLink → products PCIe
  → write products). Writeup slave can overlap next graph's compute to avoid pod idle.
- 2026-11-02 — **Phase 1 hidden ladder DONE** (papers100M/NVLink). Ceilings at the 900 s
  cap: gat H\*=4096, sage H\*=8192. Headline reproduces the V100 loss on A100: sage
  `dgl@0` < `split@0` at every hidden (uncached gsplit slower than replicated DGL); gsplit
  wins only the cached pair. Fixed a `record_timeout.py`/`profiler.py` crash that had
  dropped all ceiling rows; backfilled. Launched NVLink **depth** ladder at H\*/2
  (gat@2048, sage@4096). Caught + fixed a data-path bug (must `export WORKSPACE_DIR=/root`).
