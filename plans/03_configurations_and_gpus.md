# Plan 3 — Configurations & GPU spectrum: when each strategy wins

## The core trade-off
The best parallelism strategy is a function of the **compute cost vs. inter-GPU
communication cost** on the target hardware. The split shuffle trades *redundant
computation* for *communication*; so:

- **Fast compute + slow/expensive inter-GPU comm → redundant computation (DP) wins.**
  Recomputing a node's embedding locally on each GPU is cheaper than shuffling it. This is
  the regime where **Plan 1's DP layers** (no shuffle) pay off — push `k` up (more DP).
- **Fast inter-GPU comm + (relatively) slow compute → gsplit / split wins.**
  Shuffling to avoid redundant recompute is cheap, so split from layer 0 (`k=0`) wins.
- **Both high and comparable → pipeline (Plan 2) wins.**
  When comm ≈ compute, overlapping one stream's compute with the other's communication
  hides the most (ideal ≈ `max(comm,compute)` instead of `comm+compute`).

So the three mechanisms aren't competitors — each owns a region of the (compute,
bandwidth) plane, and the **framework should pick per (GPU, graph, hidden, depth)**.

## RunPod GPU spectrum (from `runpodctl gpu list`)
Interconnect/compute figures below are **nominal vendor specs to confirm empirically**
(via the probes in the last section), not measured this session. "Inter-GPU BW" is the
2-GPU link that the shuffle rides.

| Archetype | Example GPU(s) | Inter-GPU link | ~BF16 TFLOPS | $/hr (secure) | Favors |
|---|---|---|---|---|---|
| **A: high compute / slow comm** | RTX 4090 (24G), RTX 5090 (32G), L40S, RTX 6000 Ada, A40/A6000 | **PCIe only** (~25–64 GB/s), no NVLink | 165–360+ | 0.59–1.19 | **DP / redundant (Plan 1, large k)** |
| **B: fast comm / moderate compute** | **A100 SXM** (80G), A100 SXM 40G | **NVLink** ~600 GB/s (NV12×25) | ~312 | 1.79 | **gsplit / split (k=0)** |
| **C: both high (comparable)** | **H100 SXM** (80G), **H200 SXM** (141G) | **NVLink** ~900 GB/s | ~990 | 3.99–5.29 | **pipeline (Plan 2)** |
| **D: middle / PCIe datacenter** | A100 PCIe (80G), H200 NVL | PCIe / partial NVLink | ~312–990 | 1.79–3.79 | sweeps / boundary cases |

Notes:
- **RTX 4090** is the cleanest "compute-cheap, comm-expensive" box: high FP throughput, no
  NVLink (multi-GPU over PCIe), cheap ($0.89), and the only **High**-stock card — ideal to
  demonstrate DP/redundant beating split.
- **A100-SXM** (what we've been on) is the "fast-comm" box where split shines.
- **H100/H200-SXM** have both huge compute and NVLink → the regime where pipeline's
  compute‖comm overlap gives the biggest win.
- **MI350X / B300** exist but MI350X is AMD/ROCm (not the CUDA+NCCL stack); treat as out of
  scope unless the framework is ported.

## Does the simulator capture this? (your products/pipeline hypothesis)
**Mostly yes, with one enhancement.** The two cost models already have the right shape:
- `simulation/hybrid_dp_sim.py` sweeps `k` (DP layers vs split layers) and prices the
  all-to-all comm per split layer — it will show the DP region win when comm is expensive.
- `simulation/hybrid_pipe_sim.py` uses `pipeline_depth=2` and overlaps per-stage comm with
  the other unit's compute (`max(comm, compute)`) — it will show pipeline win **when comm ≈
  compute**, which is exactly why you expect **products** (dense 100-dim features, large
  hidden → substantial compute *and* substantial shuffle volume) to benefit from pipeline.
  So the structure captures your hypothesis.

What it needs to be trustworthy across GPUs:
1. **Compute** is already GPU-correct: the model *measures* layer compute on the actual
   GPU via CUDA events (validated within ~2–5% for SAGE at large hidden after the Plan-0
   fix). Run the sim *on* the target GPU and compute is right for free.
2. **Communication** currently uses the constants `NVLINK_BW` / `PCIE_BW`. These must be set
   to the **target GPU's** inter-GPU bandwidth. Enhancement:
   - make them a config/CLI knob (not hardcoded), and
   - add a **GPU↔GPU NCCL all-to-all bandwidth probe** (analogous to the host→GPU
     feature-load probe we already have in `profile_model.py`, which measured ~25 GB/s) to
     auto-calibrate per pod. On an NVLink box it should read ~hundreds of GB/s; on a
     PCIe-only box ~25–50 GB/s — which is what flips the DP-vs-split decision.
3. With (1)+(2), running `hybrid_dp_sim` (k-sweep) + `hybrid_pipe_sim` (pipeline) on each
   archetype GPU yields the **predicted winning strategy per (GPU, graph, hidden, depth)**
   — the catalog below — cheaply (short probe, no full training).

Caveat: to compare GPUs *without renting each*, you could scale compute by a TFLOPS ratio
and comm by a bandwidth ratio, but that's less reliable than a 2-minute probe on each
archetype; prefer the probe.

## "Interesting situations" matrix (what to actually run)
Target: a table of (GPU archetype × graph × hidden/depth) → measured winning strategy,
used both to validate the simulator's prediction and as the paper's config study.

| GPU | Graph | Hidden | Expected winner | Why |
|---|---|---|---|---|
| RTX 4090 ×2 (PCIe) | products | 256–512 | **DP / large k** | comm expensive → recompute locally |
| RTX 4090 ×2 (PCIe) | papers100M | 256 | DP / large k | same, more pronounced (big shuffle) |
| A100-SXM ×2 (NVLink) | products | 256 | **split (k=0)** | comm cheap → shuffle beats recompute |
| A100-SXM ×2 | arxiv | 256 | split | small graph, comm cheap |
| H100-SXM ×2 (NVLink) | products | 512 | **pipeline** | comm≈compute → overlap wins |
| H100-SXM ×2 | papers100M | 256–512 | pipeline / split | large shuffle to overlap |
| A100-PCIe ×2 | products | 256 | boundary (DP↔split) | moderate both — tests the crossover |

For each cell: (a) simulator prediction (cheap), then (b) a real micro/short run to confirm
— prioritizing the cheap archetypes (RTX 4090, A100-SXM) for the crossover story.

## Next steps (after Plans 1 & 2 micro-tests pass)
1. Add the configurable/auto-probed bandwidth to the simulator (enhancement 2 above).
2. Run the simulator's k-sweep + pipeline model on 2–3 archetype GPUs (RTX 4090, A100-SXM,
   H100-SXM) for products/arxiv/papers → fill the predicted-winner matrix.
3. Spot-check the predictions with short real runs (cheap GPUs first) → the config study.
This is what lets the framework *choose* DP / split / pipeline per deployment.
