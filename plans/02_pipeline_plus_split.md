# Plan 2 — Pipeline + Split (2-stream micro-batch pipelining)

## Goal & intuition
Split each minibatch into **2 micro-batches** and run them **alternately on two CUDA
streams** through the *same* model, so that the **split all-to-all shuffle (comm) of one
micro-batch overlaps with the compute of the other**. The two backward passes
**accumulate gradients**, and a single optimizer step is taken per minibatch pair.

The split system's cost is dominated by the shuffle communication at each layer;
pipelining two micro-batches hides that comm behind compute. The cost-model spec already
assumes this exact design: `experiment/simulation/hybrid_pipe_sim.py` uses
`pipeline_depth = 2` and overlaps per-unit comm with the other unit's compute
(`max(comm[0][layer], compute[1][layer])`).

## How gsplit works today (what we extend)
- **Driver is one-batch-at-a-time** (`experiment/nodepred/split_trainer.py:83–101`): one
  `model(blocks, feat)` → one `cross_entropy` → `optimizer.zero_grad()` →
  `loss.backward()` → `optimizer.step()`, with a `dist.barrier()` mid-step (line 89). No
  streams, no micro-batching, no grad accumulation, no `no_sync`.
- **Grad reduction is DDP all-reduce** fused into `backward()` (`model = DDP(...)`, line 49).
- **Everything runs on the default CUDA stream.** All C++/NCCL
  (`all2all.cu`, `array_scatter.cc`, `split_sampler.h`) uses
  `runtime::getCurrentCUDAStream()` — i.e. it picks up whatever stream Python has current.
- **The shuffle is encapsulated in the model** (`Shuffle` per layer), so the driver never
  touches scatter arrays directly — good: pipelining is a driver concern.

## Design (driver-only; single NCCL comm, compute‖comm overlap)
The driver changes; the **model is not touched**. Key schedule insight (per review): we
overlap **one stream's compute with the other stream's communication**, staggered by one
stage — so at any instant **exactly one all-to-all is in flight**, and a **single NCCL
communicator is sufficient** (no second communicator, no concurrent-comm deadlock).

1. **Sampling is done in sequence, not overlapped.** Sampling gives no overlap benefit,
   so sample A fully, then B fully (`GetBlocks` for each), up front — this also sidesteps
   the sampler's single-`_batch` constraint (caveat 1). Pipelining applies only to the
   forward/backward compute vs. the shuffle comm.
2. **Two streams, staggered by one stage.** `sA`, `sB`. Run the per-layer loop so that
   while micro-batch A runs the **conv (compute)** of layer *i* on `sA`, micro-batch B
   runs the **shuffle (all-to-all comm)** of layer *i* on `sB`; next step A shuffles layer
   *i+1* while B computes layer *i*. Because the C++ reads the *current* CUDA stream
   (`getCurrentCUDAStream`), wrapping each op in `with torch.cuda.stream(...)` is enough —
   no model change. At every step only one stream is communicating → one all-to-all at a
   time → the single `_nccl_comm` is safe.
3. **Gradient accumulation (single accumulator):** wrap A's backward in `model.no_sync()`
   (DDP) so it does not all-reduce; let B's backward trigger the all-reduce of the summed
   grads; one `optimizer.step()`; `zero_grad` once per pair. Grads of A+B are summed and
   reduced once.
4. Use CUDA events to order the stagger (A's layer-*i* conv waits on B's layer-*i* shuffle
   only where there is a true data dependence); a `dist.barrier()` before the optimizer
   step.

### Caveat 1 — single global batch in the sampler (handled by sequential sampling)
`SplitSampler` keeps one `_batch` (`split_sampler.h:63`), overwritten each
`sampleOneBatch`, with `getBlock/getScatteredArray` asserting `_batch->_batch_id == batch_id`.
Because we **sample A and B sequentially and fully materialize their blocks + scattered
arrays before any pipelined compute** (item 1), this is a non-issue — the per-batch
`ScatteredArray` objects are independent once fetched.

### Single NCCL communicator is enough (why no 2nd comm / accumulator)
Earlier draft worried about a second NCCL communicator for concurrent all-to-alls. With
the **compute‖comm stagger** there is never more than one all-to-all in flight (A computes
while B communicates and vice-versa), so the existing single `_nccl_comm` suffices and no
second communicator is needed. The only accumulation is the ordinary gradient sum over the
two micro-batches (item 3).

## Micro-tests first (the deliverable before touching the driver)
Same philosophy as Plan 1: prove the mechanism on a toy problem, not the GNN.

### Toy problem
- **Graph:** n-clique `K_n` (shared with Plan 1).
- **Toy op / "model":** a tiny shared-weight message-passing layer stack (shared `W`),
  fully differentiable, analytically reproducible.
- Two micro-batches = two disjoint seed shards of the clique.

### Micro-test MT1 — gradient accumulation equivalence (no streams)
Run micro-batch A then B through the toy model, accumulate grads (A under `no_sync`, B
reduces), one step.
- **Oracle:** a single backward over the *concatenated* A∪B batch (standard training).
- **Pass if** accumulated `W.grad` equals the concatenated-batch `W.grad` (atol 1e-5) for
  world_size ∈ {1,2}. This proves the accumulation/`no_sync` semantics are correct —
  independent of streams.

### Micro-test MT2 — staggered two-stream overlap correctness
Sample A and B sequentially (fully materialize both), then run the **staggered schedule**:
A's layer-*i* conv on `sA` concurrent with B's layer-*i* shuffle on `sB`, alternating, on
a **single** NCCL comm (with the real `Shuffle`/all-to-all if built, else a torch
`all_to_all` stand-in), using CUDA events for the cross-stream dependencies.
- **Oracle:** the MT1 (single-stream) result.
- **Pass if** forward outputs and grads are identical to MT1 regardless of stream
  interleaving — streams change *timing only, not results* — and there is **no deadlock**
  on the single `_nccl_comm` (confirming the one-all-to-all-at-a-time stagger).
- Optionally measure wall-clock to confirm the compute‖comm overlap actually helps vs MT1,
  mirroring `hybrid_pipe_sim.py`'s `pipeline_depth=2` prediction.

Tests under `tests/pipelines/test_pipe_split_mt*.py`; run on a **cheap 2-GPU box**.

## Implementation path (staged; after plans approved)
1. **MT1** (accumulation equivalence) — pure torch, no repo changes.
2. **MT2** (staggered two streams, single NCCL comm, compute‖comm) on the toy model.
3. Port the pattern into `split_trainer.py` as a `--pipeline_depth=2` driver variant:
   sample A then B up front, run the staggered compute‖comm loop, `no_sync` accumulation,
   single step — **model untouched**.
4. Real SAGE/GAT: compare epoch time vs non-pipelined split (expect speedup where comm is
   a large share — large hidden/depth, and on GPUs where comm≈compute).

## Risks / open questions
- **Stagger scheduling:** correctly expressing "A-compute(i) ‖ B-comm(i)" with CUDA
  streams + events so the data dependencies hold and only one all-to-all is ever in
  flight (so the single `_nccl_comm` never sees concurrent ops).
- **DDP + manual accumulation:** `no_sync` on the first micro-batch, reduce on the second;
  confirm the reduction sums (not averages incorrectly) across the pair and across ranks.
- **Determinism:** atomic scatter-add in `shuffle_backward` (`array_scatter.cc`) can make
  grads order-sensitive at float tolerance; use atol, not exact equality.
- **Correctness-vs-timing:** streams must never change results — MT2 guards this.
- **Overlap payoff depends on the GPU:** the benefit is largest when comm and compute are
  comparable in magnitude; on a fast-compute/slow-comm GPU comm dominates (less to hide),
  on fast-comm/slow-compute comm is already cheap. (Config-dependence is the separate
  GPU-spectrum discussion, paused.)
