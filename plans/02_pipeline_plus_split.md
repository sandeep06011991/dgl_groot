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

## Design (driver-only, with two known caveats)
The driver changes; the **model is not touched**.
1. **Micro-batching:** obtain two micro-batches A and B (either two dataloader steps, or
   one minibatch split in half). **Fully materialize both** (`GetBlocks` for each) before
   proceeding — see caveat 1.
2. **Two streams:** create `sA = torch.cuda.Stream()`, `sB`. Run A's forward under
   `with torch.cuda.stream(sA)` and B's under `sB`; because the C++ reads the *current*
   stream, the shuffle/all-to-all of A and the compute of B can overlap. Alternate
   per-stage (layer) for true pipelining, or at minimum per-micro-batch.
3. **Gradient accumulation:** wrap A's backward in `model.no_sync()` (DDP) so it does not
   all-reduce; let B's backward trigger the all-reduce of the *summed* grads; then one
   `optimizer.step()`; `zero_grad` once per pair. Net effect: grads of A+B are summed and
   reduced once.
4. Keep (or carefully place) the `dist.barrier()` so streams synchronize before the step.

### Caveat 1 — single global batch in the sampler
`SplitSampler` keeps one `_batch` (`split_sampler.h:63`), overwritten each
`sampleOneBatch`, and `getBlock/getScatteredArray` assert `_batch->_batch_id == batch_id`.
⇒ The driver must **fully fetch A's blocks+scattered arrays, then B's, before advancing**
the sampler further; it cannot interleave *sampling* of B between *block-fetches* of A.
Once fetched, the per-batch `ScatteredArray` objects are independent and safe to use
concurrently.

### Caveat 2 — single NCCL communicator
There is one `_nccl_comm` per rank (`split_sampler.h:72`). Two concurrent all-to-alls on
different streams share that communicator; NCCL can deadlock/serialize without care.
- **Minimal version (still useful):** overlap A's *compute* with B's *compute* and the
  non-NCCL parts; let the all-to-alls serialize (they still pipeline against compute).
- **Full overlap:** add a **second NCCL communicator** so A and B shuffle on independent
  comms — this is C++/init work (driver-side NCCL init), **not** a model change. Treat as
  a stretch goal after the minimal version is proven.

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

### Micro-test MT2 — two-stream overlap correctness
Run A on `sA` and B on `sB` (with the real `Shuffle`/all-to-all if built, else a torch
`all_to_all` stand-in), with proper events/sync.
- **Oracle:** the MT1 (single-stream) result.
- **Pass if** forward outputs and grads are identical to MT1 regardless of stream
  interleaving — i.e. streams change *timing only, not results*. Also assert no deadlock
  with the shared NCCL comm (and, in the stretch version, with a 2nd comm).
- Optionally measure wall-clock to confirm the comm/compute overlap actually helps
  (compare to MT1), mirroring `hybrid_pipe_sim.py`'s `pipeline_depth=2` prediction.

Tests under `tests/pipelines/test_pipe_split_mt*.py`; run on a **cheap 2-GPU box**.

## Implementation path (staged; after plans approved)
1. **MT1** (accumulation equivalence) — pure torch, no repo changes.
2. **MT2 minimal** (two streams, shared NCCL comm, comm serialized) on the toy model.
3. Port the pattern into `split_trainer.py` as a `--pipeline_depth=2` driver variant
   (micro-batch, two streams, `no_sync` accumulation, single step) — **model untouched**.
4. **Stretch:** add a 2nd NCCL communicator for true shuffle overlap; re-run MT2.
5. Only then: real SAGE/GAT, compare epoch time vs non-pipelined split (expect speedup
   where comm is the bottleneck — large hidden/depth).

## Risks / open questions
- **Sampler single-`_batch`** (caveat 1): the fetch-before-advance ordering must be
  enforced in the driver; otherwise `batch_id` asserts fire.
- **Single NCCL comm** (caveat 2): true overlap needs a 2nd communicator; the minimal
  version still pipelines comp/comp and comp/comm partially.
- **DDP + manual accumulation:** `no_sync` on all but the last micro-batch; confirm the
  reduction sums (not averages incorrectly) across the pair and across ranks.
- **Determinism:** atomic scatter-add in `shuffle_backward` (`array_scatter.cc`) can make
  grads order-sensitive at float tolerance; use atol, not exact equality.
- **Correctness-vs-timing:** streams must never change results — MT2 guards this.
