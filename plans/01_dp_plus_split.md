# Plan 1 — Data-Parallelism + Split (hybrid "DP-then-split")

## Goal & intuition
Run the first **k** GNN layers as **pure data parallelism** (each GPU processes its own
micro-batch of seeds, sampling/computing locally with **no cross-GPU communication**),
then switch to **split** (partition-aware shuffle) for the remaining layers. At the
switch point the per-GPU micro-batches are conceptually concatenated into the global
batch, and from there each GPU owns a partition and features are shuffled to owners.

Why: the split all-to-all shuffle is only worth its cost at the layers where the
cross-partition fan-in is large; the earliest layers (closest to seeds, smallest
frontier) can stay replicated/DP and avoid communication entirely. `k` is the
split-from-layer: `k=0` = today's full split, `k=L` = pure DP.

The authoritative cost-model spec already exists: `experiment/simulation/hybrid_dp_sim.py::simulate_optimal_k`
("k=0 full split … k=L data parallelism"; microbatches concatenated at layer k).

## How gsplit works today (what we extend)
- **Per-layer shuffle in the model.** `python/dgl/dev/splitmodel.py` `SRC_TO_DEST.forward`
  applies `Shuffle.apply(block.scattered_src, x, rank, world_size)` **once per layer,
  before the conv** (lines 59–60). `Shuffle` (lines 16–31) calls the C++
  `_CAPI_Split_ScatterForward` / `_CAPI_Split_ScatterBackward`.
- **The hook already exists but is disabled.** The model stores `num_dp` but
  `forward` asserts `assert(self.num_dp == 0)` (line 56) and there is a `Todo` (line 57)
  to handle `num_dp != 0`. So every layer shuffles today.
- **Sampler side.** `SplitSampler::sampleOneBatch` (`src/spara/split_sampler.h:149–204`)
  computes a fresh `ScatteredArray` per layer from `IndexSelect(_partition_map, unique_src)`,
  and already has a DP branch `if (layer < _num_dp)` — but it is stubbed:
  `CHECK_EQ(_num_dp, 0) << "does not support hybrid sampling for now"` (line 171).
  `getScatteredArray` already offsets by `layer - _num_dp` (line 282).
- **`_num_dp` is never set.** `_CAPI_Split_SetNumDP` exists (`split_sampler.cc:124–129`)
  but nothing in Python calls it; `Config.num_redundant_layer` (config.py:27) flows only
  into the model's `num_dp`, not the sampler.
- **DDP already wraps the split model** (`split_trainer.py:49`), so parameter-gradient
  all-reduce is handled; "data parallel for the first k layers" needs no new reduction.

## Design (semantics)
- Layers `0 .. k-1` (**DP**): each rank samples the neighborhoods of *its own seed shard*
  and computes the conv on **local, replicated** features — **no `Shuffle`**. (Model
  params are replicated and kept in sync by DDP.)
- Layer `k` (**switch**): the global batch is the union of all ranks' shards; from here
  each node is owned by its partition (`partition_map`), and features are shuffled to
  owners before each conv — exactly today's split path, but starting at layer `k`.
- Layers `k .. L-1` (**split**): unchanged from today.

## Micro-tests first (the deliverable before touching the real model)
**Do not modify the GNN model/sampler first.** Capture the intent on a toy problem and
prove correctness, then graduate.

### Toy problem
- **Graph:** an **n-clique** `K_n` (e.g. n=16), every node adjacent to every other.
  Under any `partition_map`, every node has neighbors in every partition → maximal
  shuffle stress and trivial to reason about.
- **Toy op per layer:** message passing with a **shared weight** `W` (small, e.g.
  `d×d`): `h'[v] = W · mean_{u∈N(v)∪{v}} h[u]`. No SAGE/GAT, no dropout. Fully
  differentiable and analytically reproducible.
- **Features:** small fixed `d` (e.g. 4), deterministic init; `W` shared/replicated.

### Micro-test MT1 — pure-PyTorch DP-then-split logic (cheap, no gsplit build)
Implements the mechanism with `torch.distributed` only (`all_to_all_single` for the
shuffle), on 2 GPUs (or gloo/CPU for logic), parameterized by `k`:
- layers `<k`: each rank holds the full `K_n` features (replicated) and computes the toy
  op on its **seed shard** locally;
- at `k`: concat shards (all-gather the shard outputs into the global node set);
- layers `≥k`: partition nodes by `partition_map`, `all_to_all` the needed neighbor
  features to the owning rank, aggregate, apply `W`.
- **Oracle:** a single-process run of the identical toy op over all seeds for all L
  layers (no parallelism). **Pass if** forward outputs match (atol 1e-5) **and** `W.grad`
  matches (atol 1e-5) for **every k in 0..L** and world_size ∈ {1,2,(4)}.
This proves the DP↔split switch and gradient flow are correct independent of the C++.

### Micro-test MT2 — real-CAPI shuffle at layer k (validates the actual code path)
Uses gsplit's real scatter on the n-clique:
- build each layer's frontier on `K_n`, call the `Scatter` builder + `_CAPI_Split_ScatterForward/Backward`
  (via the `Shuffle` autograd fn) for layers `≥k`, and skip it for layers `<k`;
- exercises the new `SetNumDP(k)` + the sampler DP branch + `getScatteredArray(layer-_num_dp)`.
- **Oracle:** same single-process reference. Confirms the real all-to-all + the
  `num_dp` plumbing produce identical results to MT1/reference.

Both micro-tests go under `tests/` (new, e.g. `tests/pipelines/test_dp_split_mt*.py`),
run on a **cheap 2-GPU box**, finish in seconds.

## Implementation path (staged; after plans approved)
1. **MT1** (pure torch) — no repo changes; proves the algorithm + grads.
2. **Enable the seams** for MT2 / real model (the three hard-blocked sites from the
   exploration):
   - `splitmodel.py`: remove `assert(self.num_dp==0)`; guard shuffle with `if i >= self.num_dp:`
     (and run the conv on local features for `i < num_dp`).
   - Python loader: add a `SetNumDP` wrapper and call `SetNumDP(config.num_redundant_layer)`
     in `SplitGraphLoader.__init__`; make `GetBlocks`/`_CAPI_GetBlockScatteredSrc` tolerate
     DP layers having no `scattered_src`.
   - `split_sampler.h`: implement the `layer < _num_dp` branch (sample locally, build no
     `ScatteredArray`); keep the `layer - _num_dp` offset consistent.
3. **MT2** against the rebuilt extension on the n-clique.
4. Only then: run the real SAGE/GAT split model with `num_redundant_layer=k>0` and
   compare epoch metrics/accuracy to `k=0`.

## Risks / open questions
- **DP-layer sampling semantics:** for `layer < k`, does each rank sample only its seed
  shard's neighbors (true DP), and are those frontiers consistent with the global batch
  formed at layer k? The sampler DP branch must produce frontiers that concatenate
  correctly at the switch.
- **Reindexing at the switch:** node-id spaces for the DP (local) layers vs the split
  (global/partitioned) layers must line up at layer k.
- **Accuracy parity:** `k>0` must match `k=0` accuracy (same math, different comm), which
  MT1/MT2 verify on the toy op before trusting the real model.
- DDP already reduces param grads; confirm no double-counting when early layers are DP.
