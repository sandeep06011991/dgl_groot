# Hybrid Parallelism Feature Spec

## Overview

This document specifies two new parallelism strategies for Spara:

1. **Hybrid Split+DDP** — bottom `k` GNN layers run with split parallelism (partition-aware NCCL scatter), top `n-k` layers run with data parallelism. A sparse hidden-state pull at the transition boundary replaces an all-gather.
2. **Pipeline Double Buffering** — within a single forward pass, two microbatches are in flight simultaneously. The Shuffle (NCCL all-to-all) of one microbatch overlaps the GNN computation of the other, using Python-level CUDA stream double buffering.

A **simulator** is also specified to predict the optimal `k` for Hybrid Split+DDP given a graph and model configuration.

---

## Feature 1: Hybrid Split+DDP

### Motivation

In the current split system, every GNN layer calls `Shuffle.apply()` which issues an NCCL scatter/gather to exchange features across partition boundaries. The communication volume at each layer is:

```
num_cross_partition_src_nodes × dim_at_layer
```

For the first layer, `dim_at_layer = feat_dim` (e.g., 100 for `products`). For subsequent layers, `dim_at_layer = hidden_dim` (e.g., 256). However, the number of cross-partition nodes grows with fanout depth. Beyond a certain layer `k`, it may be cheaper to abandon split parallelism and switch to DDP, accepting a one-time sparse pull of hidden states from remote GPUs instead of issuing a full NCCL scatter per layer.

### Design

#### Configuration

Add `num_split_layers: int` to `SampleConfig` and `Config`:

```python
# SampleConfig (python/dgl/dev/util.py)
num_split_layers: int = -1  # -1 means all layers use split (current behavior)
```

```python
# Config (experiment/utils/config.py)
self.num_split_layers = -1
```

CLI argument in `experiment/utils/args.py`:
```
--num_split_layers  int  Number of bottom layers using split parallelism (-1 = all layers)
```

#### Forward Pass (`SRC_TO_DEST.forward()`)

Current behavior (all layers use Shuffle):
```
for each layer i:
    x = Shuffle.apply(block.scattered_src, x, rank, world_size)
    x = layer(block, x)
```

New behavior:
```
for each layer i:
    if i < num_split_layers:
        x = Shuffle.apply(block.scattered_src, x, rank, world_size)
        x = layer(block, x)
    elif i == num_split_layers:
        x = SparseHiddenPull(block, x, rank, world_size)  # transition
        x = layer(block, x)
    else:
        x = layer(block, x)  # pure local DDP compute
```

#### Transition: Sparse Hidden State Pull (`SparseHiddenPull`)

At layer `k` (the transition boundary), each GPU holds hidden states only for its locally-owned src nodes. The top DDP layers need hidden states for all src nodes in the local block, including those owned by remote GPUs.

Instead of all-gather (which would pull all hidden states), each GPU:
1. Identifies which src nodes in its local block are missing (owned by other partitions).
2. Sends requests to the owning GPUs for only those node IDs.
3. Receives the corresponding hidden state vectors.

This is a sparse point-to-point exchange, similar in structure to `Shuffle`/`_CAPI_Split_ScatterForward` but driven by the set of missing node IDs rather than the full partition scatter.

**New Python class** in `python/dgl/dev/splitmodel.py`:
```python
class SparseHiddenPull(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block, hidden, partition_map, rank, world_size):
        # 1. Compute missing_ids = src nodes in block not owned by rank
        # 2. All-to-all exchange of missing_ids and their hidden vectors
        # 3. Stitch result into full src hidden tensor
        ...

    @staticmethod
    def backward(ctx, grads):
        # reverse the sparse pull: route gradients back to owning GPU
        ...
```

The actual exchange can reuse the existing NCCL all-to-all infrastructure (`_CAPI_Split_ScatterForward`) or be implemented as a new C++ API call if the sparse pattern requires it.

#### Backward Pass

- Layers below the transition (`i < k`): backward through `Shuffle` as today — `_CAPI_Split_ScatterBackward`.
- Transition layer (`i == k`): `SparseHiddenPull.backward()` routes gradients back to the owning GPU for the pulled nodes.
- Layers above the transition (`i > k`): standard DDP backward (AllReduce of gradients via PyTorch DDP).

#### Key Invariant

`num_split_layers = num_layers` (i.e., `k = n`) reproduces the current split system exactly. `num_split_layers = 1` uses split only for the first layer and DDP for all remaining layers.

---

## Feature 2: Pipeline Double Buffering

### Motivation

In the current forward pass, `Shuffle.apply()` (NCCL all-to-all) and `layer()` (GPU compute) execute sequentially. For large graphs over PCIe (non-NVLink), NCCL latency is a significant fraction of per-step time. By overlapping the Shuffle of one microbatch with the compute of another, we can hide this communication latency.

### Design

#### High-Level Flow

Two microbatches, A and B, are sampled upfront. At each GNN layer, the forward pass proceeds as:

```
Layer i:
  [comm stream]  Shuffle microbatch B, layer i      ──────────────────┐
  [main stream]  layer(block_A[i], x_A)  ────────────────────────────┤
                                                                       ↓
                                   sync streams

  [comm stream]  Shuffle microbatch A, layer i+1    ──────────────────┐
  [main stream]  layer(block_B[i], x_B)  ────────────────────────────┤
                                                                       ↓
                                   sync streams
  ...
```

Batch size per microbatch = `global_batch_size / 2`. Effective throughput is the same as a full batch, but communication is overlapped.

#### Python-Level Implementation

No C++ changes required. Uses two `torch.cuda.Stream` objects:

```python
main_stream = torch.cuda.current_stream()
comm_stream = torch.cuda.Stream()
```

**Sampling:** Both microbatches are sampled sequentially before the forward pass begins:
```python
batch_id_A = SampleBatch(seeds_A, replace)
blocks_A = GetBlocks(batch_id_A, reindex=True, layers=num_layers)
feat_A = GetFeature(batch_id_A)

batch_id_B = SampleBatch(seeds_B, replace)
blocks_B = GetBlocks(batch_id_B, reindex=True, layers=num_layers)
feat_B = GetFeature(batch_id_B)
```

**Forward pass (pipelined):**
```python
x_A, x_B = feat_A, feat_B

for i, layer in enumerate(layers):
    # Issue Shuffle for B on comm_stream (async)
    with torch.cuda.stream(comm_stream):
        x_B_ready = async_shuffle(blocks_B[i].scattered_src, x_B, rank, world_size)

    # Compute layer for A on main_stream
    x_A = layer(blocks_A[i], x_A)

    # Sync: ensure B's Shuffle is done before its compute
    main_stream.wait_stream(comm_stream)
    x_B = layer(blocks_B[i], x_B_ready)

    # Swap roles for next layer
    x_A, x_B = x_B, x_A
    blocks_A, blocks_B = blocks_B, blocks_A
```

> **Note:** The swap means each microbatch alternates between the "fast" path (overlapped comm) and the "slow" path (compute while other's comm happens). Over many layers both microbatches are treated symmetrically.

#### Async Shuffle

`Shuffle.apply()` currently uses synchronous NCCL calls. To support overlap, an `async_shuffle` variant is needed that launches NCCL on the comm stream and returns a future/tensor that becomes valid once the comm stream is synced.

```python
class AsyncShuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scattered_array, feat, rank, world_size, stream):
        with torch.cuda.stream(stream):
            out = from_dgl_nd(_CAPI_Split_ScatterForward(scattered_array, to_dgl_nd(feat), rank, world_size))
        ctx.save_for_backward(...)
        return out
```

The existing `_CAPI_Split_ScatterForward` must be called with NCCL communicators associated with the correct CUDA stream. This may require a small C++ change to pass the stream handle to the NCCL all-to-all call.

#### Training Loop Changes

The training loop in `split_trainer.py` needs to be updated to:
1. Draw two seed batches per step from `IdxLoader`.
2. Call the pipelined forward pass instead of the single-batch forward.
3. Accumulate loss over both microbatches before backward.

#### Memory Overhead

Two microbatches of blocks and features are live simultaneously. Estimated peak overhead: `2 × batch_size × fanout_product × hidden_dim × 4 bytes`. For `batch_size=512` (half of 1024), `fanout=(15,15,15)`, `hidden=256`: roughly 2× the current working set for intermediate activations.

---

## Simulator: Predicting Optimal `k`

### Goal

Given a graph, model config, and hardware, predict the value of `num_split_layers` **before training begins**, using an offline analytical simulation over graph statistics. No profiling steps during training are required.

### Cost Model

For each candidate `k ∈ {1, 2, ..., num_layers}`, estimate total per-step time:

```
T(k) = T_sample + T_split_layers(k) + T_transition(k) + T_ddp_layers(k)
```

**T_sample:** Constant — sampling cost does not depend on `k`.

**T_split_layers(k):** Sum over layers `i = 0..k-1`:
```
T_split(i) = comm(i) + compute(i)

comm(i)    = cross_partition_src_nodes(i) × dim(i) / bandwidth
compute(i) = edges(i) × hidden_dim / gpu_throughput

dim(0) = feat_dim
dim(i>0) = hidden_dim
```

**T_transition(k):** Cost of the sparse hidden pull at boundary:
```
T_transition(k) = missing_nodes(k) × hidden_dim / bandwidth
```
where `missing_nodes(k)` = number of src nodes at layer `k` owned by remote partitions.

**T_ddp_layers(k):** Sum over layers `i = k+1..num_layers-1`:
```
T_ddp(i) = compute(i) + allreduce_grad(i)

allreduce_grad(i) = param_count(i) × 4 bytes / bandwidth
```
No per-node communication; only parameter gradient AllReduce.

### Graph Statistics Required

The simulator needs per-layer estimates of:
- `cross_partition_src_nodes(i)` — number of src nodes crossing partition boundaries at layer `i`
- `missing_nodes(i)` — subset of those that are missing on the local GPU
- `edges(i)` — number of edges in sampled block at layer `i`

These can be estimated using the existing **simulation infrastructure** (`experiment/simulate_main.py`, `simulation/simulate.py`, `CntSampler`). The `simulate` function already profiles per-layer sampling statistics; it needs to be extended to output per-layer cross-partition node counts.

### Simulator Interface

```python
# experiment/simulate_main.py or a new experiment/hybrid_sim.py

def simulate_optimal_k(cfg: Config) -> int:
    """
    Runs offline simulation over the graph to estimate T(k) for all k.
    Returns the k that minimizes estimated per-step time.
    """
    ...
```

Inputs from `cfg`:
- `graph_name`, `data_dir`, `partition_type` — for loading graph + partition map
- `fanouts`, `batch_size`, `world_size`
- `feat_dim` (loaded from graph), `hid_size`
- `bandwidth` (NVLink vs PCIe, detected or passed via `nvlink` flag)

Output: integer `k ∈ {1..num_layers}`.

### Validation

After predicting `k` analytically, run `train_main.py` with `--num_split_layers=k` and adjacent values (`k-1`, `k+1`) and log throughput to `logs/hybrid_ablation.csv` to verify the simulator's prediction.

---

## New CLI Arguments Summary

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_split_layers` | int | -1 | Number of bottom layers using split parallelism (-1 = all layers, reproduces current behavior) |
| `--pipeline` | flag | False | Enable pipeline double buffering within forward pass |
| `--simulate_k` | flag | False | Run simulator to predict optimal `num_split_layers` before training |

---

## File Change Summary

| File | Change |
|---|---|
| `python/dgl/dev/splitmodel.py` | Add `SparseHiddenPull`, `AsyncShuffle`; modify `SRC_TO_DEST.forward()` to branch on `num_split_layers` |
| `python/dgl/dev/util.py` | Add `num_split_layers` to `SampleConfig` |
| `python/dgl/dev/splitloader.py` | Add support for sampling two microbatches for pipeline mode |
| `experiment/nodepred/split_trainer.py` | Add pipeline training loop; pass `num_split_layers` through |
| `experiment/utils/args.py` | Add `--num_split_layers`, `--pipeline`, `--simulate_k` |
| `experiment/utils/config.py` | Add `num_split_layers`, `pipeline` fields to `Config` |
| `experiment/hybrid_sim.py` | New file: simulator for predicting optimal `k` |
| `src/spara/split_sampler.h/.cc` | (If needed) expose stream-aware NCCL call for async Shuffle |

---

## Open Questions

1. **Backward through pipeline** — with two microbatches interleaved, the backward pass needs to track which activations belong to which microbatch. Does each microbatch get its own backward call, or are losses summed before a single backward?
3. **Stream safety of NCCL** — `_CAPI_Split_ScatterForward` currently runs on whichever stream is current. Confirm that passing an explicit stream handle is sufficient, or whether the NCCL communicator itself needs to be re-created per stream.
4. **Interaction between Feature 1 and Feature 2** — can pipeline double buffering be combined with Hybrid Split+DDP (i.e., pipeline the split layers only, then transition to DDP for top layers)?
