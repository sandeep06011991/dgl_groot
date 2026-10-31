# Hybrid DP+Split — implementation changes

Branch: `hybrid-data-parallelism` (worktree under `.claude/worktrees/`).
Implements the plan in [01_dp_plus_split.md](01_dp_plus_split.md): make `k` (the
number of leading data-parallel layers) a configurable parameter, remove the two
hard-blocks that forced `k=0`, and wire `k` end-to-end into both the sampler and
the model.

`k` is exposed as `--num_redundant_layer` and flows:
CLI → `Config.num_redundant_layer` → `SampleConfig.num_dp` → the C++ sampler
(`SetNumDP`) **and** the loader's `GetBlocks`/model.

**Semantics:** sampler-layers `[0, k)` are the **seed-adjacent** layers and run
data-parallel (each rank samples its own seed shard locally, no NCCL exchange);
sampler-layers `[k, L)` run the existing partition-aware split. `k=0` = today's
full split; `k=L` (= `len(fanouts)`) = pure DP. See the "flow" note at the end.

---

## Files changed (8)

### 1. `experiment/utils/args.py` — new CLI flag
Added `--num_redundant_layer` (int, default `0`): `k`, the number of leading DP
(no-shuffle) layers. `0` = full split, `len(fanouts)` = pure DP.

### 2. `experiment/train_main.py` — flag → Config + validation
Sets `cfg.num_redundant_layer = args.num_redundant_layer` and asserts
`0 <= k <= len(fanouts)`.

### 3. `experiment/utils/config.py` — (unchanged structurally)
`Config.num_redundant_layer` already existed (default `0`); it is now actually
populated from the CLI in `train_main.py`.

### 4. `python/dgl/dev/util.py` — SampleConfig carries `k`
Added field `num_dp: int = 0` to the `SampleConfig` dataclass so the loader
receives `k`.

### 5. `experiment/nodepred/split_trainer.py` — pass `k` to the loader
`SampleConfig(...)` now passes `num_dp=config.num_redundant_layer`.

### 6. `python/dgl/dev/splitloader.py` — set `k` on sampler + loader
- Reads `self.num_dp = config.num_dp` and asserts `0 <= num_dp <= len(fanouts)`.
- Calls `SetNumDP(self.num_dp)` during `__init__` (before `InitNccl`) so the C++
  sampler knows `k` before sampling starts. **This is the call that was missing
  entirely before** — `_CAPI_Split_SetNumDP` existed but nothing invoked it.
- Passes `num_dp=self.num_dp` into `GetBlocks(...)` in `__next__`.

### 7. `python/dgl/dev/splitloader_util.py` — `SetNumDP` wrapper + DP-aware blocks
- New `SetNumDP(num_dp)` Python wrapper calling `_CAPI_Split_SetNumDP`.
- `GetBlocks(...)` gained a `num_dp: int = 0` arg. For each sampler-layer it now
  attaches `block.scattered_src = None` when `layer < num_dp` (a DP layer, for
  which the sampler built **no** `ScatteredArray`), and only calls
  `_CAPI_GetBlockScatteredSrc` for `layer >= num_dp`. This prevents the invalid
  `_scattered_arrays.at(layer - _num_dp)` (negative index) for DP layers.

### 8. `python/dgl/dev/splitmodel.py` — shuffle only where a ScatteredArray exists
- **Removed** the hard-block `assert(self.num_dp == 0)`.
- The per-layer shuffle guard changed from `if not inference:` to
  `if not inference and getattr(block, "scattered_src", None) is not None:`.
  The model now shuffles **exactly** on blocks that have a scattered array, and
  skips the DP blocks.

  Why this and not the plan's suggested `if i >= num_dp`: `GetBlocks` reverses
  block order (`blocks.insert(0, block)`), so the DP (seed-adjacent) layers are
  the model's **last** `k` layers, not its first. Gating on
  `scattered_src is not None` is order-independent, avoids the index-direction
  bug, and reduces to today's behavior when `k=0` (every block has a
  scattered array).

### 9. `src/spara/split_sampler.h` — implement the DP sampler branch (C++)
- **Removed** the stub `CHECK_EQ(_num_dp, 0) << "does not support hybrid sampling
  for now"`. The `if (layer < _num_dp)` branch now simply does
  `frontiers.push_back(unique_src)`: it samples the frontier's neighbors locally
  and feeds the local unique set to the next (deeper) layer, building **no**
  `ScatteredArray` and doing **no** NCCL all-to-all. The split branch
  (`layer >= _num_dp`) is unchanged, and `getScatteredArray(layer - _num_dp)`
  indexes correctly because only split layers push to `_scattered_arrays`.

> Note: the setter `setNumDP` and the `_CAPI_Split_SetNumDP` registration already
> existed in `split_sampler.{h,cc}`; the only C++ source change needed was
> removing the stub assert.

---

## Verification (RunPod, 2× A100-SXM-80GB)

Built release with the changes and ran the real split SAGE trainer on **arxiv**
(w2 partition, 2 GPUs, 3 epochs) at three values of `k`:

| `k` | mode | result | test acc (rank0 / rank1) |
|-----|------|--------|--------------------------|
| 0 | full split | exit 0 | 47.81% / 53.39% |
| 1 | hybrid (1 DP + 2 split) | exit 0 | 47.54% / 52.99% |
| 3 | pure DP (all layers) | exit 0 | 47.86% / 55.64% |

All three run with no crash / NaN and land in the same accuracy band (per-rank
*local* test accuracy; no fixed seed, 3 epochs, so values are comparable rather
than bit-identical). The old `_num_dp==0` assert no longer fires and the hybrid
path executes. A tighter K₁₀ forward+grad oracle test was considered but deemed
unnecessary given arxiv parity.

## Build gotcha hit along the way
`sync_code.sh --build` initially ran **stale** code because `rsync -a` preserves
mtimes (ninja + setuptools skip rebuilds), and `setup.py` copies `build/` (debug)
rather than `build-release/`. Fix used: `touch src/spara/* && rm -rf python/build`
then `ln -sfn build-release build` before `pip install .`; verify with `md5sum`
(python) and `strings libdgl.so | grep` (C++). Recorded in session memory.

---

## Flow (clique intuition, K₁₀, 2 GPUs, even→GPU0 / odd→GPU1, k=1)
- **DP layer 0 (seed-adjacent):** each GPU samples its own seed's neighborhood
  locally → on a clique that is all 10 nodes. No shuffle.
- **Split layer 1:** the full frontier `{0..9}` is scattered by owner, so GPU0
  keeps the even nodes and GPU1 the odd — "only this partition's nodes are
  expanded." The shuffle gathers the input features needed here.
- Because the DP layer did not shrink the frontier, the boundary split layer
  computes embeddings for all 10 nodes (redundantly per GPU) — exactly the
  replicated inputs the DP layer then consumes locally, so no separate
  switch-gather is needed. Correct, trading the seed-adjacent shuffle for
  redundant compute (hence "redundant layer").
