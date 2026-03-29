## Spara/Groot 

Spara primarily consists of two folders, python/ folder which contains dgl folder and the setup.py. Also contains cpp files which are stored in the src folder. 

```
python setup.py build_ext --inplace
```

### Prerequisites
- CUDA 11.8+ with `CUDA_HOME` set
- PyTorch 2.0.1, PyG, torchmetrics, ogb
- C++17 compiler, CMake 3.18+, Ninja
- Third-party deps: GKlib, METIS, oneTBB, NCCL (built by `init.sh`)

### Build Steps

**Step 1: Build third-party dependencies (only needed once)**
```bash
bash init.sh
```
Builds GKlib, METIS, oneTBB, and NCCL into `third_party/build/`.

**Step 2: Build DGL/Spara**
```bash
export CUDA_HOME=/path/to/cuda   # or conda env path
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=debug -DBUILD_TYPE=debug
cmake --build build -j
cd python && pip install . && cd ../
```

Or simply run:
```bash
bash build.sh
```
Note: `build.sh` currently has `exit 1` after `pip install .` — lines after that are vestigial.

**Verify installation:**
```bash
python -c "import dgl; print(dgl.__path__)"
```

### CMake Build Types
- `dev` — debug symbols, `-O0 -g3` (default in `build.sh`)
- `dogfood` — intermediate testing, `-O2`
- `release` — production, `-O2`

Key CMake options: `USE_CUDA=ON`, `BUILD_TORCH=ON` (builds tensoradapter for PyTorch), `BUILD_SPARSE/BUILD_GRAPHBOLT` (disabled by default).

### Repository Layout

```
src/spara/          # Spara C++ core (the main contribution)
python/dgl/dev/     # Spara Python API layer
experiment/         # Training scripts, benchmarks, dataset prep
  nodepred/         # Node prediction trainers for all systems
  utils/            # Config, dataloading, profiling utilities
  script/           # Bash scripts for running experiments
  prepare_dataset/  # Scripts to download and preprocess graphs
third_party/build/  # Built artifacts for GKlib, METIS, oneTBB, NCCL
```

## Code Architecture

### C++ / CUDA Layer (`src/spara/`)

Core components registered as DGL API functions and called from Python via `_init_api`:

| File | Class | Role |
|---|---|---|
| `sampler.h/.cc` | `Sampler` | Single-GPU neighbor sampler; holds CSC graph, fanouts, and current `GraphBatch` |
| `split_sampler.h/.cc` | `SplitSampler` | Multi-GPU partition-aware sampler; uses NCCL all-to-all to exchange neighbor data across GPUs |
| `cnt_sampler.h/.cc` | `CntSampler` | Frequency-weighted sampler (used for simulation/analysis) |
| `batch_sampler.h/.cc` | `BatchSampler` | Batched variant of the sampler |
| `feature_loader.h/.cc` | — | Feature loading with GPU cache |
| `preprocess.h/.cc` | — | Graph preprocessing utilities |
| `coo2csr.h/.cc` | — | COO to CSR format conversion |
| `array_scatter.h/.cc` | `ScatteredArray` | Tracks which nodes belong to which partition after NCCL scatter |

CUDA kernels in `src/spara/cuda/`:
- `all2all.cu` — NCCL-based all-to-all communication for partition-aware neighbor exchange
- `batch_rowwise_sampling.cu` — GPU neighbor sampling kernel
- `bitmap.cu` / `bytemap.h` — GPU bitmap/bytemap for fast set-unique operations
- `feat_cache.cu` — GPU feature cache with LRU/frequency-based eviction
- `index_select.cu` — GPU index-select for feature gathering
- `map_edges.cu` — Remapping edge endpoints to local indices (reindexing)
- `gather.cu` — Feature gather from pinned host memory (UVA mode)
- `partition.cu` — CSR partitioning by partition map

All `Sampler`, `SplitSampler`, etc. are **singletons** accessed via `::Global()`.

### Python Layer (`python/dgl/dev/`)

Wraps the C++ API via `_init_api("dgl.dev", __name__)`:

| File | Class | Role |
|---|---|---|
| `dataloader.py` | `GraphDataloader` | Single-GPU dataloader; wraps `Sampler` singleton |
| `splitloader.py` | `SplitGraphLoader` | Multi-GPU dataloader; wraps `SplitSampler`; initializes NCCL via `InitNccl` |
| `splitmodel.py` | `SRC_TO_DEST`, `Shuffle` | DDP-aware GNN model with `Shuffle` autograd function that calls NCCL scatter/gather in forward/backward |
| `util.py` | `SampleConfig`, `IdxLoader` | Config dataclass and per-epoch index shuffler |
| `cnt_sampler.py` | — | Python wrapper for `CntSampler` |
| `partition.py` | — | Partition map utilities |

### Sampling Modes

Two data modes are supported, configured via `SampleConfig.mode`:
- `"uva"` — Graph stays on CPU pinned memory; GPU accesses via CUDA Unified Virtual Addressing (lower GPU memory use)
- `"gpu"` — Graph is copied to GPU (faster for small-enough graphs)

### Distributed Training Flow (Spara `split` system)

1. Load graph and partition map (`load_partition_map`) — a `.npy` file of shape `[num_nodes]` mapping each node to a partition ID (0-indexed, uint8)
2. Each GPU process initializes `SplitGraphLoader` with its rank and the shared NCCL unique ID
3. Each GPU owns the training nodes assigned to its partition (`partition_map[train_idx] == rank`)
4. `SplitGraphLoader.__next__()` calls `SplitSampleBatch` → C++ runs local sampling, then NCCL all-to-all to gather remote neighbors
5. Model forward pass in `SRC_TO_DEST.forward()` calls `Shuffle.apply()` which runs `_CAPI_Split_ScatterForward` (NCCL scatter of features to owning GPU), then runs the GNN layer locally
6. Backward through `Shuffle` reverses the scatter via `_CAPI_Split_ScatterBackward`



## Key Development Notes

- The C++ singleton pattern (`Sampler::Global()`, `SplitSampler::Global()`) means only one sampler instance exists per process. Multi-GPU training uses `torch.multiprocessing.spawn` so each spawned process has its own singleton.
- `_init_api("dgl.dev", __name__)` binds C++ functions registered under the `"dgl.dev"` namespace into the calling Python module. All `_CAPI_*` functions in `python/dgl/dev/` are provided this way.
- `SplitGraphLoader` requires a NCCL unique ID shared across all ranks before initialization (`GetUniqueId()` in the parent process, passed via `spawn`).
- The `SampleConfig.reindex=True` setting remaps sampled node IDs to 0-indexed local IDs using a GPU hash table — required for correct GNN layer computation. Do not disable unless you only need the raw sampled subgraph topology.
- Build artifacts land in `build/` (cmake output) and the installed package is at the Python env's `site-packages/dgl`.