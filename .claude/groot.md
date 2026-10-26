## Spara/Groot 

Spara primarily consists of two folders, python/ folder which contains dgl folder and the setup.py. Also contains cpp files which are stored in the src folder. 

```
python setup.py build_ext --inplace
```

### Prerequisites
- CUDA 12.4 with `CUDA_HOME` set (the conda env created by `environment_setup.sh` provides this)
- PyTorch 2.4.0 (cu124), PyG, torchmetrics, ogb
- C++17 compiler, CMake 3.18+, Ninja
- Third-party deps: GKlib, METIS, oneTBB, NCCL (built by `init.sh`)

### One-time environment setup (from scratch)

RunPod pods are ephemeral: on termination the container filesystem (conda, the
env, and this repo, all under `/home` on the overlay mount) is wiped, so the
whole setup is redone on each new pod. The flow:

```bash
# 1. Install Miniconda (if not already present) and create + ACTIVATE an env.
#    conda itself and the env are your manual steps; the script installs INTO
#    whatever env is active.
conda create -y -p /home/groot_env python=3.11
conda activate /home/groot_env

# 2. Clone the repo and run the setup from inside the activated env.
git clone <repo-url> /home/dgl_groot
cd /home/dgl_groot
bash environment_setup.sh
```

`environment_setup.sh` installs CUDA 12.4 toolkit + gxx + CMake/Ninja, PyTorch
2.4.0 (cu124), PyG and `ogb` into the active env, builds the third-party deps
via `init.sh`, then does the first DGL build via `build.sh`.

- **Which env it installs into:** by default the currently-activated conda env
  (`ENV_PATH` defaults to `$CONDA_PREFIX`). Override with `ENV_PATH=/path/to/env`
  to target a different one, or `ENV_NAME=...` to create a sibling env.
- **No `bash -i` needed:** `conda init` only wires up the `conda` command for
  *interactive* shells (via `~/.bashrc`), so a plain non-interactive
  `bash environment_setup.sh` historically failed at the conda check. The script
  now locates conda's base itself (from `$_CONDA_ROOT`/`$CONDA_EXE`/common install
  paths) and sources its `conda.sh`, so running it inside the active env works
  directly. (`bash -i environment_setup.sh` still works if you prefer.)

### Rebuilding during development
The compile is in its own script so you can re-run it as often as you like:
```bash
./build.sh            # debug build  -> build/          (default; for development)
./build.sh release    # release build -> build-release/ (for performance measurements)
./build.sh debug --clean   # wipe the build dir and reconfigure from scratch
```
`build.sh` resolves `CUDA_HOME` automatically from the active conda env
(`CONDA_PREFIX`) — this is what prevents the `/lib/libcudart.so missing` link
error that occurs when `CUDA_HOME` is empty at cmake-configure time. Debug and
release use separate build dirs so both can coexist.

**Verify installation:**
```bash
python -c "import dgl; print(dgl.__path__)"
```

### CMake Build Types
DGL's `BUILD_TYPE` option accepts `dev`, `dogfood`, `release` (see `CMakeLists.txt`);
`build.sh` passes `debug`/`release` as both `CMAKE_BUILD_TYPE` and `BUILD_TYPE`.
`CMAKE_BUILD_TYPE` controls optimization/debug flags (`debug` → `-O0 -g`,
`release` → optimized).

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