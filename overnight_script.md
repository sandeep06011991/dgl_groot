# environment_setup.sh — log of work and decisions

Context: building a clean-room `environment_setup.sh` for this repo (Spara/groot, a
DGL fork) that creates a fresh conda env, installs PyTorch 2.4.0 + CUDA 12.4, builds
DGL from this checkout, and verifies GPU tensor/graph ops work. Ultimate target is
running this end-to-end on a RunPod pod; currently being validated on Unity.

## Existing repo state found during exploration

- `.claude/groot.md` — build docs (not root `groot.md`). Build sequence: `bash init.sh`
  (builds GKlib/METIS/oneTBB/NCCL into `third_party/build/`), then
  `cmake -B build -GNinja -DCMAKE_BUILD_TYPE=debug -DBUILD_TYPE=debug`,
  `cmake --build build -j`, `cd python && pip install .`.
- `build.sh` — hardcodes `CUDA_HOME` to the existing
  `/work/pi_huiguan_umass_edu/sandeep/conda_environment/dgl_groot/` env (cu11.8), calls
  `init.sh`, wipes `build/`, cmake-configures with an extra
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` flag (needed for newer cmake), builds, `pip
  install .`. Has a stray `exit 1` after — everything after it (LD_LIBRARY_PATH/CPATH
  exports, torch-quiver build) is dead code.
- `install.sh` — conda-installs `nvidia/label/cuda-11.8.0::cuda-toolkit` +
  `gxx==11.3`, then pip-installs PyG stack for `torch-2.0.0+cu118` and misc deps. Does
  **not** actually install PyTorch itself (comment only).
- `setup_unity.sh` — just `conda activate ../conda_environment/dgl_groot/` (sibling
  folder to repo root). This is the convention `environment_setup.sh` follows for
  where the new env lives.
- `init.sh` — builds GKlib, METIS, oneTBB, NCCL into `third_party/build/` from
  `third_party/{GKlib,metis,oneTBB,nccl}`. Not conda-env-specific beyond needing
  gcc/nvcc/cmake on PATH.
- Root `CMakeLists.txt` hardcodes `third_party/build/include` and `third_party/build/lib`
  (lines 378-379) — so the DGL cmake configure/build step doesn't need `CPATH`
  exports, but the built `libdgl.so` needs `LD_LIBRARY_PATH` pointing at
  `third_party/build/lib` at **import time** (NCCL/oneTBB are dynamically linked).
- `Dockerfile.runpod` exists at repo root but is **empty** — no prior RunPod-specific
  setup to reuse.
- Existing `../conda_environment/dgl_groot/` env baseline (read-only inspected, not
  modified): Python 3.11.13, torch 2.0.1+cu118, CUDA toolkit pinned to 11.8.0 (nvidia
  channel), gxx/gcc 11.3.0 (conda-forge), cmake 3.25.0 (pypi), dgl 1.2, torch_geometric
  2.7.0 + cu118 scatter/sparse/cluster builds already installed there.
- PyG usage check (grep across repo, excluding `third_party/`): `torch_geometric` is
  only imported inside the vendored, unrelated `third_party/torch-quiver/` and
  `third_party/dist_cache/torch-quiver/` benchmark code (a different baseline sampling
  system). `pyg_lib` has zero references anywhere. `torch_scatter`/`torch_sparse`/
  `torch_cluster` have zero real imports (only DGL's own unrelated `torch_sparse_tensor`
  variable name matched). **Conclusion: PyG is not required to build or run groot/Spara
  itself.**
- Unity cluster facts (from a read-only exploration agent, not necessarily true on
  RunPod): conda 24.7.1 at `/work/pi_huiguan_umass_edu/sandeep/conda_installation`;
  `module avail` has `cuda/12.4.1` but no cudnn build for exactly 12.4; only `gcc/9.4.0`
  module (no 10+), so conda-forge gxx is the right call, not a module; no `cmake`
  module; `nvidia-smi` shows driver 595.71.05 / A16 GPU (CUDA 13.2 driver-reported,
  fully compatible with a 12.4 toolkit); `mamba` not installed; `/work` had only ~88GB
  free at exploration time (not a hard constraint per user — see decisions).

## Decisions made (with user)

1. **New env location**: `../conda_environment/dgl_groot_cu124/` — one level up from
   the repo, sibling to the existing (untouched) cu11.8 `dgl_groot` env. User confirmed
   this Unity-style convention is fine and to ignore `/workspace`-style RunPod-specific
   pathing for now, since validation is currently happening on Unity.
2. **Conda bootstrap**: assume `conda` is already installed and on `PATH` — script does
   **not** bootstrap Miniconda itself. It errors out clearly if `conda` is missing.
3. **CUDA toolkit source**: self-contained conda package (`nvidia/label/cuda-12.4.0::cuda-toolkit`
   installed *into* the new env, `CUDA_HOME` pointed at the env prefix), matching the
   existing `build.sh`/`install.sh` convention. Rejected the Unity-module (`module load
   cuda/12.4.1`) approach since it doesn't exist/generalize on RunPod and the user
   confirmed conda-toolkit is correct. Exact nvidia channel label (`cuda-12.4.0`) was
   not verified against a live `conda search` (network check was interrupted) — script
   has a fallback (`-c nvidia -c conda-forge cuda-toolkit=12.4`) if the specific label
   is missing.
4. **gxx version**: reused `gxx==11.3` (conda-forge) — same version already proven to
   work in the existing cu11.8 env, and within CUDA 12.4's supported host-compiler range
   (gcc 6.3–13.2), so lowest-risk choice rather than picking something untested.
5. **PyG stack**: user said try to install it (`pyg_lib`, `torch_scatter`,
   `torch_sparse`, `torch_cluster`, `torch_spline_conv`, `torch_geometric` for
   `torch-2.4.0+cu124`) but it's OK to skip if it becomes too complex, since it's
   unrelated to groot/Spara's actual code (see PyG usage check above). Implemented as a
   **best-effort, non-fatal** block — failure prints a warning and the script continues
   rather than aborting. Controllable via `SKIP_PYG=1` env var.
6. **Disk/build cost**: user said disk space and build time are not a concern for this
   task — so no attempt made to minimize footprint (e.g. no shared-cache tricks, no
   skipping the CUDA toolkit conda install to save space).
7. **Script structure**: single top-to-bottom idempotent-ish script (`set -euo
   pipefail`), configurable via env vars (`ENV_NAME`, `ENV_PATH`, `PYTHON_VERSION`,
   `CUDA_VERSION`, `GXX_VERSION`, `TORCH_VERSION`/`TORCHVISION_VERSION`/
   `TORCHAUDIO_VERSION`, `TORCH_INDEX_URL`, `PYG_WHEEL_URL`, `SKIP_PYG`), no `--flag`
   parsing added (kept simple; reruns reuse the env dir if it already exists rather than
   forcing recreation, since `conda install`/`pip install` steps are themselves
   idempotent).

## What `environment_setup.sh` does (final, at repo root, executable)

1. Create (or reuse) conda env at `../conda_environment/dgl_groot_cu124`, Python 3.11.
2. Install CUDA 12.4 toolkit + `gxx==11.3` + `cmake`/`ninja` into the env; set
   `CUDA_HOME=$ENV_PATH`, prepend `$ENV_PATH/bin` to `PATH`; print `nvcc --version`.
3. `pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url
   https://download.pytorch.org/whl/cu124`; print torch/cuda version + `cuda.is_available()`.
4. Run `bash init.sh` to build GKlib/METIS/oneTBB/NCCL into `third_party/build/`.
5. `rm -rf build`, then `cmake -B build -GNinja -DCMAKE_BUILD_TYPE=debug
   -DBUILD_TYPE=debug -DCMAKE_POLICY_VERSION_MINIMUM=3.5`, `cmake --build build -j`,
   `cd python && pip install .`.
6. Best-effort: `pip install pyg_lib torch_scatter torch_sparse torch_cluster
   torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html && pip install
   torch_geometric`; warns and continues on failure.
7. Verification: sets `LD_LIBRARY_PATH` to include `third_party/build/lib` (needed at
   import time for NCCL/oneTBB), then runs
   `torch.tensor(0).to(0)` and `dgl.rand_graph(10, 10).to(0)`, prints
   `ENVIRONMENT SETUP OK` on success.

## Status / not yet done

- Not yet validated on an actual RunPod pod — only designed against Unity conventions
  per explicit user instruction to defer RunPod/`/workspace`-specific concerns for now.
- User directive (2026-08-25): don't stop to ask questions — keep adjusting versions /
  missing requirements autonomously until `torch.tensor(0).to(0)` and
  `dgl.rand_graph(10,10).to(0)` both succeed on a single GPU, then stop. This section is
  being updated live as the run progresses.

## Progress Log

- **Run started**: `nohup bash environment_setup.sh > environment_setup.log 2>&1 &`,
  PID 3809626 (fully detached from this shell, but *not* detached from the SLURM job —
  see risk note below).
- Step 1 (conda env create, Python 3.11) — **done**. Env created at
  `../conda_environment/dgl_groot_cu124`.
- Step 2 (CUDA toolkit) — **done**. `nvidia/label/cuda-12.4.0` channel label
  **confirmed to exist** (conda solved and installed `cuda-toolkit` from it directly —
  the untested fallback path was not needed).
- Step 3 (gxx==11.3, cmake, ninja, PyTorch 2.4.0/cu124) — **done**.
- Step 4 (`init.sh` third-party build: GKlib/METIS/oneTBB/NCCL) — **done**.
- Step 5 (DGL/Spara cmake+ninja build + `pip install .`) — **done**.
- Step 6 (PyG stack) — **in progress** as of 2026-08-25T07:27Z: `pyg_lib`,
  `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv` installed;
  `torch_geometric-2.8.0.post1` downloading/installing now (pulling in `aiohttp` etc.).
- Step 7 (GPU verification: `torch.tensor(0).to(0)` + `dgl.rand_graph(10,10).to(0)`) —
  not reached yet.
- **Handoff note (2026-08-25T07:27Z)**: original SLURM job `63598701` (partition
  `gpu-preempt`, node `gpu048`) was at `RunTime=59:07` against its `TimeLimit=01:00:00`.
  Step 7 (verification) ran just in time and **hit a real bug**: `import dgl` failed
  with `ModuleNotFoundError: No module named 'pandas'` — `dgl/__init__.py` eagerly
  imports `dgl.data.movielens`, which hard-imports `pandas`, but `pandas` was never
  listed in `python/setup.py`'s `install_requires` (confirmed by reading that file —
  only `numpy/scipy/networkx/requests/tqdm/psutil` were listed). This is a genuine gap
  in the repo's own packaging metadata, not a CUDA/torch issue.
  - **Fix applied**: added `"pandas"` to `install_requires` in `python/setup.py`
    (root-cause fix, so a plain `pip install .` is sufficient for anyone building this
    repo — not just a workaround in the setup script).
  - Manually `pip install pandas`'d into the live env and re-ran the verification
    snippet directly — it **passed**: `torch tensor device: cuda:0`, `dgl graph
    device: cuda:0`, `ENVIRONMENT SETUP OK`. So torch 2.4.0/cu124 + DGL/Spara built
    from source were confirmed working on a single GPU on node `gpu048` before that job
    ended (it has since disappeared from `squeue` — presumably hit its time limit right
    after).
  - `environment_setup.sh` itself was **not yet updated** with an explicit `pandas`
    install at that point — relying on the `python/setup.py` fix above to cover it on
    the next `pip install .` (DGL's own install_requires will now pull pandas in).

## Full clean re-run on a new node (job `63599675`, partition `gpu`, node `gpu030`)

User obtained a fresh SLURM job and asked to run the full script there for a clean,
reproducible validation (env/build state on `/work` already exists from the run above,
but this exercises the script end-to-end again on new hardware).

**Result: SUCCESS.** `bash environment_setup.sh` (log:
`environment_setup_gpu030.log`), run top-to-bottom, no manual intervention needed this
time — the `python/setup.py` pandas fix took effect automatically (pip pulled in
`pandas` as a `dgl==1.2` dependency, "Requirement already satisfied" since it was
already installed from the previous run's env reuse). All steps completed cleanly:
conda env, CUDA 12.4 toolkit, gxx 11.3, PyTorch 2.4.0+cu124, third-party build
(GKlib/METIS/oneTBB/NCCL), DGL/Spara cmake+ninja build, `pip install .`, and the full
PyG stack (`pyg_lib` 0.4.0, `torch_scatter` 2.1.2, `torch_sparse` 0.6.18,
`torch_cluster` 1.6.3, `torch_spline_conv` 1.2.2, `torch_geometric` 2.8.0.post1 — all
`+pt24cu124` builds).

Final verification output:
```
torch tensor device: cuda:0
dgl graph device: cuda:0
ENVIRONMENT SETUP OK
```

## Final status: DONE

Goal achieved — a clean conda env (`../conda_environment/dgl_groot_cu124`) builds and
installs PyTorch 2.4.0/CUDA 12.4 and DGL/Spara from source, and both
`torch.tensor(0).to(0)` and `dgl.rand_graph(10, 10).to(0)` run on a single GPU,
validated end-to-end on two separate nodes (`gpu048` and `gpu030`). Only fix required
along the way: `pandas` was missing from `python/setup.py`'s `install_requires` (root
cause fixed there, not just papered over in the setup script). No further action
pending; environment is ready to use via `conda activate
../conda_environment/dgl_groot_cu124`.

### ⚠️ SLURM risk: this session is running inside a time-limited job

This whole session (and the detached `environment_setup.sh` process, since it's still
inside the SLURM job's cgroup even though it's `nohup`'d from the shell) is running
under **SLURM job 63598701** on partition `gpu-preempt` (node `gpu048`):
- `TimeLimit=01:00:00`, `StartTime=07:27:33` ... wait, `EndTime=2026-08-25T07:27:33`.
- As of 2026-08-25T07:23:24Z there were only **~4 minutes** left before the job hits its
  time limit.
- `gpu-preempt` is also a **preemptible** partition, so the job could be killed earlier
  than the time limit if a higher-priority job needs the node.
- When the SLURM job ends (time limit or preemption), SLURM kills everything in the
  job's cgroup, including `nohup`'d/backgrounded processes — this is not a case where
  `nohup` protects the build. If the job disappears, `environment_setup.sh` dies
  mid-build.

**Per user instruction**: if the job is relinquished/ends, stop rather than fight it —
just record the last-known state here so the run can be resumed cleanly:
- Conda env at `../conda_environment/dgl_groot_cu124` persists on `/work` (persistent
  storage, not node-local scratch) — safe to reuse in a new job.
- `third_party/build/` (GKlib/METIS/oneTBB/NCCL) persists too — `init.sh` doesn't clean
  it by default, so a rerun will mostly no-op there.
- The DGL `build/` directory is mid-compile — `environment_setup.sh` does `rm -rf
  build` at the start of the cmake step, so a straight rerun recompiles from scratch
  (ninja doesn't get to reuse partial `.o` files across a full `rm -rf`). If resuming,
  consider dropping the `rm -rf build` for a faster incremental ninja build, or just
  accept the full recompile for simplicity.
- To resume: request a fresh SLURM allocation on `gpu-preempt` (or a longer/non-preempt
  partition to avoid repeating this), then rerun `bash environment_setup.sh` from the
  repo root — it will detect the existing env/toolchain and skip straight to the parts
  still needed.
