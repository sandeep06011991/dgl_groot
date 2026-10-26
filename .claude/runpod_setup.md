# RunPod Setup & Dataset Preprocessing

End-to-end recipe to bring up Spara/groot on a fresh **multi-GPU RunPod** pod and
produce the preprocessed datasets. Distilled from `environment_setup.sh`,
`build.sh`, `init.sh`, and `experiment/prepare_dataset/`.

---

## 1. What container to spin up on RunPod

- **GPUs:** a multi-GPU pod with **identical** GPUs (e.g. 2–8× A40 / A100 / L40S).
  They must be identical because `build.sh` reads the compute capability of the
  *first* GPU (`nvidia-smi ... | head -1`) and compiles for that single arch only
  (A40→sm_86, A100→sm_80). NCCL is built with `-arch=native`.
- **Image:** anything with **CUDA ≥ 12.4** and **Ubuntu 22.04**. Two good options:
  - RunPod official **PyTorch 2.4 / CUDA 12.4.1 / Ubuntu 22.04** template, or
  - base **`nvidia/cuda:12.4.1-devel-ubuntu22.04`** (then install Miniconda, below).

  The conda env installs its *own* CUDA 12.4 toolkit + nvcc, so the host only needs
  a driver new enough for CUDA 12.4. (Reference box: driver 595.91.07 / CUDA 13.2,
  so 12.4 is fine.)
- **Storage:** attach a large **persistent/NFS volume mounted at `/workspace`**.
  Sizing: arxiv ~0.1 GB, products ~a few GB, **papers100M needs ~150 GB+**
  (≈57 GB raw zip + ≈70–80 GB numpy — they must coexist during conversion).
  METIS partitioning is RAM-hungry (≈23 GB for *arxiv* alone; papers100M needs
  hundreds of GB — pick a high-RAM pod).

> **Do the heavy work on local/container disk, then copy the finished numpy +
> partition maps to the `/workspace` NFS.** Don't preprocess directly on NFS.

---

## 2. Create the conda environment

Prereq: `conda` on PATH. If the image has no conda, install Miniconda first:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/miniconda3
source /home/miniconda3/etc/profile.d/conda.sh
```

Clone the repo **with submodules** (third-party deps are submodules):

```bash
git clone --recurse-submodules https://github.com/sandeep06011991/dgl_groot.git
cd dgl_groot
# if you forgot --recurse-submodules:
git submodule update --init --recursive
```

Provision everything with one script. It creates the env, installs CUDA 12.4
toolkit + gxx 11.3 + cmake/ninja, PyTorch 2.4.0 (cu124), builds the third-party
deps (GKlib, METIS, oneTBB, NCCL) via `init.sh`, compiles DGL/Spara via
`build.sh`, then installs PyG + ogb and verifies torch+dgl on GPU:

```bash
# installs into the ACTIVE conda env; or pass ENV_PATH / ENV_NAME to make a new one
ENV_PATH=/home/groot_env bash environment_setup.sh
```

Key defaults (override via env vars): `PYTHON_VERSION=3.11`, `CUDA_VERSION=12.4.0`,
`GXX_VERSION=11.3`, `TORCH_VERSION=2.4.0` (cu124). Resulting env: Python 3.11,
torch 2.4.0+cu124.

Activate and verify:

```bash
conda activate /home/groot_env
python -c "import dgl; print(dgl.__path__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Rebuild during development (no re-provision needed):

```bash
./build.sh            # debug  -> build/
./build.sh release    # optimized -> build-release/   (use for perf numbers)
./build.sh debug --clean
```

---

## 3. Process the datasets

Pipeline per graph: **download → numpy (`get_npgraph.py`) → freq weights
(`get_weight_fast.py`, GPU) → METIS partition map (`get_partition.py`)**.
OGB graph names: `arxiv`, `products`, `papers100M`. (`orkut`, `friendster` are
SNAP graphs, not OGB.)

### Recommended: `download.sh` (avoids the OGB name-collision — see note)

`download.sh` wgets+unzips the raw OGB zip and renames it to `ogbn_<name>` *before*
calling `get_npgraph.py`, so OGB finds a cached dataset and never performs the
extract/rename that otherwise clobbers the output dir.

```bash
# 1) point the staging dir at local/container disk (NOT /workspace) by editing
#    experiment/script/env.sh:  WORKSPACE_DIR=/home/datasets
# 2) to use all GPUs for the weight computation on RunPod (no SLURM):
export SLURM_GPUS_PER_NODE=<NGPU>        # e.g. 4; default is 1 if unset

cd experiment/prepare_dataset
./download.sh arxiv
./download.sh products
./download.sh papers        # == papers100M
```

Each `download.sh <graph>` runs the full chain and writes the standard partition
map (`node=dst, edge=freq, bal=xbal, w4`).

### Manual (explicit `--data_dir`, what this session used)

```bash
cd experiment/prepare_dataset
DATA=/home/datasets/graph                 # local staging, outside the repo
NGPU=1                                     # set to your pod's GPU count

# numpy  — for arxiv/products run TWICE (first run hits the collision, see note).
# products/papers100M also prompt "download X GB? (y/N)", so pipe `yes`:
yes | python get_npgraph.py --data_dir=$DATA --graph_name=arxiv
yes | python get_npgraph.py --data_dir=$DATA --graph_name=arxiv   # re-run if feat.npy missing

# frequency weights (uses NGPU GPUs via torch spawn)
python get_weight_fast.py --data_dir=$DATA --graph_name=arxiv \
    --fanouts=15,15,15 --num_epoch=5 --world_size=$NGPU

# standard partition map -> partition_map/arxiv/arxiv_w4_ndst_efreq_xbal.npy
python get_partition.py --graph_name=arxiv --data_dir=$DATA \
    --num_partition=4 --node_weight=dst --edge_weight=freq --bal=xbal
```

For the full partition sweep (uniform/degree/dst × uniform/freq × bal/xbal), run
`bash partition.sh` (edit its graph list); for extra weight configs, `weight.sh`.

### ⚠️ OGB out_dir name-collision (calling `get_npgraph.py` directly)

`get_npgraph.py` creates `out_dir = <data_dir>/<name>` *before* invoking OGB. For
**arxiv** and **products**, OGB extracts its zip into a folder of the same base
name and then renames it to `ogbn_<name>` — moving your freshly-created `out_dir`
away, so the first run dies with `FileNotFoundError: .../feat.npy`. Fixes (pick one):
- use `download.sh` (pre-fetches → OGB skips extract/rename → no collision), or
- just run `get_npgraph.py` **twice** (the cached download no longer collides).

papers100M (`papers100M-bin`) does **not** collide.

### Output layout (per graph)

```
<data_dir>/<graph>/                 feat, label, indptr_xsym, indices_xsym,
                                    train/valid/test_idx, node_weight, edge_weight
<data_dir>/partition_map/<graph>/   <graph>_w4_ndst_efreq_xbal.npy
<data_dir>/ogbn_<graph>/            raw OGB download (not needed downstream)
```

---

## 4. Back up to the `/workspace` NFS

Copy the processed numpy + partition maps (skip the raw `ogbn_*` download):

```bash
mkdir -p /workspace/graph
cp -av /home/datasets/graph/<graph>              /workspace/graph/
cp -av /home/datasets/graph/partition_map        /workspace/graph/   # merges per-graph
```

Experiments then point `data_dir` at `/workspace/graph` (see
`experiment/script/env.sh` and `.claude/experiment.md`).
