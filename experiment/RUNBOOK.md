# Spara/groot Experiment Runbook

End-to-end workflow for running the Spara (`split`/gsplit) GNN experiments on a
fresh GPU box (e.g. a RunPod A100-SXM pod): install software → partition data →
back up → restore → run → (how GPU placement is decided). Data is **regenerated
from OGB** by default (fast + deterministic); Backblaze B2 is used to **back up
results and partition maps**.

## TL;DR (zero → results)

```bash
# A. CONTROL MACHINE: get a pod (retries through the A100-SXM capacity shortage)
bash experiment/script/acquire_pod.sh groot-run        # prints POD_SSH + an ssh command
ssh -i ~/.runpod/ssh/runpodctl-ssh-key root@<ip> -p <port>

# B. ON THE POD: build everything (Miniconda, env, clone, DGL release, torchmetrics)
curl -fsSL https://raw.githubusercontent.com/sandeep06011991/dgl_groot/final_dissertation/experiment/script/bootstrap_pod.sh | bash
source /home/miniconda3/etc/profile.d/conda.sh && conda activate /home/groot_env && cd /home/dgl_groot

# C. ON THE POD: partition/prep (regenerate from OGB) then run
export data_dir=/root/graph
bash experiment/script/provision_data.sh products "2 4"
bash experiment/script/run_split_experiment.sh products 2 "3 4 5" 5
#    -> experiment/logs/products_split_w2_d{3,4,5}.csv

# D. CONTROL MACHINE: stop billing when done (results are already backed up — see below)
runpodctl pod delete <POD_ID>
```

`run_split_experiment.sh` auto-copies the result CSVs to B2 (`gsplit-data/groot/logs/`)
if an `gsplit-b2` rclone remote is configured; otherwise it prints a reminder to `scp`
them off before deleting the pod. The scripts self-activate the conda env, but in a
hand-typed shell run `source /home/miniconda3/etc/profile.d/conda.sh` first.

Supported OGB graphs: `arxiv`, `products`, `papers100M`. Scripts are labelled
**control-machine** (runpodctl) vs **on-pod** below.

## 0. Provisioning a pod (control machine)

Needs `runpodctl` installed and authenticated **on your laptop/WSL** (not the pod):

```bash
# install (one-time)
wget -qO- https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
  -O /usr/local/bin/runpodctl && chmod +x /usr/local/bin/runpodctl
runpodctl config --apiKey <YOUR_RUNPOD_API_KEY>   # or: export RUNPOD_API_KEY=...  (never commit the key)
runpodctl ssh add-key                              # registers an SSH key under ~/.runpod/ssh/
```

Then acquire a pod — A100-SXM is often "Low"/empty, so **use the retry script** rather
than a single `pod create`:

```bash
bash experiment/script/acquire_pod.sh groot-run
# knobs: GPU_COUNT (default 2), DISK_GB (200), DCS (US-MD-1,US-MO-1,US-KS-2,US-WA-1,EUR-IS-1),
#        MAX_ATTEMPTS (30), RETRY_SECONDS (60). Prints POD_ID, POD_SSH, and an ssh command.
```

Notes: the split **measurement needs GPU_COUNT ≥ 2** (world_size 2 → `w2` map; `w4` needs 4
GPUs). No US A100-SXM datacenter supports network volumes, so persistence is via B2 (§3),
not a pod volume. Pods are billed hourly ($3.58/hr for 2× A100-SXM) — **`runpodctl pod delete <id>`
when done** (data you care about should already be in B2 / pulled local).

Supported OGB graphs: `arxiv`, `products`, `papers100M`.

---

## 1. Software setup (one-time per pod)

**Fastest path: `bash experiment/script/bootstrap_pod.sh`** (or the `curl | bash` one-liner
in the TL;DR) does everything in this section automatically on a fresh pod. The manual
steps below are what it runs, for reference / debugging.

RunPod pods are ephemeral — conda env, repo, and container disk are wiped on
termination, so this is redone on each new pod. Start from a CUDA-12.4 image
(e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`).

```bash
# OS packages
apt-get update && apt-get install -y unzip build-essential ca-certificates wget

# Miniconda (accept channel ToS — required by recent Miniconda)
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p /home/miniconda3
source /home/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# env + repo
conda create -y -p /home/groot_env python=3.11 && conda activate /home/groot_env
git clone --recurse-submodules https://github.com/sandeep06011991/dgl_groot.git /home/dgl_groot
cd /home/dgl_groot && git checkout final_dissertation && git submodule update --init --recursive

# build (installs everything below into the env)
ENV_PATH=/home/groot_env bash environment_setup.sh
./build.sh release
python -c "import dgl; print(dgl.__path__)"      # verify
```

`environment_setup.sh` installs, into the active conda env:
- **CUDA 12.4 toolkit** + **gxx 11.3** (sets `CUDA_HOME` to the env), `cmake`, `ninja`
- **PyTorch 2.4.0 (cu124)** + torchvision/torchaudio
- **third-party native deps** via `init.sh`: GKlib, METIS (i64), oneTBB, NCCL → `third_party/build/`
- first **DGL/Spara build** (debug); then `./build.sh release` for perf
- **PyG** (pyg_lib, torch_scatter/sparse/cluster/spline, torch_geometric)
- **ogb** (dataset download/convert) and **torchmetrics** (trainers import it)

GPU arch is auto-detected by `build.sh` (A100 → sm_80); it needs `nvidia-smi` or
`CUDA_ARCH_BIN=80`, so build on the actual GPU box.

## 2. Partitioning the data

`provision_data.sh` (Section 4) runs the whole prep pipeline. The individual steps,
all reading/writing `$data_dir/<graph>/`:

1. **`get_npgraph.py`** — download OGB, convert to CSC numpy: `indptr_xsym.npy`,
   `indices_xsym.npy`, `train/valid/test_idx.npy`, `feat.npy`, `label.npy`.
2. **`get_weight_fast.py --fanouts=15,15,15 --num_epoch=5 --world_size=<G>`** — runs
   the GPU frequency sampler and writes `node_weight.npy`, `edge_weight.npy`
   (how often each node/edge is touched during sampling). *Needs a GPU.*
3. **`get_partition.py --num_partition=<k> --node_weight=dst --edge_weight=freq --bal=xbal`**
   — METIS partitions the graph into `k` parts, weighting vertices by dst-sampling
   frequency and edges by sampling frequency so partitions are **sampling/locality
   aware**. *CPU + lots of RAM* (papers100M peak ~65 GB). Output:
   `partition_map/<graph>/<graph>_w<k>_ndst_efreq_xbal.npy` — an `int8` array mapping
   each node → partition id. Build one map per world size you'll run (2, 4, …).

The map filename encodes **partition count + weight strategy only** — not depth/fanout.
Depth affects a partition only indirectly, via the step-2 frequency weights.

## 3. Where to back up (Backblaze B2)

Bucket **`gsplit-data`**, endpoint `s3.us-east-005.backblazeb2.com`. Configure rclone
on the pod (creds live locally at `~/.config/groot/b2.env`, chmod 600 — **never commit;
the repo is public**):

```bash
curl https://rclone.org/install.sh | bash
source ~/.config/groot/b2.env
rclone config create gsplit-b2 b2 account "$B2_KEY_ID" key "$B2_APP_KEY"
```

Bucket layout: `gsplit-data/groot/{graph/<g>/, partition_map/<g>/, logs/}`. Back up:
- **partition maps** (`partition_map/<g>/`) — worth keeping for reproducibility (METIS
  is not guaranteed bit-identical across runs).
- **result CSVs + run logs** (`logs/`).
- **full `graph/<g>/`** only if you want it (optional; the 57 GB `papers100M/feat.npy`
  regenerates from OGB, and B2 free tier has a storage cap). Push with `PUSH_B2=1`:
  `PUSH_B2=1 bash experiment/script/provision_data.sh papers100M "2 4"`.

## 4. Getting the data back (default: regenerate from OGB)

```bash
export data_dir=/root/graph
bash experiment/script/provision_data.sh <graph> "<world_sizes>"   # e.g. papers100M "2 4"
```

Idempotent — only rebuilds what's missing. By default it **regenerates from OGB**
(download the zip, unzip, rename to `ogbn_<g>`, then steps 1–3 of Section 2). OGB is
**deterministic**: regenerated `feat.npy`/topology are byte-identical with the same
node ordering, so any previously-saved partition map still aligns. METIS is retried
once (it can segfault nondeterministically on huge graphs).

Opt-in B2 restore instead of OGB (e.g. to reuse exact maps): `RESTORE_B2=1 bash .../provision_data.sh <graph> "2 4"`.

OGB download sources (direct zips):

| graph | URL | unzip → rename |
|---|---|---|
| arxiv | `http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip` | `ogbn_arxiv` |
| products | `http://snap.stanford.edu/ogb/data/nodeproppred/products.zip` | `ogbn_products` |
| papers100M | `http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip` | `ogbn_papers100M` |

Sizes: arxiv ~0.1 GB; products a few GB; papers100M ~57 GB zip / ~70 GB npy / ~65 GB
METIS peak RAM (needs a high-RAM box).

## 5. Running the experiment

```bash
bash experiment/script/run_split_experiment.sh <graph> <world_size> "<depths>" <num_epoch>
# arxiv:      run_split_experiment.sh arxiv 2 "3 4 5" 5
# papers100M: run_split_experiment.sh papers100M 2 "3 4 5" 3
```

Ensures data (Section 4), then runs `train_main.py --system=split --model=sage` once
per depth (fanout = depth copies of 15), writing one row per depth to
`experiment/logs/<graph>_split_w<ws>_d<depth>.csv`. Columns include per-epoch
`duration/sampling/feature/forward/backward` seconds, `allocated/reserved` MB,
`test accuracy %`, `edges_computed`, and `edge_skew`.

`world_size` = number of GPUs **and** selects the `w<world_size>` partition map, because
`train_main.py` forces `num_partition = world_size`. A 2-GPU pod can only run `w2`;
`w4` needs a 4-GPU pod.

## 6. How data placement on the GPU is decided

Three independent decisions control what lives on each GPU vs. in host memory:

1. **Graph topology — `--sample_mode` (default `uva`).**
   - `uva`: the CSC graph stays in **pinned host memory**; GPUs read it via CUDA
     Unified Virtual Addressing (low GPU memory, used for large graphs).
   - `gpu`: the whole CSC graph is copied onto each GPU (faster, only for graphs that fit).
   Each training step the *sampled* subgraph (the `blocks`) is materialized on the GPU
   and reindexed to local ids (`reindex=True`).

2. **Which nodes a GPU "owns" — the partition map.** `partition_map[node] == rank`
   defines rank-`r`'s partition. A rank's training seeds are the `train_idx` nodes in
   its partition. METIS built this map (Section 2) with **dst-sampling-frequency vertex
   weights + sampling-frequency edge weights**, so partitions are balanced *and*
   locality-aware (minimize cross-GPU neighbor traffic). This is the core of
   partition-aware sampling/feature-loading.

3. **Which node features sit on the GPU — the feature cache (`--cache_size`).** In
   `split_trainer.py`, each rank caches features for the nodes **it owns**:
   ```
   local_ids       = nodes where partition_map == rank
   num_ids_cached  = min( cache_size_bytes / (feat_dim * 4 bytes), len(local_ids) )
   dataloader.init_featloader(feat, local_ids[:num_ids_cached])
   ```
   So `--cache_size` (e.g. `10G`) sets how many of this rank's partition-local node
   features are pinned on the GPU. Any feature **not** cached is fetched from pinned
   host memory on demand each step (`get_feature()` → gather over PCIe/UVA). Bigger
   `cache_size` and a locality-aware partition ⇒ more cache hits ⇒ less PCIe traffic.

4. **During the model forward (cross-GPU) — the `Shuffle` scatter.** `SRC_TO_DEST` /
   `Shuffle` (in `dgl/dev/splitmodel.py`) NCCL-scatters boundary-node features to the
   owning GPU so each layer computes on locally-resident data, and reverses it on the
   backward pass. This is what makes the partition map pay off at compute time.

Net: `sample_mode` decides graph residency; the **partition map** decides node
ownership/compute locality; `cache_size` decides how much of each partition's features
are resident vs. gathered from host; and `Shuffle` moves only the boundary features
across GPUs during forward/backward.

---

## Data locations

```
$data_dir/
  <graph>/
    indptr_xsym.npy  indices_xsym.npy        # CSC topology (*_sym for orkut/friendster)
    train_idx.npy  valid_idx.npy  test_idx.npy
    feat.npy  label.npy                      # real OGB features/labels
    node_weight.npy  edge_weight.npy         # freq weights (intermediate; maps only)
  partition_map/<graph>/
    <graph>_w<k>_ndst_efreq_xbal.npy         # one per world size k
```

## Gotchas

- `get_weight_fast.py` ends with `os._exit(1)` — **always** non-zero even on success;
  gate on `node_weight.npy`/`edge_weight.npy`, not the exit code.
- `torchmetrics` is imported by the trainers; now installed by `environment_setup.sh`.
- `load_data` (`utils/dataloading.py`) uses **random** features unless the graph is in
  the real-feature list (`products`, `arxiv`, `papers100M`). Random feats give valid
  timing but meaningless accuracy.
- The accuracy eval in `split_trainer.py` was gated on a non-existent `"products2"`
  (test acc always 0); now enabled for the real-feature graphs.
- METIS can segfault nondeterministically on huge graphs (papers100M `w4` did once,
  succeeded on retry) — `provision_data.sh` retries once.
- No US A100-SXM datacenter supports network volumes; persistence uses B2, not NFS.
- `--cache_size` must be like `10G` (the trainer strips a trailing `G`); do not pass `10GB`.
- A100-SXM capacity is frequently empty — use `acquire_pod.sh` (retries), not a single `pod create`.
