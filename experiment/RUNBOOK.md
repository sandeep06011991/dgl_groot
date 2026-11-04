# Spara/groot Experiment Runbook

End-to-end workflow for running the Spara (`split`/gsplit) GNN experiments on a
fresh GPU box (e.g. a RunPod A100-SXM pod): install software → partition data →
back up → restore → run → (how GPU placement is decided). Data is **regenerated
from OGB** by default (fast + deterministic); Backblaze B2 is used to **back up
results and partition maps**.

## TL;DR (zero → results)

```bash
# A. CONTROL MACHINE: get a pod (retries through the A100-SXM capacity shortage)
bash experiment/script/cloud/runpod/acquire_pod.sh groot-run        # prints POD_SSH + an ssh command
ssh -i ~/.runpod/ssh/runpodctl-ssh-key root@<ip> -p <port>

# B. ON THE POD: build everything (Miniconda, env, clone, DGL release, torchmetrics)
curl -fsSL https://raw.githubusercontent.com/sandeep06011991/dgl_groot/final_dissertation/experiment/script/cloud/runpod/bootstrap_pod.sh | bash
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
bash experiment/script/cloud/runpod/acquire_pod.sh groot-run
# knobs: GPU_COUNT (default 2), DISK_GB (200), DCS (US-MD-1,US-MO-1,US-KS-2,US-WA-1,EUR-IS-1),
#        MAX_ATTEMPTS (30), RETRY_SECONDS (60). Prints POD_ID, POD_SSH, and an ssh command.
```

Notes: the split **measurement needs GPU_COUNT ≥ 2** (world_size 2 → `w2` map; `w4` needs 4
GPUs). No US A100-SXM datacenter supports network volumes, so persistence is via B2 (§3),
not a pod volume. Pods are billed hourly ($3.58/hr for 2× A100-SXM) — **`runpodctl pod delete <id>`
when done** (data you care about should already be in B2 / pulled local).

Supported OGB graphs: `arxiv`, `products`, `papers100M`.

---

## 0b. Massed Compute (alternative to RunPod)

Use Massed Compute (MC) when A100-SXM capacity on RunPod is dry. MC is a **VM**
provider (not containers), so the image is a bare Ubuntu with only the NVIDIA
**kernel driver** — the CUDA toolkit, torch, and the DGL/Spara build all come from
the conda env (`environment_setup.sh`), exactly as on RunPod.

**Prereq — the API token must be *exported*.** Scripts read `MC_API_TOKEN` from the
environment. If it lives in `~/.profile` as a bare assignment (no `export`), a child
process (the scripts) won't inherit it — you'll get `ERROR: export MC_API_TOKEN first`
even though `echo ${#MC_API_TOKEN}` is nonzero. Fix:

```bash
source ~/.profile && export MC_API_TOKEN     # ensure children inherit it
```

**Which path are you on — COLD or WARM?** Check whether a prebuilt env snapshot already exists:

```bash
source ~/.config/groot/b2.env && rclone config create gsplit-b2 b2 account "$B2_KEY_ID" key "$B2_APP_KEY" >/dev/null 2>&1
rclone lsf gsplit-b2:gsplit-data/groot/env/     # prints mc_a100_sm80.tar.zst if a B2 snapshot exists
curl -s -H "Authorization: Bearer $MC_API_TOKEN" https://vm.massedcompute.com/api/v1/storage-volumes  # volume id/status
```

- **COLD (first time):** no `mc_a100_sm80.tar.zst` in B2 **and** no seeded volume → **Path A**
  (build once ~30–45 min, then seed persistence so you never build again).
- **WARM (persistence exists):** the snapshot is in B2 and/or on the `groot-sec1` volume
  (id `12cd1bbd-5fc8-4f74-86fe-7b106a5ad02f`) → **Path B** (restore, skip the build, minutes).

VM facts common to both: user `Ubuntu` (sudo), image = bare Ubuntu, paths `repo /home/dgl_groot`,
`env /home/groot_env`, `data /home/data` (match `run_worker2.sh`). Only `gpu_8x_A100_SXM4`
($11.04/hr, 8×A100-80GB, 800 GB RAM, 5 TB disk) can host a 4-GPU (`w4`) run — MC has NO 4× product
(1× is $1.38/hr but can't do `world_size=4`; it IS enough to seed/validate a volume cheaply).

#### Path A — first time / COLD (no persistence yet)

```bash
# 1. CONTROL: create a volume once (POST /storage-volumes or the web console), or reuse
#    groot-sec1's id; launch WITH it so it mounts at /persistent/<name> (attach is launch-only).
VOLUME_IDS=<volume-id> bash experiment/script/cloud/mc/mc_launch.sh groot-run
# 2. ON THE VM: FULL build ~30–45 min (no snapshot yet -> mc_bootstrap builds). Logs to ~/boot.out.
curl -fsSL https://raw.githubusercontent.com/sandeep06011991/dgl_groot/final_dissertation/experiment/script/cloud/mc/mc_bootstrap.sh | bash
# 3. provision data, then SEED persistence so every later launch is WARM:
export data_dir=/home/data && bash experiment/script/provision_data.sh papers100M "4"
bash experiment/script/cloud/mc/mc_volume.sh snapshot-env            # env  -> volume
bash experiment/script/cloud/mc/mc_snapshot_env.sh                   # env  -> B2 (offsite backup)
bash experiment/script/cloud/mc/mc_volume.sh snapshot-data papers100M  # data -> volume (optional)
# 4. run (see the sweep block below); terminate when idle — volume + B2 survive.
```

#### Path B — subsequent / WARM (snapshot exists → skip the build)

```bash
# 1. CONTROL: relaunch WITH the volume (mounts at /persistent/<name>).
VOLUME_IDS=12cd1bbd-5fc8-4f74-86fe-7b106a5ad02f bash experiment/script/cloud/mc/mc_launch.sh groot-run
# 2. ON THE VM: mc_bootstrap takes the FAST-PATH — finds the volume tarball (then B2), untars to
#    local /home, verifies import dgl, and SKIPS the ~40-min build (minutes, validated 2026-11-03).
curl -fsSL https://raw.githubusercontent.com/sandeep06011991/dgl_groot/final_dissertation/experiment/script/cloud/mc/mc_bootstrap.sh | bash
#    - B2 restore (empty/no volume) needs the gsplit-b2 remote configured first; the volume path doesn't.
#    - SKIP_RESTORE=1 forces a clean rebuild; if C++/CUDA sources changed since the snapshot,
#      ./build.sh release + re-run snapshot-env (a plain git pull leaves the .so stale).
# 3. restore data from the volume (fast) or B2, run, terminate when idle:
export data_dir=/home/data && bash experiment/script/cloud/mc/mc_volume.sh restore-data papers100M
```

**Always confirm billing state with `mc_status.sh`** (its `runningInstances` parsing is fixed) — if it
ever prints "(no instances)" while a VM is live you have a stale copy, so re-check before assuming
nothing bills. **Terminate** (not just "stop" — see below) when done: `mc_terminate.sh <uuid>`.

**Running the Section-1 perf sweep on the VM** (the gsplit-vs-DGL @0%-cache grid —
see §5.1), across the 8 GPUs as two concurrent 4-GPU shards:

```bash
export data_dir=/home/data
bash experiment/script/provision_data.sh papers100M "4"      # B2 restore; no feat (random_feat)
cd experiment/script && python3 gen_settings.py              # writes sec1_settings.jsonl (missing cells only)
# shard 0 -> GPUs 0-3, shard 1 -> GPUs 4-7, distinct TRAIN_MASTER_PORT so DDP rendezvous don't collide
bash run_worker2.sh 0 0,1,2,3 29500 sec1_settings.jsonl papers_shard0.csv papers
bash run_worker2.sh 1 4,5,6,7 29600 sec1_settings.jsonl papers_shard1.csv papers
```

### Billing: there is NO "Stop" on MC — idle = full rate until you terminate

MC bills per-minute and has **no pause/hibernate state**: the only instance actions are
`restart` and `terminate` (`GET /api/v1` lists no stop endpoint; MC's own docs say
"stop effectively means terminate — there is no pause-and-resume, no hibernation").
So "stop the box to save the build and restart later" is **not possible** — an idle
`gpu_8x_A100_SXM4` keeps billing the full **$11.04/hr** until `instance/terminate`,
which **destroys the VM disk**. The only way to stop GPU billing is to terminate.
=> The right pattern is **snapshot → terminate-when-idle → fast-restore-on-next-launch**
(below), never "leave it stopped."

### Persistent-storage volume (beta): what it is and how it attaches

A **Persistent Storage volume** survives termination and can hold the build tarball +
datasets so the next VM comes up fast without a B2 round-trip. Confirmed mechanics
(VM API + live pod, Nov 2026):

- **List/create/delete via API:** `GET/POST /storage-volumes`, `DELETE /storage-volumes/{id}`
  (all accept the bearer token). `GET /storage-volumes` shows `id`, `name`, `status`,
  `mountPath`, and `attachedTo`. (Note: `/storage`, `/volumes`, `/account` only exist in
  the web console and 307-redirect to `/login` with the API token.)
- **Mounts at `/persistent/<volume-name>`** on the VM (object-storage backed; the image
  ships `/usr/local/bin/geesefs`).
- **Attach is LAUNCH-ONLY.** The launch field is `volumeIds` (camelCase, array, **max 8**;
  `volume_ids` → 422). Attaching to an **already-running** VM only updates the volume's
  metadata (`attachedTo` fills in) — it does **NOT** hot-mount: `/persistent/<name>` stays
  empty and is not a mountpoint until the VM is **relaunched** with that id in `volumeIds`.
  (This is exactly why a volume attached to a live box is invisible there — `lsblk` shows
  only `sda`, `/persistent` is empty. Detach is web-console-only.)
- **Survives terminate; billed (storage only) until DELETEd.** A volume still attached to
  any instance can't be deleted (409).
- **The env/`.so` can't run off the geesefs mount** (not relocatable) — use the volume as a
  *tarball store*: snapshot env → volume, and on a fresh VM copy the tarball OFF the volume
  to local `/home` and untar. Dataset `.npy` MAY be read straight off the mount.

Helper (on the VM): `experiment/script/cloud/mc/mc_volume.sh {detect|snapshot-env|restore-env|snapshot-data <g>|restore-data <g>|seed-from-b2}`.

### Snapshot / restore reference (the mechanics Paths A & B use)

The env is persisted as a **tarball** (`mc_a100_sm80.tar.zst`) keyed to the exact paths + GPU
arch, kept on the **volume** (primary fast path) and in **B2** (offsite). Validated 2026-11-03.

- **Create a snapshot** (Path A, on a built VM): `mc_volume.sh snapshot-env` (→ volume) and
  `mc_snapshot_env.sh` (→ B2). Both verify `import dgl` before trusting the build.
- **Seed a fresh volume straight from B2** (no local build): `mc_volume.sh seed-from-b2`.
- **Restore** (Path B) is automatic inside `mc_bootstrap.sh` (volume → B2 → build);
  `SKIP_RESTORE=1` forces a clean build.
- **Restore is MC→MC only** — A100 sm_80 + image 84. A different arch/image must rebuild.
- **Point-in-time:** if the C++/CUDA sources changed since the snapshot, `./build.sh release`
  and re-run `snapshot-env` (a plain `git pull` leaves the `.so` stale).
- **geesefs gotchas** (encoded in `seed-from-b2`): never `rclone copyto B2 → the geesefs mount`
  (it rejects `chtimes`); stage the download under `$HOME` (fresh `/home` is root-owned until
  `mc_bootstrap` chowns it), then plain `cp` onto the volume.

---

## 1. Software setup (one-time per pod)

**Fastest path: `bash experiment/script/cloud/runpod/bootstrap_pod.sh`** (or the `curl | bash` one-liner
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

> **ALWAYS pull the partition map from B2 — do NOT repartition each run.** METIS
> is **not bit-identical across runs** (verified: a fresh `get_partition.py` for
> `arxiv w2` produced a different md5 than the B2 copy), so regenerating silently
> changes node→partition assignments and makes accuracy/timing non-comparable to
> prior results. It is also slow and RAM-hungry on big graphs (papers100M peak
> ~65 GB). The map in `gsplit-data/groot/partition_map/<graph>/` is the canonical
> source of truth. On a **reused** pod, `rclone` may be installed but the
> `gsplit-b2` remote is often **not configured** — configure it first (§3), then
> restore (`RESTORE_B2=1 provision_data.sh ...`, or `rclone copyto` the single
> map). Only run `get_partition.py` (below) to create a map that does **not yet
> exist in B2** — then push it back (it is Tier-A2, auto-pushed by
> `provision_data.sh`).

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

## 4. Getting the data back (default: restore from B2)

```bash
export data_dir=/root/graph
bash experiment/script/provision_data.sh <graph> "<world_sizes>"   # e.g. papers100M "2 4"
```

Idempotent — only fetches what's missing. **By default it restores from B2** (`RESTORE_B2=1`,
the `provision_data.sh` default): the partition map is the non-reproducible METIS artifact and
B2 is its source of truth, so we never rebuild it by default. **OGB regeneration is opt-in**
(`ALLOW_OGB=1`) and only kicks in on a B2 miss; otherwise a miss hard-fails rather than silently
starting a ~70 GB download. **METIS is opt-in too** (`ALLOW_METIS=1`) — a missing map hard-fails
unless it's set. Features/labels are pulled **only when `NEED_FEAT=1`** (real-feature/correctness
runs); the Section-1 perf sweep uses `--random_feat`, so `feat.npy`/`label.npy` (e.g. ~57 GB for
papers100M) are never fetched. See `provision_data.sh`'s header for all flags and
[[data-provisioning-default]].

If you *do* want OGB regeneration (e.g. to rebuild topology from scratch): `ALLOW_OGB=1 bash
.../provision_data.sh <graph> "2 4"`. OGB is **deterministic** — regenerated `feat.npy`/topology
are byte-identical with the same node ordering, so a previously-saved partition map still aligns.

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

### 5.1 The Section-1 sweep (gsplit vs DGL @ 0% cache)

`run_split_experiment.sh` above is for single/ad-hoc runs. The **dissertation
Section-1 grid** is driven by a dedicated, idempotent pair — do not hand-loop it:

- **`gen_settings.py`** enumerates the planned grid and diffs it against the
  consolidated results (`defense_latex/results/sweep_all.csv`), writing
  `sec1_settings.jsonl` with **only the cells not already measured**. The grid is
  two *separate* single-axis ablations per (graph, model, interconnect, system),
  meeting at the pivot (L=3, h=1024): a hidden sweep `h∈{256,1024,4096}` at L=3 and a
  depth sweep `L∈{2,3,4}` at h=1024. Systems are **`dgl` and `split` (gsplit, k=0),
  both at `--cache_size=0G`** — no quiver, no k>0. Protocol: 3 epochs, world_size 4,
  `--random_feat`, 1200 s/cell cap. (Edit the grid constants there, kept in sync with
  `defense_latex/gen_sec1_tables.py`.)
- **`run_settings.py`** iterates `sec1_settings.jsonl`, running each cell under
  `timeout`, recording `timeout`/`oom` **sentinel rows** (so a capped/OOM'd cell is
  never retried), and **skipping any cell already present** in `sweep_all.csv` or the
  log — so it resumes cleanly after an interruption. Each cell runs in its own process
  group (never a global `pkill`), and `--shard I/N` + `CUDA_VISIBLE_DEVICES` split the
  grid across concurrent workers (see `run_worker2.sh`, and §0b for the 8-GPU layout).
- **Consolidate + render:** merge the per-run CSVs into `sweep_all.csv`, then
  `defense_latex/gen_sec1_tables.py` regenerates `tables/sec1_products.tex` /
  `sec1_papers100M.tex` and `results/losing_cells.csv` from that single file.

The older `sunday_sweep.sh` / `sunday_ledger.md` describe a **superseded** stress-to-OOM
ladder (it ablated hidden and depth together); the grid above is the current design.

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
- A100-SXM capacity is frequently empty — use `cloud/runpod/acquire_pod.sh` (retries), not a single `pod create`.
