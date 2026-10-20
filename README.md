# Artifacts for Spara

This repository contains the source code of Spara, an efficient distributed GNN training system. Scripts to reproduce the Figures and Tables in the paper are also provided.

# Project Structure
Spara is highly integrated into DGL. The Python part for Spara is in the `python/dgl/dev` directory whereas the cpp source code are in the `src/spara` directory.

# Software Requirements

- Linux with an NVIDIA GPU. Developed and tested on A40 (`sm_86`) and A100
  (`sm_80`); runs end-to-end on a RunPod pod as well as the Unity cluster.
- An NVIDIA driver that supports CUDA 12.4.
- `conda` (Miniconda/Anaconda) on `PATH`.

The CUDA 12.4 toolkit, the C++ compiler, PyTorch 2.4.0 (cu124) and PyG are all
installed into a conda environment by the setup script below, so you do **not**
need a system-wide CUDA install.

# How to build

Spara builds from source in two steps: a one-time environment setup, then the
DGL/Spara compile (which you re-run during development).

## 1. One-time environment setup

From the repo root:
```bash
bash environment_setup.sh
```
This creates (or reuses) a conda env with CUDA 12.4, `gxx`, PyTorch 2.4.0
(cu124) and PyG, builds the third-party dependencies (GKlib, METIS, oneTBB,
NCCL) via `init.sh`, and then does the first Spara build by calling `build.sh`.
It is designed to run end-to-end on a fresh GPU box such as a RunPod pod.

By default it installs into the currently-activated conda env; override the
target with `ENV_PATH=/path/to/env`, or set `ENV_NAME` to create a sibling env.

## 2. Build / rebuild during development

The compile lives in its own script so you can re-run it as often as you like:
```bash
./build.sh                 # debug build   -> build/          (default; for development)
./build.sh release         # release build -> build-release/  (for performance measurements)
./build.sh debug --clean   # wipe the build dir and reconfigure from scratch
```
`build.sh` auto-detects `CUDA_HOME` from the active conda env and auto-detects
the GPU's compute capability from `nvidia-smi`, building **only** for the arch
actually present (e.g. A40 → `sm_86`, A100 → `sm_80`). On a machine with no
visible GPU (e.g. a cluster login node), export `CUDA_ARCH_BIN` (e.g. `=80` for
A100, `=86` for A40) to choose the target arch.

## 3. Verify the install

```bash
python -c "import dgl; print(dgl.__path__)"
```
This should print the `dgl` package path inside your conda env, e.g.
`['/home/groot_env/lib/python3.11/site-packages/dgl']`.

For more build detail and the C++/Python code architecture, see
[`.claude/groot.md`](./.claude/groot.md).

# Download Dataset

The datasets consist of the graph topology data and the partition maps.

**Option I — generate from raw datasets (recommended):** see
[experiment/prepare_dataset/README.md](./experiment/prepare_dataset/README.md).
(Processing the large graphs can take several days.)

**Option II — preprocessed datasets (stale):** the original SC24 artifact
published preprocessed graphs and partition maps on Amazon S3 via `download.sh`.
**Those S3 artifacts are no longer maintained and the links are likely dead** —
use Option I to regenerate the data.

Set the `dataset_dir` variable in [`env.sh`](./experiment/script/env.sh) to
choose where data is stored. The graph topology lands in `${dataset_dir}/graph`
and the partition maps in `${dataset_dir}/graph/partition_map`; the
`partition_map` directory must live inside the `graph` directory so the
dataloader can find the maps correctly.

# Artifact Execution

<ul>
<li>S1 The first step is to obtain the input datasets, which
include the graph topology data and partition maps. We
provide pre-processed datasets that can be downloaded
from Amazon S3. The script download.sh can be used
to download those files automatically. Alternatively, you
can use scripts in the repository to generate the prepared
datasets. Notice that this would take several days. </li>
<li>S2 After obtaining the prepared datasets, you can run
the main experiment running the bash scripts experi-
ment/script/main.sh. This script runs all the baselines and
generates the log file. (Expected time: 120 min, depends
on S1.)</li>
<li>S3 Postprocess the logs from S2. Generate Figures 3 using
the notebook plot/time breakdown. (depends on S2). </li>
<li>S4 Postprocess the logs from S2 using the notebook
plot/main to generate Table 3. (depends on S2). </li>
<li> S5 Run the sampling simulation
experiment/sample main, to generate the varying
edges computed and features loaded for varying batch
sizes and graphs to generate the datapoints in table 1.
(Expected time: 30min, depends on S1) </li>
<li> S6 Run the script experiment/scripts/ablation.sh to run
all the ablation experiments on papers graph. (depends
on S1</li>
<li>S7 Postproces the logs generated in the previous step with
the jupyter notebook plot/final_ablation (Expected
time: 3 hours, depends on S6) </li>
<li> S8 : Run the python file experiment/simulate main for
the friendster and various partitioning schemes to collect
the workload characteristics. (command line arguments
details provided in the README).(depends on S1) </li>
<li> S9 : Post process the workloads generated in step S8 with
the notebook plot/simulation plot to generate Figures
5 </li>
<li> S10 : Run the script experiment/partition ablation to col-
lect the training logs for varying partitioning strategies.
(depends S1) </li>
<li> S11 : Run the notebook plot/partitioning to generate table
4 from the training logs. (depends on S10) </li>
</ul>







## Experiment Result Generation

## Data Analysis and Visualization


