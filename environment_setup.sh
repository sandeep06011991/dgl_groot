#!/usr/bin/env bash
# One-time environment provisioning for DGL/Spara: create (or reuse) a conda env
# with PyTorch 2.4.0 + CUDA 12.4, build the third-party deps, then compile and
# verify DGL from this checkout.
#
# This is the *setup* script, run once per machine/env. The actual DGL compile is
# delegated to build.sh, which you re-run as often as you like during development
# (./build.sh debug | ./build.sh release). This script just calls it for you once.
#
# Assumes `conda` is already installed and on PATH. Designed to run end-to-end on a
# fresh GPU box (e.g. a RunPod pod) as well as on Unity.
#
# By default it installs into the currently-activated conda env. Override the
# target env with ENV_PATH=/path/to/env, or set ENV_NAME to create a sibling env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# ---- Configuration (override via env vars) ----
ENV_NAME="${ENV_NAME:-dgl_groot_cu124}"
# Default to the already-active conda env (unless it's base); otherwise fall back
# to a sibling conda_environment/<ENV_NAME> that this script will create.
if [ -z "${ENV_PATH:-}" ] && [ -n "${CONDA_PREFIX:-}" ] \
   && [ "$(basename "${CONDA_PREFIX}")" != "base" ]; then
  ENV_PATH="${CONDA_PREFIX}"
fi
ENV_PATH="${ENV_PATH:-${SCRIPT_DIR}/../conda_environment/${ENV_NAME}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.4.0}"
GXX_VERSION="${GXX_VERSION:-11.3}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
PYG_WHEEL_URL="${PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu124.html}"
SKIP_PYG="${SKIP_PYG:-0}"

log() { printf '\n=== %s ===\n\n' "$*"; }

# ---- 1. Conda environment ----
log "Setting up conda env at ${ENV_PATH}"
# Make conda usable in this *non-interactive* shell. `conda` is normally only
# wired up by the 'conda init' block in ~/.bashrc, which bash does NOT read
# unless started with -i. So a plain `bash environment_setup.sh` would otherwise
# die here even though conda is installed. Locate conda's base from the env that
# `conda init` leaves behind (or common install paths) and source its profile
# script, so `conda activate` works without requiring `bash -i`.
if ! command -v conda &>/dev/null; then
  for base in "${_CONDA_ROOT:-}" "${CONDA_EXE:+$(dirname "$(dirname "${CONDA_EXE}")")}" \
              "${HOME}/miniconda3" "${HOME}/anaconda3" /home/miniconda3 /opt/conda; do
    if [ -n "${base}" ] && [ -f "${base}/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "${base}/etc/profile.d/conda.sh"
      break
    fi
  done
fi
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install Miniconda first, then re-run this script." >&2
  echo "  (If conda IS installed, run it from a shell where conda is initialized," >&2
  echo "   e.g. 'bash -i environment_setup.sh'.)" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "${ENV_PATH}" ]; then
  conda create -y -p "${ENV_PATH}" "python=${PYTHON_VERSION}"
else
  echo "Env already exists at ${ENV_PATH}, reusing it."
fi
conda activate "${ENV_PATH}"

# Ensure the env owns its own Python + pip, so the env's pip (not a system one on
# PATH) is used below and `pip install .` installs DGL *into this env* — the whole
# point of isolating from the container. Also covers reusing an env that only had
# the CUDA toolkit installed.
conda install -y "python=${PYTHON_VERSION}" pip
echo "Using pip: $(command -v pip)"

# ---- 2. CUDA toolkit + compiler + build tools (inside the env, self-contained) ----
log "Installing CUDA ${CUDA_VERSION} toolkit, gxx==${GXX_VERSION}, cmake, ninja"
if ! conda install -y -c "nvidia/label/cuda-${CUDA_VERSION}" cuda-toolkit; then
  echo "nvidia/label/cuda-${CUDA_VERSION} not found, falling back to nvidia channel with version pin" >&2
  conda install -y -c nvidia -c conda-forge "cuda-toolkit=${CUDA_VERSION%.*}"
fi
conda install -y -c conda-forge "gxx==${GXX_VERSION}"
pip install -U cmake ninja

export CUDA_HOME="${ENV_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
echo "CUDA_HOME=${CUDA_HOME}"
nvcc --version

# ---- 3. PyTorch 2.4.0 / cu124 ----
log "Installing torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} (cu124)"
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda available', torch.cuda.is_available())"

# ---- 4. Third-party deps: GKlib, METIS, oneTBB, NCCL ----
log "Building third-party dependencies via init.sh"
bash "${SCRIPT_DIR}/init.sh"

# ---- 5. Build + install DGL/Spara from this checkout ----
# The compile itself lives in build.sh so it can be re-run during development.
# Pass CUDA_HOME through so build.sh links against this env's CUDA toolkit.
log "Building DGL/Spara from source (debug) via build.sh"
CUDA_HOME="${CUDA_HOME}" bash "${SCRIPT_DIR}/build.sh" debug --clean

# ---- 6. PyG extension stack (best-effort; nothing in groot/Spara depends on it) ----
if [ "${SKIP_PYG}" != "1" ]; then
  log "Installing PyG extension libraries (best-effort, matching torch-${TORCH_VERSION}+cu124)"
  if pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "${PYG_WHEEL_URL}" \
     && pip install torch_geometric; then
    echo "PyG stack installed successfully."
  else
    echo "WARNING: PyG install failed - continuing without it (not required by groot/Spara)." >&2
  fi
else
  echo "Skipping PyG install (SKIP_PYG=1)"
fi

# ---- 7. ogb (Open Graph Benchmark) ----
# Required by the dataset-prep scripts, e.g. experiment/prepare_dataset/get_npgraph.py
# (`from ogb.nodeproppred import DglNodePropPredDataset`).
log "Installing ogb (Open Graph Benchmark, used by the dataset-prep scripts)"
pip install ogb

# torchmetrics is imported by the node-prediction trainers (nodepred/*_trainer.py);
# without it, training aborts with ModuleNotFoundError: No module named 'torchmetrics'.
log "Installing torchmetrics (required by experiment/nodepred trainers)"
pip install torchmetrics

# ---- 8. Verification: torch + dgl on GPU ----
log "Verifying torch + dgl on GPU"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/third_party/build/lib:${LD_LIBRARY_PATH:-}"
python - <<'EOF'
import torch
import dgl

x = torch.tensor(0).to(0)
g = dgl.rand_graph(10, 10).to(0)
print("torch tensor device:", x.device)
print("dgl graph device:", g.device)
print("ENVIRONMENT SETUP OK")
EOF

log "Done. Activate this environment with: conda activate ${ENV_PATH}"
