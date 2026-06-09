#!/usr/bin/env bash
# Build a clean conda environment for DGL/Spara against PyTorch 2.4.0 + CUDA 12.4,
# build DGL from this checkout, and verify torch/dgl can both use the GPU.
#
# Assumes `conda` is already installed and on PATH. Designed to run end-to-end on a
# fresh GPU box (e.g. a RunPod pod) as well as on Unity.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# ---- Configuration (override via env vars) ----
ENV_NAME="${ENV_NAME:-dgl_groot_cu124}"
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
if ! command -v conda &>/dev/null; then
  echo "conda not found on PATH. This script assumes conda is already installed." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "${ENV_PATH}" ]; then
  conda create -y -p "${ENV_PATH}" "python=${PYTHON_VERSION}"
else
  echo "Env already exists at ${ENV_PATH}, reusing it."
fi
conda activate "${ENV_PATH}"

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
log "Building DGL/Spara from source"
rm -rf "${SCRIPT_DIR}/build"
cmake -B "${SCRIPT_DIR}/build" -S "${SCRIPT_DIR}" -GNinja \
  -DCMAKE_BUILD_TYPE=debug -DBUILD_TYPE=debug \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "${SCRIPT_DIR}/build" -j
(cd "${SCRIPT_DIR}/python" && pip install .)

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

# ---- 7. Verification: torch + dgl on GPU ----
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
