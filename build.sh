#!/usr/bin/env bash
# Build DGL/Spara from this checkout. Run this as often as you like during
# development — one-time environment provisioning lives in environment_setup.sh.
#
# Usage:
#   ./build.sh [debug|release] [--clean]
#
#   debug    (default) -O0 -g, assertions on. Use this for development.
#   release            optimized build; use this for performance measurements.
#   --clean            wipe the build dir and reconfigure from scratch.
#
# debug and release use separate build dirs (build/ and build-release/) so you
# can keep a release build for measurements alongside a debug build you develop
# against, without a full recompile every time you switch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# ---- Parse args ----
BUILD_TYPE="debug"
CLEAN=0
for arg in "$@"; do
  case "${arg}" in
    debug|release) BUILD_TYPE="${arg}" ;;
    --clean)       CLEAN=1 ;;
    -h|--help)     sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: ${arg}" >&2
       echo "Usage: $0 [debug|release] [--clean]" >&2; exit 1 ;;
  esac
done

if [ "${BUILD_TYPE}" = "release" ]; then
  BUILD_DIR="${SCRIPT_DIR}/build-release"
else
  BUILD_DIR="${SCRIPT_DIR}/build"
fi

# Target GPU architecture for nvcc. We build for the single arch of the GPU
# actually present on this machine, instead of DGL's full "All" list
# (sm_60/70/75/80/86/90 + PTX), which compiles every CUDA .cu file ~7x and
# dominates build time. The compute capability is read straight from nvidia-smi
# so the binary always matches the hardware (A100 -> sm_80, A40 -> sm_86, ...);
# nothing about the arch is hardcoded. We build the real binary AND the matching
# PTX for the detected arch, so no load-time JIT is needed on this GPU.
#
# Override (skips auto-detection) by exporting an explicit sm number:
#   CUDA_ARCH_BIN=86 [CUDA_ARCH_PTX=86]
if [ -z "${CUDA_ARCH_BIN:-}" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found, so the GPU compute capability can't be" >&2
    echo "  detected. Run build.sh on a node with the target GPU visible, or" >&2
    echo "  export CUDA_ARCH_BIN (e.g. =80 for A100, =86 for A40) and re-run." >&2
    exit 1
  fi
  # e.g. "8.6" -> "86"; take the first GPU (they are identical on our nodes).
  CUDA_ARCH_BIN="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -n1 | tr -d ' .')"
  if [ -z "${CUDA_ARCH_BIN}" ]; then
    echo "ERROR: nvidia-smi reported no GPU compute capability. Ensure a GPU is" >&2
    echo "  visible here, or export CUDA_ARCH_BIN (e.g. =86 for A40) and re-run." >&2
    exit 1
  fi
  echo "Detected GPU compute capability sm_${CUDA_ARCH_BIN}; building for that arch only."
fi
# DGL's cmake uses CUDA_ARCH_NAME=Manual to take explicit CUDA_ARCH_BIN/PTX.
CUDA_ARCH_NAME="Manual"
CUDA_ARCH_PTX="${CUDA_ARCH_PTX:-${CUDA_ARCH_BIN}}"

# ---- Resolve CUDA_HOME ----
# cmake/util/FindCUDA.cmake builds the CUDA library paths straight from
# $CUDA_HOME, e.g. ${CUDA_HOME}/lib/libcudart.so. If CUDA_HOME is empty when
# cmake configures, it bakes the bogus path /lib/libcudart.so into build.ninja
# and the link fails with:
#   ninja: error: '/lib/libcudart.so', needed by 'libdgl.so', missing
# Default to the active conda env, which is where environment_setup.sh installs
# the CUDA toolkit.
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -e "${CONDA_PREFIX}/lib/libcudart.so" ]; then
    CUDA_HOME="${CONDA_PREFIX}"
  elif [ -e /usr/local/cuda/lib/libcudart.so ] || [ -e /usr/local/cuda/lib64/libcudart.so ]; then
    CUDA_HOME="/usr/local/cuda"
  fi
fi
export CUDA_HOME="${CUDA_HOME:-}"

if [ -z "${CUDA_HOME}" ] || [ ! -e "${CUDA_HOME}/lib/libcudart.so" ]; then
  echo "ERROR: could not locate a CUDA toolkit with lib/libcudart.so." >&2
  echo "  CUDA_HOME='${CUDA_HOME}'" >&2
  echo "  Activate the conda env (so CONDA_PREFIX is set), or export CUDA_HOME" >&2
  echo "  to a toolkit that contains lib/libcudart.so, then re-run." >&2
  exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "Build type: ${BUILD_TYPE}    Build dir: ${BUILD_DIR}"

# ---- Third-party deps (GKlib/METIS/oneTBB/NCCL), built once into third_party/build ----
# Check for the actual headers/libs DGL links against, not just the directory:
# a partial init.sh (e.g. GKlib/METIS failing while TBB/NCCL succeed) leaves the
# dir present but DGL's compile then fails on missing GKlib.h / metis.h.
TP="${SCRIPT_DIR}/third_party/build"
third_party_ready() {
  [ -f "${TP}/include/GKlib.h" ] \
    && [ -f "${TP}/include/metis.h" ] \
    && [ -f "${TP}/include/nccl.h" ] \
    && ls "${TP}"/lib/libmetis.*  >/dev/null 2>&1 \
    && ls "${TP}"/lib/libnccl.so* >/dev/null 2>&1 \
    && ls "${TP}"/lib/libtbb.*    >/dev/null 2>&1
}
if ! third_party_ready; then
  echo "Third-party deps incomplete in ${TP}; building them via init.sh ..."
  bash "${SCRIPT_DIR}/init.sh"
  if ! third_party_ready; then
    echo "ERROR: init.sh ran but third-party deps are still incomplete." >&2
    echo "  Expected GKlib.h, metis.h, nccl.h in ${TP}/include and the" >&2
    echo "  matching libs in ${TP}/lib. Check init.sh output above." >&2
    exit 1
  fi
fi
echo "Third-party deps present in ${TP}."

# ---- Configure + build ----
if [ "${CLEAN}" = "1" ]; then
  rm -rf "${BUILD_DIR}"
fi
echo "CUDA arch: ${CUDA_ARCH_NAME} (sm_${CUDA_ARCH_BIN}, ptx ${CUDA_ARCH_PTX})"
cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" -GNinja \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DBUILD_TYPE="${BUILD_TYPE}" \
  -DCUDA_ARCH_NAME="${CUDA_ARCH_NAME}" \
  -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
  -DCUDA_ARCH_PTX="${CUDA_ARCH_PTX}" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "${BUILD_DIR}" -j

# ---- Install the python package ----
(cd "${SCRIPT_DIR}/python" && pip install .)

echo "Build (${BUILD_TYPE}) complete. Verify with: python -c 'import dgl; print(dgl.__path__)'"
