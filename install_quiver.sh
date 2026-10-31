#!/usr/bin/env bash
# Install torch-quiver (the "quiver" / "dist_cache" baseline) into the active
# groot conda env (torch 2.4.0 + CUDA 12.4). Run AFTER environment_setup.sh,
# with the env activated (CONDA_PREFIX set) and a GPU visible.
#
# torch-quiver was last tested upstream on torch 1.x / CUDA 10-11, so it needs
# three fixes to build against the groot env. This script applies them to a
# working copy of the upstream quiver-team/torch-quiver (vendored at
# third_party/torch-quiver, chosen over the dist_cache fork for stability) and
# installs it. Verified on RunPod A100 (sm_80), env at /home/groot_env.
#
#   ./install_quiver.sh [path-to-torch-quiver-checkout]
#
# With no argument it uses the third_party/torch-quiver submodule, checking it
# out at its PINNED commit via `git submodule update --init` if it is empty
# (a fresh git worktree or a non-recursive clone leaves submodules empty).
set -euo pipefail

: "${CUDA_HOME:=${CONDA_PREFIX:?activate the groot conda env first}}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
# A100=80, A40=86; auto-detect so the extension matches the GPU.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .' | sed 's/\(.\)\(.\)/\1.\2/')}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
QUIVER_SRC="${1:-${SCRIPT_DIR}/third_party/torch-quiver}"
if [ ! -f "${QUIVER_SRC}/setup.py" ] && [ "${QUIVER_SRC}" = "${SCRIPT_DIR}/third_party/torch-quiver" ]; then
  echo "torch-quiver submodule not checked out; initializing it at the pinned commit."
  git -C "${SCRIPT_DIR}" submodule update --init --recursive third_party/torch-quiver || true
fi
if [ ! -f "${QUIVER_SRC}/setup.py" ]; then
  echo "torch-quiver source unavailable at ${QUIVER_SRC}; cloning upstream (HEAD)."
  rm -rf "${QUIVER_SRC}"
  git clone --recurse-submodules https://github.com/quiver-team/torch-quiver.git "${QUIVER_SRC}"
fi

echo "=== build deps: ninja + PyG (quiver.pyg needs torch_geometric AND its C++ exts) ==="
pip install ninja torch_geometric
# quiver.pyg.GraphSageSampler imports torch_sparse/torch_scatter; install the
# prebuilt PyG extension wheels matching torch 2.4.0 + cu124.
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

echo "=== patch fork for CUDA 12 / torch 2.4 ==="
QUIVER_SRC="${QUIVER_SRC}" python - <<'PYEOF'
import os, re
src = os.environ["QUIVER_SRC"]

# (1) CUDA 12's thrust no longer pulls these in transitively.
thrust = ("#include <thrust/sort.h>\n#include <thrust/unique.h>\n"
          "#include <thrust/sequence.h>\n#include <thrust/execution_policy.h>\n"
          "#include <thrust/device_ptr.h>\n")
for rel in ["srcs/cpp/include/quiver/quiver.cu.hpp",
            "srcs/cpp/include/quiver/reindex.cu.hpp",
            "srcs/cpp/src/quiver/cuda/quiver_sample.cu"]:
    f = os.path.join(src, rel); s = open(f).read()
    if "thrust/sort.h" in s:
        continue
    m = re.search(r'^\s*#pragma once\s*\n', s)
    s = (s[:m.end()] + thrust + s[m.end():]) if m else thrust + s
    open(f, "w").write(s); print("thrust-patched:", rel)

# (2) nccl.h/libnccl come from the nvidia-nccl-cu12 wheel, not $CUDA_HOME.
sp = os.path.join(src, "setup.py"); s = open(sp).read()
anchor = "        extra_cxx_flags += ['-DHAVE_CUDA']\n"
inject = ("        nccl_home = os.getenv('QUIVER_NCCL_HOME')\n"
          "        if nccl_home:\n"
          "            include_dirs += [os.path.join(nccl_home, 'include')]\n"
          "            library_dirs += [os.path.join(nccl_home, 'lib')]\n"
          "            libraries += ['nccl']\n")
if "QUIVER_NCCL_HOME" not in s and anchor in s:
    open(sp, "w").write(s.replace(anchor, anchor + inject)); print("setup.py patched for nccl")
PYEOF

# (3) the wheel ships only libnccl.so.2; create the -lnccl symlink.
NCCL_HOME="$(python -c 'import os,nvidia.nccl as n; print(os.path.dirname(n.__file__))')"
export QUIVER_NCCL_HOME="${NCCL_HOME}"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
[ -e "${NCCL_HOME}/lib/libnccl.so" ] || ln -s "${NCCL_HOME}/lib/libnccl.so.2" "${NCCL_HOME}/lib/libnccl.so"

echo "=== build + install (no build isolation: setup.py imports torch) ==="
( cd "${QUIVER_SRC}" && QUIVER_ENABLE_CUDA=1 pip install -v --no-build-isolation . )

echo "=== verify ==="
python - <<'EOF'
import quiver, quiver.pyg
from quiver.pyg import GraphSageSampler
assert all(hasattr(quiver, n) for n in ("CSRTopo", "Feature", "init_p2p"))
print("quiver OK:", quiver.__file__)
EOF
echo "install_quiver.sh: done"
