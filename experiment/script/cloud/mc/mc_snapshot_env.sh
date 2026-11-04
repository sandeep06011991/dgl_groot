#!/usr/bin/env bash
# mc_snapshot_env.sh — RUN ON A BUILT MC VM. Snapshot the built conda env + repo/build
# to B2 so a future MC VM can RESTORE instead of rebuilding (~40 min + ~$7 saved).
#
# The conda env + compiled .so CANNOT run off the geesefs persistent volume and are NOT
# relocatable, so this is a tarball keyed to the EXACT paths + GPU arch/driver:
#   /home/groot_env  +  /home/dgl_groot   (A100 sm_80, MC image 84).
# Restore is therefore MC->MC only; MC->RunPod (different paths/base) must rebuild.
#
# Prereq on the VM: rclone with the `gsplit-b2` remote configured (bootstrap installs
# rclone; configure the remote from ~/.config/groot/b2.env). Verify the env imports
# before snapshotting — never persist a broken build.
#
#   Usage (on the VM):  bash mc_snapshot_env.sh            # create + push
#                       KEY=mc_a100_sm80 bash mc_snapshot_env.sh
set -uo pipefail
ENV_PATH="${ENV_PATH:-/home/groot_env}"
REPO_DIR="${REPO_DIR:-/home/dgl_groot}"
KEY="${KEY:-mc_a100_sm80}"
REMOTE="${REMOTE:-gsplit-b2:gsplit-data/groot/env}"
TARBALL="/home/${KEY}.tar.zst"

source /home/miniconda3/etc/profile.d/conda.sh && conda activate "$ENV_PATH"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/build/lib:${LD_LIBRARY_PATH:-}"

echo "==== verify the build imports before snapshotting ===="
python -c "import torch, dgl, torchmetrics; print('ok torch',torch.__version__,'cuda',torch.cuda.is_available()); print('dgl',dgl.__path__)" \
  || { echo "ERROR: env does not import — refusing to snapshot a broken build"; exit 1; }

echo "==== tar + zstd (env + repo/build) ===="
command -v zstd >/dev/null 2>&1 || conda install -y -q zstd || true
# -C / so paths restore as /home/groot_env and /home/dgl_groot exactly.
tar -C / --zstd -cf "$TARBALL" \
    home/groot_env \
    home/dgl_groot
ls -lh "$TARBALL"

echo "==== push to B2 ($REMOTE) ===="
rclone copyto "$TARBALL" "$REMOTE/${KEY}.tar.zst" --progress
echo "snapshot pushed: $REMOTE/${KEY}.tar.zst"
echo "restore on a fresh MC VM:  rclone copyto $REMOTE/${KEY}.tar.zst /home/${KEY}.tar.zst && tar -C / --zstd -xf /home/${KEY}.tar.zst"
