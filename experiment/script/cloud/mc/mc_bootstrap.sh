#!/usr/bin/env bash
# mc_bootstrap.sh  —  RUN ON A FRESH MASSED COMPUTE VM (user `Ubuntu`, has sudo).
#
# Massed Compute equivalent of cloud/runpod/bootstrap_pod.sh. MC's image 84
# ("Ubuntu Server 22.04 w/ drivers") is a bare Ubuntu VM: it ships the NVIDIA
# *kernel driver* only, NOT a CUDA toolkit and NOT a conda/python stack. So, as on
# RunPod, the whole toolchain (CUDA 12.4 toolkit + gxx + cmake/ninja + torch + the
# DGL/Spara build) is installed INTO the conda env by environment_setup.sh — the
# conda env is self-contained; the system supplies only the driver. The only real
# differences from the RunPod script are: (1) we are NOT root (apt + making /home
# writable need sudo), and (2) logs/paths live under /home for the `Ubuntu` user.
#
# Keeps the SAME install paths the runner hardcodes (run_worker2.sh):
#   REPO=/home/dgl_groot  ENV=/home/groot_env  MINICONDA=/home/miniconda3  DATA=/home/data
#
# Fetch + run on a fresh MC VM (repo is public):
#   curl -fsSL https://raw.githubusercontent.com/sandeep06011991/dgl_groot/final_dissertation/experiment/script/cloud/mc/mc_bootstrap.sh | bash
# or scp this file over and: bash mc_bootstrap.sh
#
# FAST-PATH (skip the ~40-min build): if a prebuilt env tarball is reachable — on a
# persistent-storage VOLUME mounted at /persistent/<name>/groot/env/<KEY>.tar.zst, or
# on B2 — it is copied to local /home, untarred, and verified; the build is then
# skipped. Falls back to the full build when neither is present (unchanged behavior).
# Set SKIP_RESTORE=1 to force a clean build. See mc_volume.sh and RUNBOOK §0b.
#
# Env: ENV_PATH (/home/groot_env) REPO_DIR (/home/dgl_groot) DATA_DIR (/home/data)
#      BRANCH (final_dissertation) SNAPSHOT_KEY (mc_a100_sm80) SKIP_RESTORE (0)
set -euo pipefail

ENV_PATH="${ENV_PATH:-/home/groot_env}"
REPO_DIR="${REPO_DIR:-/home/dgl_groot}"
DATA_DIR="${DATA_DIR:-/home/data}"
MINICONDA="${MINICONDA:-/home/miniconda3}"
BRANCH="${BRANCH:-final_dissertation}"
SNAPSHOT_KEY="${SNAPSHOT_KEY:-mc_a100_sm80}"
SKIP_RESTORE="${SKIP_RESTORE:-0}"
REPO_URL="${REPO_URL:-https://github.com/sandeep06011991/dgl_groot.git}"
LOG_DIR="${LOG_DIR:-/home/overnight_logs}"

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
export DEBIAN_FRONTEND=noninteractive

# --- make /home writable for this user so conda/miniconda/repo can be created there,
#     and pre-create the dirs the runner expects. (MC /home is the large VM disk.)
$SUDO chown "$(id -un):$(id -gn)" /home 2>/dev/null || true
mkdir -p "$DATA_DIR" "$LOG_DIR"

exec > >(tee -a "$LOG_DIR/setup.log") 2>&1
echo "================ [$(date -u)] mc_bootstrap START (user=$(id -un)) ================"
echo "driver check:"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || \
  echo "WARN: nvidia-smi failed — driver may be missing (build arch detection needs it)"

echo "---- [1/6] apt build tools (sudo) ----"
$SUDO apt-get update -y
$SUDO apt-get install -y unzip build-essential ca-certificates wget rsync git zstd
# rclone for B2 push/restore (provision_data.sh); official installer = newer than apt.
command -v rclone >/dev/null 2>&1 || { curl -fsSL https://rclone.org/install.sh | $SUDO bash || true; }

echo "---- [2/6] Miniconda (+ channel ToS) ----"
if [ ! -x "$MINICONDA/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -f -p "$MINICONDA"
fi
source "$MINICONDA/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# ---- [FAST-PATH] restore a prebuilt env instead of building (volume first, then B2) ----
# A restored tarball supplies BOTH /home/groot_env and /home/dgl_groot (repo+build), so a
# success lets us skip steps [3]-[5]. restore_tarball untars a LOCAL copy (the env/.so
# cannot run off the geesefs volume mount) and only returns 0 if the env imports.
RESTORED=0
restore_tarball() {   # $1 = local path to a <KEY>.tar.zst
  echo ">> restoring prebuilt env from $1"
  tar -C / --zstd -xf "$1" || return 1
  source "$MINICONDA/etc/profile.d/conda.sh"; conda activate "$ENV_PATH" || return 1
  export LD_LIBRARY_PATH="$REPO_DIR/third_party/build/lib:${LD_LIBRARY_PATH:-}"
  python -c "import torch,dgl,torchmetrics; print('restored dgl',dgl.__path__)" || return 1
  return 0
}
if [ "$SKIP_RESTORE" != "1" ]; then
  LOCAL_TB="/home/$SNAPSHOT_KEY.tar.zst"
  # (a) persistent-storage VOLUME: /persistent/<name>/groot/env/<KEY>.tar.zst
  VOL_DIR="${VOL_MOUNT:-$(ls -d /persistent/*/ 2>/dev/null | head -1)}"; VOL_DIR="${VOL_DIR%/}"
  VOL_TB="${VOL_DIR:+$VOL_DIR/groot/env/$SNAPSHOT_KEY.tar.zst}"
  if [ -n "${VOL_TB:-}" ] && [ -f "$VOL_TB" ]; then
    echo "---- [FAST-PATH] found env tarball on volume: $VOL_TB ----"
    cp "$VOL_TB" "$LOCAL_TB" && restore_tarball "$LOCAL_TB" && RESTORED=1 || echo "WARN: volume restore failed; will try B2 / build"
  fi
  # (b) B2 fallback: gsplit-data/groot/env/<KEY>.tar.zst
  if [ "$RESTORED" = 0 ] && command -v rclone >/dev/null 2>&1 && \
     rclone lsf "gsplit-b2:gsplit-data/groot/env/$SNAPSHOT_KEY.tar.zst" >/dev/null 2>&1; then
    echo "---- [FAST-PATH] found env tarball on B2 ----"
    rclone copyto "gsplit-b2:gsplit-data/groot/env/$SNAPSHOT_KEY.tar.zst" "$LOCAL_TB" --progress \
      && restore_tarball "$LOCAL_TB" && RESTORED=1 || echo "WARN: B2 restore failed; will build"
  fi
fi

if [ "$RESTORED" = 1 ]; then
  echo "---- [3-5/6] SKIPPED (restored prebuilt env; set SKIP_RESTORE=1 to force a build) ----"
  cd "$REPO_DIR"
else
  echo "---- [3/6] conda env $ENV_PATH ----"
  [ -d "$ENV_PATH" ] || conda create -y -p "$ENV_PATH" python=3.11
  conda activate "$ENV_PATH"

  echo "---- [4/6] clone $REPO_URL ($BRANCH) ----"
  [ -d "$REPO_DIR/.git" ] || git clone --recurse-submodules "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
  git fetch origin "$BRANCH" && git checkout "$BRANCH" && git pull --ff-only origin "$BRANCH"
  git submodule update --init --recursive

  echo "---- [5/6] environment_setup.sh (CUDA toolkit + torch into the env) + release build ----"
  ENV_PATH="$ENV_PATH" bash environment_setup.sh
  source "$MINICONDA/etc/profile.d/conda.sh" && conda activate "$ENV_PATH"
  cd "$REPO_DIR" && ./build.sh release
  pip install -q torchmetrics || true   # trainers import it
fi

echo "---- [6/6] verify ----"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/build/lib:${LD_LIBRARY_PATH:-}"
python -c "import torch, dgl, torchmetrics, ogb; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('dgl',dgl.__path__)"
# make conda + LD_LIBRARY_PATH available in future (fresh) SSH logins
BRC="$HOME/.bashrc"
grep -q "miniconda3/etc/profile.d/conda.sh" "$BRC" 2>/dev/null || {
  echo "source $MINICONDA/etc/profile.d/conda.sh" >> "$BRC"
  echo "export LD_LIBRARY_PATH=$REPO_DIR/third_party/build/lib:\${LD_LIBRARY_PATH:-}" >> "$BRC"
}

echo "================ [$(date -u)] mc_bootstrap DONE ================"
echo "Activate:  source $MINICONDA/etc/profile.d/conda.sh && conda activate $ENV_PATH"
echo "Data dir:  $DATA_DIR   Repo: $REPO_DIR"
