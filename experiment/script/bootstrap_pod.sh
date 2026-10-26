#!/usr/bin/env bash
# bootstrap_pod.sh  —  RUN ON A FRESH POD (as root).
#
# Takes a bare RunPod CUDA-12.4 pod to a working groot/Spara install:
# Miniconda -> conda env -> clone repo -> environment_setup.sh -> release build.
# Idempotent; logs to /root/overnight_logs/setup.log. Safe to re-run.
#
# Fetch + run on a fresh pod (repo is public):
#   curl -fsSL https://raw.githubusercontent.com/sandeep06011991/dgl_groot/final_dissertation/experiment/script/bootstrap_pod.sh | bash
# or copy this file over and: bash bootstrap_pod.sh
#
# Env: ENV_PATH (/home/groot_env), REPO_DIR (/home/dgl_groot), BRANCH (final_dissertation)
set -euo pipefail
ENV_PATH="${ENV_PATH:-/home/groot_env}"
REPO_DIR="${REPO_DIR:-/home/dgl_groot}"
BRANCH="${BRANCH:-final_dissertation}"
REPO_URL="${REPO_URL:-https://github.com/sandeep06011991/dgl_groot.git}"
mkdir -p /root/overnight_logs
exec > >(tee -a /root/overnight_logs/setup.log) 2>&1
echo "================ [$(date -u)] bootstrap_pod START ================"
export DEBIAN_FRONTEND=noninteractive

echo "---- [1/6] apt build tools ----"
apt-get update -y && apt-get install -y unzip build-essential ca-certificates wget rsync
# rclone for B2 push/restore (provision_data.sh / run_split_experiment.sh); official
# installer gives a newer build than apt. Best-effort — B2 steps no-op without it.
command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | bash || true

echo "---- [2/6] Miniconda (+ channel ToS) ----"
if [ ! -x /home/miniconda3/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /home/miniconda3
fi
source /home/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "---- [3/6] conda env $ENV_PATH ----"
[ -d "$ENV_PATH" ] || conda create -y -p "$ENV_PATH" python=3.11
conda activate "$ENV_PATH"

echo "---- [4/6] clone $REPO_URL ($BRANCH) ----"
[ -d "$REPO_DIR/.git" ] || git clone --recurse-submodules "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
git checkout "$BRANCH"
git submodule update --init --recursive

echo "---- [5/6] environment_setup.sh + release build ----"
ENV_PATH="$ENV_PATH" bash environment_setup.sh
source /home/miniconda3/etc/profile.d/conda.sh && conda activate "$ENV_PATH"
cd "$REPO_DIR" && ./build.sh release
pip install -q torchmetrics || true   # safety: trainers import it

echo "---- [6/6] verify ----"
export LD_LIBRARY_PATH="$REPO_DIR/third_party/build/lib:${LD_LIBRARY_PATH:-}"
python -c "import torch, dgl, torchmetrics, ogb; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('dgl',dgl.__path__)"
# make `conda` available in future (fresh) SSH logins
grep -q "miniconda3/etc/profile.d/conda.sh" /root/.bashrc 2>/dev/null || \
  echo "source /home/miniconda3/etc/profile.d/conda.sh" >> /root/.bashrc

echo "================ [$(date -u)] bootstrap_pod DONE ================"
echo "Activate with: source /home/miniconda3/etc/profile.d/conda.sh && conda activate $ENV_PATH"
