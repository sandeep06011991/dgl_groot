#!/usr/bin/env bash
# run_profile.sh  (ON THE POD)  —  validate the simulator's compute + feature-load
# cost against ground truth, swept over model, hidden size, and feature size, with
# an nsys cross-check.
#
#   Usage:  run_profile.sh [graph] [depth] ["hiddens"] ["models"]
#   e.g.    run_profile.sh products 3 "64 128 256 512" "sage gat"
set -uo pipefail
GRAPH="${1:-products}"
DEPTH="${2:-3}"
HIDDENS="${3:-64 128 256 512}"
MODELS="${4:-sage gat}"
: "${data_dir:=/root/graph}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PD="$REPO/experiment"
export LD_LIBRARY_PATH="$REPO/third_party/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PD:${PYTHONPATH:-}"
: "${ENV_PATH:=/home/groot_env}"
if [ -f /home/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/miniconda3/etc/profile.d/conda.sh; conda activate "$ENV_PATH" 2>/dev/null || true
fi
cd "$PD"
WEIGHT_WS=2 bash "$PD/script/provision_data.sh" "$GRAPH" 2 >/dev/null 2>&1 || true
fan=$(printf '15%.0s,' $(seq 1 "$DEPTH")); fan=${fan%,}

echo "===== compute (fused vs sim-style) + feature-load sweep: $GRAPH depth=$DEPTH fanout=$fan ====="
for model in $MODELS; do
  for hid in $HIDDENS; do
    python3 profile_model.py --graph_name="$GRAPH" --data_dir="$data_dir" \
      --fanouts="$fan" --hid_size="$hid" --model="$model" --batch_size=1024
  done
done

echo "===== nsys cross-check (sage, hid=256) ====="
if command -v nsys >/dev/null 2>&1; then
  nsys profile -f true -o "/root/overnight_logs/nsys_${GRAPH}_d${DEPTH}" --stats=true \
    python3 profile_model.py --graph_name="$GRAPH" --data_dir="$data_dir" \
      --fanouts="$fan" --hid_size=256 --model=sage --batch_size=1024 2>&1 \
    | grep -iE "Time \(%\)|gemm|sgemm|aggregat|GraphSAGE|reduce|Kernel Summary|cutlass|elementwise" | head -25 || true
else
  echo "nsys not available"
fi
echo "PROFILE_ALL_DONE"
