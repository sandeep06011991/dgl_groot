#!/usr/bin/env bash
# run_simulation.sh  (ON THE POD)  —  run the hybrid-parallelism cost-model
# simulator across sampling depths for one graph+model, writing an e2e-style CSV
# (same Config columns as the measured runs, plus predicted per-k costs).
#
#   Usage:  run_simulation.sh <graph> [world_size] ["depths"] [num_epoch] [model] [hid_size]
#   e.g.    run_simulation.sh products 2 "3 4 5" 1 sage 256
#
# Output: experiment/logs/<graph>_<model>_sim_w<ws>.csv  (one row per depth x split-point k).
set -uo pipefail

GRAPH="${1:?usage: run_simulation.sh <graph> [world_size] [depths] [num_epoch] [model] [hid_size]}"
WS="${2:-2}"
DEPTHS="${3:-3 4 5}"
NUM_EPOCH="${4:-1}"
MODEL="${5:-sage}"
HID="${6:-256}"
: "${data_dir:=/root/graph}"
: "${MACHINE_NAME:=a100sxm}"; export MACHINE_NAME

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PD="$REPO/experiment"
export LD_LIBRARY_PATH="$REPO/third_party/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PD:${PYTHONPATH:-}"
: "${ENV_PATH:=/home/groot_env}"
if [ -f /home/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV_PATH" 2>/dev/null || true
fi

# ensure the data + the w<WS> partition map exist (same as the measured runner)
WEIGHT_WS="$WS" bash "$PD/script/provision_data.sh" "$GRAPH" "$WS"

cd "$PD"
CSV="logs/${GRAPH}_${MODEL}_sim_w${WS}.csv"
rm -f "$CSV"   # fresh file; rows appended per depth below
for d in $DEPTHS; do
  fan=$(printf '15%.0s,' $(seq 1 "$d")); fan=${fan%,}
  echo "== SIM $GRAPH model=$MODEL ws=$WS depth=$d fanout=$fan =="
  python3 get_best_strategy.py --graph_name="$GRAPH" --data_dir="$data_dir" \
    --fanouts="$fan" --world_size="$WS" --hid_size="$HID" --model="$MODEL" \
    --node_weight=dst --edge_weight=freq --bal=xbal --nvlink --num_epoch="$NUM_EPOCH" \
    --log_file="${GRAPH}_${MODEL}_sim_w${WS}.csv" \
    || echo "WARN sim depth $d exited nonzero"
done

echo "== simulator predictions ($GRAPH $MODEL w$WS) =="
cat "$CSV" 2>/dev/null || echo "(missing)"

: "${B2_REMOTE:=gsplit-b2}"; : "${B2_PREFIX:=gsplit-data/groot}"
if command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -q "^${B2_REMOTE}:"; then
  rclone copy "$PD/logs" "$B2_REMOTE:$B2_PREFIX/logs" --include "${GRAPH}_${MODEL}_sim_w${WS}.csv" -P || echo "WARN sim-log backup failed"
fi
