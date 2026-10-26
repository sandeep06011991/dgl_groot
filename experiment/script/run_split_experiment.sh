#!/usr/bin/env bash
# run_split_experiment.sh — run the Spara "split" (gsplit) measurement for one
# graph across sampling depths. Ensures data first (provision_data.sh), then
# runs train_main.py once per depth, writing one CSV row each.
#
#   Usage:  run_split_experiment.sh <graph> [world_size] ["depths"] [num_epoch] [model]
#   e.g.    run_split_experiment.sh arxiv 2 "3 4 5" 5 sage
#           run_split_experiment.sh products 2 "3 4 5" 5 gat
#
# world_size selects the partition map (w<world_size>) and the #GPUs. model is
# sage|gat. Results:
#   experiment/logs/<graph>_<model>_w<ws>_d<depth>.csv
# Requires the conda env active (so LD_LIBRARY_PATH/CUDA/torch resolve); see RUNBOOK.md.
set -uo pipefail

GRAPH="${1:?usage: run_split_experiment.sh <graph> [world_size] [depths] [num_epoch] [model]}"
WS="${2:-2}"
DEPTHS="${3:-3 4 5}"
NUM_EPOCH="${4:-3}"
MODEL="${5:-sage}"
: "${data_dir:=/root/graph}"
: "${MACHINE_NAME:=a100sxm}"; export MACHINE_NAME
# Partition scheme — the SINGLE source of truth for both provisioning (which builds
# the map) and training (which looks it up). Default ndst_efreq_xbal. Forwarded to
# both below so the two can never disagree on the map filename.
: "${PART_NODE:=dst}"
: "${PART_EDGE:=freq}"
: "${PART_BAL:=xbal}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PD="$REPO/experiment"
export LD_LIBRARY_PATH="$REPO/third_party/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PD:${PYTHONPATH:-}"

# ensure the conda env is active (scripts are often run in a fresh SSH shell)
: "${ENV_PATH:=/home/groot_env}"
if [ -f /home/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV_PATH" 2>/dev/null || true
fi

# 1. make sure the data + the w<WS> partition map exist (B2 restore or OGB regen)
WEIGHT_WS="$WS" PART_NODE="$PART_NODE" PART_EDGE="$PART_EDGE" PART_BAL="$PART_BAL" \
  bash "$PD/script/provision_data.sh" "$GRAPH" "$WS"

# 2. run the measurement per depth
cd "$PD"
for d in $DEPTHS; do
  # fanout = d copies of "15"
  fan=$(printf '15%.0s,' $(seq 1 "$d")); fan=${fan%,}
  echo "== $GRAPH  split  model=$MODEL  world_size=$WS  depth=$d  fanout=$fan =="
  python3 train_main.py --system=split --model="$MODEL" --fanouts="$fan" \
    --graph_name="$GRAPH" --data_dir="$data_dir" --cache_size=10G --batch_size=1024 \
    --world_size="$WS" --num_epoch="$NUM_EPOCH" \
    --node_weight="$PART_NODE" --edge_weight="$PART_EDGE" --bal="$PART_BAL" \
    --log_file="${GRAPH}_${MODEL}_w${WS}_d${d}.csv" \
    || echo "WARN train $MODEL depth $d exited nonzero"
done

echo "== results ($GRAPH $MODEL w$WS) =="
for d in $DEPTHS; do
  echo "-- d$d --"
  tail -n1 "logs/${GRAPH}_${MODEL}_w${WS}_d${d}.csv" 2>/dev/null || echo "(missing)"
done

# Back up result CSVs to B2 (pods are ephemeral + no network volume, so the logs
# are lost on `pod delete` unless copied off). Best-effort.
: "${B2_REMOTE:=gsplit-b2}"; : "${B2_PREFIX:=gsplit-data/groot}"
if command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -q "^${B2_REMOTE}:"; then
  echo "== backing up logs to B2 =="
  rclone copy "$PD/logs" "$B2_REMOTE:$B2_PREFIX/logs" --include "${GRAPH}_${MODEL}_w${WS}_d*.csv" -P || echo "WARN log backup failed"
else
  echo "NOTE: no '$B2_REMOTE' rclone remote; CSVs are only in $PD/logs/ — scp them off before 'pod delete'."
fi
