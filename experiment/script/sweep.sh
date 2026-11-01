#!/bin/bash
# sweep.sh — Section 1 performance sweep (dissertation eval plan, Step 1b).
#
# Runs ONE hardware pass for ONE graph across every system/config cell, writing a
# per-pass CSV that 1b consolidates into defense_latex/results/sweep_all.csv.
#
#   Usage:  HARDWARE=nvlink  WORLD_SIZE=4 sweep.sh <graph>
#           HARDWARE=pcie    WORLD_SIZE=4 sweep.sh <graph>   # NCCL_P2P_DISABLE=1
#
# Env knobs (defaults in parens):
#   HARDWARE    (nvlink)  nvlink | pcie   -- pcie forces comm onto PCIe via
#                         NCCL_P2P_DISABLE=1 and drops --nvlink (so Config records
#                         hardware=pcie_emu and quiver uses device_replicate).
#   WORLD_SIZE  (4)       GPUs on this pod.
#   GRAPH       ($1)      products | papers100M | ...
#   CACHE_SIZE  (10G)     absolute cache for split/dgl/p3.
#   FANOUT      (15,15,15)
#   K_GRID      ("0 1 2") data-split k = num_redundant_layer values for split.
#   QUIVER_FRACS ("0.25 0.10")  quiver cache fractions.
#   SYSTEMS     ("split dgl quiver p3 gsplit_pipepar")
#   MODELS      ("sage gat")
#   NUM_EPOCH   (5)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

GRAPH="${1:-${GRAPH:-products}}"
HARDWARE="${HARDWARE:-nvlink}"
WORLD_SIZE="${WORLD_SIZE:-4}"
CACHE_SIZE="${CACHE_SIZE:-10G}"
FANOUT="${FANOUT:-15,15,15}"
K_GRID="${K_GRID:-0 1 2}"
QUIVER_FRACS="${QUIVER_FRACS:-0.25 0.10}"
SYSTEMS="${SYSTEMS:-split dgl quiver p3 gsplit_pipepar}"
MODELS="${MODELS:-sage gat}"
NUM_EPOCH="${NUM_EPOCH:-5}"

# --- hardware pass: NVLink as-is, PCIe emulated by disabling NCCL P2P ----------
NVLINK_FLAG="--nvlink"
if [ "$HARDWARE" == "pcie" ] || [ "$HARDWARE" == "pcie_emu" ]; then
  export NCCL_P2P_DISABLE=1
  NVLINK_FLAG=""
  HW_TAG="pcie_emu"
else
  unset NCCL_P2P_DISABLE 2>/dev/null || true
  HW_TAG="nvlink"
fi

LOG_FILE="sweep_${GRAPH}_${HW_TAG}.csv"
QUIVER_PYPATH="${SCRIPT_DIR}/../../python:${SCRIPT_DIR}/../../third_party/torch-quiver/srcs/python"
BASE_PYPATH="${SCRIPT_DIR}/../../python"

echo "=============================================================="
echo " sweep: graph=$GRAPH hardware=$HW_TAG world_size=$WORLD_SIZE"
echo "        fanout=$FANOUT cache=$CACHE_SIZE epochs=$NUM_EPOCH"
echo "        log=logs/$LOG_FILE  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset}"
echo "=============================================================="

run() {   # run <extra-pythonpath> <system> <cache_arg> <k> [extra flags...]
  local pypath="$1"; shift
  local system="$1"; shift
  local cache_arg="$1"; shift
  local k="$1"; shift
  echo ">>> system=$system model=$MODEL k=$k $cache_arg $*"
  PYTHONPATH="$pypath" python3 "${python_dir}/train_main.py" \
    --system="$system" --model="$MODEL" --fanouts="$FANOUT" \
    --graph="$GRAPH" --data_dir="$data_dir" \
    --batch_size=1024 --world_size="$WORLD_SIZE" --num_epoch="$NUM_EPOCH" \
    --num_redundant_layer="$k" $cache_arg $NVLINK_FLAG $* \
    --log_file="$LOG_FILE"
}

for MODEL in $MODELS; do
  for system in $SYSTEMS; do
    case "$system" in
      split)
        for k in $K_GRID; do
          run "$BASE_PYPATH" split "--cache_size=$CACHE_SIZE" "$k"
        done ;;
      gsplit_pipepar)
        # pipeline-split: overlap on, full split (k=0). Measure across the sweep
        # (Risk 5: under-characterized at scale -> run, then conclude from data).
        run "$BASE_PYPATH" gsplit_pipepar "--cache_size=$CACHE_SIZE" 0 ;;
      dgl)
        run "$BASE_PYPATH" dgl "--cache_size=$CACHE_SIZE" 0 ;;
      p3)
        run "$BASE_PYPATH" p3 "--cache_size=$CACHE_SIZE" 0 ;;
      quiver)
        for frac in $QUIVER_FRACS; do
          run "$QUIVER_PYPATH" quiver "--cache_frac=$frac" 0
        done ;;
      *)
        echo "unknown system: $system" >&2 ;;
    esac
  done
done

echo "sweep pass done -> ${python_dir}/logs/${LOG_FILE}"
