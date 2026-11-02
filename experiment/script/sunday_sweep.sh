#!/bin/bash
# sunday_sweep.sh — Section-1 training-time sweep, run until OOM.
#
# Four situations only (the columns of every table), training time only. Run once per
# hardware pass: HARDWARE=nvlink then HARDWARE=pcie (pcie = NCCL_P2P_DISABLE=1 emulation).
#   1. dgl    @ 0%  cache   (--cache_size=0G)
#   2. split  @ 0%  cache   (--cache_size=0G)            <- gsplit, uncached
#   3. quiver @ 25% cache   (--cache_frac=0.25)
#   4. split  @ 25% cache   (--cache_frac=0.25)          <- gsplit, cached
# On a 4-GPU run quiver's 25%-per-GPU partitioned cache == the whole graph resident
# across the clique, matching split's partition residency -> (4 vs 3) is apples-to-apples.
#
# ONE axis per invocation, climbed ASCENDING until the gsplit (split) cell OOMs or
# hits the 900s cap (larger/deeper is monotonically worse, so that ends the climb).
# An underperform crossover is RECORDED, never a stop -- we stress to OOM.
#
#   MODE=hidden FIXED_DEPTH=3 HID_LADDER="256 512 1024 2048 4096" sunday_sweep.sh <graph>
#   MODE=depth  FIXED_HID=<H*/2> DEPTH_LADDER="2 3 4 5 6"         sunday_sweep.sh <graph>
#
# Models: gat first, then sage (gat OOMs earlier -> finds the ceiling fast).
#
# Env knobs (defaults in parens):
#   HARDWARE     (nvlink)  nvlink | pcie   (pcie -> NCCL_P2P_DISABLE=1, drops --nvlink)
#   MODE         (hidden)  hidden | depth
#   HID_LADDER   ("256 512 1024 2048 4096 8192")   hidden axis (MODE=hidden)
#   FIXED_DEPTH  (3)                                depth held fixed (MODE=hidden)
#   DEPTH_LADDER ("2 3 4 5 6")                      depth axis (MODE=depth)
#   FIXED_HID    (512)                              hidden held fixed (MODE=depth; set H*/2)
#   MODELS       ("gat sage")   WORLD_SIZE (4)   FPL (15)   BATCH (1024)
#   CACHE_FRAC   (0.25)         NUM_EPOCH (5)    RUN_TIMEOUT (900)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

GRAPH="${1:-${GRAPH:-papers100M}}"
HARDWARE="${HARDWARE:-nvlink}"
MODE="${MODE:-hidden}"
HID_LADDER="${HID_LADDER:-256 512 1024 2048 4096 8192}"
FIXED_DEPTH="${FIXED_DEPTH:-3}"
DEPTH_LADDER="${DEPTH_LADDER:-2 3 4 5 6}"
FIXED_HID="${FIXED_HID:-512}"
MODELS="${MODELS:-gat sage}"
WORLD_SIZE="${WORLD_SIZE:-4}"
FPL="${FPL:-15}"
BATCH="${BATCH:-1024}"
CACHE_FRAC="${CACHE_FRAC:-0.25}"
NUM_EPOCH="${NUM_EPOCH:-5}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"   # per-cell wall-clock cap (s); 900 = 15 min

# hardware pass: NVLink as-is; PCIe emulated by disabling NCCL P2P (forces comm onto
# PCIe). No C++ rebuild needed between passes -- split comm is runtime NCCL and respects
# P2P-disable; --nvlink only flips the hardware label + quiver's cache policy (dropping it
# makes quiver use the PCIe-safe device_replicate path instead of the p2p clique).
NVLINK_FLAG="--nvlink"
if [ "$HARDWARE" = "pcie" ] || [ "$HARDWARE" = "pcie_emu" ]; then
  export NCCL_P2P_DISABLE=1
  NVLINK_FLAG=""
  HW_TAG="pcie_emu"
else
  unset NCCL_P2P_DISABLE 2>/dev/null || true
  HW_TAG="nvlink"
fi

LOG_FILE="sunday_${GRAPH}_${HW_TAG}_${MODE}.csv"
BASE_PYPATH="${SCRIPT_DIR}/../../python"
QUIVER_PYPATH="${BASE_PYPATH}:${SCRIPT_DIR}/../../third_party/torch-quiver/srcs/python"

fanout_of() { local d="$1" out="$FPL" i; for ((i=1;i<d;i++)); do out="${out},${FPL}"; done; echo "$out"; }

# kill orphaned DDP workers from the previous cell (any exit, not just timeouts) so a
# crashed/OOM'd run can't poison the next cell's GPU memory. Bracket patterns so this
# shell doesn't match itself.
cleanup() {
  pkill -9 -f '[t]rain_main.py' 2>/dev/null || true
  pkill -9 -f '[m]ultiprocessing.spawn' 2>/dev/null || true
  sleep 3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/    gpu /'
}

# run <pypath> <system> <cache-arg> <model> <hid> <fanout> ; echoes rc, records timeout rows.
run() {
  local pypath="$1" system="$2" cache_arg="$3" model="$4" hid="$5" fanout="$6"
  local depth; depth=$(echo "$fanout" | tr ',' ' ' | wc -w)
  local -a A=(--system="$system" --model="$model" --fanouts="$fanout" --hid_size="$hid"
    --graph="$GRAPH" --data_dir="$data_dir" --batch_size="$BATCH" --world_size="$WORLD_SIZE"
    --num_epoch="$NUM_EPOCH" --num_redundant_layer=0 $NVLINK_FLAG --random_feat
    $cache_arg --log_file="$LOG_FILE")
  echo ">>> system=$system model=$model depth=$depth hid=$hid $cache_arg (cap ${RUN_TIMEOUT}s)"
  PYTHONPATH="$pypath" timeout -k 30 "$RUN_TIMEOUT" python3 "${python_dir}/train_main.py" "${A[@]}"
  local rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "!!! TIMEOUT ${RUN_TIMEOUT}s (rc=$rc) -> record timeout row"
    PYTHONPATH="$pypath" python3 "${python_dir}/record_timeout.py" "${A[@]}" \
      || echo "warn: could not record timeout row"
  elif [ "$rc" -ne 0 ]; then
    echo "!!! NONZERO rc=$rc (likely OOM; os._exit records no row)"
  fi
  cleanup
  return "$rc"
}

# one cell = all 4 situations at (model,hid,depth); returns the gsplit(uncached) rc,
# which drives the early-break (its OOM/timeout = ceiling for this axis).
cell() {
  local model="$1" hid="$2" fanout="$3" split_rc=0
  run "$BASE_PYPATH"   split  "--cache_size=0G"           "$model" "$hid" "$fanout"; split_rc=$?
  run "$BASE_PYPATH"   dgl    "--cache_size=0G"           "$model" "$hid" "$fanout" || true
  run "$QUIVER_PYPATH" quiver "--cache_frac=$CACHE_FRAC"  "$model" "$hid" "$fanout" || true
  run "$BASE_PYPATH"   split  "--cache_frac=$CACHE_FRAC"  "$model" "$hid" "$fanout" || true
  return "$split_rc"
}

echo "=============================================================="
echo " sunday_sweep: graph=$GRAPH hardware=$HW_TAG mode=$MODE world_size=$WORLD_SIZE epochs=$NUM_EPOCH"
if [ "$MODE" = "hidden" ]; then echo "  hidden ladder=[$HID_LADDER] @ depth=$FIXED_DEPTH"; \
  else echo "  depth ladder=[$DEPTH_LADDER] @ hidden=$FIXED_HID"; fi
echo "  models=[$MODELS]  log=logs/$LOG_FILE  cache_frac=$CACHE_FRAC"
echo "=============================================================="

for MODEL in $MODELS; do
  echo "######## model=$MODEL ########"
  if [ "$MODE" = "hidden" ]; then
    fanout="$(fanout_of "$FIXED_DEPTH")"
    for h in $HID_LADDER; do
      echo "---- $MODEL hidden=$h (depth=$FIXED_DEPTH) ----"
      if ! cell "$MODEL" "$h" "$fanout"; then
        echo "==== gsplit hit the ceiling at hidden=$h -> stop climbing ($MODEL) ===="
        break
      fi
    done
  else
    for d in $DEPTH_LADDER; do
      fanout="$(fanout_of "$d")"
      echo "---- $MODEL depth=$d (hidden=$FIXED_HID) ----"
      if ! cell "$MODEL" "$FIXED_HID" "$fanout"; then
        echo "==== gsplit hit the ceiling at depth=$d -> stop climbing ($MODEL) ===="
        break
      fi
    done
  fi
done

echo "sunday_sweep done -> ${python_dir}/logs/${LOG_FILE}"
