#!/bin/bash
# sweep.sh — Section 1 performance sweep (dissertation eval plan, Step 1b).
#
# ONE hardware pass for ONE graph over the Section-1 systems and the hidden/depth
# ablation, writing a per-pass CSV that 1b consolidates into
# defense_latex/results/sweep_all.csv.
#
# SCOPE (per user, kept lean): systems = split (full split, k=0), dgl, quiver.
# No p3, no pipeline-split, no k-variants here — the hybrid ablations (k,
# pipeline) are chosen by the SIMULATOR (Section 3) and measured only at the
# predicted config (Section 4).
#
# Ablation = two 1-D sweeps crossing at (depth=3, hidden=256):
#   hidden ∈ HID_SIZES at depth DEPTH_BASE, and depth ∈ DEPTHS at hidden HID_BASE.
#
#   Usage:  HARDWARE=nvlink WORLD_SIZE=4 sweep.sh <graph>
#           HARDWARE=pcie   WORLD_SIZE=4 sweep.sh <graph>   # NCCL_P2P_DISABLE=1
#
# Env knobs (defaults in parens):
#   HARDWARE     (nvlink)  nvlink | pcie  -- pcie forces comm onto PCIe via
#                          NCCL_P2P_DISABLE=1 + drops --nvlink (Config hardware=pcie_emu,
#                          quiver uses device_replicate).
#   WORLD_SIZE   (4)       GPUs on this pod.
#   GRAPH        ($1)      products | papers100M | ...
#   CACHE_SIZE   (10G)     absolute cache for split/dgl.
#   HID_SIZES    ("256 512 1024")  hidden-size axis (at DEPTH_BASE).
#   DEPTHS       ("2 3 4")         depth axis (at HID_BASE).
#   DEPTH_BASE   (3)   HID_BASE (256)   crossing point of the two axes.
#   FPL          (15)             fanout per layer.
#   QUIVER_FRACS ("0.25 0.10")    quiver cache fractions.
#   SYSTEMS      ("split dgl quiver")
#   MODELS       ("sage gat")
#   NUM_EPOCH    (5)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

GRAPH="${1:-${GRAPH:-products}}"
HARDWARE="${HARDWARE:-nvlink}"
WORLD_SIZE="${WORLD_SIZE:-4}"
CACHE_SIZE="${CACHE_SIZE:-10G}"
# Note: '-' not ':-' so an explicit empty value (HID_SIZES="") is honored, letting
# a top-up run do only the CROSS corner cells without re-running the 1-D sweeps.
HID_SIZES="${HID_SIZES-256 512 1024}"
DEPTHS="${DEPTHS-2 3 4}"
DEPTH_BASE="${DEPTH_BASE:-3}"
HID_BASE="${HID_BASE:-256}"
FPL="${FPL:-15}"
QUIVER_FRACS="${QUIVER_FRACS:-0.25 0.10}"
SYSTEMS="${SYSTEMS:-split dgl quiver}"
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

# Build the (depth:hidden) ablation list = two 1-D sweeps crossing at BASE,
# PLUS explicit deep x large-hidden CORNER cells (CROSS) — the regime the chapter
# thesis targets ("gsplit bad at deeper layers AND larger hidden"), which the two
# 1-D sweeps alone skip.
CROSS="${CROSS:-4:512 4:1024}"
declare -A SEEN
CONFIGS=()
for h in $HID_SIZES; do k="${DEPTH_BASE}:${h}"; [ -n "${SEEN[$k]:-}" ] || { CONFIGS+=("$k"); SEEN[$k]=1; }; done
for d in $DEPTHS;    do k="${d}:${HID_BASE}"; [ -n "${SEEN[$k]:-}" ] || { CONFIGS+=("$k"); SEEN[$k]=1; }; done
for k in $CROSS;     do                       [ -n "${SEEN[$k]:-}" ] || { CONFIGS+=("$k"); SEEN[$k]=1; }; done

fanout_of() { local d="$1" out="$FPL" i; for ((i=1;i<d;i++)); do out="${out},${FPL}"; done; echo "$out"; }

echo "=============================================================="
echo " sweep: graph=$GRAPH hardware=$HW_TAG world_size=$WORLD_SIZE epochs=$NUM_EPOCH"
echo "        systems=[$SYSTEMS] configs(depth:hid)=[${CONFIGS[*]}]"
echo "        log=logs/$LOG_FILE  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset}"
echo "=============================================================="

run() {   # run <extra-pythonpath> <system> <cache_arg> <hid> <fanout>
  local pypath="$1" system="$2" cache_arg="$3" hid="$4" fanout="$5"
  echo ">>> system=$system model=$MODEL depth=$(echo "$fanout" | tr ',' ' ' | wc -w) hid=$hid $cache_arg"
  PYTHONPATH="$pypath" python3 "${python_dir}/train_main.py" \
    --system="$system" --model="$MODEL" --fanouts="$fanout" --hid_size="$hid" \
    --graph="$GRAPH" --data_dir="$data_dir" \
    --batch_size=1024 --world_size="$WORLD_SIZE" --num_epoch="$NUM_EPOCH" \
    --num_redundant_layer=0 $cache_arg $NVLINK_FLAG \
    --log_file="$LOG_FILE"
}

for MODEL in $MODELS; do
  for cfg in "${CONFIGS[@]}"; do
    depth="${cfg%%:*}"; hid="${cfg##*:}"; fanout="$(fanout_of "$depth")"
    for system in $SYSTEMS; do
      case "$system" in
        split) run "$BASE_PYPATH" split "--cache_size=$CACHE_SIZE" "$hid" "$fanout" ;;
        dgl)   run "$BASE_PYPATH" dgl   "--cache_size=$CACHE_SIZE" "$hid" "$fanout" ;;
        quiver)
          for frac in $QUIVER_FRACS; do
            run "$QUIVER_PYPATH" quiver "--cache_frac=$frac" "$hid" "$fanout"
          done ;;
        *) echo "unknown system: $system (skipped)" >&2 ;;
      esac
    done
  done
done

echo "sweep pass done -> ${python_dir}/logs/${LOG_FILE}"
