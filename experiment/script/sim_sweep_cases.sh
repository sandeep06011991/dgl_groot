#!/usr/bin/env bash
# sim_sweep_cases.sh  (ON THE POD)  —  run the (calibrated) cost-model simulator
# over the Section-1 "gsplit loses" product cases, for BOTH hybrids:
#   * DP-hybrid  : split-point k sweep   (default / --cost-estimation-mode hybriddata)
#   * pipeline   : pipeline+split overlap (--cost-estimation-mode pipeline, one k=-1 row)
#
# One CSV accumulates every row (Config columns identify graph/hw/model/hid/depth/cache;
# KProfiler column "Split from layer" is k, with k=-1 marking a pipeline row).
#
#   Usage:  sim_sweep_cases.sh [world_size]
#   Env:    data_dir (/root/graph), CACHE_SIZE (10G, matches the Section-1 sweep),
#           CELLS (override the default cell list), OUT (csv basename).
set -uo pipefail

WS="${1:-4}"
: "${data_dir:=/root/graph}"
: "${CACHE_SIZE:=10G}"
: "${MACHINE_NAME:=a100sxm}"; export MACHINE_NAME
: "${OUT:=products_simcases_w${WS}.csv}"

# Each cell: "hardware model depth hidden"  (hardware in {nvlink,pcie_emu}).
# Default = the 10 products losing cells (defense_latex/results/losing_cells.csv)
# plus L=5 deep + large-hidden extrapolation targets for the Section-4 table.
DEFAULT_CELLS="
nvlink   gat  2 256
nvlink   sage 2 256
pcie_emu gat  2 256
pcie_emu gat  3 1024
pcie_emu sage 2 256
pcie_emu sage 3 256
pcie_emu sage 3 512
pcie_emu sage 3 1024
pcie_emu sage 4 512
pcie_emu sage 4 1024
pcie_emu sage 5 256
pcie_emu sage 5 1024
nvlink   sage 5 1024
"
CELLS="${CELLS:-$DEFAULT_CELLS}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PD="$REPO/experiment"
export LD_LIBRARY_PATH="$REPO/third_party/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PD:${PYTHONPATH:-}"
: "${ENV_PATH:=/home/groot_env}"
if [ -f /home/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV_PATH" 2>/dev/null || true
fi

GRAPH=products
WEIGHT_WS="$WS" bash "$PD/script/provision_data.sh" "$GRAPH" "$WS"

cd "$PD"
CSV="logs/$OUT"
rm -f "$CSV"   # fresh accumulation
run_one () {  # hw model depth hid mode
  local hw="$1" model="$2" depth="$3" hid="$4" mode="$5"
  local fan; fan=$(printf '15%.0s,' $(seq 1 "$depth")); fan=${fan%,}
  local nvflag=""; [ "$hw" = nvlink ] && nvflag="--nvlink"
  local modeflag=""; [ -n "$mode" ] && modeflag="--cost-estimation-mode=$mode"
  echo "== SIM $GRAPH hw=$hw model=$model L=$depth h=$hid cache=$CACHE_SIZE mode=${mode:-dp-ksweep} =="
  python3 get_best_strategy.py --graph_name="$GRAPH" --data_dir="$data_dir" \
    --fanouts="$fan" --world_size="$WS" --hid_size="$hid" --model="$model" \
    --cache_size="$CACHE_SIZE" $nvflag $modeflag \
    --node_weight=dst --edge_weight=freq --bal=xbal --num_epoch=1 \
    --log_file="$OUT" \
    || echo "WARN $hw $model L$depth h$hid ${mode:-dp} exited nonzero"
}

while read -r hw model depth hid; do
  [ -z "${hw:-}" ] && continue
  run_one "$hw" "$model" "$depth" "$hid" ""          # DP-hybrid k-sweep (k=0..L)
  run_one "$hw" "$model" "$depth" "$hid" "pipeline"  # pipeline+split (k=-1)
done <<< "$CELLS"

echo "== all simulator rows ($GRAPH w$WS) =="
cat "$CSV" 2>/dev/null || echo "(missing)"

: "${B2_REMOTE:=gsplit-b2}"; : "${B2_PREFIX:=gsplit-data/groot}"
if command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -q "^${B2_REMOTE}:"; then
  rclone copy "$PD/logs" "$B2_REMOTE:$B2_PREFIX/logs" --include "$OUT" -P || echo "WARN sim-log backup failed"
fi
