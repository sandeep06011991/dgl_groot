#!/usr/bin/env bash
# run_hybrid_autoterm.sh — RUN ON THE VM (Massed Compute). Unattended gsplit-hybrid
# ablation sweep: provision papers100M (B2) + products (OGB) -> run two concurrent 4-GPU
# shards over sec1_hybrid_settings.jsonl -> push results to B2 -> SELF-TERMINATE.
#
# Self-terminates on completion, on fatal provision error, and at a hard MAX_HOURS cap
# (fail-toward-terminate) so the $11.04/hr box can never idle-bill. Pair with
# idle_watchdog.sh as an independent backstop. Results pushed to B2 before every terminate.
#
# Required env: MC_API_TOKEN, SELF_UUID (from ~/.config/groot/mc_run.env).
# Optional: MAX_HOURS=12, SETTINGS (default sec1_hybrid_settings.jsonl),
#           GRAPHS="papers100M products", NO_TERMINATE=1 (debug).
set -uo pipefail
API="https://vm.massedcompute.com/api/v1"
REPO=/home/dgl_groot
SETTINGS="${SETTINGS:-$REPO/experiment/script/sec1_hybrid_settings.jsonl}"
MAX_HOURS="${MAX_HOURS:-12}"
GRAPHS="${GRAPHS:-papers100M products}"
export data_dir="${data_dir:-/home/data}"
LOGDIR="$REPO/experiment/logs"
mkdir -p "$LOGDIR"
exec > >(tee -a "$REPO/hybrid_autoterm.out") 2>&1
echo "[hyb $(date -u +%FT%TZ)] START settings=$SETTINGS graphs='$GRAPHS' max_hours=$MAX_HOURS self=${SELF_UUID:-UNSET}"

setup_b2() {
  if ! command -v rclone >/dev/null 2>&1; then
    curl -s https://rclone.org/install.sh | sudo bash >/dev/null 2>&1 || true
  fi
  if command -v rclone >/dev/null 2>&1 && ! rclone listremotes 2>/dev/null | grep -q '^gsplit-b2:'; then
    [ -f "$HOME/.config/groot/b2.env" ] && { source "$HOME/.config/groot/b2.env"; \
      rclone config create gsplit-b2 b2 account "${B2_KEY_ID:-}" key "${B2_APP_KEY:-}" >/dev/null 2>&1 \
      && echo "[hyb] gsplit-b2 configured"; }
  fi
}
push_b2() {
  command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -q '^gsplit-b2:' || { echo "[hyb] no rclone/remote, cannot push"; return 1; }
  rclone copy "$LOGDIR" gsplit-b2:gsplit-data/groot/logs/sec1_autoterm/ --include "hyb_shard*.csv" -v 2>&1 | tail -4
}
terminate_self() {
  local why="${1:-unknown}"
  echo "[hyb $(date -u +%FT%TZ)] TERMINATE ($why)"
  push_b2 || echo "[hyb] WARN push failed before terminate"
  [ -n "${NO_TERMINATE:-}" ] && { echo "[hyb] NO_TERMINATE set"; return 0; }
  if [ -z "${MC_API_TOKEN:-}" ] || [ -z "${SELF_UUID:-}" ]; then
    echo "[hyb] FATAL no token/uuid; poweroff fallback"; sudo poweroff 2>/dev/null || poweroff 2>/dev/null || true; return 1
  fi
  curl -s --max-time 30 -X POST -H "Authorization: Bearer $MC_API_TOKEN" -H "Content-Type: application/json" \
    -d "{\"instanceUuids\":[\"$SELF_UUID\"]}" "$API/instance/terminate" -w $'\nHTTP %{http_code}\n'
}

# hard watchdog
( sleep "$(( MAX_HOURS*3600 ))"; echo "[hyb $(date -u +%FT%TZ)] MAX_HOURS cap"; terminate_self "max-hours" ) &

setup_b2
source /home/miniconda3/etc/profile.d/conda.sh && conda activate /home/groot_env
export LD_LIBRARY_PATH="$REPO/third_party/build/lib:${LD_LIBRARY_PATH:-}"

# 1) provision data (ALLOW_OGB=1 so products regenerates topology from OGB; papers100M is in
#    B2 so OGB never triggers for it; --random_feat => no feature download for either).
for g in $GRAPHS; do
  echo "[hyb] provision $g (world_size 4, ALLOW_OGB=1)"
  if ! ALLOW_OGB=1 bash "$REPO/experiment/script/provision_data.sh" "$g" "4"; then
    echo "[hyb] provision FAILED for $g"; terminate_self "provision-fail:$g"; exit 1
  fi
done

# 2) run two 4-GPU shards concurrently over the hybrid grid
cd "$REPO/experiment/script"
echo "[hyb] launching shards"
bash run_worker2.sh 0 0,1,2,3 29500 "$SETTINGS" hyb_shard0.csv hyb & P0=$!
bash run_worker2.sh 1 4,5,6,7 29600 "$SETTINGS" hyb_shard1.csv hyb & P1=$!
wait "$P0"; r0=$?; wait "$P1"; r1=$?
echo "[hyb $(date -u +%FT%TZ)] shards done rc0=$r0 rc1=$r1"
terminate_self "complete"
