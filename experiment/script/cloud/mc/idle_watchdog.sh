#!/usr/bin/env bash
# idle_watchdog.sh — ON THE VM. Sole terminate guarantee for the box once the original
# self-terminating wrapper has been removed. Terminates the MC instance (after pushing
# logs to B2) when it has been idle too long — no train_main.py running AND no BUSY
# marker — or when an absolute max-life is hit. Prevents idle-billing per kill-on-inaction.
#
# Reads MC_API_TOKEN / SELF_UUID from ~/.config/groot/mc_run.env.
# Env: IDLE_LIMIT=1800 (30 min idle -> terminate), MAXLIFE=28800 (8 h absolute backstop).
set -uo pipefail
# shellcheck disable=SC1090
source "$HOME/.config/groot/mc_run.env"
API="https://vm.massedcompute.com/api/v1"
REPO=/home/dgl_groot
IDLE_LIMIT="${IDLE_LIMIT:-1800}"
MAXLIFE="${MAXLIFE:-28800}"
LOGDIR="$REPO/experiment/logs"
WD="$REPO/idle_watchdog.out"
log(){ echo "[idle-wd $(date -u +%FT%TZ)] $*" >> "$WD"; }
push(){ rclone copy "$LOGDIR" gsplit-b2:gsplit-data/groot/logs/sec1_autoterm/ --include "sec1_*shard*.csv" >>"$WD" 2>&1 || true
        rclone copy "$LOGDIR" gsplit-b2:gsplit-data/groot/logs/${B2_SUBDIR:-sec1_autoterm}/ --include "${LOG_PREFIX:-hyb}_shard*.csv" >>"$WD" 2>&1 || true; }
term(){ log "TERMINATE: $1"; push; curl -s --max-time 30 -X POST \
  -H "Authorization: Bearer $MC_API_TOKEN" -H "Content-Type: application/json" \
  -d "{\"instanceUuids\":[\"$SELF_UUID\"]}" "$API/instance/terminate" >>"$WD" 2>&1; }
start=$(date +%s); idle=0
log "start idle_limit=${IDLE_LIMIT}s maxlife=${MAXLIFE}s self=${SELF_UUID:-UNSET}"
while true; do
  sleep 60
  now=$(date +%s)
  [ $((now-start)) -ge "$MAXLIFE" ] && { term "maxlife ${MAXLIFE}s"; exit 0; }
  if pgrep -f train_main.py >/dev/null 2>&1 || [ -f "$REPO/BUSY" ]; then idle=0; else idle=$((idle+60)); fi
  [ $((idle%300)) -eq 0 ] && [ "$idle" -gt 0 ] && log "idle=${idle}s (train_main=$(pgrep -cf train_main.py) busy=$([ -f "$REPO/BUSY" ] && echo 1 || echo 0))"
  [ "$idle" -ge "$IDLE_LIMIT" ] && { term "idle>=${IDLE_LIMIT}s"; exit 0; }
done
