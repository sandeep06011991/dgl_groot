#!/usr/bin/env bash
# run_sec1_autoterm.sh — RUN ON THE VM (Massed Compute). Unattended Section-1 sweep:
#   provision data -> run two concurrent 4-GPU shards -> push results to B2 -> SELF-TERMINATE.
#
# Built for overnight / unattended runs: it self-terminates the instance on completion,
# on a fatal setup error, AND on a hard MAX_HOURS cap (fail-toward-terminate), so the
# $11.04/hr box can never idle-bill after the work is done or stalled. Results are pushed
# to B2 *before* every terminate so nothing is lost when the disk is destroyed.
#
# Required env:
#   MC_API_TOKEN   MC API token (to self-terminate via the VM API)
#   SELF_UUID      this instance's uuid (mc_launch.sh prints it as POD_ID=...)
# Optional env:
#   MAX_HOURS=10                  hard cap: force-terminate after this long no matter what
#   SETTINGS=<path>              settings jsonl (default: sec1_settings.jsonl in the repo)
#   GRAPHS="papers100M products" graphs to provision, in order (papers100M first)
#   NO_TERMINATE=1               debug: run everything but never terminate
# For pushing results off the ephemeral disk, one of:
#   - a gsplit-b2 rclone remote already configured, OR
#   - ~/.config/groot/b2.env holding B2_KEY_ID / B2_APP_KEY (scp it over before launch)
set -uo pipefail
API="https://vm.massedcompute.com/api/v1"
REPO=/home/dgl_groot
SETTINGS="${SETTINGS:-$REPO/experiment/script/sec1_settings.jsonl}"
MAX_HOURS="${MAX_HOURS:-10}"
GRAPHS="${GRAPHS:-papers100M products}"
export data_dir="${data_dir:-/home/data}"
LOGDIR="$REPO/experiment/logs"
mkdir -p "$LOGDIR"
exec > >(tee -a "$REPO/autoterm.out") 2>&1
echo "[autoterm $(date -u +%FT%TZ)] START settings=$SETTINGS graphs='$GRAPHS' max_hours=$MAX_HOURS self=${SELF_UUID:-UNSET}"

setup_b2() {
  if ! command -v rclone >/dev/null 2>&1; then
    echo "[autoterm] installing rclone"
    curl -s https://rclone.org/install.sh | sudo bash >/dev/null 2>&1 \
      || curl -s https://rclone.org/install.sh | bash >/dev/null 2>&1 || true
  fi
  if command -v rclone >/dev/null 2>&1 && ! rclone listremotes 2>/dev/null | grep -q '^gsplit-b2:'; then
    if [ -f "$HOME/.config/groot/b2.env" ]; then
      # shellcheck disable=SC1091
      source "$HOME/.config/groot/b2.env"
      rclone config create gsplit-b2 b2 account "${B2_KEY_ID:-}" key "${B2_APP_KEY:-}" >/dev/null 2>&1 \
        && echo "[autoterm] configured gsplit-b2 remote" || echo "[autoterm] WARN rclone config failed"
    else
      echo "[autoterm] WARN no gsplit-b2 remote and no b2.env -> results CANNOT be pushed"
    fi
  fi
}

push_b2() {
  command -v rclone >/dev/null 2>&1 || { echo "[autoterm] no rclone, cannot push"; return 1; }
  rclone listremotes 2>/dev/null | grep -q '^gsplit-b2:' || { echo "[autoterm] no gsplit-b2 remote, cannot push"; return 1; }
  echo "[autoterm] pushing CSVs -> B2 gsplit-data/groot/logs/sec1_autoterm/"
  rclone copy "$LOGDIR" gsplit-b2:gsplit-data/groot/logs/sec1_autoterm/ --include "sec1_shard*.csv" -v 2>&1 | tail -5
}

terminate_self() {
  local why="${1:-unknown}"
  echo "[autoterm $(date -u +%FT%TZ)] TERMINATE ($why)"
  push_b2 || echo "[autoterm] WARN push failed before terminate"
  if [ -n "${NO_TERMINATE:-}" ]; then echo "[autoterm] NO_TERMINATE set; staying up"; return 0; fi
  if [ -z "${MC_API_TOKEN:-}" ] || [ -z "${SELF_UUID:-}" ]; then
    echo "[autoterm] FATAL cannot self-terminate (missing MC_API_TOKEN/SELF_UUID); poweroff fallback"
    sudo poweroff 2>/dev/null || poweroff 2>/dev/null || true
    return 1
  fi
  curl -s --max-time 30 -X POST -H "Authorization: Bearer $MC_API_TOKEN" -H "Content-Type: application/json" \
    -d "{\"instanceUuids\":[\"$SELF_UUID\"]}" "$API/instance/terminate" -w $'\nHTTP %{http_code}\n'
}

# hard watchdog: terminate after MAX_HOURS regardless of job state (never idle-bill overnight)
( sleep "$(( MAX_HOURS*3600 ))"; echo "[autoterm $(date -u +%FT%TZ)] MAX_HOURS cap hit"; terminate_self "max-hours" ) &
WATCHDOG=$!

setup_b2

source /home/miniconda3/etc/profile.d/conda.sh && conda activate /home/groot_env
export LD_LIBRARY_PATH="$REPO/third_party/build/lib:${LD_LIBRARY_PATH:-}"

# 1) provision data for each graph (B2 restore; --random_feat sweep => no features pulled)
for g in $GRAPHS; do
  echo "[autoterm] provision $g (world_size 4)"
  if ! bash "$REPO/experiment/script/provision_data.sh" "$g" "4"; then
    echo "[autoterm] provision FAILED for $g"
    terminate_self "provision-fail:$g"; exit 1
  fi
done

# 2) run the two 4-GPU shards concurrently (shard 0 -> GPUs 0-3, shard 1 -> GPUs 4-7)
cd "$REPO/experiment/script"
echo "[autoterm] launching shards"
bash run_worker2.sh 0 0,1,2,3 29500 "$SETTINGS" sec1_shard0.csv sec1 & P0=$!
bash run_worker2.sh 1 4,5,6,7 29600 "$SETTINGS" sec1_shard1.csv sec1 & P1=$!
wait "$P0"; r0=$?
wait "$P1"; r1=$?
echo "[autoterm $(date -u +%FT%TZ)] shards done rc0=$r0 rc1=$r1"

kill "$WATCHDOG" 2>/dev/null || true
terminate_self "complete"
