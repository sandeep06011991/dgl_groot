#!/usr/bin/env bash
# run_pipe_rerun.sh — RUN ON A RUNPOD POD (root, 4 GPUs). Unattended re-run of the PURE
# pipeline (gsplit_pipepar, k=0) cells after the micro-batch fix (240e2e2a: each of the two
# micro-batches now holds batch_size//2 seeds). 40 cells = 5 (L,h) x 2 graphs x 2 interconnects
# x 2 models, one 4-GPU worker, serial.
#
# provision papers100M (B2) + products (OGB topology, B2 map) -> run settings -> push the log
# to B2 -> touch DONE -> try to delete this pod. The control machine ALSO runs a watcher that
# deletes the pod on DONE / hard cap (the primary guarantee; see the session ledger).
#
# Env: SETTINGS (sec1_pipe_rerun_settings.jsonl), LOG (pipefix_runpod.csv),
#      B2_SUBDIR (pipe_rerun), GRAPHS ("papers100M products"), data_dir (/home/data)
set -uo pipefail
REPO=/home/dgl_groot
SETTINGS="${SETTINGS:-$REPO/experiment/script/sec1_pipe_rerun_settings.jsonl}"
LOG="${LOG:-pipefix_runpod.csv}"
B2_SUBDIR="${B2_SUBDIR:-pipe_rerun}"
GRAPHS="${GRAPHS:-papers100M products}"
export data_dir="${data_dir:-/home/data}"
LOGDIR="$REPO/experiment/logs"; mkdir -p "$LOGDIR" "$data_dir"
exec > >(tee -a "$REPO/pipe_rerun.out") 2>&1
echo "[pipe $(date -u +%FT%TZ)] START settings=$SETTINGS log=$LOG graphs='$GRAPHS'"

push_b2() { rclone copy "$LOGDIR" "gsplit-b2:gsplit-data/groot/logs/$B2_SUBDIR/" --include "$LOG" -v 2>&1 | tail -3; }
finish() {
  echo "[pipe $(date -u +%FT%TZ)] FINISH ($1)"
  push_b2 || echo "[pipe] WARN B2 push failed"
  touch "$REPO/DONE.pipe.$1"
  if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl pod delete "$RUNPOD_POD_ID" 2>&1 | tail -2 || runpodctl remove pod "$RUNPOD_POD_ID" 2>&1 | tail -2 || true
  fi
}

source /home/miniconda3/etc/profile.d/conda.sh && conda activate /home/groot_env
export LD_LIBRARY_PATH="$REPO/third_party/build/lib:${LD_LIBRARY_PATH:-}"

for g in $GRAPHS; do
  echo "[pipe] provision $g (w4)"
  if ! ALLOW_OGB=1 bash "$REPO/experiment/script/provision_data.sh" "$g" "4"; then
    echo "[pipe] provision FAILED for $g"; finish "provision-fail"; exit 1
  fi
done

export CUDA_VISIBLE_DEVICES=0,1,2,3 TRAIN_MASTER_PORT=29500
cd "$REPO/experiment/script"
python3 run_settings.py --settings "$SETTINGS" --data_dir "$data_dir" \
  --log_file "$LOG" --master /nonexistent --timeout 1200
echo "[pipe $(date -u +%FT%TZ)] runner rc=$?"
finish "complete"
