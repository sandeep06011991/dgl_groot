#!/usr/bin/env bash
# phase2_products.sh — ON THE VM. After the papers100M shards finish (DONE.sec1.{0,1}),
# provision products (OGB topology + B2 map) and run the missing products cells across the
# two 4-GPU shards. Termination is owned by idle_watchdog.sh; this script only does the
# work and pushes results to B2. A BUSY marker keeps the watchdog from killing the box
# during the (train_main-less) provisioning gap.
set -uo pipefail
REPO=/home/dgl_groot
export data_dir=/home/data
LOGDIR="$REPO/experiment/logs"
SETTINGS="$REPO/experiment/script/sec1_settings_products.jsonl"
BUSY="$REPO/BUSY"
exec >> "$REPO/phase2.out" 2>&1
echo "[phase2 $(date -u +%FT%TZ)] START"
touch "$BUSY"
trap 'rm -f "$BUSY"' EXIT

source /home/miniconda3/etc/profile.d/conda.sh && conda activate /home/groot_env
export LD_LIBRARY_PATH="$REPO/third_party/build/lib:${LD_LIBRARY_PATH:-}"

# 1) wait for papers100M shards to finish
echo "[phase2] waiting for papers DONE flags (DONE.sec1.0 & DONE.sec1.1)..."
while [ ! -f "$REPO/DONE.sec1.0" ] || [ ! -f "$REPO/DONE.sec1.1" ]; do sleep 30; done
echo "[phase2 $(date -u +%FT%TZ)] papers shards done"

# 2) push papers logs to B2 (belt-and-suspenders; idle_watchdog also pushes)
rclone copy "$LOGDIR" gsplit-b2:gsplit-data/groot/logs/sec1_autoterm/ --include "sec1_shard*.csv" 2>&1 | tail -3

# 3) provision products (OGB topology since B2 has no products graph dir; w4 map from B2)
echo "[phase2] provisioning products (ALLOW_OGB=1, world_size 4)"
if ! ALLOW_OGB=1 bash "$REPO/experiment/script/provision_data.sh" products "4"; then
  echo "[phase2] products provision FAILED -> leaving box to idle_watchdog"; exit 1
fi

# 4) run products shards (fresh logs; --master /nonexistent in run_worker2 runs every queued cell)
cd "$REPO/experiment/script"
echo "[phase2 $(date -u +%FT%TZ)] launching products shards"
bash run_worker2.sh 0 0,1,2,3 29500 "$SETTINGS" sec1_prod_shard0.csv prod & P0=$!
bash run_worker2.sh 1 4,5,6,7 29600 "$SETTINGS" sec1_prod_shard1.csv prod & P1=$!
wait "$P0"; r0=$?; wait "$P1"; r1=$?
echo "[phase2 $(date -u +%FT%TZ)] products shards done rc0=$r0 rc1=$r1"

# 5) push products logs to B2
rclone copy "$LOGDIR" gsplit-b2:gsplit-data/groot/logs/sec1_autoterm/ --include "sec1_prod_shard*.csv" 2>&1 | tail -3
echo "[phase2 $(date -u +%FT%TZ)] DONE (idle_watchdog terminates after idle window)"
