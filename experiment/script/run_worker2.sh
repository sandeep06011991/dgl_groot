#!/usr/bin/env bash
# run_worker2.sh SHARD GPUS PORT SETTINGS LOG DONETAG
#   SHARD=0|1 (of /2), GPUS=CUDA_VISIBLE_DEVICES, PORT=TRAIN_MASTER_PORT,
#   SETTINGS=settings jsonl, LOG=log_file under experiment/logs/, DONETAG=flag suffix.
# Fresh clean re-run with the process-group-scoped runner: --master /nonexistent so NO cell
# is skipped (we want every cell re-measured into a clean log; originals are untouched).
set -uo pipefail
SHARD=$1; GPUS=$2; PORT=$3; SETTINGS=$4; LOG=$5; DONETAG=$6
source /home/miniconda3/etc/profile.d/conda.sh && conda activate /home/groot_env
export LD_LIBRARY_PATH=/home/dgl_groot/third_party/build/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=$GPUS
export TRAIN_MASTER_PORT=$PORT
cd /home/dgl_groot/experiment/script
python3 run_settings.py --settings "$SETTINGS" --data_dir /home/data --shard "$SHARD"/2 \
  --log_file "$LOG" --master /nonexistent --timeout 1200
rc=$?
echo "WORKER $SHARD ($DONETAG) FINISHED rc=$rc"
touch "/home/dgl_groot/DONE.${DONETAG}.${SHARD}"
