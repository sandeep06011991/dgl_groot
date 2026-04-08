#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "Start optimal k Experiment on $graph"

# Default parameters
fanout="10,10,10"
batch_size=1024
graphs=("arxiv" "arxiv")
graphs=("products")

mode="hybriddp"
for model in gat; do
  for graph in $graphs; do
    python3 ${python_dir}/get_best_strategy.py  --model=${model}  \
    --graph=${graph}  --data_dir=${data_dir}  \
    --batch_size=${batch_size} --log_file=${python_dir}/logs/${mode}.csv \
    --fanout=${fanout}
  done
done
