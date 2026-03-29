#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "Start optimal k Experiment on $graph"


for model in sage gat; do
  for graph in arxiv product papers friendster; do
    python3 ${python_dir}/get_best_strategy.py --system=${system} --model=${model} --fanout="15,15,15" \
    --graph=${graph} --world_size=${world_size} --data_dir=${data_dir}  
     --batch_size=${batch_size} --log_file=${python_dir}/logs/batch_size.csv
  done
done
