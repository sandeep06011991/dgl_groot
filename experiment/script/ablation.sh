#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"
graph=friendster
echo "Start Ablation Experiment on $graph"

PYTHONPATH="${SCRIPT_DIR}/../../python:${SCRIPT_DIR}/../../third_party/torch-quiver/srcs/python"
echo $PYTHONPATH
export PYTHONPATH=$PYTHONPATH 
for model in sage gat; do
  for system in split dgl quiver; do
    batch_size=(1024)
    cache_sizes=(10G 8G 6G)
    if [$system == "dgl"]; then
        cache_sizes=(10G 8G)
    fi
    for cache_size in ${cache_sizes[@]}; do
      for fanout in "10,10" "10,10,10" "10,10,10,10"; do
        python3 ${python_dir}/train_main.py --system=${system} --model=${model} --fanout=${fanout} --graph=${graph} --world_size=${world_size} --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=${batch_size} --log_file=depth.csv
      done
      for hidden in 64 128 512; do
        python3 ${python_dir}/train_main.py --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph} --world_size=${world_size} --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=${batch_size} --hid_size=${hidden} --log_file=hidden.csv
      done
      for batch_size in 512 2048; do
        python3 ${python_dir}/train_main.py --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph} --world_size=${world_size} --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=${batch_size} --log_file=batch_size.csv
      done
      python3 ${python_dir}/train_main.py --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph} --world_size=2 --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=512 --log_file=scalability.csv
    done
  done
done
