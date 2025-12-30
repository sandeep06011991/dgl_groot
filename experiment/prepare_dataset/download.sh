#!/bin/bash

# Script downloads and processes files. 

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${WORK_DIR}/../script/env.sh"

# Download Dataset
mkdir -p $data_dir
pushd $data_dir
# aria2c -x 16 -s 16 http://snap.stanford.edu/ogb/data/nodeproppred/products.zip
# aria2c -x 16 -s 16 https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
# aria2c -x 16 -s 16 http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
# aria2c -x 16 -s 16 https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
echo $1
if [[ -z "$1" ]]; then
  echo "Dataset name is empty"
  # You can add an exit command here if necessary
  exit 0
fi

if [[ "$1" == "arxiv" ]]; then
    if [ ! -d "ogbn_arxiv" ]; then 
        echo "Downloading" $1
        wget http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
        unzip arxiv.zip && mv -f arxiv ogbn_arxiv
    fi
    graph_name="arxiv"
fi 
if [[ "$1" == "papers" ]]; then 
    if [ !  -d "ogbn_papers100M" ]; then 
        echo "Downloading" $1
        wget http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
        unzip papers100M-bin.zip && mv -f papers100M-bin ogbn_papers100M
    fi
    graph_name="papers100M"
fi 
if [[ "$1" == "products" ]]; then 
    if [ !  -d "ogbn_products" ]; then 
        wget http://snap.stanford.edu/ogb/data/nodeproppred/products.zip
        unzip products.zip && mv -f products ogbn_products
    fi
    graph_name="products"
fi 
if [[ "$1" == "friendster" ]]; then 
    if [ !  -d "friendster" ]; then 
        wget https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
        gzip -d com-friendster.ungraph.txt.gz && mkdir -p friendster && mv -f com-friendster.ungraph.txt friendster/friendster.txt
    fi 
    graph_name="friendster"
fi 
if [[ "$1" == "orkut" ]]; then 
    if [ !  -d "orkut" ]; then 
        wget https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
        gzip -d com-orkut.ungraph.txt.gz && mkdir -p orkut && mv -f com-orkut.ungraph.txt orkut/orkut.txt
    fi 
    graph_name="orkut"
fi 



# rm -f *.zip
# rm -f *.gz
popd

num_epoch=5
world_size=1

echo ${graph_name} "Downloaded"

# python3 ${python_dir}/prepare_dataset/get_npgraph.py --data_dir=$data_dir --graph_name=$graph_name

echo ${graph_name} "Processing done"

python3 ${python_dir}/prepare_dataset/get_weight_fast.py --data_dir=$data_dir --graph_name=$graph_name --fanouts=15,15,15  --num_epoch=${num_epoch} --world_size=${world_size}

echo ${graph_name} "Partitioning done"

node_weight="dst"
edge_weight="freq"
bal="xbal"

PYTHONFAULTHANDLER=1  python3 ${python_dir}/prepare_dataset/get_partition.py --graph_name=$graph_name --data_dir=$data_dir --node_weight=$node_weight --edge_weight=$edge_weight --bal=$bal

