

### Dataset Format

Graphs are stored as NumPy arrays in `<data_dir>/<graph_name>/`:
- `indptr_xsym.npy` / `indptr_sym.npy` — CSC indptr
- `indices_xsym.npy` / `indices_sym.npy` — CSC indices
- `feat.npy`, `label.npy` — node features and labels
- `train_idx.npy`, `valid_idx.npy`, `test_idx.npy` — split indices
- `edge_weight.npy`, `node_weight_*.npy` — frequency weights for partitioning

Partition maps live in `<data_dir>/partition_map/<graph_name>/`:
- Filename convention: `<graph_name>_w<num_partitions>_n<node_weight>_e<edge_weight>_<bal>.npy`
- e.g., `products_w4_ndst_efreq_xbal.npy`

Supported graph names: `products`, `papers100M`, `orkut`, `friendster`, `arxiv`.

## Running Experiments

**Configure data paths** in `experiment/script/env.sh`:
```bash
WORKSPACE_DIR="/path/to/workspace"
dataset_dir=$WORKSPACE_DIR/dataset/
data_dir=$WORKSPACE_DIR/graph/
```

**Run a single training job:**
```bash
cd experiment
python3 train_main.py \
  --system=split \
  --model=sage \
  --fanout="15,15,15" \
  --graph=products \
  --data_dir=/path/to/graph \
  --cache_size=10G \
  --batch_size=1024 \
  --world_size=4 \
  --log_file=my_run.csv
```

`--system` choices: `split` (Spara), `dgl`, `quiver`, `p3`
`--model` choices: `sage`, `gat`
`--sample_mode` choices: `uva` (default), `gpu`

**Run all main benchmarks:**
```bash
bash experiment/script/main.sh
```

**Dataset preparation:**
```bash
cd experiment/prepare_dataset
python3 get_npgraph.py --data_dir=../../dataset/graph --graph_name=products
python3 get_weight.py --data_dir=../../dataset/graph --graph_name=products
python3 get_partition.py --num_partition=4 --graph_name=products \
    --data_dir=../../dataset/graph --node_mode=dst --edge_mode=freq --bal=xbal
```

## Python Tests

```bash
# Run DGL Python unit tests (pytorch backend)
cd tests/python/pytorch
python3 -m pytest <test_file.py> -v

# Run a single test
python3 -m pytest tests/python/pytorch/test_graph.py::test_create_graph -v
```

The `tests/` directory mirrors DGL upstream structure. Spara-specific tests are not separated into their own directory — they live alongside experiment scripts.

### Baseline Systems (in `experiment/nodepred/`)

| File | System | Description |
|---|---|---|
| `dgl_trainer.py` | `dgl` | Standard DGL multi-GPU with replicated graph |
| `split_trainer.py` | `split` | Spara partition-aware system |
| `quiver_trainer.py` | `quiver` / `dist_cache` | Quiver-based baseline |
| `p3_trainer.py` | `p3` | P3 pipeline-parallel baseline |