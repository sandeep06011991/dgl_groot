"""
# Run a simulation to investigate the proposed hybrid pipeline and data parallelisms. 
# Configuration provides the max configuration. 

Usage:
    cd experiment
    python3 get_best_strategy.py \
        --graph_name=products \
        --data_dir=/path/to/graph \
        --fanouts="15,15,15" \
        --world_size=4 \
        --hid_size=256 \
        --model=sage \
        --node_weight=dst \
        --edge_weight=freq \
        --bal=xbal

Done post partitioning. 
"""

import os
import torch
import numpy as np
from utils import get_args, get_partition_type, Config
from utils.dataloading import get_feat_dim, load_partition_map, load_topo 
from simulation import simulate_optimal_k
import dgl
from nodepred import Sage,Gat




def run_simulation_hybrid_dp_split(cfg: Config,  device: str = "cuda:0") -> int:
    """
    Predict the optimal num_split_layers (k) for Hybrid Split+DDP.

    Args:
        cfg:         Training config (graph_name, world_size, fanouts, hid_size, …).
        device:      CUDA device for compute benchmarking (default "cuda:0").
    Returns:        
        Optimal k (int in {1, …, num_layers}).
    """
    # Step 1: Load the dgl graph from data loading. 

    graph, train_idx, valid_idx, test_idx = load_topo(cfg, is_pinned=False)
    input_feat_size =  get_feat_dim(cfg)
    partition_map = load_partition_map(cfg)
    
    batch_size = cfg.batch_size
    fanouts = cfg.fanouts
    num_devices = cfg.world_size
    if cfg.model == "sage":
        print(cfg.in_feat, cfg.hid_size, len(cfg.fanouts), cfg.num_classes)
        model = Sage(cfg.in_feat, cfg.hid_size, len(cfg.fanouts), cfg.num_classes).to(device)
    if cfg.model == "gat":
        model = Gat(cfg.in_feat, cfg.hid_size, len(cfg.fanouts), cfg.num_classes).to(device)
    model_layers = list(model.layers)
    device = 0
    for k in range(2,len(cfg.fanouts) + 1):
        num_nvlinks = 0
        simulate_optimal_k( graph.to(device), train_idx.to(device), partition_map.to(device), device, k,
                batch_size, fanouts, model_layers, num_devices, num_nvlinks, input_feat_size, cfg)
        print("Layer was ok.", k)
    return 

def run_simulation_hybrid_pipeline_parallelism(cfg:Config, device: str) -> int:
    """
    Predicts the optimal hidden size for pipeline parallleims to have an effect.
    """
    # Iterate through hidden size variations. 
    pass 

if __name__ == "__main__":
    args = get_args()
    fanouts     = [int(f) for f in args.fanouts.split(",")]
    
    programming_model = args.cost_estimation_mode 

    cfg = Config(
        graph_name     = args.graph_name,
        world_size     = args.world_size,
        num_partition  = args.world_size,
        num_epoch      = args.num_epoch,
        fanouts        = fanouts,
        batch_size     = args.batch_size,
        system         = "split",
        model          = args.model,
        cache_size     = "0GB",
        hid_size       = args.hid_size,
        log_path       = args.log_file,
        data_dir       = args.data_dir,
        nvlink         = args.nvlink,
        sample_mode    = args.sample_mode,
    )
    cfg.in_feat = get_feat_dim(cfg)
    cfg.num_classes = 172
    assert(cfg.in_feat > 0)
    programming_model = "hybriddata"

    if programming_model == "hybriddata":
        run_simulation_hybrid_dp_split(cfg)
    else:
        print("temporary code")
        assert(False)

