"""
Run the hybrid-parallelism cost-model simulator and write results the same way
the e2e tests do (Config columns + predicted costs, via utils.write_to_csv).

The simulator predicts, per split-point k, the per-epoch:
  - computation_cost     (model-layer compute, timed on GPU)
  - communication_cost   (cross-partition feature volume / interconnect BW)
  - dataloading_cost     (frontier feature transfer / PCIe)
k is the layer to "split from": k=0 is full split parallelism (what the e2e
`split` system runs), k=num_layers is full data parallelism.

Usage (run after partitioning; same flags as the e2e runner):
    cd experiment
    python3 get_best_strategy.py --graph_name=products --data_dir=/root/graph \
        --fanouts="15,15,15" --world_size=2 --hid_size=256 --model=sage \
        --node_weight=dst --edge_weight=freq --bal=xbal --nvlink --num_epoch=1 \
        --log_file=products_sage_sim.csv
"""
import os
import time
import torch
from utils import get_args, get_partition_type, Config, write_to_csv
from utils.dataloading import get_feat_dim, load_partition_map, load_topo
from simulation import simulate_optimal_k
from nodepred import Sage, Gat

# OGB node-classification class counts (output dim for the last layer).
NUM_CLASSES = {"products": 47, "arxiv": 40, "papers100M": 172, "orkut": 100, "friendster": 100}


def run_simulation_hybrid_dp_split(cfg: Config, device: int = 0):
    """Sweep the split-point k and write one predicted-cost row per k."""
    s_time = time.time()
    graph, train_idx, _, _ = load_topo(cfg, is_pinned=False)
    input_feat_size = get_feat_dim(cfg)
    partition_map = load_partition_map(cfg)
    print(f"{cfg.graph_name} loaded in {time.time() - s_time:.2f}s; "
          f"nodes={graph.num_nodes()} edges={graph.num_edges()}")

    if cfg.model == "sage":
        model = Sage(cfg.in_feat, cfg.hid_size, len(cfg.fanouts), cfg.num_classes).to(device)
    elif cfg.model == "gat":
        model = Gat(cfg.in_feat, cfg.hid_size, len(cfg.fanouts), cfg.num_classes).to(device)
    else:
        raise ValueError(f"unsupported model {cfg.model}")
    model_layers = list(model.layers)

    # A100-SXM (NVLink) vs PCIe interconnect for the communication model.
    num_nvlinks = 4 if cfg.nvlink else 0
    graph = graph.to(device)
    train_idx = train_idx.to(device)
    partition_map = partition_map.to(device)

    # k = split-from-layer: 0 = full split (matches the e2e split system),
    # len(fanouts) = full data parallelism.
    for k in range(0, len(cfg.fanouts) + 1):
        simulate_optimal_k(graph, train_idx, partition_map, device, k,
                           cfg.batch_size, cfg.fanouts, model_layers,
                           cfg.world_size, num_nvlinks, input_feat_size, cfg)
        print(f"simulated k={k} ok", flush=True)


if __name__ == "__main__":
    args = get_args()
    fanouts = [int(f) for f in args.fanouts.split(",")]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_file = args.log_file if args.log_file else f"{args.graph_name}_{args.model}_sim.csv"
    log_path = os.path.join(dir_path, "logs", log_file)
    os.makedirs(os.path.join(dir_path, "logs"), exist_ok=True)

    cfg = Config(
        graph_name    = args.graph_name,
        world_size    = args.world_size,
        num_partition = args.world_size,
        num_epoch     = args.num_epoch,
        fanouts       = fanouts,
        batch_size    = args.batch_size,
        system        = "split-sim",
        model         = args.model,
        cache_size    = "0GB",
        hid_size      = args.hid_size,
        log_path      = log_path,
        data_dir      = args.data_dir,
        nvlink        = args.nvlink,
        partition_type= get_partition_type(args.node_weight, args.edge_weight, args.bal),
        sample_mode   = args.sample_mode,
    )
    cfg.in_feat = get_feat_dim(cfg)
    cfg.num_classes = NUM_CLASSES.get(cfg.graph_name, 172)
    assert cfg.in_feat > 0

    run_simulation_hybrid_dp_split(cfg)
    print("simulation done ->", log_path)
