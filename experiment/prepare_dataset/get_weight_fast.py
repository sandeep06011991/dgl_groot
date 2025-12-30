import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import torch.distributed as dist
import torch
from torch.multiprocessing import spawn
from utils import *
from dgl.dev import *
from dgl.dev.cnt_sampler import *

def freq(config: Config):
    graph, train_idx, valid_idx, test_idx = load_topo(config, is_pinned=True)
    avg_deg = graph.num_edges() // graph.num_nodes()
    
    fanouts = None
    if "," in config.fanouts:
        fanouts = config.fanouts.split(",")
    elif "-" in config.fanouts:
        fanouts = config.fanouts.split("-")
        
    for idx, fanout in enumerate(fanouts):
        fanouts[idx] = int(fanout)
        
    config.fanouts = fanouts
    
    if config.world_size == 1:
        _freq_single(config, graph, train_idx)
    else:
        try:
            spawn(_freq, args=(config, graph, train_idx), nprocs=config.world_size)
        except Exception as e:
            print(f"error encountered with {config=}:", e)
            exit(-1)

def _freq_single(config: Config, graph: dgl.DGLGraph, train_idx: torch.Tensor):
    torch.cuda.set_device(0)
    device = torch.cuda.current_device()
    rank = 0
    print(config)
        
    sample_config = SampleConfig(rank=rank, batch_size=config.batch_size, world_size=config.world_size, mode=config.sample_mode, fanouts=config.fanouts, reindex=False)
    dataloader = CntSampler(graph, train_idx, sample_config)
    step_per_epoch = dataloader.max_step_per_epoch
    
    print(f"sampling on device: {device}")        
    timer = Timer()
    epoch_num = config.num_epoch
    for epoch in range(epoch_num):
        for batch_id in dataloader:
            log_step(rank, epoch, batch_id + 1, step_per_epoch, timer)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"get weight in {timer.duration()} secs")
    node_weight = CntGetNodeFreq()
    edge_weight = CntGetEdgeFreq()
    print(f"{node_weight=}")
    print(f"{edge_weight=}")
    out_dir = os.path.join(config.data_dir, config.graph_name)
    # out_dir = os.path.join(config.data_dir, "../weight", "-".join(str(fanout) for fanout in config.fanouts), config.graph_name)
    print("saving to", out_dir)
    os.makedirs(name=out_dir, exist_ok=True)
    save_numpy(node_weight, f"{out_dir}/node_weight.npy")
    save_numpy(edge_weight, f"{out_dir}/edge_weight.npy")
    print("All saving done !")

def _freq(rank: int, config: Config, graph: dgl.DGLGraph, train_idx: torch.Tensor):
    ddp_setup(rank, config.world_size, "nccl")
    device = torch.cuda.current_device()

    if rank == 0:
        print(config)
        
    sample_config = SampleConfig(rank=rank, batch_size=config.batch_size, world_size=config.world_size, mode=config.sample_mode, fanouts=config.fanouts, reindex=False)
    dataloader = CntSampler(graph, train_idx, sample_config)
    step = 0
    step_per_epoch = dataloader.max_step_per_epoch
    
    print(f"sampling on device: {device}")        
    timer = Timer()
    epoch_num = config.num_epoch
    for epoch in range(epoch_num):
        for batch_id in dataloader:
            step += 1
            log_step(rank, epoch, step, step_per_epoch, timer)
    
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"get weight in {timer.duration()} secs")
    
    node_weight = CntGetNodeFreq()
    edge_weight = CntGetEdgeFreq()
    
    dist.all_reduce(node_weight)
    dist.all_reduce(edge_weight)
    
    if rank == 0:
        out_dir = os.path.join(config.data_dir, config.graph_name)
        # out_dir = os.path.join(config.data_dir, "../weight", "-".join(str(fanout) for fanout in config.fanouts), config.graph_name)
        print("saving to", out_dir)
        os.makedirs(name=out_dir, exist_ok=True)
        save_numpy(node_weight, f"{out_dir}/node_weight.npy")
        save_numpy(edge_weight, f"{out_dir}/edge_weight_epoch.npy")
    ddp_exit()

if __name__ == "__main__":
    args = get_args()
    graph_name = str(args.graph_name)
    data_dir = args.data_dir
    batch_size = args.batch_size
    fanouts = None
    if "-" in args.fanouts:
        fanouts = args.fanouts.split('-')
    elif "," in args.fanouts:
        fanouts = args.fanouts.split(',')

    num_epoch=args.num_epoch
    world_size =args.world_size
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "logs/exp.csv")
    cfg = Config(graph_name=graph_name,
                       world_size=world_size,
                       num_epoch=num_epoch,
                       num_partition=1,
                       fanouts=args.fanouts,
                       batch_size=batch_size,
                       system="dgl-sample",
                       model="none",
                       cache_size="0GB",
                       hid_size=256,
                       log_path=log_path,
                       data_dir=data_dir)
    print("Go to freq")
    freq(cfg)
    print("Return from freq")
    import os 
    os._exit(1)
