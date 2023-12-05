from dgl._ffi.function import _init_api
_init_api("dgl.groot", __name__)
import nvtx
import torch.multiprocessing
import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.distributed as dist
import gc
import argparse
import torch
import dgl.backend as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from dgl.utils import pin_memory_inplace, gather_pinned_tensor_rows

from torch import Tensor
from dgl.heterograph import DGLBlock
from dgl.utils import pin_memory_inplace
from sampling_util import *

def bench(configs: list[Config], test_acc=False):
    for config in configs:
        assert(config.system == configs[0].system and config.graph_name == configs[0].graph_name)
    in_dir = os.path.join(configs[0].data_dir, configs[0].graph_name)
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
    graph = load_dgl_graph(in_dir, is32=True, wsloop=False)
    # feat, label, num_label = load_feat_label(in_dir)
    graph.create_formats_()
    graph.pin_memory_()
    
    for config in configs:
        # Default settings
        try:
            sampling(config, graph, train_idx, test_idx, valid_idx)
        except Exception as e:
            if "out of memory" in str(e):
                print("out of memory for", config)
                write_to_csv(config.log_path, [config], [oom_profiler()])
                continue
            else:
                write_to_csv(config.log_path, [config], [empty_profiler()])
                with open(f"exceptions/{config.get_file_name()}", 'w') as fp:
                    fp.write(str(e))
                continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def sampling(config: Config, graph, train_idx, test_idx, valid_idx):
    print(f"{config=}")
    graph_sampler = dgl.dataloading.NeighborSampler(fanouts=config.fanouts, replace=config.replace)
    dataloader = dgl.dataloading.DataLoader(
        graph=graph,               # The graph
        indices=train_idx.to(0),         # The node IDs to iterate over in minibatches
        graph_sampler=graph_sampler,     # The neighbor sampler
        device=0,      # Put the sampled MFGs on CPU or GPU
        use_ddp=False, # enable ddp if using mutiple gpus
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=config.batch_size,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=True,    # Whether to drop the last incomplete batch
        use_uva=True,
        num_workers=0,
    )
    timer = Timer()
    num_edges = 0
    for epoch in range(config.num_epoch):
        for input_nodes, output_nodes, blocks in dataloader:
            for block in blocks:
                num_edges += block.num_edges()
    print(f"{num_edges=}")
    duration = timer.duration()
    sampling_time = duration
    profiler = Profiler(num_epoch=config.num_epoch, duration=duration, sampling_time=sampling_time, feature_time=0.0, forward_time=0.0, backward_time=0.0, test_acc=0)
    write_to_csv(config.log_path, [config], [profiler])
    
def get_configs(graph_name, batch_size, system, log_path, data_dir, replace):
    # fanouts = [ [10, 10, 10], [15, 15, 15], [20, 20, 20]]
    # batch_sizes = [256, 1024, 4096, 8192]
    fanouts = [ [20, 20, 20]]
    batch_sizes = [batch_size]
    configs = []
    for fanout in fanouts:
        for batch_size in batch_sizes:
            config = Config(graph_name=graph_name, 
                            world_size=1, 
                            num_epoch=1, 
                            fanouts=fanout, 
                            batch_size=batch_size, 
                            system=system, 
                            model="sage",
                            hid_size=128, 
                            cache_size=0, 
                            log_path=log_path,
                            data_dir=data_dir,
                            pool_size=1,
                            batch_layer=1,
                            replace=replace)
            configs.append(config)
    return configs

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--graph_name', default="ogbn-products", type=str, help="Input graph name any of ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']", choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'])
    parser.add_argument('--epoch', default=1, type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=256, type=int, help='Input batch size on each device (default: 256)')
    parser.add_argument('--replace', default=True, type=bool, help='use replace sampling or not')
    parser.add_argument('--pool_size', default=4, type=int, help='pool size on each device (default: 1)')
    parser.add_argument('--batch_layer', default=0, type=int, help='at which layer starts to use batch loading')
    
    args = parser.parse_args()
    pool_size=int(args.pool_size)
    batch_size=int(args.batch_size)
    replace=bool(args.replace)
    graph_name=str(args.graph_name)
    batch_layer=int(args.batch_layer)
    log_file_path = os.path.join(os.path.dirname(__file__), "exp.csv")
    data_dir = "/mnt/homes/juelinliu/dataset/OGBN/processed"
    configs = get_configs(graph_name, batch_size, "dgl", log_file_path, data_dir, replace)
    bench(configs=configs)
    # configs = get_configs("ogbn-products", "dgl", "/mnt/homes/juelinliu/project/dgl_groot/python/exp.csv", "/mnt/homes/juelinliu/dataset/OGBN/processed")
    # bench(configs=configs)