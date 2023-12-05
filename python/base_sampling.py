from dgl._ffi.function import _init_api
_init_api("dgl.groot", __name__)
import nvtx
import torch.multiprocessing
import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.distributed as dist
import gc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from dgl.utils import pin_memory_inplace, gather_pinned_tensor_rows
import torch
import dgl.backend as F
from torch import Tensor
from dgl.heterograph import DGLBlock
from dgl.utils import pin_memory_inplace
from sampling_util import *
import argparse

def init_groot_dataloader(rank: int, world_size: int, block_type: int, device_id: int, fanouts: list[int],
                    batch_size: int, num_redundant_layers: int, max_pool_size: int,
                    indptr: Tensor, indices: Tensor, feats: Tensor, labels: Tensor,
                    train_idx: Tensor, valid_idx: Tensor, test_idx: Tensor, partition_map: Tensor):
    # assert(partition_map != None)
    if partition_map is None:
        # todo: partition map should be read from disk
        # todo: partition map must be consistent with others
        print("Using synthetic product map")
        partition_map = torch.arange(indptr.shape[0] - 1) % world_size
    print("Temp by pass")
    if num_redundant_layers == len(fanouts):
        assert(block_type == 0)
    else:
        assert(block_type == 1 or block_type == 2)

    return _CAPI_InitDataloader(rank, world_size, block_type, device_id, 
                                fanouts, batch_size, num_redundant_layers, max_pool_size,
                                F.zerocopy_to_dgl_ndarray(indptr),
                                F.zerocopy_to_dgl_ndarray(indices),
                                F.zerocopy_to_dgl_ndarray(feats),
                                F.zerocopy_to_dgl_ndarray(labels),
                                F.zerocopy_to_dgl_ndarray(train_idx.to(device_id)),
                                F.zerocopy_to_dgl_ndarray(valid_idx.to(device_id)),
                                F.zerocopy_to_dgl_ndarray(test_idx.to(device_id)),
                                F.zerocopy_to_dgl_ndarray(partition_map.to(device_id)))

def init_groot(fanouts: list[int], batch_size: int, max_pool_size: int,
                    indptr: Tensor, indices: Tensor, feats: Tensor, labels: Tensor,
                    train_idx: Tensor, valid_idx: Tensor, test_idx: Tensor):
    rank: int = 0
    world_size: int = 1
    block_type: int = 0
    device_id: int = 0
    num_redundant_layers: int = len(fanouts)
    partition_map: Tensor = Tensor()
    
    return init_groot_dataloader(rank, world_size, block_type, device_id, fanouts, batch_size, num_redundant_layers, max_pool_size,
                                indptr, indices, feats, labels, train_idx, valid_idx, test_idx, partition_map)

def get_batch_graph(key: int, layers: int = 3) -> list[DGLBlock]:
    blocks = []
    for i in range(layers):
        gidx = _CAPI_GetBlock(key, i)
        block = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        blocks.insert(0, block)
    return blocks

def get_feat_label(key:int) -> Tensor:
    def extract_batch_feat_label(key: int, is_async: bool) -> None:
        return _CAPI_ExtractFeatLabel(key, is_async)
    
    extract_batch_feat_label(key, True)
    labels = _CAPI_GetLabel(key)
    feat = _CAPI_GetFeat(key)
    return F.zerocopy_from_dgl_ndarray(feat), F.zerocopy_from_dgl_ndarray(labels)

def sample_batch(replace: bool) -> int:
    return _CAPI_NextSync(replace)

def sample_batches(num_batches: int, batch_layer: int, replace: bool) -> int:
    return _CAPI_BatchNextSync(num_batches, batch_layer, replace)

def bench(configs: list[Config], test_acc=False):
    for config in configs:
        assert(config.system == configs[0].system and config.graph_name == configs[0].graph_name)
    in_dir = os.path.join(configs[0].data_dir, configs[0].graph_name)
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
    indptr, indices, _label = load_graph(in_dir, is32=True, wsloop=True)
    feat, label, num_label = load_feat_label(in_dir)
    _indptr = pin_memory_inplace(indptr)
    _indices = pin_memory_inplace(indices)
    _feat = pin_memory_inplace(feat)
    _label = pin_memory_inplace(label)
    
    for config in configs:
        # Default settings
        try:
            base_sampling(config, indptr, indices, train_idx, test_idx, valid_idx, feat, label)
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
        
def base_sampling(config: Config, indptr, indices, train_idx, test_idx, valid_idx, feat, label):
    print(f"{config=}")
    groot = init_groot(config.fanouts, config.batch_size, config.pool_size, indptr, indices, feat, label, train_idx, valid_idx, test_idx)
    step = int(train_idx.shape[0] / config.batch_size)
    print("start base sampling")
    timer = Timer()
    key = 0
    num_edges = 0
    for epoch in range(config.num_epoch):
        sampled_minibatch = 0
        while ((key + 1) // step == epoch):            
            key = sample_batch(replace=config.replace)
            blocks = get_batch_graph(key)
            for block in blocks:
                num_edges += block.num_edges()
                
            sampled_minibatch += 1
            print(f"base {sampled_minibatch=} {key=}")
    
    print(f"base {num_edges=}")
    torch.cuda.synchronize()
    duration = timer.duration()
    sampling_time = duration
    profiler = Profiler(num_epoch=config.num_epoch, duration=duration, sampling_time=sampling_time, feature_time=0.0, forward_time=0.0, backward_time=0.0, test_acc=0)
    write_to_csv(config.log_path, [config], [profiler])
    
def get_configs(graph_name, batch_size, system, log_path, data_dir, pool_size, replace):
    fanouts = [ [20, 20, 20]]
    pool_sizes = [pool_size]
    configs = []
    for fanout in fanouts:
        for pool_size in pool_sizes:
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
                            pool_size=pool_size,
                            batch_layer=0,
                            replace=replace)
            configs.append(config)
    return configs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--epoch', default=1, type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=256, type=int, help='Input batch size on each device (default: 256)')
    parser.add_argument('--pool_size', default=1, type=int, help='pool size on each device (default: 1)')
    parser.add_argument('--replace', default=True, type=bool, help='use replace sampling or not')
    parser.add_argument('--graph_name', default="ogbn-products", type=str, help="Input graph name any of ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']", choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'])
    args = parser.parse_args()
    pool_size=int(args.pool_size)
    batch_size=int(args.batch_size)
    replace=bool(args.replace)
    graph_name=str(args.graph_name)
    configs = get_configs(graph_name, batch_size, "base", "/mnt/homes/juelinliu/project/dgl_groot/python/exp.csv", "/mnt/homes/juelinliu/dataset/OGBN/processed", pool_size, replace)
    bench(configs=configs)
    # configs = get_configs("ogbn-papers100M", "batch", "/mnt/homes/juelinliu/project/dgl_groot/python/log/batch.csv", "/mnt/homes/juelinliu/dataset/OGBN/processed")
    # bench_groot_batch(configs=configs)