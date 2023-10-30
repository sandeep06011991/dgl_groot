from dataclasses import dataclass
import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from ogb.nodeproppred import  DglNodePropPredDataset
import dgl
import argparse
import time
import torchmetrics.functional as MF
import numpy as np

def avg_ignore_first(s):
    if(np.var(s[1:]) >= .10 * np.mean(s[1:])):
        print("Variance overflow")
        print(s)
    assert(np.var(s[1:]) < .10 * np.mean(s[1:]))
    return sum(s[1:])/(len(s) - 1)

class Timer:

    def __init__(self) -> None:
        self.accumulated = 0
        self.e1 = torch.cuda.Event(enable_timing= True)
        self.e2 = torch.cuda.Event(enable_timing= True)

    def start(self) -> None:
        self.start_time = time.time()
        self.e1.record()

    def reset(self) -> None:
        self.accumulated = 0
        self.async_accumulated  = 0
        self.start_time = 0

    def stop(self) -> None:
        self.accumulated += (time.time() - self.start_time)
        self.start_time = 0
        self.e2.record()

    def get_accumulated(self) -> float:
        return self.accumulated

    def get_async_accumulated(self) -> float:
        return self.async_accumulated

    def accumulate_async(self) -> None:
        self.async_accumulated \
                += (self.e1.elapsed_time(self.e2)/1000)

@dataclass
class Measurements:
    sampling_time: int
    training_time: int
    epoch_time: int

    def reset(self):
        self.sampling_time = 0
        self.training_time = 0





@dataclass
class RunConfig:
    rank: int = 0
    world_size: int = 1
    system: str = "dgl-uva" # choice of ["pyg", "dgl-cpu", "dgl-uva", "quiver", "groot-uva", "groot-cache"]
    graph_name: str = "ogbn-arxiv" # choice of ["ogbn-products", "ogbn-papers100M", "mag240m"]
    model_type: str = "graphsage" # choice of ["graphsage", "gat"]
    batch_size: int = 5
    hid_feat: int = 256
    num_layer: int = 3
    num_epoch : int = 3
    sample_only: bool = False
    test_acc: bool = True
    cache_percentage: float = 0.1
    in_feat: int = -1 # must be set correctly
    num_classes: int = -1 # must be set correctly
    fanouts: list[int] = None # must be set correctly
    num_redundant_layers: int  = 0# range of [0, len(fanouts)]
    num_partitions: int = 4
    random_partition: bool = True
    def is_valid(self):
        if self.in_feat < 0:
            return False
        if self.num_classes < 0:
            return False
        if self.fanouts is None:
            return False
        return True
    
# @dataclass
# class DGLDataset:
#     graph: dgl.DGLGraph = None
#     train_idx: torch.tensor = None
#     valid_idx: torch.tensor = None
#     test_idx: torch.tensor = None
#     edge_id: torch.tensor = None
#     indptr: torch.tensor = None
#     indices: torch.tensor = None
#     partition_map: torch.tensor = None
#     def get_containing_tensors(self):
#         return self.graph, self.train_idx, self.test_idx, self.valid_idx,\
#                 self.edge_id, self.indptr, self.indices, self.partition_map
#     def set_containing_tensors(self, graph, train_idx, test_idx, valid_idx, edge_id, indptr, indices, partition_map):
#         self.graph = graph
#         self.train_idx = train_idx
#         self.valid_idx = valid_idx
#         self.edge_id = edge_id
#         self.indptr = indptr
#         self.indices = indices
#         self.partition_map = partition_map
#         self.test_idx = test_idx


def get_parser():
    parser = argparse.ArgumentParser(description='benchmarking script')
    parser.add_argument('--batch', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--system', default="groot-cache", type=str, help='System setting', choices=["dgl-uva", "dgl-cpu", "dgl-gpu","pyg", "quiver", "groot-gpu", "groot-uva", "groot-cache"])
    parser.add_argument('--model', default="gat", type=str, help='Model type: graphsage,gat, gcn, hgt', choices=['graphsage', 'gat', 'gcn', 'hgt'])
    parser.add_argument('--graph', default="ogbn-arxiv", type=str, help="Input graph name any of ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']", choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'])
    parser.add_argument('--world_size', default=1, type=int, help='Number of GPUs')
    parser.add_argument('--hid_feat', default=256, type=int, help='Size of hidden feature')
    parser.add_argument('--cache_rate', default=0.1, type=float, help="percentage of feature data cached on each gpu")
    parser.add_argument('--sample_only', default=False, type=bool, help="whether test system on sampling only mode", choices=[True, False])
    parser.add_argument('--test_acc', default=True, type=bool, help="whether test model accuracy", choices=[True, False])
    parser.add_argument('--num_redundant_layers', default = 0, type = int, help = "number of redundant layers")
    parser.add_argument('--random_partition', default = False, type = bool)
    parser.add_argument('--fanout', default='5-5-5', type = str, help = "fanout during neighbour sampling ")
    return parser

def get_config():
    parser = get_parser()
    args = parser.parse_args()
    config = RunConfig()
    config.num_partitions = args.world_size
    config.batch_size = args.batch
    config.system = args.system
    config.model_type = args.model
    config.graph_name = args.graph
    config.hid_feat = args.hid_feat
    config.sample_only = args.sample_only
    config.cache_percentage = args.cache_rate
    config.test_acc = args.test_acc
    config.fanouts = [int(f) for f in args.fanout.split("-")]
    config.world_size = args.world_size
    config.num_redundant_layers = args.num_redundant_layers
    config.random_partition = args.random_partition
    return config

def get_block_type(type:str):
    if type.lower() == "dp":
        return 0
    elif type.lower() == "src_to_dst":
        return 1
    elif type.lower() == "dst_to_src":
        return 2
    else:
        print("block type must be one of: dp / src_to_dst / dst_to_src")
        exit(-1) 
        
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_exit():
    destroy_process_group()



def load_dgl_dataset(config: RunConfig):
    torch_type = torch.int32
    dataset = DglNodePropPredDataset(config.graph_name, root="/data/ogbn")
    graph: dgl.DGLGraph = dataset[0][0].astype(torch_type)
    feat = graph.ndata.pop('feat')
    label: torch.tensor = dataset[0][1]
    label = torch.flatten(label).type(torch.int64)
    torch.nan_to_num_(label, nan=-1)
    print(label[:30])
    # No garbage
    assert(torch.all(label < 10000))
    graph.ndata['label'] = label
    dataset_idx_split = dataset.get_idx_split()
    partition_map = None
    cached_ids = None
    train_idx = dataset_idx_split.pop("train").type(torch_type)
    if "groot" in config.system:
        if config.random_partition:
            print("Using random partition")
            partition_map = torch.randint(0, 4, (graph.num_nodes(),))
        else:
            print("using metis partition")
            partition_map = get_metis_partition(config)
        train_idx_partition = []
        for i in range(4):
            train_idx_partition.append(train_idx[torch.where(partition_map[train_idx] == i)[0]])
        if config.cache_percentage != 0:
            print("need to cache")
            cached_ids = get_cache_ids_by_sampling(config, graph, train_idx_partition)
            print(cached_ids)
        train_idx = train_idx_partition
        print("Label test", label[train_idx_partition[0]])
    if config.model_type == "gat" or config.model_type == "gcn":
        print("Adding self loop")
        graph = dgl.add_self_loop(graph)
    test_idx = dataset_idx_split.pop("test").type(torch_type)
    valid_idx = dataset_idx_split.pop("valid").type(torch_type)
    label = graph.ndata.pop("label")
    shared_graph = graph
    if "groot" not in config.system:
        shared_graph = graph.shared_memory(config.graph_name)
    indptr, indices, edge_id = None, None, None
    if "groot" in config.system:
        indptr, indices, edge_id = graph.adj_tensors("csc")
    config.in_feat = feat.shape[1]
    config.num_classes = dataset.num_classes
    return indptr, indices, edge_id, shared_graph, train_idx, test_idx, \
            valid_idx, feat, label, partition_map, cached_ids

def load_pyg_graph(config: RunConfig):
    dataset = PygNodePropPredDataset(config.graph_name, root="/data")
    return dataset

def test_model_accuracy(config: RunConfig, model:torch.nn.Module, dataloader: dgl.dataloading.DataLoader):
    if config.test_acc == False:
        print("Skip model accuracy test")
        return
    
    print("Testing model accuracy")
    model.eval()
    ys = []
    y_hats = []
    
    for input_nodes, output_nodes, blocks in dataloader:
        with torch.no_grad():
            batch_feat = blocks[0].srcdata["feat"]
            batch_label = blocks[-1].dstdata["label"]
            ys.append(batch_label)
            batch_pred = model(blocks, batch_feat, inference = True)
            y_hats.append(batch_pred)
    acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=config.num_classes)
    print(f"test accuracy={round(acc.item() * 100, 2)}%")
    return acc

def get_cache_ids_by_sampling(config: RunConfig, 
                              graph: dgl.DGLGraph, 
                              seeds: list[torch.Tensor]):
    use_uva = False
    device = torch.device('cpu')
    if use_uva:
        graph.pin_memory_()
        device = torch.device(config.rank)
    num_nodes = graph.num_nodes()
    num_cached = int(config.cache_percentage * num_nodes)
    sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    cached_ids = []
    for local_seeds in seeds:
        dataloader = dgl.dataloading.DataLoader(graph = graph,
                                            indices = local_seeds,
                                            graph_sampler = sampler,
                                            use_uva=use_uva,
                                            batch_size=config.batch_size)
        sample_freq = torch.zeros(num_nodes, dtype=torch.int32, device=device )
        for batch_input, batch_seeds, batch_blocks in dataloader:
            sample_freq[batch_input] += 1

        freq, node_id = sample_freq.sort(descending = True)
        cached_id = node_id[:num_cached]
        cached_ids.append(cached_id)
    return cached_ids

def get_metis_partition(config: RunConfig):
    path = f"/data/ogbn/{config.graph_name}".replace("-", "_")
    assert (os.path.exists(f"{path}/partition_map"))
    p_map = torch.load(f"{path}/partition_map")
    assert(torch.all(p_map < 4))
    return p_map

def metis_partition(config:RunConfig):
    print(config)
    num_partitions = 4
    path = f"/data/ogbn/{config.graph_name}".replace("-", "_")
    dataset = DglNodePropPredDataset(config.graph_name, root="/data/ogbn")
    print("read success full")
    torch_type = torch.int64
    graph: dgl.DGLGraph = dataset[0][0].astype(torch_type)
    feat = graph.ndata.pop('feat')
    del feat
    import gc
    gc.collect()
    print("feature deleted")
    dataset_idx_split = dataset.get_idx_split()
    ntype = torch.zeros(graph.num_nodes(), dtype=torch_type)
    training_nodes = dataset_idx_split.pop("train").type(torch_type)
    ntype[training_nodes] = 1
    partitions = dgl.metis_partition(graph, num_partitions, balance_ntypes=ntype)
    p_map = torch.zeros(graph.num_nodes(), dtype=torch_type)
    print(partitions)
    for p_id in partitions.keys():
        nodes = partitions[p_id].ndata['_ID']
        p_map[nodes] = p_id
        print(f"In partiiton {p_id}: nodes{nodes.shape}")
    p_map = p_map.to(torch.int32)
    print(f"Saving to {path}/partition_map")
    torch.save(p_map, f"{path}/partition_map")

    return p_map
