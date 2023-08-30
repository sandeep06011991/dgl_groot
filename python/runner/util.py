from dataclasses import dataclass
import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset
import dgl
import argparse
import time

class Timer:
    def __init__(self) -> None:
        self.start_time = time.time()
    
    def reset(self) -> None:
        self.start_time = time.time()
        
    def passed(self) -> float:
        end_time = time.time()
        return end_time - self.start_time
    
@dataclass
class RunConfig:
    system: str = "dgl-uva" # choice of ["pyg", "dgl-cpu", "dgl-uva", "quiver", "groot"]
    graph_name: str = "ogbn-products" # choice of ["ogbn-products", "ogbn-papers100M", "mag240m"]
    model_type: str = "graphsage" # choice of ["graphsage", "gat"]
    batch_size: int = 1024
    hid_feat: int = 256
    num_layer: int = 3
    num_epoch : int = 3
    sample_only: bool = False
    in_feat: int = -1 # needs to be set after loading feature data
    num_classes: int = -1 # must be set correctly
    fanouts: list[int] = None
    def is_valid(self):
        if self.in_feat < 0:
            return False
        if self.num_classes < 0:
            return False
        if self.fanouts is None:
            return False
        return True
    
@dataclass
class DGLDataset:
    graph: dgl.DGLGraph = None
    train_idx: torch.tensor = None
    valid_idx: torch.tensor = None
    test_idx: torch.tensor = None
    
def get_parser():
    parser = argparse.ArgumentParser(description='benchmarking script')
    parser.add_argument('--batch', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--system', default="dgl-uva", type=str, help='System setting', choices=["dgl-uva", "dgl-cpu", "pyg", "quiver", "groot"])
    parser.add_argument('--model', default="graphsage", type=str, help='Model type: graphsage or gat', choices=['graphsage', 'gat'])
    parser.add_argument('--graph', default="ogbn-products", type=str, help="Input graph name any of ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']", choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'])
    parser.add_argument('--nprocs', default=1, type=int, help='Number of GPUs')
    parser.add_argument('--hid_feat', default=256, type=int, help='Size of hidden feature')
    parser.add_argument('--sample_only', default=False, type=bool, choices=[True, False])
    return parser

def get_config():
    parser = get_parser()
    args = parser.parse_args()
    config = RunConfig()
    config.batch_size = args.batch
    config.system = args.system
    config.model_type = args.model
    config.graph_name = args.graph
    config.hid_feat = args.hid_feat
    config.fanouts = [5, 10, 15]
    config.sample_only = args.sample_only
    return config

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_exit():
    destroy_process_group()

def load_dgl_dataset(config: RunConfig) -> DGLDataset:
    dataset = DglNodePropPredDataset(config.graph_name, root="/data")
    graph: dgl.DGLGraph = dataset[0][0]
    label: torch.tensor = dataset[0][1]
    label = torch.flatten(label).type(torch.int64)
    torch.nan_to_num_(label, nan=-1)
        
    if config.model_type == "gat":
        graph = dgl.add_self_loop(graph)
    
    graph.ndata['label'] = label
    dataset_idx_split = dataset.get_idx_split()
    dgl_dataset = DGLDataset()
    dgl_dataset.train_idx = dataset_idx_split.pop("train")
    dgl_dataset.test_idx = dataset_idx_split.pop("test")
    dgl_dataset.valid_idx = dataset_idx_split.pop("valid")
    dgl_dataset.graph = graph
    config.in_feat = graph.ndata["feat"].shape[1]
    config.num_classes = dataset.num_classes
    return dgl_dataset

def load_pyg_graph(config: RunConfig):
    dataset = PygNodePropPredDataset(config.graph_name, root="/data")
    return dataset