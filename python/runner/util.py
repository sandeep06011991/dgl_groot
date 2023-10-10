from dataclasses import dataclass
import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset
import dgl
import argparse
import time
import torchmetrics.functional as MF

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
    num_partitions: int = 0
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
    parser.add_argument('--model', default="gat", type=str, help='Model type: graphsage or gat', choices=['graphsage', 'gat'])
    parser.add_argument('--graph', default="ogbn-arxiv", type=str, help="Input graph name any of ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']", choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'])
    parser.add_argument('--world_size', default=1, type=int, help='Number of GPUs')
    parser.add_argument('--hid_feat', default=256, type=int, help='Size of hidden feature')
    parser.add_argument('--cache_rate', default=0.1, type=float, help="percentage of feature data cached on each gpu")
    parser.add_argument('--sample_only', default=False, type=bool, help="whether test system on sampling only mode", choices=[True, False])
    parser.add_argument('--test_acc', default=False, type=bool, help="whether test model accuracy", choices=[True, False])
    parser.add_argument('--num_redundant_layers', default = 0, type = int, help = "number of redundant layers")
    parser.add_argument('--random_partition', default = True, type = bool)
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
    config.fanouts = [5,5,5]
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
    torch_type = torch.int64
    if config.graph_name != "test-data":
        print("Attempting to read")
        dataset = DglNodePropPredDataset(config.graph_name, root="/ssd/ogbn")
        print("read success full")
        graph: dgl.DGLGraph = dataset[0][0].astype(torch_type)
        label: torch.tensor = dataset[0][1]
        label = torch.flatten(label).type(torch.int64)
        torch.nan_to_num_(label, nan=-1)
        if config.model_type == "gat":
            graph = dgl.add_self_loop(graph)

        graph.ndata['label'] = label
        dataset_idx_split = dataset.get_idx_split()

    else:
        print("reading synthetic ")
        clique_size = 16
        e_list = []
        for i in range(clique_size):
            for j in range(4):
            #     if i == j:
            #         continue
                e_list.append([i,(i + 1+ j)%clique_size])
        graph = dgl.DGLGraph(e_list)
        label = torch.zeros(clique_size).to(torch.int64)
        graph.ndata['feat'] =\
            (torch.arange(graph.num_nodes()) * torch.ones((2,graph.num_nodes()),dtype = torch.float32)).T.contiguous()
        print(graph.ndata['feat'].shape)
        graph.ndata['label'] = label
        dataset_idx_split = {}
        for i in ["train", "test", "valid"]:
            dataset_idx_split[i] = torch.arange(clique_size)
        class Dataset:
            pass
        dataset = Dataset()
        dataset.num_classes = 1

    train_idx = dataset_idx_split.pop("train").type(torch_type)
    test_idx = dataset_idx_split.pop("test").type(torch_type)
    valid_idx = dataset_idx_split.pop("valid").type(torch_type)
    feat = graph.ndata.pop("feat")
    label = graph.ndata.pop("label")
    indptr, indices, edge_id = None, None, None
    if "groot" in config.system:
        indptr, indices, edge_id = graph.adj_tensors("csc")
    shared_graph = None
    if "groot" not in config.system:
        shared_graph = graph.shared_memory(config.graph_name)

    partition_map = None
    if config.random_partition:
        print("random partition")
        partition_map = torch.randint(0,4, (graph.num_nodes(),))
    else:
        partition_map = get_metis_partition(config, graph, f"/ssd/ogbn/{config.graph_name}".replace("-","_"), train_idx)

    config.in_feat = feat.shape[1]
    config.num_classes = dataset.num_classes

    return indptr, indices, edge_id, shared_graph, train_idx, test_idx, valid_idx, feat, label, partition_map

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
                              seeds: torch.Tensor):
    print("Start cache sampling")
    graph.pin_memory_()
    num_nodes = graph.num_nodes()
    num_cached = int(config.cache_percentage * num_nodes)
    sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    dataloader = dgl.dataloading.DataLoader(graph = graph, 
                                        indices = seeds,
                                        graph_sampler = sampler, 
                                        use_uva=True,
                                        batch_size=config.batch_size)
    sample_freq = torch.zeros(num_nodes, dtype=torch.int32, device=config.rank)

    for batch_input, batch_seeds, batch_blocks in dataloader:
        sample_freq[batch_input] += 1
    
    freq, node_id = sample_freq.sort(descending = True)
    cached_id = node_id[:num_cached]
    return cached_id

def get_metis_partition(config: RunConfig,\
                        graph: dgl.DGLGraph, path:str , training_nodes: torch.Tensor):
    if os.path.exists(f"{path}/partition_map"):
        return torch.load(f"{path}/partition_map")
    print(config)
    ntype = torch.zeros(graph.num_nodes(), dtype = torch.int32)
    ntype[training_nodes] = 1
    partitions = dgl.metis_partition(graph, config.num_partitions, balance_ntypes = ntype )
    p_map = torch.zeros(graph.num_nodes(), dtype = torch.int32)
    for p,nodes in enumerate(partitions):
        p_map[nodes] = p
    print(f"Saving to {path}/partition_map") 
    torch.save(p_map,f"{path}/partition_map")

    return p_map
