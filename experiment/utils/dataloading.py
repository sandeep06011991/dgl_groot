import torch
import dgl
import os
from .config import Config
from dgl.utils import pin_memory_inplace, gather_pinned_tensor_rows
from .timer import Timer
import numpy as np

def load_numpy(path):
    return torch.from_numpy(np.asarray(np.load(path)))

def save_numpy(out:torch.Tensor, outpath: str):
    assert(outpath.endswith(".npy"))
    np.save(outpath, out.numpy(force=True))
    
def load_graph(in_dir, is32=False, wsloop=False, is_sym=False, load_edge_weight=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    symtype_str = "sym" if is_sym else "xsym"
    indptr = load_numpy(os.path.join(in_dir, f"indptr_{symtype_str}.npy"))
    indices = load_numpy(os.path.join(in_dir, f"indices_{symtype_str}.npy"))
    edges = torch.empty(0, dtype=indices.dtype)
    if load_edge_weight:
        edges = load_numpy(os.path.join(in_dir, "edge_weight.npy")).type(torch.int64)
    if wsloop and is_sym == False:
        graph: dgl.DGLGraph = dgl.graph(("csc", (indptr, indices, edges)))
        graph = dgl.add_self_loop(graph)
        indptr, indices, edges = graph.adj_tensors("csc")
    if is32:
        return indptr.type(torch.int32), indices.type(torch.int32), edges.type(torch.int32)
    else:
        return indptr, indices, edges

def load_dgl_graph(in_dir, is32=False, wsloop=False, is_sym=False, load_edge_weight=False)->dgl.DGLGraph:
    indptr, indices, edges = load_graph(in_dir, is32, wsloop, is_sym, load_edge_weight)
    return dgl.graph(("csc", (indptr, indices, edges)))

def load_idx_split(in_dir, is32=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print("load idx split from", in_dir)
    train_idx = load_numpy(os.path.join(in_dir, f"train_idx.npy"))
    valid_idx = load_numpy(os.path.join(in_dir, f"valid_idx.npy"))
    test_idx = load_numpy(os.path.join(in_dir, f"test_idx.npy"))
    if is32:
        return train_idx.type(torch.int32), valid_idx.type(torch.int32), test_idx.type(torch.int32)
    else:
        return train_idx, valid_idx, test_idx

def load_feat_label(in_dir) -> tuple[torch.Tensor, torch.Tensor, int]:
    feat = load_numpy(os.path.join(in_dir, f"feat.npy"))
    label = load_numpy(os.path.join(in_dir, f"label.npy"))
    num_labels = torch.unique(label).shape[0]
    return feat, label, num_labels


def load_partition_map(config: Config):
    in_dir = os.path.join(config.data_dir, "partition_map", config.graph_name)
    file_name = f"{config.graph_name}_w{config.num_partition}_{config.partition_type}.npy"
    return load_numpy(os.path.join(in_dir, file_name)).type(torch.uint8)

def gen_rand_feat(v_num, feat_dim):
    return torch.empty((v_num, feat_dim))

def gen_rand_label(v_num, num_classes):
    return torch.randint(low=0, high=num_classes, size=(v_num,))

def gen_feat_label(v_num, feat_dim, num_classes=10):
    feat = gen_rand_feat(v_num, feat_dim)
    label = gen_rand_label(v_num, num_classes)
    return feat, label, num_classes

def load_topo(config: Config, is_pinned=False):
    print("Start graph topology loading")
    t1 = Timer()
    in_dir = os.path.join(config.data_dir, config.graph_name)
    wsloop = "gat" in config.model
    is32 = False
    is_sym = config.graph_name in ["orkut", "friendster"]
    graph = load_dgl_graph(in_dir, is32=is32, wsloop=wsloop, is_sym=is_sym)
    train_idx, valid_idx, test_idx = load_idx_split(in_dir, is32=is32)
    if is_pinned:
        graph = graph.pin_memory_()
        
    return graph, train_idx, valid_idx, test_idx

def get_feat_dim(config: Config):
    if config.graph_name == "orkut":
        return 512
    elif config.graph_name == "friendster":
        return 128
    elif config.graph_name == "papers100M":
        return 128
    elif config.graph_name == "products":
        return 100
    else:
        print("Invalid graph name", config.graph_name)
        exit(-1)

def get_feat_bytes(feat:torch.Tensor):
    res = feat.element_size()
    for dim in feat.shape:
        res *= dim
    return res        

def get_feat_bytes_str(feat:torch.Tensor):
    feat_bytes = get_feat_bytes(feat)
    num_gb = round(feat_bytes / 1024 / 1024 / 1024)
    return f"{num_gb}GB"

def load_data(config: Config, is_pinned=False):
    print("Start data loading")
    t1 = Timer()
    in_dir = os.path.join(config.data_dir, config.graph_name)
    wsloop = "gat" in config.model
    is32 = False
    is_sym = config.graph_name in ["orkut", "friendster"]
    graph = load_dgl_graph(in_dir, is32=is32, wsloop=wsloop, is_sym=is_sym)
    train_idx, valid_idx, test_idx = load_idx_split(in_dir, is32=is32)    
    feat = None
    label = None
    num_label = None
    if config.graph_name in ["products"]: # also supports papers100M
        feat, label, num_label = load_feat_label(os.path.join(config.data_dir, config.graph_name))
    else:
        v_num = graph.num_nodes()
        feat_dim = get_feat_dim(config)
        feat = gen_rand_feat(v_num, feat_dim)
        num_label = 10
        label = gen_rand_label(v_num, 10)
        
    config.in_feat = feat.shape[1]
    print(f"Data loading total time {t1.duration()} secs")
    
    if is_pinned:
        print("pining data")
        nd_feat = pin_memory_inplace(feat)
        nd_label = pin_memory_inplace(label)
        graph = graph.pin_memory_()
    return graph, feat, label, train_idx, valid_idx, test_idx, num_label

def gather_tensor(intensor:torch.Tensor, row:torch.Tensor):
    if intensor.is_pinned():
        return gather_pinned_tensor_rows(intensor, row)
    else:
        return intensor[row]
