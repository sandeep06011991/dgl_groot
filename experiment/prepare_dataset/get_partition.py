from dgl.dev.coo2csr import CompactCSR, MakeSym, ExpandIndptr
from dgl.partition import metis_partition_assignment_capi
import torch, os, time, dgl, argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from utils import *

def load_metis_graph(config:Config, node_mode: str, edge_mode: str):
    print("Start graph topology loading", flush=True)
    timer = Timer()
    in_dir = os.path.join(config.data_dir, config.graph_name)
    # weight_dir = os.path.join(config.data_dir, "weight","15-15-15",config.graph_name)
    wsloop = False
    is32 = False
    is_sym = config.graph_name in ["orkut", "friendster"]
    
    load_edge_weight = True if edge_mode == "freq" else False
    indptr, indices, edge_weight = load_graph(in_dir, is32=is32, wsloop=wsloop, is_sym=is_sym, load_edge_weight=load_edge_weight)
    
    v_num = indptr.shape[0] - 1
    e_num = indices.shape[0]
    print(f"load graph topology in {timer.duration()} secs", flush=True)
    timer.reset()
    degree = indptr[1:] - indptr[:-1]
    avg_deg = torch.sum(degree) // degree.shape[0]
    edge_pruned = False
    if load_edge_weight:
        print("prunning edges")
        flag = edge_weight > 0
        remain_ratio = flag.sum() / flag.shape[0] * 100
        e_num = flag.sum()
        indices = indices[flag].clone()
        edge_weight = edge_weight[flag].type(torch.int64).clone()
        indptr = CompactCSR(indptr, flag)
        edge_pruned=True
        print(f"remove {round(100 - remain_ratio.item())}% edges in {timer.duration()} secs", flush=True)

    timer.reset()
    if is_sym == False:
        # remove self edges
        src = ExpandIndptr(indptr)
        flag = src != indices
        indices = indices[flag].clone()
        if load_edge_weight:
            edge_weight = edge_weight[flag].type(torch.int64).clone()
        indptr = CompactCSR(indptr, flag)
        print(f"remove self edges in {timer.duration()} secs", flush=True)
        indices = indices.to(torch.int)
        edge_weight =edge_weight.to(torch.int)
        indptr, indices, edge_weight = MakeSym(indptr, indices, edge_weight)
        print(f"convert graph org_enum={e_num} to sym_enum={indices.shape[0]} in {timer.duration()} secs", flush=True)
        
    if is_sym == True and load_edge_weight and edge_pruned:
        indptr, indices, edge_weight = MakeSym(indptr, indices, edge_weight)
        print(f"convert graph org_enum={e_num} to sym_enum={indices.shape[0]} in {timer.duration()} secs", flush=True)

    
    node_weight = None
    if node_mode == "uniform":
        node_weight = torch.ones((v_num,), dtype=torch.int64)
    elif node_mode == "degree":
        node_weight = degree
    elif node_mode in ["src", "dst", "input"]:
        node_weight = load_numpy(os.path.join(in_dir, "node_weight.npy")).type(torch.int64)
        node_weight = node_weight * avg_deg + 1
    return node_weight, indptr, indices, edge_weight

def partition(config: Config, node_mode:str, edge_mode:str, bal: str):
    assert node_mode in ["uniform", "degree", "src", "dst", "input"]    
    assert edge_mode in ["uniform", "freq"]
    assert bal in ["bal", "xbal"]
    
    timer = Timer()
    node_weight, indptr, indices, edge_weight = load_metis_graph(config, node_mode, edge_mode)
    indices =indices.to(torch.int64)
    edge_weight = edge_weight.to(torch.int64)
    graph = dgl.graph(("csr", (indptr, indices, edge_weight)))

    print(f"load graph in {timer.duration()} secs", flush=True)
    if bal == "bal":
        v_num = indptr.shape[0] - 1
        vwgts = [node_weight]
        in_dir = os.path.join(config.data_dir, config.graph_name)
        train, valid, test = load_idx_split(in_dir)
        for target_idx in [train, valid, test]:
            wgt = torch.zeros((v_num,), dtype=torch.int64)
            wgt[target_idx] = node_weight[target_idx]
            vwgts.append(wgt)
        node_weight = torch.concat(vwgts).clone()
    assert(node_weight.is_contiguous())
    if edge_mode == "uniform":
        edge_weight = torch.empty((0,), dtype=torch.int64)
    
    timer.reset()
    assignment: torch.Tensor = metis_partition_assignment_capi(sym_g=graph, k=config.num_partition, vwgt=node_weight, mode="k-way", objtype="cut", use_edge_weight=edge_mode != "uniform")
    file_name = f"{config.graph_name}_w{config.num_partition}_{config.partition_type}.npy"
    out_dir = os.path.join(config.data_dir, "partition_map", config.graph_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"saving file to {out_dir}/{file_name}", flush=True)
    save_numpy(assignment.type(torch.int8), os.path.join(out_dir, file_name))
    # torch.save(assignment, os.path.join(out_dir, file_name))

# def partition(config: Config, node_mode:str, edge_mode:str, bal: str):
#     assert node_mode in ["uniform", "degree", "src", "dst", "input"]    
#     assert edge_mode in ["uniform", "freq"]
#     assert bal in ["bal", "xbal"]
    
#     timer = Timer()
#     node_weight, graph = load_metis_graph(config, node_mode, edge_mode)
#     print(f"load graph in {timer.duration()} secs", flush=True)
#     if bal == "bal":
#         v_num = graph.num_nodes()
#         vwgts = [node_weight]
#         in_dir = os.path.join(config.data_dir, config.graph_name)
#         train, valid, test = load_idx_split(in_dir)
#         for target_idx in [train, valid, test]:
#             wgt = torch.zeros((v_num,), dtype=torch.int64)
#             wgt[target_idx] = node_weight[target_idx]
#             vwgts.append(wgt)
#         node_weight = torch.concat(vwgts).clone()
#     assert(node_weight.is_contiguous())
#     assignment: torch.Tensor = metis_partition_assignment_capi(sym_g=graph, k=config.num_partition, vwgt=node_weight, mode="k-way", objtype="cut", use_edge_weight=edge_mode != "uniform")
#     file_name = f"{config.graph_name}_w{config.num_partition}_{config.partition_type}.npy"
#     out_dir = os.path.join(config.data_dir, "partition_map", config.graph_name)
#     os.makedirs(out_dir, exist_ok=True)
#     print(f"saving file to {out_dir}/{file_name}", flush=True)
#     save_numpy(assignment.type(torch.int8), os.path.join(out_dir, file_name))
#     # torch.save(assignment, os.path.join(out_dir, file_name))
    
def get_args():
    parser = argparse.ArgumentParser(description='local run script')
    parser.add_argument('--graph_name', default="products", type=str, help="Input graph name", choices=["products", "papers100M", "orkut", "friendster","arxiv"])
    parser.add_argument('--data_dir', required=True, type=str, help="Input graph directory")
    parser.add_argument('--num_partition', default=4, type=int, help='Number of partitions')
    parser.add_argument('--node_weight', default="uniform", type=str, help="Node weight configuraion", choices=["uniform", "degree", "src", "dst", "input"] )
    parser.add_argument('--edge_weight', default="uniform", type=str, help="Edge weight configuraion", choices=["uniform", "freq"])
    parser.add_argument('--bal', default="bal", type=str, help='Balance target idx on each partition or not', choices=["bal", "xbal"])
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    
    print(f"{args=}")
    graph_name = str(args.graph_name)
    data_dir = args.data_dir
    bal = args.bal
    node_weight = args.node_weight
    edge_weight = args.edge_weight
    num_partition = args.num_partition
    if "random" in node_weight:
        print("skip random")
        exit(0)
    batch_size = 0
    fanouts = [0]
    num_epoch = 0
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "logs/exp.csv")
    config = Config(graph_name=graph_name,
                       world_size=1,
                       num_epoch=num_epoch,
                       fanouts=fanouts,
                       batch_size=batch_size,
                       num_partition=num_partition,
                       system="dgl-sample",
                       model="none",
                       cache_size="0GB",
                       hid_size=256,
                       log_path=log_path,
                       data_dir=data_dir,
                       partition_type=get_partition_type(node_weight=node_weight, edge_weight=edge_weight, bal=bal))
           
    partition(config, node_weight, edge_weight, bal)
