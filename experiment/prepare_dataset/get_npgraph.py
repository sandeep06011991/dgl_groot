import torch
import os
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dev.coo2csr import LoadSNAP
import os, argparse

def save_numpy(out:torch.Tensor,  outpath: str):
    assert(outpath.endswith(".npy"))
    np.save(outpath, out.numpy())
    
def prep_snap_graph(in_dir, out_dir, filename, to_sym):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    in_path = f"{in_dir}/{filename}"
    indptr, indices = LoadSNAP(in_path, to_sym)
    if to_sym:
        save_numpy(indptr, f"{out_dir}/indptr_sym.npy")
        save_numpy(indices, f"{out_dir}/indices_sym.npy")
    else:
        save_numpy(indptr, f"{out_dir}/indptr_xsym.npy")
        save_numpy(indices, f"{out_dir}/indices_xsym.npy")
    
    v_num = indptr.shape[0] - 1
    # randomly generate 10% nodes as train_idx, 5% as valid_idx and test_idx and save them to out_dir
    generate_idx_split(v_num, 0.1, out_dir)
        
def prep_ogbn_graph(in_dir, out_dir, graph_name):
    assert(graph_name in ["ogbn-products", "ogbn-papers100M","ogbn-arxiv"])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dataset = DglNodePropPredDataset(graph_name, in_dir)
    graph = dataset[0][0]
    
    feat: torch.Tensor = graph.dstdata.pop("feat")
    save_numpy(feat, os.path.join(out_dir, "feat.npy"))
    del feat
        
    indptr, indices, _ = graph.adj_tensors("csc")
    save_numpy(indptr, f"{out_dir}/indptr_xsym.npy")
    save_numpy(indices, f"{out_dir}/indices_xsym.npy")
    
    # generate idx split
    idx_split = dataset.get_idx_split()
    train_idx = idx_split["train"]
    valid_idx = idx_split["valid"]
    test_idx = idx_split["test"]
    
    save_numpy(train_idx, os.path.join(out_dir, f"train_idx.npy"))
    save_numpy(valid_idx, os.path.join(out_dir, f"valid_idx.npy"))
    save_numpy(test_idx, os.path.join(out_dir, f"test_idx.npy"))
    
    node_labels: torch.Tensor = dataset[0][1]
    node_labels = node_labels.flatten().clone()
    torch.nan_to_num_(node_labels, nan=0.0)
    node_labels: torch.Tensor = node_labels.type(torch.int64)
    save_numpy(node_labels, os.path.join(out_dir, "label.npy"))

def gen_rand_feat(v_num, feat_dim):
    return torch.empty((v_num, feat_dim))

def gen_rand_label(v_num, num_classes):
    return torch.randint(low=0, high=num_classes, size=(v_num,))

def generate_idx_split(v_num, ratio, out_dir):
    # generate idx split
    num_train = int(v_num * ratio)
    num_val   = int(v_num * 0.5 * ratio)
    num_test  = int(v_num * 0.5 * ratio)
    rand_idx  = torch.randperm(v_num)
    train_idx = rand_idx[0 : num_train].clone()
    test_idx  = rand_idx[num_train : num_train + num_test].clone()
    val_idx   = rand_idx[num_train + num_test : num_train + num_test + num_val].clone()
    
    os.makedirs(f"{out_dir}", exist_ok=True)
    save_numpy(train_idx, f"{out_dir}/train_idx.npy")
    save_numpy(test_idx,  f"{out_dir}/test_idx.npy")
    save_numpy(val_idx,   f"{out_dir}/valid_idx.npy")


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='local run script')
    parser.add_argument('--data_dir', required=True, type=str, help="Input graph directory")
    parser.add_argument('--graph_name', required=True, type=str, help="Input graph name")
    args = parser.parse_args()
    data_dir = args.data_dir
    graph_name = args.graph_name
    
    print(f"{args=}")
    
    if graph_name in ["orkut", "friendster"]:
        filedir = os.path.join(data_dir, graph_name)
        prep_snap_graph(in_dir=filedir, out_dir=filedir, filename=f"{graph_name}.txt", to_sym=True)

    if graph_name in ["products", "papers100M","arxiv"]:
        out_dir = os.path.join(data_dir, graph_name)
        prep_ogbn_graph(in_dir=data_dir, out_dir=out_dir, graph_name=f"ogbn-{graph_name}")
    print("finished processing ", graph_name)