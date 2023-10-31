
import os
import torch
import re
import tqdm
import dgl
from ogb.nodeproppred import  DglNodePropPredDataset

def preprocess(out_dir, graph_name, ogbn_out_dur):
    OUT_DIR = os.path.join(out_dir, graph_name)
    edge_list = []
    if os.path.exists(f"{OUT_DIR}/edge_list.pt"):
        edge_list = torch.load(f"{OUT_DIR}/edge_list.pt")
    else:
        with open(f"{OUT_DIR}/{graph_name}.ungraph.txt",'r') as f:
            for line in tqdm.tqdm(f):
                if 'Nodes' in line:
                    n_nodes = int(re.findall(r"Nodes: (\d+)",line)[0])
                    n_edges = int(re.findall(r"Edges: (\d+)", line)[0])
                if line.startswith('#'):
                    continue
                nd1, nd2 = line.split()
                nd1, nd2 = (int(nd1),int(nd2))
                edge_list.append([nd1,nd2])
        edge_list = torch.tensor(edge_list)
        torch.save( edge_list ,f"{OUT_DIR}/edge_list.pt")
    n_nodes = torch.max(edge_list) + 1
    print(n_nodes)
    graph = dgl.DGLGraph((edge_list[:,0], edge_list[:,1]), num_nodes = n_nodes)
    idtype_str = "64"
    assert(graph.idtype == torch.int64)
    indptr, indices, _ = graph.adj_tensors("csc")
    ws_self_loop = False
    torch.save(indptr, OUT_DIR + f'/indptr_{idtype_str}_{ws_self_loop}.pt')
    torch.save(indices, OUT_DIR + f'/indices_{idtype_str}_{ws_self_loop}.pt')
    # torch.save(edges, SAVE_PATH + '/edges')
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    ws_self_loop = True
    indptr, indices, _ = graph.adj_tensors("csc")
    torch.save(indptr, OUT_DIR + f'/indptr_{idtype_str}_{ws_self_loop}.pt')
    torch.save(indices, OUT_DIR + f'/indices_{idtype_str}_{ws_self_loop}.pt')
    graph = graph.remove_self_loop()

    if graph_name == "com-orkut":
        dataset = DglNodePropPredDataset('ogbn-products', root=ogbn_out_dur)
    if graph_name == "com-friendster":
        dataset = DglNodePropPredDataset('ogbn-papers100M', root = ogbn_out_dur)

    split = dataset.get_idx_split()
    ref_num_nodes = (dataset[0][0].num_nodes())
    start = 0
    offsets = {}
    for k in split.keys():
        offsets[k] = start, start + split[k].shape[0]/ref_num_nodes
        start = start + split[k].shape[0]/ref_num_nodes
    print(offsets)
    random_idx = torch.rand(n_nodes)
    ntype = torch.zeros(n_nodes)
    count = 1
    for k in offsets.keys():
        selected_nodes = torch.where((random_idx >= offsets[k][0]) & (random_idx < offsets[k][1]))[0]
        torch.save(selected_nodes,f"{OUT_DIR}/{k}_idx_{idtype_str}.pt")
        ntype[selected_nodes] = count
        count = count + 1
        print(selected_nodes.shape)
    torch.save(ntype, f'{OUT_DIR}/ntype.pt')
    print(ntype.shape)
    print(graph.num_nodes())
    num_nodes = indptr.shape[0] -1
    feat = torch.rand(num_nodes, 128, dtype = torch.float32)
    torch.save(feat, f"{OUT_DIR}/feat.pt")
    labels = torch.randint(0, 172, (num_nodes,), dtype = torch.int64)
    torch.save(labels, f"{OUT_DIR}/labels.pt")
    print("All done")




if __name__ == "__main__":
    preprocess()