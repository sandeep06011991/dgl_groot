
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
    global_degree = indptr[1:] - indptr[:-1]
    ws_self_loop = False
    torch.save(indptr, OUT_DIR + f'/indptr_{idtype_str}.pt')
    torch.save(indices, OUT_DIR + f'/indices_{idtype_str}.pt')
    # torch.save(edges, SAVE_PATH + '/edges')
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    ws_self_loop = True
    indptr, indices, _ = graph.adj_tensors("csc")
    ld = torch.where(indptr[1:] - indptr[:-1] == 1)[0]
    assert(torch.all(indices[indptr[ld]] == ld))
    torch.save(indptr, OUT_DIR + f'/indptr_{idtype_str}_wsloop.pt')
    torch.save(indices, OUT_DIR + f'/indices_{idtype_str}_wsloop.pt')

    if graph_name == "com-orkut":
        ref_graph = "ogbn-products"
    if graph_name == "com-friendster":
        ref_graph = "ogbn-papers100M"
    ref_train_nodes = torch.load(f'{ogbn_out_dur}/{ref_graph}/train_idx_64.pt')
    high_deg_train_nodes = torch.where(global_degree > 7)[0]
    train_nodes = high_deg_train_nodes[torch.randint(0, high_deg_train_nodes.shape[0],(ref_train_nodes.shape))]
    torch.save(train_nodes, f"{OUT_DIR}/train_idx_{idtype_str}.pt")
    print(graph.num_nodes())
    num_nodes = indptr.shape[0] -1
    if graph_name == "com-orkut":
        feat = torch.rand(num_nodes, 1280, dtype = torch.float32)
    if graph_name == "com-friendster":
        feat = torch.rand(num_nodes, 128, dtype = torch.float32)
    torch.save(feat, f"{OUT_DIR}/feat.pt")
    labels = torch.randint(0, 172, (num_nodes,), dtype = torch.int64)
    torch.save(labels, f"{OUT_DIR}/labels.pt")
    print("All done")




if __name__ == "__main__":
    out_dir = "/data/sandeep/groot_data/snap"
    ogb_out  = "/data/sandeep/groot_data/ogbn-processed"
    graph = "com-friendster"
    preprocess(out_dir, graph, ogb_out)
    print(graph, "is done")