import torch 
import os 

graph_name = "ogbn-products"
DATA_DIR = "/data/sandeep"

#MACHINE_NAME = os.environ.get('SLURMD_NODENAME')
#JOB_ID = os.environ.get('SLURM_JOB_ID')
#DATA_DIR = f"/scratch/{MACHINE_NAME}/{JOB_ID}"

import dgl 
from ogb.nodeproppred import DglNodePropPredDataset
import dgl

num_partitions = 4
path = f"{DATA_DIR}/{graph_name}".replace("-", "_")
dataset = DglNodePropPredDataset(graph_name, root=DATA_DIR)
print("read success full", path)
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
balance_edges = False
partitions = dgl.metis_partition(graph, num_partitions, balance_ntypes=ntype, balance_edges = False)
p_map = torch.zeros(graph.num_nodes(), dtype=torch_type)
print(partitions)
for p_id in partitions.keys():
    nodes = partitions[p_id].ndata['_ID']
    p_map[nodes] = p_id
    print(f"In partiiton {p_id}: nodes{nodes.shape}")
p_map = p_map.to(torch.int32)
print(f"Saving to {path}/partition_map")
if balance_edges:
    torch.save(p_map, f"{path}/partition_map_ebal")
else:
    torch.save(p_map, f"{path}/partition_map")
