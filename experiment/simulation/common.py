from utils import *
from torch.utils.data import DataLoader
from .measurements import KProfiler
# Approximate unidirectional bandwidth in bytes/sec
NVLINK_BW       = 300e9   # NVLink 3.0  ~300 GB/s
PCIE_BW         =  16e9   # PCIe 4.0 x16 ~16 GB/s
BYTES_PER_FLOAT = 4         
BARRIER_COST = 1e-6 # 1ms 

# subgraph -> bipartite graph
# model layer is a torch representation 
def measure_compute_cost_layer(subgraph, model_layer, seeds):
    # 1. re_index the subgraph [0, N], where N is unique nodes.
    block = dgl.to_block(subgraph, seeds)
    # 2. Create a block with dummy input features sized to the layer's input dim.
    device = next(model_layer.parameters()).device
    num_src = block.num_src_nodes()
    num_dest = block.num_dst_nodes()
    assert(num_dest == seeds.shape[0])
    in_feats = model_layer._in_src_feats
    feat = torch.empty((num_src, in_feats), device=device)
    block = block.to(device)

    # 3. Run the layer and use events to return the cost (ms).
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    model_layer(block, feat)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)/1000