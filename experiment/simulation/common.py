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


def get_communication_cost(comm_matrix,  feat_size, n_gpus, num_nvlink):
    # case_1. pci_e
    if num_nvlink == 0: 
        total_pcie = 0
        for src in range(n_gpus):
            for dest in range(n_gpus):
                # total_bytes * two way 
                if src != dest:
                    total_pcie += (comm_matrix[src][dest] * feat_size * 4 ) / PCIE_BW

        return total_pcie + BARRIER_COST
    if num_nvlink == 2:
        total_nvlink = 0
        total_pcie = 0
        for src in range(n_gpus):
            for dest in range(n_gpus):
                if src!= dest and src%num_nvlink == dest%num_nvlink:
                    total_nvlink = max(total_nvlink, comm_matrix[src][dest] * feat_size * 4  / NVLINK_BW)
                if src!= dest and src%num_nvlink != dest%num_nvlink:
                    total_pcie = total_pcie + comm_matrix[src][dest] * feat_size * 4 * 2 / PCIE_BW    
        return total_nvlink + total_pcie + BARRIER_COST 
    
    if num_nvlink == 4:
        total_nvlink = 0
        for src in range(n_gpus):
            for dest in range(n_gpus):
                if src!= dest:
                    total_nvlink = max(total_nvlink, comm_matrix[src][dest] * feat_size * 4 * 2 / NVLINK_BW)
        return total_nvlink + BARRIER_COST 