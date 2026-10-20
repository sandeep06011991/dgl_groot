from utils import *
from torch.utils.data import DataLoader
from .measurements import KProfiler
# Approximate unidirectional bandwidth in bytes/sec
NVLINK_BW       = 300e9   # NVLink 3.0  ~300 GB/s
PCIE_BW         =  25e9   # PCIe 4.0 x16, measured ~25-26 GB/s host->GPU UVA gather
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
    feat = torch.empty((num_src, in_feats), device=device, requires_grad=True)
    block = block.to(device)

    # 3. Time a full training step for this layer: forward + backward, so the
    #    estimate matches the measured train time (fwd+bwd), not just forward.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = model_layer(block, feat)
    out.sum().backward()
    end.record()
    torch.cuda.synchronize()
    model_layer.zero_grad(set_to_none=True)
    return start.elapsed_time(end)/1000


def cache_bytes(cache_size: str) -> int:
    """Parse a cache-size string like '10G', '10GB', '512M', '0GB' into bytes."""
    s = str(cache_size).upper().strip()
    if s.endswith("B"):
        s = s[:-1]
    mult = 1
    if s.endswith("G"):
        mult, s = 1024 ** 3, s[:-1]
    elif s.endswith("M"):
        mult, s = 1024 ** 2, s[:-1]
    elif s.endswith("K"):
        mult, s = 1024, s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return 0


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
