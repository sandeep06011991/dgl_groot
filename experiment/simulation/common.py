from utils import *
from torch.utils.data import DataLoader
from .measurements import KProfiler
# Approximate unidirectional bandwidth in bytes/sec
NVLINK_BW       = 300e9   # NVLink 3.0  ~300 GB/s
PCIE_BW         =  25e9   # PCIe 4.0 x16, measured ~25-26 GB/s host->GPU UVA gather
BYTES_PER_FLOAT = 4
BARRIER_COST = 1e-6 # 1us (one all-reduce/barrier sync)

# Effective GPU<->GPU all-to-all (feature shuffle) bandwidth, bytes/sec. This is
# DISTINCT from PCIE_BW/NVLINK_BW above: the host->GPU feature gather (PCIE_BW)
# runs near line rate, but the partition-aware all-to-all shuffle pays NCCL
# launch/latency + (under PCIe emulation) host-staging overhead, so its EFFECTIVE
# bandwidth is far below line rate. These constants are CALIBRATED BY SUBTRACTION
# against measured split runs: the shuffle cost per cell is isolated as
# (pcie_fwdbwd - nvlink_fwdbwd) in sweep_all.csv, holding compute fixed, and the
# effective BW is fit so the simulator's volume/BW comm term reproduces it.
# (A direct 4-GPU all-to-all microbenchmark would refine these; not available on
# the 1-GPU calibration pod, so they are empirical effective values.)
ALLTOALL_PCIE_BW   = 1.95e9  # fit to products split shuffle under PCIe emulation
ALLTOALL_NVLINK_BW = 18e9    # fit to products split shuffle under NVLink

# Compute runs across num_devices GPUs in parallel, but the simulator times it on
# the FULL global batch on ONE GPU. The achieved per-GPU wall-clock is NOT
# full/num_devices: partition boundaries replicate halo nodes and small batches
# underutilize the GPU, so the measured parallel-efficiency factor is well below
# num_devices. Fit empirically so split-layer compute (full-batch timing /
# factor) matches the measured per-GPU fwd+bwd (nvlink) on products sage: the
# ratio is ~1.6-1.8 on the large (compute-heavy) cells that drive the hybrid
# decision, ~1 on the small ones. A single value is a compromise; residuals are
# reported. DP layers instead divide by num_devices (genuinely parallel
# redundant work: each GPU runs its own equal-size microbatch).
COMPUTE_PARALLEL_FACTOR = 1.7

# subgraph -> bipartite graph
# model layer is a torch representation 
def measure_compute_cost_layer(subgraph, model_layer, seeds, input_requires_grad=True):
    # 1. re_index the subgraph [0, N], where N is unique nodes.
    block = dgl.to_block(subgraph, seeds)
    # 2. Create a block with dummy input features sized to the layer's input dim.
    device = next(model_layer.parameters()).device
    num_src = block.num_src_nodes()
    num_dest = block.num_dst_nodes()
    assert(num_dest == seeds.shape[0])
    in_feats = model_layer._in_src_feats
    # The true input layer reads raw features (a leaf, no input gradient); inner
    # layers propagate a gradient to their input. Model that so the backward cost
    # matches the fused step rather than over-counting an input grad everywhere.
    feat = torch.empty((num_src, in_feats), device=device, requires_grad=input_requires_grad)
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


def sample_neighbors_timed(graph, seeds, fanout):
    """dgl.sampling.sample_neighbors, timed on the GPU. Returns (subgraph, seconds).
    The simulator omitted sampling cost entirely, but measured sampling is 1-4s
    (large at depth 4); timing the real sampler gives it, in the same
    'measure, don't fit' spirit as measure_compute_cost_layer."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    subgraph = dgl.sampling.sample_neighbors(graph, seeds, fanout)
    end.record()
    torch.cuda.synchronize()
    return subgraph, start.elapsed_time(end) / 1000


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
    # Uses the EFFECTIVE all-to-all bandwidths (ALLTOALL_*), not the line-rate
    # PCIE_BW/NVLINK_BW, because the shuffle pays NCCL/host-staging overhead that
    # those line rates do not capture (see the constant definitions above).
    # case_1. pci_e
    if num_nvlink == 0:
        total_pcie = 0
        for src in range(n_gpus):
            for dest in range(n_gpus):
                # total_bytes * two way
                if src != dest:
                    total_pcie += (comm_matrix[src][dest] * feat_size * 4 ) / ALLTOALL_PCIE_BW

        return total_pcie + BARRIER_COST
    if num_nvlink == 2:
        total_nvlink = 0
        total_pcie = 0
        for src in range(n_gpus):
            for dest in range(n_gpus):
                if src!= dest and src%num_nvlink == dest%num_nvlink:
                    total_nvlink = max(total_nvlink, comm_matrix[src][dest] * feat_size * 4  / ALLTOALL_NVLINK_BW)
                if src!= dest and src%num_nvlink != dest%num_nvlink:
                    total_pcie = total_pcie + comm_matrix[src][dest] * feat_size * 4 * 2 / ALLTOALL_PCIE_BW
        return total_nvlink + total_pcie + BARRIER_COST

    if num_nvlink == 4:
        total_nvlink = 0
        for src in range(n_gpus):
            for dest in range(n_gpus):
                if src!= dest:
                    total_nvlink = max(total_nvlink, comm_matrix[src][dest] * feat_size * 4 * 2 / ALLTOALL_NVLINK_BW)
        return total_nvlink + BARRIER_COST
