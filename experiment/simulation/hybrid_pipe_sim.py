from utils import *
from torch.utils.data import DataLoader
from .common import *
from .measurements import KProfiler

# Bandwidth constants come from common.py (single source of truth) via the
# `from .common import *` above — do NOT redefine them here. This file used to
# shadow PCIE_BW=16e9, which disagreed with common.py's 25e9 for the same
# host->GPU feature-gather path; that inconsistency is now removed.


def _layer_costs(graph, partition_map, seeds, fanout, conv_layer, input_feat_size,
                 num_devices, num_nvlinks, is_input_layer):
    """Compute (sampling_cost, compute_cost, comm_cost, next_frontier) for one hop.

    Mirrors the accounting in hybrid_dp_sim.simulate_optimal_k: a real fwd+bwd
    on a dummy block for compute, and cross-partition feature volume / bandwidth
    for the split all-to-all shuffle, plus the timed neighbor sampling.
    """
    subgraph, sampling_cost = sample_neighbors_timed(graph, seeds, fanout)
    compute_cost = measure_compute_cost_layer(
        subgraph, conv_layer, seeds, input_requires_grad=not is_input_layer)
    torch.cuda.empty_cache()
    src, dest = subgraph.edges()
    in_src_feats = conv_layer._in_src_feats
    # Per-(dst,src)-device cross-partition edge counts -> comm volume matrix.
    comm_volume = []
    for dst_device in range(num_devices):
        row = []
        filter_edges = src[torch.where(partition_map[dest] == dst_device)]
        for src_device in range(num_devices):
            row.append(torch.sum(partition_map[filter_edges.unique()] == src_device).item())
        comm_volume.append(row)
    comm_cost = get_communication_cost(comm_volume, in_src_feats, num_devices, num_nvlinks)
    next_frontier = torch.cat([subgraph.edges()[0], subgraph.edges()[1]]).unique()
    return sampling_cost, compute_cost, comm_cost, next_frontier


# Estimate the communication and computation cost of pipeline+split parallelism.
# Two micro-batches (pipeline_depth=2) run on two CUDA streams staggered by one
# stage, so one unit's split all-to-all shuffle (comm) overlaps the other unit's
# conv (compute). Per the hybrids design (hybrids.tex), pipelining makes the
# per-layer cost max(compute, comm) rather than compute + comm.
#
# We model this on the FULL global batch's per-layer costs (same batch the DP
# simulator uses at k=0), NOT on two independently-sampled half-batches: at deep
# layers a half-batch frontier already saturates the graph, so sampling two of
# them would nearly DOUBLE compute (a micro-batching artifact that swamps the
# overlap). The full-batch per-layer max(compute, comm) is the design-doc model
# and isolates the overlap benefit cleanly.
def get_costs_for_pipeline_parallelism(graph, train_idx, partition_map, device,
                                       micro_batch_size, fanouts, model_layers,
                                       num_devices, num_nvlinks, input_feat_size,
                                       config: Config):
    L = len(fanouts)
    lasthop = L - 1
    total_cached = (cache_bytes(config.cache_size) // (input_feat_size * 4)) * num_devices
    uncached_frac = max(0.0, 1.0 - min(1.0, total_cached / graph.num_nodes()))
    print(f"[pipe] cache_size={config.cache_size} -> uncached_frac={uncached_frac:.4f}")

    # Components attributed so their sum == the overlapped total:
    #   compute_total   : full per-layer compute (always on the critical path)
    #   exposed_comm    : comm NOT hidden behind compute = sum(max(C,M)-C) >= 0
    #   dataloading     : outermost-hop feature transfer (additive, like the DP sim)
    #   sampling        : neighbor sampling (additive)
    compute_total = 0.0
    exposed_comm = 0.0
    dataloading_total = 0.0
    sampling_total = 0.0
    total_batches = train_idx.shape[0] // (num_devices * micro_batch_size)

    for _ in range(config.num_epoch):
        chunks = train_idx.chunk(num_devices)
        dataloaders = [iter(DataLoader(c, batch_size=micro_batch_size, shuffle=True)) for c in chunks]
        for _batch in range(total_batches):
            microbatches = [next(dataloaders[i]) for i in range(num_devices)]
            frontier = torch.cat(microbatches).unique()   # full global batch (as DP k=0)
            for hop, fanout in enumerate(fanouts):
                li = L - 1 - hop                          # hop -> conv layer L-1-hop
                s, c, m, nxt = _layer_costs(
                    graph, partition_map, frontier, fanout, model_layers[li],
                    input_feat_size, num_devices, num_nvlinks, is_input_layer=(li == 0))
                c = c / COMPUTE_PARALLEL_FACTOR           # per-GPU wall-clock (split layer)
                sampling_total += s
                compute_total += c
                exposed_comm += max(c, m) - c             # per-GPU comm not hidden behind compute
                frontier = nxt
                if hop == lasthop:
                    dataloading_total += nxt.shape[0] * uncached_frac * input_feat_size * 4 / PCIE_BW

    profiler = KProfiler(k=-1, num_nvlinks=num_nvlinks,
                         computation_cost=compute_total,
                         communication_cost=exposed_comm,
                         dataloading_cost=dataloading_total,
                         sampling_cost=sampling_total)
    write_to_csv(config.log_path, [config], [profiler])
    # Overlapped per-epoch total == (sampling + compute + exposed_comm + dataloading)/num_epoch.
    return sampling_total, compute_total, exposed_comm, dataloading_total


# Plain split (no overlap) baseline on the SAME cost primitives: per-layer cost
# is compute + comm (summed, not maxed). Kept for an apples-to-apples comparison
# against the pipeline total above; equivalent to hybrid_dp_sim at k=0.
def get_costs_for_split_parallelism(graph, train_idx, partition_map, device,
                                    micro_batch_size, fanouts, model_layers,
                                    num_devices, num_nvlinks, input_feat_size,
                                    config: Config):
    L = len(fanouts)
    lasthop = L - 1
    total_cached = (cache_bytes(config.cache_size) // (input_feat_size * 4)) * num_devices
    uncached_frac = max(0.0, 1.0 - min(1.0, total_cached / graph.num_nodes()))
    compute_cost = 0.0
    communication_cost = 0.0
    dataloading_cost = 0.0
    sampling_cost = 0.0
    total_batches = train_idx.shape[0] // (micro_batch_size * num_devices)
    for _ in range(config.num_epoch):
        dataloaders = iter(DataLoader(train_idx, batch_size=micro_batch_size * num_devices, shuffle=True))
        for _batch in range(total_batches):
            frontier = next(dataloaders)
            for hop, fanout in enumerate(fanouts):
                li = L - 1 - hop
                s, c, m, nxt = _layer_costs(
                    graph, partition_map, frontier, fanout, model_layers[li],
                    input_feat_size, num_devices, num_nvlinks, is_input_layer=(li == 0))
                sampling_cost += s
                compute_cost += c
                communication_cost += m
                frontier = nxt
                if hop == lasthop:
                    dataloading_cost += nxt.shape[0] * uncached_frac * input_feat_size * 4 / PCIE_BW
    return sampling_cost, compute_cost, communication_cost, dataloading_cost
