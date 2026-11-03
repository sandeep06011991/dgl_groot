
from utils import *
from .common import *
from torch.utils.data import DataLoader
from .measurements import KProfiler
# Bandwidth + barrier constants come from common.py (single source of truth) via
# `from .common import *` above — do NOT redefine them here.



    
    



# Returns the cost of performing hybrid split parallelism. 
# k is the split parallelism layer. 
# for 3 layer network 10-10-10
# k = 0 = no data parallellism , full split parallelism throughout from layer 0
# k = 3  data parallelism 
def simulate_optimal_k( graph, train_idx, partition_map, device, k,
                             micro_batch_size, fanouts,  model_layers,
                                 num_devices, num_nvlinks, input_feat_size, config:Config) -> int:
    compute_dp = 0        # DP-layer compute, summed over the num_devices microbatches
    compute_split = 0     # split-layer compute, timed on the full global batch
    communication_cost = 0
    dataloading_cost = 0
    sampling_cost = 0
    total_batches = train_idx.shape[0]//(num_devices * micro_batch_size)
    print("total num batches", total_batches)
    # Fraction of input features NOT resident in the per-GPU feature cache.
    # Each GPU caches its partition's nodes up to cache_size; collectively the
    # GPUs cache min(1, total_cached / N) of all nodes, and only the uncached
    # remainder is charged PCIe transfer. (With cache_size=0 this is 1.0 =
    # transfer everything, the previous behaviour.)
    total_cached = (cache_bytes(config.cache_size) // (input_feat_size * 4)) * num_devices
    uncached_frac = max(0.0, 1.0 - min(1.0, total_cached / graph.num_nodes()))
    print(f"cache_size={config.cache_size} -> uncached_frac={uncached_frac:.4f}")
    # try:
    if (graph.in_degrees() == 0).any():
        print("warning: graph has zero-in-degree nodes; sampling/GATConv handle this "
              "(allow_zero_in_degree), proceeding with the cost estimate")
    if True:
        for _ in range(config.num_epoch):
            chunks = train_idx.chunk(num_devices)
            dataloaders = [iter(DataLoader(chunk, batch_size=micro_batch_size, shuffle=True)) for chunk in chunks]
            for batch in range(total_batches):
                current_microbatches = []
                for i in range(num_devices):
                    current_microbatches.append(next(dataloaders[i]))
                lastlayer = len(fanouts) - 1
                for layer_no, fanout in enumerate(fanouts):
                    if layer_no < k:
                        # Do data parallelism
                        new_microbatches = []
                        for layer_nds in current_microbatches:
                            subgraph, _samp = sample_neighbors_timed(graph, layer_nds, fanout)
                            sampling_cost += _samp
                            li = len(fanouts) - 1 - layer_no   # hop j (seeds-outward) -> conv L-1-j
                            compute_dp += measure_compute_cost_layer(subgraph, model_layers[li], layer_nds, input_requires_grad=(li != 0))
                            torch.cuda.empty_cache()
                            frontier = torch.cat([subgraph.edges()[0], subgraph.edges()[1]]).unique()
                            new_microbatches.append(frontier)
                            if layer_no == lastlayer:
                                dataloading_cost += frontier.shape[0] * uncached_frac * input_feat_size * 4 / PCIE_BW
                        current_microbatches = new_microbatches
                        
                    if layer_no == k:
                        # Union the per-device DP frontiers into the global split
                        # batch. .unique() is required: for k>=1 the sampled
                        # frontiers overlap across devices, and feeding duplicate
                        # seeds to to_block/SAGEConv builds a malformed block
                        # (h_self vs h_neigh size mismatch at depth 4). At k=0 the
                        # initial per-device microbatches are disjoint, so this is
                        # a no-op there and the k=0 calibration is unchanged.
                        minibatch = torch.cat(current_microbatches).unique()

                    if layer_no >= k:
                        layer_nds = minibatch
                        subgraph, _samp = sample_neighbors_timed(graph, layer_nds, fanout)
                        sampling_cost += _samp
                        li = len(fanouts) - 1 - layer_no   # hop j (seeds-outward) -> conv L-1-j
                        in_src_feats = model_layers[li]._in_src_feats
                        compute_split += measure_compute_cost_layer(subgraph, model_layers[li], layer_nds, input_requires_grad=(li != 0))
                        frontier = torch.cat([subgraph.edges()[0], subgraph.edges()[1]]).unique()
                        minibatch = (frontier)
                        if layer_no == lastlayer:
                            dataloading_cost += frontier.shape[0] * uncached_frac * input_feat_size * 4 / PCIE_BW
                        src, dest = subgraph.edges()
                        # compute_cost of split parallelism.
                        comm_volume = []
                        for dst_device in range(num_devices):
                            row = []
                            filter_edges = src[torch.where(partition_map[dest] == dst_device)]
                            for src_device in range(num_devices):
                                row.append(torch.sum(partition_map[filter_edges.unique()] == src_device).item())
                            comm_volume.append(row)
                        
                        communication_cost += get_communication_cost(comm_volume, in_src_feats, num_devices, num_nvlinks)
    # except:
    #     computation_cost  = -1
    #     communication_cost = -1 
    #     dataloading_cost = -1
    # Per-GPU wall-clock compute: DP layers run num_devices equal microbatches in
    # parallel (-> /num_devices); split layers partition the full batch across GPUs
    # with boundary/underutilization overhead (-> /COMPUTE_PARALLEL_FACTOR).
    computation_cost = compute_dp / num_devices + compute_split / COMPUTE_PARALLEL_FACTOR
    profiler = KProfiler(k = k, num_nvlinks = num_nvlinks, communication_cost = communication_cost, computation_cost = computation_cost, dataloading_cost = dataloading_cost, sampling_cost = sampling_cost)
    write_to_csv(config.log_path, [config], [profiler])
