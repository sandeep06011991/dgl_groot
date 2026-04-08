
from utils import *
from .common import *
from torch.utils.data import DataLoader
from .measurements import KProfiler
# Approximate unidirectional bandwidth in bytes/sec
NVLINK_BW       = 300e9   # NVLink 3.0  ~300 GB/s
PCIE_BW         =  16e9   # PCIe 4.0 x16 ~16 GB/s
BYTES_PER_FLOAT = 4         
BARRIER_COST = 1e-6 # 1ms 



    
    



# Returns the cost of performing hybrid split parallelism. 
# k is the split parallelism layer. 
# for 3 layer network 10-10-10
# k = 0 = no data parallellism , full split parallelism throughout from layer 0
# k = 3  data parallelism 
def simulate_optimal_k( graph, train_idx, partition_map, device, k,
                             micro_batch_size, fanouts,  model_layers,
                                 num_devices, num_nvlinks, input_feat_size, config:Config) -> int:
    computation_cost = 0
    communication_cost = 0
    dataloading_cost = 0
    total_batches = train_idx.shape[0]//(num_devices * micro_batch_size)
    print("total num batches", total_batches)
    # try:
    assert(not (graph.in_degrees() ==0).any())
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
                            subgraph = dgl.sampling.sample_neighbors(graph, layer_nds, fanout)
                            computation_cost += measure_compute_cost_layer(subgraph, model_layers[-layer_no], layer_nds )
                            torch.cuda.empty_cache()
                            frontier = torch.cat([subgraph.edges()[0], subgraph.edges()[1]]).unique()
                            new_microbatches.append(frontier)
                            if layer_no == lastlayer:
                                dataloading_cost += frontier.shape[0] * input_feat_size * 4 / PCIE_BW
                        current_microbatches = new_microbatches
                        
                    if layer_no == k:
                        minibatch = torch.cat(current_microbatches)

                    if layer_no >= k:
                        layer_nds = minibatch
                        subgraph = dgl.sampling.sample_neighbors(graph, layer_nds, fanout)
                        in_src_feats = model_layers[-layer_no]._in_src_feats
                        computation_cost += measure_compute_cost_layer(subgraph, model_layers[-layer_no], layer_nds )
                        frontier = torch.cat([subgraph.edges()[0], subgraph.edges()[1]]).unique()
                        minibatch = (frontier)
                        if layer_no == lastlayer:
                            dataloading_cost += frontier.shape[0] * input_feat_size * 4 / PCIE_BW
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
    profiler = KProfiler(k = k, num_nvlinks = num_nvlinks, communication_cost = communication_cost, computation_cost = computation_cost, dataloading_cost = dataloading_cost)
    write_to_csv(config.log_path, [config], [profiler])
