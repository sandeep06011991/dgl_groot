from utils import *
from torch.utils.data import DataLoader
from .common import *

# Approximate unidirectional bandwidth in bytes/sec
NVLINK_BW       = 300e9   # NVLink 3.0  ~300 GB/s
PCIE_BW         =  16e9   # PCIe 4.0 x16 ~16 GB/s
BYTES_PER_FLOAT = 4         
NVLINK_BARRIER_COST = 1e-6 # 1ms 

# Estimate the commmunication and computation cost of pipeline parellelism.
def get_costs_for_pipeline_parallelism( graph, train_idx, partition_map, device, k,
                             micro_batch_size, fanouts,  model_layers, input_feat_size , num_devices) -> int:
    pipeline_depth = 2
    chunks = train_idx.chunk(pipeline_depth)
    dataloaders = [iter(DataLoader(chunk, batch_size=micro_batch_size * num_devices //pipeline_depth, shuffle=True)) for chunk in chunks]
    num_nvlinks = 4
    total_batches = train_idx.shape[0]//(micro_batch_size * num_devices)
    for batch_num in range(total_batches):
        frontier = []
        pipe_layer_wise_communication_costs = []
        pipe_layer_wise_computation_costs = []
        pipe_dataloading_costs = []
        for layer_no in range(len(fanouts)):
            for split_unit in range(pipeline_depth):
                if layer_no== 0:
                    frontier.append(next(dataloaders))
                    pipe_layer_wise_communication_costs.append([])
                    pipe_layer_wise_computation_costs.append([])
                layer_nds = frontier[split_unit]
                subgraph = dgl.sampling.sample_neighbors(graph, layer_nds, fanouts[layer_no])
                in_src_feats = model_layers[-layer_no]._in_src_feats
                pipe_layer_wise_computation_costs[split_unit].append(measure_compute_cost_layer(subgraph, model_layers[-layer_no], layer_nds ))
                next_minibatch = torch.cat([subgraph.edges()[0], subgraph.edges()[1]]).unique()
                frontier[split_unit] = next_minibatch
                if layer_no == len(fanouts) -1:
                    pipe_dataloading_costs.append( frontier.shape[0] * input_feat_size * 4 / PCIE_BW)
                src, dest = subgraph.edges()
                # compute_cost of split parallelism.
                comm_volume = []
                for dst_device in range(num_devices):
                    row = []
                    filter_edges = src[torch.where(partition_map[dest] == dst_device)]
                    for src_device in range(num_devices):
                        row.append(torch.sum(partition_map[filter_edges.unique()] == src_device).item())
                    comm_volume.append(row)
                
                pipe_layer_wise_communication_costs.append(get_communication_cost(comm_volume, in_src_feats, num_devices, num_nvlinks))
        total_costs = pipe_layer_wise_computation_costs[0][0]
        for layer_id in range(len(fanouts)) - 1:
            total_costs += max(pipe_layer_wise_communication_costs[0][layer_id] , pipe_layer_wise_computation_costs[1][layer_id])
            total_costs += max(pipe_layer_wise_computation_costs[0][layer_id + 1], pipe_layer_wise_communication_costs[1][layer_id])
        total_costs += max(pipe_layer_wise_communication_costs[0][-1], pipe_layer_wise_computation_costs[1] [-1])
        total_costs += max(pipe_dataloading_costs[0], pipe_layer_wise_communication_costs[1][-1])
        total_costs += pipe_dataloading_costs[1]
    return total_costs


def get_costs_for_split_parallelism(graph, train_idx, partition_map,  device, k,
                             micro_batch_size, fanouts,  model_layers, input_feat_size , num_devices):
    Dataloaders = (DataLoader(train_idx, batch_size=micro_batch_size * num_devices, shuffle=True))
    total_batches = train_idx.shape[0]//(micro_batch_size * num_devices)
    compute_cost = 0
    communication_volume = 0
    dataloading_cost = 0
    num_epochs = 2
    num_nvlinks = 4
    for _ in range(num_epochs):
        dataloaders = iter(Dataloaders)
        for batch_num in range(total_batches):
            frontier = next(dataloaders)
            for layer_no in range(len(fanouts)):
                subgraph = dgl.sampling.sample_neighbors(graph, frontier, fanouts[layer_no])
                compute_cost += measure_compute_cost_layer(subgraph, model_layers[-layer_no],frontier)
                next_minibatch = torch.cat([subgraph.edges()[0], subgraph.edges()[1]]).unique()
                in_src_feats = model_layers[-layer_no]._in_src_feats
                frontier = next_minibatch
                if layer_no == len(fanouts) -1:
                        dataloading_cost +=( frontier.shape[0] * input_feat_size * 4 / PCIE_BW)
                src, dest = subgraph.edges()
                # compute_cost of split parallelism.
                comm_volume = []
                for dst_device in range(num_devices):
                    row = []
                    filter_edges = src[torch.where(partition_map[dest] == dst_device)]
                    for src_device in range(num_devices):
                        row.append(torch.sum(partition_map[filter_edges.unique()] == src_device).item())
                    comm_volume.append(row)    
                communication_costs += get_communication_cost(comm_volume, in_src_feats, num_devices, num_nvlinks)
    return compute_cost + communication_costs + dataloading_cost


def run_pipeline_parallelism_simulation():
    pass 

