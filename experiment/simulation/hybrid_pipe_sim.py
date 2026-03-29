from utils import *
from torch.utils.data import DataLoader

# Approximate unidirectional bandwidth in bytes/sec
NVLINK_BW       = 300e9   # NVLink 3.0  ~300 GB/s
PCIE_BW         =  16e9   # PCIe 4.0 x16 ~16 GB/s
BYTES_PER_FLOAT = 4         
NVLINK_BARRIER_COST = 1e-6 # 1ms 

# Estimate the commmunication and computation cost of pipeline parellelism.
def simulate_optimal_h( graph, train_idx, partition_map, device, k,
                             micro_batch_size, fanouts,  model_layers, num_devices) -> int:
    pipeline_depth = 2
    chunks = train_idx.chunk(pipeline_depth)
    dataloaders = [iter(DataLoader(chunk, batch_size=micro_batch_size//pipeline_depth, shuffle=True)) for chunk in chunks]

    total_batches = train_idx.shape[0]//(micro_batch_size)
    total_computation_cost = 0
    total_communication_cost = 0
    for batch_num in range(total_batches):
        frontier = []
        for l_id, model_layer  in enumerate(model_layers):
            for split_unit in range(pipeline_depth):
                if l_id == 0:
                    frontier.append(next(dataloaders))
                for pipeline_layer in range(pipeline_depth):
                communication_cost = ..
                computation_cost = .. 
    pass
    