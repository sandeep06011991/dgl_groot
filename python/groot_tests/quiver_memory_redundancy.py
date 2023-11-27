from dgl._ffi.function import _init_api
import dgl
_init_api("dgl.groot", __name__)
_init_api("dgl.ds", __name__)
import torch.cuda
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn

from exp.util import *
from exp.quiver_sampler import *
import torch.distributed as dist
import torch.multiprocessing as mp


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_exit():
    dist.destroy_process_group()

def train_ddp(rank: int,\
              sampler: quiver.pyg.GraphSageSampler, feat: quiver.Feature,
              train_idx: torch.Tensor, \
              offset, world_size, batch_size, num_epoch):
    ddp_setup(rank, world_size)
    device = torch.cuda.current_device()
    e2eTimer = Timer()
    start_time = time.time()
    print("Going into DDP")
    print("start")
    thread_num = 1
    enable_kernel_control = False
    enable_comm_control = False
    enable_profiler = False
    _CAPI_DGLDSInitialize(rank, world_size, thread_num , enable_kernel_control, enable_comm_control, enable_profiler)

    dataloader = QuiverDglSageSample(rank=rank, world_size=world_size, batch_size=batch_size, nids=train_idx, sampler=sampler)
    num_nodes = feat.shape[0]
    p_map = torch.arange(num_nodes).to(rank) % 4
    num_partitions = 4
    world_size = 4
    print(f"pre-heating model on device: {device}")
    # for input_nodes, output_nodes, blocks in dataloader:
    #     batch_feat = feat[input_nodes]
    #     batch_label = label[output_nodes]
    #     batch_pred = model(blocks, batch_feat)
    #     batch_loss = F.cross_entropy(batch_pred, batch_label)
    #     optimizer.zero_grad()
    #     batch_loss.backward()
    #     optimizer.step()

    print(f"training model on device: {device}")
    timer = Timer()
    sampling_timers = []
    feature_timers = []
    forward_timers = []
    backward_timers = []
    edges_computed = []
    for epoch in range(num_epoch):
        if rank == 0 and (epoch + 1) % 5 == 0:
            print(f"start epoch {epoch}")
        sampling_timer = CudaTimer()
        edges_computed_epoch = 0
        for input_nodes, output_nodes, blocks in dataloader:
            sampling_timer.end()
            feat_timer = CudaTimer()
            batch_feat = feat[input_nodes]
            local_ordering = feat.feature_order[input_nodes]
            if feat.cache_policy == "device_replicate":
                cache_hit = torch.where(local_ordering < offset.end_)
                cache_miss = torch.where(local_ordering > offset.end_)
                print("Cache ratio", cache_hit.shape[0], cache_miss.shape[0])
            if feat.cache_policy == "p2p_clique_replicate":
                cache_hit = torch.where(local_ordering < offset.end_)[0]
                cache_miss = torch.where(local_ordering > offset.end_)[0]
                hit_nodes = input_nodes[cache_hit]
                miss_nodes = input_nodes[cache_miss]
                scattered_array = _CAPI_getScatteredArrayObject \
                    (dgl.backend.zerocopy_to_dgl_ndarray(hit_nodes), \
                     dgl.backend.zerocopy_to_dgl_ndarray(p_map[hit_nodes]), \
                     num_partitions, rank, world_size)
                print("Cache ratio", cache_hit.shape[0], cache_miss.shape[0], scattered_array.unique_array.shape)

    ddp_exit()



if __name__== "__main__":
    graph_name = "ogbn-products"
    system = "quiver-gpu"
    cache_policy = "p2p_clique_replicate"
    cache_policy = "device_replicate"
    in_dir = "/data/ogbn/processed"
    in_dir = os.path.join(in_dir, graph_name)
    graph = load_dgl_graph(in_dir, is32=False, wsloop=True)
    row, col = graph.adj_tensors("coo")
    csr_topo = quiver.CSRTopo(edge_index=(col, row))
    feat, label, num_label = load_feat_label(in_dir)
    cache_size = "100M"
    world_size = 4
    batch_size = 1024
    num_epoch = 5
    quiver.init_p2p(device_list= list(range(world_size)))
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=False)

    quiver_feat = quiver.Feature(0, device_list=list(range(world_size)), \
                            cache_policy=cache_policy, device_cache_size= cache_size, csr_topo = csr_topo)
    quiver_feat.from_cpu_tensor(feat)
    if cache_policy == "p2p_clique_replicate":
        feat_width = feat.shape[1]
        shard = quiver_feat.clique_tensor_list[0]
        offset = shard.shard_tensor_config.tensor_offset_device[3]
        print(offset)
    else:

        for i in range(4):
            shard = quiver_feat.device_tensor_list[i]
            offset = shard.shard_tensor_config.tensor_offset_device[i]
            print(offset.end_)

        assert(False)
        # Use devece tensor list   assert "uva" in config.system or "gpu"  in config.system
    fanouts = [20,20,20]
    if "uva" in system:
        print("Using uva sampler")
        quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo=csr_topo, sizes= fanouts, mode="UVA")
    if "gpu" in system:
        print("Using gpu sampler")
        quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo=csr_topo, sizes= fanouts, mode="GPU")

    spawn(train_ddp, args=( quiver_sampler, quiver_feat, \
                         train_idx, \
                         offset, world_size, batch_size, num_epoch), nprocs=world_size)
