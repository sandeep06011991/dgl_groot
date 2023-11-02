import nvtx
import torch.multiprocessing
import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.distributed as dist
import gc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from dgl.utils import pin_memory_inplace, gather_pinned_tensor_rows

from runner.util import *
from .util import *
from dgl.groot import *
from dgl.groot.models import get_distributed_model
from torch.nn.parallel import DistributedDataParallel

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_exit():
    dist.destroy_process_group()

def bench_groot_batch(configs: list[Config], test_acc=False):
    for config in configs:
        assert(config.system == configs[0].system and config.graph_name == configs[0].graph_name)
    in_dir = os.path.join(configs[0].data_dir, configs[0].graph_name)
    graph = load_dgl_graph(in_dir, is32=True, wsloop=True)
    partition_map = get_metis_partition(in_dir, config, graph)
    # partition_map = torch.randint(0, configs[0].world_size, (graph.num_nodes(),))
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
    indptr, indices, edges = load_graph(in_dir, is32=True, wsloop=True)
    feat, label, num_label = load_feat_label(in_dir)
    max_cache_fraction_queue = torch.multiprocessing.Queue(1)
    train_idx_list = []
    for p in range(config.world_size):
        train_idx_list.append(train_idx[partition_map[train_idx] == p])
    for config in configs:
        if config.graph_name == "ogbn-products" or config.graph_name == "com-orkut":
            config.cache_rate = 1
            config.system = "groot-gpu"
        if config.graph_name == "ogbn-papers100M" or config.graph_name == "com-friendster":
            config.cache_rate  = .05
            config.system = "groot-uva"
        config.cache_percentage = config.cache_rate
        cached_ids = get_cache_ids_by_sampling(config, graph, train_idx_list)
        try:
            spawn(train_ddp, args=(config, test_acc, graph, feat, label, \
                                   num_label, train_idx_list, valid_idx, test_idx, \
                                   indptr, indices, edges, None , partition_map, cached_ids), \
                  nprocs=config.world_size, daemon=True, join= True)

        except Exception as e:
            write_to_csv(config.log_path, [config], [empty_profiler()])
            print(e)
            # assert(False)
        gc.collect()
        torch.cuda.empty_cache()

def train_ddp(rank: int, config: Config, test_acc: bool,
              graph: dgl.DGLGraph, feat: torch.Tensor, label: torch.Tensor, num_label: int, 
              train_idx_list: [torch.Tensor], valid_idx: torch.Tensor, test_idx: torch.Tensor,
              indptr: torch.Tensor, indices: torch.Tensor, edges:torch.Tensor, \
                queue: torch.multiprocessing.Queue, partition_map: torch.Tensor, cached_ids):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    e2eTimer = Timer()
    indptr_handle = pin_memory_inplace(indptr)
    indices_handle = pin_memory_inplace(indices)
    edge_id_handle = pin_memory_inplace(edges)

    if "uva" in config.system:
        feat_nd  = pin_memory_inplace(feat)
        label_nd = pin_memory_inplace(label)
    
    elif "gpu" in config.system:
        feat = feat.to(device)
        label = label.to(device)

    max_pool_size = 1
    num_layers = len(config.fanouts)
    block_type = get_block_type("src_to_dst")
    step = min([(int(idx.shape[0] / config.batch_size) + 1) for idx in train_idx_list])
    train_idx = train_idx_list[rank].to(rank)
    partition_map = partition_map.to(rank)
    dataloader = init_groot_dataloader(
        rank, config.world_size, block_type, rank, config.fanouts,
        config.batch_size,  config.num_redundant_layer, max_pool_size,
        indptr, indices, feat, label,
        train_idx, valid_idx, test_idx, partition_map
    )
    if cached_ids != None:

        cache_id = cached_ids[rank].to(rank)
        # cache_id = get_cache_ids_by_sampling(config, graph, train_idx)
        init_groot_dataloader_cache(cache_id.to(indptr.dtype))

    n_classes = torch.max(label) + 1
    model = get_distributed_model( config.model, feat.shape[1], num_layers, config.hid_size, \
                                   n_classes, config.num_redundant_layer,
                                   rank, config.world_size).to(rank)
    model = DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    timer = Timer()
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    sampling_timers = []
    feature_timers = []
    forward_timers = []
    backward_timers = []
    edges_computed = []
    max_memory_used = []
    print(f"training model on {device}")
    for epoch in range(config.num_epoch):
        if rank == 0 and (epoch + 1) % 1 == 0:
            print(f"start epoch {epoch}")
        randInt = torch.randperm(train_idx.shape[0], device = rank)
        shuffle_training_nodes(randInt)
        edges_per_epoch = 0
        for _ in range(step):
            nvtx.push_range('minibatch_Start')
            sampling_timer = CudaTimer()
            
            extract_feat_label_flag = False
            key = sample_batch_sync(extract_feat_label_flag)
            sampling_timer.end()
            
            feat_timer = CudaTimer()
            is_async = False
            extract_batch_feat_label(key, is_async)
            feat_timer.end()

            blocks, batch_feat, batch_label = get_batch(key, layers = num_layers, \
                                n_redundant_layers = config.num_redundant_layer , mode = "SRC_TO_DEST")
            
            if config.num_redundant_layer == num_layers:
                local_blocks = blocks
            else:
                local_blocks, _, _ = blocks
            for block in local_blocks:
                edges_per_epoch += block.num_edges()
            forward_timer = CudaTimer()
            pred = model(blocks, batch_feat)
            loss = torch.nn.functional.cross_entropy(pred, batch_label)
            optimizer.zero_grad()
            forward_timer.end()
            backward_timer = CudaTimer()
            loss.backward()
            backward_timer.end()
            optimizer.step()
            nvtx.pop_range()
            sampling_timers.append(sampling_timer)
            feature_timers.append(feat_timer)
            forward_timers.append(forward_timer)
            backward_timers.append(backward_timer)
            max_memory_used.append(torch.cuda.memory_reserved())
            
    torch.cuda.synchronize()
    duration = timer.duration()
    edges_computed.append(edges_per_epoch)
    sampling_time = get_duration(sampling_timers)
    feature_time = get_duration(feature_timers)
    forward_time = get_duration(forward_timers)
    backward_time = get_duration(backward_timers)
    profiler = Profiler(duration=duration, sampling_time=sampling_time,
                        feature_time=feature_time, forward_time=forward_time, backward_time=backward_time, test_acc=0)
    profiler.edges_computed = sum(edges_computed)/len(edges_computed)
    edges_computed_max  = torch.tensor(edges_computed).to(rank)
    edges_computed_min  = torch.tensor(edges_computed).to(rank)
    edges_computed_avg  = torch.tensor(edges_computed).to(rank)
    dist.all_reduce(edges_computed_max, op = dist.ReduceOp.MAX)
    dist.all_reduce(edges_computed_min, op = dist.ReduceOp.MIN)
    dist.all_reduce(edges_computed_avg, op = dist.ReduceOp.SUM)
    profiler.edges_computed = edges_computed_avg.item()/4
    profiler.edge_skew = (edges_computed_max.item() - edges_computed_min.item()) / profiler.edges_computed
    print(config.cache_rate, "cache rate check")

    if config.cache_rate == 0:
        max_available_memory = 15 * (1024 ** 3) - (max(max_memory_used))
        percentage_of_nodes = min(1.0, max_available_memory/(feat.shape[0] * feat.shape[1] * 4))
        print("Can cache percentage of nodes", percentage_of_nodes)
        cache_rate = torch.tensor(percentage_of_nodes).to(rank)
        dist.all_reduce(cache_rate, op=dist.ReduceOp.MIN)
        if rank == 0:
            print("calculated cache size", cache_rate.item())
            queue.put(cache_rate.item())
    if rank == 0:
        print(f"train for {config.num_epoch} epochs in {duration}s")
        print(f"finished experiment {config} in {e2eTimer.duration()}s")

    if not test_acc:
            write_to_csv(config.log_path, [config], [profiler])
    
    dist.barrier()
    
    if test_acc :
        print(f"testing model accuracy on {device}")
        graph_sampler = dgl.dataloading.NeighborSampler(fanouts=config.fanouts)
        dataloader, graph = get_dgl_sampler(graph=graph, train_idx=test_idx, \
                                            graph_samler=graph_sampler, system=config.system, \
                                            batch_size=config.batch_size, use_dpp=True)
        model.eval()
        ys = []
        y_hats = []
        for input_nodes, output_nodes, blocks in dataloader:
            with torch.no_grad():
                batch_feat = None
                batch_label = None
                if "uva" in config.system:
                    batch_feat = gather_pinned_tensor_rows(feat, input_nodes)
                    batch_label = gather_pinned_tensor_rows(label, output_nodes)
                elif "gpu" in config.system:
                    batch_feat = feat[input_nodes]
                    batch_label = label[output_nodes]
                else:
                    batch_feat = feat[input_nodes].to(device)
                    batch_label = label[output_nodes].to(device)
                    blocks = [block.to(device) for block in blocks]
                ys.append(batch_label)
                batch_pred = model(blocks, batch_feat, inference = True)
                y_hats.append(batch_pred)  
        acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=num_label)
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        acc = round(acc.item() * 100 / config.world_size, 2)
        profiler.test_acc = acc  
        if rank == 0:
            print(f"test accuracy={acc}%")
            write_to_csv(config.log_path, [config], [profiler])
    ddp_exit()

