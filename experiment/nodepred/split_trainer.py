import torch
import torchmetrics.functional as MF
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from utils import *
from dgl.dev.splitloader import *
from dgl.dev.splitmodel import get_distributed_model
from dgl.dev import CudaProfilerStart, CudaProfilerStop

def bench_split(config: Config):
    graph, feat, label, train_idx, valid_idx, test_idx, num_label = load_data(config, is_pinned=True)
    config.num_classes = num_label
    
    partition_map = None
    if "random" not in config.partition_type:
        partition_map = load_partition_map(config)
    else:
        print(f"using random with {config.num_partition} partitions")
        v_num = graph.num_nodes()
        partition_map = torch.randint(low=0, high=config.num_partition, size=(v_num,), dtype=torch.int8)
    try:
        unique_id = GetUniqueId()
        spawn(train_split_ddp, args=(config, unique_id, graph, feat, label, train_idx, test_idx, partition_map),
              nprocs=config.world_size)
    except Exception as e:
        print(f"error encountered with {config=}:", e)
        exit(-1)


def train_split_ddp(rank: int, config: Config, unique_id, 
                    graph: dgl.DGLGraph, feat:torch.Tensor, label: torch.Tensor, 
                    train_idx: torch.Tensor, test_idx: torch.Tensor, 
                    partition_map: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    if rank == 0:
        print(config)
    mode = "uva"
    graph = graph.pin_memory_()
    feat_handle  = pin_memory_inplace(feat)
    label = label.to(device)
        
    assert(partition_map.shape[0] == graph.num_nodes())
    
    model = get_distributed_model(rank, config.world_size, config.num_redundant_layer, 
                                  config.model, feat.shape[1], len(config.fanouts), config.hid_size, config.num_classes)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample_config = SampleConfig(rank=rank, num_partition=config.num_partition, batch_size=config.batch_size,
                                 world_size=config.world_size, mode=mode, fanouts=config.fanouts, reindex=True,
                                 drop_last=True)

    dataloader = SplitGraphLoader(graph, partition_map, train_idx, unique_id, sample_config)
    local_ids = torch.empty([], dtype=torch.int64)
    
    if "G" in config.cache_size:
        ids = torch.arange(0, partition_map.shape[0])
        local_ids = ids[partition_map == rank]
        size = int(config.cache_size.removesuffix("G")) * (1024 ** 3)
        num_ids_cached = min(size // (feat.shape[1] * 4), local_ids.shape[0])
        local_ids = local_ids[:num_ids_cached].clone()
    
    print(f"rank {rank} cached {local_ids.shape} nodes")

    dataloader.init_featloader(feat, local_ids)
    step = 0
    step_per_epoch = dataloader.max_step_per_epoch
    dist.barrier()

    CudaProfilerStart()
    print(f"sampling on device: {device}", flush=True)
    timer = Timer()
    sampling_timers = []
    feature_timers = []
    forward_timers = []
    backward_timers = []
    edges_computed = []
    for epoch in range(config.num_epoch):
        edges_computed_epoch = 0
        sampling_timer = CudaTimer()
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1
            sampling_timer.end()
            feat_timer = CudaTimer()
            batch_feat = dataloader.get_feature()
            batch_label = label[output_nodes]
            dist.barrier()
            feat_timer.end()
            forward_timer = CudaTimer()
            batch_pred = model(blocks, batch_feat)
            for block in blocks:
                edges_computed_epoch += block.num_edges()
            forward_timer.end()
            backward_timer = CudaTimer()
            batch_loss = torch.nn.functional.cross_entropy(batch_pred, batch_label)
            optimizer.zero_grad()
            batch_loss.backward()
            backward_timer.end()
            optimizer.step()
            sampling_timers.append(sampling_timer)
            feature_timers.append(feat_timer)
            forward_timers.append(forward_timer)
            backward_timers.append(backward_timer)
            sampling_timer = CudaTimer()
            log_step(rank, epoch, step, step_per_epoch, timer)
        edges_computed.append(edges_computed_epoch)
    dist.barrier()
    CudaProfilerStop()
    duration = timer.duration()

    sampling_time = get_duration(sampling_timers)
    feature_time = get_duration(feature_timers)
    forward_time = get_duration(forward_timers)
    backward_time = get_duration(backward_timers)

    acc = 0
    if config.graph_name in ["products", "arxiv", "papers100M"]:
        # proof-of-correctness: per-rank local test accuracy on real labels.
        # collective-free (no all_reduce/barrier) and guarded so a failure here
        # can never lose the timing row written below.
        try:
            t2 = Timer()
            print(f"testing model accuracy on {device}")
            model.eval()
            ys = []
            y_hats = []
            dataloader.set_target_idx(test_idx)
            step = 0
            step_per_epoch = dataloader.max_step_per_epoch

            for input_nodes, output_nodes, blocks in dataloader:
                with torch.no_grad():
                    batch_feat = dataloader.get_feature()
                    batch_label = label[output_nodes]
                    ys.append(batch_label)
                    batch_pred = model(blocks, batch_feat)
                    y_hats.append(batch_pred)
                step+=1
                log_step(rank, "testing", step, step_per_epoch, timer)

            acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=config.num_classes)
            acc = round(acc.item() * 100, 2)
            print(f"[rank {rank}] test accuracy={acc}% in {t2.duration()} secs")
        except Exception as e:
            print(f"WARN accuracy eval failed on rank {rank}: {e}")
            acc = 0

    if rank == 0:
        profiler = Profiler(duration=duration, sampling_time=sampling_time,
                            feature_time=feature_time, forward_time=forward_time, backward_time=backward_time, test_acc=acc)
        profile_edge_skew(edges_computed, profiler, rank)
        write_to_csv(config.log_path, [config], [profiler])
    ddp_exit()
