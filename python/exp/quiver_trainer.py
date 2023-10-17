import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.distributed as dist
import gc
import quiver

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn

from .dgl_model import *
from .util import *
from .quiver_sampler import *

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_exit():
    dist.destroy_process_group()

def get_cache_size(feat: torch.Tensor, config: Config):
    res = 1
    if torch.float32 == feat.dtype or torch.int32 == feat.dtype:
        res *= 4
    elif torch.float64 == feat.dtype or torch.int64 == feat.dtype:
        res *= 8
    
    for s in feat.shape:
        res *= s
    
    res /= 1024 * 1024
    res = int(res * config.cache_rate)
    return f"{res}MB"

def bench_quiver_batch(configs: list[Config], test_acc=False):
    for config in configs:
        assert(config.system == configs[0].system and config.graph_name == configs[0].graph_name)
    
    quiver.init_p2p(device_list=list(range(config.world_size)))
    in_dir = os.path.join(config.data_dir, config.graph_name)
    feat, label, num_label = load_feat_label(in_dir)
    cache_size = get_cache_size(feat, config)
    cache_size = "0MB"
    # quiver_feat = quiver.Feature(0, device_list=list(range(config.world_size)), cache_policy="p2p_clique_replicate", device_cache_size=cache_size)
    quiver_feat = quiver.Feature(0, device_list=list(range(config.world_size)), cache_policy="device_replicate", device_cache_size=cache_size)
    quiver_feat.from_cpu_tensor(feat)
    print(f"feature cache size is {cache_size}")
    feat_width = feat.shape[1]
    del feat

    graph = load_dgl_graph(in_dir, is32=False, wsloop=True)
    row, col = graph.adj_tensors("coo")
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=False)
    csr_topo = quiver.CSRTopo(edge_index=(col, row))

    for config in configs:
        try:
            quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo=csr_topo, sizes=config.fanouts, mode="UVA")
            spawn(train_ddp, args=(config, test_acc, quiver_sampler, quiver_feat, feat_width, label, num_label, train_idx, valid_idx, test_idx), nprocs=config.world_size)
        except Exception as e:
            print(e)
            write_to_csv(config.log_path, [config], [empty_profiler()])
        gc.collect()
        torch.cuda.empty_cache()
            
def train_ddp(rank: int, config: Config, test_acc: bool,
              sampler: quiver.pyg.GraphSageSampler, feat: quiver.Feature, feat_width:int, label: torch.Tensor, num_label: int, 
              train_idx: torch.Tensor, valid_idx: torch.Tensor, test_idx: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    e2eTimer = Timer()
    dataloader = QuiverDglSageSample(rank=rank, world_size=config.world_size, batch_size=config.batch_size, nids=train_idx, sampler=sampler)
    model = None
    if config.model == "gat":
        model = DGLGat(in_feats=feat_width, hid_feats=config.hid_size, num_layers=len(config.fanouts), out_feats=num_label, num_heads=4)
    elif config.model == "sage":
        model = DGLGraphSage(in_feats=feat_width, hid_feats=config.hid_size, num_layers=len(config.fanouts), out_feats=num_label)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    label = label.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    
    print(f"pre-heating model on device: {device}")
    for input_nodes, output_nodes, blocks in dataloader:
        batch_feat = feat[input_nodes]
        batch_label = label[output_nodes]
        batch_pred = model(blocks, batch_feat)
        batch_loss = F.cross_entropy(batch_pred, batch_label)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    print(f"training model on device: {device}")        
    timer = Timer()
    sampling_timers = []
    feature_timers = []
    forward_timers = []
    backward_timers = []
    
    for epoch in range(config.num_epoch):
        if rank == 0 and (epoch + 1) % 5 == 0:
            print(f"start epoch {epoch}")
        sampling_timer = CudaTimer()
        for input_nodes, output_nodes, blocks in dataloader:
            sampling_timer.end()
            
            feat_timer = CudaTimer()
            batch_feat = feat[input_nodes]
            batch_label = label[output_nodes]
            feat_timer.end()            
            
            forward_timer = CudaTimer()
            batch_pred = model(blocks, batch_feat)
            batch_loss = F.cross_entropy(batch_pred, batch_label)
            forward_timer.end()
            
            backward_timer = CudaTimer()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            backward_timer.end()
            
            sampling_timers.append(sampling_timer)
            feature_timers.append(feat_timer)
            forward_timers.append(forward_timer)
            backward_timers.append(backward_timer)

            sampling_timer = CudaTimer()
            
    torch.cuda.synchronize()
    duration = timer.duration()
    sampling_time = get_duration(sampling_timers)
    feature_time = get_duration(feature_timers)
    forward_time = get_duration(forward_timers)
    backward_time = get_duration(backward_timers)
    profiler = Profiler(duration=duration, sampling_time=sampling_time, feature_time=feature_time, forward_time=forward_time, backward_time=backward_time, test_acc=0)
    if rank == 0:
        print(f"train for {config.num_epoch} epochs in {duration}s")
        print(f"finished experiment {config} in {e2eTimer.duration()}s")
        if not test_acc:
            write_to_csv(config.log_path, [config], [profiler])
    
    dist.barrier()
    
    if test_acc:
        print(f"testing model accuracy on {device}")
        dataloader = QuiverDglSageSample(rank=rank, world_size=config.world_size, batch_size=config.batch_size, nids=test_idx, sampler=sampler)
        model.eval()
        ys = []
        y_hats = []
        
        for input_nodes, output_nodes, blocks in dataloader:
            with torch.no_grad():
                batch_feat = feat[input_nodes]
                batch_label = label[output_nodes]
                ys.append(batch_label)
                batch_pred = model(blocks, batch_feat)
                y_hats.append(batch_pred)  
        acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=num_label)
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        acc = round(acc.item() * 100 / config.world_size, 2)
        profiler.test_acc = acc  
        if rank == 0:
            print(f"test accuracy={acc}%")
            write_to_csv(config.log_path, [config], [profiler])
    ddp_exit()