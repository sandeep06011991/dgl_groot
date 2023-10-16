import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.distributed as dist
import gc

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from dgl.utils import pin_memory_inplace, gather_pinned_tensor_rows

from .dgl_model import *
from .util import *

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_exit():
    dist.destroy_process_group()

def bench_dgl_batch(configs: list[Config], test_acc=False):
    for config in configs:
        assert(config.system == configs[0].system and config.graph_name == configs[0].graph_name)
    
    in_dir = os.path.join(config.data_dir, config.graph_name)
    graph = load_dgl_graph(in_dir, is32=True, wsloop=True)
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
    feat, label, num_label = load_feat_label(in_dir)
    for config in configs:
        try:
            spawn(train_ddp, args=(config, test_acc, graph, feat, label, num_label, train_idx, valid_idx, test_idx), nprocs=config.world_size, daemon=True)

        except Exception as e:
            print(e)
        gc.collect()
        torch.cuda.empty_cache()
            
def train_ddp(rank: int, config: Config, test_acc: bool,
              graph: dgl.DGLGraph, feat: torch.Tensor, label: torch.Tensor, num_label: int, 
              train_idx: torch.Tensor, valid_idx: torch.Tensor, test_idx: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    e2eTimer = Timer()
    graph_sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    dataloader, graph = get_dgl_sampler(graph=graph, train_idx=train_idx, graph_samler=graph_sampler, system=config.system, batch_size=config.batch_size, use_dpp=True)    
    if "uva" in config.system:
        feat_nd  = pin_memory_inplace(feat)
        label_nd = pin_memory_inplace(label)
    
    elif "gpu" in config.system:
        feat = feat.to(device)
        label = label.to(device)
        
    timer = Timer()
    model = None
    if config.model == "gat":
        model = DGLGat(in_feats=feat.shape[1], hid_feats=config.hid_size, num_layers=len(config.fanouts), out_feats=num_label, num_heads=4)
    elif config.model == "sage":
        model = DGLGraphSage(in_feats=feat.shape[1], hid_feats=config.hid_size, num_layers=len(config.fanouts), out_feats=num_label)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    sampling_timers = []
    feature_timers = []
    forward_timers = []
    backward_timers = []
    print(f"training model on {device}")
    for epoch in range(config.num_epoch):
        if rank == 0 and (epoch + 1) % 5 == 0:
            print(f"start epoch {epoch}")
        sampling_timer = CudaTimer()
        for input_nodes, output_nodes, blocks in dataloader:
            sampling_timer.end()
            
            feat_timer = CudaTimer()
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
        dataloader, graph = get_dgl_sampler(graph=graph, train_idx=test_idx, graph_samler=graph_sampler, system=config.system, batch_size=config.batch_size, use_dpp=True)    
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

# def bench_dgl_single(config: Config, test_acc=False):
#     if config.world_size == 1:
#         return train_single_gpu(config, config.data_dir, test_acc)

#     in_dir = os.path.join(config.data_dir, config.graph_name)
#     graph = load_dgl_graph(in_dir, is32=True, wsloop=True)
#     train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
#     feat, label, num_label = load_feat_label(in_dir)
#     spawn(train_ddp, args=(config, test_acc, graph, feat, label, num_label, train_idx, valid_idx, test_idx), nprocs=config.world_size, daemon=True)
    
# def train_single_gpu(config: Config, data_dir, test_acc):
#     device = torch.cuda.device(0)
#     e2eTimer = Timer()
#     in_dir = os.path.join(data_dir, config.graph_name)
#     graph = load_dgl_graph(in_dir, is32=True, wsloop=True)
#     train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
#     graph_sampler = dgl.dataloading.NeighborSampler(config.fanouts)
#     dataloader, graph = get_dgl_sampler(graph, train_idx, graph_sampler, config.system, config.batch_size, False)
#     feat, label, num_label = load_feat_label(in_dir)
    
#     if "uva" in config.system:
#         feat_nd  = pin_memory_inplace(feat)
#         label_nd = pin_memory_inplace(label)
    
#     elif "gpu" in config.system:
#         feat = feat.to(device)
#         label = label.to(device)
        
#     timer = Timer()
    
#     model = None
#     if config.model == "gat":
#         model = DGLGat(in_feats=feat.shape[1], hid_feats=config.hid_size, num_layers=len(config.fanouts), out_feats=num_label, num_heads=4)
#     elif config.model == "sage":
#         model = DGLGraphSage(in_feats=feat.shape[1], hid_feats=config.hid_size, num_layers=len(config.fanouts), out_feats=num_label)
#     model = model.to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     sampling_timers = []
#     feature_timers = []
#     forward_timers = []
#     backward_timers = []
#     print(f"train on device {device}")
#     for epoch in range(config.num_epoch):
#         print(f"start epoch {epoch}")
#         sampling_timer = CudaTimer()
#         for input_nodes, output_nodes, blocks in dataloader:
#             sampling_timer.end()
            
#             feat_timer = CudaTimer()
#             batch_feat = None
#             batch_label = None
#             if "uva" in config.system:
#                 batch_feat = gather_pinned_tensor_rows(feat, input_nodes)
#                 batch_label = gather_pinned_tensor_rows(label, output_nodes)
#             elif "gpu" in config.system:
#                 batch_feat = feat[input_nodes]
#                 batch_label = label[output_nodes]
#             else:
#                 print(f"{input_nodes=}")
#                 batch_feat = feat[input_nodes].to(device)
#                 batch_label = label[output_nodes].to(device)
#                 blocks = [block.to(device) for block in blocks]
                
#             feat_timer.end()            
            
#             forward_timer = CudaTimer()
#             batch_pred = model(blocks, batch_feat)
#             batch_loss = F.cross_entropy(batch_pred, batch_label)
#             forward_timer.end()
            
#             backward_timer = CudaTimer()
#             optimizer.zero_grad()
#             batch_loss.backward()
#             optimizer.step()
#             backward_timer.end()
            
#             sampling_timers.append(sampling_timer)
#             feature_timers.append(feat_timer)
#             forward_timers.append(forward_timer)
#             backward_timers.append(backward_timer)

#             sampling_timer = CudaTimer()
            
#     torch.cuda.synchronize()
#     duration = timer.duration()
#     sampling_time = get_duration(sampling_timers)
#     feature_time = get_duration(feature_timers)
#     forward_time = get_duration(forward_timers)
#     backward_time = get_duration(backward_timers)
#     profiler = Profiler(duration=duration, sampling_time=sampling_time, feature_time=feature_time, forward_time=forward_time, backward_time=backward_time, test_acc=0)
    
#     print(f"train for {config.num_epoch} epochs in {duration}s")
#     print(f"finished experiment {config} in {e2eTimer.duration()}s")
    
#     if test_acc:
#         print("testing model accuracy")
#         dataloader, graph = get_dgl_sampler(graph, test_idx, graph_sampler, config.system, config.batch_size, False)
#         model.eval()
#         ys = []
#         y_hats = []
        
#         for input_nodes, output_nodes, blocks in dataloader:
#             with torch.no_grad():
#                 batch_feat = None
#                 batch_label = None
#                 if "uva" in config.system:
#                     batch_feat = gather_pinned_tensor_rows(feat, input_nodes)
#                     batch_label = gather_pinned_tensor_rows(label, output_nodes)
#                 elif "gpu" in config.system:
#                     batch_feat = feat[input_nodes]
#                     batch_label = label[output_nodes]
#                 else:
#                     batch_feat = feat[input_nodes].to(device)
#                     batch_label = label[output_nodes].to(device)
#                     blocks = [block.to(device) for block in blocks]
#                 ys.append(batch_label)
#                 batch_pred = model(blocks, batch_feat)
#                 y_hats.append(batch_pred)
                
#         acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=num_label)
#         acc = round(acc.item() * 100, 2)
#         profiler.test_acc = acc  
#         print(f"test accuracy={acc}%")
              
#     write_to_csv(config.log_path, [config], [profiler])
