import os, dgl, torch, time, csv, gc
from ogb.nodeproppred import DglNodePropPredDataset

def _build_dgl_graph(indptr, indices, edges) -> dgl.DGLGraph:
    graph = dgl.graph(("csc", (indptr, indices, edges)))
    return graph

def preprocess(graph_name, in_dir, out_dir) -> None:
    out_dir = os.path.join(out_dir, graph_name)
    try:
        os.mkdir(out_dir)
    except Exception as e:
        print(e)
    
    id_type = torch.int64
    idtype_str = "64"
    dataset = DglNodePropPredDataset(graph_name, in_dir)
    graph = dataset[0][0]
    if graph_name == "ogbn-proteins":
        feat = graph.edata.pop("feat")
        torch.save(feat, os.path.join(out_dir, "feat.pt"))
        species = graph.ndata["species"]
        torch.save(species, os.path.join(out_dir, "species.pt"))
    else:
        feat: torch.Tensor = graph.dstdata.pop("feat")
        torch.save(feat, os.path.join(out_dir, "feat.pt"))
        del feat
    
    graph.remove_self_loop()    
    graph.add_self_loop()    
    node_labels: torch.Tensor = dataset[0][1]
    node_labels = node_labels.flatten().clone()
    torch.nan_to_num_(node_labels, nan=0.0)
    node_labels: torch.Tensor = node_labels.type(torch.int64)
    
    torch.save(node_labels, os.path.join(out_dir, "label.pt"))

    idx_split = dataset.get_idx_split()
    train_idx = idx_split["train"].type(id_type)
    valid_idx = idx_split["valid"].type(id_type)
    test_idx = idx_split["test"].type(id_type)
    
    torch.save(train_idx, os.path.join(out_dir, f"train_idx_{idtype_str}.pt"))
    torch.save(valid_idx, os.path.join(out_dir, f"valid_idx_{idtype_str}.pt"))
    torch.save(test_idx, os.path.join(out_dir, f"test_idx_{idtype_str}.pt"))

    indptr, indices, edges = graph.adj_tensors("csc")
    indptr = indptr.type(id_type)
    indices = indices.type(id_type)
    edges = edges.type(id_type)
    
    torch.save(indptr, os.path.join(out_dir, f"indptr_{idtype_str}.pt"))
    torch.save(indices, os.path.join(out_dir, f"indices_{idtype_str}.pt"))
    torch.save(edges, os.path.join(out_dir, f"edges_{idtype_str}.pt"))

def add_self_loop(in_dir, out_dir=None):
    id_type = torch.int64
    idtype_str = "64"
    graph = load_dgl_graph(in_dir)
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    indptr, indices, edges = graph.adj_tensors("csc")
    indptr = indptr.type(id_type)
    indices = indices.type(id_type)
    edges = edges.type(id_type)
    if out_dir == None:
        out_dir = in_dir
    torch.save(indptr, os.path.join(out_dir, f"indptr_{idtype_str}_wsloop.pt"))
    torch.save(indices, os.path.join(out_dir, f"indices_{idtype_str}_wsloop.pt"))
    torch.save(edges, os.path.join(out_dir, f"edges_{idtype_str}_wsloop.pt"))
    
def load_graph(in_dir, is32=False, wsloop=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    idtype_str = "64"
    indptr = None
    indices = None
    if not wsloop: 
        indptr = torch.load(os.path.join(in_dir, f"indptr_{idtype_str}.pt"))
        indices = torch.load(os.path.join(in_dir, f"indices_{idtype_str}.pt"))
        # edges = torch.load(os.path.join(in_dir, f"edges_{idtype_str}.pt"))
        edges = torch.empty(0, dtype=indices.dtype)
    else:
        # with self loop
        indptr = torch.load(os.path.join(in_dir, f"indptr_{idtype_str}_wsloop.pt"))
        indices = torch.load(os.path.join(in_dir, f"indices_{idtype_str}_wsloop.pt"))
        # edges = torch.load(os.path.join(in_dir, f"edges_{idtype_str}_wsloop.pt"))
        edges = torch.empty(0, dtype=indices.dtype)
    if is32:
        return indptr.type(torch.int32), indices.type(torch.int32), edges.type(torch.int32)
    else:
        return indptr, indices, edges

def load_idx_split(in_dir, is32=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    idtype_str = "64"
    train_idx = torch.load(os.path.join(in_dir, f"train_idx_{idtype_str}.pt"))
    valid_idx = torch.load(os.path.join(in_dir, f"valid_idx_{idtype_str}.pt"))
    test_idx = torch.load(os.path.join(in_dir, f"test_idx_{idtype_str}.pt"))
    if is32:
        return train_idx.type(torch.int32), valid_idx.type(torch.int32), test_idx.type(torch.int32)
    else:
        return train_idx, valid_idx, test_idx

def load_feat_label(in_dir) -> (torch.Tensor, torch.Tensor, int):
    feat = torch.load(os.path.join(in_dir, f"feat.pt"))
    label = torch.load(os.path.join(in_dir, f"label.pt"))
    num_labels = torch.unique(label).shape[0]
    return feat, label, num_labels

def load_dgl_graph(in_dir, is32=False, wsloop=False) -> dgl.DGLGraph:
    indptr, indices, edges = load_graph(in_dir, is32, wsloop)
    graph = _build_dgl_graph(indptr, indices, edges)
    if is32:
        return graph.int()
    else:
        return graph
    
def get_dataset(graph_name, in_dir):
    dataset = DglNodePropPredDataset(graph_name, in_dir)
    return dataset

def get_dgl_sampler(graph: dgl.DGLGraph, train_idx: torch.Tensor, graph_samler: dgl.dataloading.Sampler, system:str = "cpu", batch_size:int=1024, use_dpp=False) -> dgl.dataloading.dataloader.DataLoader:
    device = torch.cuda.current_device()
    dataloader = None
    drop_last = True
    
    if device == torch.cuda.device(0):
        print(f"before dataloader init graph formats: {graph.formats()}")

    if system == "cpu":
        dataloader = dgl.dataloading.DataLoader(
            graph=graph,               # The graph
            indices=train_idx,         # The node IDs to iterate over in minibatches
            graph_sampler=graph_samler,     # The neighbor sampler
            device="cpu",      # Put the sampled MFGs on CPU or GPU
            use_ddp=use_dpp, # enable ddp if using mutiple gpus
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=drop_last,    # Whether to drop the last incomplete batch
            use_uva=False,
            num_workers=1,
        )
    elif system == "uva":
        graph.pin_memory_()
        assert(graph.is_pinned())
        dataloader = dgl.dataloading.DataLoader(
            graph=graph,               # The graph
            indices=train_idx.to(device),         # The node IDs to iterate over in minibatches
            graph_sampler=graph_samler,     # The neighbor sampler
            device=device,      # Put the sampled MFGs on CPU or GPU
            use_ddp=use_dpp, # enable ddp if using mutiple gpus
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=drop_last,    # Whether to drop the last incomplete batch
            use_uva=True,
            num_workers=0,
        )
    elif system == "gpu":
        graph = graph.to(device)
        dataloader = dgl.dataloading.DataLoader(
            graph=graph,               # The graph
            indices=train_idx.to(device),         # The node IDs to iterate over in minibatches
            graph_sampler=graph_samler,     # The neighbor sampler
            device=device,      # Put the sampled MFGs on CPU or GPU
            use_ddp=use_dpp, # enable ddp if using mutiple gpus
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=drop_last,    # Whether to drop the last incomplete batch
            use_uva=False,
            num_workers=0,
        )
    if device == torch.cuda.device(0):
        print(f"after dataloader init graph formats: {graph.formats()}")
    return dataloader, graph

def get_memory_info(device=torch.cuda.current_device(), rd=0):
    allocated_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024
    allocated_mb = round(allocated_mb, rd)
    reserved_mb = round(reserved_mb, rd)
    return allocated_mb, reserved_mb

class Timer:
    def __init__(self):
        self.start = time.time()
    def duration(self, rd=3):
        return round(time.time() - self.start, rd)
    def reset(self):
        self.start = time.time()

class CudaTimer:
    def __init__(self, stream=torch.cuda.current_stream()):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.stream = stream
        self.end_recorded = False
        self.start_event.record(stream=self.stream)

    def start(self):
        self.start_event.record(stream=self.stream)
        
    def end(self):
        self.end_event.record(stream=self.stream)
        self.end_recorded = True
        
    def reset(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_recorded = False
        
    def duration(self):
        assert(self.end_recorded)
        self.start_event.synchronize()
        self.end_event.synchronize()
        duration_ms = self.start_event.elapsed_time(self.end_event)
        duration_s = duration_ms / 1000
        return duration_s
    
class Config:
    def __init__(self, graph_name, world_size, num_epoch, fanouts, batch_size, system, model, hid_size, cache_rate, log_path, data_dir):
        self.graph_name = graph_name
        self.world_size = world_size
        self.num_epoch = num_epoch
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.system = system
        self.model = model
        self.hid_size = hid_size
        self.cache_rate = cache_rate
        self.log_path = log_path
        self.data_dir = data_dir
        
    def header(self):
        return ["graph_name", "world_size", "num_epoch", "fanouts", "batch_size", "system", "model", "hid_size", "cache_rate"]
    
    def content(self):
        return [self.graph_name, self.world_size, self.num_epoch, self.fanouts, self.batch_size, self.system, self.model, self.hid_size, self.cache_rate]

    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += "\n"
        return res
    
class Profiler:
    def __init__(self, duration: float, sampling_time : float, feature_time: float, forward_time: float, backward_time: float, test_acc):
        self.duration = duration
        self.sampling_time = sampling_time
        self.feature_time = feature_time
        self.forward_time = forward_time
        self.backward_time = backward_time
        self.test_acc = test_acc
        self.allocated_mb, self.reserved_mb = get_memory_info()
        
    def header(self):
        header = ["duration (s)", "sampling (s)", "feature (s)", "forward (s)", "backward (s)", "allocated (MB)", "reserved (MB)", "test accuracy %"]
        return header
    
    def content(self):
        content = [self.duration, self.sampling_time, self.feature_time, self.forward_time, self.backward_time, self.allocated_mb, self.reserved_mb, self.test_acc]
        return content
    
    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += "\n"
        return res


def get_duration(timers: list[CudaTimer], rb=3)->float:
    res = 0.0
    for timer in timers:
        res += timer.duration()
    return round(res, rb)

def write_to_csv(out_path, configs: list[Config], profilers: list[Profiler]):
    assert(len(configs) == len(profilers))
    def get_row(header, content):
        res = {}
        for k, v in zip(header, content):
            res[k] = v
        return res
    
    has_header = os.path.isfile(out_path)
    with open(out_path, 'a') as f:
        header = configs[0].header() + profilers[0].header()
        writer = csv.DictWriter(f, fieldnames=header)        
        if not has_header:
            writer.writeheader()
        for config, profiler in zip(configs, profilers):
            row = get_row(config.header() + profiler.header(), config.content() + profiler.content())
            writer.writerow(row)
    print("Experiment result has been written to: ", out_path)


# This is identical to the run function except it assumes the graph has been loaded into the memory
# def run_batch_helper(config: Config, graph, train_idx):
#     e2eTimer = Timer()
#     graph_sampler = dgl.dataloading.NeighborSampler(config.fanouts)
#     dataloader, graph = get_dataloader(graph, train_idx, graph_sampler, config.system, config.batch_size, False)
#     # start pre-heating
#     for input_nodes, output_nodes, blocks in dataloader:
#         continue
#     # end pre-heating
#     dst_nodes = [0] * len(config.fanouts) 
#     src_nodes = [0] * len(config.fanouts) 
#     num_edges = [0] * len(config.fanouts) 
#     timer = Timer()
#     for epoch in range(config.num_epoch):
#         print(f"start epoch {epoch}")
#         for input_nodes, output_nodes, blocks in dataloader:
#             for i, block in enumerate(blocks):
#                 dst_nodes[i] += block.num_dst_nodes()
#                 src_nodes[i] += block.num_src_nodes()
#                 num_edges[i] += block.num_edges()   
                
#     duration = timer.duration()
#     profiler = Profiler(duration=duration, sampling_time=duration, training_time=0, feature_time=0, dst_nodes=dst_nodes, src_nodes=src_nodes, num_edges=num_edges)
#     print(f"run_batch_helper: sample {config.num_epoch} epochs in {duration}s")
#     print(f"run_batch_helper: finished experiment {config.content()} in {e2eTimer.duration()}s")
#     return profiler, graph

# def run_batch(configs: list[Config]):
#     for config in configs:
#         assert(config.system == configs[0].system)
#         assert(config.graph_name == configs[0].graph_name)
    
#     data_dir = "/mnt/homes/juelinliu/dataset/OGBN/processed"
#     in_dir = os.path.join(data_dir, config.graph_name)
#     graph = load_dgl_graph(in_dir, is32=True)
#     train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
#     profilers = []
#     for config in configs:
#         profiler, graph = run_batch_helper(config, graph, train_idx)
#         profilers.append(profiler)
#         gc.collect()
#         torch.cuda.empty_cache()
#     return profilers

# def get_configs(graph_name, system, num_epoch=1):
#     fanout_list = [[10, 10, 10], [15, 15, 15], [20,20,20]]
#     batch_size_list = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
#     configs = []
#     for fanouts in fanout_list:
#         for batch_size in batch_size_list:
#             config = Config(graph_name=graph_name, num_epoch=num_epoch, fanouts=fanouts, batch_size=batch_size, system=system)
#             configs.append(config)
            
#     return configs

# def bench(graph_name, system, num_epoch=20, log_dir="/mnt/homes/juelinliu/project/dgl/python/log"):
#     configs = get_configs(graph_name=graph_name, system=system, num_epoch=num_epoch)
#     profilers = run_batch(configs)
#     log_name = f"{graph_name}_{system}.csv"
#     log_path = os.path.join(log_dir, log_name)
#     write_to_csv(log_path, configs, profilers)
#     gc.collect()
#     torch.cuda.empty_cache()


