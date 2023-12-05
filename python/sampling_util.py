import os, dgl, torch, time, csv, gc
from ogb.nodeproppred import DglNodePropPredDataset
import pandas as pd
import numpy as np

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

    node_labels: torch.Tensor = dataset[0][1]
    node_labels = node_labels.flatten().clone()
    torch.nan_to_num_(node_labels, nan=0.0)
    node_labels: torch.Tensor = node_labels.type(torch.int64)
    
    torch.save(node_labels, os.path.join(out_dir, "label.pt"))

    idx_split = dataset.get_idx_split()
    train_idx = idx_split["train"].type(id_type)
    valid_idx = idx_split["valid"].type(id_type)
    test_idx = idx_split["test"].type(id_type)

    ntype = torch.zeros(graph.num_nodes(), dtype = torch.int64)
    count = 0
    for k in ["train", "valid", "test"]:
        ids = idx_split[k].type(id_type)
        ntype[ids] = count
        count  = count + 1
    torch.save(ntype, os.path.join(out_dir, f'ntype.pt'))


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
    add_self_loop(out_dir, out_dir)

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

def get_metis_partition(in_dir, config, graph):
    assert config.partition_type in ["edge_balanced", "node_balanced", "random"]
    if config.partition_type == "random":
        return torch.randint(0, 4, (graph.num_nodes(),), dtype = torch.int32)
    if config.partition_type == "edge_balanced":
        edge_balanced = True
        return torch.load(f'{in_dir}/partition_map_{edge_balanced}').to(torch.int32)
    if config.partition_type == "node_balanced":
        edge_balanced = False
        return torch.load(f'{in_dir}/partition_map_{edge_balanced}').to(torch.int32)

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
    elif "uva" in system:
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
    elif "gpu" in system:
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
    def __init__(self, graph_name, world_size, num_epoch, fanouts,
                 batch_size, system, model, hid_size, cache_size, log_path, data_dir, pool_size, batch_layer, replace):
        try:
            self.machine_name = os.environ['MACHINE_NAME']
        except Exception as e:
            self.machine_name = "Elba"
        self.graph_name = graph_name
        self.world_size = world_size
        self.num_epoch = num_epoch
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.system = system
        self.model = model
        self.in_feat = -1
        self.num_classes = -1
        self.cache_size = cache_size
        self.hid_size = hid_size
        self.log_path = log_path
        self.data_dir = data_dir
        self.num_redundant_layer = len(self.fanouts)
        self.partition_type = "edge_balanced"
        self.pool_size = pool_size
        self.batch_layer = batch_layer
        self.replace = replace
        
    def get_file_name(self):
        if "groot" not in self.system:
            return (f"{self.system}_{self.graph_name}_{self.model}_{self.batch_size}_{self.hid_size}_" + \
                     f"{len(self.fanouts)}x{self.fanouts[0]}_{self.cache_size}")
        else:
            return (f"{self.system}_{self.graph_name}_{self.model}_{self.batch_size}_{self.hid_size}_" + \
                    f"{len(self.fanouts)}x{self.fanouts[0]}_{self.num_redundant_layer}_{self.cache_size}")

    def header(self):
        return ["machine_name", "system", "graph_name", "world_size", "num_epoch", "fanouts", "batch_size", "pool_size", "batch_layer", "replace"]
    
    def content(self):
        return [ self.machine_name, self.system, self.graph_name, self.world_size, self.num_epoch, self.fanouts, self.batch_size, self.pool_size, self.batch_layer, self.replace]

    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += f"num_classes={self.num_classes}"
        res += "\n"
        return res    

class Profiler:
    def __init__(self, num_epoch: int, duration: float, sampling_time : float, feature_time: float,\
                 forward_time: float, backward_time: float, test_acc: float):
        self.num_epoch = num_epoch
        self.duration = duration
        self.sampling_time = sampling_time
        self.feature_time = feature_time
        self.forward_time = forward_time
        self.backward_time = backward_time
        self.test_acc = test_acc
        self.allocated_mb, self.reserved_mb = get_memory_info()
    def header(self):
        header = ["epoch (s)", "sampling (s)", "feature (s)", "forward (s)", "backward (s)", \
                  "allocated (MB)", "reserved (MB)", "test accuracy %", ]
        return header
    
    def content(self):
        content = [self.duration / self.num_epoch, \
                   self.sampling_time / self.num_epoch, \
                   self.feature_time / self.num_epoch, \
                   self.forward_time / self.num_epoch, \
                   self.backward_time / self.num_epoch, \
                   self.allocated_mb, \
                   self.reserved_mb, \
                   self.test_acc]
        return content
    
    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += "\n"
        return res

def empty_profiler():
    empty = -1
    profiler = Profiler(num_epoch=1, duration=empty, sampling_time=empty, feature_time=empty, forward_time=empty, backward_time=empty, test_acc=empty)
    return profiler

def oom_profiler():
    oom = "oom"
    profiler = Profiler(num_epoch=1, duration=oom, sampling_time=oom, feature_time=oom, forward_time=oom, backward_time=oom, test_acc=oom)
    return profiler

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

def get_configs(graph_name, system, log_path, data_dir, pool_size, batch_layer, replace):
    fanouts = [[10, 10, 10]]
    pool_sizes = [pool_size]
    configs = []
    for fanout in fanouts:
        for pool_size in pool_sizes:
            config = Config(graph_name=graph_name, 
                            world_size=1, 
                            num_epoch=1, 
                            fanouts=fanout, 
                            batch_size=1024, 
                            system=system, 
                            model="sage",
                            hid_size=128, 
                            cache_size=0, 
                            log_path=log_path,
                            data_dir=data_dir,
                            pool_size=pool_size,
                            batch_layer=batch_layer,
                            replace=replace)
            configs.append(config)
    return configs