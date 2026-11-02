# import torch
from torch.cuda import max_memory_allocated, max_memory_reserved, current_device
from torch import tensor

def get_memory_info(device=None, rd=0):
    if device == None:
        device = current_device()
    allocated_mb = max_memory_allocated(device) / 1024 / 1024
    reserved_mb = max_memory_reserved(device) / 1024 / 1024
    allocated_mb = round(allocated_mb, rd)
    reserved_mb = round(reserved_mb, rd)
    return allocated_mb, reserved_mb

class Profiler:
    def __init__(self, duration: float, sampling_time : float, feature_time: float,\
                 forward_time: float, backward_time: float, test_acc):
        self.duration = duration
        self.sampling_time = sampling_time
        self.feature_time = feature_time
        self.forward_time = forward_time
        self.backward_time = backward_time
        self.test_acc = test_acc
        self.allocated_mb, self.reserved_mb = get_memory_info()
        self.edges_computed = 0
        self.edges_computed_min = 0
        self.edges_computed_max = 0
        self.edge_skew = 0
        self.run_time = 0
        self.epoch_num = -1
        
    def header(self):
        header = ["duration (s)", "sampling (s)", "feature (s)", "forward (s)", "backward (s)",\
                    "allocated (MB)", "reserved (MB)", "test accuracy %", "edges_computed", "edge_skew", "min_edge", "max_edge","run_time"]
        return header
    
    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num
        
    def content(self):
        def avg(t):
            # sentinel rows (timeout_profiler/oom_profiler) carry string markers
            # like "timeout"/"oom" and never ran an epoch -> emit them verbatim.
            if not isinstance(t, (int, float)):
                return t
            assert(self.epoch_num != -1)
            return round(t / self.epoch_num, 2)
        
        content = [avg(self.duration), 
                   avg(self.sampling_time), 
                   avg(self.feature_time), 
                   avg(self.forward_time),
                   avg(self.backward_time), 
                   self.allocated_mb, 
                   self.reserved_mb, 
                   self.test_acc,
                   self.edges_computed, 
                   self.edge_skew, 
                   self.edges_computed_min, 
                   self.edges_computed_max, 
                   self.run_time]
        return content
    
    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += "\n"
        return res

def profile_edge_skew(edges_computed: int, profiler:Profiler, rank:int):
    profiler.edges_computed = sum(edges_computed)/len(edges_computed)
    edges_computed = sum(edges_computed)/len(edges_computed)
    edges_computed_max  = tensor(edges_computed).to(rank)
    edges_computed_min  = tensor(edges_computed).to(rank)
    edges_computed_avg  = tensor(edges_computed).to(rank)

    profiler.edges_computed_min = edges_computed_min.item()
    profiler.edges_computed_max = edges_computed_max.item()
    profiler.edges_computed = edges_computed_avg.item() / 4
    profiler.edge_skew = (edges_computed_max.item() - edges_computed_min.item()) / profiler.edges_computed

def empty_profiler():
    empty = -1
    profiler = Profiler(duration=empty, sampling_time=empty, feature_time=empty, forward_time=empty, backward_time=empty, test_acc=empty)
    return profiler

def oom_profiler():
    oom = "oom"
    profiler = Profiler(duration=oom, sampling_time=oom, feature_time=oom, forward_time=oom, backward_time=oom, test_acc=oom)
    return profiler

def timeout_profiler():
    # Marks a cell that exceeded the per-run wall-clock cap (see sweep.sh RUN_TIMEOUT).
    # Mirrors oom_profiler so the row has the full schema with "timeout" sentinels.
    t = "timeout"
    profiler = Profiler(duration=t, sampling_time=t, feature_time=t, forward_time=t, backward_time=t, test_acc=t)
    profiler.allocated_mb = t
    profiler.reserved_mb = t
    return profiler
