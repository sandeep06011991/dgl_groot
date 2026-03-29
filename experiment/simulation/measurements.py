# import torch
from torch.cuda import max_memory_allocated, max_memory_reserved, current_device


def get_memory_info(device=None, rd=0):
    if device == None:
        device = current_device()
    allocated_mb = max_memory_allocated(device) / 1024 / 1024
    reserved_mb = max_memory_reserved(device) / 1024 / 1024
    allocated_mb = round(allocated_mb, rd)
    reserved_mb = round(reserved_mb, rd)
    return allocated_mb, reserved_mb

class Profiler:
    def __init__(self, computation_cost: float, communication_cost : float, epoch_num: int):
        self.computation_cost = computation_cost
        self.communication_cost = communication_cost
        # self.edge_skew = 0
        # self.run_time = 0
        self.epoch_num = epoch_num 
        
    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num

    def header(self):
        header = ["computation_cost (s)", "communication_cost (s)"]
        return header
    

        
    def content(self):
        assert(self.epoch_num != -1)
        
        def avg(t):
            return round(t / self.epoch_num, 2)
        
        content = [avg(self.computation_cost), 
                   avg(self.communication_cost)]
        return content
    
    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += "\n"
        return res


class KProfiler(Profiler):
    def __init__(self, k = -1, num_nvlinks = -1, computation_cost= -1, communication_cost = -1, epoch_num = -1):
        super().__init__(computation_cost, communication_cost, epoch_num)
        self.k = k
        self.num_nvlinks = num_nvlinks

    def header(self):
        header = ["Split from layer","num_nvlinks"]
        return header + super().header()



    def content(self):
        assert(self.epoch_num != -1)

        content = [self.k, self.num_nvlinks]
        return content + super().content()
