import torch.cuda

import dgl
import  dgl.backend as F
import torch.multiprocessing as mp
from dgl.groot import *
from dgl.groot.models import *
from dgl._ffi.function import _init_api
_init_api("dgl.groot", __name__)
_init_api("dgl.ds", __name__)
def run_proc(rank, world_size):
    # Initalize()
    thread_num = 1
    enable_kernel_control = False
    enable_comm_control = False
    enable_profiler = False
    _CAPI_DGLDSInitialize(rank, world_size, thread_num , enable_kernel_control, enable_comm_control, enable_profiler)
    # Test Scatter Object()
    s1 = torch.cuda.Stream()
    with (torch.cuda.stream(s1)):
        frontier =F.zerocopy_to_dgl_ndarray(torch.arange(16, device = rank))
        partition_map = F.zerocopy_to_dgl_ndarray(torch.arange(16, device = rank) % 4)
        num_partitions = 4
        scattered_array = _CAPI_getScatteredArrayObject\
                (frontier, partition_map, num_partitions, rank, world_size)
        print(scattered_array.unique_array)
        feat = torch.ones((4,128), device = rank, requires_grad = True)

        forward = Shuffle.apply( scattered_array, feat, rank, world_size)
        forward.sum().backward()
              # ).backward()
        print(feat.grad)
        s1.synchronize()
        print("shape ", forward.shape)
     #    Create all torch layers
     #Test forward and Backward Pass on this object

def run_naive(rank):
    pass
#   Init distributed process group
#   get random frontier in range


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_proc,(world_size,),nprocs = world_size)



