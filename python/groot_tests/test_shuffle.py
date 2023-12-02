import torch.cuda

import dgl
import  dgl.backend as F
import torch.multiprocessing as mp
from dgl.groot import *
from dgl.groot.models import *
from dgl._ffi.function import _init_api
import dgl
import  dgl.backend as F
import torch.multiprocessing as mp
from dgl.groot import *
from dgl.groot.models import *
from dgl._ffi.function import _init_api
import torch.cuda
import os
import torch.distributed as dist
import numpy as np
_init_api("dgl.groot", __name__)
_init_api("dgl.ds", __name__)

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_exit():
    dist.destroy_process_group()



class ShuffleAlt(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  feat, frontier_send_idx, frontier_recv_idx,  rank, world_size, original_frontier_size):
        ctx.rank = rank
        ctx.unique_frontier_size = feat.shape[0]
        ctx.world_size = world_size
        ctx.frontier_send_idx = frontier_send_idx
        ctx.frontier_recv_idx = frontier_recv_idx
        to_send = []
        to_recv = []
        for recv in frontier_recv_idx:
            to_send.append(feat[recv])
        torch.cuda.synchronize()
        for i,send_size in enumerate(frontier_send_idx):
            to_recv.append(torch.empty((send_size.shape[0], feat.shape[1]), device = rank, dtype  = feat.dtype))
        dist.all_to_all(to_recv, to_send)
        out = torch.zeros((original_frontier_size,feat.shape[1]), device = rank, dtype = feat.dtype)
        for id, recv in enumerate(to_recv):
            out[frontier_send_idx[id]] = recv
        out.requires_grad = True
        return out

    @staticmethod
    def backward(ctx, grads):
        rank = ctx.rank
        world_size = ctx.world_size
        unique_frontier_size = ctx.unique_frontier_size
        frontier_send_idx = ctx.frontier_send_idx
        frontier_recv_idx = ctx.frontier_recv_idx
        to_send = []
        to_recv = []
        for send in frontier_send_idx:
            to_send.append(grads[send])
        for recv in frontier_recv_idx:
            to_recv.append(torch.empty((recv.shape[0], grads.shape[1]), device = rank, dtype = grads.dtype))
        obj = dist.all_to_all(to_recv, to_send)
        out = torch.zeros((unique_frontier_size, grads.shape[1]), device = rank, dtype = grads.dtype)
        for id, recv in enumerate(to_recv):
            out[frontier_recv_idx[id]] += recv
        return out, None, None, None, None, None

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
        key = 0
        for key in range(1,2):
            frontier = torch.from_numpy(
                np.fromfile(f'/data/log/_frontier_sample{key}_rank{rank}', dtype = np.int32)).to(rank)
            partition_map = torch.from_numpy(
                np.fromfile(f'/data/log/_partition_index_sample{key}_rank{rank}', dtype = np.int32)).to(rank)

            frontier =F.zerocopy_to_dgl_ndarray(frontier)
            partition_map = F.zerocopy_to_dgl_ndarray(partition_map)
            e1 = torch.cuda.Event(enable_timing=  True)
            e2 = torch.cuda.Event(enable_timing= True)
            for _ in range(1):
                e1.record()
                # frontier =F.zerocopy_to_dgl_ndarray(torch.arange(16, device = rank))
                # partition_map = F.zerocopy_to_dgl_ndarray(torch.arange(16, device = rank) % 4)
                num_partitions = 4
                scattered_array = _CAPI_getScatteredArrayObject\
                        (frontier, partition_map, num_partitions, rank, world_size)
                e2.record()
                e2.synchronize()
                if rank == 0:
                    print("Total time my api", e1.elapsed_time(e2)/1000)
                uq = F.zerocopy_from_dgl_ndarray(scattered_array.unique_array)
                feat = torch.ones((uq.shape[0],128), device = rank, requires_grad = True)
                #
                forward = Shuffle.apply( scattered_array, feat, rank, world_size)
                print(forward.sum())
                forward.sum().backward()
                      # ).backward()
                # print(feat.grad)
                # s1.synchronize()
                # print("shape ", forward.shape)
     #    Create all torch layers
     #Test forward and Backward Pass on this object

def run_naive(rank, world_size):
    ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)
    key = 0
    for key in range(1):

        frontier = torch.from_numpy(
            np.fromfile(f'/data/log/_frontier_sample{key}_rank{rank}', dtype = np.int32)).to(rank)
        partition_map = torch.from_numpy(
            np.fromfile(f'/data/log/_partition_index_sample{key}_rank{rank}', dtype = np.int32)).to(rank)

        e1 = torch.cuda.Event(enable_timing=  True)
        e2 = torch.cuda.Event(enable_timing= True)
        for _ in range(3):
            e1.record()

        # frontier = torch.arange(1000).to(rank)
    # partition_map = frontier % world_size
            num_partitions = 4
            local_idx_frontier_send = []
            idx_size = []
            to_send_frontier = []
            for i in range(4):
                idx = (torch.where(partition_map == i)[0])
                to_send_frontier.append(frontier[idx])
                idx_size.append(idx.shape[0])
                local_idx_frontier_send.append(idx)
            send_size = torch.tensor(idx_size).to(torch.int32).to(rank)
            recv_size = torch.empty(send_size.shape, dtype = torch.int32, device = rank)
            dist.all_to_all_single(recv_size, send_size)
            recv_frontiers = []
            for i in recv_size.tolist():
                recv_frontiers.append(torch.empty(i, dtype = torch.int32, device = rank))
            dist.all_to_all(recv_frontiers, to_send_frontier)
            unique_frontier = torch.unique(torch.cat(recv_frontiers, dim = 0))
            frontier_recieved_idx = []
            for frontier_l in recv_frontiers:
                frontier_recieved_idx.append(torch.searchsorted(unique_frontier, frontier_l))

            e2.record()
            ### All frontier data exchanged.
            input_features = torch.ones(unique_frontier.shape[0], 128, requires_grad= True).to(rank)
            out = ShuffleAlt.apply(input_features, local_idx_frontier_send, frontier_recieved_idx, rank, world_size, frontier.shape[0])
            # out.sum().backward()
            print("Out ", out.sum())
            e2.synchronize()
            if rank == 0:
                print(e1.elapsed_time(e2)/1000, "naive index creation time")
    #   Init distributed process group
#   get random frontier in range


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_proc, (world_size,),nprocs = world_size)
    mp.spawn(run_naive, (world_size,),nprocs = world_size)



