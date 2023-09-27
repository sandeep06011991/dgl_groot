from torch.multiprocessing import Queue
import torch
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import time,datetime
from layers.shuffle_functional import *

class Shuffle(torch.autograd.Function):

    # from_sizes: Shapes exppected from other gpus.
    # To offsets that iwll be shuffled.
    @staticmethod
    def forward(ctx, remote_t, device_id, num_gpus, layer_id ,from_nds_size,\
                to_tensor_offset, shbuffs, barrier, async_dict):
        
        recv = []
        recv_g = []
        send_dict = []
        #new_remote = torch.empty((remote_t.shape), device = device_id)
        #remote_t.detach().copy_(new_remote, non_blocking = True)
        #remote_t  = new_remote
        remote_t = remote_t.detach()
        for i in range(num_gpus):
            # Do I do work allocation here ?
            if i == device_id:
                recv.append(torch.empty([0,*remote_t.shape[1:]], device = device_id, dtype = torch.int32))
                recv_g.append(torch.empty([0,*remote_t.shape[1:]], device = device_id, dtype = torch.int32))
                send_dict.append(None)
            else:
                # remote has the same shape as local
                recv.append(torch.empty((from_nds_size[i], *remote_t.shape[1:]) \
                    , device = device_id, dtype = torch.int32))
                recv_g.append(torch.empty((to_tensor_offset[i+1] - to_tensor_offset[i], *remote_t.shape[1:]) \
                    , device = device_id, dtype = torch.int32))
                send_dict.append(remote_t[to_tensor_offset[i]:to_tensor_offset[i+1]])
        if shbuffs is None:
            async_op = shuffle_functional(device_id, send_dict, recv,num_gpus)
        else:
            shuffle_functional_buffers(device_id, send_dict, recv, num_gpus, shbuffs, barrier)
        ctx.device_id = device_id
        ctx.layer_id = layer_id
        ctx.recv_g = recv_g
        ctx.num_gpus = num_gpus
        ctx.shbuffs = shbuffs
        ctx.barrier = barrier
        async_dict['async_op'] = async_op
        # torch.cuda.current_stream().synchronize()
        return tuple(recv)

    @staticmethod
    def backward(ctx, *grads):
        send_grads = [grad.detach() for grad in grads]
        device_id = ctx.device_id
        recv_g = ctx.recv_g
        layer_id = ctx.layer_id
        num_gpus = ctx.num_gpus
        shbuffs = ctx.shbuffs
        barriers = ctx.barrier
        if shbuffs is None:
            shuffle_functional(device_id,send_grads, recv_g,num_gpus)
        else:
            shuffle_functional_buffers(device_id, send_grads, recv_g, num_gpus, shbuffs, barriers)
        grads = []
        for i in range(num_gpus):
            if i!= device_id:
                grads.append(recv_g[i])
        remote_g = torch.cat(grads, dim = 0)
        # torch.cuda.current_stream().synchronize()
        return  remote_g, None, None, None, None, None, None, None, None


class ToySingle(torch.nn.Module):

    def __init__(self,  device_id):
        super(ToySingle, self).__init__()
        self.ll = torch.nn.Linear(10000,10000)
        self.device_id = device_id
        self.ll.weight = torch.nn.Parameter(
            torch.ones(self.ll.weight.shape))

    def forward(self, local_input, remote_input):
        local_input = self.ll(local_input)
        remote_input = self.ll(remote_input)
        to_offsets = [0]
        from_sizes = []
        for i in range(4):
            if i == self.device_id:
                to_offsets.append(to_offsets[-1])
                from_sizes.append(None)
            else:
                to_offsets.append(to_offsets[i] + 25)
                from_sizes.append(25)
        r1,r2,r3,r4 = Shuffle.apply(remote_input, self.device_id, 0,  from_sizes,\
                    to_offsets)
        r = [r1,r2,r3,r4]
        for i in range(4):
            if i == self.device_id:
                continue
            local_input += r[i]
        return local_input


# Not  a real correctness test. Just for me to know the shuffle works
# Hence not migrating
def test_single(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus
    pg = th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)

    model = ToySingle(proc_id).to(proc_id)
    model = DistributedDataParallel(model, device_ids = [proc_id])
    local = torch.ones((25,10000),requires_grad = True, device = proc_id)
    remote = torch.ones((75,10000), requires_grad = True, device = proc_id)

    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    for i in range(100):
        e1.record()
        out = model.forward(local, remote)
        out.sum().backward()
        e2.record()
        print("time",e1.elapsed_time(e2/1000))
    print(local.grad, remote.grad)

if __name__ == "__main__":
    # Create unit test which can handle  shuffling
    procs = []
    n_gpus = 4
    print("Launch multiple gpus")
    for proc_id in range(n_gpus):
        p = mp.Process(target=(test_single),
                       args=(proc_id, n_gpus))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
