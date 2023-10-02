# What is the best format for communication between processes
# Torch distributed sending which can potentially speed up using nvlink
import torch as th
import torch.multiprocessing as mp
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

comm_map = {
    0:[1,2,3],
    1:[0,3,2],
    2:[3,0,1],
    3:[2,1,0]
}

comm_map = {
    0:[1,2,3,4,5,6,7],
    1:[0,3,2,5,4,7,6],
    2:[3,0,1,6,7,4,5],
    3:[2,1,0,7,6,5,4],
    4:[5,6,7,0,1,2,3],
    5:[4,7,6,1,0,3,2],
    6:[7,4,5,2,3,0,1],
    7:[6,5,4,3,2,1,0],
}


def shuffle_functional(device_id, send_dict, recv_dict, num_devices):
    input_splits = []
    input_tensors = []
    output_tensors = []
    output_splits = []
    torch.cuda.nvtx.range_push("all to all {}".format(device_id)) 
    for i in range(num_devices):
        if i== device_id:
            send_dict[i] = recv_dict[i].clone()
        input_splits.append(send_dict[i].shape[0])
        output_splits.append(recv_dict[i].shape[0])
        input_tensors.append(send_dict[i])
        output_tensors.append(recv_dict[i])
    send = torch.cat(input_tensors)
    recv = torch.cat(output_tensors)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e1.record()
    async_all = torch.distributed.all_to_all_single(recv, send, output_splits, input_splits, async_op = True)
    e2.record()
    e2.synchronize()
    async_all.wait()
    print("Total time ", e1.elapsed_time(e2)/1000)
    torch.cuda.nvtx.range_pop()
    s = 0
    torch.cuda.nvtx.range_push("merge {}".format(output_splits[num_devices - 1]))
    for i in range(num_devices):
        recv_dict[i] = recv[s : s + output_splits[i]]
        s = s + output_splits[i]
    torch.cuda.nvtx.range_pop()
    return async_all 


def using_dist_async(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    GB  = 1024 * 1024 * 1024
    GB =  1024 * 1024
    j = 0
    device_id = proc_id
    send_data = {}
    recv_data = {}

    num_tries = 10
    for k in range(num_tries):
        for i in range(n_gpus):
            send_data[i] = torch.ones(((int)(GB / 4) ,) ,device = proc_id) *   proc_id
            recv_data[i] = torch.rand(((int)(GB / 4) ,),device = proc_id)
        t1 = time.time()
        shuffle_functional(device_id, send_data, recv_data, n_gpus)
        torch.cuda.synchronize()
        t2 = time.time()
        print("Time ", t2-t1, "Bandwidth", ((n_gpus-1) * n_gpus * 1)/(t2-t1))
        for i in range(n_gpus):
            if i!=proc_id:
                print(recv_data[i][(int)(GB/4) - 1] == i * k)



'''
Single gpu transfer total time .4 secnds total, bandwitdh 2 gbps, .04 seconds per movement
two to four gpus transfer time  .4 - .8 seconds per gpu. bandwirdh 2 gps .06 - .08 seconds per iteration
'''
if __name__ == "__main__":
    n_gpus = 4
    procs = []
    # assert(False)
    test_functions = [using_dist_async]
    for f in test_functions:
        for proc_id in range(n_gpus):
            p = mp.Process(target=(f),
                           args=(proc_id, n_gpus))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
