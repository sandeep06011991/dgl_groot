import torch
import numpy as np
from dgl.utils import pin_memory_inplace, gather_pinned_tensor_rows
import time, os
import torch_quiver as torch_qv
import quiver

def test_zerocopy_gather():
    rank = 0
    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600

    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(rank)

    host_tensor = np.random.randint(0,
                                    high=10,
                                    size=(2 * NUM_ELEMENT, FEATURE_DIM))
    print("Tensor size", NUM_ELEMENT * 2 * FEATURE_DIM * 4/ 1024 ** 3)
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    print("host data size", host_tensor.size * 4 // 1024 // 1024, "MB")

    device_indices = indices.to(rank)

    ############################
    # define a quiver.Feature
    ###########################
    t = pin_memory_inplace(tensor)
    feature = tensor
    ####################
    # Indexing
    ####################
    res = gather_pinned_tensor_rows(feature, device_indices)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)

    start = time.time()
    e1.record()
    res = gather_pinned_tensor_rows(feature, device_indices)
    e2.record()
    e2.synchronize()
    consumed_time = time.time() - start
    consumed_time = e1.elapsed_time(e2)/1000
    res = res.cpu().numpy()
    feature_gt = tensor[indices].numpy()
    print("Correctness Check : ", np.array_equal(res, feature_gt))
    print(
        f"Process {os.getpid()}: TEST SUCCEED!, With Memory Bandwidth = {res.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s"
    )

def test_feature_quiver_basic():
    rank = 0

    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600

    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(rank)

    host_tensor = np.random.randint(0,
                                    high=10,
                                    size=(2 * NUM_ELEMENT, FEATURE_DIM))
    print("Tensor size", NUM_ELEMENT * 2 * FEATURE_DIM * 4/ 1024 ** 3)
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    print("host data size", host_tensor.size * 4 // 1024 // 1024, "MB")

    device_indices = indices.to(rank)

    ############################
    # define a quiver.Feature
    ###########################
    feature = quiver.Feature(rank=rank,
                             device_list=[0, 1, 2, 3],
                             device_cache_size="0.01G",
                             cache_policy="p2p_clique_replicate")
    feature.from_cpu_tensor(tensor)

    ####################
    # Indexing
    ####################
    res = feature[device_indices]
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)

    start = time.time()
    e1.record()
    res = feature[device_indices]
    e2.record()
    e2.synchronize()
    consumed_time = time.time() - start
    consumed_time = e1.elapsed_time(e2)/1000
    res = res.cpu().numpy()
    feature_gt = tensor[indices].numpy()
    print("Correctness Check : ", np.array_equal(res, feature_gt))
    print(
        f"Process {os.getpid()}: TEST SUCCEED!, With Memory Bandwidth = {res.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s"
    )


import torch_quiver as torch_qv
import multiprocessing as mp

if __name__ == "__main__":
    test_zerocopy_gather()
    # mp.set_start_method("spawn")
    # torch_qv.init_p2p([0, 1, 2, 3])
    # test_feature_quiver_basic()
