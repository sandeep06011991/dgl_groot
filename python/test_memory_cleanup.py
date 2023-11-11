
import quiver
import time
import torch
# seperate into another process to force garbage collection of ipc
def func(rank):
    feat = torch.rand( 1032 * 1032, 1032)
    print("got feat")
    for i in range(4):
        quiver_feat = quiver.Feature(0, device_list=list(range(4)), \
                                     cache_policy="p2p_clique_replicate", device_cache_size="1032M")

        quiver_feat.from_cpu_tensor(feat)



if __name__ == "__main__":

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    #best_configuration()
    # all_experiments()

    quiver.init_p2p(device_list=list(range(4)))
    mp.spawn(func, (), join = True)
    torch.cuda.ipc_collect()
    import gc
    gc.collect()
    while True:
        pass

    print("returned")
    print("Start of sleeping")


