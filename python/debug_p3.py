from torch import multiprocessing as mp
import os
import torch
import torch.distributed as dist

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_exit():
    dist.destroy_process_group()


def test(rank):
    ddp_setup(rank, 4)

    b = [rank * torch.ones(100000, dtype = torch.int32, device = rank) for i in range(4)]
    dist.all_gather(tensor = b[rank],tensor_list =  b)
    if rank == 1:
        print(b)
    #dist.barrier(device_ids=[i for i in range(4 )])
    print("barrier cleared")
    ddp_exit()
    print("All done")
if __name__ == "__main__":
    mp.spawn(test, args=(),
          nprocs=4, daemon=True, join= True)
