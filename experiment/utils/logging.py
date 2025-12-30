from torch import Tensor
# from .timer import Timer, CudaTimer

def get_tensor_size(intensor: Tensor):
    num_bytes = intensor.element_size() * intensor.nelement()
    return f"{round(num_bytes / 1e9, 1)}GB"

def log_step(rank: int, epoch: str | int, step: int, step_per_epoch: int, timer):
    if rank == 0 and (step % step_per_epoch) % 100 == 0:
        cur_step = step % step_per_epoch
        if cur_step == 0:
            cur_step = step_per_epoch
        print(f"{epoch=} {cur_step=} / {step_per_epoch} time={timer.duration()} secs", flush=True)