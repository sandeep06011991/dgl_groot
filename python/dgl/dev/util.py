from torch import randperm, device, Tensor
from torch.cuda.nvtx import range_pop, range_push
from dataclasses import dataclass

@dataclass
class SampleConfig:
    rank: int = 0
    world_size: int = 1
    num_partition: int = 1
    batch_size: int = 1024
    replace: bool = False
    mode: str = "uva"  # must be one of ["uva", "gpu"]
    reindex: bool = True  # whether to make the sampled vertex to be 0 indexed
    fanouts: list[int] = None
    drop_last: bool = False
    num_dp: int = 0  # k = number of leading DP (no-shuffle) layers for hybrid split

    @staticmethod
    def header():
        return ["rank", "world_size", "num_partition", "batch_size", "replace", "mode", "reindex", "fanouts"]

    def content(self):
        return [self.rank,
                self.world_size,
                self.num_partition,
                self.batch_size,
                self.replace,
                self.mode,
                self.reindex,
                self.fanouts
                ]


class IdxLoader:
    def __init__(self, d: device, target_idx: Tensor, batch_size: int, shuffle: bool, max_step_per_epoch: int):
        assert (target_idx.device == device("cpu"))
        assert(max_step_per_epoch * batch_size <= target_idx.shape[0])
        self.target_idx = target_idx
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_step_per_epoch = max_step_per_epoch
        self.cur_step = 0
        self.device = d
        self.nids = target_idx.to(self.device)
        
    def __iter__(self):
        self.cur_step = 0
        if self.shuffle:
            range_push("shuffle idx")
            idx = randperm(self.target_idx.shape[0])
            self.nids = self.target_idx[idx].to(self.device)
            range_pop()
        return self

    def __next__(self):
        if self.cur_step < self.max_step_per_epoch:
            range_push("get next seeds")
            seeds = self.nids[self.cur_step * self.batch_size: (self.cur_step + 1) * self.batch_size]
            self.cur_step += 1
            range_pop()
            return seeds
        else:
            raise StopIteration