import torch, quiver, dgl
from dgl import create_block
def partition_ids(rank: int, world_size: int, nids: torch.Tensor) -> torch.Tensor:
    step = int(nids.shape[0] / world_size)
    start_idx = rank * step
    end_idx = start_idx + step
    loc_ids = nids[start_idx : end_idx]
    return loc_ids.to(rank)

class QuiverGraphSageSampler():
    def __init__(self, sampler: quiver.pyg.GraphSageSampler):
        self.sampler = sampler
    
    def sample_dgl(self, seeds):
        """Sample k-hop neighbors from input_nodes

        Args:
            input_nodes (torch.LongTensor): seed nodes ids to sample from
        Returns:
            Tuple: Return results are the same with Dgl's sampler
            1. input_ndoes # to extract features
            2. output_nodes # to prefict label
            3. blocks # dgl blocks
        """
        self.sampler.lazy_init_quiver()
        adjs = []
        nodes = seeds

        for size in self.sampler.sizes:
            out, cnt = self.sampler.sample_layer(nodes, size)
            frontier, row_idx, col_idx = self.sampler.reindex(nodes, out, cnt)
            block = create_block(('coo', (col_idx, row_idx)), num_dst_nodes=nodes.shape[0], num_src_nodes=frontier.shape[0], device=self.sampler.device)
            adjs.append(block)
            nodes = frontier
        return nodes, seeds, adjs[::-1]
    
class QuiverDglSageSample():
    def __init__(self, 
                 rank: int,
                 world_size: int,
                 batch_size: int, 
                 nids:torch.Tensor, 
                 sampler: quiver.pyg.GraphSageSampler,
                 shuffle=True,
                 partition=True):
        self.rank = rank
        if partition:
            self.nids = partition_ids(rank, world_size, nids)
        else:
            self.nids = nids.to(rank) # train_nids
        self.cur_idx = 0
        self.max_idx = nids.shape[0]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sampler = QuiverGraphSageSampler(sampler)     
        # self.sampler = sampler    

    def __iter__(self):
        self.cur_idx = 0
        if self.shuffle:
            dim = 0
            idx = torch.randperm(self.nids.shape[dim]).to(self.rank)
            self.nids = self.nids[idx]
        return self

    def __next__(self):
        if self.cur_idx < self.max_idx:
            seeds = self.nids[self.cur_idx : self.cur_idx + self.batch_size]
            self.cur_idx += self.batch_size
            return self.sampler.sample_dgl(seeds)
        else:
            raise StopIteration