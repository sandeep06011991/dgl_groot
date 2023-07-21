from .. import backend as F
from .._ffi.function import _init_api
import torch
from .. import utils 


    # >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15])
    # >>> dataloader = dgl.dataloading.DataLoader(
    # ...     g, train_nid, sampler,
    # ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    # >>> for input_nodes, output_nodes, blocks in dataloader:
    # ...     train_on(blocks)
    # class GRootSampler(NeighborSampler):

    #     
    # def sample_blocks(self, g, seed_nodes, exclude_eids=None):
    #         output_nodes = seed_nodes
    #         blocks = []
    #         for fanout in reversed(self.fanouts):
    #             if mode = "REDUNDANT":
    #                 frontier = g.sample_neighbors(
    #                     seed_nodes,
    #                     fanout,
    #                     edge_dir=self.edge_dir,
    #                     prob=self.prob,
    #                     replace=self.replace,
    #                     output_device=self.output_device,
    #                     exclude_edges=exclude_eids,
    #                 )
    #             if mode == "src_to_dest":
    #             if mode == "dest_to_src":
    #             eid = frontier.edata[EID]
    #             block = to_block(frontier, seed_nodes)
    #             block.edata[EID] = eid
    #             seed_nodes = block.srcdata[NID]
    #             blocks.insert(0, block)

    #         return seed_nodes, output_nodes, blocks

def ScatterArray(input_array : torch.Tensor, 
        partition_map: torch.Tensor, num_partitions: int):
    print("check recieved")
    print(_CAPI_testffi())
    # Call CAPI print revieved arguments and return dummy values.
    pass 

def test(a : torch.tensor, b: torch.tensor, N: int):
    # print(hex(a.data_ptr()
    print(torch.device(a.device),)
    assert(torch.device(a.device) == torch.device(0))
    a = F.zerocopy_to_dgl_ndarray(a)
    b = F.zerocopy_to_dgl_ndarray(b)
    return _CAPI_ScatterObjectCreate(a,b,N)
    # Examples 
    # torch.Tensor([1,2,3,6,7,8,9], [0,1,3,1,3,2], 4)
    # returns (scattered_arrays)
    pass 



def wrap_function():
    a = _CAPI_testffi()
    print(a)

_init_api("dgl.groot", __name__)
