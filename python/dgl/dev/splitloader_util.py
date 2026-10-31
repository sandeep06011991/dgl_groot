import torch
from dgl.backend import to_dgl_nd, from_dgl_nd
from torch import Tensor
from ..heterograph import DGLBlock
from .._ffi.function import _init_api
_init_api("dgl.dev", __name__)
from torch.cuda import nvtx

def SetRank(rank: int, world_size: int):
    _CAPI_Split_SetRank(rank, world_size)

def GetUniqueId():
    return from_dgl_nd(_CAPI_GetUniqueId())

def InitNccl(rank: int,  nrank : int, unique_id: Tensor):
    _CAPI_Split_InitNccl(rank, nrank, to_dgl_nd(unique_id))

def SetPartitionMap(num_partition: int, partition_map: Tensor):
    _CAPI_Split_SetPartitionMap(num_partition, to_dgl_nd(partition_map))

def SetGraph(indptr: Tensor, indices: Tensor, data: Tensor = None) -> None:
    if data == None:
        data = Tensor([])
    _CAPI_Split_SetGraph(to_dgl_nd(indptr),
                   to_dgl_nd(indices),
                   to_dgl_nd(data))

def SetFanout(fanout: list[int]) -> None:
    _CAPI_Split_SetFanout(fanout)

def SetNumDP(num_dp: int) -> None:
    # k = number of leading DP (no-shuffle) layers; sampler builds no
    # ScatteredArray for sampler-layers [0, num_dp).
    _CAPI_Split_SetNumDP(num_dp)

def SampleBatch(seeds: Tensor, replace : bool = False) -> int:
    return _CAPI_Split_SampleBatch(to_dgl_nd(seeds), replace)

# def UseBitmap(use_bitmap: bool) -> None:
#     return _CAPI_Split_UseBitmap(use_bitmap)

def PartitionCSR(indptr: Tensor, indices: Tensor, flag: Tensor) -> (Tensor, Tensor):
    ret = _CAPI_PartitionCSR(to_dgl_nd(indptr), to_dgl_nd(indices), to_dgl_nd(flag))
    out_indptr = from_dgl_nd(ret(0))
    out_indices = from_dgl_nd(ret(1))
    return (out_indptr, out_indices)

def GetBlockData(batch_id: int, layer: int):
    data = _CAPI_Split_GetBlockData(batch_id, layer)
    return from_dgl_nd(data)

def GetBlocks(batch_id: int, reindex: bool = True, layers: int = 3, edge_data: bool = False, num_dp: int = 0) -> (Tensor, Tensor, list[DGLBlock]):
    input_node = _CAPI_Split_GetInputNode(batch_id)
    input_node = from_dgl_nd(input_node)
    output_node = _CAPI_Split_GetOutputNode(batch_id)
    output_node = from_dgl_nd(output_node)
    blocks = []
    for layer in range(layers):
        nvtx.range_push("capi get block")
        gidx = _CAPI_Split_GetBlock(batch_id, layer, reindex)
        nvtx.range_pop()

        nvtx.range_push("dgl create block")
        block = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        # sampler-layers [0, num_dp) are data-parallel: no ScatteredArray was
        # built for them, so there is nothing to shuffle (scattered_src=None).
        if layer >= num_dp:
            block.scattered_src = _CAPI_GetBlockScatteredSrc(batch_id, layer)
        else:
            block.scattered_src = None
        nvtx.range_pop()

        nvtx.range_push("dgl get block data")
        if edge_data:
            block.edata["_ID"] = GetBlockData(batch_id, layer)
        nvtx.range_pop()

        blocks.insert(0, block)
    return input_node, output_node, blocks

def GetFeature(batch_id: int)->Tensor:
    return from_dgl_nd(_CAPI_Split_GetFeature(batch_id))

def InitFeatloader(pinned_feat: Tensor, cached_ids: Tensor):
    _CAPI_Split_InitFeatloader(to_dgl_nd(pinned_feat), to_dgl_nd(cached_ids))