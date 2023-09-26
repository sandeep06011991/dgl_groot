from .._ffi.function import _init_api
_init_api("dgl.groot", __name__)

import dgl.backend as F
from torch import Tensor
from ..heterograph import DGLBlock

def init_groot_dataloader(rank: int, world_size: int, block_type: int, device_id: int, fanouts: list[int],
                    batch_size: int, num_redundant_layers: int, max_pool_size: int,
                    indptr: Tensor, indices: Tensor, feats: Tensor, labels: Tensor,
                    train_idx: Tensor, valid_idx: Tensor, test_idx: Tensor, partition_map: Tensor):
    if partition_map is None:
        partition_map = torch.randint(0, world_size, (indptr.shape[0] - 1,))
    if num_redundant_layers == len(fanouts):
        assert(block_type == 0)
    else:
        assert(block_type == 1 or block_type == 2)
        
    return _CAPI_InitDataloader(rank, world_size, block_type, device_id, 
                                fanouts, batch_size, num_redundant_layers, max_pool_size,
                                F.zerocopy_to_dgl_ndarray(indptr),
                                F.zerocopy_to_dgl_ndarray(indices),
                                F.zerocopy_to_dgl_ndarray(feats),
                                F.zerocopy_to_dgl_ndarray(labels),
                                F.zerocopy_to_dgl_ndarray(train_idx.to(device_id)),
                                F.zerocopy_to_dgl_ndarray(valid_idx.to(device_id)),
                                F.zerocopy_to_dgl_ndarray(test_idx.to(device_id)),
                                F.zerocopy_to_dgl_ndarray(partition_map.to(device_id)))

def init_groot_dataloader_cache(cache_idx: Tensor):
    _CAPI_InitCache(F.to_dgl_nd(cache_idx))
    
def get_batch(key: int, layers: int = 3):
    blocks = []
    for i in range(layers):
        gidx = _CAPI_GetBlock(key, i)
        block = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        blocks.insert(0, block)
        
    feat = _CAPI_GetFeat(key)
    labels = _CAPI_GetLabel(key)
    return blocks, F.zerocopy_from_dgl_ndarray(feat), F.zerocopy_from_dgl_ndarray(labels)

def sample_batch_async() -> int:
    return _CAPI_NextAsync()

def sample_batch_sync() -> int:
    return _CAPI_NextSync()