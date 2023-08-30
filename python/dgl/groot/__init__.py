from .._ffi.function import _init_api
_init_api("dgl.groot", __name__)

import dgl.backend as F
from torch import Tensor
from ..heterograph import DGLBlock

def init_groot_dataloader(rank: int, world_size: int, device_id: int, fanouts: list[int],
                    batch_size: int, max_pool_size: int,
                    indptr: Tensor, indices: Tensor, feats: Tensor, labels: Tensor,
                    train_idx: Tensor, valid_idx: Tensor, test_idx: Tensor):
    
    return _CAPI_InitDataloader(rank, world_size, device_id, 
                                fanouts, batch_size, max_pool_size,
                                F.zerocopy_to_dgl_ndarray(indptr),
                                F.zerocopy_to_dgl_ndarray(indices),
                                F.zerocopy_to_dgl_ndarray(feats),
                                F.zerocopy_to_dgl_ndarray(labels),
                                F.zerocopy_to_dgl_ndarray(train_idx.to(device_id)),
                                F.zerocopy_to_dgl_ndarray(valid_idx.to(device_id)),
                                F.zerocopy_to_dgl_ndarray(test_idx.to(device_id)))

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