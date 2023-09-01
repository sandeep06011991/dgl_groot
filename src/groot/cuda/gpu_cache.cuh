//
// Created by juelin on 8/31/23.
//

#ifndef DGL_GPU_CACHE_CUH
#define DGL_GPU_CACHE_CUH
#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

#include "cuda_index_select.cuh"
#include "../block.h"
#include "../../runtime/cuda/cuda_common.h"
namespace dgl::groot
{
    class GpuCache {
    private:
        NDArray _cache_ids; // cached ids sorted by value (int32 / int64)
        NDArray _cache_idx; // cached ids sorted by value (int64)
        NDArray _cpu_feats;
        NDArray _gpu_feats;
        DGLContext _ctx;
        bool _initialized{false};
    public:
        GpuCache() = default;
        bool IsInitialized() {return _initialized;};
        void Init(NDArray cpu_feats, NDArray cache_ids) {
            LOG(INFO) << "Initializing GpuCache";
            CHECK_EQ(cache_ids->ctx.device_type, kDGLCUDA) << "cached ids must be on gpu";
            CHECK_EQ(cache_ids->ndim, 1) << "cached ids must have dimension 1";
            const auto [sorted_ids, sorted_idx] = dgl::aten::Sort(cache_ids, 0);
            _ctx = cache_ids->ctx;
            _cache_ids = sorted_ids;
            _cache_idx = dgl::aten::Range(0, _cache_ids->shape[0], 64, _ctx);
            _cpu_feats = cpu_feats;
            _gpu_feats = IndexSelect(_cpu_feats, _cache_ids, runtime::getCurrentCUDAStream());
            runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, runtime::getCurrentCUDAStream());
            _initialized = true;
        }
        // Input:
        // 1. query_ids : unsorted (int64 / int32)
        // Depends on:
        // 1. _cache_idx (int64)
        // 2. _cache_ids (int64 / int32)
        // return
        // 1. hit id (int64 / int32) : not very useful but return anyway for debugging
        // 2. hit id's index in the cache_ids (int64): used for loading from gpu_feat
        // 3. hit id's index in the query_ids (int64): used for writing cpu_feat to output buffer (int64)
        // 4. missed id's in the query_ids (int64 / int32) : used for loading from cpu_feat
        // 5. missed id's index in the query_ids (int64) : used for writing gpu_feat ot the output buffer (int64)
        // TODO: implement this in one kernel
        std::tuple<IdArray, IdArray, IdArray, IdArray, IdArray> Query(NDArray query_ids,
                                                                      std::shared_ptr<BlocksObject> blocksPtr) const;

        void IndexSelectWithNoCache(NDArray query_ids, std::shared_ptr<BlocksObject> blocksPtr, cudaStream_t uva_stream){
            IndexSelect(_cpu_feats, query_ids, blocksPtr->_feats, uva_stream);
        }

        void IndexSelectWithLocalCache(NDArray query_ids, std::shared_ptr<BlocksObject> blocksPtr, cudaStream_t gpu_stream, cudaStream_t uva_stream){
            auto [hit_id, hit_cidx, hit_qidx, missed_qid, missed_qidx] = Query(query_ids, blocksPtr);
            IndexSelect(_gpu_feats, hit_cidx, hit_qidx, blocksPtr->_feats, gpu_stream);
            IndexSelect(_cpu_feats, missed_qid, missed_qidx, blocksPtr->_feats, uva_stream);
            blocksPtr->_feats->shape[0] = query_ids->shape[0];
        };
    };
}

#endif //DGL_GPU_CACHE_CUH
