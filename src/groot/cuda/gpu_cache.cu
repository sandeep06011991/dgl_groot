//
// Created by juelin on 8/31/23.
//

#include "gpu_cache.cuh"
#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/set_operations.h>
namespace dgl::groot {
    std::tuple<IdArray, IdArray, IdArray, IdArray, IdArray> GpuCache::Query(NDArray query_ids,
                                                                                    std::shared_ptr<BlocksObject> blocksPtr) const {
        nvtx3::scoped_range query{"GpuCacheQuery"};
        CHECK_EQ(query_ids->ndim, 1) << "queried ids must have dimension 1";
        CHECK_EQ(query_ids->ctx.device_type, kDGLCUDA) << "queried ids must on gpu";
        CHECK_EQ(query_ids->ctx.device_id, _cache_ids->ctx.device_id) << "queried ids must be on same device";
        CHECK_EQ(query_ids->dtype, _cache_ids->dtype) << "queried ids must have same dtype with cached ids";
        auto [sorted_query_ids, sorted_query_idx] = dgl::aten::Sort(query_ids, 0);
        auto& ret_hit_id = blocksPtr->_query_buffer._hit_id;
        auto& ret_hit_cidx = blocksPtr->_query_buffer._hit_cidx;
        auto& ret_hit_qidx = blocksPtr->_query_buffer._hit_qidx;
        auto& ret_missed_qid = blocksPtr->_query_buffer._missed_qid;
        auto& ret_missed_qidx = blocksPtr->_query_buffer._missed_qidx;
        ATEN_ID_TYPE_SWITCH(_cache_ids->dtype, IdType, {
            using DevIdPtr = thrust::device_ptr<IdType>;
            using DevIdxPtr = thrust::device_ptr<int64_t >;
            DevIdPtr cid_start = DevIdPtr (_cache_ids.Ptr<IdType>());
            DevIdPtr cid_end = DevIdPtr(_cache_ids.Ptr<IdType>() + _cache_ids->shape[0]);
            DevIdPtr qid_start = DevIdPtr (sorted_query_ids.Ptr<IdType>());
            DevIdPtr qid_end = DevIdPtr (sorted_query_ids.Ptr<IdType>() + sorted_query_ids->shape[0]);
            DevIdxPtr cidx_range = DevIdxPtr(_cache_idx.Ptr<int64_t >());
            DevIdxPtr qidx_start = DevIdxPtr (sorted_query_idx.Ptr<int64_t >());
            DevIdPtr hit_id_start = DevIdPtr(ret_hit_id.Ptr<IdType >());
            DevIdPtr missed_qid_start = DevIdPtr (ret_missed_qid.Ptr<IdType>());
            DevIdxPtr hit_cidx_start = DevIdxPtr(ret_hit_cidx.Ptr<int64_t >());
            DevIdxPtr hit_qidx_start = DevIdxPtr(ret_hit_qidx.Ptr<int64_t >());
            DevIdxPtr missed_qidx_start = DevIdxPtr (ret_missed_qidx.Ptr<int64_t>());
            
            auto [hit_id_end, hit_cidx_end] = thrust::set_intersection_by_key(
                                                                              cid_start, cid_end,
                                                                              qid_start, qid_end,
                                                                              cidx_range,
                                                                              hit_id_start, hit_cidx_start);

            auto [hit_id_end_v2, hit_qidx_end] = thrust::set_intersection_by_key(
                                                                                 qid_start, qid_end,
                                                                                 cid_start, cid_end,
                                                                                 qidx_start,
                                                                                 hit_id_start, hit_qidx_start);
            
            auto [missed_qid_end, missed_qidx_end] = thrust::set_difference_by_key(
                                                                                   qid_start, qid_end,
                                                                                   cid_start, cid_end,
                                                                                   qidx_start, cidx_range,
                                                                                   missed_qid_start, missed_qidx_start);
            int64_t hit_id_size = hit_id_end - hit_id_start;
            int64_t hit_cidx_size = hit_cidx_end - hit_cidx_start;
            int64_t hit_qidx_size = hit_qidx_end - hit_qidx_start;
            int64_t missed_qid_size = missed_qid_end - missed_qid_start;
            int64_t missed_qidx_size = missed_qidx_end - missed_qidx_start;

            CHECK_EQ(hit_id_size, hit_cidx_size);
            CHECK_EQ(hit_id_end_v2, hit_id_end);
            CHECK_EQ(hit_cidx_size, hit_qidx_size);
            CHECK_EQ(missed_qid_size, missed_qidx_size);
            CHECK_EQ(hit_id_size + missed_qid_size, sorted_query_ids->shape[0]);
            
            ret_hit_cidx->shape[0] = hit_cidx_size;
            ret_hit_id->shape[0] = hit_id_size;
            ret_hit_qidx->shape[0] = hit_qidx_size;
            ret_missed_qidx->shape[0] = missed_qidx_size;
            ret_missed_qid->shape[0] = missed_qid_size;

            // ret_missed_qid are in the same order of the sorted version
            // need to convert them back up unsorted version
            IndexSelect(query_ids, ret_missed_qidx, ret_missed_qid, blocksPtr->_stream);
            runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, blocksPtr->_stream);
            return std::make_tuple(ret_hit_id, ret_hit_cidx, ret_hit_qidx, ret_missed_qid, ret_missed_qidx);
        });
    };
}
