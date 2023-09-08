//
// Created by juelin on 8/31/23.
//

#include "gpu_cache.cuh"
#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/set_operations.h>
namespace dgl::groot {

namespace impl {
/** @brief compute the feature data that is cached
 * @tparam IdType The type of node and indexes.
 * @param query_ids the set of ids to fetch feature data.
 * @param num_query_ids The number of ids to fetch.
 * @param ret_hit_mask 1 -> cached | 0: uncached
 * @param ret_num_hit The length of ret_hit_id
 * @param ret_num_missed The length of ret_missed_qid
 *
 * */
template <typename IdType>
__global__ void _CacheMask(const IdType *id_to_cidx, IdType *const ret_hit_mask,
                           const IdType *qids, int64_t num_qids) {
  int qidx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (qidx < num_qids) {
    IdType qid = qids[qidx];
    if (id_to_cidx[qid] < 0) {
      // cache miss
      ret_hit_mask[qidx] = 0;
    } else {
      // cache hit
      ret_hit_mask[qidx] = 1;
    }
    qidx += stride_x;
  };
}

template <typename IdType>
__global__ void _QueryIndex(const IdType * id_to_cidx, const IdType * hit_ptr,
                            const IdType * qids, int64_t num_qids,
                            IdType * const ret_hit_id, IdType * const ret_hit_cidx,
                            IdType * const ret_hit_qidx, IdType * const ret_missed_qidx,
                            IdType * const ret_missed_qid) {
  int qidx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;

  while (qidx < num_qids) {
    IdType qid = qids[qidx];
    IdType cidx = id_to_cidx[qid];
    IdType cur_num_hit = hit_ptr[qidx];
    IdType cur_num_missed = qidx - cur_num_hit;
    if (cidx >= 0) {
      // cache hit
      ret_hit_cidx[cur_num_hit] = cidx;
      ret_hit_qidx[cur_num_hit] = qidx;
      ret_hit_id[cur_num_hit] = qid;
    } else {
      // cache missed
      ret_missed_qid[cur_num_missed] = qid;
      ret_missed_qidx[cur_num_missed] = qidx;
    }
    qidx += stride_x;
  }
} // QueryIndex

template <typename IdType>
std::tuple<IdArray, IdArray, IdArray, IdArray, IdArray>
QueryIndex(NDArray id_to_cidx_ndarr, NDArray qid_ndarr,
           std::shared_ptr<BlocksObject> blocksPtr) {
  //  LOG(INFO) << "enter query index";
  auto ctx = qid_ndarr->ctx;
  auto stream = blocksPtr->_stream;
  int64_t num_qids = qid_ndarr->shape[0];
  NDArray hit_mask_ndarr =
      NDArray::Empty({num_qids + 1}, qid_ndarr->dtype, qid_ndarr->ctx);
  NDArray hit_ptr_ndarr =
      NDArray::Empty({num_qids + 1}, qid_ndarr->dtype, qid_ndarr->ctx);
  NDArray pinned_num_hit =
      NDArray::PinnedEmpty({1}, qid_ndarr->dtype, DGLContext{kDGLCPU, 0});

  IdType *qids = static_cast<IdType *>(qid_ndarr->data);
  IdType *id_to_cidx = static_cast<IdType *>(id_to_cidx_ndarr->data);
  IdType *ret_hit_mask = static_cast<IdType *>(hit_mask_ndarr->data);
  IdType *ret_hit_ptr = static_cast<IdType *>(hit_ptr_ndarr->data);
  constexpr int64_t blockDim{1024};
  constexpr int64_t gridDim{256};

  //  LOG(INFO) << "calling CacheMask";
  _CacheMask<IdType><<<gridDim, blockDim, 0, stream>>>(id_to_cidx, ret_hit_mask, qids, num_qids);
//  runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, stream);

  //  LOG(INFO) << "prefix sum";
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size,
                                          ret_hit_mask, ret_hit_ptr,
                                          num_qids + 1, stream));
  void *prefix_temp =
      runtime::DeviceAPI::Get(ctx)->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_temp, prefix_temp_size,
                                          ret_hit_mask, ret_hit_ptr,
                                          num_qids + 1, stream));
//  runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, stream);

  CUDA_CALL(cudaMemcpyAsync(pinned_num_hit->data, ret_hit_ptr + num_qids,
                            sizeof(IdType), cudaMemcpyDeviceToHost, stream));
//  runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, stream);

  auto &ret_hit_id_ndarr = blocksPtr->_query_buffer._hit_id;
  auto &ret_hit_cidx_ndarr = blocksPtr->_query_buffer._hit_cidx;
  auto &ret_hit_qidx_ndarr = blocksPtr->_query_buffer._hit_qidx;
  auto &ret_missed_qid_ndarr = blocksPtr->_query_buffer._missed_qid;
  auto &ret_missed_qidx_ndarr = blocksPtr->_query_buffer._missed_qidx;
  IdType *ret_hit_id = static_cast<IdType *>(ret_hit_id_ndarr->data);
  IdType *ret_hit_cidx = static_cast<IdType *>(ret_hit_cidx_ndarr->data);
  IdType *ret_hit_qidx = static_cast<IdType *>(ret_hit_qidx_ndarr->data);
  IdType *ret_missed_qid = static_cast<IdType *>(ret_missed_qid_ndarr->data);
  IdType *ret_missed_qidx = static_cast<IdType *>(ret_missed_qidx_ndarr->data);

  //  LOG(INFO) << "query index kernel";
  _QueryIndex<IdType><<<gridDim, blockDim, 0, stream>>>(
      id_to_cidx, ret_hit_ptr, qids, num_qids, ret_hit_id, ret_hit_cidx,
      ret_hit_qidx, ret_missed_qidx, ret_missed_qid);
  //  LOG(INFO) << "finished query index";

  runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, stream);
  runtime::DeviceAPI::Get(ctx)->FreeWorkspace(ctx, prefix_temp);

  int64_t _num_hit = static_cast<IdType *>(pinned_num_hit->data)[0];
  CHECK_LE(_num_hit, num_qids);
  int64_t _num_missed = num_qids - _num_hit;
//  LOG(INFO) << "num_qid=" << num_qids << " | hit=" << _num_hit << " | missed=" << _num_missed << " | hit_rate=" << 1.0 * _num_hit / num_qids;

  blocksPtr->_query_buffer._hit_id->shape[0] = _num_hit;
  blocksPtr->_query_buffer._hit_cidx->shape[0] = _num_hit;
  blocksPtr->_query_buffer._hit_qidx->shape[0] = _num_hit;
  blocksPtr->_query_buffer._missed_qid->shape[0] = _num_missed;
  blocksPtr->_query_buffer._missed_qidx->shape[0] = _num_missed;
  return std::make_tuple(
      blocksPtr->_query_buffer._hit_id, blocksPtr->_query_buffer._hit_cidx,
      blocksPtr->_query_buffer._hit_qidx, blocksPtr->_query_buffer._missed_qid,
      blocksPtr->_query_buffer._missed_qidx);
}
template std::tuple<IdArray, IdArray, IdArray, IdArray, IdArray>
    QueryIndex<int32_t>(NDArray, NDArray, std::shared_ptr<BlocksObject>);
template std::tuple<IdArray, IdArray, IdArray, IdArray, IdArray>
    QueryIndex<int64_t>(NDArray, NDArray, std::shared_ptr<BlocksObject>);

template <typename IdType> struct MaskFunctor {
  IdType *_cache_mask{nullptr};
  IdType *_output{nullptr};
  int64_t _num_qid{0};
  MaskFunctor(IdType *cache_mask, IdType *output, int64_t num_qid)
      : _cache_mask{cache_mask}, _output{output}, _num_qid{num_qid} {};

  template <typename Tuple> __device__ void operator()(Tuple t) {
    IdType qidx = thrust::get<0>(t);
    IdType qid = thrust::get<1>(t);
    if (_cache_mask[qidx] >= 0) {
      _output[qidx] = 1;
    } else {
      _output[qidx] = 0;
    }
    if (qidx == 0) {
      _output[_num_qid] = 0; // make prefix sum works
    }
  }
};

template class MaskFunctor<int32_t>;
template class MaskFunctor<int64_t>;

template <typename IdType> struct IndexFunctor {
  IdType *_cache_mask{nullptr};
  IdType *_hit_id{nullptr};
  IdType *_hit_cidx{nullptr};
  IdType *_hit_qidx{nullptr};
  IdType *_missed_qid{nullptr};
  IdType *_missed_qidx{nullptr};
  IdType *_indptr{nullptr};

  IndexFunctor(IdType *cache_mask, IdType *hit_id, IdType *hit_cidx,
               IdType *hit_qidx, IdType *missed_qid, IdType *missed_qidx,
               IdType *indptr)
      : _cache_mask{cache_mask}, _hit_id{hit_id}, _hit_cidx{hit_cidx},
        _hit_qidx{hit_qidx}, _missed_qid{missed_qid}, _missed_qidx{missed_qidx},
        _indptr{indptr} {};

  template <typename Tuple> __device__ void operator()(Tuple t) {
    IdType qidx = thrust::get<0>(t);
    IdType qid = thrust::get<1>(t);
    IdType cidx = _cache_mask[qid];
    IdType cur_num_hit = _indptr[qidx];
    IdType cur_num_missed = qidx - cur_num_hit;
    if (cidx >= 0) {
      // cache hit
      _hit_cidx[cur_num_hit] = cidx;
      _hit_qidx[cur_num_hit] = qidx;
      _hit_id[cur_num_hit] = qid;
    } else {
      // cache missed
      _missed_qid[cur_num_missed] = qid;
      _missed_qidx[cur_num_missed] = qidx;
    }
  }
};
template class IndexFunctor<int32_t>;
template class IndexFunctor<int64_t>;

} // namespace impl

std::tuple<IdArray, IdArray, IdArray, IdArray, IdArray>
GpuCache::QueryV2(NDArray query_ids,
                  std::shared_ptr<BlocksObject> blocksPtr) const {
  nvtx3::scoped_range query{"GpuCacheQueryV2"};
  ATEN_ID_TYPE_SWITCH(query_ids->dtype, IdType, {
    return impl::QueryIndex<IdType>(_id_to_idx, query_ids, blocksPtr);
  });

  //  ATEN_ID_TYPE_SWITCH(query_ids->dtype, IdType, {
  //    using DevIdPtr = thrust::device_ptr<IdType>;
  //    int64_t num_qid = query_ids->shape[0];
  //    DevIdPtr qid_start = DevIdPtr(query_ids.Ptr<IdType>());
  //    DevIdPtr qid_end = DevIdPtr(query_ids.Ptr<IdType>() + num_qid);
  //    NDArray indptr_arr =
  //        NDArray::Empty({num_qid}, query_ids->dtype, query_ids->ctx);
  //    NDArray num_hit_buf = NDArray::Empty({1}, query_ids->dtype,
  //    DGLContext{kDGLCPU, 0}); IdType * indptr = indptr_arr.Ptr<IdType >();
  //    auto exec = thrust::cuda::par.on(blocksPtr->_stream);
  //    thrust::counting_iterator<IdType> first{0};
  //    thrust::counting_iterator<IdType> end = first + num_qid;
  //
  //    IdType * cache_mask = _id_to_idx.Ptr<IdType>();
  //    auto mask_functor = impl::MaskFunctor<IdType>(cache_mask, indptr,
  //    num_qid); thrust::for_each(
  //        exec, thrust::make_zip_iterator(first, qid_start),
  //        thrust::make_zip_iterator(end, qid_end), mask_functor);
  //
  //
  //    DevIdPtr indptr_start = DevIdPtr(indptr);
  //    DevIdPtr indptr_end = DevIdPtr(indptr + num_qid);
  //    thrust::exclusive_scan(exec, indptr_start, indptr_end, indptr_start);
  //
  //    IdType *hit_id = blocksPtr->_query_buffer._hit_id.Ptr<IdType >();
  //    IdType *hit_cidx = blocksPtr->_query_buffer._hit_cidx.Ptr<IdType>();
  //    IdType *hit_qidx = blocksPtr->_query_buffer._hit_qidx.Ptr<IdType>();
  //    IdType *missed_qid = blocksPtr->_query_buffer._missed_qid.Ptr<IdType>();
  //    IdType *missed_qidx =
  //    blocksPtr->_query_buffer._missed_qidx.Ptr<IdType>(); auto index_functor
  //    = impl::IndexFunctor<IdType>(cache_mask, hit_id, hit_cidx, hit_qidx,
  //    missed_qid, missed_qidx, indptr); thrust::for_each(
  //        exec, thrust::make_zip_iterator(first, qid_start),
  //        thrust::make_zip_iterator(end, qid_end), index_functor);
  //    CUDA_CALL(cudaMemcpyAsync(num_hit_buf->data, indptr + num_qid,
  //    sizeof(IdType), cudaMemcpyDeviceToHost, blocksPtr->_stream));
  //    runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, blocksPtr->_stream);
  //    IdType _num_hit = static_cast<IdType*>(num_hit_buf->data)[0];
  //    IdType _num_missed = num_qid - _num_hit;
  //    blocksPtr->_query_buffer._hit_id->shape[0] = _num_hit;
  //    blocksPtr->_query_buffer._hit_cidx->shape[0] = _num_hit;
  //    blocksPtr->_query_buffer._hit_qidx->shape[0] = _num_hit;
  //    blocksPtr->_query_buffer._missed_qid->shape[0] = _num_missed;
  //    blocksPtr->_query_buffer._missed_qidx->shape[0] = _num_missed;
  //  });
  //
  //  return std::make_tuple(blocksPtr->_query_buffer._hit_id,
  //                         blocksPtr->_query_buffer._hit_cidx,
  //                         blocksPtr->_query_buffer._hit_qidx,
  //                         blocksPtr->_query_buffer._missed_qid,
  //                         blocksPtr->_query_buffer._missed_qidx);
}

std::tuple<IdArray, IdArray, IdArray, IdArray, IdArray>
GpuCache::QueryV1(NDArray query_ids,
                  std::shared_ptr<BlocksObject> blocksPtr) const {
  nvtx3::scoped_range query{"GpuCacheQueryV1"};
  CHECK_EQ(query_ids->ndim, 1) << "queried ids must have dimension 1";
  CHECK_EQ(query_ids->ctx.device_type, kDGLCUDA) << "queried ids must on gpu";
  CHECK_EQ(query_ids->ctx.device_id, _cache_ids->ctx.device_id)
      << "queried ids must be on same device";
  CHECK_EQ(query_ids->dtype, _cache_ids->dtype)
      << "queried ids must have same dtype with cached ids";
  auto [sorted_query_ids, sorted_query_idx] = dgl::aten::Sort(query_ids, 0);
  auto &ret_hit_id = blocksPtr->_query_buffer._hit_id;
  auto &ret_hit_cidx = blocksPtr->_query_buffer._hit_cidx;
  auto &ret_hit_qidx = blocksPtr->_query_buffer._hit_qidx;
  auto &ret_missed_qid = blocksPtr->_query_buffer._missed_qid;
  auto &ret_missed_qidx = blocksPtr->_query_buffer._missed_qidx;
  ATEN_ID_TYPE_SWITCH(_cache_ids->dtype, IdType, {
    using DevIdPtr = thrust::device_ptr<IdType>;
    using DevIdxPtr = thrust::device_ptr<int64_t>;
    DevIdPtr cid_start = DevIdPtr(_cache_ids.Ptr<IdType>());
    DevIdPtr cid_end =
        DevIdPtr(_cache_ids.Ptr<IdType>() + _cache_ids->shape[0]);
    DevIdPtr qid_start = DevIdPtr(sorted_query_ids.Ptr<IdType>());
    DevIdPtr qid_end =
        DevIdPtr(sorted_query_ids.Ptr<IdType>() + sorted_query_ids->shape[0]);
    DevIdxPtr cidx_range = DevIdxPtr(_cache_idx.Ptr<int64_t>());
    DevIdxPtr qidx_start = DevIdxPtr(sorted_query_idx.Ptr<int64_t>());
    DevIdPtr hit_id_start = DevIdPtr(ret_hit_id.Ptr<IdType>());
    DevIdPtr missed_qid_start = DevIdPtr(ret_missed_qid.Ptr<IdType>());
    DevIdxPtr hit_cidx_start = DevIdxPtr(ret_hit_cidx.Ptr<int64_t>());
    DevIdxPtr hit_qidx_start = DevIdxPtr(ret_hit_qidx.Ptr<int64_t>());
    DevIdxPtr missed_qidx_start = DevIdxPtr(ret_missed_qidx.Ptr<int64_t>());
    auto exec = thrust::cuda::par.on(blocksPtr->_stream);
    auto [hit_id_end, hit_cidx_end] = thrust::set_intersection_by_key(
        exec, cid_start, cid_end, qid_start, qid_end, cidx_range, hit_id_start,
        hit_cidx_start);

    auto [hit_id_end_v2, hit_qidx_end] = thrust::set_intersection_by_key(
        exec, qid_start, qid_end, cid_start, cid_end, qidx_start, hit_id_start,
        hit_qidx_start);

    auto [missed_qid_end, missed_qidx_end] = thrust::set_difference_by_key(
        exec, qid_start, qid_end, cid_start, cid_end, qidx_start, cidx_range,
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
    return std::make_tuple(ret_hit_id, ret_hit_cidx, ret_hit_qidx,
                           ret_missed_qid, ret_missed_qidx);
  });
};
} // namespace dgl::groot
