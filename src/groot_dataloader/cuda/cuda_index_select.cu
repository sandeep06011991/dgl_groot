//
// Created by juelin on 8/15/23.
//

#include "cuda_index_select.cuh"
#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include <assert.h>
#include "../../runtime/cuda/cuda_common.h"
#include "../../array/cuda/utils.h"
namespace dgl {
namespace groot
{
namespace impl {

template <typename DType, typename IdType>
__global__ void IndexSelectSingleKernel(
    const DType* array, const IdType* index, const int64_t length,
    const int64_t arr_len, DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    assert(index[tx] >= 0 && index[tx] < arr_len);
    out[tx] = array[index[tx]];
    tx += stride_x;
  }
}

template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernel(
    const DType* const array, const int64_t num_feat, const IdType* const index,
    const int64_t length, const int64_t arr_len, DType* const out) {
  int64_t out_row = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = index[out_row];
    assert(in_row >= 0 && in_row < arr_len);
    while (col < num_feat) {
      out[out_row * num_feat + col] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    out_row += stride;
  }
}

template <typename DType, typename IdType>
__global__ void IndexScatterSingleKernel(
    const DType* array, const IdType* index, const int64_t length,
    const int64_t arr_len, DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    assert(index[tx] >= 0 && index[tx] < arr_len);
    out[index[tx]] = array[tx];
    tx += stride_x;
  }
}

template <typename DType, typename IdType>
__global__ void IndexScatterMultiKernel(
    const DType* const array, const int64_t num_feat, const IdType* const index,
    const int64_t length, const int64_t arr_len, DType* const out) {
  int64_t in_row = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (in_row < length) {
    int64_t col = threadIdx.x;
    const int64_t out_row = index[in_row];
    assert(out_row >= 0 && out_row < arr_len);
    while (col < num_feat) {
      out[out_row * num_feat + col] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    in_row += stride;
  }
}

}  // namespace impl
template <typename DType, typename IdType>
NDArray IndexSelect(NDArray array, IdArray index, cudaStream_t stream) {
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  int64_t num_feat = 1;
  std::vector<int64_t> shape{len};
  for (int d = 1; d < array->ndim; ++d) {
    num_feat *= array->shape[d];
    shape.emplace_back(array->shape[d]);
  }

  // use index->ctx for pinned array
  NDArray ret = NDArray::Empty(shape, array->dtype, index->ctx);
  if (len == 0 || arr_len * num_feat == 0) return ret;
  DType* ret_data = static_cast<DType*>(ret->data);

  const DType* array_data = static_cast<DType*>(cuda::GetDevicePointer(array));
  const IdType* idx_data = static_cast<IdType*>(index->data);
  if (num_feat == 1) {
    const int nt = cuda::FindNumThreads(len);
    const int nb = (len + nt - 1) / nt;
    CUDA_KERNEL_CALL(
        impl::IndexSelectSingleKernel, nb, nt, 0, stream, array_data, idx_data, len,
        arr_len, ret_data);
  } else {
    dim3 block(256, 1);
    while (static_cast<int64_t>(block.x) >= 2 * num_feat) {
      block.x /= 2;
      block.y *= 2;
    }
    const dim3 grid((len + block.y - 1) / block.y);
    CUDA_KERNEL_CALL(
        impl::IndexSelectMultiKernel, grid, block, 0, stream, array_data, num_feat,
        idx_data, len, arr_len, ret_data);
  }
  return ret;
}
template NDArray IndexSelect<int8_t , int32_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<int8_t , int64_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<int16_t , int32_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<int16_t , int64_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<int32_t, int32_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<int32_t, int64_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<int64_t, int32_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<int64_t, int64_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<__half, int32_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<__half, int64_t>(
    NDArray, IdArray, cudaStream_t stream);
#if BF16_ENABLED
template NDArray IndexSelect<__nv_bfloat16, int32_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<__nv_bfloat16, int64_t>(
    NDArray, IdArray, cudaStream_t stream);
#endif  // BF16_ENABLED
template NDArray IndexSelect<float, int32_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<float, int64_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<double, int32_t>(
    NDArray, IdArray, cudaStream_t stream);
template NDArray IndexSelect<double, int64_t>(
    NDArray, IdArray, cudaStream_t stream);
}
}