//
// Created by juelin on 8/15/23.
//

#include "../../array/cuda/utils.h"
#include "../../runtime/cuda/cuda_common.h"
#include "cuda_index_select.cuh"
#include <assert.h>
#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

namespace dgl {
namespace groot {
namespace impl {

inline std::pair<dim3, dim3>
FindUVAKernelConfig(int feat_len, int feat_width, int feat_bytes = 4,
                    int CUDA_MAX_THREAD_NUM = 1024) {
  CHECK_GE(feat_width, 0);
  if (feat_width == 0)
    return std::make_pair(dim3(feat_len), dim3(1));
  constexpr int warp_size = 32;
  // TODO can we automatically detach this?
  constexpr int max_pcie_requests =
      256; // PCIe3.0 : 256. PCIe4.0: 768, PCIe 5.0: 768?
  // round num threads to the nearest 32 that is no larger than
  // CUDA_MAX_THREAD_NUM
  int num_threads = ((feat_width + warp_size - 1) / warp_size) * warp_size;
  int per_block_request = (num_threads * feat_bytes + 127) / 128;
  ;
  if (num_threads < feat_width) {
    const int scale = (num_threads + feat_width - 1) / feat_width;
    per_block_request *= scale;
  }
  int num_blocks =
      (max_pcie_requests + per_block_request - 1) / per_block_request;
  return std::make_pair(dim3(num_blocks), dim3(num_threads));
}

template <typename DType, typename IdType>
__global__ void IndexSelectSingleKernel(const DType *array, const IdType *index,
                                        const int64_t length,
                                        const int64_t arr_len, DType *out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    assert(index[tx] >= 0 && index[tx] < arr_len);
    out[tx] = array[index[tx]];
    tx += stride_x;
  }
}

template <typename DType, typename IdType>
__global__ void
IndexSelectMultiKernel(const DType *const array, const int64_t num_feat,
                       const IdType *const index, const int64_t length,
                       const int64_t arr_len, DType *const out) {
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
__global__ void
IndexSelectSingleKernel(const DType *array, const IdType *in_index,
                        const IdType *out_index, const int64_t length,
                        const int64_t arr_len, DType *out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    assert(in_index[tx] >= 0 && in_index[tx] < arr_len);
    out[out_index[tx]] = array[in_index[tx]];
    tx += stride_x;
  }
}

template <typename DType, typename IdType>
__global__ void
IndexSelectMultiKernel(const DType *const array, const int64_t num_feat,
                       const IdType *const in_index,
                       const IdType *const out_index, const int64_t length,
                       const int64_t arr_len, DType *const out) {
  int64_t task_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int64_t stride = blockDim.y * gridDim.x;

  while (task_id < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = in_index[task_id];
    const int64_t out_row = out_index[task_id];
    assert(in_row >= 0 && in_row < arr_len);
    while (col < num_feat) {
      out[out_row * num_feat + col] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    task_id += stride;
  }
}

template <typename DType, typename IdType>
__global__ void IndexScatterSingleKernel(const DType *array,
                                         const IdType *index,
                                         const int64_t length,
                                         const int64_t arr_len, DType *out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    assert(index[tx] >= 0 && index[tx] < arr_len);
    out[index[tx]] = array[tx];
    tx += stride_x;
  }
}

template <typename DType, typename IdType>
__global__ void
IndexScatterMultiKernel(const DType *const array, const int64_t num_feat,
                        const IdType *const index, const int64_t length,
                        const int64_t arr_len, DType *const out) {
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

} // namespace impl
template <typename DType, typename IdType>
NDArray _IndexSelect(NDArray array, IdArray index, cudaStream_t stream) {
  bool is_pinned = array.IsPinned();
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
  if (len == 0 || arr_len * num_feat == 0)
    return ret;
  DType *ret_data = static_cast<DType *>(ret->data);

  const DType *array_data = static_cast<DType *>(cuda::GetDevicePointer(array));
  const IdType *idx_data = static_cast<IdType *>(index->data);
  if (num_feat == 1) {
    const int nt = cuda::FindNumThreads(len);
    const int nb = (len + nt - 1) / nt;
    CUDA_KERNEL_CALL(impl::IndexSelectSingleKernel, nb, nt, 0, stream,
                     array_data, idx_data, len, arr_len, ret_data);
  } else if (!is_pinned) {
    dim3 block(256, 1);
    while (static_cast<int64_t>(block.x) >= 2 * num_feat) {
      block.x /= 2;
      block.y *= 2;
    }
    const dim3 grid((len + block.y - 1) / block.y);
    CUDA_KERNEL_CALL(impl::IndexSelectMultiKernel, grid, block, 0, stream,
                     array_data, num_feat, idx_data, len, arr_len, ret_data);
  } else {
    // array is pinned
    const auto config =
        impl::FindUVAKernelConfig(len, num_feat, array->dtype.bits / 8);
    const auto grid = config.first;
    const auto block = config.second;
    CUDA_KERNEL_CALL(impl::IndexSelectMultiKernel, grid, block, 0, stream,
                     array_data, num_feat, idx_data, len, arr_len, ret_data);
  }
  return ret;
}

template <typename DType, typename IdType>
void _IndexSelect(NDArray array, IdArray index, NDArray &ret,
                  cudaStream_t stream) {
  bool is_pinned = array.IsPinned();
  CHECK_EQ(array->ndim , ret->ndim);
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  int64_t num_feat = 1;
  //            std::vector<int64_t> shape{len};
  for (int d = 1; d < array->ndim; ++d) {
    num_feat *= array->shape[d];
    //                shape.emplace_back(array->shape[d]);
  }

  ret->shape[1] = num_feat;
  ret->shape[0] = len;
  // use index->ctx for pinned array
  if (len == 0 || arr_len * num_feat == 0)
    return;
  DType *ret_data = static_cast<DType *>(ret->data);

  const DType *array_data = static_cast<DType *>(cuda::GetDevicePointer(array));
  const IdType *idx_data = static_cast<IdType *>(index->data);
  if (num_feat == 1) {
    const int nt = cuda::FindNumThreads(len);
    const int nb = (len + nt - 1) / nt;
    CUDA_KERNEL_CALL(impl::IndexSelectSingleKernel, nb, nt, 0, stream,
                     array_data, idx_data, len, arr_len, ret_data);
  } else if (!is_pinned) {
    dim3 block(256, 1);
    while (static_cast<int64_t>(block.x) >= 2 * num_feat) {
      block.x /= 2;
      block.y *= 2;
    }
    const dim3 grid((len + block.y - 1) / block.y);
    CUDA_KERNEL_CALL(impl::IndexSelectMultiKernel, grid, block, 0, stream,
                     array_data, num_feat, idx_data, len, arr_len, ret_data);
  } else {
    // array is pinned
    const auto config =
        impl::FindUVAKernelConfig(len, num_feat, array->dtype.bits / 8);
    const auto grid = config.first;
    const auto block = config.second;
    CUDA_KERNEL_CALL(impl::IndexSelectMultiKernel, grid, block, 0, stream,
                     array_data, num_feat, idx_data, len, arr_len, ret_data);
  }
}

template <typename DType, typename IdType>
void _IndexSelect(NDArray array, IdArray in_index, IdArray out_index,
                  NDArray &ret, cudaStream_t stream) {
  bool is_pinned = array.IsPinned();
  const int64_t arr_len = array->shape[0];
  const int64_t len = in_index->shape[0];
  int64_t num_feat = 1;
  //            std::vector<int64_t> shape{len};
  for (int d = 1; d < array->ndim; ++d) {
    num_feat *= array->shape[d];
    //                shape.emplace_back(array->shape[d]);
  }

  ret->shape[1] = num_feat;
  ret->shape[0] = len;
  // use in_index->ctx for pinned array
  if (len == 0 || arr_len * num_feat == 0)
    return;
  DType *ret_data = static_cast<DType *>(ret->data);

  const DType *array_data = static_cast<DType *>(cuda::GetDevicePointer(array));
  const IdType *in_index_data = static_cast<IdType *>(in_index->data);
  const IdType *out_index_data = static_cast<IdType *>(out_index->data);

  if (num_feat == 1) {
    const int nt = cuda::FindNumThreads(len);
    const int nb = (len + nt - 1) / nt;
    CUDA_KERNEL_CALL(impl::IndexSelectSingleKernel, nb, nt, 0, stream,
                     array_data, in_index_data, out_index_data, len, arr_len,
                     ret_data);
  } else if (!is_pinned) {
    dim3 block(256, 1);
    while (static_cast<int64_t>(block.x) >= 2 * num_feat) {
      block.x /= 2;
      block.y *= 2;
    }
    const dim3 grid((len + block.y - 1) / block.y);
    CUDA_KERNEL_CALL(impl::IndexSelectMultiKernel, grid, block, 0, stream,
                     array_data, num_feat, in_index_data, out_index_data, len,
                     arr_len, ret_data);
  } else {
    // array is pinned
    const auto config =
        impl::FindUVAKernelConfig(len, num_feat, array->dtype.bits / 8);
    const auto grid = config.first;
    const auto block = config.second;
    CUDA_KERNEL_CALL(impl::IndexSelectMultiKernel, grid, block, 0, stream,
                     array_data, num_feat, in_index_data, out_index_data, len,
                     arr_len, ret_data);
  }
}

template NDArray _IndexSelect<int8_t, int32_t>(NDArray, IdArray,
                                               cudaStream_t stream);

template NDArray _IndexSelect<int8_t, int64_t>(NDArray, IdArray,
                                               cudaStream_t stream);

template NDArray _IndexSelect<int16_t, int32_t>(NDArray, IdArray,
                                                cudaStream_t stream);

template NDArray _IndexSelect<int16_t, int64_t>(NDArray, IdArray,
                                                cudaStream_t stream);

template NDArray _IndexSelect<int32_t, int32_t>(NDArray, IdArray,
                                                cudaStream_t stream);

template NDArray _IndexSelect<int32_t, int64_t>(NDArray, IdArray,
                                                cudaStream_t stream);

template NDArray _IndexSelect<int64_t, int32_t>(NDArray, IdArray,
                                                cudaStream_t stream);

template NDArray _IndexSelect<int64_t, int64_t>(NDArray, IdArray,
                                                cudaStream_t stream);

template NDArray _IndexSelect<__half, int32_t>(NDArray, IdArray,
                                               cudaStream_t stream);

template NDArray _IndexSelect<__half, int64_t>(NDArray, IdArray,
                                               cudaStream_t stream);

#if BF16_ENABLED

template NDArray _IndexSelect<__nv_bfloat16, int32_t>(NDArray, IdArray,
                                                      cudaStream_t stream);

template NDArray _IndexSelect<__nv_bfloat16, int64_t>(NDArray, IdArray,
                                                      cudaStream_t stream);

#endif // BF16_ENABLED

template NDArray _IndexSelect<float, int32_t>(NDArray, IdArray,
                                              cudaStream_t stream);

template NDArray _IndexSelect<float, int64_t>(NDArray, IdArray,
                                              cudaStream_t stream);

template NDArray _IndexSelect<double, int32_t>(NDArray, IdArray,
                                               cudaStream_t stream);

template NDArray _IndexSelect<double, int64_t>(NDArray, IdArray,
                                               cudaStream_t stream);

// with buffer

template void _IndexSelect<int8_t, int32_t>(NDArray, IdArray, NDArray &,
                                            cudaStream_t stream);

template void _IndexSelect<int8_t, int64_t>(NDArray, IdArray, NDArray &,
                                            cudaStream_t stream);

template void _IndexSelect<int16_t, int32_t>(NDArray, IdArray, NDArray &,
                                             cudaStream_t stream);

template void _IndexSelect<int16_t, int64_t>(NDArray, IdArray, NDArray &,
                                             cudaStream_t stream);

template void _IndexSelect<int32_t, int32_t>(NDArray, IdArray, NDArray &,
                                             cudaStream_t stream);

template void _IndexSelect<int32_t, int64_t>(NDArray, IdArray, NDArray &,
                                             cudaStream_t stream);

template void _IndexSelect<int64_t, int32_t>(NDArray, IdArray, NDArray &,
                                             cudaStream_t stream);

template void _IndexSelect<int64_t, int64_t>(NDArray, IdArray, NDArray &,
                                             cudaStream_t stream);

template void _IndexSelect<__half, int32_t>(NDArray, IdArray, NDArray &,
                                            cudaStream_t stream);

template void _IndexSelect<__half, int64_t>(NDArray, IdArray, NDArray &,
                                            cudaStream_t stream);

#if BF16_ENABLED

template void _IndexSelect<__nv_bfloat16, int32_t>(NDArray, IdArray, NDArray &,
                                                   cudaStream_t stream);

template void _IndexSelect<__nv_bfloat16, int64_t>(NDArray, IdArray, NDArray &,
                                                   cudaStream_t stream);

#endif // BF16_ENABLED

template void _IndexSelect<float, int32_t>(NDArray, IdArray, NDArray &,
                                           cudaStream_t stream);

template void _IndexSelect<float, int64_t>(NDArray, IdArray, NDArray &,
                                           cudaStream_t stream);

template void _IndexSelect<double, int32_t>(NDArray, IdArray, NDArray &,
                                            cudaStream_t stream);

template void _IndexSelect<double, int64_t>(NDArray, IdArray, NDArray &,
                                            cudaStream_t stream);

// with buff and out index

template void _IndexSelect<int8_t, int32_t>(NDArray, IdArray, IdArray,
                                            NDArray &, cudaStream_t stream);

template void _IndexSelect<int8_t, int64_t>(NDArray, IdArray, IdArray,
                                            NDArray &, cudaStream_t stream);

template void _IndexSelect<int16_t, int32_t>(NDArray, IdArray, IdArray,
                                             NDArray &, cudaStream_t stream);

template void _IndexSelect<int16_t, int64_t>(NDArray, IdArray, IdArray,
                                             NDArray &, cudaStream_t stream);

template void _IndexSelect<int32_t, int32_t>(NDArray, IdArray, IdArray,
                                             NDArray &, cudaStream_t stream);

template void _IndexSelect<int32_t, int64_t>(NDArray, IdArray, IdArray,
                                             NDArray &, cudaStream_t stream);

template void _IndexSelect<int64_t, int32_t>(NDArray, IdArray, IdArray,
                                             NDArray &, cudaStream_t stream);

template void _IndexSelect<int64_t, int64_t>(NDArray, IdArray, IdArray,
                                             NDArray &, cudaStream_t stream);

template void _IndexSelect<__half, int32_t>(NDArray, IdArray, IdArray,
                                            NDArray &, cudaStream_t stream);

template void _IndexSelect<__half, int64_t>(NDArray, IdArray, IdArray,
                                            NDArray &, cudaStream_t stream);

#if BF16_ENABLED

template void _IndexSelect<__nv_bfloat16, int32_t>(NDArray, IdArray, IdArray,
                                                   NDArray &,
                                                   cudaStream_t stream);

template void _IndexSelect<__nv_bfloat16, int64_t>(NDArray, IdArray, IdArray,
                                                   NDArray &,
                                                   cudaStream_t stream);

#endif // BF16_ENABLED

template void _IndexSelect<float, int32_t>(NDArray, IdArray, IdArray, NDArray &,
                                           cudaStream_t stream);

template void _IndexSelect<float, int64_t>(NDArray, IdArray, IdArray, NDArray &,
                                           cudaStream_t stream);

template void _IndexSelect<double, int32_t>(NDArray, IdArray, IdArray,
                                            NDArray &, cudaStream_t stream);

template void _IndexSelect<double, int64_t>(NDArray, IdArray, IdArray,
                                            NDArray &, cudaStream_t stream);

NDArray IndexSelect(NDArray array, IdArray index, cudaStream_t stream) {
  std::string header = "IndexSelect";
  if (index.NumElements() <= 8192) {
    header += "Label";
  } else {
    if (array.IsPinned()) {
      header += "FeatureUVA";
    } else {
      header += "FeatureGPU";
    }
  }
  nvtx3::scoped_range index_select{header};
  ATEN_DTYPE_BITS_ONLY_SWITCH(array->dtype, DType, "values", {
    ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
      return _IndexSelect<DType, IdType>(array, index, stream);
    });
  });
};

void IndexSelect(NDArray array, IdArray index, NDArray &out_buff,
                 cudaStream_t stream) {
  std::string header = "IndexSelect";
  if (index.NumElements() <= 8192) {
    header += "Label";
  } else {
    if (array.IsPinned()) {
      header += "FeatureUVA";
    } else {
      header += "FeatureGPU";
    }
  }
  nvtx3::scoped_range index_select{header};
  ATEN_DTYPE_BITS_ONLY_SWITCH(array->dtype, DType, "values", {
    ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
      return _IndexSelect<DType, IdType>(array, index, out_buff, stream);
    });
  });
};

void IndexSelect(NDArray array, IdArray in_index, IdArray out_index,
                 NDArray &out_buff, cudaStream_t stream) {
  std::string header = "IndexSelect";
  CHECK_EQ(in_index->dtype, out_index->dtype);
  CHECK_EQ(in_index->shape[0], out_index->shape[0])
      << "in_index and out_index must have the same shape";
  if (in_index.NumElements() <= 8192) {
    header += "Label";
  } else {
    if (array.IsPinned()) {
      header += "FeatureUVA";
    } else {
      header += "FeatureGPU";
    }
  }
  nvtx3::scoped_range index_select{header};
  ATEN_DTYPE_BITS_ONLY_SWITCH(array->dtype, DType, "values", {
    ATEN_ID_TYPE_SWITCH(in_index->dtype, IdType, {
      return _IndexSelect<DType, IdType>(array, in_index, out_index, out_buff,
                                         stream);
    });
  });
};
} // namespace groot
} // namespace dgl