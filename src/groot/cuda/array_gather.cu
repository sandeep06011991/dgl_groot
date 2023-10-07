#include <iostream>
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "../../array/cuda/utils.h"
#include "../array_scatter.h"
using namespace dgl::runtime;
namespace dgl{
namespace groot {
namespace impl {

template <typename IdType>
__global__ void
gather_accumulate_kernel(const IdType *out_accumulated_grads,
                         const IdType *in_grads, const size_t in_grads_size,
                         const size_t feat_size, const long *in_grads_index) {
  int tx = blockIdx.x;
  int stride_x = gridDim.x;
  while (tx < in_grads_size) {
    int ty = threadIdx.x;
    while (ty < feat_size) {
      atomicAdd(
          (IdType *)&out_accumulated_grads[in_grads_index[tx] * feat_size + ty],
          (IdType)in_grads[tx * feat_size + ty]);
      ty += blockDim.x;
    }
    tx += stride_x;
  }
}


template <DGLDeviceType XPU, typename IdType>
IdArray gather_atomic_accumulate(IdArray accumulated_grads,
                                 IdArray idx_unique_to_shuffled,
                                 IdArray grad_shuffled_reshape) {
  const IdType *out_ptr = accumulated_grads.Ptr<IdType>();
  assert(idx_unique_to_shuffled->dtype.code == 0);
  const long *index_ptr = idx_unique_to_shuffled.Ptr<long>();
  const IdType *grad_ptr = grad_shuffled_reshape.Ptr<IdType>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();
  assert(accumulated_grads->ndim == 2);
  assert(grad_shuffled_reshape->ndim == 2);

  const int nt = cuda::FindNumThreads(grad_shuffled_reshape->shape[1]);
  const int TILE_SIZE = 1024;
  const int nb = min((int)grad_shuffled_reshape->shape[0] ,  TILE_SIZE);
  CUDA_KERNEL_CALL(gather_accumulate_kernel, nb, nt, 0, stream, out_ptr,
                   grad_ptr, grad_shuffled_reshape->shape[0],
                   grad_shuffled_reshape->shape[1], index_ptr);
  return accumulated_grads;
}




    }
  }
}