//
// Created by juelin on 8/14/23.
//

#ifndef DGL_ROWWISE_SAMPLING_CUH
#define DGL_ROWWISE_SAMPLING_CUH

#include "../../runtime/cuda/cuda_common.h"
#include "../block.h"
// #include <dgl/runtime/ndarray.h>

namespace dgl {
namespace groot {
template <DGLDeviceType XPU, typename IdType>
void CSRRowWiseSamplingUniform(
    NDArray indptr, NDArray indices, NDArray rows, const int64_t num_picks,
    const bool replace, std::shared_ptr<BlockObject> block,
    cudaStream_t stream);
}  // namespace groot
}  // namespace dgl
#endif  // DGL_ROWWISE_SAMPLING_CUH
