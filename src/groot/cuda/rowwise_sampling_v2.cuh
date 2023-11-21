//
// Created by juelinliu on 11/18/23.
//

#ifndef DGL_ROWWISE_SAMPLING_V2_CUH
#define DGL_ROWWISE_SAMPLING_V2_CUH

#include "../../runtime/cuda/cuda_common.h"
#include "../block.h"
#include <dgl/runtime/ndarray.h>

#include <utility>

namespace dgl {
    namespace groot {

        template<DGLDeviceType XPU, typename IdType>
        void CSRRowWiseSamplingUniform(NDArray indptr, NDArray indices, std::vector<NDArray> rows,
                                                                      const int64_t num_picks, const bool replace,
                                                                      std::vector<std::shared_ptr<BlockObject>> blocks,
                                                                      cudaStream_t stream = runtime::getCurrentCUDAStream());


    } // namespace groot
} // namespace dgl

#endif //DGL_ROWWISE_SAMPLING_V2_CUH
