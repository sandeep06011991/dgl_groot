//
// Created by juelin on 8/15/23.
//

#ifndef DGL_CUDA_INDEX_SELECT_CUH
#define DGL_CUDA_INDEX_SELECT_CUH

#include <dgl/array.h>
#include <nvtx3/nvtx3.hpp>

namespace dgl {
    namespace groot {
        NDArray IndexSelect(NDArray array, IdArray index, cudaStream_t stream);

        void IndexSelect(NDArray array, IdArray index, NDArray &out, cudaStream_t stream);
// TODO Implemet this
// template <typename DType, typename IdType>
// NDArray IndexScatter(NDArray input_arr, NDArray output_arr, IdArray index);
    }  // namespace groot
}  // namespace dgl
#endif  // DGL_CUDA_INDEX_SELECT_CUH
