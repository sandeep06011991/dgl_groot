//
// Created by juelin on 8/15/23.
//

#ifndef DGL_CUDA_MAPPING_CUH
#define DGL_CUDA_MAPPING_CUH

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

#include "cuda_hashtable.cuh"

namespace dgl {
namespace groot {

void GPUMapEdges(
    NDArray row, NDArray ret_row, NDArray col, NDArray ret_col,
    std::shared_ptr<CudaHashTable> mapping, cudaStream_t stream);
}

}  // namespace dgl
#endif  // DGL_CUDA_MAPPING_CUH
