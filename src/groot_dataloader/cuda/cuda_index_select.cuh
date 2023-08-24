//
// Created by juelin on 8/15/23.
//

#ifndef DGL_CUDA_INDEX_SELECT_CUH
#define DGL_CUDA_INDEX_SELECT_CUH
#include <dgl/array.h>
#include <nvtx3/nvtx3.hpp>

namespace dgl {
namespace groot {
template <typename DType, typename IdType>
NDArray _IndexSelect(NDArray array, IdArray index, cudaStream_t stream);

inline NDArray IndexSelect(NDArray array, IdArray index, cudaStream_t stream) {
  std::string header = "IndexSelect";
  if (index.NumElements() <= 8192) {
    header += "Label";
  } else {
    header += "Feature";
  }
  nvtx3::scoped_range index_select{header};
 ATEN_DTYPE_BITS_ONLY_SWITCH(array->dtype, DType, "values", {
    ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
      return _IndexSelect<DType, IdType>(array, index, stream);
    });
  });
}
// TODO Implemet this
// template <typename DType, typename IdType>
// NDArray IndexScatter(NDArray input_arr, NDArray output_arr, IdArray index);
}  // namespace groot
}  // namespace dgl
#endif  // DGL_CUDA_INDEX_SELECT_CUH
