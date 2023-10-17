#ifndef DGL_GROOT_ARRAY_SCATTER_H_
#define DGL_GROOT_ARRAY_SCATTER_H_

#include "../c_api_common.h"
#include "../groot/cuda/cuda_hashtable.cuh"

#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/parallel_for.h>
#include <dgl/runtime/registry.h>
#include <memory>
#include <vector>

#include "cuda/alltoall.h"

using namespace dgl::runtime;

namespace dgl {
namespace groot {


namespace impl {

template <DGLDeviceType XPU, typename IdType, typename IndexType>
IdArray  gather_atomic_accumulate(IdArray accumulated_grads,IdArray idx_unique_to_shuffled,\
                                IdArray grad_shuffled_reshape, cudaStream_t stream);


template <DGLDeviceType XPU, typename  IdType, typename IndexType>
std::tuple<IdArray,IdArray,IdArray>
compute_partition_continuous_indices(IdArray partition_map, int num_partitions,cudaStream_t stream);

} // namespace impl

class ScatteredArrayObject : public runtime::Object {

public:
  size_t scattered_tensor_dim;
  size_t unique_tensor_dim;

  DGLDataType dtype;
  DGLContext ctx;
  int num_partitions;
  cudaStream_t stream;
  int expectedSize;
  bool debug = false;
  ~ScatteredArrayObject() {
//    LOG(INFO) << "Destructor called on scattered object" << this->expectedSize
//              << "\n";
  }
  ScatteredArrayObject(int expectedSize, int num_partitions, DGLContext ctx,
                       DGLDataType dtype, cudaStream_t stream = runtime::getCurrentCUDAStream()) {
    this->dtype = dtype;
    this->ctx = ctx;
    this->num_partitions = num_partitions;
    this->stream = stream;
    this->expectedSize = expectedSize;
//    this->table =
//        std::make_shared<CudaHashTable>(dtype, ctx, expectedSize, stream);
  }
  bool isScattered;
  // original array has no duplicates
  // original array is partition discontinuous
  NDArray originalArray;
  // Partition map does not refer to global IDS, this is local here so that when
  // we move blocks, partition map compute for dest vertices can be reused
  NDArray partitionMap;

  // original array is scattered such that partitions are contiguous, resulting
  // in partitionContinuos array
  NDArray partitionContinuousArray;
  NDArray partitionContinuousOffsets;
  bool computeIdx;
  // Idx are only needed when moving Vertices, edges dont require this.
  NDArray gather_idx_in_part_disc_cont;// shape of original array
  NDArray scatter_idx_in_part_disc_cont; // shape of scattered array


  // After NCCL Comm
  NDArray shuffled_array;
  NDArray shuffled_recv_offsets;

  // Possible received array after shuffling has duplicates
//  std::shared_ptr<CudaHashTable> table;
  NDArray unique_array;
  NDArray gather_idx_in_unique_out_shuffled;

  NDArray shuffle_forward(NDArray feat, int rank, int world_size);

  NDArray shuffle_backward(NDArray back_grad, int rank,int world_size);

  void VisitAttrs(AttrVisitor *v) final {
    v->Visit("original_array", &originalArray);
    v->Visit("partition_map", &partitionMap);
    v->Visit("partitionContinuousArray", &partitionContinuousArray);
    v->Visit("gather_idx_in_part_disc_cont", &gather_idx_in_part_disc_cont);
    v->Visit("scatter_idx_in_part_disc_cont", & scatter_idx_in_part_disc_cont);
    v->Visit("partition_continuous_offsets",
             &partitionContinuousOffsets);
    v->Visit("unique_array", &unique_array);
    v->Visit("gather_idx_in_unique_out_shuffled", &gather_idx_in_unique_out_shuffled);
  }

  static constexpr const char *_type_key = "ScatteredArray";
  DGL_DECLARE_OBJECT_TYPE_INFO(ScatteredArrayObject, Object);
};

class ScatteredArray : public ObjectRef {
public:
  DGL_DEFINE_OBJECT_REF_METHODS(ScatteredArray, runtime::ObjectRef,
                                ScatteredArrayObject);
  static ScatteredArray Create(int expected_size_on_single_gpu, int num_partitions,
                               DGLContext ctx, DGLDataType dtype,
                               cudaStream_t stream) {
    std::cout << "Creating scattered object with expected size " << expected_size_on_single_gpu << "\n";
    return ScatteredArray(std::make_shared<ScatteredArrayObject>(
            expected_size_on_single_gpu * num_partitions, num_partitions, ctx, dtype, stream));
  }
};

void Scatter(ScatteredArray array, NDArray frontier, NDArray _partition_map,
             int num_partitions, int rank, int world_size);

  } // namespace groot
} // namespace dgl

#endif // DGL_GROOT_ARRAY_SCATTER_H_
