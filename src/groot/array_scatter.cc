#include "array_scatter.h"
#include "../groot/cuda/cuda_mapping.cuh"
#include "cuda/alltoall.h"
#include "../groot/cuda/cuda_index_select.cuh"

namespace dgl {
namespace groot {
using namespace runtime;

IdArray getBoundaryOffsets(IdArray index_cum_sums, int num_partitions) {
  IdArray ret;
  ATEN_ID_TYPE_SWITCH(index_cum_sums->dtype, IdType, {
    ret = impl::getBoundaryOffsetsLocal<kDGLCUDA, IdType>(index_cum_sums,
                                                          num_partitions);
  });
  return ret;
}

IdArray gatherArray(IdArray values, IdArray index, IdArray out_idx,
                    int num_partitions) {
  IdArray ret;
  ATEN_ID_TYPE_SWITCH(values->dtype, IdType, {
    ret = impl::gatherIndexFromArray<kDGLCUDA, IdType>(values, index, out_idx,
                                                       num_partitions);
  });
  return ret;
}



IdArray scatter_index(IdArray partition_map, int num_partitions) {
  IdArray ret;
  ATEN_ID_TYPE_SWITCH(partition_map->dtype, IdType, {
    ret = impl::scatter_index<kDGLCUDA, IdType>(partition_map, num_partitions);
  });
  return ret;
}

IdArray  atomic_accumulation(IdArray accumulated_grads,IdArray idx_unique_to_shuffled,\
                            IdArray grad_shuffled_reshape){
  IdArray  ret;
  assert(idx_unique_to_shuffled->dtype.bits == 64);
  ATEN_FLOAT_TYPE_SWITCH(accumulated_grads->dtype, IdType, "accumulated_grads", {
    ret = impl::gather_atomic_accumulate<kDGLCUDA, IdType>(accumulated_grads, idx_unique_to_shuffled, \
                                                      grad_shuffled_reshape);
  });
  return ret;

}

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      LOG(FATAL) << "Cuda error " << cudaGetErrorString(e);                    \
    }                                                                          \
  } while (false);

NDArray ScatteredArrayObject::shuffle_forward(dgl::runtime::NDArray feat, int rank, int world_size) {
      assert(feat->shape[0] == unique_array->shape[0]);
      assert(feat->ndim == 2);
      cudaStream_t stream = runtime::getCurrentCUDAStream();

      NDArray toShuffle = NDArray::Empty({idx_unique_to_shuffled->shape[0], feat->shape[1]},
                                          feat->dtype, feat->ctx);

      IndexSelect(feat, idx_unique_to_shuffled, toShuffle, stream);

      NDArray  feat_shuffled;
      NDArray feat_offsets;

      std::tie(feat_shuffled, feat_offsets) \
          = ds::Alltoall(toShuffle, shuffled_recv_offsets, \
                                      feat->shape[1], rank, world_size,
                         to_send_offsets_partition_continuous_array, stream);
      const int num_nodes = feat_shuffled->shape[0] / feat->shape[1];

      NDArray feat_shuffled_reshape = feat_shuffled.CreateView(
                                                       {num_nodes, feat->shape[1]}, feat->dtype, 0);

      NDArray partDiscFeat =
          NDArray::Empty({feat_shuffled_reshape->shape[0], feat->shape[1]}, feat_shuffled->dtype, feat->ctx);

      IndexSelect(feat_shuffled_reshape,idx_part_cont_to_original, partDiscFeat, stream);
      return partDiscFeat;
}
NDArray ScatteredArrayObject::shuffle_backward(NDArray back_grad, int rank,int world_size){
      cudaStream_t stream = runtime::getCurrentCUDAStream();
      NDArray part_cont = NDArray::Empty(\
            {partitionContinuousArray->shape[0], back_grad->shape[1]}, back_grad->dtype, back_grad->ctx);
      IndexSelect(back_grad, idx_original_to_part_cont, part_cont, stream);

      NDArray  grad_shuffled;
      NDArray grad_offsets;
      cudaStreamSynchronize(stream);

      std::tie(grad_shuffled, grad_offsets) \
          = ds::Alltoall(part_cont, to_send_offsets_partition_continuous_array, \
                       back_grad->shape[1], rank, world_size, \
                          shuffled_recv_offsets, stream);

      const int num_nodes = grad_shuffled->shape[0] / back_grad->shape[1];

      NDArray grad_shuffled_reshape = grad_shuffled.CreateView(
          {num_nodes, back_grad->shape[1]}, back_grad->dtype, 0);

      cudaStreamSynchronize(stream);

      // offsets are always long
      auto grad_offsets_v = grad_offsets.ToVector<int64_t >();
      assert(back_grad->dtype.bits == 32);
      NDArray accumulated_grads = aten::Full((float) 0.0, unique_array->shape[0] * back_grad->shape[1],
                                              back_grad->ctx);
      accumulated_grads = accumulated_grads.CreateView({unique_array->shape[0], back_grad->shape[1]}, \
                                                        accumulated_grads->dtype, 0);
//    Todo can be optimized as self nodes are rarely written
//    Before doing this optimization have a unit test in place for this
      atomic_accumulation(accumulated_grads, idx_unique_to_shuffled, grad_shuffled_reshape);
      return accumulated_grads;
}


void Scatter(ScatteredArray array, NDArray frontier, NDArray _partition_map,
             int num_partitions, int rank, int world_size) {
  assert(array->dtype == frontier->dtype);
  array->originalArray = frontier;
  array->partitionMap = _partition_map;
  // Compute partition continuos array
  assert(frontier->dtype.bits == _partition_map->dtype.bits);
  NDArray scattered_index = scatter_index(array->partitionMap, num_partitions);

  // idx always in 64 bits
  array->idx_original_to_part_cont = NDArray::Empty(
      std::vector<int64_t>{frontier->shape[0]}, frontier->dtype, frontier->ctx);
  assert(frontier->dtype.bits == scattered_index->dtype.bits);
  array->partitionContinuousArray =
      gatherArray(frontier, scattered_index, array->idx_original_to_part_cont,
                  num_partitions);
  array->idx_part_cont_to_original = gatherArray(
      aten::Range(0, frontier->shape[0], frontier->dtype.bits, frontier->ctx),
      scattered_index, array->idx_original_to_part_cont, num_partitions);
//  std::cout << "found boundaries\n";
  array->to_send_offsets_partition_continuous_array =
      getBoundaryOffsets(scattered_index, num_partitions);
  NDArray boundary_offsets =
      getBoundaryOffsets(scattered_index, num_partitions);
  CUDACHECK(cudaDeviceSynchronize());
  if (world_size != 1) {
    std::tie(array->shuffled_array, array->shuffled_recv_offsets) = ds::Alltoall\
        (array->partitionContinuousArray, boundary_offsets, 1, rank, world_size);
  } else {
    array->shuffled_array = array->partitionContinuousArray;
    array->shuffled_recv_offsets = boundary_offsets;
  }

  CUDACHECK(cudaDeviceSynchronize());
  bool reindex = true;
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;
  if (reindex) {
    array->table->_stream = stream;
    array->table->Reset();
    // TODO: Why is table stream different from CUDA stream.
    cudaStreamSynchronize(stream);
    array->table->FillWithDuplicates(array->shuffled_array,
                                     array->shuffled_array->shape[0]);
    array->unique_array = array->table->RefUnique();
    array->idx_unique_to_shuffled = IdArray::Empty(
        std::vector<int64_t>{array->shuffled_array->shape[0]},
        array->shuffled_array->dtype, array->shuffled_array->ctx);
    GPUMapEdges(array->shuffled_array, array->idx_unique_to_shuffled,
                array->table, stream);
  }
}


DGL_REGISTER_GLOBAL("groot._CAPI_ScatterForward")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      ScatteredArray scatter_array = args[0];
      NDArray feat = args[1];
      int rank = args[2];
      int world_size = args[3];
      *rv = scatter_array->shuffle_forward(feat, rank, world_size);
    });

DGL_REGISTER_GLOBAL("groot._CAPI_ScatterBackward")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      ScatteredArray scatter_array = args[0];
      NDArray grads = args[1];
      int rank = args[2];
      int world_size = args[3];
      *rv = scatter_array->shuffle_backward(grads, rank, world_size);
    });

DGL_REGISTER_GLOBAL("groot._CAPI_getScatteredArrayObject")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray frontier = args[0];
      NDArray partition_map = args[1];
      int num_partitions = args[2];
      int rank = args[3];
      int  world_size = args[4];
      cudaStream_t stream = runtime::getCurrentCUDAStream();
      CUDAThreadEntry::ThreadLocal()->stream = stream;
      ScatteredArray scatter_array = ScatteredArray::Create(frontier->shape[0], 4,\
                                                            frontier->ctx, frontier->dtype, stream);
      Scatter(scatter_array, frontier, partition_map, num_partitions, rank , world_size);
      *rv = scatter_array;
    });

} // namespace groot
} // namespace dgl