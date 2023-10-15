#include "array_scatter.h"
#include "../groot/cuda/cuda_mapping.cuh"
#include "cuda/alltoall.h"
#include "../groot/cuda/cuda_index_select.cuh"

namespace dgl {
namespace groot {
using namespace runtime;

typedef  int64_t IndexType;

template<typename IndexType>
std::tuple<IdArray,IdArray,IdArray> compute_partition_continuous_indices(IdArray partition_map,
                 int num_partitions, cudaStream_t stream) {
  std::tuple<IdArray ,IdArray , IdArray > ret;
  ATEN_ID_TYPE_SWITCH(partition_map->dtype, IdType, {
    ret = impl::compute_partition_continuous_indices<kDGLCUDA, IdType, IndexType>\
        (partition_map, num_partitions, stream);
  });
  return ret;
}

template<typename IndexType>
IdArray  gather_atomic_accumulation(IdArray accumulated_grads,IdArray gather_idx_in_unique,\
                            IdArray grad_shuffled_reshape, cudaStream_t stream){
  IdArray  ret;
  ATEN_FLOAT_TYPE_SWITCH(accumulated_grads->dtype, IdType, "accumulated_grads", {
    ret = impl::gather_atomic_accumulate<kDGLCUDA, IdType, IndexType>(
        accumulated_grads, gather_idx_in_unique, grad_shuffled_reshape,stream);
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
      assert(feat->shape[0] == unique_tensor_dim);
      cudaStream_t stream = runtime::getCurrentCUDAStream();

      NDArray toShuffle = NDArray::Empty({gather_idx_in_unique_out_shuffled->shape[0], feat->shape[1]},
                                          feat->dtype, feat->ctx);

      IndexSelect(feat, gather_idx_in_unique_out_shuffled, toShuffle, stream);

      NDArray  feat_shuffled;
      NDArray feat_offsets;
      cudaDeviceSynchronize();
      assert(shuffled_recv_offsets.ToVector<int64_t>()[4] == toShuffle->shape[0]);

      std::tie(feat_shuffled, feat_offsets) \
          = ds::Alltoall(toShuffle, shuffled_recv_offsets, \
                                      feat->shape[1], rank, world_size,
                         partitionContinuousOffsets, stream);

      const int num_nodes = feat_shuffled->shape[0] / feat->shape[1];

      NDArray feat_shuffled_reshape = feat_shuffled.CreateView(
                                                       {num_nodes, feat->shape[1]}, feat->dtype, 0);
      NDArray partDiscFeat =
          NDArray::Empty({feat_shuffled_reshape->shape[0], feat->shape[1]}, feat_shuffled->dtype, feat->ctx);
      IndexSelect(feat_shuffled_reshape, scatter_idx_in_part_disc_cont, partDiscFeat, stream);

      assert(partDiscFeat->shape[0] == scattered_tensor_dim);
      return partDiscFeat;
}
NDArray ScatteredArrayObject::shuffle_backward(NDArray back_grad, int rank,int world_size){
      assert(back_grad->shape[0] == scattered_tensor_dim);
      cudaStream_t stream = runtime::getCurrentCUDAStream();
      // backgrad is part disccont
      NDArray part_cont = NDArray::Empty(\
            {partitionContinuousArray->shape[0], back_grad->shape[1]}, back_grad->dtype, back_grad->ctx);
//      atomic_accumulation(part_cont, idx_original_to_part_cont, back_grad);
//      assert(idx_original_to_part_cont->shape[0]!=back_grad->shape[0]);
      IndexSelect(back_grad, gather_idx_in_part_disc_cont, part_cont, stream);

      NDArray  grad_shuffled;
      NDArray grad_offsets;

      std::tie(grad_shuffled, grad_offsets) \
          = ds::Alltoall(part_cont, partitionContinuousOffsets, \
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
      gather_atomic_accumulation<IndexType>(accumulated_grads,\
                                            gather_idx_in_unique_out_shuffled, grad_shuffled_reshape, stream);
      assert(accumulated_grads->shape[0] == unique_tensor_dim);
      assert(accumulated_grads->shape[1] == back_grad->shape[1]);
      return accumulated_grads;
}


void Scatter(ScatteredArray array, NDArray frontier, NDArray _partition_map,
             int num_partitions, int rank, int world_size) {
  assert(array->dtype == frontier->dtype);
  assert(frontier->shape[0] > 0);
  assert(frontier->shape[0] < array->expectedSize);
  if(array->debug){
    array->table->Reset();
    array->table->FillWithDuplicates(frontier, frontier->shape[0]);
    CHECK_EQ(array->table->RefUnique()->shape[0], frontier->shape[0]);
    array->table->Reset();
  }
  array->scattered_tensor_dim = frontier->shape[0];
  array->originalArray = frontier;
  array->partitionMap = _partition_map;

  // Compute partition continuos array
  cudaStream_t stream =  runtime::getCurrentCUDAStream();
  // Todo: Why not runtime stream
  auto [boundary_offsets, gather_idx_in_part_disc_cont, scatter_idx_in_part_disc_cont]\
      = compute_partition_continuous_indices<IndexType>(array->partitionMap, num_partitions, stream);

  array->partitionContinuousArray = IndexSelect(frontier, gather_idx_in_part_disc_cont , stream);
  array->gather_idx_in_part_disc_cont = gather_idx_in_part_disc_cont;
  array->scatter_idx_in_part_disc_cont = scatter_idx_in_part_disc_cont;
  array->partitionContinuousOffsets  = boundary_offsets;

  if(array->debug){
    cudaStreamSynchronize(stream);
    assert(boundary_offsets.ToVector<int64_t>()[4] == array->partitionContinuousArray->shape[0]);
  }

  if (world_size != 1) {
    std::tie(array->shuffled_array, array->shuffled_recv_offsets) = ds::Alltoall\
        (array->partitionContinuousArray, boundary_offsets, 1, rank, world_size);
  } else {
    array->shuffled_array = array->partitionContinuousArray;
    array->shuffled_recv_offsets = boundary_offsets;
  }


  // Todo: Why table stream is different from main stream
  array->table->_stream = stream;
  array->table->Reset();
  array->table->FillWithDuplicates(array->shuffled_array,
                                   array->shuffled_array->shape[0]);
  array->unique_array = array->table->RefUnique();
  array->unique_tensor_dim = array->unique_array->shape[0];
  array->gather_idx_in_unique_out_shuffled = IdArray::Empty(
      std::vector<int64_t>{array->shuffled_array->shape[0]},
      array->shuffled_array->dtype, array->shuffled_array->ctx);
  GPUMapEdges(array->shuffled_array, array->gather_idx_in_unique_out_shuffled,
              array->table, stream);
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