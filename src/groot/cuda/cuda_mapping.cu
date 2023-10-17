//
// Created by juelin on 8/15/23.
//

#include "cuda_mapping.cuh"

namespace dgl {
namespace groot {

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__device__ void map_node_ids(const IdType *const global,
                             IdType *const new_global, const size_t num_input,
                             const DeviceOrderedHashTable<IdType> &table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Bucket = typename OrderedHashTable<IdType>::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = min(TILE_SIZE * (blockIdx.x + 1), num_input);

  for (size_t idx = threadIdx.x + block_start; idx < block_end;
       idx += BLOCK_SIZE) {
    const Bucket &bucket = *table.SearchO2N(global[idx]);
    new_global[idx] = bucket.local;
  }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
map_edge_ids(const IdType *const global_src, IdType *const new_global_src,
             const IdType *const global_dst, IdType *const new_global_dst,
             const size_t num_edges, DeviceOrderedHashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(2 == gridDim.y);

  if (blockIdx.y == 0) {
    map_node_ids<IdType, BLOCK_SIZE, TILE_SIZE>(global_src, new_global_src,
                                                num_edges, table);
  } else {
    map_node_ids<IdType, BLOCK_SIZE, TILE_SIZE>(global_dst, new_global_dst,
                                                num_edges, table);
  }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
map_edge_ids(const IdType *const global_src, IdType *const new_global_src,
             const size_t num_edges, DeviceOrderedHashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(1 == gridDim.y);

  map_node_ids<IdType, BLOCK_SIZE, TILE_SIZE>(global_src, new_global_src,
                                                num_edges, table);
}

void GPUMapEdges(NDArray row, NDArray ret_row, NDArray col, NDArray ret_col,
                 std::shared_ptr<CudaHashTable> mapping, cudaStream_t stream) {
  int64_t num_edges = row.NumElements();
  const size_t num_tiles =
      (num_edges + Constant::kCudaTileSize - 1) / Constant::kCudaTileSize;
  const dim3 grid(num_tiles, 2);
  const dim3 block(Constant::kCudaBlockSize);
  CHECK(row->dtype == ret_row->dtype);
  CHECK(col->dtype == ret_col->dtype);
  CHECK(col->dtype == row->dtype);
  CHECK(row->shape[0] == ret_row->shape[0]);
  CHECK(col->shape[0] == ret_col->shape[0]);

  ATEN_ID_TYPE_SWITCH(row->dtype, IdType, {
    const IdType *const global_src = row.Ptr<IdType>();
    IdType *const new_global_src = ret_row.Ptr<IdType>();
    const IdType *const global_dst = col.Ptr<IdType>();
    IdType *const new_global_dst = ret_col.Ptr<IdType>();
    auto table = mapping->DeviceHandle<IdType>();
    map_edge_ids<IdType, Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, stream>>>(global_src, new_global_src, global_dst,
                                     new_global_dst, num_edges, table);
  });
} // GPUMapEdges

void GPUMapEdges(NDArray row, NDArray ret_row, std::shared_ptr<CudaHashTable> mapping, cudaStream_t stream) {
  int64_t num_edges = row.NumElements();
  const size_t num_tiles =
      (num_edges + Constant::kCudaTileSize - 1) / Constant::kCudaTileSize;
  const dim3 grid(num_tiles, 1);
  const dim3 block(Constant::kCudaBlockSize);
  CHECK(row->dtype == ret_row->dtype);
  CHECK(row->shape[0] == ret_row->shape[0]);

  ATEN_ID_TYPE_SWITCH(row->dtype, IdType, {
    const IdType *const global_src = row.Ptr<IdType>();
    IdType *const new_global_src = ret_row.Ptr<IdType>();
    auto table = mapping->DeviceHandle<IdType>();
    map_edge_ids<IdType, Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, stream>>>(global_src, new_global_src, num_edges, table);
  });
    runtime::DeviceAPI::Get(row->ctx)->StreamSync(row->ctx, stream);
} // GPUMapEdges

} // namespace groot
} // namespace dgl