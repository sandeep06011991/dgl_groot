#include <iostream>
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "../../array/cuda/utils.h"
#include "../array_scatter.h"
using namespace dgl::runtime;
namespace dgl{
    namespace groot{
        namespace impl{

// Checks the assigned partition map and scatters the array indices.
template<typename IdType>
__global__
void scatter_index_kernel(
    const IdType * partition_map, size_t partition_map_size, \
         IdType * index_out, int n_partitions){
        int tx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = gridDim.x * blockDim.x;
        while (tx < partition_map_size) {
            auto p_id = partition_map[tx];
            assert(p_id < n_partitions);
            index_out[p_id * partition_map_size + tx] = 1;
            tx += stride_x;
    }
}
template<typename IDType>
__inline__
__device__
  bool is_selected(const IDType * index, int sz){
  if(sz == 0) {
            return index[sz] == 1;
    }
    return index[sz] != index[sz - 1];
}


template<typename IdType>
__global__
    void gather_index_kernel(
        const IdType * source, size_t source_size,\
        const IdType * index, size_t index_size, \
        IdType * out, IdType * out_idx, int n_partitions){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = gridDim.x * blockDim.x;
    while (tx < index_size) {
            if(is_selected(index, tx)){
              auto value_idx = tx % source_size;
              assert(index[tx] - 1 < source_size);  
              out[index[tx] - 1] = source[value_idx];
              out_idx[index[tx]-1] = value_idx;
            }
            tx += stride_x;
    }
}

template<typename IdType>
__global__
void get_boundary_offsets_kernel(
    const IdType * index_cum, size_t num_partitions, size_t partition_map_size, \
         IdType * index_out){
        int tx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = gridDim.x * blockDim.x;
        while (tx < num_partitions) {
            index_out[tx + 1] = index_cum[partition_map_size -1 + tx * partition_map_size];
            tx += stride_x;
    }
}


template<DGLDeviceType XPU, typename IdType>
IdArray scatter_index(IdArray partition_map,int num_partitions){
    // uint8_t nbits = 32;
    auto nbits = partition_map->dtype.bits;

    IdArray index_out = aten::Full<IdType>(0, partition_map->shape[0] * num_partitions, \
            partition_map->ctx);
    size_t partition_map_size = partition_map->shape[0];
    const IdType* partition_map_idx = partition_map.Ptr<IdType>();
    IdType* index_out_idx = index_out.Ptr<IdType>();

    cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;

    const int nt = cuda::FindNumThreads(partition_map_size);
    const int nb = (partition_map_size + nt - 1) / nt;
    CUDA_KERNEL_CALL(scatter_index_kernel, nb, nt, 0, stream,\
        partition_map_idx, partition_map_size, index_out_idx, num_partitions);
    size_t workspace_size = 0;
    auto device = runtime::DeviceAPI::Get(index_out->ctx);
    CUDA_CALL(cub::DeviceScan::InclusiveSum(
        nullptr, workspace_size,\
                index_out_idx, index_out_idx,\
                    partition_map_size * num_partitions, stream));
    void* workspace = device->AllocWorkspace(index_out->ctx, workspace_size);
    CUDA_CALL(cub::DeviceScan::InclusiveSum(
        workspace , workspace_size,\
                index_out_idx, index_out_idx,\
                    partition_map_size * num_partitions, stream));
    device->FreeWorkspace(index_out->ctx, workspace);
    cudaStreamSynchronize(stream);
    return index_out;
}


template<DGLDeviceType XPU, typename IdType>
IdArray  getBoundaryOffsetsLocal(IdArray index_cum_sums,int num_partitions){    
    IdArray index_out = aten::Full(0,  num_partitions + 1, index_cum_sums->dtype.bits, \
            index_cum_sums->ctx);

    const IdType* index_cum_sums_idx = index_cum_sums.Ptr<IdType>();
    IdType* index_out_idx = index_out.Ptr<IdType>();
    cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;
    const int nt = cuda::FindNumThreads(num_partitions);
    const int nb = (num_partitions + nt - 1) / nt;
    CUDA_KERNEL_CALL(get_boundary_offsets_kernel, nb, nt, 0, stream,\
        index_cum_sums_idx, num_partitions, index_cum_sums->shape[0]/num_partitions, index_out_idx);
//  Todo Use Pinned memory. save on memory copy
    return index_out;
//    .CopyTo(DGLContext{kDGLCPU, 0});
}



template<DGLDeviceType XPU, typename  IdType>
IdArray gatherIndexFromArray(IdArray values, IdArray index, IdArray out_idx, int num_partitions){
//  Todo : Error of node tye
//  Steram line this.
    assert(values->dtype.bits == index->dtype.bits);
    IdArray out = aten::NewIdArray(values->shape[0], index->ctx, values->dtype.bits);

    const IdType * values_ptr = values.Ptr<IdType>();
    const IdType * index_ptr = index.Ptr<IdType>();
    IdType * out_ptr = out.Ptr<IdType>();
    IdType * out_idx_ptr = out_idx.Ptr<IdType>();
    cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;
    const int nt = cuda::FindNumThreads(index->shape[0]);
    const int nb = (index->shape[0] + nt - 1) / nt;

    CUDA_KERNEL_CALL(gather_index_kernel, nb, nt, 0, stream,\
                     values_ptr, values->shape[0],\
                         index_ptr, index->shape[0], out_ptr, out_idx_ptr, num_partitions);
    return out;
}

template<DGLDeviceType  XPU, typename IdType>
IdArray gather_atomic_accumulate(IdArray floats, IdArray values, IdArray empty_full){

}

    template IdArray gather_atomic_accumulate<kDGLCUDA, float>(IdArray floats, IdArray values, IdArray empty_full);
    template IdArray gather_atomic_accumulate<kDGLCUDA, double>(IdArray floats, IdArray values, IdArray empty_full);

    template IdArray scatter_index<kDGLCUDA, int32_t>(IdArray , int);
    template IdArray scatter_index<kDGLCUDA, int64_t>(IdArray , int);

    template IdArray gatherIndexFromArray<kDGLCUDA, int32_t>(IdArray, IdArray, IdArray, int);
    template IdArray gatherIndexFromArray<kDGLCUDA, int64_t>(IdArray, IdArray, IdArray, int);

    template IdArray getBoundaryOffsetsLocal<kDGLCUDA, int32_t>(IdArray ,int );
    template IdArray getBoundaryOffsetsLocal<kDGLCUDA, int64_t>(IdArray ,int );
    
        }
    }
}

