#include "array_scatter.h"
#include <iostream>
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "../../array/cuda/utils.h"
// #include "../array_scatter.h"
#include <iostream>
using namespace dgl::runtime;

namespace dgl{
    namespace groot{
        namespace impl{

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

template<typename IdType>
__global__
void get_boundary_offsets_kernel(
    const IdType * index_cum, size_t num_partitions, size_t partition_map_size, \
         IdType * index_out){
        int tx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = gridDim.x * blockDim.x;
        while (tx < num_partitions) {
            index_out[tx] = index_cum[partition_map_size -1 + tx * partition_map_size];
            tx += stride_x;
    }
}


template<typename IdType>
IdArray scatter_index_local(IdArray partition_map,int num_partitions, uint8_t nbits){
    // uint8_t nbits = 32;
    IdArray index_out = aten::Full(0, partition_map->shape[0] * num_partitions, nbits, \
            partition_map->ctx);
    size_t partition_map_size = partition_map->shape[0];
    const IdType* partition_map_idx = partition_map.Ptr<IdType>();
    IdType* index_out_idx = index_out.Ptr<IdType>();

    cudaStream_t stream = runtime::getCurrentCUDAStream();
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
    cudaStreamSynchronize(stream);
    return index_out;
}


    


template IdArray scatter_index_local<int32_t>(IdArray ,int, uint8_t);
template IdArray scatter_index_local<int64_t>(IdArray ,int, uint8_t);

IdArray scatter_index(IdArray partition_map,int num_partitions){
    IdArray ret;
    // Todo Switching here is incorrect 
    if(partition_map->dtype.bits == 32){
        ret = scatter_index_local<int>(partition_map, num_partitions, 32);
    }
    if(partition_map->dtype.bits == 64){
        ret = scatter_index_local<long>(partition_map, num_partitions, 64);
    }
   
    return ret;
}   
template<DGLDeviceType XPU, typename IdType>
IdArray  getBoundaryOffsetsLocal(IdArray index_cum_sums,int num_partitions){    
    IdArray index_out = aten::Full(0,  num_partitions, index_cum_sums->dtype.bits, \
            index_cum_sums->ctx);

    const IdType* index_cum_sums_idx = index_cum_sums.Ptr<IdType>();
    IdType* index_out_idx = index_out.Ptr<IdType>();
    cudaStream_t stream = runtime::getCurrentCUDAStream();
    const int nt = cuda::FindNumThreads(num_partitions);
    const int nb = (num_partitions + nt - 1) / nt;
    CUDA_KERNEL_CALL(get_boundary_offsets_kernel, nb, nt, 0, stream,\
        index_cum_sums_idx, num_partitions, index_cum_sums->shape[0]/num_partitions, index_out_idx);

    return index_out.CopyTo(DGLContext{kDGLCPU, 0});  
}

    template IdArray getBoundaryOffsetsLocal<kDGLCUDA, int32_t>(IdArray ,int );
    template IdArray getBoundaryOffsetsLocal<kDGLCUDA, int64_t>(IdArray ,int );
    
        }
    }
}