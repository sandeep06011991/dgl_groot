
#include <iostream>
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "../../array/cuda/utils.h"
#include "../array_scatter.h"
#include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>

/**
 * @brief Search for the insertion positions for needle in the hay.
 *
 * The hay is a list of sorted elements and the result is the insertion position
 * of each needle so that the insertion still gives sorted order.
 *
 * It essentially perform binary search to find upper bound for each needle
 * elements.
 *
 * For example:
 * hay = [0, 0, 1, 2, 2]
 * needle = [0, 1, 2, 3]
 * then,
 * out = [2, 3, 5, 5]
 */
template <typename IdType, typename IndexType>
__global__ void _SortedSearchKernelUpperBound(
    const IdType* hay, int64_t hay_size, const IdType* needles,
    int64_t num_needles, IndexType* pos) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_needles) {
    const IdType ele = needles[tx];
    // binary search
    IdType lo = 0, hi = hay_size;
    while (lo < hi) {
      IdType mid = (lo + hi) >> 1;
      if (hay[mid] <= ele) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    pos[tx] = lo;
    tx += stride_x;
  }
}

using namespace dgl::runtime;
namespace dgl{
    namespace groot{
        namespace impl {

        // Checks the assigned partition map and scatters the array indices.
        template <typename PIdType, typename IndexType>
        __global__ void scatter_partition_continuous_index_kernel(const PIdType *partition_map,
                                             size_t partition_map_size,
                                             IndexType *index_out, int n_partitions) {
          int tx = blockIdx.x * blockDim.x + threadIdx.x;
          int stride_x = gridDim.x * blockDim.x;
          while (tx < partition_map_size) {
            auto p_id = partition_map[tx];
            assert(p_id < n_partitions);
            index_out[p_id * partition_map_size + tx] = 1;
            tx += stride_x;
          }
        }

        template <typename IDType>
        __inline__ __device__ bool is_selected(const IDType *index, int sz) {
          if (sz == 0) {
            return index[sz] == 1;
          }
          return index[sz] != index[sz - 1];
        }

        // gather_src_idx allows (src[gather_idx] = out
        // scatter_src_idx out[scatter_src_idx] = src
        template <typename IdType>
        __global__ void compute_partition_continuous_index_kernel(size_t source_size, const IdType *index,
            size_t index_size,IdType *gather_src_idx, IdType *scatter_src_idx, IdType * boundary_idx) {
          int tx = blockIdx.x * blockDim.x + threadIdx.x;
          int stride_x = gridDim.x * blockDim.x;
          if(tx == 0){
            boundary_idx[0] = 0;
          }
          while (tx < index_size) {
            if (is_selected(index, tx)) {
              auto value_idx = tx % source_size;
              assert(index[tx] - 1 < source_size);
              gather_src_idx[index[tx] - 1] = value_idx;
              scatter_src_idx[value_idx] = index[tx]-1;

            }
            if((tx + 1) % source_size == 0){
              boundary_idx[(tx + 1)/source_size] = index[tx];
            }
            tx += stride_x;
          }
        }


        // Given a partition map of
        // partition_map = [1,2,3,4,1,2,3,4]
        // returns partition gather index and partition sizes
        // gather_idx = [0,4,1,5,2,6,3,7]
        // scatter_idx = [0,2,4,6,1,3,5,7]
        // gather_idx_sizes = [4,8]
        // out = gather(gather_idx, in)
        // in = scatter(gather_idx, out)
        // in  = gather(scatter_ix, out)
        // out = scatter(gather_idx, in)
        // Naming convention
        // (OP)_in_out_idx, when used in !OP context, we can ensure that in_and_out are flipped
        // such that IndexSelect (F, gather_idx) will result in partition continuos ids
        template <DGLDeviceType XPU, typename IndexType, typename PIdType>
        std::tuple<IdArray,IdArray,IdArray>
            compute_partition_continuous_indices(IdArray partition_map, \
                                              int num_partitions,cudaStream_t stream) {
          // uint8_t nbits = 32;
          CHECK_EQ(partition_map->ndim , 1);
          auto index_data_type = DGLDataTypeTraits<IndexType>::dtype;
          IdArray expanded_index_out = aten::Full<IndexType>(
              0, partition_map->shape[0] * num_partitions, partition_map->ctx);
          size_t partition_map_size = partition_map->shape[0];

          const PIdType *partition_map_idx = partition_map.Ptr<PIdType>();
          IndexType *expanded_idx_ptr = expanded_index_out.Ptr<IndexType>();

          int nt = cuda::FindNumThreads(partition_map_size);
          int nb = (partition_map_size + nt - 1) / nt;
          CUDA_KERNEL_CALL(scatter_partition_continuous_index_kernel, nb, nt, 0, stream,
                           partition_map_idx, partition_map_size, expanded_idx_ptr ,
                           num_partitions);
          size_t workspace_size = 0;
          auto device = runtime::DeviceAPI::Get(partition_map->ctx);
          CUDA_CALL(cub::DeviceScan::InclusiveSum(
              nullptr, workspace_size, expanded_idx_ptr , expanded_idx_ptr ,
              partition_map_size * num_partitions, stream));
          void *workspace = device->AllocWorkspace(partition_map->ctx, workspace_size);
          CUDA_CALL(cub::DeviceScan::InclusiveSum(
              workspace, workspace_size, expanded_idx_ptr ,expanded_idx_ptr ,
              partition_map_size * num_partitions, stream));
          device->FreeWorkspace(partition_map->ctx, workspace);
          cudaStreamSynchronize(stream);


          IdArray gather_idx_in_part_disc_cont =
              IdArray::Empty({partition_map->shape[0]}, index_data_type, partition_map->ctx);
          IdArray scatter_idx_in_part_disc_cont =
              IdArray::Empty({partition_map->shape[0]}, index_data_type, partition_map->ctx);
          IdArray boundary_offsets = IdArray::Empty({num_partitions + 1}, index_data_type, partition_map->ctx);


          IndexType *gather_idx_ptr = gather_idx_in_part_disc_cont.Ptr<IndexType>();
          IndexType *scatter_idx_ptr = scatter_idx_in_part_disc_cont.Ptr<IndexType>();
          IndexType *boundary_offsets_ptr = boundary_offsets.Ptr<IndexType>();
          nt = cuda::FindNumThreads(expanded_index_out->shape[0]);
          nb = (expanded_index_out->shape[0] + nt - 1) / nt;

          CUDA_KERNEL_CALL(compute_partition_continuous_index_kernel, nb, nt, 0, stream,
                               partition_map->shape[0], expanded_idx_ptr,
                                expanded_index_out->shape[0], gather_idx_ptr, scatter_idx_ptr,
                              boundary_offsets_ptr);

          return std::tuple(boundary_offsets, gather_idx_in_part_disc_cont, scatter_idx_in_part_disc_cont);

        }


        template <DGLDeviceType XPU, typename PIdType>
        std::tuple<IdArray,IdArray,IdArray>
        compute_partition_continuous_indices_strawman(IdArray partition_map, \
                                             int num_partitions,cudaStream_t stream) {
          typedef PIdType IndexType;
          int num_items = partition_map->shape[0];
          // Declare, allocate, and initialize device-accessible pointers for sorting data
          // e.g., 7
          PIdType  *d_key_buf = partition_map.Ptr<PIdType>();         // e.g., [8, 6, 7, 5, 3, 0, 9]
          IdArray d_key_buf_array_partition = IdArray::Empty({num_items}, partition_map->dtype, partition_map->ctx);
          PIdType  *d_key_alt_buf_partition = d_key_buf_array_partition.Ptr<PIdType>();     // e.g., [        ...        ]
          auto index_data_type = DGLDataTypeTraits<IndexType>::dtype;
          IdArray range = aten::Range(0, num_items, index_data_type.bits, partition_map->ctx);       // e.g., [0, 1, 2, 3, 4, 5, 6]
          IndexType * d_value_buf = range.Ptr<IndexType>();
          IdArray gather_idx_in_part_disc_cont = IdArray::Empty({num_items}, partition_map->dtype, partition_map->ctx);
          IndexType  *d_value_alt_buf = gather_idx_in_part_disc_cont.Ptr<IndexType>(); // e.g., [        ...        ]
//
//          // Create a set of DoubleBuffers to wrap pairs of device pointers
          cub::DoubleBuffer<PIdType> d_keys1(d_key_buf, d_key_alt_buf_partition);
          cub::DoubleBuffer<IndexType> d_values1(d_value_buf, d_value_alt_buf);
          void     *d_temp_storage = NULL;
          size_t   temp_storage_bytes = 0;

          cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys1, d_values1, num_items, 0, index_data_type.bits, stream);
          auto device = runtime::DeviceAPI::Get(partition_map->ctx);
          d_temp_storage = device->AllocWorkspace(partition_map->ctx,temp_storage_bytes);

      //          // Run sorting operation
          cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys1, d_values1, num_items,  0, index_data_type.bits, stream);
          device->FreeWorkspace(partition_map->ctx, d_temp_storage);

//
          d_key_buf = d_value_alt_buf;
          IdArray d_key_buf_array = IdArray::Empty({num_items}, partition_map->dtype, partition_map->ctx);
          PIdType  *d_key_alt_buf = d_key_buf_array.Ptr<PIdType>();     // e.g., [        ...
          IdArray scatter_idx_in_part_disc_cont = aten::Range(0, num_items, index_data_type.bits, partition_map->ctx);       // e.g., [0, 1, 2, 3, 4, 5, 6]
          d_value_buf = range.Ptr<IndexType>();
          d_value_alt_buf = scatter_idx_in_part_disc_cont.Ptr<IndexType>();
//
          cub::DoubleBuffer<PIdType> d_keys2(d_key_buf, d_key_alt_buf);
          cub::DoubleBuffer<IndexType> d_values2(d_value_buf, d_value_alt_buf);
//
          cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys2, d_values2, num_items, 0, index_data_type.bits, stream);
          d_temp_storage = device->AllocWorkspace(partition_map->ctx,temp_storage_bytes);

          cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys2, d_values2, num_items, 0, index_data_type.bits, stream);
          device->FreeWorkspace(partition_map->ctx, d_temp_storage);


//        Sorting upper bound kernel compute boudary offsets;
          IdArray needle = IdArray::FromVector(std::vector<IndexType >{0,1,2,3}, partition_map->ctx);
          IdArray boundary_offsets = IdArray::FromVector(std::vector<int64_t>{0,0,0,0,0}, partition_map->ctx);

          const int nt = cuda::FindNumThreads(4);
          const int nb = (4 + nt - 1) / nt;
          CUDA_KERNEL_CALL(
              _SortedSearchKernelUpperBound, nb, nt, 0, stream,  d_key_buf_array_partition.Ptr<IndexType>(),
                  num_items, needle.Ptr<IndexType>(), 4, boundary_offsets.Ptr<int64_t>() + 1);
//          const IdType* hay, int64_t hay_size, const IdType* needles,
//              int64_t num_needles, IdType* pos
          return std::tuple(boundary_offsets, gather_idx_in_part_disc_cont, scatter_idx_in_part_disc_cont);
        }

            template
        std::tuple<IdArray,IdArray,IdArray>
        compute_partition_continuous_indices<DGLDeviceType::kDGLCUDA, int32_t ,int64_t>(IdArray partition_map, \
                                             int num_partitions,cudaStream_t stream);
        template
            std::tuple<IdArray,IdArray,IdArray>
            compute_partition_continuous_indices<DGLDeviceType::kDGLCUDA, int64_t ,int64_t>(IdArray partition_map, \
                                                                                            int num_partitions,cudaStream_t stream);


        template
            std::tuple<IdArray,IdArray,IdArray>
            compute_partition_continuous_indices<DGLDeviceType::kDGLCUDA, int32_t ,int32_t>(IdArray partition_map, \
                                                                                            int num_partitions,cudaStream_t stream);
        template
            std::tuple<IdArray,IdArray,IdArray>
            compute_partition_continuous_indices<DGLDeviceType::kDGLCUDA, int64_t ,int32_t>(IdArray partition_map, \
                                                                                            int num_partitions,cudaStream_t stream);
        template
            std::tuple<IdArray,IdArray,IdArray>
            compute_partition_continuous_indices_strawman<DGLDeviceType::kDGLCUDA, int64_t >(IdArray partition_map, \
                                                                                            int num_partitions,cudaStream_t stream);
        template
            std::tuple<IdArray,IdArray,IdArray>
            compute_partition_continuous_indices_strawman<DGLDeviceType::kDGLCUDA, int32_t >(IdArray partition_map, \
                                                                                            int num_partitions,cudaStream_t stream);



        }
    }
}

