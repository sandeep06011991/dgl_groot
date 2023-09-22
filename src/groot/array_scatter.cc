#include "array_scatter.h"
#include "../groot_dataloader/cuda/cuda_mapping.cuh"
#include "cuda/alltoall.h"

namespace dgl{
    namespace groot{


        IdArray getBoundaryOffsets(IdArray index_cum_sums,int num_partitions){
            IdArray ret;
                ATEN_ID_TYPE_SWITCH(index_cum_sums->dtype, IdType, {
                    ret = impl::getBoundaryOffsetsLocal<kDGLCUDA, IdType>( index_cum_sums , num_partitions);
            });
            return ret;
        }

        IdArray gatherArray(IdArray values, IdArray index, IdArray  out_idx, int num_partitions){
            IdArray ret;
            ATEN_ID_TYPE_SWITCH(values->dtype, IdType, {
               ret = impl::gatherIndexFromArray<kDGLCUDA, IdType>( values, index, out_idx, num_partitions);
            });
            return ret;
        }

        IdArray scatter_index(IdArray partition_map,int num_partitions){
            IdArray  ret;
            ATEN_ID_TYPE_SWITCH(partition_map->dtype, IdType, {
              ret = impl::scatter_index<kDGLCUDA, IdType>( partition_map , num_partitions);
            });
            return ret;
        }

        #define CUDACHECK(cmd)                                      \
          do {                                                      \
            cudaError_t e = cmd;                                    \
            if (e != cudaSuccess) {                                 \
              LOG(FATAL) << "Cuda error " << cudaGetErrorString(e); \
            }                                                       \
          } while (false);

        void Scatter(ScatteredArray array, NDArray frontier, NDArray _partition_map, int num_partitions,\
                          int rank, int world_size){
            std::cout << "inside scatter \n";
            array->originalArray = frontier;
            array->partitionMap = _partition_map;
            // Compute partition continuos array
            std::cout << "Start scattering index\n";
            assert(frontier->dtype.bits == _partition_map->dtype.bits);
            NDArray scattered_index = scatter_index(array->partitionMap, num_partitions);

            std::cout << "gather array\n";

            array->idx_original_to_part_cont = NDArray::Empty(\
                  std::vector<int64_t>{frontier->shape[0]}, frontier->dtype, frontier->ctx);
            assert(frontier->dtype.bits == scattered_index->dtype.bits);
            array->partitionContinuousArray = gatherArray(frontier, scattered_index, array->idx_original_to_part_cont,  num_partitions);

            std::cout << "Error os here \n";
            array->idx_part_cont_to_original = gatherArray(aten::Range(0,frontier->shape[0], frontier->dtype.bits, frontier->ctx), scattered_index,\
                                                              array->idx_original_to_part_cont,
                                                            num_partitions);
            std::cout << "Boundary offsets \n";

            array->to_send_offsets_partition_continuous_array = getBoundaryOffsets(scattered_index, num_partitions);

            NDArray boundary_offsets = getBoundaryOffsets(scattered_index, num_partitions);
//
            std::cout << "shuffled  size\n";
            std::cout << array->partitionContinuousArray->shape[0] << "\n";
            std::cout << boundary_offsets <<"\n";
            CUDACHECK(cudaDeviceSynchronize());
            std::tie(array->shuffled_array,array->shuffled_recv_offsets) = \
                ds::Alltoall(array->partitionContinuousArray, boundary_offsets\
                                                          , 1, rank, world_size);

            CUDACHECK(cudaDeviceSynchronize());
            bool reindex = true;
            std::cout << "shuffle ok \n";
            if(reindex) {
              array->table = std::make_shared<CudaHashTable>();

              array->table->Reset();
              array->table->FillWithDuplicates(
                  array->shuffled_array, array->shuffled_array->shape[0]);
              array->unique_array = array->table->GetUnique();
              cudaStream_t stream = dgl::runtime::getCurrentCUDAStream();
              array->idx_unique_to_shuffled = IdArray::Empty(
                  std::vector<int64_t>{array->shuffled_array->shape[0]},
                  array->shuffled_array->dtype, array->shuffled_array->ctx);
              GPUMapNodes(
                  array->shuffled_array, array->idx_unique_to_shuffled,
                  array->table, stream);
              std::cout << "Printing unique array for RANK"
                        << array->unique_array << "\n";
              cudaStreamSynchronize(stream);
              std::cout << "Hello wosrld of scatter entry  \n";
            }
        }
    }
}