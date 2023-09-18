#ifndef DGL_GROOT_ARRAY_SCATTER_H_
#define DGL_GROOT_ARRAY_SCATTER_H_

#include <dgl/array.h>
#include <dgl/runtime/container.h>
#include "../c_api_common.h"
#include <dgl/array.h>
#include <vector>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/parallel_for.h>
#include <dgl/runtime/registry.h>
#include <dgl/packed_func_ext.h>
#include <memory>
#include "./cuda/array_scatter.h"

using namespace dgl::runtime;

namespace dgl{
    namespace groot{
    IdArray getBoundaryOffsets(IdArray index_cum_sums, int num_partitions);

    IdArray gatherArray(IdArray values, IdArray index, int num_partitions);

    namespace impl{
    template<DGLDeviceType XPU, typename  IdType>
    IdArray gatherIndexFromArray(IdArray values, IdArray index, int num_partitions);

    template<DGLDeviceType XPU, typename IdType>
    IdArray getBoundaryOffsetsLocal(IdArray index_cum_sums, int num_partitions);
    }

    void ScatterIndex();

    class ScatteredArrayObject : public runtime::Object {

     public:
      ScatteredArrayObject(){}
      ScatteredArrayObject(NDArray a, NDArray b, int num_partitions){

      }
      //original array has no duplicates
       NDArray originalArray;
       // Partition map does not refer to global IDS, this is local here so that when we move blocks, 
       // partition map compute for dest vertices can be reused 
       NDArray partitionMap; 

	// original array is scattered such that partitions are contiguous, resulting in partitionContinuos array
       NDArray partitionContinuousArray;
       bool computeIdx;
       // Idx are only needed when moving Vertices, edges dont require this. 

       NDArray idx_original_to_part_cont; // shape of original array 
       NDArray idx_part_cont_to_original; // shape of scattered array
       // to send offsets 
       NDArray to_send_offsets_partition_continuous_array;
       
       // After NCCL Comm
       NDArray shuffled_array;
       NDArray shuffled_recv_offsets;

       // Possible recved array after shuffling has  has duplicates
       NDArray unique_array;
       NDArray idx_unique_to_shuffled;
       int num_partitions;

      void shuffle_forward(NDArray array){}
	
      void shuffle_backward(NDArray array){}

      void VisitAttrs(AttrVisitor *v) final {
            v->Visit("original_array", &originalArray);
            v->Visit("partition_map", &partitionMap);
            v->Visit("partitionContinuousArray", &partitionContinuousArray);
            v->Visit("idx_original_to_part_cont", &idx_original_to_part_cont);
            v->Visit("to_send_offsets_partition_continuos_array", &to_send_offsets_partition_continuous_array);
      }

      static constexpr const char* _type_key = "ScatteredArray";
      DGL_DECLARE_OBJECT_TYPE_INFO(ScatteredArrayObject, Object);
    };


    class ScatteredArray : public ObjectRef {
     public:
      DGL_DEFINE_OBJECT_REF_METHODS(ScatteredArray,runtime::ObjectRef, ScatteredArrayObject);

    };

    void Scatter(ScatteredArray array, NDArray frontier, NDArray _partition_map, int num_partitions);

    void ScatterWithDuplicates(ScatteredArray array, NDArray frontier, NDArray _partition_map, int num_partitions);


    class ScatterBlockEdges{
           public:
//            Circular dependecy
              ScatterBlockEdges(NDArray row, NDArray col, NDArray partition_map){}
        };


    }
}



#endif  // DGL_GROOT_ARRAY_SCATTER_H_
