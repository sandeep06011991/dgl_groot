#include "array_scatter.h"
#include <dgl/runtime/c_runtime_api.h>

namespace dgl{
    namespace groot{


        IdArray getBoundaryOffsets(IdArray index_cum_sums,int num_partitions){
            IdArray ret;
                ATEN_ID_TYPE_SWITCH(index_cum_sums->dtype, IdType, {
                    ret = impl::getBoundaryOffsetsLocal<kDGLCUDA, IdType>( index_cum_sums , num_partitions);
            });
            return ret;
        }

        IdArray gatherArray(IdArray values, IdArray index, int num_partitions){
            IdArray ret;
            ATEN_ID_TYPE_SWITCH(values->dtype, IdType, {
               ret = impl::gatherIndexFromArray<kDGLCUDA, IdType>( values, index, num_partitions);
            });
            return ret;

        }

    }


    
}