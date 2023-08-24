

#pragma once
#include <dgl/array.h>

// Todo: This file is not required as we dont share decleration but use templating 
// to find the correct implementation
namespace dgl{
    namespace groot{
        namespace impl{

        // e.g. partition_map = [0,1,0,1,0,1]
        // scatters index = [1,0,1,0,1,0,0,1,0,1,0,1]
         IdArray scatter_index(IdArray partition_map,int num_partitions);
         IdArray index_select(IdArray values, IdArray index);
    // template<Devyce
        }
    }
}
