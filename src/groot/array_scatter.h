#ifndef DGL_GROOT_ARRAY_SCATTER_H_
#define DGL_GROOT_ARRAY_SCATTER_H_

#include <dgl/array.h>

namespace dgl{
    namespace groot{
        IdArray   getBoundaryOffsets(IdArray index_cum_sums, int num_partitions);

        namespace impl{
            template<DGLDeviceType XPU, typename IdType>
            IdArray getBoundaryOffsetsLocal(IdArray index_cum_sums, int num_partitions);
        }
    }
}



#endif  // DGL_GROOT_ARRAY_SCATTER_H_