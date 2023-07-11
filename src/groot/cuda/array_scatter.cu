
#include "array_scatter.h"



namespace dgl{
    namespace groot{

        template<IdType>
        ___global__
        void array_scatter_index_in(IdType *values, size_t sz, \
            IdType *index, int n_partitions){
        
        }

        IdArray scatter_index(IdArray src, IdArray part ){

            array_scatter_index_in<int>(src.Ptr<int>(), 10, index.Ptr<int>(),10);
        }

    }
}


