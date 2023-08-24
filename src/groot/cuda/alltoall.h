#ifndef DGL_DS_CUDA_ALLTOALL_H_
#define DGL_DS_CUDA_ALLTOALL_H_

#include <dgl/array.h>



namespace dgl {
namespace ds {

/**
 * @brief Unified alltoall communication.
 * 
 * @param input input on GPU
 * @param send_offset send_offset on GPU
 * @param rank
 * @param world_size
 * 
 * @return Tuple of (received buff, recv_sizes, recv_offset)
 */
// FIXME: wrap the low-level communicator
std::pair<IdArray, IdArray> Alltoall(IdArray input, IdArray send_offset,\
     int expand_size, int rank, int world_size, IdArray recv_offset=aten::NullArray());

}
}

#endif
