#include "./alltoall.h"

#include <dmlc/logging.h>
#include <thread>
#define _CG_ABI_EXPERIMENTAL // enable experimental API
#include <cooperative_groups.h>

// #include "../comm/comm_info.h"
#include <dgl/runtime/c_runtime_api.h>
#include "../utils.h"
#include "../../runtime/cuda/cuda_common.h"
#include <nccl.h>
#include "../context.h"
// #include "./ds_kernel.h"
// #include "../schedule.h"

using namespace dgl::runtime;
using namespace cooperative_groups;


using IdType = int64_t;

namespace dgl {
namespace ds {

__global__ 
void _DiffKernel(IdType* out, IdType* in, int size) {
  int tid = threadIdx.x;
  if(tid < size) {
    out[tid] = in[tid + 1] - in[tid];
  }
}

IdArray Diff(IdArray prefix_sum) {
  auto stream = CUDAThreadEntry::ThreadLocal()->stream;
  int size = prefix_sum->shape[0] - 1;
  IdArray ret = IdArray::Empty({size}, prefix_sum->dtype, prefix_sum->ctx);
  _DiffKernel<<<1, 32, 0, stream>>>(ret.Ptr<IdType>(), prefix_sum.Ptr<IdType>(), size);
  CUDACHECK(cudaGetLastError());
  return ret;
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void NCCLAllToAll(IdArray send_buffer, IdArray send_offset, IdArray recv_buffer,\
 IdArray recv_offset, int expand_size, int rank, int world_size, ncclComm_t nccl_comm) {
  auto stream = CUDAThreadEntry::ThreadLocal()->stream;
  auto data_copy_stream = CUDAThreadEntry::ThreadLocal()->data_copy_stream;
  T* send_buffer_ptr = send_buffer.Ptr<T>();
  T* recv_buffer_ptr = recv_buffer.Ptr<T>();
  int type_bytes = sizeof(T);
  IdType* send_offset_ptr = send_offset.Ptr<IdType>();
  IdType* recv_offset_ptr = recv_offset.Ptr<IdType>();
  ncclGroupStart();
  for(int r = 0; r < world_size; ++r) {
    if(r != rank) {
      IdType send_size = (send_offset_ptr[r+1] - send_offset_ptr[r]) * expand_size;
      IdType send_ptr = send_offset_ptr[r] * expand_size;
      IdType recv_size = (recv_offset_ptr[r+1] - recv_offset_ptr[r]) * expand_size;
      IdType recv_ptr = recv_offset_ptr[r] * expand_size;
      ncclSend(send_buffer_ptr + send_ptr, send_size, NCCL_DATA_TYPE, r, nccl_comm, stream);
      ncclRecv(recv_buffer_ptr + recv_ptr, recv_size, NCCL_DATA_TYPE, r, nccl_comm, stream);
    }
  }
  ncclGroupEnd();

  CUDACHECK(cudaMemcpyAsync(recv_buffer_ptr + recv_offset_ptr[rank] * expand_size, 
                       send_buffer_ptr + send_offset_ptr[rank] * expand_size, 
                       (send_offset_ptr[rank + 1] - send_offset_ptr[rank]) * expand_size * type_bytes, cudaMemcpyDeviceToDevice, data_copy_stream));
  CUDACHECK(cudaStreamSynchronize(data_copy_stream));
}

std::pair<IdArray, IdArray> Alltoall(IdArray input, IdArray send_offset,\
           int expand_size, int rank, int world_size, IdArray recv_offset, cudaStream_t stream) {
    // NCCL
    CHECK(send_offset->dtype.bits == 64);
    auto send_sizes = Diff(send_offset);
    cudaStream_t data_copy_stream;
    if(stream == nullptr){
      stream = CUDAThreadEntry::ThreadLocal()->stream;
      data_copy_stream = CUDAThreadEntry::ThreadLocal()->data_copy_stream;
    }else{
      std::cout << "using runtime stream \n";
      data_copy_stream = stream;
    }
    auto dgl_context = input->ctx;
    auto *ds_context = DSContext::Global();
    auto host_dgl_context = DGLContext{kDGLCPU, 0};
    int comm_token = CUDAThreadEntry::ThreadLocal()->thread_id;
    int thread_id = CUDAThreadEntry::ThreadLocal()->thread_id;

    ncclComm_t nccl_comm = ds_context->nccl_comm[thread_id];
    auto* ds_thread_local = DSThreadEntry::ThreadLocal();

    // NOTE: to guarantee the send_offset is ready
    CUDACHECK(cudaStreamSynchronize(stream));
    // auto host_send_offset = send_offset.CopyTo(host_dgl_context, data_copy_stream);
    auto host_send_offset = CopyArrayToPinned(send_offset, stream);
    CUDACHECK(cudaStreamSynchronize(data_copy_stream));

    if(IsNullArray(recv_offset)) {
      IdArray recv_sizes = IdArray::Empty({world_size}, send_offset->dtype, dgl_context);
      IdArray range_seq = Range(0, world_size + 1, 64, host_dgl_context);
      // scheduler->TryComm(thread_id);
      NCCLAllToAll<int64_t, ncclInt64>(send_sizes, range_seq, \
                                       recv_sizes, range_seq, 1, rank, world_size, nccl_comm);
      CUDACHECK(cudaStreamSynchronize(stream));
      // scheduler->FinishComm();
      recv_offset = CumSum(recv_sizes, true);
    }

    //IdArray host_recv_offset = recv_offset.CopyTo(host_dgl_context, stream);
    auto host_recv_offset = CopyArrayToPinned(recv_offset, stream);
    CUDACHECK(cudaStreamSynchronize(stream));
    auto* host_recv_offset_ptr = host_recv_offset.Ptr<IdType>();
    int n_recv = host_recv_offset_ptr[world_size] * expand_size;
    auto recvbuff = IdArray::Empty({n_recv}, input->dtype, dgl_context);

    //   scheduler->TryComm(thread_id);
    if(input->dtype.bits == 32) {
      NCCLAllToAll<int, ncclInt32>(input, host_send_offset, recvbuff, host_recv_offset, expand_size, rank, world_size, nccl_comm);
    } else {
      NCCLAllToAll<int64_t, ncclInt64>(input, host_send_offset, recvbuff, host_recv_offset, expand_size, rank, world_size, nccl_comm);
    }
    CUDACHECK(cudaStreamSynchronize(stream));
    // scheduler->FinishComm();
    return {recvbuff, recv_offset};
}

}
}
