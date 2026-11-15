//
// Created by juelin on 2/14/24.
//

#include "../../runtime/cuda/cuda_common.h"
#include "./all2all.h"

/**
 * Dispatch according to data type (int32, int64, float32 or float64):
 *
 * ATEN_DTYPE_SWITCH(array->dtype, DType, {
 *   // Now DType is the type corresponding to data type in array.
 *   // For instance, one can do this for a CPU array:
 *   DType *data = static_cast<DType *>(array->data);
 *   NCCL_DATA_TYPE is assigned to the associated nccl data type
 * });
 */
#define ATEN_NCCL_TYPE_SWITCH(val, DType, ...)                       \
  do {                                                               \
    if ((val).code == kDGLInt && (val).bits == 32) {                 \
      typedef int32_t DType;                                         \
      auto NCCL_DATA_TYPE = ncclInt32;                               \
      { __VA_ARGS__ }                                                \
    } else if ((val).code == kDGLInt && (val).bits == 64) {          \
      typedef int64_t DType;                                         \
      auto NCCL_DATA_TYPE = ncclInt64;                               \
      { __VA_ARGS__ }                                                \
    } else if ((val).code == kDGLFloat && (val).bits == 32) {        \
      typedef float DType;                                           \
      auto NCCL_DATA_TYPE = ncclFloat32;                             \
      { __VA_ARGS__ }                                                \
    } else if ((val).code == kDGLFloat && (val).bits == 64) {        \
      typedef double DType;                                          \
      auto NCCL_DATA_TYPE = ncclFloat64;                             \
      { __VA_ARGS__ }                                                \
    } else {                                                         \
      LOG(FATAL) << " can only be int32, int64, float32 or float64"; \
    }                                                                \
  } while (0)

namespace dgl::dev {
template <typename IdType>
__global__ void DiffKernel(IdType* out, const IdType* in, int64_t size) {
  int tid = threadIdx.x;
  if (tid < size) {
    out[tid] = in[tid + 1] - in[tid];
  }
}

IdArray Diff(const IdArray& prefix_sum) {
  auto stream = runtime::getCurrentCUDAStream();
  int64_t size = prefix_sum->shape[0] - 1;
  IdArray ret = IdArray::Empty({size}, prefix_sum->dtype, prefix_sum->ctx);
  ATEN_ID_TYPE_SWITCH(ret->dtype, IdType, {
    CUDA_KERNEL_CALL(
        DiffKernel, 1, 256, 0, stream, ret.Ptr<IdType>(),
        prefix_sum.Ptr<IdType>(), size);
  });
  return ret;
}

NDArray NCCLAllToAll(
    int64_t rank, int64_t world_size, const NDArray& input,
    const IdArray& send_indptr, const IdArray& recv_indptr, cudaStream_t stream,
    ncclComm_t nccl_comm) {
  using IdType = int64_t;
  CHECK_EQ(send_indptr->ctx, recv_indptr->ctx);
  const auto expand_size = input.NumElements() / input->shape[0];
  const auto send_offset_ptr = send_indptr.Ptr<IdType>();
  const auto recv_offset_ptr = recv_indptr.Ptr<IdType>();
  const auto recv_num = recv_indptr.Ptr<IdType>()[world_size] * expand_size;
  NDArray ret = NDArray::Empty({recv_num}, input->dtype, input->ctx);
//  LOG(INFO) << "rank: " << rank << " ret buffer size " << recv_num << " ctx: " << ret->ctx;

  ATEN_NCCL_TYPE_SWITCH(input->dtype, DType, {
    const auto* send_buffer_ptr = input.Ptr<DType>();
    auto* recv_buffer_ptr = ret.Ptr<DType>();
    ncclGroupStart();
    for (int r = 0; r < world_size; r++) {
      IdType send_size =
          (send_offset_ptr[r + 1] - send_offset_ptr[r]) * expand_size;
      IdType send_ptr = send_offset_ptr[r] * expand_size;
      IdType recv_size =
          (recv_offset_ptr[r + 1] - recv_offset_ptr[r]) * expand_size;
      IdType recv_ptr = recv_offset_ptr[r] * expand_size;
//      LOG(INFO) << "rank " << rank << "recv_ptr " << recv_ptr << " send_ptr " << send_ptr;
      ncclSend(
          send_buffer_ptr + send_ptr, send_size, NCCL_DATA_TYPE, r, nccl_comm,
          stream);
      ncclRecv(
          recv_buffer_ptr + recv_ptr, recv_size, NCCL_DATA_TYPE, r, nccl_comm,
          stream);
    }
    ncclGroupEnd();
  });

//  CUDA_CALL(cudaStreamSynchronize(stream));
  return ret;
}

std::pair<IdArray, IdArray> Alltoall(
    int64_t rank, int64_t world_size, const IdArray& input,
    const IdArray& send_offset, IdArray recv_offset, cudaStream_t stream,
    ncclComm_t nccl_comm) {
  // NCCL
  CHECK_NE(nccl_comm, nullptr) << "Must call initNccl after spawn the process";
  CHECK(send_offset->data != nullptr);
  auto send_sizes = Diff(send_offset);
  auto host_ctx = DGLContext{kDGLCPU, 0};
  auto host_send_offset = send_offset.CopyTo(host_ctx); // blocking
//  LOG(INFO) << "send sizes: " << send_sizes;
//  CUDA_CALL(cudaStreamSynchronize(stream));

  if (aten::IsNullArray(recv_offset)) {
    auto comm_offset =
        aten::Range(0, world_size + 1, send_offset->dtype.bits, host_ctx);
    auto recv_sizes = NCCLAllToAll(
        rank, world_size, send_sizes, comm_offset, comm_offset, stream,
        nccl_comm);
    recv_offset = aten::CumSum(recv_sizes, true);
  }

  auto host_recv_offset = recv_offset.CopyTo(host_ctx); // blocking
//  LOG(INFO) << "recv offset: " << recv_offset;

  auto retbuff = NCCLAllToAll(
      rank, world_size, input, host_send_offset, host_recv_offset, stream,
      nccl_comm);
  return {retbuff, recv_offset};
}
}  // namespace dgl::dev