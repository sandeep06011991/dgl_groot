
#include <iostream>
#include <dgl/runtime/device_api.h>
#include <dgl/array.h>
#include <gtest/gtest.h>
#include "groot/array_scatter.h"
#include "groot/utils.h"
#include "groot/context.h"
#include "runtime/cuda/cuda_common.h"
#include "groot_dataloader/loc_dataloader.h"
#include <memory>
#include "groot/cuda/alltoall.h"
#include "groot_dataloader/cuda/cuda_hashtable.cuh"
#include "groot_dataloader/cuda/cuda_mapping.cuh"
#include <mpi.h>
#include "groot/context.h"
#include "groot/core.h"


using namespace dgl::ds;
using namespace dgl::aten;
using namespace dgl::groot;
using namespace dgl;
//    Test CUDAHashTable


int test_hash_table(){
  int rank = 0;
  auto context = DGLContext ({kDGLCUDA, rank});
  auto device_api = DeviceAPI::Get(context);
  CUDAThreadEntry::ThreadLocal()->stream =  static_cast<cudaStream_t>(device_api->CreateStream(context));
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;
  const size_t size = 6;
  std::vector<int32_t> array  = {5,6,7,5,6,7};
  auto dgl_array = IdArray::FromVector(array).CopyTo(context, stream);
  NDArray unique = IdArray::Empty({size}, DGLDataType { 0, 32, 1}, context);
  std::shared_ptr<CudaHashTable> _table = std::make_shared<CudaHashTable>(DGLDataType { 0, 32, 1},  context, 100, stream);
  _table->FillWithDuplicates( dgl_array, dgl_array->shape[0]);
  GPUMapNodes(dgl_array, unique, _table, stream );
//  _table->replace_nodes(dgl_array);
  cudaStreamSynchronize(stream);
//  std::cout << unique <<"\n";
  cudaStreamSynchronize(stream);
  std::cout << "testing.. hash_table \n";
}

int test_graph_sampling() {
  MPI_Init(NULL, NULL);
  // // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int rank = world_rank;
  auto context = DGLContext({kDGLCUDA, rank});
  auto device_api = DeviceAPI::Get(context);
  CUDAThreadEntry::ThreadLocal()->stream =
      static_cast<cudaStream_t>(device_api->CreateStream(context));
  CUDAThreadEntry::ThreadLocal()->thread_id = 0;
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;
  int thread_num = 1;
  bool enable_kernel_control = false;
  bool enable_comm_control = false;
  bool enable_profiler = false;

  ds::Initialize(
      world_rank, world_size, thread_num, enable_kernel_control,
      enable_comm_control, enable_profiler);

  std::vector<long> _indptrv;
  std::vector<long> _indicesv;
  std::vector<float> _featv;
  std::vector<long> _labelsv;
  std::vector<long> _seedsv;
  std::vector<long> _pmapv;
  int max_pool_size = 1;
  int num_nodes = 8;

  int batch_size = num_nodes;
  int feat_dim = 32;

  _indptrv.push_back(0);
  int edges = 0;

  for (int i = 0; i < num_nodes; i++) {
    edges += (num_nodes - 1);
    _indptrv.push_back(edges);
    for (int j = 0; j < num_nodes; j++) {
      if (i == j) continue;
      _indicesv.push_back(j);
    }
    for (int f = 0; f < feat_dim; f++) {
      _featv.push_back(i * 1.0);
    }
    _labelsv.push_back(1);
    _pmapv.push_back(i % 4);
  }
  _seedsv.push_back(rank);
  NDArray indptr = IdArray::FromVector(_indptrv).CopyTo(context, stream);
  std::cout << indptr << "\n";
  NDArray indices = IdArray::FromVector(_indicesv).CopyTo(context, stream);
  std::cout << indices << "\n";
  NDArray feat = IdArray::FromVector(_featv)
                     .CopyTo(context, stream)
                     .CreateView(
                         std::vector<int64_t>{num_nodes, feat_dim},
                         DGLDataType{2, 32, 1}, 0);
  NDArray seeds = IdArray::FromVector(_seedsv).CopyTo(context, stream);
  NDArray labels = IdArray::FromVector(_labelsv).CopyTo(context, stream);
  NDArray pmap = IdArray::FromVector(_pmapv).CopyTo(context, stream);
  std::cout << pmap << "\n";
  auto blockType = BlocksObject::BlockType::SRC_TO_DEST;
  groot::DataloaderObject obj(
      context, indptr, indices, feat, labels, seeds, pmap,
      std::vector<int64_t>{10, 10}, batch_size, max_pool_size, blockType, 1);
  std::cout << "Object created Check ! \n";
  auto key = obj.SyncSample();
  auto blocks = obj.AwaitGetBlocks(key);
  std::cout << blocks->_output_nodes << "\n";
  //  std::cout << "All done for single layer sample\n";
  MPI_Finalize();
  return 0;
}

void test_scatter(int rank, int world_size){
  cudaSetDevice(rank);
  auto context = DGLContext ({kDGLCUDA, rank});
  auto device_api = DeviceAPI::Get(context);
  CUDAThreadEntry::ThreadLocal()->stream =  static_cast<cudaStream_t>(device_api->CreateStream(context));
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;
  CUDAThreadEntry::ThreadLocal()->thread_id=0;

  int thread_num = 1;
  bool enable_kernel_control = false;
  bool enable_comm_control = false;
  bool enable_profiler = false;
  ds::Initialize(rank, world_size, thread_num, \
                 enable_kernel_control, enable_comm_control, enable_profiler);

  ScatteredArray array = ScatteredArray::Create(100, 4 , context, DGLDataType {0 , 64, 1}, stream);
  std::vector<long> _frontier_v {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  NDArray frontier = NDArray::FromVector<int64_t>(_frontier_v).CopyTo(context, stream);
  std::vector<long> _partition_map{0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
  NDArray partition_map = dgl::runtime::NDArray::FromVector(_partition_map).\
                            CopyTo(context, stream);;
  int num_partitions = 4;
  assert(partition_map->dtype.bits == 64);
  cudaStreamSynchronize(stream);

  std::vector<long> _offsets {0,4,8,12,16};
  NDArray offsets = NDArray::FromVector(_offsets).CopyTo(context,stream);
//  cudaStreamSynchronize(stream);
//  cudaDeviceSynchronize();

  std::cout << "Todo: Scatter function also takes in a stream";
  groot::Scatter(array, frontier, partition_map, num_partitions, rank, world_size);

  cudaDeviceSynchronize();
}

// Note:
// Always keep mpi template the same
int main() {
    int initialized, finalized;
    //  MPI_Initialized(&initialized);
    //  if (!initialized)
    MPI_Init(NULL, NULL);
    // // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    test_scatter(world_rank, world_size);
//  test_graph_sampling()
//  You also need this when your program is about to exit
    MPI_Finalize();
}


