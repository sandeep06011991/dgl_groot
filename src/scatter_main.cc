
#include <iostream>
#include <dgl/runtime/device_api.h>
#include <dgl/array.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <chrono>
#include <thread>
#include "groot/cuda/array_scatter.h"
#include "groot/array_scatter.h"
#include "groot/utils.h"
//#include "./test_utils.h"
#include "groot/context.h"
#include "runtime/cuda/cuda_common.h"
#include "groot_dataloader/loc_dataloader.h"
#include <memory>

#include "groot_dataloader/cuda/cuda_hashtable.cuh"
#include "groot_dataloader/cuda/cuda_mapping.cuh"
#include <mpi.h>

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

int main_v3(){
  MPI_Init(NULL, NULL);
  // // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int rank = world_rank;
  std::cout << "World rank " << rank <<"\n";
  auto context = DGLContext ({kDGLCUDA, rank});
  auto device_api = DeviceAPI::Get(context);
  CUDAThreadEntry::ThreadLocal()->stream =  static_cast<cudaStream_t>(device_api->CreateStream(context));
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;

  std::vector<int> _indptrv;
  std::vector<int> _indicesv;
  std::vector<float> _featv;
  std::vector<int> _labelsv;
  std::vector<int> _seedsv;
  std::vector<int> _pmapv;
  int max_pool_size = 1;
  int num_nodes = 8;

  int batch_size = num_nodes;
  int feat_dim = 32;

  _indptrv.push_back(0);
  int edges = 0;

  for(int i = 0; i < num_nodes; i ++){
    edges += (num_nodes - 1);
    _indptrv.push_back(edges);
    for(int j = 0; j < num_nodes; j ++){
      if (i == j)continue;
      _indicesv.push_back(j);
    }
    for(int f = 0; f < feat_dim; f++){
      _featv.push_back(i * 1.0);
    }
    _labelsv.push_back(1);
    _pmapv.push_back(i % 4);
  }
  _seedsv.push_back(rank);
  NDArray indptr = IdArray::FromVector(_indptrv).CopyTo(context, stream);
  std::cout << indptr <<"\n";
  NDArray indices = IdArray::FromVector(_indicesv).CopyTo(context, stream);
  std::cout << indices <<"\n";
  NDArray feat = IdArray::FromVector(_featv).CopyTo(context,stream).CreateView(std::vector<int64_t>{num_nodes, feat_dim},
      DGLDataType{2,32,1}, 0);
  NDArray seeds = IdArray::FromVector(_seedsv).CopyTo(context, stream);
  NDArray labels = IdArray::FromVector(_labelsv).CopyTo(context, stream);
  NDArray pmap = IdArray::FromVector(_pmapv).CopyTo(context, stream);
  std::cout << pmap <<"\n";
   auto blockType = BlocksObject::BlockType::SRC_TO_DEST;

  groot::DataloaderObject obj(
      context, indptr, indices, feat, labels, seeds, pmap,\
          std::vector<int64_t>{10}, batch_size, max_pool_size, blockType, 1);
          std::cout << "Object created Check ! \n";
//  auto key = obj.SyncSample();
//  auto blocks = obj.AwaitGetBlocks(key);
//  std::cout << "All done for single layer sample\n";
}

int main_v2(){
  int rank  = 0;
  std::cout << "Hello World\n";
  //Create an array of 100000 Elements NDARray
  auto context = DGLContext ({kDGLCUDA, rank});
  auto device_api = DeviceAPI::Get(context);
  CUDAThreadEntry::ThreadLocal()->stream =  static_cast<cudaStream_t>(device_api->CreateStream(context));
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;
  int size = 1000;
  int num_partitions = 4;
  std::vector<int32_t> array(size, 0);
  std::vector<int32_t> pmap(size, 0);
  for(int i = 0; i < size; ++i) {
    array[i] = i ;
    pmap[i] = i % num_partitions;
  }
  auto dgl_array = IdArray::FromVector(array).CopyTo(context, stream);
  auto dgl_array_idx = IdArray::FromVector(array).CopyTo(context, stream);
  auto dgl_pmap = IdArray::FromVector(pmap).CopyTo(context, stream);
  std::cout << "check\n";
  auto scattered_index = groot::scatter_index(dgl_pmap, 4);
  std::cout << scattered_index <<"\n";

  auto partitioned_out = groot::gatherArray(dgl_array, scattered_index, dgl_array_idx,  num_partitions);
  auto offsets = groot::getBoundaryOffsets(scattered_index, num_partitions);
  cudaDeviceSynchronize();
//  std::cout << "Start index select" << offsets <<"\n";
  std::cout << partitioned_out <<"\n";

  //Call functional required object
  std::cout << "all done\n";

}
















int main_v1(){
  MPI_Init(NULL, NULL);
  // // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int rank = world_rank;
  std::cout << "World rank " << rank <<"\n";
  auto context = DGLContext ({kDGLCUDA, rank});
  auto device_api = DeviceAPI::Get(context);
  CUDAThreadEntry::ThreadLocal()->stream =  static_cast<cudaStream_t>(device_api->CreateStream(context));
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;

  cudaSetDevice(rank);
  ScatteredArray array = ScatteredArray::Create();
  std::vector<int> _frontier_v {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  NDArray frontier = NDArray::FromVector(_frontier_v).CopyTo(context, stream);
  NDArray partition_map = dgl::runtime::NDArray::FromVector(std::vector<int>{0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3}).\
                            CopyTo(context, stream);;
  int num_partitions = 4;
  std::cout << "Start scatterring \n";
  cudaStreamSynchronize(stream);
  std::cout << "Stream synchronize ok !\n";
  groot::Scatter(array, frontier, partition_map, num_partitions, rank, world_size);
  cudaDeviceSynchronize();
  std::cout << "All done\n";
}











