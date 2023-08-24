
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

using namespace dgl::ds;
using namespace dgl;
using namespace dgl::aten;
using namespace dgl;

int main(){
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
  auto dgl_pmap = IdArray::FromVector(pmap).CopyTo(context, stream);
  std::cout << "check\n";
  auto scattered_index = groot::impl::scatter_index(dgl_pmap, 4);
  std::cout << scattered_index <<"\n";

  auto partitioned_out = groot::gatherArray(dgl_array, scattered_index, num_partitions);
  auto offsets = groot::getBoundaryOffsets(scattered_index, num_partitions);
  cudaDeviceSynchronize();
  std::cout << "Start index select" << offsets <<"\n";
  std::cout << partitioned_out <<"\n";

  //Call functional required object
  std::cout << "all done\n";

}