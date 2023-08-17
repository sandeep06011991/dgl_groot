//
// Created by juelin on 8/16/23.
//
#include <stdio.h>
#include <dgl/array.h>
#include <dgl/aten/array_ops.h>

#include "loc_dataloader.h"
int main(int argc, char * argv[]) {
  LOG(INFO) << "start initialing";
  using namespace dgl::groot;
  DGLContext ctx = DGLContext{kDGLCUDA, 0};
  DGLDataType dtype = DGLDataType {kDGLInt, 32, 1};
  auto device = dgl::runtime::DeviceAPI::Get(ctx);
  CHECK(device->IsAvailable()) << "device is not available";
  cudaStream_t stream = static_cast<cudaStream_t > (device->GetStream());
  auto arr = dgl::aten::NullArray();
  auto feat = dgl::runtime::NDArray::Empty({10, 10}, DGLDataType{kDGLFloat, 32, 1}, ctx);
  auto blocks = BlocksObject(ctx, {10, 10}, 1024, dtype, dtype, stream);
//  auto table = CudaHashTable(dtype, ctx, 1000 * 1024, stream);
  LOG(INFO) << "finished";
  return 0;
}