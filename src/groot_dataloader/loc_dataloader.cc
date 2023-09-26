 //
// Created by juelin on 8/6/23.
//

#include "loc_dataloader.h"
namespace dgl {
namespace groot {
using namespace runtime;


std::shared_ptr<DataloaderObject> DataloaderObject::single_instance = std::make_shared<DataloaderObject>();


DGL_REGISTER_GLOBAL("groot._CAPI_CreateLocDataloader")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      int device_id = args[0];
      DGLContext ctx{kDGLCUDA, device_id};
      NDArray indptr = args[1];
      NDArray indices = args[2];
      NDArray feats = args[3];
      NDArray labels = args[4];
      NDArray seeds = args[5];
      NDArray partition_map = args[6];
      List<Value> fanout_list = args[7];
      std::vector<int64_t> fanouts;
      for (const auto& fanout : fanout_list) {
        fanouts.push_back(static_cast<int64_t>(fanout->data));
      }
      BlocksObject::BlockType blocktypes;

      int64_t batch_size = args[8];
      int64_t max_pool_size = args[9];
      int64_t n_redundant_layers = args[10];
      int64_t block_type = args[11];
      if(block_type == 0){
        std::cout << "Creating block object data parallel \n";
        blocktypes = BlocksObject::BlockType::DATA_PARALLEL;
      }
      if(block_type == 1){
        std::cout << "Creating src to dest \n";
        blocktypes = BlocksObject::BlockType::SRC_TO_DEST;
      }
      if(block_type == 1) {
        blocktypes == BlocksObject::BlockType::DEST_TO_SRC;
      }

      auto device_api = DeviceAPI::Get(ctx);
      CUDAThreadEntry::ThreadLocal()->stream =  static_cast<cudaStream_t>(device_api->CreateStream(ctx));
      auto o = std::make_shared<DataloaderObject>(
          ctx, indptr, indices, feats, labels, seeds, partition_map, fanouts, batch_size,
          max_pool_size, blocktypes, n_redundant_layers);
      *rv = o;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_InitLocDataloader")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      int64_t device_id = args[0];
      DGLContext ctx{kDGLCUDA, (int32_t ) device_id};
      NDArray indptr = args[1];
      NDArray indices = args[2];
      NDArray feats = args[3];
      NDArray labels = args[4];
      NDArray seeds = args[5];
      NDArray partition_map = args[6];
      List<Value> fanout_list = args[7];
      std::vector<int64_t> fanouts;
      BlocksObject::BlockType blocktypes;
      std::cout << "How to pass python enums here ?\n";

      for (const auto& fanout : fanout_list) {
        fanouts.push_back(static_cast<int64_t>(fanout->data));
      }
      int64_t batch_size = args[8];

      int64_t max_pool_size = args[9];
      int64_t n_redundant_layers = args[10];
      int64_t block_type = args[11];
      if(block_type == 0){
        std::cout << "Creating block object data parallel \n";
        blocktypes = BlocksObject::BlockType::DATA_PARALLEL;
      }
      if(block_type == 1){
        std::cout << "Creating src to dest \n";
        blocktypes = BlocksObject::BlockType::SRC_TO_DEST;
      }
      if(block_type == 1) {
        blocktypes == BlocksObject::BlockType::DEST_TO_SRC;
      }

      auto device_api = DeviceAPI::Get(ctx);
      CUDAThreadEntry::ThreadLocal()->stream =  static_cast<cudaStream_t>(device_api->CreateStream(ctx));
      DataloaderObject::Global()->Init(
          ctx, indptr, indices, feats, labels, seeds, partition_map, fanouts, batch_size,
          max_pool_size, blocktypes, n_redundant_layers);

      *rv = DataloaderObject::Global();
    });

DGL_REGISTER_GLOBAL("groot._CAPI_NextAsync")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = DataloaderObject::Global()->AsyncSample();
      *rv = key;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_NextSync")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = DataloaderObject::Global()->SyncSample();
      *rv = key;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetBlock")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      const int64_t layer = args[1];
      *rv = DataloaderObject::Global()
                ->AwaitGetBlocks(key)->GetBlock(layer)->_block_ref;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetBlocks")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      *rv = DataloaderObject::Global()->AwaitGetBlocks(key);
    });


DGL_REGISTER_GLOBAL("groot._CAPI_GetInputNodes")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      *rv = DataloaderObject::Global()->AwaitGetBlocks(key)->_input_nodes;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetInputNodeLabels")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      *rv = DataloaderObject::Global()->AwaitGetBlocks(key)->_labels;
    });
DGL_REGISTER_GLOBAL("groot._CAPI_GetOutputNodes")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      *rv = DataloaderObject::Global()->AwaitGetBlocks(key)->_output_nodes;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetOutputNodeFeats")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      *rv = DataloaderObject::Global()->AwaitGetBlocks(key)->_feats;
    });

}  // namespace groot

}  // namespace dgl
