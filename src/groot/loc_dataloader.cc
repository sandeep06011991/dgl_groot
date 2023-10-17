//
// Created by juelin on 8/6/23.
//

#include "loc_dataloader.h"

namespace dgl::groot {
using namespace runtime;

DGL_REGISTER_GLOBAL("groot._CAPI_InitDataloader")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int idx = 0;
      int64_t rank = args[idx++];
      int64_t world_size = args[idx++];
      int64_t block_type = args[idx++];
      int64_t device_id = args[idx++];
      List<Value> fanout_list = args[idx++];
      int64_t batch_size = args[idx++];
      int64_t num_redundant_layers = args[idx++];
      int64_t max_pool_size = args[idx++];

      NDArray indptr = args[idx++];
      NDArray indices = args[idx++];
      NDArray feats = args[idx++];
      NDArray labels = args[idx++];
      NDArray train_idx = args[idx++];
      NDArray valid_idx = args[idx++];
      NDArray test_idx = args[idx++];
      NDArray partition_map = args[idx++];

      std::vector<int64_t> fanouts;
      for (const auto &fanout : fanout_list) {
        fanouts.insert(fanouts.begin(), static_cast<int64_t>(fanout->data));
      }

      DGLContext ctx{kDGLCUDA, (int32_t)device_id};

      DataloaderObject::Global()->Init(
          rank, world_size, block_type, ctx, fanouts, batch_size, num_redundant_layers, max_pool_size, indptr,
          indices, feats, labels, train_idx, valid_idx, test_idx, partition_map);

      *rv = DataloaderObject::Global();
    });

DGL_REGISTER_GLOBAL("groot._CAPI_NextAsync")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = DataloaderObject::Global()->AsyncSample();
      *rv = key;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_ShuffleIDX")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray train_idx = args[0];
      DataloaderObject::Global()->ShuffleTrainingNodes(train_idx);
    });


DGL_REGISTER_GLOBAL("groot._CAPI_NextSync")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = DataloaderObject::Global()->Sample();
      *rv = key;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetBlock")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = args[0];
      const int64_t layer = args[1];
      *rv = DataloaderObject::Global()
                ->AwaitGetBlocks(key)
                ->GetBlock(layer)
                ->_block_ref;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetBlocksUniqueId")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = args[0];
      const int64_t layer = args[1];
      *rv = DataloaderObject::Global()
                ->AwaitGetBlocks(key)
                ->GetBlock(layer)
                ->_true_node_ids;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetBlockScatteredSrc")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = args[0];
      const int64_t layer = args[1];
      *rv = DataloaderObject::Global()
                ->AwaitGetBlocks(key)
                ->GetBlock(layer)
                ->_scattered_src;
    });

//DGL_REGISTER_GLOBAL("groot._CAPI_GetBlockScatteredDest")
//    .set_body([](DGLArgs args, DGLRetValue *rv) {
//      const int64_t key = args[0];
//      const int64_t layer = args[1];
//      *rv = DataloaderObject::Global()
//                ->AwaitGetBlocks(key)
//                ->GetBlock(layer)
//                ->_scattered_dest;
//    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetBlocksFrontier")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = args[0];
      const int64_t layer = args[1];
      *rv = DataloaderObject::Global()
                ->AwaitGetBlocks(key)
                ->_scattered_frontier;
    });


DGL_REGISTER_GLOBAL("groot._CAPI_GetBlocks")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = args[0];
      *rv = DataloaderObject::Global()->AwaitGetBlocks(key);
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetInputNode")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = args[0];
      *rv = DataloaderObject::Global()->AwaitGetBlocks(key)->_input_nodes;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetLabel")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = args[0];
      *rv = DataloaderObject::Global()->AwaitGetBlocks(key)->_labels;
    });
DGL_REGISTER_GLOBAL("groot._CAPI_GetOutputNode")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = args[0];
      *rv = DataloaderObject::Global()->AwaitGetBlocks(key)->_output_nodes;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetFeat")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const int64_t key = args[0];
      *rv = DataloaderObject::Global()->AwaitGetBlocks(key)->_feats;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_InitCache")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray cached_ids = args[0];
      DataloaderObject::Global()->InitFeatCache(cached_ids);
    });

} // namespace dgl::groot
