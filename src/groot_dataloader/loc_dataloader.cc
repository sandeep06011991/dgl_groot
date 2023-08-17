//
// Created by juelin on 8/6/23.
//

#include "loc_dataloader.h"
namespace dgl {
namespace groot {
using namespace runtime;
DGL_REGISTER_GLOBAL("groot._CAPI_CreateLocDataloader")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      int device_id = args[0];
      DGLContext ctx{kDGLCUDA, device_id};
      NDArray indptr = args[1];
      NDArray indices = args[2];
      NDArray feats = args[3];
      NDArray labels = args[4];
      NDArray seeds = args[5];
      List<Value> fanout_list = args[6];
      std::vector<int64_t> fanouts;
      for (const auto& fanout : fanout_list) {
        fanouts.push_back(static_cast<int64_t>(fanout->data));
      }
      int64_t batch_size = args[7];
      int64_t max_pool_size = args[8];

      auto o = std::make_shared<LocDataloaderObject>(
          ctx, indptr, indices, feats, labels, seeds, fanouts, batch_size,
          max_pool_size);
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
      List<Value> fanout_list = args[6];
      std::vector<int64_t> fanouts;
      for (const auto& fanout : fanout_list) {
        fanouts.push_back(static_cast<int64_t>(fanout->data));
      }
      int64_t batch_size = args[7];
      int64_t max_pool_size = args[8];

      LocDataloaderObject::Global()->Init(
          ctx, indptr, indices, feats, labels, seeds, fanouts, batch_size,
          max_pool_size);

      *rv = LocDataloaderObject::Global();
    });

DGL_REGISTER_GLOBAL("groot._CAPI_Next")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = LocDataloaderObject::Global()->Sample();
      *rv = key;
    });


DGL_REGISTER_GLOBAL("groot._CAPI_GetBlock")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      const int64_t layer = args[1];
      *rv = LocDataloaderObject::Global()->GetBlocks(key)->GetBlock(layer)->_block_ref;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetBlocks")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      *rv = LocDataloaderObject::Global()->GetBlocks(key);
    });


DGL_REGISTER_GLOBAL("groot._CAPI_GetInputNodes")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      *rv = LocDataloaderObject::Global()->GetBlocks(key)->_input_nodes;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetInputNodeLabels")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      *rv = LocDataloaderObject::Global()->GetBlocks(key)->_labels;
    });
DGL_REGISTER_GLOBAL("groot._CAPI_GetOutputNodes")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      *rv = LocDataloaderObject::Global()->GetBlocks(key)->_output_nodes;
    });

DGL_REGISTER_GLOBAL("groot._CAPI_GetOutputNodeFeats")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t key = args[0];
      *rv = LocDataloaderObject::Global()->GetBlocks(key)->_feats;
    });

}  // namespace groot

}  // namespace dgl
