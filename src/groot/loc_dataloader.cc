//
// Created by juelin on 8/6/23.
//

#include "loc_dataloader.h"

namespace dgl {
    namespace groot {
        using namespace runtime;
        DGL_REGISTER_GLOBAL("groot._CAPI_CreateDataloader")
        .set_body([](DGLArgs args, DGLRetValue *rv) {
            int device_id = args[0];
            DGLContext ctx{kDGLCUDA, device_id};
            NDArray indptr = args[1];
            NDArray indices = args[2];
            NDArray feats = args[3];
            NDArray labels = args[4];
            NDArray seeds = args[5];
            List<Value> fanout_list = args[6];
            std::vector<int64_t> fanouts;
            for (const auto &fanout: fanout_list) {
                fanouts.push_back(static_cast<int64_t>(fanout->data));
            }
            int64_t batch_size = args[7];
            int64_t max_pool_size = args[8];

            auto o = std::make_shared<DataloaderObject>(
                    ctx, indptr, indices, feats, labels, seeds, fanouts, batch_size,
                    max_pool_size);
            *rv = o;
        });

        DGL_REGISTER_GLOBAL("groot._CAPI_InitDataloader")
        .set_body([](DGLArgs args, DGLRetValue *rv) {
            int64_t device_id = args[0];
            DGLContext ctx{kDGLCUDA, (int32_t) device_id};
            NDArray indptr = args[1];
            NDArray indices = args[2];
            NDArray feats = args[3];
            NDArray labels = args[4];
            NDArray seeds = args[5];
            List<Value> fanout_list = args[6];
            std::vector<int64_t> fanouts;
            for (const auto &fanout: fanout_list) {
                fanouts.push_back(static_cast<int64_t>(fanout->data));
            }
            int64_t batch_size = args[7];
            int64_t max_pool_size = args[8];

            DataloaderObject::Global()->Init(
                    ctx, indptr, indices, feats, labels, seeds, fanouts, batch_size,
                    max_pool_size);

            *rv = DataloaderObject::Global();
        });

        DGL_REGISTER_GLOBAL("groot._CAPI_NextAsync")
        .set_body([](DGLArgs args, DGLRetValue *rv) {
            const int64_t key = DataloaderObject::Global()->AsyncSample();
            *rv = key;
        });

        DGL_REGISTER_GLOBAL("groot._CAPI_NextSync")
        .set_body([](DGLArgs args, DGLRetValue *rv) {
            const int64_t key = DataloaderObject::Global()->SyncSample();
            *rv = key;
        });

        DGL_REGISTER_GLOBAL("groot._CAPI_GetBlock")
        .set_body([](DGLArgs args, DGLRetValue *rv) {
            const int64_t key = args[0];
            const int64_t layer = args[1];
            *rv = DataloaderObject::Global()->AwaitGetBlocks(key)->GetBlock(layer)->_block_ref;
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

    }  // namespace groot

}  // namespace dgl
