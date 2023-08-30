//
// Created by juelin on 8/6/23.
//

#ifndef DGL_LOC_DATALOADER_H
#define DGL_LOC_DATALOADER_H

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>

#include <memory>
#include <mutex>

#include "../c_api_common.h"
#include "block.h"
#include "cuda/cuda_index_select.cuh"
#include "cuda/cuda_mapping.cuh"
#include "cuda/rowwise_sampling.cuh"

namespace dgl {
    namespace groot {
        class DataloaderObject : public runtime::Object {
        public:
            std::vector<std::shared_ptr<BlocksObject>> _blocks_pool;
            std::vector<cudaStream_t> _sampling_streams;
            std::vector<cudaStream_t> _cpu_feat_streams;
            std::vector<cudaStream_t> _gpu_feat_streams;
            std::vector<bool> _syncflags;
            std::vector<int64_t> _fanouts;

            NDArray _indptr;
            NDArray _indices;
            NDArray _seeds;
            NDArray _cpu_feats;
            NDArray _gpu_feats;
            NDArray _labels;

            int64_t _max_pool_size;
            int64_t _batch_size;
            int64_t _next_key;
            int64_t _num_seeds;
            int64_t _table_capacity;

            DGLContext _ctx;
            DGLDataType _id_type;
            DGLDataType _label_type;

            std::vector<cudaStream_t> CreateStreams(size_t len) {
                std::vector<cudaStream_t> res;
                for (size_t i = 0; i < len; i++) {
                    auto stream = static_cast<cudaStream_t>(
                            runtime::DeviceAPI::Get(_ctx)->CreateStream(_ctx));
                    res.push_back(stream);
                }
                return res;
            }

            DataloaderObject() {};

            ~DataloaderObject() {}

            DataloaderObject(
                    DGLContext ctx, NDArray indptr, NDArray indices, NDArray feats,
                    NDArray labels, NDArray seeds, std::vector<int64_t> fanouts,
                    int64_t batch_size, int64_t max_pool_size) {
                Init(
                        ctx, indptr, indices, feats, labels, seeds, fanouts, batch_size,
                        max_pool_size);
            };

            void Init(
                    DGLContext ctx, NDArray indptr, NDArray indices, NDArray feats,
                    NDArray labels, NDArray seeds, std::vector<int64_t> fanouts,
                    int64_t batch_size, int64_t max_pool_size) {
                _ctx = ctx;
                _indptr = indptr;
                _indices = indices;
                _id_type = indices->dtype;
                _cpu_feats = feats;
                _batch_size = batch_size;
                _max_pool_size = max_pool_size;

                int64_t _feat_width = 1;
                for (int d = 1; d < feats->ndim; ++d) {
                    _feat_width *= feats->shape[d];
                }
                DGLDataType _feat_type = feats->dtype;
                _labels = labels;
                _label_type = labels->dtype;
                _seeds = seeds;
                _num_seeds = _seeds.NumElements();
                std::reverse(fanouts.begin(), fanouts.end());
                _fanouts = fanouts;
                _table_capacity = _batch_size;
                for (auto fanout: _fanouts) {
                    _table_capacity *= (fanout + 1);
                }

//    LOG(INFO) << "Table capacity" << ToReadableSize(_table_capacity * 3);

                _blocks_pool.clear();
                std::vector<std::mutex> mutexes(_max_pool_size);
                _syncflags.resize(_max_pool_size);
                _cpu_feat_streams = CreateStreams(_max_pool_size);
                _gpu_feat_streams = CreateStreams(_max_pool_size);
                _sampling_streams = CreateStreams(_max_pool_size);
                for (int64_t i = 0; i < _max_pool_size; i++) {
                    _syncflags.at(i) = false;
                    auto blocks = std::make_shared<BlocksObject>(
                            _ctx, _fanouts, _batch_size, _feat_width, _id_type, _label_type, _feat_type,
                            _sampling_streams.at(i));
                    _blocks_pool.push_back(blocks);
                }
                _next_key = 0;
            }

            static std::shared_ptr<DataloaderObject> const Global() {
                static auto single_instance = std::make_shared<DataloaderObject>();
                return single_instance;
            }

            std::shared_ptr<BlocksObject> AwaitGetBlocks(int64_t key) {
                int blk_idx = key % _max_pool_size;
                SyncBlocks(key);
                return _blocks_pool.at(blk_idx);
            }

            // Q: shall we sample multiple instances or only the top two layers
            // more fine-grained pipelining might be needed
            int64_t AsyncSample() {
                int64_t key = _next_key++;
                AsyncSampleOnce(key);
                return key;
            }

            void SyncBlocks(int key) {
                int blk_idx = key % _max_pool_size;
                if (_syncflags.at(blk_idx) == false) {
                    runtime::DeviceAPI::Get(_ctx)->StreamSync(
                            _ctx, _sampling_streams.at(blk_idx));
                    runtime::DeviceAPI::Get(_ctx)->StreamSync(
                            _ctx, _cpu_feat_streams.at(blk_idx));
                    runtime::DeviceAPI::Get(_ctx)->StreamSync(
                            _ctx, _gpu_feat_streams.at(blk_idx));
                    // create unit-graph that can be turned into dgl blocks
                    for (int64_t layer = 0; layer < (int64_t) _fanouts.size(); layer++) {
                        auto blockPtr = _blocks_pool.at(blk_idx)->GetBlock(layer);
                        dgl_format_code_t code = COO_CODE; // COO_CODE | CSR_CODE | CSC_CODE
                        auto graph_idx = CreateFromCOO(2,
                                                       blockPtr->num_src,
                                                       blockPtr->num_dst,
                                                       blockPtr->_new_col,
                                                       blockPtr->_new_row,
                                                       false,
                                                       false,
                                                       code);
                        blockPtr->_block_ref = HeteroGraphRef{graph_idx};
                    }
                    _syncflags.at(blk_idx) = true;
                }
            }

            // TODO sample multiple instance at once
            // Q: shall we sample mutiple instances or only the top two layers
            // more fine grained pipelining might be needed
            int64_t SyncSample() {
                int64_t key = _next_key++;
                AsyncSampleOnce(key);
                SyncBlocks(key);
                return key;
            }

            // TODO: shuffle the seeds for every epoch
            NDArray GetNextSeeds(int64_t key) {
                const int64_t start_idx = key * _batch_size % _num_seeds;
                const int64_t end_idx = std::min(_num_seeds, start_idx + _batch_size);
                return _seeds.CreateView({end_idx - start_idx}, _id_type, start_idx * _id_type.bits / 8);
            }

            void AsyncSampleOnce(int64_t key) {
                int blk_idx = key % _max_pool_size;
                _syncflags.at(blk_idx) = false;
                cudaStream_t sampling_stream = _sampling_streams.at(blk_idx);

                NDArray frontier = GetNextSeeds(key);  // seeds to sample subgraph
                auto blocksPtr = _blocks_pool.at(blk_idx);
                blocksPtr->_input_nodes = frontier;
//                auto table = std::make_shared<CudaHashTable>(_id_type, _ctx, _table_capacity, sampling_stream);
                auto table = blocksPtr->_table;
                table->Reset();
                table->FillWithUnique(frontier, frontier->shape[0]);
                for (int64_t layer = 0; layer < (int64_t) _fanouts.size(); layer++) {
                    int64_t num_picks = _fanouts.at(layer);
                    std::shared_ptr<BlockObject> blockPtr = blocksPtr->_blocks.at(layer);
                    blockPtr->num_dst = frontier->shape[0];
                    ATEN_ID_TYPE_SWITCH(_id_type, IdType, {
                        CSRRowWiseSamplingUniform<kDGLCUDA, IdType>(
                                _indptr, _indices,
                                frontier, num_picks, false,
                                blockPtr,
                                sampling_stream);
                    });
                    // get the unique rows as frontier
                    table->FillWithDuplicates(blockPtr->_col, blockPtr->_col.NumElements());
                    frontier = table->RefUnique();
                    blockPtr->num_src = frontier.NumElements();
                }


                // must wait for the sampling_stream to be done before starting Mapping and Feature extraction
                // since the hash table must be populated correctly to provide the mapping and unique nodes
                // copy is necessary as table will be freed after
                blocksPtr->_output_nodes = table->CopyUnique(); // sampling_stream sync here

                // MapEdges to 0 based indexing
                for (int64_t layer = 0; layer < (int64_t) _fanouts.size(); layer++) {
                    auto blockPtr = blocksPtr->GetBlock(layer);
                    GPUMapEdges(
                            blockPtr->_row, blockPtr->_new_row, blockPtr->_col,
                            blockPtr->_new_col, table, sampling_stream);
                }

                // those two kernels are not sync until later BatchSync is called
                IndexSelect(_labels, blocksPtr->_input_nodes, blocksPtr->_labels , _gpu_feat_streams.at(blk_idx));
                IndexSelect(_cpu_feats, blocksPtr->_output_nodes, blocksPtr->_feats,_cpu_feat_streams.at(blk_idx));
                runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);
            }

            void VisitAttrs(runtime::AttrVisitor *v) final {
                v->Visit("indptr", &_indptr);
                v->Visit("indices", &_indices);
                v->Visit("labels", &_labels);
                v->Visit("feats", &_cpu_feats);
                v->Visit("seeds", &_seeds);
                v->Visit("max_pool_size", &_max_pool_size);
            }

            static constexpr const char *_type_key = "LocDataloader";

            DGL_DECLARE_OBJECT_TYPE_INFO(DataloaderObject, Object);
        };  // DataloaderObject

        class LocDataloader : public runtime::ObjectRef {
        public:
            const DataloaderObject *operator->() const {
                return static_cast<const DataloaderObject *>(obj_.get());
            }

            using ContainerType = DataloaderObject;
        };
    }  // namespace groot
}  // namespace dgl
#endif  // DGL_LOC_DATALOADER_H
