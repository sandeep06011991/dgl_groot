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
#include "array_scatter.h"
#include "block.h"
#include "cuda/cuda_index_select.cuh"
#include "cuda/cuda_mapping.cuh"
#include "cuda/gpu_cache.cuh"
#include "cuda/rowwise_sampling.cuh"
#include "cuda/rowwise_sampling_batch.cuh"
#include <thread>
#include "core.h"

namespace dgl::groot {
    class DataloaderObject : public runtime::Object {
    public:
        std::vector<std::shared_ptr<BlocksObject>> _blocks_pool;
        std::vector<int64_t> _fanouts;
        NDArray _indptr;  // graph indptr
        NDArray _indices; // graph indices
        NDArray _train_idx;
        NDArray _valid_idx;
        NDArray _test_idx;
        NDArray _cpu_feats;
        NDArray _labels;
        NDArray _partition_map; // TODO initialize partition map

        int64_t _rank, _world_size;
        int64_t _num_redundant_layers; // TODO: initialize _num_redundant_layers
        int64_t _max_pool_size;
        int64_t _batch_size;
        int64_t _next_key;
        int64_t _valid_start_key{0};
        int64_t _valid_end_key{0};

        int64_t _table_capacity;
        int64_t _feat_width;
        BlockType _block_type; // TODO initalize this
        DGLContext _ctx;
        DGLDataType _id_type;
        DGLDataType _label_type;
        DGLDataType _feat_type;
        GpuCache gpu_cache;

        DataloaderObject() = default;

        ~DataloaderObject() = default;

        DataloaderObject(int64_t rank, int64_t world_size, int64_t blockType,
                         DGLContext ctx,
                         std::vector<int64_t> fanouts, int64_t batch_size,
                         int64_t num_redundant_layer,
                         int64_t max_pool_size, NDArray indptr, NDArray indices,
                         NDArray feats, NDArray labels, NDArray train_idx,
                         NDArray valid_idx, NDArray test_idx, NDArray partition_map) {
            Init(rank, world_size, blockType, ctx, fanouts, batch_size, num_redundant_layer, max_pool_size, indptr,
                 indices, feats, labels, train_idx, valid_idx, test_idx, partition_map);
        };

        void InitBuffer() {
            // initialize buffer
            _blocks_pool.clear();
            for (int64_t i = 0; i < _max_pool_size; i++) {
                auto blocks = std::make_shared<BlocksObject>(
                        _ctx, _world_size, _num_redundant_layers, _fanouts, _batch_size, _feat_width, _id_type,
                        _label_type, _feat_type, _block_type, runtime::getCurrentCUDAStream());
                _blocks_pool.push_back(blocks);
            }
            _next_key = 0;
        }

        void Init(int64_t rank, int64_t world_size, int64_t block_type, DGLContext ctx,
                  std::vector<int64_t> fanouts, int64_t batch_size, int64_t num_redundant_layers,
                  int64_t max_pool_size, NDArray indptr, NDArray indices,
                  NDArray feats, NDArray labels, NDArray train_idx, NDArray valid_idx,
                  NDArray test_idx, NDArray partition_map) {
            // initialize meta data
            _rank = rank;
            _world_size = world_size;
            _block_type = static_cast<BlockType>(block_type);
            _ctx = ctx;
            _fanouts = fanouts;
            _batch_size = batch_size;
            _num_redundant_layers = num_redundant_layers;
            _max_pool_size = max_pool_size;
            _id_type = indices->dtype;
            _label_type = labels->dtype;
            _feat_type = feats->dtype;

            _feat_width = 1;
            for (int d = 1; d < feats->ndim; ++d) {
                _feat_width *= feats->shape[d];
            }
            _table_capacity = _batch_size;
            for (auto fanout: _fanouts) {
                _table_capacity *= (fanout + 1);
            }

            // initialize graph data
            _indptr = indptr;
            _indices = indices;
            _cpu_feats = feats;
            _labels = labels;
            _train_idx = train_idx;
            _valid_idx = valid_idx;
            _test_idx = test_idx;
            _partition_map = partition_map;

            InitBuffer();

            LOG(INFO) << "Initialized dataloader at rank " << _rank << " world_size "
                      << _world_size << " with pool size " << _max_pool_size;

            if (world_size > 1) {
                int thread_num = 1;
                bool enable_kernel_control = false;
                bool enable_comm_control = false;
                bool enable_profiler = false;
                ds::Initialize(rank, world_size, thread_num, enable_kernel_control, enable_comm_control,
                               enable_profiler);

            }
        }

        void InitFeatCache(NDArray cached_ids) {
            gpu_cache.Init(_cpu_feats, cached_ids);
        }

        static std::shared_ptr<DataloaderObject> const Global() {
            static auto single_instance = std::make_shared<DataloaderObject>();
            return single_instance;
        }

        std::shared_ptr<BlocksObject> AwaitGetBlocks(int64_t key) {
            int64_t blk_idx = key % _max_pool_size;
            return _blocks_pool.at(blk_idx);
        }

        // Q: shall we sample mutiple instances or only the top two layers
        // more fine grained pipelining might be needed
        int64_t Sample(bool replace) {
            int64_t key = _next_key++;
            GetBatch(key, replace);
            return key;
        }

        // TODO sample multiple instance at once
        // Q: shall we sample mutiple instances or only the top two layers
        // more fine grained pipelining might be needed
        std::pair<int64_t, int64_t> BatchSample(int64_t num_batches, int64_t batch_layer, bool replace) {
            int64_t start_key = _next_key;
            int64_t end_key = GetBatchesDP(start_key, num_batches, batch_layer, replace);
            _next_key = end_key + 1;
            return std::make_pair(start_key, end_key);
        }

        void ShuffleTrainingNodes(NDArray randInt) {
            auto stream = runtime::getCurrentCUDAStream();
            _train_idx = IndexSelect(_train_idx, randInt, stream);
        }

        // TODO: shuffle the seeds for every epoch
        NDArray GetNextSeeds(int64_t key) {
            auto num_elements = (_train_idx.NumElements() / _batch_size) * _batch_size;
            const int64_t start_idx = key * _batch_size % num_elements;
            const int64_t end_idx =
                    std::min(_train_idx.NumElements(), start_idx + _batch_size);
            // Having unequal initial batch sizes causes all2all to be unpredictable
            assert(end_idx - start_idx == _batch_size);
            return _train_idx.CreateView({end_idx - start_idx}, _id_type,
                                         start_idx * _id_type.bits / 8);
        }

        std::vector<NDArray> GetNextSeeds(int64_t key, int64_t num_batches) {
            auto num_steps = _train_idx.NumElements() / _batch_size;
            int64_t num_epoch = key / num_steps;
            std::vector<NDArray> res;
            int64_t num_seeds = 0;
            while (num_seeds < num_batches && (num_seeds + key) / num_steps == num_epoch) {
                res.emplace_back(GetNextSeeds(key + num_seeds));
                num_seeds++;
            }

            for (auto & seed: res) {
                CHECK(seed.NumElements()>0) << "Frontier should have a length greater than 0";
            }

            return res;
        }

        void GetBatch(int64_t key, bool replace) {
            if (_num_redundant_layers == _fanouts.size())
                GetBatchDP(key, replace);
            else
                GetBatchHybrid(key,  replace);
        }

        void GetBatchDP(int64_t key, bool replace) {
            int blk_idx = key % _max_pool_size;
            cudaStream_t sampling_stream = runtime::getCurrentCUDAStream();
            CUDAThreadEntry::ThreadLocal()->stream = sampling_stream;
            CUDAThreadEntry::ThreadLocal()->data_copy_stream = sampling_stream;
            NDArray frontier = GetNextSeeds(key); // seeds to sample subgraph
            auto blocksPtr = _blocks_pool.at(blk_idx);
            blocksPtr->_input_nodes = frontier;
            int64_t est_output_nodes = frontier->shape[0];
            for (auto fanout: _fanouts) est_output_nodes *= (fanout + 1);
            auto table = std::make_shared<CudaHashTable>(_id_type, _ctx, est_output_nodes, sampling_stream);
            table->Reset();
            table->FillWithUnique(frontier, frontier->shape[0]);
            for (int64_t layer = 0; layer < (int64_t) _fanouts.size(); layer++) {
                int64_t num_picks = _fanouts.at(layer);
                std::shared_ptr<BlockObject> blockPtr = blocksPtr->_blocks.at(layer);
                blockPtr->num_dst = frontier->shape[0];
                ATEN_ID_TYPE_SWITCH(_id_type, IdType, {
                    CSRRowWiseSamplingUniform<kDGLCUDA, IdType>(_indptr, _indices, frontier,
                                                                num_picks, replace, blockPtr,
                                                                sampling_stream);
                });
                // get the unique rows as frontier
                table->FillWithDuplicates(blockPtr->_col, blockPtr->_col.NumElements());
                frontier = table->RefUnique();
                blockPtr->num_src = frontier.NumElements();
            }
            // must wait for the sampling_stream to be done before starting Mapping
            // and Feature extraction
            blocksPtr->_output_nodes = table->CopyUnique();
            // MapEdges to 0 based indexing
            for (int64_t layer = 0; layer < (int64_t) _fanouts.size(); layer++) {
                auto blockPtr = blocksPtr->GetBlock(layer);
                GPUMapEdges(blockPtr->_row, blockPtr->_row,
                            blockPtr->_col, blockPtr->_col,
                            table, sampling_stream);
            }
            runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);
            for (int64_t layer = 0; layer < (int64_t) _fanouts.size(); layer++) {
                auto blockPtr = _blocks_pool.at(blk_idx)->GetBlock(layer);
                dgl_format_code_t code = COO_CODE; // COO_CODE | CSR_CODE | CSC_CODE
                auto graph_idx = CreateFromCOO(2, blockPtr->num_src, blockPtr->num_dst,
                                               blockPtr->_col, blockPtr->_row,
                                               false, false, code);
                blockPtr->_block_ref = HeteroGraphRef{graph_idx};
            }
            runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);
        }

        int64_t GetBatchesDP(int64_t key, int64_t num_batches, int64_t batch_layer, bool replace, bool use_hashmap=false) {
            CHECK(num_batches <= this->_max_pool_size)
                << "Number of batches should be no greater than the capacity of the pool";
            CHECK(num_batches >= 1 )
                << "Number of batches should be at least 1";

            cudaStream_t sampling_stream = runtime::getCurrentCUDAStream();
            CUDAThreadEntry::ThreadLocal()->stream = sampling_stream;
            CUDAThreadEntry::ThreadLocal()->data_copy_stream = sampling_stream;

            std::vector<NDArray> frontiers = GetNextSeeds(key, num_batches); // seeds to sample subgraph
            const int64_t blk_idx = key % _max_pool_size;

            for (int64_t layer = 0; layer < (int64_t) _fanouts.size(); layer++){
                int num_picks = _fanouts.at(layer);
                std::vector<std::shared_ptr<BlockObject>> block_vec;
                for (size_t idx = 0; idx < frontiers.size(); idx++){
                    int cur_idx = (blk_idx + idx) % _max_pool_size;
                    auto batchPtr = _blocks_pool.at(cur_idx);
                    auto blockPtr = batchPtr->_blocks.at(layer);
                    block_vec.push_back(blockPtr);
                    if (layer == 0) batchPtr->_output_nodes = frontiers.at(idx);
                }

                ATEN_ID_TYPE_SWITCH(_id_type, IdType, {
                    if (layer < batch_layer) {
                        // without using batch laoding
                        CSRRowWiseSamplingUniformBatchV0<kDGLCUDA, IdType>(_indptr, _indices, frontiers,
                                                                           num_picks, replace, block_vec,
                                                                           sampling_stream);
                    } else if (use_hashmap){
                        // Use batch loading
                        // Use hashmap to handle mapping
                        CSRRowWiseSamplingUniformBatchV1<kDGLCUDA, IdType>(_indptr, _indices, frontiers,
                                                                           num_picks, replace, block_vec,
                                                                           sampling_stream);
                    } else {
                        // Use batch loading
                        // Use indptr to handle mapping
                        CSRRowWiseSamplingUniformBatchV2<kDGLCUDA, IdType>(_indptr, _indices, frontiers,
                                                                           num_picks, replace, block_vec,
                                                                           sampling_stream);
                    }
                });

                std::vector<NDArray> new_frontiers;
                for (size_t idx = 0; idx < frontiers.size(); idx++) {
                    auto& blockPtr = block_vec.at(idx);
                    auto& frontier = frontiers.at(idx);
                    int64_t est_src_nodes = frontier.NumElements() + blockPtr->_col.NumElements() + 1;
                    auto table = std::make_shared<CudaHashTable>(_id_type, _ctx, est_src_nodes, sampling_stream);
                    table->FillWithUnique(frontier, frontier.NumElements());
                    table->FillWithDuplicates(blockPtr->_col, blockPtr->_col.NumElements());
                    new_frontiers.emplace_back(table->CopyUnique());
                    blockPtr->num_dst = frontier.NumElements();
                    blockPtr->num_src = table->NumItem();
                    GPUMapEdges(blockPtr->_row, blockPtr->_row,
                                blockPtr->_col, blockPtr->_col,
                                table, sampling_stream);
//                    dgl_format_code_t code = COO_CODE; // COO_CODE | CSR_CODE | CSC_CODE
                    auto graph_idx = CreateFromCOO(2, blockPtr->num_src, blockPtr->num_dst,
                                                   blockPtr->_col, blockPtr->_row,
                                                   false, false, COO_CODE);
                    blockPtr->_block_ref = HeteroGraphRef{graph_idx};
                }
                frontiers = new_frontiers;
            }

            for (size_t idx = 0; idx < frontiers.size(); idx++) {
                int cur_idx = (blk_idx + idx) % _max_pool_size;
                auto batchPtr = _blocks_pool.at(cur_idx);
                batchPtr->_input_nodes = frontiers.at(idx);
            }

            runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);
            _valid_start_key = key;
            _valid_end_key = key + frontiers.size() - 1;
            return _valid_end_key;
        }

        void ExtractFeatLabel(int64_t key, bool async) {
            int blk_idx = key % _max_pool_size;
            auto sampling_stream = runtime::getCurrentCUDAStream();
            auto blocksPtr = _blocks_pool.at(blk_idx);
            // those two kernels are not sync until later BatchSync is called
            IndexSelect(_labels, blocksPtr->_input_nodes, blocksPtr->_labels,
                        sampling_stream);
            blocksPtr->_feats = NDArray::Empty({blocksPtr->_output_nodes->shape[0], _feat_width}, \
                                        _feat_type, _ctx);
            if (gpu_cache.IsInitialized()) {
                gpu_cache.IndexSelectWithLocalCache(blocksPtr->_output_nodes, blocksPtr,
                                                    sampling_stream, sampling_stream);
            } else {
                IndexSelect(_cpu_feats, blocksPtr->_output_nodes, blocksPtr->_feats,
                            sampling_stream);
            }

            if (!async) {
                runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);
            }
        }

        void GetBatchHybrid(int64_t key, bool replace) {
            int blk_idx = key % _max_pool_size;
            int num_partitions = _world_size;
            auto blocksPtr = _blocks_pool.at(blk_idx);
            NDArray frontier = GetNextSeeds(key); // seeds to sample subgraph
            blocksPtr->_input_nodes = frontier;
            auto sampling_stream = runtime::getCurrentCUDAStream();
//            blocksPtr->_stream = sampling_stream;
            CHECK_LE(_num_redundant_layers, _fanouts.size() - 1);
            for (int64_t layer = 0; layer < (int64_t) _fanouts.size(); layer++) {
                int64_t num_picks = _fanouts.at(layer);
                std::shared_ptr<BlockObject> blockPtr = blocksPtr->_blocks.at(layer);
                auto blockTable = std::make_shared<CudaHashTable>(_id_type, _ctx,
                                                                  frontier.NumElements() * num_picks * num_partitions,
                                                                  sampling_stream);
                blockTable->_stream = sampling_stream;
                blockTable->Reset();
                if (layer == _num_redundant_layers) {
                    auto partition_index =
                            IndexSelect(_partition_map, frontier, sampling_stream);
                    blocksPtr->_scattered_frontier = ScatteredArray::Create(frontier->shape[0], num_partitions, _ctx,
                                                                            _id_type, sampling_stream);
                    Scatter(blocksPtr->_scattered_frontier, frontier, partition_index,
                            num_partitions, _rank, _world_size);
                    frontier = blocksPtr->_scattered_frontier->unique_array;
                }
                ATEN_ID_TYPE_SWITCH(_id_type, IdType, {
                    CSRRowWiseSamplingUniform<kDGLCUDA, IdType>(_indptr, _indices, frontier,
                                                                num_picks, replace, blockPtr,
                                                                sampling_stream);
                });
                if (layer >= _num_redundant_layers && blocksPtr->_blockType == BlockType::SRC_TO_DEST) {
                    // Todo:: Formally verify this method of insertion
                    blockTable->FillWithUnique(frontier, frontier.NumElements());
                    blockPtr->num_dst = blockTable->NumItem();
                    blockTable->FillWithDuplicates(blockPtr->_col, blockPtr->_col->shape[0]);
                    auto unique_src = blockTable->CopyUnique();
                    blockPtr->num_src = unique_src->shape[0];
                    auto partition_index =
                            IndexSelect(_partition_map, unique_src, sampling_stream);
                    blockPtr->_scattered_src = ScatteredArray::Create(blockPtr->num_src, num_partitions, _ctx, _id_type,
                                                                      sampling_stream);
                    Scatter(blockPtr->_scattered_src, unique_src, partition_index,
                            num_partitions, _rank, _world_size);
                    frontier = blockPtr->_scattered_src->unique_array;
                } else {
                    blockTable->Reset();
                    blockTable->FillWithUnique(frontier, frontier.NumElements());
                    assert(blockTable->NumItem() == frontier->shape[0]);
                    blockPtr->num_dst = blockTable->NumItem();
                    blockTable->FillWithDuplicates(blockPtr->_col,
                                                   blockPtr->_col.NumElements());
                    frontier = blockTable->CopyUnique();
                    blockPtr->num_src = frontier.NumElements();
                }
                runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);

                // MapEdges to 0 based indexing
                blockPtr->_true_node_ids = blockTable->CopyUnique();
                GPUMapEdges(blockPtr->_row, blockPtr->_row,
                            blockPtr->_col, blockPtr->_col, blockTable, sampling_stream);
                runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);
            }
            blocksPtr->_output_nodes = frontier;


            // must wait for the sampling_stream to be done before starting Mapping and
            // Feature extraction since the hash table must be populated correctly to
            // provide the mapping and unique nodes fetch feature data and label data0
            runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);
            for (int64_t layer = 0; layer < (int64_t) _fanouts.size(); layer++) {
                auto blockPtr = _blocks_pool.at(blk_idx)->GetBlock(layer);
                dgl_format_code_t code = COO_CODE; // COO_CODE | CSR_CODE | CSC_CODE
                auto graph_idx = CreateFromCOO(2, blockPtr->num_src, blockPtr->num_dst,
                                               blockPtr->_col, blockPtr->_row,
                                               false, false, code);
                blockPtr->_block_ref = HeteroGraphRef{graph_idx};
            }
        }

        void VisitAttrs(runtime::AttrVisitor *v) final {
            v->Visit("indptr", &_indptr);
            v->Visit("indices", &_indices);
            v->Visit("labels", &_labels);
            v->Visit("feats", &_cpu_feats);
            v->Visit("seeds", &_train_idx);
            v->Visit("max_pool_size", &_max_pool_size);
        }

        static constexpr const char *_type_key = "LocDataloader";

        DGL_DECLARE_OBJECT_TYPE_INFO(DataloaderObject, Object);
    }; // DataloaderObject

    class LocDataloader : public runtime::ObjectRef {
    public:
        const DataloaderObject *operator->() const {
            return static_cast<const DataloaderObject *>(obj_.get());
        }

        using ContainerType = DataloaderObject;
    };
} // namespace dgl::groot

#endif // DGL_LOC_DATALOADER_H
