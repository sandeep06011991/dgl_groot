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
#include <thread>
#include "core.h"
namespace dgl::groot {
class DataloaderObject : public runtime::Object {
public:
  std::vector<std::shared_ptr<BlocksObject>> _blocks_pool;
  std::vector<cudaStream_t> _sampling_streams;
  //  std::vector<cudaStream_t> _uva_feat_streams;
  //  std::vector<cudaStream_t> _gpu_feat_streams;
  std::vector<bool> _syncflags;
  std::vector<int64_t> _fanouts;
  std::vector<std::thread> _workers;
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
  int64_t _table_capacity;
  int64_t _feat_width;
  BlockType _block_type; // TODO initalize this
  DGLContext _ctx;
  DGLDataType _id_type;
  DGLDataType _label_type;
  DGLDataType _feat_type;
  GpuCache gpu_cache;
  std::vector<cudaStream_t> CreateStreams(size_t len) const {
    std::vector<cudaStream_t> res;
    auto stream = static_cast<cudaStream_t>(
        runtime::DeviceAPI::Get(_ctx)->CreateStream(_ctx));
    for (size_t i = 0; i < len; i++) {
      res.push_back(stream);
    }
    return res;
  }

  DataloaderObject() = default;
  ~DataloaderObject() {
    for (auto &worker : _workers) {
      if (worker.joinable())
        worker.join();
    }
  }
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
    std::vector<std::mutex> mutexes(_max_pool_size);
    _syncflags.resize(_max_pool_size);
    //    _uva_feat_streams = CreateStreams(_max_pool_size);
    //    _gpu_feat_streams = CreateStreams(_max_pool_size);
    _sampling_streams = CreateStreams(_max_pool_size);
    for (int64_t i = 0; i < _max_pool_size; i++) {
      _syncflags.at(i) = false;
      auto blocks = std::make_shared<BlocksObject>(
          _ctx, _world_size, _num_redundant_layers, _fanouts, _batch_size, _feat_width, _id_type,
          _label_type, _feat_type, _block_type, _sampling_streams.at(i));
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
    _workers.resize(_max_pool_size);
    _id_type = indices->dtype;
    _label_type = labels->dtype;
    _feat_type = feats->dtype;

    _feat_width = 1;
    for (int d = 1; d < feats->ndim; ++d) {
      _feat_width *= feats->shape[d];
    }
    _table_capacity = _batch_size;
    for (auto fanout : _fanouts) {
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
              << _world_size;

    if(world_size != 0) {
      int thread_num = 1;
      bool enable_kernel_control = false;
      bool enable_comm_control = false;
      bool enable_profiler = false;
      ds::Initialize(rank, world_size, thread_num, enable_kernel_control, enable_comm_control, enable_profiler);
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

  void SyncBlocks(int64_t key) {
    int blk_idx = key % _max_pool_size;
    if (_syncflags.at(blk_idx) == false) {
      auto stream = _sampling_streams.at(
          blk_idx); // dataloading is using the sampling stream as well
      runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, stream);
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
    const int64_t start_idx = key * _batch_size % _train_idx.NumElements();
    const int64_t end_idx =
        std::min(_train_idx.NumElements(), start_idx + _batch_size);
    return _train_idx.CreateView({end_idx - start_idx}, _id_type,
                                 start_idx * _id_type.bits / 8);
  }

  void AsyncSampleOnce(int64_t  key) {
    if (_num_redundant_layers == _fanouts.size())
      AsyncSampleOnceDP(key);
    else
      AsyncSampleOnceHybrid(key);
  }

  void AsyncSampleOnceDP(int64_t key) {
    int blk_idx = key % _max_pool_size;
    _syncflags.at(blk_idx) = false;
    cudaStream_t sampling_stream = _sampling_streams.at(blk_idx);
    CUDAThreadEntry::ThreadLocal()->stream = sampling_stream;
    CUDAThreadEntry::ThreadLocal()->data_copy_stream = sampling_stream;

    NDArray frontier = GetNextSeeds(key); // seeds to sample subgraph
    auto blocksPtr = _blocks_pool.at(blk_idx);
    blocksPtr->_input_nodes = frontier;
    auto table = blocksPtr->_table;
    table->Reset();
    table->FillWithUnique(frontier, frontier->shape[0]);
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      int64_t num_picks = _fanouts.at(layer);
      std::shared_ptr<BlockObject> blockPtr = blocksPtr->_blocks.at(layer);
      blockPtr->num_dst = frontier->shape[0];
      ATEN_ID_TYPE_SWITCH(_id_type, IdType, {
        CSRRowWiseSamplingUniform<kDGLCUDA, IdType>(_indptr, _indices, frontier,
                                                    num_picks, false, blockPtr,
                                                    sampling_stream);
      });
      // get the unique rows as frontier
      table->FillWithDuplicates(blockPtr->_col, blockPtr->_col.NumElements());
      frontier = table->RefUnique();
      blockPtr->num_src = frontier.NumElements();
    }
    // must wait for the sampling_stream to be done before starting Mapping
    // and Feature extraction
    blocksPtr->_output_nodes = table->RefUnique();
    // MapEdges to 0 based indexing
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      auto blockPtr = blocksPtr->GetBlock(layer);
      GPUMapEdges(blockPtr->_row, blockPtr->_new_row, blockPtr->_col,
                  blockPtr->_new_col, table, sampling_stream);
    }

    runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      auto blockPtr = _blocks_pool.at(blk_idx)->GetBlock(layer);
      dgl_format_code_t code = COO_CODE; // COO_CODE | CSR_CODE | CSC_CODE
      auto graph_idx = CreateFromCOO(2, blockPtr->num_src, blockPtr->num_dst,
                                     blockPtr->_new_col, blockPtr->_new_row,
                                     false, false, code);
      blockPtr->_block_ref = HeteroGraphRef{graph_idx};
    }

    // those two kernels are not sync until later BatchSync is called
    IndexSelect(_labels, blocksPtr->_input_nodes, blocksPtr->_labels,
                sampling_stream);

    if (gpu_cache.IsInitialized()) {
      gpu_cache.IndexSelectWithLocalCache(blocksPtr->_output_nodes, blocksPtr,
                                          sampling_stream, sampling_stream);
    } else {
      IndexSelect(_cpu_feats, blocksPtr->_output_nodes, blocksPtr->_feats,
                  sampling_stream);
    }
  }

  void AsyncSampleOnceHybrid(int64_t key) {
    int blk_idx = key % _max_pool_size;
    int num_partitions = _world_size;
    auto blocksPtr = _blocks_pool.at(blk_idx);
    NDArray frontier = GetNextSeeds(key); // seeds to sample subgraph
    blocksPtr->_input_nodes = frontier;
    cudaStream_t sampling_stream = _sampling_streams.at(blk_idx);
    CHECK_LE(_num_redundant_layers, _fanouts.size() - 1);
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      int64_t num_picks = _fanouts.at(layer);
      std::shared_ptr<BlockObject> blockPtr = blocksPtr->_blocks.at(layer);
      auto blockTable = blockPtr->_table;
      blockTable->Reset();
      if (layer == _num_redundant_layers) {
        auto partition_index =
            IndexSelect(_partition_map, frontier, sampling_stream);
        Scatter(blocksPtr->_scattered_frontier, frontier, partition_index,
                num_partitions, _rank, _world_size);
        frontier = blocksPtr->_scattered_frontier->unique_array;
      }
      ATEN_ID_TYPE_SWITCH(_id_type, IdType, {
        CSRRowWiseSamplingUniform<kDGLCUDA, IdType>(_indptr, _indices, frontier,
                                                    num_picks, false, blockPtr,
                                                    sampling_stream);
      });
//      cudaDeviceSynchronize();
      if (layer >= _num_redundant_layers) {
        if (blocksPtr->_blockType == BlockType::SRC_TO_DEST) {
          // Todo:: Formally verify this method of insertion
          blockTable->Reset();
          blockTable->FillWithUnique(frontier, frontier.NumElements());
//          blockTable->FillWithDuplicates(blockPtr->_row,blockPtr->_row->shape[0]);
          blockPtr->num_dst = blockTable->RefUnique().NumElements();
          blockTable->FillWithDuplicates(blockPtr->_col,blockPtr->_col->shape[0]);
          auto unique_src = blockTable->RefUnique();
          auto partition_index =
              IndexSelect(_partition_map, unique_src, sampling_stream);
          Scatter(blockPtr->_scattered_src, unique_src, partition_index,
                  num_partitions, _rank, _world_size);
          frontier = blockPtr->_scattered_src->unique_array;
        } else {
          LOG(FATAL) << "Dont need to reindex here ";
        }
      }else{
        blockTable->Reset();
        blockTable->FillWithUnique(frontier, frontier.NumElements());
        assert(blockTable->RefUnique().NumElements() == frontier->shape[0]);
        blockPtr->num_dst = blockTable->RefUnique().NumElements();
        blockTable->FillWithDuplicates(blockPtr->_col,
                                       blockPtr->_col.NumElements());
        frontier = blockTable->RefUnique();
      }
//        if (blocksPtr->_blockType == BlockType::DEST_TO_SRC) {
//          LOG(FATAL) << ("Dont need to reindex here ");
//          NDArray p_map =
//              IndexSelect(blockPtr->_col, _partition_map, sampling_stream);
//          Scatter(blockPtr->_scattered_dest, blockPtr->_col, p_map,
//                  num_partitions, _rank, _world_size);
//          Scatter(blockPtr->_scattered_dest, blockPtr->_row, p_map,
//                  num_partitions, _rank, _world_size);
//          blockPtr->_col = blockPtr->_scattered_dest->shuffled_array;
//          blockPtr->_row = blockPtr->_scattered_dest->shuffled_array;
//          blockTable->FillWithDuplicates(blockPtr->_row,blockPtr->_row.NumElements());
//        }
//      } else {
//
//      }
      blockPtr->num_src = frontier.NumElements();
    }
    blocksPtr->_output_nodes = frontier;
    // MapEdges to 0 based indexing
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      auto blockPtr = blocksPtr->GetBlock(layer);
      GPUMapEdges(blockPtr->_row, blockPtr->_new_row, blockPtr->_col,
                  blockPtr->_new_col, blockPtr->_table, sampling_stream);
    }

    // must wait for the sampling_stream to be done before starting Mapping and
    // Feature extraction since the hash table must be populated correctly to
    // provide the mapping and unique nodes fetch feature data and label data0
    runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      auto blockPtr = _blocks_pool.at(blk_idx)->GetBlock(layer);
      dgl_format_code_t code = COO_CODE; // COO_CODE | CSR_CODE | CSC_CODE
      auto graph_idx = CreateFromCOO(2, blockPtr->num_src, blockPtr->num_dst,
                                     blockPtr->_new_col, blockPtr->_new_row,
                                     false, false, code);
      blockPtr->_block_ref = HeteroGraphRef{graph_idx};
    }

    // those two kernels are not sync until later BatchSync is called
    IndexSelect(_labels, blocksPtr->_input_nodes, blocksPtr->_labels,
                sampling_stream);

    if (gpu_cache.IsInitialized()) {
      gpu_cache.IndexSelectWithLocalCache(blocksPtr->_output_nodes, blocksPtr,
                                          sampling_stream, sampling_stream);
    } else {
      IndexSelect(_cpu_feats, blocksPtr->_output_nodes, blocksPtr->_feats,
                  sampling_stream);
    }
    //      blocksPtr->_output_nodes =frontier;
    //      blocksPtr->_labels = IndexSelect(_labels, blocksPtr->_input_nodes,
    //      _gpu_feat_streams.at(blk_idx)); blocksPtr->_feats =
    //      IndexSelect(_cpu_feats, blocksPtr->_output_nodes,
    //      _cpu_feat_streams.at(blk_idx));
    //
    //      // MapEdges to 0 based indexing
    //      for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
    //        auto blockPtr = blocksPtr->GetBlock(layer);
    //        GPUMapEdges(
    //            blockPtr->_row, blockPtr->_new_row, blockPtr->_col,
    //            blockPtr->_new_col, blockPtr->_table,
    //            _reindex_streams.at(blk_idx));
    //      }
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
