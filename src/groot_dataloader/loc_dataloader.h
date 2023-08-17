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
class LocDataloaderObject : public runtime::Object {
 public:
  std::vector<std::shared_ptr<BlocksObject>> _blocks_pool;
  std::vector<cudaStream_t> _sampling_streams;
  std::vector<cudaStream_t> _reindexing_streams;
  std::vector<cudaStream_t> _cpu_feat_streams;
  std::vector<cudaStream_t> _gpu_feat_streams;
  std::vector<std::mutex> _blocks_mutex;
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
  int64_t _feat_width;
  int64_t _num_seeds;

  DGLContext _ctx;
  DGLDataType _id_type;
  DGLDataType _feat_type;
  DGLDataType _label_type;

  bool _stop;
  std::mutex _mutex;

  LocDataloaderObject() {
    LOG(INFO) << "Calling LocDataloaderObject default constructor";
  };

  ~LocDataloaderObject() {
    LOG(INFO) << "Calling LocDataloaderObject default deconstructor";
  }

  LocDataloaderObject(
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
    LOG(INFO) << "Creating LocDataloaderObject with init function";

    _ctx = ctx;
    _indptr = indptr;
    _indices = indices;
    _id_type = indices->dtype;
    _cpu_feats = feats;
    _feat_type = feats->dtype;
    _feat_width = feats->shape[1];
    _labels = labels;
    _label_type = labels->dtype;
    _seeds = seeds;
    _num_seeds = _seeds.NumElements();
    _fanouts = fanouts;
    _batch_size = batch_size;
    _max_pool_size = max_pool_size;
    _blocks_pool.clear();
    std::vector<std::mutex> mutexes(_max_pool_size);
    _blocks_mutex.swap(mutexes);

    _cpu_feat_streams.clear();
    _gpu_feat_streams.clear();
    _reindexing_streams.clear();
    _sampling_streams.clear();
    for (int64_t i = 0; i < _max_pool_size; i++) {
      auto sampling_stream = static_cast<cudaStream_t>(
          runtime::DeviceAPI::Get(_ctx)->CreateStream(_ctx));
      auto reindex_stream = static_cast<cudaStream_t>(
          runtime::DeviceAPI::Get(_ctx)->CreateStream(_ctx));
      auto cpu_feat_stream = static_cast<cudaStream_t>(
          runtime::DeviceAPI::Get(_ctx)->CreateStream(_ctx));
      auto gpu_feat_stream = static_cast<cudaStream_t>(
          runtime::DeviceAPI::Get(_ctx)->CreateStream(_ctx));
      _sampling_streams.push_back(sampling_stream);
      _reindexing_streams.push_back(reindex_stream);
      _cpu_feat_streams.push_back(cpu_feat_stream);
      _gpu_feat_streams.push_back(gpu_feat_stream);

      auto blocks = std::make_shared<BlocksObject>(
          _ctx, _fanouts, _batch_size, _id_type, _label_type, sampling_stream);

      _blocks_pool.push_back(blocks);
    }
    _next_key = 0;
  }

  static std::shared_ptr<LocDataloaderObject> const Global() {
    static auto single_instance = std::make_shared<LocDataloaderObject>();
    return single_instance;
  }

  std::shared_ptr<BlocksObject> GetBlocks(int64_t key) {
    return _blocks_pool.at(key % _max_pool_size);
  }

  // TODO sample multiple instance at once
  // Q: shall we sample mutiple instances or only the top two layers
  // more fine grained pipelining might be needed
  int64_t Sample() {
    int64_t blk_idx, ret_idx;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      ret_idx = _next_key;
      blk_idx = _next_key++;
    }
    blk_idx = blk_idx % _max_pool_size;
    SampleOneBatch(blk_idx);
    runtime::DeviceAPI::Get(_ctx)->StreamSync(
        _ctx, _sampling_streams.at(blk_idx));
    runtime::DeviceAPI::Get(_ctx)->StreamSync(
        _ctx, _cpu_feat_streams.at(blk_idx));
    runtime::DeviceAPI::Get(_ctx)->StreamSync(
        _ctx, _gpu_feat_streams.at(blk_idx));
    runtime::DeviceAPI::Get(_ctx)->StreamSync(
        _ctx, _reindexing_streams.at(blk_idx));
    return ret_idx;
  }

  // TODO: shuffle the seeds for every epoch
  NDArray GetNextSeeds() {
    std::lock_guard<std::mutex> lock(_mutex);
    const int64_t start_idx = _next_key * _batch_size % _num_seeds;
    const int64_t end_idx = std::min(_num_seeds, start_idx + _batch_size);
    return _seeds.CreateView(
        {end_idx - start_idx}, _id_type, start_idx * _id_type.bits / 8);
  }

  void SampleOneBatch(int64_t blk_idx) {
    CHECK(blk_idx < (int64_t)_blocks_pool.size());
    auto &blocksPtr = _blocks_pool.at(blk_idx);
    NDArray frontier = GetNextSeeds();  // seeds to sample subgraph

    auto table = blocksPtr->_table;
    table->Reset();
    table->FillWithUnique(frontier, frontier->shape[0]);

    cudaStream_t sampling_stream = _sampling_streams.at(blk_idx);
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      int64_t fanout = _fanouts.at(layer);
      std::shared_ptr<BlockObject> blockPtr = blocksPtr->_blocks.at(layer);
      if (_id_type.bits == 32) {
        CSRRowWiseSamplingUniform<kDGLCUDA, int32_t>(
            _indptr, _indices, frontier, fanout, false, blockPtr,
            sampling_stream);
      } else if (_id_type.bits == 64) {
        CSRRowWiseSamplingUniform<kDGLCUDA, int64_t>(
            _indptr, _indices, frontier, fanout, false, blockPtr,
            sampling_stream);
      }
      // get the unique rows as frontier
      // the _col has correct number of elements because it is synchronized through an event
      // but the ids in the _col might still be incorrect
      table->FillWithDuplicates(blockPtr->_col, blockPtr->_col->shape[0]);
      blockPtr->num_src = frontier->shape[0];
      blockPtr->num_dst = blockPtr->_col->shape[0];
      frontier = table->GetUnique();
    }

    // must wait for the sampling_stream to be done before starting Mapping and Feature extraction
    // since the hash table must be populated correctly to provide the mapping and unique nodes
    runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);

    // MapEdges to 0 based indexing
    auto reindex_stream = _reindexing_streams.at(blk_idx);
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      auto blockPtr = blocksPtr->GetBlock(layer);
      GPUMapEdges(
          blockPtr->_row, blockPtr->_new_row, blockPtr->_col,
          blockPtr->_new_col, blocksPtr->_table, reindex_stream);
    }

    // fetch feature data and label data0
    blocksPtr->_output_nodes = table->GetUnique();
    blocksPtr->_labels = IndexSelect(_labels, frontier, _gpu_feat_streams.at(blk_idx));
    blocksPtr->_feats = IndexSelect(_cpu_feats, blocksPtr->_output_nodes, _cpu_feat_streams.at(blk_idx));

    // create unit-graph that can be turned into dgl blocks
    runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, reindex_stream);
    for (int64_t layer = 0; layer < (int64_t )_fanouts.size(); layer++){
      auto blockPtr = blocksPtr->GetBlock(layer);
      dgl_format_code_t code = COO_CODE; // COO_CODE | CSR_CODE | CSC_CODE
      auto graph_idx = CreateFromCOO(2, blockPtr->num_src, blockPtr->num_dst, blockPtr->_new_row, blockPtr->_new_col, false, false, code);
      blockPtr->_block_ref = HeteroGraphRef{graph_idx};
    }
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
  DGL_DECLARE_OBJECT_TYPE_INFO(LocDataloaderObject, Object);
};  // LocDataloaderObject

class LocDataloader : public runtime::ObjectRef {
 public:
  const LocDataloaderObject *operator->() const {
    return static_cast<const LocDataloaderObject *>(obj_.get());
  }
  using ContainerType = LocDataloaderObject;
};
}  // namespace groot
}  // namespace dgl
#endif  // DGL_LOC_DATALOADER_H
