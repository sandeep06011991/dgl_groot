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
#include "../groot/array_scatter.h"

namespace dgl {
namespace groot {
class DataloaderObject : public runtime::Object {



 public:
  std::vector<std::shared_ptr<BlocksObject>> _blocks_pool;
  std::vector<cudaStream_t> _reindex_streams;
  std::vector<cudaStream_t> _cpu_feat_streams;
  std::vector<cudaStream_t> _gpu_feat_streams;
  std::vector<bool> _blocks_syncflags;
  std::vector<int64_t> _fanouts;

  NDArray _indptr;
  NDArray _indices;
  NDArray _seeds;
  NDArray _cpu_feats;
  NDArray _gpu_feats;
  NDArray _labels;

//Data structures required for slicing
  BlocksObject::BlockType _no_redundancy_type;
  int _num_redundant_layers;
  NDArray _partition_map;

  int64_t _max_pool_size;
  int64_t _batch_size;
  int64_t _next_key;
  int64_t _num_seeds;

  DGLContext _ctx;
  DGLDataType _id_type;
  DGLDataType _label_type;

  std::vector<cudaStream_t > CreateStreams(size_t len) {
    std::vector<cudaStream_t> res;
    for (size_t i = 0; i < len; i++) {
      auto stream = static_cast<cudaStream_t>(
          runtime::DeviceAPI::Get(_ctx)->CreateStream(_ctx));
      res.push_back(stream);
    }
    return res;
  }
  DataloaderObject() {
    LOG(INFO) << "Calling DataloaderObject default constructor";
  };

  ~DataloaderObject() {
    LOG(INFO) << "Calling DataloaderObject default deconstructor";
  }


  void Init(
      DGLContext ctx, NDArray indptr, NDArray indices, NDArray feats,
      NDArray labels, NDArray seeds, NDArray pmap, std::vector<int64_t> fanouts,
      int64_t batch_size, int64_t max_pool_size, BlocksObject::BlockType blockType,\
      int numRedundantLayers) {
    LOG(INFO) << "Creating DataloaderObject with init function";

    _ctx = ctx;
    _indptr = indptr;
    _indices = indices;
    _id_type = indices->dtype;
    _cpu_feats = feats;
    _labels = labels;
    _label_type = labels->dtype;
    _seeds = seeds;
    _no_redundancy_type = blockType;
    _num_redundant_layers = numRedundantLayers;
    _partition_map = pmap;
    CHECK_EQ(pmap->shape[0] , (indptr->shape[0] - 1));
    _num_seeds = _seeds.NumElements();
    _fanouts = fanouts;
    _batch_size = batch_size;
    _max_pool_size = max_pool_size;
    CHECK_NE(_max_pool_size,0);
    _blocks_pool.clear();
    std::vector<std::mutex> mutexes(_max_pool_size);
    _blocks_syncflags.resize(_max_pool_size);
    _cpu_feat_streams = CreateStreams(_max_pool_size);
    _gpu_feat_streams = CreateStreams(_max_pool_size);
    _reindex_streams = CreateStreams(_max_pool_size);
    auto sampling_stream = runtime::getCurrentCUDAStream();
    for (int64_t i = 0; i < _max_pool_size; i++) {
      _blocks_syncflags.at(i) = false;
      auto blocks = std::make_shared<BlocksObject>(
          _ctx, _fanouts, _batch_size, _id_type, _label_type, sampling_stream, BlocksObject::SRC_TO_DEST);
      _blocks_pool.push_back(blocks);
    }
    _next_key = 0;
  }

  DataloaderObject(
      DGLContext ctx, NDArray indptr, NDArray indices, NDArray feats,
      NDArray labels, NDArray seeds, NDArray pmap, std::vector<int64_t> fanouts,
      int64_t batch_size, int64_t max_pool_size, BlocksObject::BlockType blockType,\
      int numRedundantLayers) {
    Init(
        ctx, indptr, indices, feats, labels, seeds, pmap,  fanouts, batch_size,
        max_pool_size, blockType, numRedundantLayers);
  };

  static std::shared_ptr<DataloaderObject> single_instance;

  static std::shared_ptr<DataloaderObject> const Global() {
//    if(single_instance == nullptr) {
//      LOG(INFO) << "Creating DataLoader object suspecting duplication \n";
//      single_instance = std::make_shared<DataloaderObject>();
//    }
    return DataloaderObject::single_instance;
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
    if (!_blocks_syncflags.at(blk_idx)) {
      runtime::DeviceAPI::Get(_ctx)->StreamSync(
          _ctx, _reindex_streams.at(blk_idx));
      runtime::DeviceAPI::Get(_ctx)->StreamSync(
          _ctx, _cpu_feat_streams.at(blk_idx));
      runtime::DeviceAPI::Get(_ctx)->StreamSync(
          _ctx, _gpu_feat_streams.at(blk_idx));
      // create unit-graph that can be turned into dgl blocks
      for (int64_t layer = 0; layer < (int64_t )_fanouts.size(); layer++){
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
      _blocks_syncflags.at(blk_idx) = true;
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
    _blocks_syncflags.at(blk_idx) = false;
    auto blocksPtr = _blocks_pool.at(blk_idx);
    NDArray frontier = GetNextSeeds(key);  // seeds to sample subgraph
    blocksPtr->_input_nodes = frontier;
    cudaStream_t sampling_stream = runtime::getCurrentCUDAStream();

    int num_partitions = 4;
    int world_size = num_partitions;
    int rank = _ctx.device_id;
    CHECK_LE(_num_redundant_layers, _fanouts.size() + 1 );
    if(_fanouts.size() == _num_redundant_layers){
      CHECK_EQ(_no_redundancy_type ,BlocksObject::BlockType::DATA_PARALLEL);
    }
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      int64_t num_picks = _fanouts.at(layer);
      std::shared_ptr<BlockObject> blockPtr = blocksPtr->_blocks.at(layer);
      auto blockTable  = blockPtr->_table;
      if(layer == _num_redundant_layers){
        auto partition_index = IndexSelect(_partition_map, frontier,  sampling_stream);
        Scatter(blocksPtr->_scattered_frontier, frontier,  partition_index, num_partitions, rank, world_size );
        frontier = blocksPtr->_scattered_frontier->unique_array;
      }
      ATEN_ID_TYPE_SWITCH(_id_type, IdType, {
        CSRRowWiseSamplingUniform<kDGLCUDA, IdType >(
            _indptr, _indices, frontier, num_picks, false, blockPtr,
            sampling_stream);
      });
      cudaDeviceSynchronize();
      if(layer >= _num_redundant_layers){
        if(blocksPtr->_blockType == BlocksObject::BlockType::SRC_TO_DEST){
            // Todo:: Formally verify this method of insertion
            blockTable->Reset();
            blockTable->FillWithDuplicates(blockPtr->_row, blockPtr->_row->shape[0]);
            blockPtr->num_dst = frontier.NumElements();
            blockTable->FillWithDuplicates(blockPtr->_col, blockPtr->_col->shape[0]);
            auto unique_src = blockTable->GetUnique();
            auto partition_index = IndexSelect(_partition_map, unique_src, sampling_stream);
            Scatter(blockPtr->_scattered_src,  unique_src,  partition_index, num_partitions, rank, world_size );
            frontier = blockPtr->_scattered_src->unique_array;
         }
         if(blocksPtr->_blockType == BlocksObject::BlockType::DEST_TO_SRC){
                  LOG(FATAL) << ( "Dont need to reindex here ");
                    NDArray p_map = IndexSelect(blockPtr->_col, _partition_map, sampling_stream);
                    Scatter(blockPtr->_scattered_dest, blockPtr->_col, p_map, num_partitions, rank, world_size );
                    Scatter(blockPtr->_scattered_dest, blockPtr->_row, p_map, num_partitions, rank, world_size );
                    blockPtr->_col = blockPtr->_scattered_dest->shuffled_array;
                    blockPtr->_row = blockPtr->_scattered_dest->shuffled_array;
                    blockTable->FillWithDuplicates(blockPtr->_row, blockPtr->_row.NumElements());
         }
      }else{
         blockTable->Reset();
         blockTable->FillWithDuplicates(blockPtr->_row, blockPtr->_row.NumElements());
         blockPtr->num_dst = blockTable->GetUnique().NumElements();
         blockTable->FillWithDuplicates(blockPtr->_col, blockPtr->_col.NumElements());
         frontier = blockTable->GetUnique();
      }
       blockPtr->num_src = frontier.NumElements();
    }

    // must wait for the sampling_stream to be done before starting Mapping and Feature extraction
    // since the hash table must be populated correctly to provide the mapping and unique nodes
    runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, sampling_stream);
    // fetch feature data and label data0
    blocksPtr->_output_nodes =frontier;
    blocksPtr->_labels = IndexSelect(_labels, blocksPtr->_input_nodes, _gpu_feat_streams.at(blk_idx));
    blocksPtr->_feats = IndexSelect(_cpu_feats, blocksPtr->_output_nodes, _cpu_feat_streams.at(blk_idx));

    // MapEdges to 0 based indexing
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      auto blockPtr = blocksPtr->GetBlock(layer);
      GPUMapEdges(
          blockPtr->_row, blockPtr->_new_row, blockPtr->_col,
          blockPtr->_new_col, blockPtr->_table, _reindex_streams.at(blk_idx));
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
  DGL_DECLARE_OBJECT_TYPE_INFO(DataloaderObject, Object);
};

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
