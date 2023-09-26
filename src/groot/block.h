//
// Created by juelin on 8/6/23.
//

#ifndef DGL_BLOCK_H
#define DGL_BLOCK_H

#include "../c_api_common.h"
#include "../graph/heterograph.h"
#include "../graph/unit_graph.h"
#include "./cuda/cuda_hashtable.cuh"
#include "array_scatter.h"

#include <dgl/array.h>
#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/device_api.h>
#include <memory>
#include <stdint.h>

namespace dgl {
namespace groot {

enum BlockType{
  DATA_PARALLEL,
  SRC_TO_DEST,
  DEST_TO_SRC
};

struct BlockObject : public runtime::Object {
  BlockObject(){};

  BlockObject(DGLContext ctx, int64_t num_partitions,
              const std::vector<int64_t> &fanouts, int64_t layer,
              int64_t batch_size, DGLDataType dtype, cudaStream_t stream) {
    int64_t est_num_src = batch_size;
    int64_t est_num_dst = batch_size;
    for (int64_t i = 0; i < layer; i++)
      est_num_src *= (fanouts[i] + 1);
    for (int64_t i = 0; i <= layer; i++)
      est_num_dst *= (fanouts[i] + 1);
    _row = NDArray::Empty({est_num_dst}, dtype, ctx);
    _col = NDArray::Empty({est_num_dst}, dtype, ctx);
    _new_row = NDArray::Empty({est_num_dst}, dtype, ctx);
    _new_col = NDArray::Empty({est_num_dst}, dtype, ctx);
    _new_len_tensor = NDArray::PinnedEmpty({1}, dtype, DGLContext{kDGLCPU, 0});
    _data_or_idx = NDArray::Empty({est_num_dst}, dtype, ctx);
    _indptr = NDArray::Empty({est_num_src + 1}, dtype, ctx);
    _outdeg = NDArray::Empty({est_num_src}, dtype, ctx);

    _table = std::make_shared<CudaHashTable>(dtype, ctx, est_num_dst, stream);
    // Todo SRC to DEST data strucutures are not needed for redudnant blocks
    _scattered_dest =
        ScatteredArray::Create(est_num_dst, num_partitions, ctx, dtype, stream);
    _scattered_src =
        ScatteredArray::Create(est_num_src, num_partitions, ctx, dtype, stream);
  };
  int64_t num_src, num_dst; // number of src (unique) and destination (unique)
                            // for buliding the dgl block object
  NDArray _row;             // input nodes in original ids (coo format)
  NDArray _new_row;         // input nodes in reindex-ed ids (0 based indexing)
  NDArray _col;             // destination nodes in original ids (coo format)
  NDArray _new_col;         // destination nodes inreindex-ed ids (coo format)
  NDArray _data_or_idx;     // storing index / data (optional)
  NDArray _outdeg;
  NDArray _indptr;
  NDArray _new_len_tensor;
  HeteroGraphRef _block_ref;
  std::shared_ptr<CudaHashTable> _table;
  // Depending on wheter they are scattered src or dest
  ScatteredArray _scattered_src;
  ScatteredArray _scattered_dest;

  void VisitAttrs(runtime::AttrVisitor *v) final {
    v->Visit("row", &_row);
    v->Visit("col", &_col);
    v->Visit("idx", &_data_or_idx);
    v->Visit("new_col", &_new_col);
    v->Visit("new_row", &_new_row);
    v->Visit("indptr", &_indptr);
    v->Visit("outdeg", &_outdeg);
    v->Visit("gidx", &_block_ref);
  }

  static constexpr const char *_type_key = "Block";

  DGL_DECLARE_OBJECT_TYPE_INFO(BlockObject, Object);
}; // block

class Block : public runtime::ObjectRef {
public:
  const BlockObject *operator->() const {
    return static_cast<const BlockObject *>(obj_.get());
  }

  using ContainerType = BlockObject;
};

struct GpuCacheQueryBuffer {
  NDArray _hit_id; // hit id : not very useful but return anyway for debugging
  NDArray _hit_cidx; // hit id's index in the cache : used for loading from gpu_feat
  NDArray _hit_qidx; // hit id's index in the query : used for writing cpu_feat
                     // to output buffer
  NDArray _missed_qid; // missed id's in the query : used for loading from cpu_feat
  NDArray _missed_qidx; // missed id's index in the query : used for writing
                        // gpu_feat ot the output buffer
  GpuCacheQueryBuffer() = default;

  GpuCacheQueryBuffer(int64_t est_num_nodes, DGLDataType dtype,
                      DGLContext ctx) {
    Init(est_num_nodes, dtype, ctx);
  }

  void Init(int64_t est_num_nodes, DGLDataType dtype, DGLContext ctx) {
    _hit_id = NDArray::Empty({est_num_nodes}, dtype, ctx);
    _missed_qid = NDArray::Empty({est_num_nodes}, dtype, ctx);
    _hit_cidx =
        NDArray::Empty({est_num_nodes}, DGLDataType{kDGLInt, 64, 1}, ctx);
    _hit_qidx =
        NDArray::Empty({est_num_nodes}, DGLDataType{kDGLInt, 64, 1}, ctx);
    _missed_qidx =
        NDArray::Empty({est_num_nodes}, DGLDataType{kDGLInt, 64, 1}, ctx);
  }
};

struct BlocksObject : public runtime::Object {

  ~BlocksObject() { LOG(INFO) << "Calling Blocks Object destructor"; };
  int64_t key, _num_layer, _feat_width, _num_redundant_layers;
  std::vector<std::shared_ptr<BlockObject>> _blocks;
  NDArray _labels;                   // labels of input nodes
  NDArray _feats;                    // feature of output nodes
  NDArray _input_nodes;              // seeds
  NDArray _output_nodes;             // output nodes
  GpuCacheQueryBuffer _query_buffer; // store indices for query gpu cache
  std::shared_ptr<CudaHashTable> _table;
  cudaStream_t _stream;
  DGLContext _ctx;
  BlockType _blockType;
  ScatteredArray _scattered_frontier;

  BlocksObject(){};

  BlocksObject(DGLContext ctx, int64_t num_partitions, int num_redundant_layers,
               std::vector<int64_t> fanouts, int64_t batch_size,
               int64_t feat_width, DGLDataType id_type, DGLDataType label_type,
               DGLDataType feat_type, BlockType block_type, cudaStream_t stream) {
    _ctx = ctx;
    _feat_width = feat_width;
    _num_layer = fanouts.size();
    _stream = stream;
    _blockType = block_type;
    for (int64_t layer = 0; layer < _num_layer; layer++) {
      auto block = std::make_shared<BlockObject>(
          _ctx, num_partitions, fanouts, layer, batch_size, id_type, stream);
      _blocks.push_back(block);
    }

    int exp_frontier_size = batch_size;
    for (int i = 0; i < num_redundant_layers; i++) exp_frontier_size *= fanouts[i] + 1;
    _scattered_frontier = ScatteredArray::Create(exp_frontier_size, num_partitions, ctx, id_type, stream);

    int64_t est_output_nodes = batch_size;
    for (int64_t fanout : fanouts)
      est_output_nodes *= (fanout + 1);
    _table = std::make_shared<CudaHashTable>(id_type, _ctx, est_output_nodes,
                                             stream);
    _feats = NDArray::Empty({est_output_nodes, _feat_width}, feat_type, _ctx);
    _labels = NDArray::Empty({batch_size}, label_type, _ctx);
    _query_buffer.Init(est_output_nodes, id_type, _ctx);
  };

  int64_t GetSize() {
    int64_t num_bytes = 0;
    num_bytes += _feats.GetSize();
    return num_bytes;
  }

  std::shared_ptr<BlockObject> GetBlock(int64_t layer) {
    return _blocks.at(layer);
  }

  void VisitAttrs(runtime::AttrVisitor *v) final {
    v->Visit("label", &_labels);
    v->Visit("input_node", &_input_nodes);
    v->Visit("output_node", &_output_nodes);
    v->Visit("feat", &_feats);
  }

  static constexpr const char *_type_key = "Blocks";

  DGL_DECLARE_OBJECT_TYPE_INFO(BlocksObject, Object);
};

class Blocks : public runtime::ObjectRef {
public:
  const BlocksObject *operator->() const {
    return static_cast<const BlocksObject *>(obj_.get());
  }

  using ContainerType = BlocksObject;
};

} // namespace groot
} // namespace dgl

#endif // DGL_BLOCK_H
