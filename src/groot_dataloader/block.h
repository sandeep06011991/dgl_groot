//
// Created by juelin on 8/6/23.
//

#ifndef DGL_BLOCK_H
#define DGL_BLOCK_H
#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include <dgl/packed_func_ext.h>
#include <stdint.h>
#include <memory>
#include <dgl/immutable_graph.h>
#include "../graph/unit_graph.h"
#include "../graph/heterograph.h"
#include "../c_api_common.h"
#include "./cuda/cuda_hashtable.cuh"
#include "../groot/array_scatter.h"
namespace dgl {
namespace groot {

struct BlockObject : public runtime::Object {
  BlockObject(){};
  ~BlockObject(){LOG(INFO) << "Calling blocks destructor \n" ;}
  BlockObject(
      DGLContext ctx, const std::vector<int64_t>& fanouts, int64_t layer,
      int64_t batch_size, DGLDataType dtype, cudaStream_t stream) {
    int64_t est_num_src = batch_size;
    int64_t est_num_dst = batch_size;
    for (int64_t i = 0; i < layer; i++) est_num_src *= (fanouts[i] + 1);
    for (int64_t i = 0; i <= layer; i++) est_num_dst *= (fanouts[i] + 1);
    _row = NDArray::Empty({est_num_dst}, dtype, ctx);
    _new_row = NDArray::Empty({est_num_dst}, dtype, ctx);
    _col = NDArray::Empty({est_num_dst}, dtype, ctx);
    _new_col = NDArray::Empty({est_num_dst}, dtype, ctx);
    _idx = NDArray::Empty({est_num_dst}, dtype, ctx);
    _indptr = NDArray::Empty({est_num_src + 1}, dtype, ctx);
    _outdeg = NDArray::Empty({est_num_src}, dtype, ctx);
    _newlen = NDArray::Empty({1}, dtype, DGLContext{kDGLCPU, 0});
    _table = std::make_shared<CudaHashTable>(dtype, ctx, est_num_dst, stream);
    // Todo SRC to DEST data strucutures are not needed for redudnant blocks
    _scattered_dest = ScatteredArray::Create(est_num_dst, 4, ctx, dtype, stream);
    _scattered_src = ScatteredArray::Create(est_num_src, 4, ctx, dtype, stream);
    _newlen.PinMemory_();
  };

  int64_t num_src, num_dst; // number of src (unique) and destination (unique) for buliding the dgl block object

  std::shared_ptr<CudaHashTable> _table;

  NDArray _row;  // input nodes in original ids (coo format)
  NDArray _new_row; // input nodes in reindex-ed ids (0 based indexing)
  NDArray _col;  // destination nodes in original ids (coo format)
  NDArray _new_col; // destination nodes inreindex-ed ids (coo format)
  NDArray _idx;  // place holder for storing index (optional)
  NDArray _outdeg;
  NDArray _indptr;
  NDArray _newlen;
  HeteroGraphRef _block_ref;

  // Depending on wheter they are scattered src or dest
  ScatteredArray _scattered_src;
  ScatteredArray _scattered_dest;

  void VisitAttrs(runtime::AttrVisitor *v) final {
    v->Visit("row", &_row);
    v->Visit("col", &_col);
    v->Visit("idx", &_idx);
    v->Visit("new_col", &_new_col);
    v->Visit("new_row", &_new_row);
    v->Visit("indptr", &_indptr);
    v->Visit("outdeg", &_outdeg);
    v->Visit("gidx", &_block_ref);
    v->Visit("scattered_src", &_scattered_src);
    v->Visit("scattered_dest", &_scattered_dest);
  }

  static constexpr const char * _type_key = "Block";
  DGL_DECLARE_OBJECT_TYPE_INFO(BlockObject, Object);
};  // block

class Block : public runtime::ObjectRef {
 public:
  const BlockObject* operator->() const {
    return static_cast<const BlockObject*>(obj_.get());
  }
  using ContainerType = BlockObject;
};

struct BlocksObject : public runtime::Object {
  enum BlockType{
     DATA_PARALLEL,
     SRC_TO_DEST,
     DEST_TO_SRC
  };
  ~BlocksObject(){
    LOG(INFO) << "Calling Blocks Object destructor";
  };
  int64_t key, _num_layer;
  // Todo this must be ref.
  std::vector<std::shared_ptr<BlockObject>> _blocks;

  NDArray _labels;  // labels of input nodes
  NDArray _feats;   // feature of output nodes
  NDArray _input_nodes;   // seeds
  NDArray _output_nodes;  // output nodes
// std::shared_ptr<CudaHashTable> _table;
// Data structures for slicing.
  BlockType _blockType;
  int switch_layer;
  ScatteredArray _scattered_frontier;


  DGLContext _ctx;
  BlocksObject(){};
  BlocksObject(
      DGLContext ctx,
      const std::vector<int64_t>& fanouts,
      int64_t batch_size,
      DGLDataType id_type,
      DGLDataType label_type,
      cudaStream_t stream, BlocksObject::BlockType type) {
    _ctx = ctx;
    _labels = NDArray::Empty({batch_size}, label_type, _ctx);
    _num_layer = fanouts.size();
    int expected_frontier_size = batch_size * 100;
    for (int64_t layer = 0; layer < _num_layer; layer++) {
      auto block =
          std::make_shared<BlockObject>(_ctx, fanouts, layer, batch_size, id_type, stream);
      _blocks.push_back(block);
    }
    // todo: scattered frontier creation is it correct.
    _scattered_frontier = ScatteredArray::Create(expected_frontier_size, 4, ctx, id_type, stream);
    int64_t est_output_nodes = batch_size;
    for (int64_t fanout : fanouts) est_output_nodes *= (fanout + 1);
    _blockType = type;
//    _table = std::make_shared<CudaHashTable>(id_type, _ctx, est_output_nodes, stream);
  };

  std::shared_ptr<BlockObject > GetBlock(int64_t layer){
    CHECK(layer < (int64_t )_blocks.size());
    return _blocks.at(layer);
  }

  void VisitAttrs(runtime::AttrVisitor *v) final {
    v->Visit("labels", &_labels);
    v->Visit("input_nodes", &_input_nodes);
    v->Visit("output_nodes", &_output_nodes);
    v->Visit("feats", &_feats);
  }

  static constexpr const char * _type_key = "Blocks";
  DGL_DECLARE_OBJECT_TYPE_INFO(BlocksObject, Object);
};

class Blocks : public runtime::ObjectRef {
 public:
  const BlocksObject* operator->() const {
    return static_cast<const BlocksObject*>(obj_.get());
  }
  using ContainerType = BlocksObject;
};

}  // namespace groot
}  // namespace dgl

#endif  // DGL_BLOCK_H
