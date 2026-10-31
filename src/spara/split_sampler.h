//
// Created by juelin on 2/13/24.
//

#ifndef DGL_SPLIT_SAMPLER_H
#define DGL_SPLIT_SAMPLER_H
#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/container.h>
#include <nccl.h>

#include <memory>
#include <queue>
#include <utility>

#include "../graph/unit_graph.h"
#include "../runtime/cuda/cuda_common.h"
#include "../runtime/cuda/cuda_hashtable.cuh"
#include "array_scatter.h"
#include "cuda/all2all.h"
#include "cuda/bitmap.h"
#include "cuda/index_select.cuh"
#include "cuda/map_edges.cuh"
#include "cuda/feat_cache.cuh"

namespace dgl::dev {

struct SplitBatch {
  int64_t _batch_id{-1};
  int64_t _num_dp{0};
  NDArray _feat{};

  bool reindexed{false};
  std::vector<aten::COOMatrix> _blocks;
  std::vector<int64_t > _num_srcs;
  std::vector<int64_t > _num_dsts;
  std::vector<NDArray> _frontiers;
  std::vector<ScatteredArray> _scattered_arrays;
  std::vector<HeteroGraphRef> _blockrefs;
  SplitBatch() = default;
  SplitBatch(
      int64_t batch_id, int64_t num_dp,
      const std::vector<int64_t >& num_srcs,
      const std::vector<int64_t >& num_dsts,
      const std::vector<NDArray>& frontiers,
      const std::vector<aten::COOMatrix>& blocks,
      const std::vector<ScatteredArray>& scattered_arrays)
      : _batch_id{batch_id},
        _num_dp{num_dp},
        _blocks{blocks},
        _num_srcs(num_srcs),
        _num_dsts(num_dsts),
        _frontiers{frontiers},
        _scattered_arrays{scattered_arrays} {}
};

class SplitSampler {
 private:
  aten::CSRMatrix _csc;
  NDArray _partition_map;
  std::vector<int64_t> _fanouts;
  std::shared_ptr<SplitBatch> _batch;
  std::shared_ptr<FeatCache> _featloader{nullptr};
  int64_t _v_num{0};
  int64_t _e_num{0};
  int64_t _next_id{0};
  int64_t _rank{-1};
  int64_t _world_size{0};
  int64_t _num_partitions{0};
  int64_t _num_dp{0};  // number of dp layers
  ncclComm_t _nccl_comm{nullptr};
  bool _use_bitmap{false};
  bool _use_featloader{false};
  DGLContext _ctx{};  // inferred from the ctx of the seeds

 public:
  ~SplitSampler() {
    if (_nccl_comm) ncclCommDestroy(_nccl_comm);
  }

  static std::shared_ptr<SplitSampler> Global() {
    static auto single_instance = std::make_shared<SplitSampler>();
    return single_instance;
  }

  void initNcclComm(int64_t nranks, ncclUniqueId commId, int64_t rank) {
    auto res = ncclCommInitRank(&_nccl_comm, nranks, commId, rank);
    CHECK_EQ(res, ncclSuccess);
  }

  void initFeatCache(const NDArray& pinned_feat, const NDArray& cached_ids) {
    CHECK_EQ(pinned_feat.IsPinned(), true);
    CHECK_NE(_rank, -1) << "Must set sampler's rank before start sampling";

    _featloader = std::make_shared<FeatCache>(_ctx, pinned_feat, cached_ids);
    _use_featloader = true;
  }

  void setGraph(
      const NDArray& indptr, const NDArray& indices, const NDArray& data) {
    CHECK(indptr.IsPinned() || indptr->ctx.device_type != kDGLCPU)
        << "Indptr must be pinned or on gpu";
    CHECK(indices.IsPinned() || indices->ctx.device_type != kDGLCPU)
        << "Indices must be pinned or on gpu";
    CHECK(
        data.IsPinned() || data->ctx == indices->ctx || data.NumElements() == 0)
        << "Data must be either empty or pinned or on gpu";
    CHECK(
        data.NumElements() == indices.NumElements() || data.NumElements() == 0);
    int64_t nrows = indptr.NumElements() - 1;
    int64_t ncols = indices.NumElements();
    if (data.NumElements() == 0)
      _csc = aten::CSRMatrix(nrows, ncols, indptr, indices);
    else
      _csc = aten::CSRMatrix(nrows, ncols, indptr, indices, data);
    _next_id = 0;
    _v_num = nrows;
    _e_num = ncols;
  }

  void setPartitionMap(const NDArray& partition_map) {
    _partition_map = partition_map;
  }

  void setFanouts(std::vector<int64_t> fanouts) {
    _fanouts = std::move(fanouts);
    _next_id = 0;
  }

  void useBitmap(bool use_bitmap) { _use_bitmap = use_bitmap; }

  void setRank(int64_t rank, int64_t world_size) {
    _rank = rank;
    _ctx = {DGLDeviceType::kDGLCUDA, (int32_t ) rank};
    _world_size = world_size;
  }

  void setNumPartitions(int64_t num_partitions) {
    _num_partitions = num_partitions;
  }

  void setNumDP(int64_t num_dp) { _num_dp = num_dp; }

  /*
   * seeds: the root vertex to be sampled from the graph
   * replace: use replace sampling or not
   */
  int64_t sampleOneBatch(const NDArray& seeds, bool replace) {
    CHECK_NE(_rank, -1) << "Must set sampler's rank before start sampling";
    CHECK(seeds.IsPinned() || seeds->ctx.device_type != DGLDeviceType::kDGLCPU)
        << "Seeds must be pinned or in GPU memory";
    std::vector<aten::COOMatrix> blocks;
    std::vector<NDArray> frontiers = {seeds};
    std::vector<int64_t > num_srcs;
    std::vector<int64_t > num_dsts;

    std::vector<ScatteredArray> scattered_arrays;
    for (int64_t layer = 0; layer < (int64_t)_fanouts.size(); layer++) {
      int64_t fanout = _fanouts.at(layer);
      auto frontier = frontiers.at(layer);
      // assume seeds are all local nodes
      aten::COOMatrix block = aten::CSRRowWiseSampling(
          _csc, frontier, fanout, aten::NullArray(), replace);

      auto unique_src = getUnique({frontier, block.col});
      num_srcs.push_back(unique_src.NumElements());
      num_dsts.push_back(frontier.NumElements());

      if (layer < _num_dp) {
        // Data-parallel layer: each rank samples its own seed shard's
        // neighborhood locally. No cross-GPU exchange, so no ScatteredArray is
        // built; the local unique frontier feeds the next (deeper) layer.
        frontiers.push_back(unique_src);
      } else {
        auto partition_idx = IndexSelect(
            _partition_map, unique_src, runtime::getCurrentCUDAStream());
        auto scatter_arr =
            ScatteredArray::Create(_v_num, _ctx, unique_src->dtype, _nccl_comm);
        // send remote frontiers to remote gpus
        // and receive frontiers from other gpus
        // build corresponding indices for mapping
        Scatter(
            _rank, _world_size, _num_partitions, unique_src, partition_idx,
            scatter_arr);
        frontiers.push_back(scatter_arr->unique_array);
        scattered_arrays.push_back(scatter_arr);
      }
      blocks.push_back(block);
    }

    int64_t batch_id = _next_id++;
    _batch = std::make_shared<SplitBatch>(
        batch_id, _num_dp, num_srcs, num_dsts, frontiers, blocks, scattered_arrays);

    if (_use_featloader) {
      // static cudaEvent_t sampling_event{nullptr};
      // if (sampling_event == nullptr) {
      //   CUDA_CALL(cudaEventCreate(&sampling_event));
      // }
      // CUDA_CALL(cudaEventRecord(sampling_event, runtime::getCurrentCUDAStream()));
      _batch->_feat = _featloader->Prefetch(_batch->_frontiers.at(frontiers.size()-1), nullptr);
    }

    return batch_id;
  }

  HeteroGraphRef getBlock(
      int64_t batch_id, int64_t layer, bool should_reindex = true) {
    CHECK(layer < _fanouts.size());
    std::shared_ptr<SplitBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id)
        << "getBlock batch id " << batch_id << " is not found in the pool";
    CHECK(batch->_blocks.size() == _fanouts.size());
    if (should_reindex && !batch->reindexed) {
      auto dtype = batch->_frontiers.at(_fanouts.size())->dtype;
      ATEN_ID_TYPE_SWITCH(dtype, IdType, {
        auto stream = runtime::getCurrentCUDAStream();
        auto device = runtime::DeviceAPI::Get(_ctx);

          // we unfortunately need to consider all cols since we removed remote cols in frontiers
        static int64_t *d_num_item{nullptr};
        if (d_num_item == nullptr) d_num_item = static_cast<int64_t *>(device->AllocWorkspace(_ctx, sizeof(int64_t)));

          for (size_t cur_layer = 0; cur_layer < _fanouts.size(); cur_layer++) {

            auto &block = _batch->_blocks.at(cur_layer);
            const auto &unique_dst = _batch->_frontiers.at(cur_layer);
            auto all_nodes = aten::Concat({unique_dst, block.col});
            auto num_input = all_nodes.NumElements();
            auto hash_table =
                runtime::cuda::OrderedHashTable<IdType>(num_input, _ctx, stream);
            NDArray unique = NDArray::Empty({num_input}, all_nodes->dtype, _ctx);
            hash_table.FillWithDuplicates(all_nodes.Ptr<IdType>(), num_input, unique.Ptr<IdType>(), d_num_item, stream);
            GPUMapEdges<IdType>(block, hash_table, stream);
          }
      });
      batch->reindexed = true;
    }

    if (batch->_blockrefs.size() != batch->_blocks.size()) {
      // create graph ref
      for (size_t i = 0; i < _fanouts.size(); i++) {
        auto& mat = batch->_blocks.at(i);
        int64_t num_src = batch->_num_srcs.at(i );
        int64_t num_dst = batch->_num_dsts.at(i);
        auto graph = HeteroGraphRef(CreateFromCOO(
            2, num_src, num_dst, mat.col, mat.row, false, false, COO_CODE));
        batch->_blockrefs.push_back(graph);
      }
    }
    CHECK_EQ(batch->_blockrefs.size(), batch->_blocks.size());
    return batch->_blockrefs.at(layer);
  }

  NDArray getFeature(int64_t batch_id) {
    std::shared_ptr<SplitBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id);
    CHECK_NE(_featloader, nullptr);
    _featloader->Sync();
    return batch->_feat;
  }

  NDArray getInputNode(int64_t batch_id) {
    std::shared_ptr<SplitBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id);
    CHECK(batch->_frontiers.size() == _fanouts.size() + 1);
    return batch->_frontiers.at(_fanouts.size());
  }
  NDArray getOutputNode(int64_t batch_id) {
    std::shared_ptr<SplitBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id);
    CHECK(!batch->_frontiers.empty());
    return batch->_frontiers.at(0);
  }
  NDArray getBlockData(int64_t batch_id, int64_t layer) {
    std::shared_ptr<SplitBatch> batch{_batch};
    CHECK_EQ(batch->_batch_id, batch_id);
    return batch->_blocks.at(layer).data;
  }

  ScatteredArray getScatteredArray(int64_t batch_id, int64_t layer) {
    CHECK_EQ(_batch->_batch_id, batch_id);
    return _batch->_scattered_arrays.at(layer - _num_dp);
  }
};

}  // namespace dgl::dev
#endif  // DGL_SPLIT_SAMPLER_H
