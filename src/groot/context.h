#ifndef DGL_DS_CONTEXT_H_
#define DGL_DS_CONTEXT_H_

#include <atomic>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <dmlc/thread_local.h>
#include <memory>
#include <nccl.h>
#include <vector>

//#include "./comm/comm_info.h"
//#include "coordinator.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace ds {

enum FeatMode {
  kFeatModeAllCache,
  kFeatModePartitionCache,
  kFeatModeReplicateCache
};

#define ENCODE_ID(i) (-(i)-2)

#define SAMPLER_ROLE 0
#define LOADER_ROLE 1

#define THREAD_LOCAL_PINNED_ARRAY_SIZE 256
#define N_PINNED_ARRAY 10

struct DSThreadEntry {
  IdArray pinned_array[N_PINNED_ARRAY];
  int pinned_array_counter;
  int check_point  = 0;

  static DSThreadEntry *ThreadLocal();
};

struct DSContext {
  bool initialized = false;
  int world_size;
  int rank;
  int thread_num;
  std::vector<ncclComm_t> nccl_comm;
//  std::vector<std::unique_ptr<CommInfo>> comm_info;
//  std::unique_ptr<Coordinator> coordinator;
//  std::unique_ptr<Coordinator> comm_coordinator;

  // Feature related arrays
  bool feat_loaded = false;
  FeatMode feat_mode;
  IdArray dev_feats, shared_feats, feat_pos_map;
  int feat_dim;

  // Graph related arrays
  bool graph_loaded = false;
  CSRMatrix dev_graph, uva_graph;
  int64_t n_cached_nodes, n_uva_nodes;
  IdArray adj_pos_map;

  // Kernel controller
  bool enable_kernel_control;
  std::atomic<int> sampler_queue_size{0}, loader_queue_size{0};

  // Communication control
  bool enable_comm_control;

  // Profiler
  // bool enable_profiler;
  // std::unique_ptr<Profiler> profiler;

  static DSContext *Global() {
    static DSContext instance;
    return &instance;
  }
};

} // namespace ds
} // namespace dgl

#endif
