//
// Created by juelin on 2/22/24.
//

#ifndef DGL_PREPROCESS_H
#define DGL_PREPROCESS_H
#include <dgl/array.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>
#include <algorithm>
#include <numeric>
#include <thread>
#include <metis.h>
#include <filesystem>
#include <fstream>

namespace dgl::dev
{
struct EdgeWithData {
  int32_t _src{0};
  int32_t _dst{0};
  int32_t _data{0};
  EdgeWithData() = default;
  EdgeWithData(int32_t src, int32_t dst, int32_t data)
      : _src{src}, _dst{dst}, _data{data} {};
  bool operator==(const EdgeWithData &other) const {
    return other._src == _src && other._dst == _dst;
  }
  bool operator<(const EdgeWithData &other) const {
    if (_src == other._src) {
      return _dst < other._dst;
    } else {
      return _src < other._src;
    }
  }
};

struct Edge {
  int32_t _src{0};
  int32_t _dst{0};
  Edge(int32_t src, int32_t dst) : _src{src}, _dst{dst} {};
  Edge() = default;
  bool operator==(const Edge &other) const {
    return other._src == _src && other._dst == _dst;
  }
  bool operator<(const Edge &other) const {
    if (_src == other._src) {
      return _dst < other._dst;
    } else {
      return _src < other._src;
    }
  }
};


inline bool nextSNAPline(
    std::ifstream &infile, std::string &line, std::istringstream &iss,
    int64_t &src, int64_t &dest) {
  do {
    if (!getline(infile, line)) return false;
  } while (line.length() == 0 || line[0] == '#');
  iss.clear();
  iss.str(line);
  return !!(iss >> src >> dest);
}

inline void getID(std::vector<int64_t> &idMap, int64_t &id, int64_t &nextID) {
  if (idMap.size() <= id) {
    idMap.resize(id + 4096, -1);
  }

  if (idMap.at(id) == -1) {
    idMap.at(id) = nextID;
    nextID++;
  }
  id = idMap.at(id);
}

static std::pair<NDArray, NDArray> LoadSNAP(
    std::filesystem::path data_file, bool to_sym) {
  LOG(INFO) << "Loading from file: " << data_file;
  LOG(INFO) << "Convert to symmetric: " << to_sym;
  typedef std::vector<Edge> EdgeVec;
  EdgeVec edge_vec;
  std::vector<int64_t> idMap;
  int64_t max_id = 0, src = 0, dst = 0, nextID = 0, line_num = 0;
  std::string line;
  std::istringstream iss;
  std::ifstream infile(data_file.c_str());
  while (nextSNAPline(infile, line, iss, src, dst)) {
    if (src == dst) continue;
    getID(idMap, src, nextID);
    getID(idMap, dst, nextID);
    max_id = std::max(max_id, src);
    max_id = std::max(max_id, dst);
    edge_vec.push_back({(int32_t )src, (int32_t ) dst});
    if (to_sym) edge_vec.push_back({ (int32_t) dst, (int32_t ) src});  // convert to symmetric graph
    line_num++;
    if (line_num % int(2e7) == 0) {
      LOG(INFO) << "LoadSNAP Read: " << line_num / int(1e6) << "M edges";
    }
  }

  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());

  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
  edge_vec.shrink_to_fit();
  int64_t v_num = nextID;
  int64_t e_num = edge_vec.size();
  LOG(INFO) << "LoadSNAP v_num: " << v_num << " | e_num: " << e_num;

  std::vector<std::atomic<int64_t>> degree(v_num + 1);
  IdArray indptr = aten::NewIdArray(v_num + 1);
  IdArray indices = aten::NewIdArray(e_num);

  int64_t *indices_ptr = indices.Ptr<int64_t>();
  int64_t *out_start = indptr.Ptr<int64_t>();
  LOG(INFO) << "LoadSNAP compute degree";
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, edge_vec.size()),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          const Edge &e = edge_vec.at(i);
          degree.at(e._src)++;
          indices_ptr[i] = e._dst;
        }
      });

  LOG(INFO) << "LoadSNAP compute indptr";
  auto out_end =
      std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);

  CHECK_EQ(out_end - out_start, v_num + 1);
  CHECK_EQ(out_start[v_num], edge_vec.size());
  return {indptr, indices};
}

static std::tuple<IdArray, IdArray, IdArray> MakeSym(
    NDArray in_indptr, NDArray in_indices, NDArray data) {
  int64_t e_num = in_indices.NumElements();
  int64_t v_num = in_indptr.NumElements() - 1;
  LOG(INFO) << "MakeSym v_num: " << v_num << " | e_num: " << e_num;
  typedef std::vector<EdgeWithData> EdgeVec;
  EdgeVec edge_vec;
  edge_vec.resize(e_num * 2);
  auto _in_indptr = in_indptr.Ptr<int64_t >();
  auto _in_indices = in_indices.Ptr<int32_t >();
  auto data_ptr = data.Ptr<int32_t >();
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, v_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int32_t v = r.begin(); v < r.end(); v++) {
          int64_t start = _in_indptr[v];
          int64_t end = _in_indptr[v + 1];
          for (int64_t i = start; i < end; i++) {
            int32_t u = _in_indices[i];
            edge_vec.at(i * 2) = {v, u, data_ptr[i]};
            edge_vec.at(i * 2 + 1) = {u, v, data_ptr[i]};
          }
        }
      });

  LOG(INFO) << "MakeSym start sorting";
  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
  edge_vec.shrink_to_fit();
  int64_t cur_e_num = edge_vec.size();
  IdArray indptr = aten::NewIdArray(v_num + 1, DGLContext{kDGLCPU, 0}, 64);
  IdArray indices = aten::NewIdArray(cur_e_num, DGLContext{kDGLCPU, 0}, 32);
  IdArray retdata = aten::NewIdArray(cur_e_num, DGLContext{kDGLCPU, 0}, 32);

  memset(indptr.Ptr<void>(), 0, sizeof(int64_t ) * (v_num + 1));
  memset(retdata.Ptr<void>(), 0, sizeof(int32_t ) * cur_e_num);
  memset(indices.Ptr<void>(), 0, sizeof(int32_t ) * cur_e_num);
  std::vector<std::atomic<int64_t>> degree(v_num + 1);
  int32_t *indices_ptr = indices.Ptr<int32_t>();
  int32_t *retdata_ptr = retdata.Ptr<int32_t>();
  LOG(INFO) << "MakeSym compute degree";

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, edge_vec.size()),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          const auto &e = edge_vec.at(i);
          degree.at(e._src)++;
          indices_ptr[i] = e._dst;
          retdata_ptr[i] = e._data;
        }
      });

  LOG(INFO) << "MakeSym compute indptr";
  auto out_start = indptr.Ptr<int64_t>();
  auto out_end = std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
  LOG(INFO) << "MakeSym e_num after convert " << cur_e_num;

  CHECK_EQ(out_end - out_start, v_num + 1);
  CHECK_EQ(out_start[v_num], edge_vec.size());
  return {indptr, indices, retdata};
}  // MakeSym

static std::tuple<IdArray, IdArray> MakeSym(
    NDArray in_indptr, NDArray in_indices) {
  int64_t e_num = in_indices.NumElements();
  int64_t v_num = in_indptr.NumElements() - 1;
  LOG(INFO) << "MakeSym v_num: " << v_num << " | e_num: " << e_num;
  typedef std::vector<Edge> EdgeVec;
  EdgeVec edge_vec;
  edge_vec.resize(e_num * 2);
  auto _in_indptr = in_indptr.Ptr<int64_t >();
  auto _in_indices = in_indices.Ptr<int32_t >();
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, v_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int32_t v = r.begin(); v < r.end(); v++) {
          int64_t start = _in_indptr[v];
          int64_t end = _in_indptr[v + 1];
          for (int64_t i = start; i < end; i++) {
            int32_t u = _in_indices[i];
            edge_vec.at(i * 2) = {v, u};
            edge_vec.at(i * 2 + 1) = {u, v};
          }
        }
      });

  LOG(INFO) << "MakeSym start sorting";
  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
  edge_vec.shrink_to_fit();
  int64_t cur_e_num = edge_vec.size();
  IdArray indptr = aten::NewIdArray(v_num + 1, DGLContext{kDGLCPU, 0}, 64);
  IdArray indices = aten::NewIdArray(cur_e_num, DGLContext{kDGLCPU, 0}, 32);
  IdArray retdata = aten::NewIdArray(cur_e_num, DGLContext{kDGLCPU, 0}, 32);

  memset(indptr.Ptr<void>(), 0, sizeof(int64_t ) * (v_num + 1));
  memset(indices.Ptr<void>(), 0, sizeof(int32_t ) * cur_e_num);
  std::vector<std::atomic<int64_t>> degree(v_num + 1);
  int64_t *indices_ptr = indices.Ptr<int64_t>();
  LOG(INFO) << "MakeSym compute degree";

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, edge_vec.size()),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          const auto &e = edge_vec.at(i);
          degree.at(e._src)++;
          indices_ptr[i] = e._dst;
        }
      });

  LOG(INFO) << "MakeSym compute indptr";
  auto out_start = indptr.Ptr<int64_t>();
  auto out_end = std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
  LOG(INFO) << "MakeSym e_num after convert " << cur_e_num;

  CHECK_EQ(out_end - out_start, v_num + 1);
  CHECK_EQ(out_start[v_num], edge_vec.size());
  return {indptr, indices};
}  // MakeSym


IdArray MetisPartition(int64_t num_partition,
                         int64_t num_iteration,
                         int64_t num_initpart,
                         float unbalance_val,
                         bool obj_cut,
                         NDArray indptr,
                         NDArray indices,
                         NDArray node_weight,
                         NDArray edge_weight)
{
   idx_t nparts = num_partition;
   idx_t nvtxs = indptr.NumElements() - 1;
   idx_t num_edge = indices.NumElements();
  idx_t ncon = 1; // number of constraint

  if (node_weight.NumElements())
  {
    int64_t nvwgt = node_weight.NumElements();
    ncon = nvwgt / nvtxs;
    CHECK(nvwgt % nvtxs == 0);
    CHECK(node_weight->dtype.bits == 64);
  };

  if (edge_weight.NumElements())
  {
    CHECK(edge_weight.NumElements() == num_edge);
    CHECK(edge_weight->dtype.bits == 64);
  }

  NDArray ret = aten::NewIdArray(nvtxs, DGLContext{kDGLCPU, 0}, 64);
  auto part = reinterpret_cast<idx_t *>(ret->data);
  auto xadj = reinterpret_cast< idx_t *>(indptr->data); // uint64_t
  auto adjncy = reinterpret_cast< idx_t *>(indices->data); // uint32_t

  idx_t *vwgt = nullptr;
  idx_t *ewgt = nullptr;

  if (node_weight.NumElements())
    vwgt = node_weight.Ptr<idx_t>();
  if (edge_weight.NumElements())
    ewgt = edge_weight.Ptr<idx_t>();

  idx_t objval = 0;
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_ONDISK] = 1;
  options[METIS_OPTION_NITER] = num_iteration;
  options[METIS_OPTION_OBJTYPE] = obj_cut ? METIS_OBJTYPE_CUT : METIS_OBJTYPE_VOL;
  options[METIS_OPTION_DROPEDGES] = ewgt == nullptr;
  options[METIS_OPTION_NIPARTS] = num_initpart;

  // tpwgts: array of size ncon × nparts that is used to specify the fraction of vertex weight that should
  // be distributed to each sub-domain for each balance constraint. If all of the sub-domains are to be of
  // the same size for every vertex weight, then each of the ncon ×nparts elements should be set to
  // a value of 1 / nparts. If ncon is greater than 1, the target sub-domain weights for each sub-domain
  // are stored contiguously (similar to the vwgt array). Note that the sum of all of the tpwgts for a
  // give vertex weight should be one.
  std::vector<float> tpwgts(ncon * nparts, 1.0 / nparts);

  // ubvec: An array of size ncon that is used to specify the imbalance tolerance for each vertex weight, with 1
  // being perfect balance and nparts being perfect imbalance. A value of 1.05 for each of the ncon
  // weights is recommended.
  std::vector<float> ubvec(ncon, unbalance_val);

  int flag = METIS_PartGraphKway(&nvtxs,
                                 &ncon,
                                 xadj,
                                 adjncy,
                                 vwgt,
                                 NULL,
                                 ewgt,
                                 &nparts,
                                 tpwgts.data(), // tpwgts
                                 ubvec.data(), // ubvec
                                 options,
                                 &objval,
                                 part);

  float obj_scale = 1.0;
  if (ewgt != nullptr) {
    obj_scale *= std::accumulate(ewgt, ewgt + num_edge, 0ul) / num_edge;
  }
  objval /= obj_scale;

  if (obj_cut)
  {
    std::cout << "Partition a graph with " << nvtxs << " nodes and "
              << num_edge << " edges into " << num_partition << " parts and "
              << "get " << objval << " edge cuts with scale " << obj_scale << std::endl;
  }
  else
  {
    std::cout << "Partition a graph with " << nvtxs << " nodes and "
              << num_edge << " edges into " << num_partition << " parts and "
              << "the communication volume is " << objval << " with scale " << obj_scale << std::endl;
  }
  switch (flag)
  {
    case METIS_OK:
      return ret;
    case METIS_ERROR_INPUT:
      LOG(ERROR) << "Error in Metis partitioning: invalid input";
    case METIS_ERROR_MEMORY:
      LOG(ERROR) << "Error in Metis partitioning: not enough memory";
  };
  exit(-1);
};

static NDArray ExpandIndptr(NDArray in_indptr) {
  auto dtype = in_indptr->dtype;
  int64_t v_num = in_indptr.NumElements() - 1;
  ATEN_ID_TYPE_SWITCH(dtype, IdType, {
    auto _in_indptr = in_indptr.Ptr<IdType>();
    IdType e_num = _in_indptr[v_num];

    IdArray indices = NDArray::Empty({e_num}, dtype, in_indptr->ctx);
    auto _indices = indices.Ptr<IdType>();
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, v_num),
        [&](tbb::blocked_range<int64_t> r) {
          for (int64_t v = r.begin(); v < r.end(); v++) {
            IdType start = _in_indptr[v];
            IdType end = _in_indptr[v + 1];
            for (IdType i = start; i < end; i++) {
              _indices[i] = v;
            }
          }
        });
    return indices;
  });
  return aten::NullArray();
};


// Remove vertices with 0 degree
static std::pair<NDArray, NDArray> ReindexCSR(
    NDArray in_indptr, NDArray in_indices) {
  int64_t v_num = in_indptr.NumElements() - 1;
  int64_t e_num = in_indices.NumElements();
  LOG(INFO) << "ReindexCSR v_num before compact = " << v_num;

  IdArray out_indices = in_indices;
  int64_t *_in_indices = in_indices.Ptr<int64_t>();
  int64_t *_out_indices = out_indices.Ptr<int64_t>();
  const int64_t *_in_indptr = in_indptr.Ptr<int64_t>();
  std::vector<int64_t> degree(v_num, 0);
  std::vector<int64_t> org2new(v_num, 0);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, v_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          degree.at(i) = _in_indptr[i + 1] - _in_indptr[i];
        }
      });
  int64_t cur_v_num = 0;

  std::vector<int64_t> compacted_degree;

  for (int64_t i = 0; i < v_num; i++) {
    int64_t d = degree.at(i);
    if (d > 0) {
      org2new[i] = cur_v_num++;
      compacted_degree.push_back(d);
    }
  }

  compacted_degree.push_back(0);  // make exclusive scan work

  IdArray out_indptr = aten::NewIdArray(cur_v_num + 1);
  auto out_indptr_start = out_indptr.Ptr<int64_t>();
  auto out_indptr_end = std::exclusive_scan(
      compacted_degree.begin(), compacted_degree.end(), out_indptr_start, 0ll);
  CHECK_EQ(out_indptr_start[cur_v_num], e_num);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, e_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          _out_indices[i] = org2new[_in_indices[i]];
        }
      });

  LOG(INFO) << "ReindexCSR v_num after = " << cur_v_num;
  return {out_indptr, out_indices};
}

// Remove edges with 0 flag
static NDArray CompactCSR(NDArray in_indptr, NDArray flag) {
  int64_t v_num = in_indptr.NumElements() - 1;
  int64_t e_num = flag.NumElements();
  // CHECK_EQ(flag->dtype, DGLDataTypeTraits<bool>::dtype);
  LOG(INFO) << "ReindexCSR e_num before compact = " << e_num;
  bool *_in_indices = flag.Ptr<bool>();
  const int64_t *_in_indptr = in_indptr.Ptr<int64_t>();
  std::vector<int64_t> degree(v_num + 1);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, v_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          int64_t start = _in_indptr[i];
          int64_t end = _in_indptr[i + 1];
          for (int64_t j = start; j < end; j++) {
            degree.at(i) += (_in_indices[j]);
          }
        }
      });

  IdArray out_indptr = aten::NewIdArray(v_num + 1);
  auto out_indptr_start = out_indptr.Ptr<int64_t>();
  auto out_indptr_end =
      std::exclusive_scan(degree.begin(), degree.end(), out_indptr_start, 0ll);

  LOG(INFO) << "ReindexCSR e_num after = " << out_indptr_start[v_num];
  return out_indptr;
}

// Remove adjacency lists of vertices with 0 flag
static std::pair<NDArray, NDArray> PartitionCSR(NDArray in_indptr, NDArray in_indices, NDArray flag)
{
  CHECK_EQ(in_indices->dtype.bits, 64);
  CHECK_EQ(in_indptr->dtype.bits, 64);
  CHECK_EQ(flag->dtype.bits, 8);
  CHECK_EQ(flag.NumElements(), in_indptr.NumElements() - 1);

  int64_t v_num = in_indptr.NumElements() - 1;
  int64_t e_num = in_indices.NumElements();

  LOG(INFO) << "PartitionCSR e_num before compact = " << e_num;

  const auto _in_indices = in_indices.Ptr<int64_t >();
  const auto _in_indptr = in_indptr.Ptr<int64_t>();
  const auto _flag = flag.Ptr<bool >();

  std::vector<int64_t> degree(v_num + 1);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, v_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          if (_flag[i]) {
            degree.at(i) += _in_indptr[i + 1] - _in_indptr[i];
          } else {
            degree.at(i) = 0;
          }
        }
      });

  IdArray out_indptr = aten::NewIdArray(v_num + 1);
  auto _out_indptr = out_indptr.Ptr<int64_t>();
  auto _out_indptr_end = std::exclusive_scan(degree.begin(), degree.end(), _out_indptr, 0ll);
  const auto new_enum = _out_indptr[v_num];
  IdArray out_indices = aten::NewIdArray(new_enum);
  auto _out_indices = out_indices.Ptr<int64_t >();
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, v_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          if (_flag[i]) {
            auto out_start = _out_indptr[i];
            auto out_end = _out_indptr[i + 1];
            auto in_start = _in_indptr[i];
            auto in_end = _in_indptr[i + 1];
            assert(out_end - out_start == in_end - in_start);
            std::copy_n(_in_indices + in_start, in_end - in_start, _out_indices + out_start);
          }
        }
      });
  LOG(INFO) << "ReindexCSR e_num after = " << new_enum;

  return std::make_pair(out_indptr, out_indices);
}

} // dgl:dev
#endif  // DGL_PREPROCESS_H
