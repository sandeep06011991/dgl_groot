//
// Created by juelin on 8/8/23.
//

#ifndef DGL_CUDA_HASHTABLE_CUH
#define DGL_CUDA_HASHTABLE_CUH
#include <cuda_runtime.h>
#include <dgl/array.h>
#include <dgl/runtime/c_runtime_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "cuda_constant.h"

namespace dgl {
namespace groot {

template <typename IdType>
class OrderedHashTable;
template <typename IdType>
class DeviceOrderedHashTable {
 public:
  struct BucketO2N {
    IdType key;
    IdType local;
    IdType index;
    IdType version;
  };

  struct BucketN2O {
    IdType global;
  };

  typedef const BucketO2N *ConstIterator;

  DeviceOrderedHashTable(const DeviceOrderedHashTable &other) = default;
  DeviceOrderedHashTable &operator=(const DeviceOrderedHashTable &other) =
      default;

  inline __device__ ConstIterator SearchO2N(const IdType id) const {
    const IdType pos = SearchForPositionO2N(id);
    return &_o2n_table[pos];
  }

 protected:
  const BucketO2N *_o2n_table;
  const BucketN2O *_n2o_table;
  const size_t _o2n_size;
  const size_t _n2o_size;

  explicit DeviceOrderedHashTable(
      const BucketO2N *const o2n_table, const BucketN2O *const n2o_table,
      const size_t o2n_size, const size_t n2o_size);

  inline __device__ IdType SearchForPositionO2N(const IdType id) const {
    IdType pos = HashO2N(id);

    // linearly scan for matching entry
    IdType delta = 1;
    while (_o2n_table[pos].key != id) {
      assert(_o2n_table[pos].key != Constant::kEmptyKey);
      pos = HashO2N(pos + delta);
      delta += 1;
    }
    assert(pos < _o2n_size);

    return pos;
  }

  inline __device__ IdType HashO2N(const IdType id) const {
    return id % _o2n_size;
  }

  friend class OrderedHashTable<IdType>;
};

template <typename IdType>
class OrderedHashTable {
 public:
  static constexpr size_t kDefaultScale = 3;

  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;
  using BucketN2O = typename DeviceOrderedHashTable<IdType>::BucketN2O;

  OrderedHashTable(
      const size_t size, DGLContext ctx, const size_t scale = kDefaultScale);

  ~OrderedHashTable();

  // Disable copying
  OrderedHashTable(const OrderedHashTable &other) = delete;
  OrderedHashTable &operator=(const OrderedHashTable &other) = delete;

  void Reset(cudaStream_t stream);

  void FillWithDuplicates(
      const IdType *const input, const size_t num_input, IdType *const unique,
      IdType *const num_unique, cudaStream_t stream);

  void FillWithDupRevised(
      const IdType *const input, const size_t num_input,
      // IdType *const unique, IdType *const num_unique,
      cudaStream_t stream);
  void FillWithDupMutable(
      IdType *const input, const size_t num_input, cudaStream_t stream);
  void CopyUnique(IdType *const unique, cudaStream_t stream);
  void RefUnique(const IdType *&unique, IdType *const num_unique);
  /** add all neighbours of nodes in hashtable to hashtable */
  void FillNeighbours(
      const IdType *const indptr, const IdType *const indices,
      cudaStream_t stream);

  void FillWithUnique(
      const IdType *const input, const size_t num_input, cudaStream_t stream);

  size_t NumItems() const { return _num_items; }

  DeviceOrderedHashTable<IdType> DeviceHandle() const;
 private:
  DGLContext _ctx;

  BucketO2N *_o2n_table;
  BucketN2O *_n2o_table;
  size_t _o2n_size;
  size_t _n2o_size;

  IdType _version;
  IdType _num_items;
};

class CudaHashTable {
 private:
  void *_cpu_hash_table_handle{nullptr};
  DGLDataType _dtype;
  DGLContext _ctx;
  int64_t _capacity;
  cudaStream_t _stream;

 public:
  CudaHashTable(){};
  CudaHashTable(
      DGLDataType dtype, DGLContext ctx, int64_t capacity,
      cudaStream_t stream) {
    _capacity = capacity;
    _stream = stream;
    _dtype = dtype;
    _ctx = ctx;
    if (_dtype.bits == 64) {
      _cpu_hash_table_handle =
          static_cast<void *>(new OrderedHashTable<int64_t>(_capacity, _ctx));
    } else if (_dtype.bits == 32) {
      _cpu_hash_table_handle =
          static_cast<void *>(new OrderedHashTable<int32_t>(_capacity, _ctx));
    } else {
      LOG(WARNING) << "unsupported hash table type";
    }
  }

  template <typename IdType>
  DeviceOrderedHashTable<IdType> DeviceHandle() {
    CHECK(_dtype == DGLDataTypeTraits<IdType>().dtype);
    return static_cast<OrderedHashTable<IdType> *>(_cpu_hash_table_handle)
        ->DeviceHandle();
  }

  void FillWithUnique(NDArray arr, int64_t num_input) {
    if (_dtype.bits == 64) {
      CHECK(arr->dtype.bits == 64);
      auto _handle =
          static_cast<OrderedHashTable<int64_t> *>(_cpu_hash_table_handle);
      _handle->FillWithUnique(arr.Ptr<int64_t>(), num_input, _stream);
    } else if (_dtype.bits == 32) {
      CHECK(arr->dtype.bits == 32);
      auto _handle =
          static_cast<OrderedHashTable<int32_t> *>(_cpu_hash_table_handle);
      _handle->FillWithUnique(arr.Ptr<int32_t>(), num_input, _stream);
    } else {
      LOG(WARNING) << "CudaHashTable unsupported data bits number: "
                   << _dtype.bits;
    }
  }

  void FillWithDuplicates(NDArray arr, int64_t num_input) {
    if (_dtype.bits == 64) {
      CHECK(arr->dtype.bits == 64);
      auto _handle =
          static_cast<OrderedHashTable<int64_t> *>(_cpu_hash_table_handle);
      _handle->FillWithDupRevised(arr.Ptr<int64_t>(), num_input, _stream);
    } else if (_dtype.bits == 32) {
      CHECK(arr->dtype.bits == 32);
      auto _handle =
          static_cast<OrderedHashTable<int32_t> *>(_cpu_hash_table_handle);
      _handle->FillWithDupRevised(arr.Ptr<int32_t>(), num_input, _stream);
    } else {
      LOG(WARNING) << "CudaHashTable unsupported data bits number: "
                   << _dtype.bits;
    }
  }

  void Reset() {
    if (_dtype.bits == 64) {
      auto _handle =
          static_cast<OrderedHashTable<int64_t> *>(_cpu_hash_table_handle);
      _handle->Reset(_stream);
    } else if (_dtype.bits == 32) {
      auto _handle =
          static_cast<OrderedHashTable<int32_t> *>(_cpu_hash_table_handle);
      _handle->Reset(_stream);
    } else {
      LOG(WARNING) << "CudaHashTable unsupported data bits number: "
                   << _dtype.bits;
    }
  }

  NDArray GetUnique() {
    if (_dtype.bits == 64) {
      using IdType = int64_t;
      auto _handle =
          static_cast<OrderedHashTable<IdType> *>(_cpu_hash_table_handle);
      const IdType *unique;
      IdType num_unique;
      _handle->RefUnique(unique, &num_unique);
      return NDArray::CreateFromRaw(
          {num_unique}, _dtype, _ctx, (void *)(unique), false);
    } else if (_dtype.bits == 32) {
      using IdType = int64_t;
      auto _handle =
          static_cast<OrderedHashTable<IdType> *>(_cpu_hash_table_handle);
      const IdType *unique;
      IdType num_unique;
      _handle->RefUnique(unique, &num_unique);
      return NDArray::CreateFromRaw(
          {num_unique}, _dtype, _ctx, (void *)(unique), false);
    } else {
      LOG(WARNING) << "CudaHashTable unsupported data bits number: "
                   << _dtype.bits;
      return aten::NullArray();
    }
  }

  ~CudaHashTable() {
    if (_dtype.bits == 64) {
      auto _handle =
          static_cast<OrderedHashTable<int64_t> *>(_cpu_hash_table_handle);
      delete _handle;
      _cpu_hash_table_handle = nullptr;
    } else if (_dtype.bits == 32) {
      auto _handle =
          static_cast<OrderedHashTable<int32_t> *>(_cpu_hash_table_handle);
      delete _handle;
      _cpu_hash_table_handle = nullptr;
    }
  }
};  // CudaHashTable
}  // namespace groot
}  // namespace dgl
#endif  // DGL_CUDA_HASHTABLE_CUH
