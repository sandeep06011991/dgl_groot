//
// Created by juelin on 8/8/23.
//

#ifndef DGL_CUDA_HASHTABLE_CUH
#define DGL_CUDA_HASHTABLE_CUH

#include "../../runtime/cuda/cuda_common.h"
#include "cuda_constant.h"
#include <cuda_runtime.h>
#include <dgl/array.h>
#include <dgl/runtime/c_runtime_api.h>

namespace dgl {
namespace groot {
inline std::string ToReadableSize(size_t nbytes) {
  char buf[Constant::kBufferSize];
  if (nbytes > Constant::kGigabytes) {
    double new_size = (float)nbytes / Constant::kGigabytes;
    sprintf(buf, "%.2lf GB", new_size);
    return std::string(buf);
  } else if (nbytes > Constant::kMegabytes) {
    double new_size = (float)nbytes / Constant::kMegabytes;
    sprintf(buf, "%.2lf MB", new_size);
    return std::string(buf);
  } else if (nbytes > Constant::kKilobytes) {
    double new_size = (float)nbytes / Constant::kKilobytes;
    sprintf(buf, "%.2lf KB", new_size);
    return std::string(buf);
  } else {
    double new_size = (float)nbytes;
    sprintf(buf, "%.2lf Bytes", new_size);
    return std::string(buf);
  }
}

template <typename IdType> class OrderedHashTable;

template <typename IdType> class DeviceOrderedHashTable {
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

  DeviceOrderedHashTable &
  operator=(const DeviceOrderedHashTable &other) = default;

  inline __device__ ConstIterator SearchO2N(const IdType id) const {
    const IdType pos = SearchForPositionO2N(id);
    return &_o2n_table[pos];
  }

protected:
  const BucketO2N *_o2n_table;
  const BucketN2O *_n2o_table;
  const size_t _o2n_size;
  const size_t _n2o_size;

  explicit DeviceOrderedHashTable(const BucketO2N *const o2n_table,
                                  const BucketN2O *const n2o_table,
                                  const size_t o2n_size, const size_t n2o_size);

  inline __device__ IdType SearchForPositionO2N(const IdType id) const {
    IdType pos = HashO2N(id);

    // linearly scan for matching entry
    IdType delta = 1;
    while (_o2n_table[pos].key != id) {
      if(_o2n_table[pos].key == Constant::kEmptyKey){printf("not found searching %ld\n", id);}
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

template <typename IdType> class OrderedHashTable {
public:
  static constexpr size_t kDefaultScale = 3;

  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;
  using BucketN2O = typename DeviceOrderedHashTable<IdType>::BucketN2O;

  OrderedHashTable(const size_t size, DGLContext ctx,
                   const size_t scale = kDefaultScale);

  ~OrderedHashTable();

  // Disable copying
  OrderedHashTable(const OrderedHashTable &other) = delete;

  OrderedHashTable &operator=(const OrderedHashTable &other) = delete;

  void Reset(cudaStream_t stream);

  void FillWithDuplicates(const IdType *const input, const size_t num_input,
                          IdType *const unique, IdType *const num_unique,
                          cudaStream_t stream);

  void FillWithDupRevised(const IdType *const input, const int64_t num_input,
                          // IdType *const unique, IdType *const num_unique,
                          cudaStream_t stream);

  void FillWithDupMutable(IdType *const input, const size_t num_input,
                          cudaStream_t stream);

  void CopyUnique(IdType *const unique, cudaStream_t stream);

  IdType RefUnique(const IdType *&unique);

  /** add all neighbours of nodes in hashtable to hashtable */
  void FillNeighbours(const IdType *const indptr, const IdType *const indices,
                      cudaStream_t stream);

  void FillWithUnique(const IdType *const input, const size_t num_input,
                      cudaStream_t stream);

  size_t NumItems() const { return _num_items; }

  DeviceOrderedHashTable<IdType> DeviceHandle() const;

private:
  DGLContext _ctx;
  BucketO2N *_o2n_table;
  BucketN2O *_n2o_table;
  int64_t _o2n_size;
  int64_t _n2o_size;
  IdType _version;
  IdType _num_items;
};

class CudaHashTable {
private:
  void *_host_handle_ptr{nullptr};
  DGLDataType _dtype;
  DGLContext _ctx;
  int64_t _capacity;

public:
  cudaStream_t _stream;
  CudaHashTable(DGLDataType dtype, DGLContext ctx, int64_t capacity,
                cudaStream_t stream = runtime::getCurrentCUDAStream()) {
    _capacity = capacity;
    _stream = stream;
    _dtype = dtype;
    _ctx = ctx;
    ATEN_ID_TYPE_SWITCH(_dtype, IdType, {
      _host_handle_ptr =
          static_cast<void *>(new OrderedHashTable<IdType>(_capacity, _ctx));
    });
  }

  template <typename IdType> DeviceOrderedHashTable<IdType> DeviceHandle() {
    CHECK(_dtype == DGLDataTypeTraits<IdType>().dtype);
    return static_cast<OrderedHashTable<IdType> *>(_host_handle_ptr)
        ->DeviceHandle();
  }

  void FillWithUnique(NDArray arr, int64_t num_input) {
    ATEN_ID_TYPE_SWITCH(_dtype, IdType, {
      auto _handle = static_cast<OrderedHashTable<IdType> *>(_host_handle_ptr);
      const IdType *unique{nullptr};
      IdType num_unique = _handle->RefUnique(unique);
      if(num_input + num_unique >=  _capacity){std::cout << "Expected " << _capacity <<" but got " << num_input + num_unique <<"\n";}
      assert(num_input +  num_unique < _capacity);

      _handle->FillWithUnique(arr.Ptr<IdType>(), num_input, _stream);
      runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, _stream);
    });
  }

  void FillWithDuplicates(NDArray arr, int64_t num_input) {
    ATEN_ID_TYPE_SWITCH(_dtype, IdType, {
      auto _handle = static_cast<OrderedHashTable<IdType> *>(_host_handle_ptr);
      const IdType *unique{nullptr};
      IdType num_unique = _handle->RefUnique(unique);
      if(num_input + num_unique >= _capacity){std::cout << "Expected " << _capacity <<" but got " << num_input + num_unique <<"\n";}
      assert(num_input +  num_unique < _capacity);

      _handle->FillWithDupRevised(arr.Ptr<IdType>(), num_input, _stream);
      runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, _stream);
    });
  }

  void Reset() {
    ATEN_ID_TYPE_SWITCH(_dtype, IdType, {
      auto _handle = static_cast<OrderedHashTable<IdType> *>(_host_handle_ptr);
      _handle->Reset(_stream);
    });
  }

  NDArray RefUnique() {
    ATEN_ID_TYPE_SWITCH(_dtype, IdType, {
      auto _handle = static_cast<OrderedHashTable<IdType> *>(_host_handle_ptr);
      const IdType *unique{nullptr};
      IdType num_unique = _handle->RefUnique(unique);
      return NDArray::CreateFromRaw({num_unique}, _dtype, _ctx,
                                    (void *)(unique), false);
    });
  }

  int64_t NumItem() {
    ATEN_ID_TYPE_SWITCH(_dtype, IdType, {
      auto _handle = static_cast<OrderedHashTable<IdType> *>(_host_handle_ptr);
      const IdType *unique{nullptr};
      IdType num_unique = _handle->RefUnique(unique);
      return num_unique;
    });
  }
  NDArray CopyUnique() {
    ATEN_ID_TYPE_SWITCH(_dtype, IdType, {
      auto _handle = static_cast<OrderedHashTable<IdType> *>(_host_handle_ptr);
      auto ret = NDArray::Empty({(int64_t)_handle->NumItems()}, _dtype, _ctx);
      _handle->CopyUnique(static_cast<IdType *>(ret->data), _stream);
      return ret;
    });
  }

  ~CudaHashTable() {
    ATEN_ID_TYPE_SWITCH(_dtype, IdType, {
      auto _handle = static_cast<OrderedHashTable<IdType> *>(_host_handle_ptr);
      delete _handle;
      _host_handle_ptr = nullptr;
    });
  }
}; // CudaHashTable
} // namespace groot
} // namespace dgl
#endif // DGL_CUDA_HASHTABLE_CUH
