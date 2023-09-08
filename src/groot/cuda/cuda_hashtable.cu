//
// Created by juelin on 8/8/23.
//

#include <cub/cub.cuh>

#include "../../array/cuda/atomic.cuh"
#include "cuda_hashtable.cuh"
#include <dgl/runtime/tensordispatch.h>

namespace dgl {
namespace groot {
template <typename T, typename D> inline T RoundUpDiv(T target, D unit) {
  return (target + unit - 1) / unit;
}

template <typename IdType>
class MutableDeviceOrderedHashTable : public DeviceOrderedHashTable<IdType> {
public:
  typedef typename DeviceOrderedHashTable<IdType>::BucketO2N *IteratorO2N;
  typedef typename DeviceOrderedHashTable<IdType>::BucketN2O *IteratorN2O;

  explicit MutableDeviceOrderedHashTable(
      OrderedHashTable<IdType> *const host_table)
      : DeviceOrderedHashTable<IdType>(host_table->DeviceHandle()) {}

  inline __device__ IteratorO2N SearchO2N(const IdType id) {
    const IdType pos = this->SearchForPositionO2N(id);

    return GetMutableO2N(pos);
  }

  inline __device__ bool AttemptInsertAtO2N(const IdType pos, const IdType id,
                                            const IdType index,
                                            const IdType version) {
    auto iter = GetMutableO2N(pos);
    const IdType key =
        dgl::aten::cuda::AtomicCAS(&(iter->key), Constant::kEmptyKey, id);
    if (key == Constant::kEmptyKey) {
      iter->index = index;
      iter->version = version;
      return true;
    }
    return key == id;
  }

  inline __device__ IteratorO2N InsertO2N(const IdType id, const IdType index,
                                          const IdType version) {
    IdType pos = this->HashO2N(id);

    // linearly scan for an empty slot or matching entry
    IdType delta = 1;
    while (!AttemptInsertAtO2N(pos, id, index, version)) {
      pos = this->HashO2N(pos + delta);
      delta += 1;
    }
    return GetMutableO2N(pos);
  }

  inline __device__ IteratorN2O InsertN2O(const IdType pos,
                                          const IdType global) {
    GetMutableN2O(pos)->global = global;
    return GetMutableN2O(pos);
  }

  inline __device__ IdType IterO2NToPos(const IteratorO2N iter) {
    return iter - this->_o2n_table;
  }

  //  private:
  inline __device__ IteratorO2N GetMutableO2N(const IdType pos) {
    assert(pos < this->_o2n_size);
    // The parent class Device is read-only, but we ensure this can only be
    // constructed from a mutable version of OrderedHashTable<IdType>, making
    // this a safe cast to perform.
    return const_cast<IteratorO2N>(this->_o2n_table + pos);
  }

  inline __device__ IteratorN2O GetMutableN2O(const IdType pos) {
    assert(pos < this->_n2o_size);
    return const_cast<IteratorN2O>(this->_n2o_table + pos);
  }
};

/**
 * Calculate the number of buckets in the hashtable. To guarantee we can
 * fill the hashtable in the worst case, we must use a number of buckets which
 * is a power of two.
 * https://en.wikipedia.org/wiki/Quadratic_probing#Limitations
 */
size_t TableSize(const size_t num, const size_t scale) {
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
generate_hashmap_duplicates(const IdType *const items, const size_t num_items,
                            MutableDeviceOrderedHashTable<IdType> table,
                            const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      table.InsertO2N(items[index], index, version);
    }
  }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
generate_hashmap_unique(const IdType *const items, const size_t num_items,
                        MutableDeviceOrderedHashTable<IdType> table,
                        const IdType global_offset, const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using IteratorO2N =
      typename MutableDeviceOrderedHashTable<IdType>::IteratorO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const IteratorO2N bucket = table.InsertO2N(items[index], index, version);
      IdType pos = global_offset + static_cast<IdType>(index);
      // since we are only inserting unique items, we know their local id
      // will be equal to their index
      bucket->local = pos;
      table.InsertN2O(pos, items[index]);
    }
  }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(const IdType *items, const size_t num_items,
                              DeviceOrderedHashTable<IdType> table,
                              IdType *const num_unique, const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const BucketO2N &bucket = *table.SearchO2N(items[index]);
      if (bucket.index == index && bucket.version == version) {
        ++count;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    num_unique[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      num_unique[gridDim.x] = 0;
    }
  }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_count_hashmap_duplicates(
    const IdType *const items, const size_t num_items,
    MutableDeviceOrderedHashTable<IdType> table, IdType *const num_unique,
    const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      BucketO2N *iter = table.InsertO2N(items[index], index, version);
      if (iter->index == index && iter->version == version) {
        ++count;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;
  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    num_unique[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      num_unique[gridDim.x] = 0;
    }
  }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_count_hashmap_duplicates_mutable(
    IdType *items, const size_t num_items,
    MutableDeviceOrderedHashTable<IdType> table, IdType *const num_unique,
    const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      BucketO2N *iter = table.InsertO2N(items[index], index, version);
      if (iter->index == index && iter->version == version) {
        ++count;
        items[index] = table.IterO2NToPos(iter);
      } else {
        items[index] = Constant::kEmptyKey;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;
  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    num_unique[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      num_unique[gridDim.x] = 0;
    }
  }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void gen_count_hashmap_neighbour(
    const IdType *const items, const size_t num_items,
    const IdType *const indptr, const IdType *const indices,
    MutableDeviceOrderedHashTable<IdType> table, IdType *const num_unique,
    IdType *const block_max_degree, const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;
  IdType thread_max_degree = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const IdType orig_id = items[index];
      const IdType off = indptr[orig_id];
      const IdType off_end = indptr[orig_id + 1];
      thread_max_degree = (thread_max_degree > (off_end - off))
                              ? thread_max_degree
                              : (off_end - off);
      for (IdType j = off; j < off_end; j++) {
        const IdType nbr_orig_id = indices[j];
        BucketO2N *iter = table.InsertO2N(nbr_orig_id, index, version);
        if (iter->index == index && iter->version == version) {
          ++count;
        }
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space1;
  __shared__ typename BlockReduce::TempStorage temp_space2;
  IdType max_degree =
      BlockReduce(temp_space1).Reduce(thread_max_degree, cub::Max());
  count = BlockReduce(temp_space2).Sum(count);

  if (threadIdx.x == 0) {
    num_unique[blockIdx.x] = count;
    block_max_degree[blockIdx.x] = max_degree;
    if (blockIdx.x == 0) {
      num_unique[gridDim.x] = 0;
    }
  }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void gen_count_hashmap_neighbour_single_loop(
    const IdType *const items, const size_t num_items,
    const IdType *const indptr, const IdType *const indices,
    MutableDeviceOrderedHashTable<IdType> table, IdType *const num_unique,
    IdType *const, const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  const size_t thread_end = (num_items < block_end) ? num_items : block_end;

  IdType thread_count = 0;
  IdType cur_index = threadIdx.x + block_start;
  IdType cur_j = 0;
  IdType cur_off = 0;
  IdType cur_len = 0;
  if (cur_index < thread_end) {
    auto orig_id = items[cur_index];
    cur_off = indptr[orig_id];
    cur_len = indptr[orig_id + 1] - cur_off;
  }
  while (cur_index < thread_end) {
    if (cur_j >= cur_len) {
      cur_index += BLOCK_SIZE;
      if (cur_index >= thread_end)
        break;
      auto orig_id = items[cur_index];
      cur_off = indptr[orig_id];
      cur_len = indptr[orig_id + 1] - cur_off;
      cur_j = 0;
    } else {
      const IdType nbr_orig_id = indices[cur_j + cur_off];
      BucketO2N *bucket = table.InsertO2N(nbr_orig_id, cur_index, version);
      thread_count +=
          (bucket->index == cur_index && bucket->version == version);
      cur_j++;
    }
  }

  num_unique[blockIdx.x * blockDim.x + threadIdx.x] = thread_count;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    num_unique[gridDim.x * blockDim.x] = 0;
  }
}

template <typename T> struct BlockPrefixCallbackOp {
  T _running_total;

  __device__ BlockPrefixCallbackOp(const T running_total)
      : _running_total(running_total) {}

  __device__ T operator()(const T block_aggregate) {
    const T old_prefix = _running_total;
    _running_total += block_aggregate;
    return old_prefix;
  }
};

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
compact_hashmap(const IdType *const items, const size_t num_items,
                MutableDeviceOrderedHashTable<IdType> table,
                const IdType *const num_items_prefix,
                IdType *const num_unique_items, const IdType global_offset,
                const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    BucketO2N *kv;
    if (index < num_items) {
      kv = table.SearchO2N(items[index]);
      flag = kv->version == version && kv->index == index;
    } else {
      flag = 0;
    }

    if (!flag) {
      kv = nullptr;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (kv) {
      const IdType pos = global_offset + offset + flag;
      kv->local = pos;
      table.InsertN2O(pos, items[index]);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items = global_offset + num_items_prefix[gridDim.x];
  }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
compact_hashmap_revised(const IdType *const items, const size_t num_items,
                        MutableDeviceOrderedHashTable<IdType> table,
                        const IdType *const num_items_prefix,
                        const IdType global_offset, const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    BucketO2N *kv;
    if (index < num_items) {
      kv = table.SearchO2N(items[index]);
      flag = kv->version == version && kv->index == index;
    } else {
      flag = 0;
    }

    if (!flag) {
      kv = nullptr;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (kv) {
      const IdType pos = global_offset + offset + flag;
      kv->local = pos;
      table.InsertN2O(pos, items[index]);
    }
  }

  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //   *num_unique_items = global_offset + num_items_prefix[gridDim.x];
  // }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
compact_hashmap_revised_mutable(IdType *items_pos, const size_t num_items,
                                MutableDeviceOrderedHashTable<IdType> table,
                                const IdType *const num_items_prefix,
                                const IdType global_offset,
                                const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag = 0;
    BucketO2N *kv = nullptr;
    if (index < num_items && items_pos[index] != Constant::kEmptyKey) {
      kv = table.GetMutableO2N(items_pos[index]);
      flag = 1;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (kv) {
      const IdType pos = global_offset + offset + flag;
      kv->local = pos;
      table.InsertN2O(pos, kv->key);
    }
  }

  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //   *num_unique_items = global_offset + num_items_prefix[gridDim.x];
  // }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap_neighbour(
    const IdType *const items, const size_t num_items,
    const IdType *const indptr, const IdType *const indices,
    MutableDeviceOrderedHashTable<IdType> table,
    const IdType *const num_items_prefix, const IdType *const block_max_degree,
    IdType *const num_unique_items, const IdType global_offset,
    const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    IdType orig_id;
    IdType off;
    IdType len = 0;
    if (index < num_items) {
      orig_id = items[index];
      off = indptr[orig_id];
      len = indptr[orig_id + 1] - off;
    }
    assert(block_max_degree[blockIdx.x] >= len);

    for (IdType j = 0; j < block_max_degree[blockIdx.x]; j++) {
      BucketO2N *kv;
      if (j < len) {
        const IdType nbr_orig_id = indices[off + j];
        kv = table.SearchO2N(nbr_orig_id);
        flag = kv->version == version && kv->index == index;
      } else {
        flag = 0;
      }
      if (!flag)
        kv = nullptr;

      BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
      __syncthreads();

      if (kv) {
        const IdType pos = global_offset + offset + flag;
        kv->local = pos;
        table.InsertN2O(pos, items[index]);
      }
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items = global_offset + num_items_prefix[gridDim.x];
  }
}

template <typename IdType, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap_neighbour_single_loop(
    const IdType *const items, const size_t num_items,
    const IdType *const indptr, const IdType *const indices,
    MutableDeviceOrderedHashTable<IdType> table,
    const IdType *const num_items_prefix, const IdType *const,
    IdType *const num_unique_items, const IdType global_offset,
    const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BucketO2N = typename DeviceOrderedHashTable<IdType>::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  const size_t thread_offset =
      num_items_prefix[blockIdx.x * blockDim.x + threadIdx.x];
  const size_t thread_end = (num_items < block_end) ? num_items : block_end;

  IdType thread_pos = 0;
  IdType cur_index = threadIdx.x + block_start;
  IdType cur_j = 0;
  IdType cur_off = 0;
  IdType cur_len = 0;
  if (cur_index < thread_end) {
    auto orig_id = items[cur_index];
    cur_off = indptr[orig_id];
    cur_len = indptr[orig_id + 1] - cur_off;
  }
  while (cur_index < thread_end) {
    if (cur_j >= cur_len) {
      cur_index += BLOCK_SIZE;
      if (cur_index >= thread_end)
        break;
      auto orig_id = items[cur_index];
      cur_off = indptr[orig_id];
      cur_len = indptr[orig_id + 1] - cur_off;
      cur_j = 0;
    } else {
      const IdType nbr_orig_id = indices[cur_j + cur_off];
      BucketO2N *kv = table.SearchO2N(nbr_orig_id);
      if (kv->index == cur_index && kv->version == version) {
        kv->local = global_offset + thread_offset + thread_pos;
        table.InsertN2O(kv->local, nbr_orig_id);
        thread_pos++;
      }
      cur_j++;
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items =
        global_offset + num_items_prefix[gridDim.x * blockDim.x];
  }
}

// DeviceOrderedHashTable<IdType> implementation
template <typename IdType>
DeviceOrderedHashTable<IdType>::DeviceOrderedHashTable(
    const BucketO2N *const o2n_table, const BucketN2O *const n2o_table,
    const size_t o2n_size, const size_t n2o_size)
    : _o2n_table(o2n_table), _n2o_table(n2o_table), _o2n_size(o2n_size),
      _n2o_size(n2o_size) {}

template <typename IdType>
DeviceOrderedHashTable<IdType> OrderedHashTable<IdType>::DeviceHandle() const {
  return DeviceOrderedHashTable<IdType>(_o2n_table, _n2o_table, _o2n_size,
                                        _n2o_size);
}

// OrderedHashTable<IdType> implementation
template <typename IdType>
OrderedHashTable<IdType>::OrderedHashTable(const size_t size, DGLContext ctx,
                                           const size_t scale)
    : _o2n_table(nullptr),
#ifndef SXN_NAIVE_HASHMAP
      _o2n_size(TableSize(size, scale)),
#else
      _o2n_size(size),
#endif
      _n2o_size(size), _ctx(ctx), _version(0), _num_items(0) {
  // make sure we will at least as many buckets as items.
  auto device = runtime::DeviceAPI::Get(_ctx);
  _o2n_table = static_cast<BucketO2N *>(
      device->AllocWorkspace(_ctx, sizeof(BucketO2N) * _o2n_size));
  _n2o_table = static_cast<BucketN2O *>(
      device->AllocWorkspace(_ctx, sizeof(BucketN2O) * _n2o_size));

  CUDA_CALL(cudaMemset(_o2n_table, (int)Constant::kEmptyKey,
                       sizeof(BucketO2N) * _o2n_size));
  CUDA_CALL(cudaMemset(_n2o_table, (int)Constant::kEmptyKey,
                       sizeof(BucketN2O) * _n2o_size));
  // LOG(INFO) << "cuda hashtable init with " << ToReadableSize(_o2n_size) << "
  // O2N table size and " << ToReadableSize(_n2o_size) << " N2O table size";
}

template <typename IdType> OrderedHashTable<IdType>::~OrderedHashTable() {
  auto device = runtime::DeviceAPI::Get(_ctx);
  device->FreeDataSpace(_ctx, _o2n_table);
  device->FreeDataSpace(_ctx, _n2o_table);
}

template <typename IdType>
void OrderedHashTable<IdType>::Reset(cudaStream_t stream) {
  CUDA_CALL(cudaMemsetAsync(_o2n_table, (int)Constant::kEmptyKey,
                            sizeof(BucketO2N) * _o2n_size, stream));
  CUDA_CALL(cudaMemsetAsync(_n2o_table, (int)Constant::kEmptyKey,
                            sizeof(BucketN2O) * _n2o_size, stream));
  runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, stream);
  _version = 0;
  _num_items = 0;
}

template <typename IdType>
void OrderedHashTable<IdType>::FillWithDuplicates(const IdType *const input,
                                                  const size_t num_input,
                                                  IdType *const unique,
                                                  IdType *const num_unique,
                                                  cudaStream_t stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable<IdType>(this);
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  generate_hashmap_duplicates<IdType, Constant::kCudaBlockSize,
                              Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table, _version);
  device->StreamSync(_ctx, stream);

  // LOG(INFO) << "OrderedHashTable<IdType>::FillWithDuplicates
  // generate_hashmap_duplicates with " << num_input << " inputs";

  IdType *item_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * (grid.x + 1)));
  // // LOG(INFO)<< "OrderedHashTable<IdType>::FillWithDuplicates cuda
  // item_prefix malloc "<< ToReadableSize(sizeof(IdType) * (grid.x + 1));

  count_hashmap<IdType, Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      item_prefix, _version);
  device->StreamSync(_ctx, stream);

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace = device->AllocWorkspace(_ctx, workspace_bytes);
  // // LOG(INFO)<< "OrderedHashTable<IdType>::FillWithDuplicates cuda
  // item_prefix malloc "<< ToReadableSize(sizeof(IdType) * (num_input + 1));

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix, grid.x + 1,
                                          cu_stream));
  device->StreamSync(_ctx, stream);

  IdType *gpu_num_unique =
      static_cast<IdType *>(device->AllocWorkspace(_ctx, sizeof(IdType)));
  // // LOG(INFO) << "OrderedHashTable<IdType>::FillWithDuplicates cuda
  // gpu_num_unique malloc " << ToReadableSize(sizeof(IdType));

  compact_hashmap<IdType, Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      item_prefix, gpu_num_unique, _num_items,
                                      _version);
  device->StreamSync(_ctx, stream);

  device->CopyDataFromTo(gpu_num_unique, 0, num_unique, 0, sizeof(IdType), _ctx,
                         DGLContext{kDGLCPU, 0}, DGLDataType{});
  device->StreamSync(_ctx, stream);

  // If the number of input equals to 0, the kernel won't
  // be executed then the value of num_unique will be wrong.
  // We have to manually set the num_unique on this situation.
  if (num_input == 0) {
    *num_unique = _num_items;
  }

  // // LOG(INFO) << "OrderedHashTable<IdType>::FillWithDuplicates num_unique
  // "<< *num_unique;
  device->CopyDataFromTo(_n2o_table, 0, unique, 0,
                         sizeof(IdType) * (*num_unique), _ctx, _ctx,
                         DGLDataTypeTraits<IdType>::dtype);
  device->StreamSync(_ctx, stream);

  device->FreeWorkspace(_ctx, gpu_num_unique);
  device->FreeWorkspace(_ctx, item_prefix);
  device->FreeWorkspace(_ctx, workspace);

  _version++;
  _num_items = *num_unique;
}

template <typename IdType>
void OrderedHashTable<IdType>::CopyUnique(IdType *const unique,
                                          cudaStream_t stream) {
  auto device = runtime::DeviceAPI::Get(_ctx);
  device->CopyDataFromTo(_n2o_table, 0, unique, 0, sizeof(IdType) * _num_items,
                         _ctx, _ctx, DGLDataTypeTraits<IdType>::dtype);
  device->StreamSync(_ctx, stream);
}

template <typename IdType>
IdType OrderedHashTable<IdType>::RefUnique(const IdType *&unique) {
  unique = reinterpret_cast<const IdType *>(_n2o_table);
  return _num_items;
}

template <typename IdType>
void OrderedHashTable<IdType>::FillWithDupRevised(const IdType *const input,
                                                  const int64_t num_input,
                                                  cudaStream_t cu_stream) {
  if (num_input == 0)
    return;
  const int64_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  // LOG(INFO) << "FillWithDupRevised num_tiles: " << num_tiles;

  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable<IdType>(this);
  auto device = runtime::DeviceAPI::Get(_ctx);
  //  auto stream = static_cast<DGLStreamHandle>(cu_stream);
  const auto &stream = cu_stream;

  IdType *item_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * (grid.x + 1)));
  // LOG(INFO)<< "OrderedHashTable<IdType>::FillWithDupRevised cuda item_prefix
  // malloc "<< ToReadableSize(sizeof(IdType) * (grid.x + 1));

  // 1. insert into o2n table, collect each block's new insertion count
  generate_count_hashmap_duplicates<IdType, Constant::kCudaBlockSize,
                                    Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      item_prefix, _version);
  device->StreamSync(_ctx, stream);

  // LOG(INFO) << "OrderedHashTable<IdType>::FillWithDupRevised
  // generate_count_hashmap_duplicates with "<< num_input << " inputs";

  // 2. partial sum
  size_t workspace_bytes{0};
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace = device->AllocWorkspace(_ctx, workspace_bytes);
  // LOG(INFO)<< "OrderedHashTable<IdType>::FillWithDupRevised cuda workspace
  // malloc "<< ToReadableSize(workspace_bytes);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix, grid.x + 1,
                                          cu_stream));
  device->StreamSync(_ctx, stream);

  // 3. now each block knows where in n2o to put the node
  compact_hashmap_revised<IdType, Constant::kCudaBlockSize,
                          Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      item_prefix, _num_items, _version);
  device->StreamSync(_ctx, stream);

  NDArray new_len_tensor;
  if (dgl::runtime::TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty({1}, DGLDataTypeTraits<IdType>::dtype,
                                          DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty({1}, DGLDataTypeTraits<IdType>::dtype,
                                    DGLContext{kDGLCPU, 0});
  }

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));
  CUDA_CALL(cudaMemcpyAsync(new_len_tensor->data, item_prefix + grid.x,
                            sizeof(IdType), cudaMemcpyDeviceToHost, cu_stream));
  CUDA_CALL(cudaEventRecord(copyEvent, cu_stream));
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));
  auto new_element = static_cast<const IdType *>(new_len_tensor->data);
  _num_items += *new_element;
  //  if (_num_items > _n2o_size * 8) {
  //    LOG(INFO) << "OrderedHashTable<IdType>::FillWithDupRevised num_unique
  //    "<< _num_items << " new elements " << *new_element;
  //  }

  device->FreeWorkspace(_ctx, item_prefix);
  device->FreeWorkspace(_ctx, workspace);

  _version++;
}

template <typename IdType>
void OrderedHashTable<IdType>::FillWithDupMutable(IdType *input,
                                                  const size_t num_input,
                                                  cudaStream_t stream) {
  if (num_input == 0)
    return;
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable<IdType>(this);
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  IdType *item_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * (grid.x + 1)));
  // // LOG(INFO)<< "OrderedHashTable<IdType>::FillWithDupRevised cuda
  // item_prefix malloc "<< ToReadableSize(sizeof(IdType) * (grid.x + 1));

  // 1. insert into o2n table, collect each block's new insertion count
  generate_count_hashmap_duplicates_mutable<IdType, Constant::kCudaBlockSize,
                                            Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      item_prefix, _version);
  device->StreamSync(_ctx, stream);

  // // LOG(INFO) << "OrderedHashTable<IdType>::FillWithDupRevised
  // generate_count_hashmap_duplicates with "<< num_input << " inputs";

  // 2. partial sum
  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace = device->AllocWorkspace(_ctx, workspace_bytes);
  // // LOG(INFO)<< "OrderedHashTable<IdType>::FillWithDupRevised cuda workspace
  // malloc "<< ToReadableSize(workspace_bytes);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix, grid.x + 1,
                                          cu_stream));
  device->StreamSync(_ctx, stream);

  // 3. now each block knows where in n2o to put the node
  compact_hashmap_revised_mutable<IdType, Constant::kCudaBlockSize,
                                  Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      item_prefix, _num_items, _version);
  device->StreamSync(_ctx, stream);
  IdType tmp;
  device->CopyDataFromTo(item_prefix + grid.x, 0, &tmp, 0, sizeof(IdType), _ctx,
                         DGLContext{kDGLCPU, 0},
                         DGLDataTypeTraits<IdType>::dtype);
  device->StreamSync(_ctx, stream);
  _num_items += tmp;

  // // LOG(INFO) << "OrderedHashTable<IdType>::FillWithDupRevised num_unique
  // "<< _num_items;

  device->FreeWorkspace(_ctx, item_prefix);
  device->FreeWorkspace(_ctx, workspace);

  _version++;
}

template <typename IdType>
void OrderedHashTable<IdType>::FillNeighbours(const IdType *const indptr,
                                              const IdType *const indices,
                                              cudaStream_t stream) {
  const size_t num_input = _num_items;
  const IdType *const input = reinterpret_cast<IdType *>(_n2o_table);

  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable<IdType>(this);
  auto device = runtime::DeviceAPI::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  size_t n_item_prefix = grid.x * block.x + 1;
  IdType *item_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * n_item_prefix));
  // // LOG(INFO)<< "OrderedHashTable<IdType>::FillNeighbours cuda item_prefix
  // malloc "<< ToReadableSize(sizeof(IdType) * n_item_prefix);

  // 1. insert into o2n table, collect each block's new insertion count
  gen_count_hashmap_neighbour_single_loop<IdType, Constant::kCudaBlockSize,
                                          Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, indptr, indices,
                                      device_table, item_prefix, nullptr,
                                      _version);
  device->StreamSync(_ctx, stream);

  // 2. partial sum
  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), n_item_prefix, cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace = device->AllocWorkspace(_ctx, workspace_bytes);
  // // LOG(INFO)<< "OrderedHashTable<IdType>::FillNeighbours cuda item_prefix
  // malloc "<< ToReadableSize(workspace_bytes);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix,
                                          n_item_prefix, cu_stream));
  device->StreamSync(_ctx, stream);

  IdType *gpu_num_unique =
      static_cast<IdType *>(device->AllocWorkspace(_ctx, sizeof(IdType)));

  // 3.now each block knows where in n2o to put the node
  compact_hashmap_neighbour_single_loop<IdType, Constant::kCudaBlockSize,
                                        Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, indptr, indices,
                                      device_table, item_prefix, nullptr,
                                      gpu_num_unique, _num_items, _version);
  device->StreamSync(_ctx, stream);
  if (num_input != 0) {
    device->CopyDataFromTo(gpu_num_unique, 0, &_num_items, 0, sizeof(IdType),
                           _ctx, DGLContext{kDGLCPU, 0},
                           DGLDataTypeTraits<IdType>::dtype);
    device->StreamSync(_ctx, stream);
  }

  // // LOG(INFO) << "OrderedHashTable<IdType>::FillWithDuplicates num_unique
  // "<< _num_items;

  device->FreeWorkspace(_ctx, gpu_num_unique);
  device->FreeWorkspace(_ctx, item_prefix);
  device->FreeWorkspace(_ctx, workspace);

  _version++;
}

template <typename IdType>
void OrderedHashTable<IdType>::FillWithUnique(const IdType *const input,
                                              const size_t num_input,
                                              cudaStream_t stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable<IdType>(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  generate_hashmap_unique<IdType, Constant::kCudaBlockSize,
                          Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      _num_items, _version);
  runtime::DeviceAPI::Get(_ctx)->StreamSync(_ctx, stream);

  _version++;
  _num_items += num_input;

  // // LOG(INFO) << "OrderedHashTable<IdType>::FillWithUnique insert " <<
  // num_input<< " items, now " << _num_items << " in total";
}

template class OrderedHashTable<int32_t>;

template class OrderedHashTable<int64_t>;
} // namespace groot
} // namespace dgl