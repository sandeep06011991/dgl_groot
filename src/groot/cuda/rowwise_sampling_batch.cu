//
// Created by juelinliu on 11/18/23.
//

#include "rowwise_sampling_batch.cuh"
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <dgl/random.h>
#include <dgl/runtime/tensordispatch.h>

#include "../../array/cuda/atomic.cuh"
#include "../../array/cuda/utils.h"
#include <dgl/aten/array_ops.h>
#include "rowwise_sampling.cuh"
#include "cuda_mapping.cuh"
#include "cuda_index_select.cuh"

using TensorDispatcher = dgl::runtime::TensorDispatcher;
namespace dgl {
    namespace groot {
        constexpr size_t BLOCK_SIZE = 32;
        constexpr size_t TILE_SIZE = 1; // number of rows each block will cover

        namespace impl {
            /**
             * @brief Compute the size of each row in the sampled CSR, without replacement.
             *
             * @tparam IdType The type of node and edge indexes.
             * @param num_rows The number of rows to pick.
             * @param in_rows The set of rows to pick.
             * @param in_ptr The index where each row's edges start.
             * @param out_deg The size of each row in the sampled matrix, as indexed by
             * `in_rows` (output).
             */
            template<typename IdType>
            __global__ void _CSRRowWiseDegreeKernel(const int64_t num_rows,
                                                    const IdType *const in_rows,
                                                    const IdType *const in_ptr,
                                                    IdType *const out_deg) {
                const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

                if (tIdx < num_rows) {
                    const int in_row = in_rows[tIdx];
                    const int out_row = tIdx;
                    out_deg[out_row] = in_ptr[in_row + 1] - in_ptr[in_row];

                    if (out_row == num_rows - 1) {
                        // make the prefixsum work
                        out_deg[num_rows] = 0;
                    }
                }
            } // _CSRRowWiseDegreeKern

            /**
             * @brief Compute the size of each row in the sampled CSR, without replacement.
             *
             * @tparam IdType The type of node and edge indexes.
             * @param num_rows The number of rows to pick.
             * @param in_rows The set of rows to pick.
             * @param mask The mask indicating the presense of number in row
             * `in_rows` (output).
             */
            template<typename IdType>
            __global__ void _MaskIndex(const IdType * in_row, const int64_t num_rows, bool* mask) {
                const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
                if (tIdx < num_rows) {
                    mask[ in_row[tIdx] ] = 1;
                }
            }

            /**
             * @brief Compute the size of each row in the sampled CSR, without replacement.
             *
             * @tparam IdType The type of node and edge indexes.
             * @param num_rows The number of rows to pick.
             * @param in_rows The set of rows to pick.
             * @param mask The mask indicating the presense of number in row
             * `in_rows` (output).
             */
            template<typename IdType>
            __global__ void _MapIndptr(const IdType * in_row, const int64_t num_rows, IdType * out) {
                const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
                if (tIdx < num_rows) {
                    out[ in_row[tIdx] ] = tIdx;
                }
            }
            /**
             * @brief Compute the size of each row in the sampled CSR, without replacement.
             *
             * @tparam IdType The type of node and edge indexes.
             * @param num_elem The number of rows in in_ptr.
             * @param in_rows The set of rows to pick.
             * @param mask The mask indicating the presense of number in row
             * `in_rows` (output).
             */
            template<typename IdType>
            __global__ void _RemapIndex(const IdType * in_ptr, const int64_t num_elem, IdType* out) {
                const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
                if (tIdx < num_elem) {
                    const int start = in_ptr[tIdx];
                    const int end = in_ptr[tIdx + 1];
                    assert(end - start <= 1);
                    if (end - start == 1) {
                        out[start] = tIdx;
                    }
                }
            }

            /**
             * @brief Perform row-wise uniform sampling on a CSR matrix,
             * and generate a COO matrix, with replacement.
             *
             * @tparam IdType The ID type used for matrices.
             * @tparam TILE_SIZE The number of rows covered by each threadblock.
             * @param num_rows The number of rows to pick.
             * @param num_rows The number of cols to pick.
             * @param in_rows The set of rows to pick.
             * @param in_ptr The indptr array of the input CSR.
             * @param in_cols The indices array of the input CSR.
             * @param out_ptr The offset to write each row to in the output COO.
             * @param out_cols The columns of the output COO (output).
             */

            template<typename IdType, int TILE_SIZE>
            __global__ void _CSRRowWiseLoadingKernel(const int64_t num_rows, const int64_t num_cols,
                                                     const IdType *in_rows, const IdType *in_ptr, const IdType *in_cols,
                                                     const IdType *out_ptr,
                                                     IdType *const out_cols) {
                // assign one warp per row
                assert(blockDim.x == BLOCK_SIZE);
                int64_t out_row = blockIdx.x * TILE_SIZE;
                const int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);
                while (out_row < last_row) {
                    const int64_t row = in_rows[out_row];
                    const int64_t in_row_start = in_ptr[row];
                    const int64_t deg = in_ptr[row + 1] - in_ptr[row];
                    const int64_t out_row_start = out_ptr[out_row];
                    assert(out_row_start < num_cols);
                    for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
                        const IdType in_idx = in_row_start + idx;
                        const IdType out_idx = out_row_start + idx;
                        out_cols[out_idx] = in_cols[in_idx];
                    }
                    out_row += 1;
                }
            } // _CSRRowWiseLoadingKernel

        } // impl
        struct BatchCSRv1 {
            NDArray _indptr; // indptr
            NDArray _indices; // loaded adjacency lists
            NDArray _gids; //  global ids
            std::shared_ptr<CudaHashTable> _table; // map global ids to local (0 based) ids

            BatchCSRv1(NDArray indptr, NDArray indice, NDArray gids, std::shared_ptr<CudaHashTable> table) :
                    _indptr{std::move(indptr)}, _indices{std::move(indice)}, _gids{std::move(gids)},
                    _table{std::move(table)} {};
        };

        struct BatchCSRv2 {
            NDArray _indptr; // indptr for loaded adjlists
            NDArray _indices; // loaded adjacency lists
            NDArray _gids; //  global ids
            int64_t _v_num; // orignal number of vertices in the graph

            BatchCSRv2(int64_t v_num, NDArray indptr, NDArray indice, NDArray gids) :
            _v_num{v_num}, _indptr{std::move(indptr)}, _indices{std::move(indice)}, _gids{std::move(gids)} {};
        };

        // return: loaded indices and indptr
        template<DGLDeviceType XPU, typename IdType>
        BatchCSRv1 BatchCSRRowWiseLoadingV1(NDArray indptr, NDArray indices, const std::vector<NDArray> &rows,
                                            cudaStream_t stream = runtime::getCurrentCUDAStream());

        template<DGLDeviceType XPU, typename IdType>
        void
        BatchCSRRowWiseSamplingV1(const BatchCSRv1 &csr, const std::vector<NDArray> &rows, int64_t num_picks, bool replace,
                                  std::vector<std::shared_ptr<BlockObject>> blocks,
                                  cudaStream_t stream = runtime::getCurrentCUDAStream());


        template<DGLDeviceType XPU, typename IdType>
        BatchCSRv1
        BatchCSRRowWiseLoadingV1(NDArray indptr, NDArray indices, const std::vector<NDArray> &rows, cudaStream_t stream) {
            CHECK(rows.size() > 0) << "the number of rows to load must be greater than 0";
            const auto &ctx = rows[0]->ctx;
            auto idtype = rows[0]->dtype;
            auto device = runtime::DeviceAPI::Get(ctx);

            const IdType *in_ptr;
            const IdType *in_cols;

            if (indptr.IsPinned()) {
                void *ptr = indptr->data;
                CUDA_CALL(cudaHostGetDevicePointer(&ptr, ptr, 0));
                in_ptr = static_cast<IdType *>(ptr);
            } else {
                in_ptr = static_cast<IdType *>(indptr->data);
            }

            if (indices.IsPinned()) {
                void *ptr = indices->data;
                CUDA_CALL(cudaHostGetDevicePointer(&ptr, ptr, 0));
                in_cols = static_cast<IdType *>(ptr);
            } else {
                in_cols = static_cast<IdType *>(indices->data);
            }
            int64_t capacity = 0;
            for (const auto &row: rows) {
                capacity += row.NumElements();
            };
            capacity += rows.at(0).NumElements();
            bool fill_with_unique = true;
            for (const auto &row: rows){
                if (row.NumElements() != rows.at(0).NumElements()) fill_with_unique = false;
            }
            auto table = std::make_shared<CudaHashTable>(idtype, ctx, capacity, stream);
            for (const auto &row: rows) {
                table->FillWithDuplicates(row, row.NumElements());
            }
            NDArray gids = table->RefUnique();
            const IdType *const slice_rows = static_cast<const IdType *>(gids->data);
            int64_t num_rows = table->NumItem();
            IdType *out_deg = static_cast<IdType *>(device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
//            LOG(INFO) << "Compute degree";
            {
                const dim3 block(512);
                const dim3 grid((num_rows + block.x - 1) / block.x);
                CUDA_KERNEL_CALL(impl::_CSRRowWiseDegreeKernel, grid, block, 0,
                                 stream, num_rows, slice_rows, in_ptr, out_deg);
            }

            // fill out_ptr
//            LOG(INFO) << "Compute prefix sum for outptr";

            NDArray out_ptr_nd = NDArray::Empty({num_rows + 1}, idtype, ctx);
            IdType *out_ptr = static_cast<IdType *>(out_ptr_nd->data);
            size_t prefix_temp_size = 0;
            CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size, out_deg,
                                                    out_ptr, num_rows + 1, stream));
            void *prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
            CUDA_CALL(cub::DeviceScan::ExclusiveSum(
                    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
            device->FreeWorkspace(ctx, prefix_temp);
            device->FreeWorkspace(ctx, out_deg);
            device->StreamSync(ctx, stream);
            NDArray new_len_tensor = NDArray::Empty({1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
            CUDA_CALL(cudaMemcpyAsync(new_len_tensor->data, out_ptr + num_rows, sizeof(IdType), cudaMemcpyDeviceToHost, stream));
            device->StreamSync(ctx, stream);
            const IdType new_len = static_cast<const IdType *>(new_len_tensor->data)[0];
            NDArray out_cols_nd = NDArray::Empty({new_len}, idtype, ctx);
            IdType *out_cols = static_cast<IdType *>(out_cols_nd->data);
            {
                const dim3 block(BLOCK_SIZE);
                const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
                CUDA_KERNEL_CALL((impl::_CSRRowWiseLoadingKernel<IdType, TILE_SIZE>), grid, block, 0,
                                 stream, num_rows, new_len, slice_rows, in_ptr, in_cols, out_ptr, out_cols);
            }
            device->StreamSync(ctx, stream);

            return BatchCSRv1(out_ptr_nd, out_cols_nd, gids, table);
        };

        template<DGLDeviceType XPU, typename IdType>
        BatchCSRv2
        BatchCSRRowWiseLoadingV2(NDArray indptr, NDArray indices, const std::vector<NDArray> &rows, cudaStream_t stream) {
            CHECK(rows.size() > 0) << "the number of rows to load must be greater than 0";
            const auto &ctx = rows[0]->ctx;
            auto idtype = rows[0]->dtype;
            auto device = runtime::DeviceAPI::Get(ctx);

            const IdType *in_ptr;
            const IdType *in_cols;

            if (indptr.IsPinned()) {
                void *ptr = indptr->data;
                CUDA_CALL(cudaHostGetDevicePointer(&ptr, ptr, 0));
                in_ptr = static_cast<IdType *>(ptr);
            } else {
                in_ptr = static_cast<IdType *>(indptr->data);
            }

            if (indices.IsPinned()) {
                void *ptr = indices->data;
                CUDA_CALL(cudaHostGetDevicePointer(&ptr, ptr, 0));
                in_cols = static_cast<IdType *>(ptr);
            } else {
                in_cols = static_cast<IdType *>(indices->data);
            }

            int64_t v_num = indptr.NumElements() - 1;
            auto d_flags = static_cast<bool*>(device->AllocWorkspace(ctx, v_num * sizeof(bool)));
            CUDA_CALL(cudaMemset(d_flags, 0, v_num * sizeof(bool)));
            int64_t total_frontier_size = 0;
            for (const auto &row: rows) {
                const dim3 block(BLOCK_SIZE);
                const dim3 grid((row.NumElements() + block.x - 1) / block.x);
                total_frontier_size += row.NumElements();
                CUDA_KERNEL_CALL(impl::_MaskIndex<IdType>, grid, block, 0, stream, static_cast<IdType* >(row->data), row.NumElements(), d_flags);
            }

//
            IdType *d_num_unique = static_cast<IdType*>(device->AllocWorkspace(ctx, sizeof(IdType)));
            IdType h_num_unique = 0;
            size_t sum_temp_size = 0;
            CUDA_CALL(cub::DeviceReduce::Sum(nullptr, sum_temp_size, d_flags, d_num_unique, v_num, stream));
            void * sum_temp_storage = device->AllocWorkspace(ctx, sum_temp_size);
            CUDA_CALL(cub::DeviceReduce::Sum(sum_temp_storage, sum_temp_size, d_flags, d_num_unique, v_num, stream));
            device->StreamSync(ctx, stream);

            CUDA_CALL(cudaMemcpyAsync(&h_num_unique, d_num_unique, sizeof(IdType), cudaMemcpyDeviceToHost, stream));
            device->StreamSync(ctx, stream);
            const IdType num_rows = h_num_unique;
            LOG(INFO) << "frontier_size:" << total_frontier_size << " | num_unique: " << num_rows << " " << int(1.0 * num_rows / total_frontier_size * 100) <<"%";
            CHECK(num_rows > 0);

            NDArray gids = NDArray::Empty({num_rows}, idtype, ctx);
            IdType *slice_rows = static_cast<IdType*>(gids->data);
            auto d_in = cub::CountingInputIterator<IdType>(0);
            size_t   temp_storage_bytes = 0;
            cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, d_in, d_flags, slice_rows, d_num_unique, v_num, stream);
// Allocate temporary storage
            void     *d_temp_storage = device->AllocWorkspace(ctx, temp_storage_bytes);
// Run selection
            cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, slice_rows, d_num_unique, v_num, stream);
//            {
//                const dim3 block(256);
//                const dim3 grid((v_num + block.x - 1) / block.x);
//                CUDA_KERNEL_CALL(impl::_RemapIndex, grid, block, 0, stream, remap_ptr, v_num - 1, slice_rows);
//                device->StreamSync(ctx, stream);
//            }

            IdType *out_deg = static_cast<IdType *>(device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
            LOG(INFO) << "Compute degree";
            {
                const dim3 block(256);
                const dim3 grid((num_rows + block.x - 1) / block.x);
                CUDA_KERNEL_CALL(impl::_CSRRowWiseDegreeKernel, grid, block, 0,
                                 stream, num_rows, slice_rows, in_ptr, out_deg);
            }

            // fill out_ptr
//            LOG(INFO) << "Compute prefix sum for outptr";

            NDArray out_ptr_nd = NDArray::Empty({num_rows + 1}, idtype, ctx);
            IdType *out_ptr = static_cast<IdType *>(out_ptr_nd->data);
            size_t prefix_temp_size = 0;
            CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size, out_deg,
                                                    out_ptr, num_rows + 1, stream));
            void *prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
            CUDA_CALL(cub::DeviceScan::ExclusiveSum(
                    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
            device->FreeWorkspace(ctx, prefix_temp);
            device->FreeWorkspace(ctx, out_deg);
            device->StreamSync(ctx, stream);
            NDArray new_len_tensor = NDArray::Empty({1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
            CUDA_CALL(cudaMemcpyAsync(new_len_tensor->data, out_ptr + num_rows, sizeof(IdType), cudaMemcpyDeviceToHost, stream));
            device->StreamSync(ctx, stream);
            const IdType new_len = static_cast<const IdType *>(new_len_tensor->data)[0];
//            LOG(INFO) << "new len " << new_len;
            NDArray out_cols_nd = NDArray::Empty({new_len}, idtype, ctx);
            IdType *out_cols = static_cast<IdType *>(out_cols_nd->data);
            {
                const dim3 block(BLOCK_SIZE);
                const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
                CUDA_KERNEL_CALL((impl::_CSRRowWiseLoadingKernel<IdType, TILE_SIZE>), grid, block, 0,
                                 stream, num_rows, new_len, slice_rows, in_ptr, in_cols, out_ptr, out_cols);
            }
            device->FreeWorkspace(ctx, d_num_unique);
            device->FreeWorkspace(ctx, d_temp_storage);
            device->StreamSync(ctx, stream);

            return BatchCSRv2(v_num, out_ptr_nd, out_cols_nd, gids);
        };

        template<DGLDeviceType XPU, typename IdType>
        void
        BatchCSRRowWiseSamplingV1(const BatchCSRv1 &csr, const std::vector<NDArray> &rows,
                                  int64_t num_picks, bool replace,
                                  std::vector<std::shared_ptr<BlockObject>> blocks,
                                  cudaStream_t stream) {
            CHECK(rows.size() > 0) << "the number of rows to load must be greater than 0";
            CHECK(rows.size() == blocks.size()) << "the number of rows should be equal to the number of blocks";
            const auto &ctx = rows[0]->ctx;
            auto idtype = rows[0]->dtype;
            LOG(INFO) << "BatchLoadingV1 hashtable size: " << ToReadableSize(csr._table->NumBytes());

            for (size_t i = 0; i < rows.size(); i++) {
                auto &row = rows[i];
                auto &block = blocks[i];
                NDArray in_row_nd = NDArray::Empty({row.NumElements()}, idtype,
                                                   ctx); // local ids of the rows to be picked
                GPUMapEdges(row, in_row_nd, csr._table, stream);
                CSRRowWiseSamplingUniform<kDGLCUDA, IdType>(csr._indptr, csr._indices,
                                                                   in_row_nd,
                                                                   num_picks,
                                                                   replace,
                                                                   block,
                                                                   stream);
                IndexSelect(csr._gids, block->_row, block->_row, stream, "RowMapping");
                CHECK(block->_col.NumElements() == block->_row.NumElements());
            }
        };

        template<DGLDeviceType XPU, typename IdType>
        void
        BatchCSRRowWiseSamplingV2(const BatchCSRv2 &csr, const std::vector<NDArray> &rows,
                                  int64_t num_picks, bool replace,
                                  std::vector<std::shared_ptr<BlockObject>> blocks,
                                  cudaStream_t stream) {
            CHECK(rows.size() > 0) << "the number of rows to load must be greater than 0";
            CHECK(rows.size() == blocks.size()) << "the number of rows should be equal to the number of blocks";
            const auto &ctx = rows[0]->ctx;
            auto idtype = rows[0]->dtype;
            auto device = runtime::DeviceAPI::Get(ctx);
            int64_t hashTableSize = EstHashTableSize(csr._gids.NumElements(), sizeof(IdType));
            int64_t remapSize =  csr._v_num * sizeof(IdType);

            std::shared_ptr<CudaHashTable> table = nullptr;
            NDArray remap_indptr;
            if (remapSize < hashTableSize) {
                // use indptr for remapping
                remap_indptr = NDArray::Empty({csr._v_num}, idtype, ctx);
                const dim3 block(256);
                const dim3 grid((csr._v_num + block.x - 1) / block.x);
                CUDA_KERNEL_CALL(impl::_MapIndptr<IdType>, grid, block, 0, stream, csr._gids.Ptr<IdType>(), csr._gids.NumElements(), remap_indptr.Ptr<IdType>());
                LOG(INFO) << "BatchLoadingV2 remap indptr size: " << ToReadableSize(remapSize);
            } else {
                // use hashmap for remapping
                table = std::make_shared<CudaHashTable>(idtype, ctx, csr._gids.NumElements() + 1, stream);
                LOG(INFO) << "BatchLoadingV2 hashtable size: " << ToReadableSize(table->NumBytes());
                table->FillWithUnique(csr._gids, csr._gids.NumElements());
            }

            for (size_t i = 0; i < rows.size(); i++) {
                auto &row = rows[i];
                auto &block = blocks[i];
                NDArray in_row_nd = NDArray::Empty({row.NumElements()}, idtype, ctx); // local ids of the rows to be picked
                if (remap_indptr.defined()) {
                    assert(table == nullptr);
                    IndexSelect(remap_indptr, row, in_row_nd, stream);
                } else {
                    assert(table != nullptr);
                    GPUMapEdges(row, in_row_nd, table, stream);
                }
                CSRRowWiseSamplingUniform<kDGLCUDA, IdType>(csr._indptr, csr._indices,
                                                            in_row_nd,
                                                            num_picks,
                                                            replace,
                                                            block,
                                                            stream);
                IndexSelect(csr._gids, block->_row, block->_row, stream);
                CHECK(block->_col.NumElements() == block->_row.NumElements());
            }
            device->StreamSync(ctx, stream);
        };

        // do not use batchloading
        template<DGLDeviceType XPU, typename IdType>
        void CSRRowWiseSamplingUniformBatchV0(NDArray indptr, NDArray indices, std::vector<NDArray> rows,
                                              const int64_t num_picks, const bool replace,
                                              std::vector<std::shared_ptr<BlockObject>> blocks,
                                              cudaStream_t stream) {
            CHECK(rows.size() == blocks.size()) << "The size of rows and blocks should be the same";
            for (size_t i = 0; i < rows.size(); i++) {
                auto frontier = rows.at(i);
                auto block = blocks.at(i);
                    CSRRowWiseSamplingUniform<XPU, IdType>(indptr, indices, frontier,
                                                                num_picks, replace, block,
                                                                stream);
            }
        };

        // use hashmap for remapping
        template<DGLDeviceType XPU, typename IdType>
        void CSRRowWiseSamplingUniformBatchV1(NDArray indptr, NDArray indices, std::vector<NDArray> rows,
                                              const int64_t num_picks, const bool replace,
                                              std::vector<std::shared_ptr<BlockObject>> blocks,
                                              cudaStream_t stream) {
            CHECK(rows.size() == blocks.size()) << "The size of rows and blocks should be the same";
            auto csr = BatchCSRRowWiseLoadingV1<XPU, IdType>(indptr, indices, rows, stream);
            BatchCSRRowWiseSamplingV1<XPU, IdType>(csr, rows, num_picks, replace, blocks, stream);
        };

        // use indptr for remapping
        template<DGLDeviceType XPU, typename IdType>
        void CSRRowWiseSamplingUniformBatchV2(NDArray indptr, NDArray indices, std::vector<NDArray> rows,
                                              const int64_t num_picks, const bool replace,
                                              std::vector<std::shared_ptr<BlockObject>> blocks,
                                              cudaStream_t stream) {
            CHECK(rows.size() == blocks.size()) << "The size of rows and blocks should be the same";
            auto csr = BatchCSRRowWiseLoadingV2<XPU, IdType>(indptr, indices, rows, stream);
            BatchCSRRowWiseSamplingV2<XPU, IdType>(csr, rows, num_picks, replace, blocks, stream);
        };

template void CSRRowWiseSamplingUniformBatchV0<kDGLCUDA, int32_t>(
        NDArray, NDArray, std::vector<NDArray>, const int64_t, const bool,
        std::vector<std::shared_ptr<BlockObject>>, cudaStream_t);

template void CSRRowWiseSamplingUniformBatchV0<kDGLCUDA, int64_t>(
        NDArray, NDArray, std::vector<NDArray>, const int64_t, const bool,
        std::vector<std::shared_ptr<BlockObject>>, cudaStream_t);

template void CSRRowWiseSamplingUniformBatchV1<kDGLCUDA, int32_t>(
    NDArray, NDArray, std::vector<NDArray>, const int64_t, const bool,
    std::vector<std::shared_ptr<BlockObject>>, cudaStream_t);

template void CSRRowWiseSamplingUniformBatchV1<kDGLCUDA, int64_t>(
    NDArray, NDArray, std::vector<NDArray>, const int64_t, const bool,
    std::vector<std::shared_ptr<BlockObject>>, cudaStream_t);

template void CSRRowWiseSamplingUniformBatchV2<kDGLCUDA, int32_t>(
        NDArray, NDArray, std::vector<NDArray>, const int64_t, const bool,
        std::vector<std::shared_ptr<BlockObject>>, cudaStream_t);

template void CSRRowWiseSamplingUniformBatchV2<kDGLCUDA, int64_t>(
        NDArray, NDArray, std::vector<NDArray>, const int64_t, const bool,
        std::vector<std::shared_ptr<BlockObject>>, cudaStream_t);

    }
} // namespace dgl
