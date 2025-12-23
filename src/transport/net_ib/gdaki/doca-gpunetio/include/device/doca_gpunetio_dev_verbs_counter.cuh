/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file doca_gpunetio_dev_verbs_counter.cuh
 * @brief GDAKI CUDA device functions for One-sided Shared QP ops
 *
 * @{
 */

#ifndef DOCA_GPUNETIO_DEV_VERBS_COUNTER_CUH
#define DOCA_GPUNETIO_DEV_VERBS_COUNTER_CUH

#include "doca_gpunetio_dev_verbs_qp.cuh"
#include "doca_gpunetio_dev_verbs_cq.cuh"

/**
 * @brief Submit work requests to the NIC using the DB protocol.
 *
 * @param qps - Array of Queue Pair (QP)
 * @param prod_indices - Array of producer indices
 * @param num_qps - Number of Queue Pair (QP)
 */
template <unsigned int num_qps,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
          enum doca_gpu_dev_verbs_gpu_code_opt code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_db_multi_qps(
    struct doca_gpu_dev_verbs_qp **qps, uint64_t *prod_indices) {
    DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
    uint64_t old_prod_indices[num_qps];
    __be64 db_vals[num_qps];

#pragma unroll 2
    for (unsigned int i = 0; i < num_qps; i++) {
        doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qps[i]->sq_lock);
        old_prod_indices[i] = doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(
            &qps[i]->sq_wqe_pi, prod_indices[i]);
        if (old_prod_indices[i] < prod_indices[i]) {
            // Early rining of the DB to push WQEs to the NIC ASAP.
            __be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&qps[i]->sq_db);
            db_vals[i] = doca_gpu_dev_verbs_prepare_db(qps[i], prod_indices[i]);

#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
            if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
                doca_gpu_dev_verbs_async_store_release<sync_scope>((uint64_t *)db_ptr,
                                                                   (uint64_t)db_vals[i]);
            } else
#endif
            {
                doca_gpu_dev_verbs_fence_release<sync_scope>();
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
                { doca_gpu_dev_verbs_store_relaxed_mmio((uint64_t *)db_ptr, (uint64_t)db_vals[i]); }
#else
                {
                    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(
                        *((uint64_t *)db_ptr));
                    db_ptr_aref.store(db_vals[i], cuda::memory_order_relaxed);
                }
#endif
            }
        }
    }

#pragma unroll 2
    for (unsigned int i = 0; i < num_qps; i++) {
        if (old_prod_indices[i] < prod_indices[i]) {
            // In case the recovery path is triggered, the later DB ringing will cover for
            // correctness.
            doca_priv_gpu_dev_verbs_update_dbr(qps[i], prod_indices[i]);
            __be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&qps[i]->sq_db);
#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
            if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
                doca_gpu_dev_verbs_async_store_release<sync_scope>((uint64_t *)db_ptr,
                                                                   (uint64_t)db_vals[i]);
            } else
#endif
            {
                doca_gpu_dev_verbs_fence_release<sync_scope>();
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
                { doca_gpu_dev_verbs_store_relaxed_mmio((uint64_t *)db_ptr, (uint64_t)db_vals[i]); }
#else
                {
                    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(
                        *((uint64_t *)db_ptr));
                    db_ptr_aref.store(db_vals[i], cuda::memory_order_relaxed);
                }
#endif
            }
        }
        doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qps[i]->sq_lock);
    }
}

template <unsigned int num_qps,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_proxy_multi_qps(
    struct doca_gpu_dev_verbs_qp **qps, uint64_t *prod_indices) {
    DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
    doca_gpu_dev_verbs_fence_release<sync_scope>();

#pragma unroll 2
    for (unsigned int i = 0; i < num_qps; i++) {
        doca_gpu_dev_verbs_ring_proxy<resource_sharing_mode>(qps[i], prod_indices[i]);
    }
}

template <unsigned int num_qps,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_multi_qps(
    struct doca_gpu_dev_verbs_qp **qps, uint64_t *prod_indices) {
    DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
    if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
        const enum doca_gpu_dev_verbs_nic_handler qp_nic_handler =
            (enum doca_gpu_dev_verbs_nic_handler)__ldg((int *)&qps[0]->nic_handler);
        if (qp_nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB)
            doca_gpu_dev_verbs_submit_db_multi_qps<num_qps, resource_sharing_mode, sync_scope>(
                qps, prod_indices);
        else
            doca_gpu_dev_verbs_submit_proxy_multi_qps<num_qps, resource_sharing_mode, sync_scope>(
                qps, prod_indices);
    } else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
        doca_gpu_dev_verbs_submit_db_multi_qps<num_qps, resource_sharing_mode, sync_scope>(
            qps, prod_indices);
    } else {
        doca_gpu_dev_verbs_submit_proxy_multi_qps<num_qps, resource_sharing_mode, sync_scope>(
            qps, prod_indices);
    }
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static __forceinline__ void doca_gpu_dev_verbs_put_counter(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_addr raddr,
    struct doca_gpu_dev_verbs_addr laddr, size_t size, struct doca_gpu_dev_verbs_qp *companion_qp,
    struct doca_gpu_dev_verbs_addr counter_raddr, struct doca_gpu_dev_verbs_addr counter_laddr,
    uint64_t counter_val) {
    constexpr unsigned int num_qps = 2;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    uint64_t base_wqe_idx;
    uint64_t wqe_idx;
    size_t remaining_size = size;
    size_t size_;
    uint64_t num_chunks =
        doca_gpu_dev_verbs_div_ceil_aligned_pow2(size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
    num_chunks = num_chunks > 1 ? num_chunks : 1;

    // DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
    DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
    // DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

    base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks);
#pragma unroll 1
    for (uint64_t i = 0; i < num_chunks; i++) {
        wqe_idx = base_wqe_idx + i;
        size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    : remaining_size;
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        [[likely]] if (size_ > 0) {
            doca_gpu_dev_verbs_wqe_prepare_write(
                qp, wqe_ptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), raddr.key,
                laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), laddr.key, size_);
        } else {
            doca_gpu_dev_verbs_wqe_prepare_nop(qp, wqe_ptr, wqe_idx,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
        }
        remaining_size -= size_;
    }

    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);

    uint64_t companion_base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2);
    uint64_t companion_wqe_idx = companion_base_wqe_idx;

    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp, wqe_ptr, companion_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, wqe_idx,
                                        qp->cq_sq.cq_num);

    ++companion_wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companion_qp, wqe_ptr, companion_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, counter_raddr.addr, counter_raddr.key,
        counter_laddr.addr, counter_laddr.key, sizeof(uint64_t), counter_val, 0);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp, companion_base_wqe_idx,
                                                              companion_wqe_idx);

    doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
    uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};
    doca_gpu_dev_verbs_submit_multi_qps<num_qps, resource_sharing_mode,
                                        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(
        qps, prod_indices);
}

template <typename T,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static __forceinline__ void doca_gpu_dev_verbs_p_counter(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_addr raddr, T value,
    struct doca_gpu_dev_verbs_qp *companion_qp, struct doca_gpu_dev_verbs_addr counter_raddr,
    struct doca_gpu_dev_verbs_addr counter_laddr, uint64_t counter_val) {
    constexpr unsigned int num_qps = 2;
    uint64_t wqe_idx;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;

    // DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
    DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
    // DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

    wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, 1);
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(qp, wqe_ptr, wqe_idx,
                                                         DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                         raddr.addr, raddr.key, sizeof(T));
    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data<T>(qp, wqe_ptr, value);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, wqe_idx, wqe_idx);

    uint64_t companion_base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2);
    uint64_t companion_wqe_idx = companion_base_wqe_idx;

    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp, wqe_ptr, companion_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, wqe_idx,
                                        qp->cq_sq.cq_num);

    ++companion_wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companion_qp, wqe_ptr, companion_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, counter_raddr.addr, counter_raddr.key,
        counter_laddr.addr, counter_laddr.key, sizeof(uint64_t), counter_val, 0);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp, companion_base_wqe_idx,
                                                              companion_wqe_idx);

    doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
    uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};
    doca_gpu_dev_verbs_submit_multi_qps<num_qps, resource_sharing_mode,
                                        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(
        qps, prod_indices);
}

template <enum doca_gpu_dev_verbs_signal_op sig_op,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static __forceinline__ void doca_gpu_dev_verbs_put_signal_counter(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_addr raddr,
    struct doca_gpu_dev_verbs_addr laddr, size_t size, struct doca_gpu_dev_verbs_addr sig_raddr,
    struct doca_gpu_dev_verbs_addr sig_laddr, uint64_t sig_val,
    struct doca_gpu_dev_verbs_qp *companion_qp, struct doca_gpu_dev_verbs_addr counter_raddr,
    struct doca_gpu_dev_verbs_addr counter_laddr, uint64_t counter_val) {
    constexpr unsigned int num_qps = 2;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    uint64_t base_wqe_idx;
    uint64_t wqe_idx;
    size_t remaining_size = size;
    size_t size_;
    uint64_t num_chunks =
        doca_gpu_dev_verbs_div_ceil_aligned_pow2(size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
    num_chunks = num_chunks > 1 ? num_chunks : 1;

    // DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
    DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
    // DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

    // Put
    base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks + 1);
#pragma unroll 1
    for (uint64_t i = 0; i < num_chunks; i++) {
        wqe_idx = base_wqe_idx + i;
        size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    : remaining_size;
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        [[likely]] if (size_ > 0) {
            doca_gpu_dev_verbs_wqe_prepare_write(
                qp, wqe_ptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), raddr.key,
                laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), laddr.key, size_);
        } else {
            doca_gpu_dev_verbs_wqe_prepare_nop(qp, wqe_ptr, wqe_idx,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
        }
        remaining_size -= size_;
    }

    // Signal
    ++wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        qp, wqe_ptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, sig_raddr.addr, sig_raddr.key, sig_laddr.addr,
        sig_laddr.key, sizeof(uint64_t), sig_val, 0);

    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);

    // Counter
    uint64_t companion_base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2);
    uint64_t companion_wqe_idx = companion_base_wqe_idx;

    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp, wqe_ptr, companion_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, wqe_idx,
                                        qp->cq_sq.cq_num);

    ++companion_wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companion_qp, wqe_ptr, companion_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, counter_raddr.addr, counter_raddr.key,
        counter_laddr.addr, counter_laddr.key, sizeof(uint64_t), counter_val, 0);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp, companion_base_wqe_idx,
                                                              companion_wqe_idx);

    doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
    uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};
    doca_gpu_dev_verbs_submit_multi_qps<num_qps, resource_sharing_mode,
                                        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(
        qps, prod_indices);
}

template <enum doca_gpu_dev_verbs_signal_op sig_op,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static __forceinline__ void doca_gpu_dev_verbs_signal_counter(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_addr sig_raddr,
    struct doca_gpu_dev_verbs_addr sig_laddr, uint64_t sig_val,
    struct doca_gpu_dev_verbs_qp *companion_qp, struct doca_gpu_dev_verbs_addr counter_raddr,
    struct doca_gpu_dev_verbs_addr counter_laddr, uint64_t counter_val) {
    constexpr unsigned int num_qps = 2;
    uint64_t wqe_idx;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;

    // DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
    DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
    // DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

    // Signal
    wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, 1);
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        qp, wqe_ptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, sig_raddr.addr, sig_raddr.key, sig_laddr.addr,
        sig_laddr.key, sizeof(uint64_t), sig_val, 0);

    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, wqe_idx, wqe_idx);

    // Counter
    uint64_t companion_base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2);
    uint64_t companion_wqe_idx = companion_base_wqe_idx;

    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp, wqe_ptr, companion_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, wqe_idx,
                                        qp->cq_sq.cq_num);

    ++companion_wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companion_qp, wqe_ptr, companion_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, counter_raddr.addr, counter_raddr.key,
        counter_laddr.addr, counter_laddr.key, sizeof(uint64_t), counter_val, 0);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp, companion_base_wqe_idx,
                                                              companion_wqe_idx);

    doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
    uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};
    doca_gpu_dev_verbs_submit_multi_qps<num_qps, resource_sharing_mode,
                                        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(
        qps, prod_indices);
}

#endif /* DOCA_GPUNETIO_DEV_VERBS_COUNTER_CUH */
