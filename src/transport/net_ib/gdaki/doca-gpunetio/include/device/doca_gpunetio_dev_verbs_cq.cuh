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
 * @file doca_gpunetio_dev_verbs_cq.cuh
 * @brief GDAKI CUDA device functions for CQ management
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEV_VERBS_CQ_H
#define DOCA_GPUNETIO_DEV_VERBS_CQ_H

#include <errno.h>

#include "doca_gpunetio_dev_verbs_common.cuh"

/**
 * @brief Return device CQ SQ pointer from a device QP
 *
 * @param[in] qp - Dev QP pointer
 *
 * @return Dev CQ pointer
 */
__device__ static __forceinline__ struct doca_gpu_dev_verbs_cq *doca_gpu_dev_verbs_qp_get_cq_sq(
    struct doca_gpu_dev_verbs_qp *qp) {
    return &(qp->cq_sq);
}

template <enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ struct doca_gpu_dev_verbs_cq *doca_gpu_dev_verbs_qp_get_cq(
    struct doca_gpu_dev_verbs_qp *qp) {
    static_assert(qp_type == DOCA_GPUNETIO_VERBS_QP_SQ,
                  "only SQ CQ polling is currently supported");
    return doca_gpu_dev_verbs_qp_get_cq_sq(qp);
}

/**
 * @brief Increament and round up CQE id
 *
 * @param[in] cqe_idx - cqe idx
 * @param[in] increment - cqe idx increment
 *
 * @return cqe incremented idx
 */
__device__ static __forceinline__ uint32_t doca_gpu_dev_verbs_cqe_idx_inc_mask(uint32_t cqe_idx,
                                                                               uint32_t increment) {
    return (cqe_idx + increment) & DOCA_GPUNETIO_VERBS_CQE_CI_MASK;
}

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
/**
 * @brief Print error CQE values
 *
 * @param[in] cqe64 - erroneous cqe
 *
 * @return
 */
__device__ static __forceinline__ void doca_gpu_dev_verbs_cq_print_cqe_err(
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64) {
    struct doca_gpunetio_ib_mlx5_err_cqe_ex *err_cqe =
        (struct doca_gpunetio_ib_mlx5_err_cqe_ex *)cqe64;

    printf(
        "got completion with err: "
        "syndrome=%#x, vendor_err_synd=%#x, "
        "hw_err_synd=%#x, hw_synd_type=%#x, wqe_counter=%u\n",
        err_cqe->syndrome, err_cqe->vendor_err_synd, err_cqe->hw_err_synd, err_cqe->hw_synd_type,
        doca_gpu_dev_verbs_bswap16(err_cqe->wqe_counter));
}
#endif

// ==================== Poll One CQ At ====================

/**
 * @brief [Internal] Poll the Completion Queue (CQ) at a specific index.
 * This polling algorithm is for non-collapsed CQ on GPU.
 * This function does not update the SW consumer index nor guarantees the ordering.
 * It also does not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_one_cq_device_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    uint8_t *cqe = (uint8_t *)doca_gpu_dev_verbs_load_const((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = doca_gpu_dev_verbs_load_const(&cq->cqe_num);
    const uint64_t cqe_rsvd = doca_gpu_dev_verbs_load_const(&cq->cqe_rsvd);
    uint64_t cons_index_in_cq = cons_index + cqe_rsvd;
    uint32_t idx = cons_index_in_cq & (cqe_num - 1);
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 =
        (struct doca_gpunetio_ib_mlx5_cqe64 *)(cqe + (idx * DOCA_GPUNETIO_VERBS_CQE_SIZE));

    uint64_t cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);

    if (cons_index < cqe_ci) return 0;
    if (cons_index >= cqe_ci + cqe_num) return EBUSY;

    uint8_t opown;
    uint8_t opcode;
    bool observed_completion;

#if __CUDA_ARCH__ >= 900
    opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);

    observed_completion =
        !((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num));
#else
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global(&cqe64->wqe_counter_sig_op_own_raw);
    cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
    wqe_counter = cqe_chunk >> 16;
    opown = cqe_chunk & 0xff;

    observed_completion =
        !((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num)) &&
        (wqe_counter == ((uint32_t)cons_index & 0xffff));
#endif

    if (!observed_completion) return EBUSY;

    opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
    if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif
    return (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) * -EIO;
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index.
 * This polling algorithm is for non-collapsed CQ on GPU.
 * This function does not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_one_cq_device_at() returns 0. If the completion is
 * not available, returns EBUSY. If it is a completion with error, returns a
 * negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_one_cq_device_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    int status =
        doca_priv_gpu_dev_verbs_poll_one_cq_device_at<resource_sharing_mode>(cq, cons_index);
    if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire_nvidia_nic();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
    }
    return status;
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index.
 * This polling algorithm is for CQ on host.
 * This function does not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_one_cq_host_at() returns 0. If the completion is
 * not available, returns EBUSY. If it is a completion with error, returns a
 * negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_one_cq_host_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    int status = 0;
    uint64_t cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
    if (cons_index >= cqe_ci) status = EBUSY;
    doca_gpu_dev_verbs_fence_acquire_nvidia_nic();
    return status;
}

/**
 * @brief [Internal] Poll the collapsed Completion Queue (CQ) at a specific index.
 * This function does not update the SW consumer index nor guarantees the ordering.
 * It also does not wait for the completion to arrive.
 *
 * @param cq - Collapsed Completion Queue (CQ)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @param new_cqe_ci - New CQ Consumer Index
 * @return On success, doca_priv_gpu_dev_verbs_poll_cq_one_collapsed_at() returns 0. If the
 * completion is not available, returns EBUSY. If it is a completion with error, returns a
 * negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_cq_one_collapsed_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index, uint64_t *new_cqe_ci) {
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 =
        (struct doca_gpunetio_ib_mlx5_cqe64 *)doca_gpu_dev_verbs_load_const(
            (uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = doca_gpu_dev_verbs_load_const(&cq->cqe_num);
    uint64_t cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);

    [[unlikely]] if (cons_index < cqe_ci)
        return 0;
    if (cons_index >= cqe_ci + cqe_num) return EBUSY;

    uint32_t cqe_chunk =
        doca_gpu_dev_verbs_load_relaxed_sys_global(&cqe64->wqe_counter_sig_op_own_raw);
    cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
    uint16_t wqe_counter = cqe_chunk >> 16;
    uint8_t opown = cqe_chunk & 0xff;
    uint8_t opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

    if ((opcode == DOCA_GPUNETIO_IB_MLX5_CQE_INVALID) ||
        ((cqe_ci <= cons_index) &&
         ((uint16_t)((uint16_t)cons_index - wqe_counter - (uint16_t)2) < cqe_num))) {
        return EBUSY;
    }

    ++wqe_counter;
    *new_cqe_ci = ((cons_index & ~(0xFFFFULL)) | wqe_counter) +
                  (((uint16_t)cons_index > wqe_counter) ? 0x10000ULL : 0x0);

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
    if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif

    return ((opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) * -EIO);
}

/**
 * @brief Poll the collapsed Completion Queue (CQ) at a specific index.
 * This function does not wait for the completion to arrive.
 *
 * @param cq - Collapsed Completion Queue (CQ)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_one_cq_collapsed_at() returns 0. If the completion is
 * not available, returns EBUSY. If it is a completion with error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_one_cq_collapsed_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    uint64_t new_cqe_ci = 0;
    int status = doca_priv_gpu_dev_verbs_poll_cq_one_collapsed_at<resource_sharing_mode>(
        cq, cons_index, &new_cqe_ci);
    if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire_nvidia_nic();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, new_cqe_ci);
    }
    return status;
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index.
 * This function does not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_one_cq_at() returns 0. If the completion is
 * not available, returns EBUSY. If it is a completion with error, returns a
 * negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ,
          enum doca_gpu_dev_verbs_cq_type cq_type = DOCA_GPUNETIO_VERBS_CQ_UNKNOWN>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_one_cq_at(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t cons_index) {
    struct doca_gpu_dev_verbs_cq *cq = doca_gpu_dev_verbs_qp_get_cq<qp_type>(qp);
    enum doca_gpu_dev_verbs_cq_type mode = cq_type;
    if (cq_type == DOCA_GPUNETIO_VERBS_CQ_UNKNOWN) {
        mode =
            (enum doca_gpu_dev_verbs_cq_type)doca_gpu_dev_verbs_load_const((uint8_t *)&cq->cq_type);
    }

    if (mode == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST) {
        return doca_gpu_dev_verbs_poll_one_cq_host_at<resource_sharing_mode>(cq, cons_index);
    } else if (mode == DOCA_GPUNETIO_VERBS_CQ_64B) {
        return doca_gpu_dev_verbs_poll_one_cq_device_at<resource_sharing_mode>(cq, cons_index);
    } else if (mode == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED) {
        return doca_gpu_dev_verbs_poll_one_cq_collapsed_at<resource_sharing_mode>(cq, cons_index);
    }
    return EINVAL;
}

// ==================== Poll CQ At ====================

/**
 * @brief [Internal] Poll the Completion Queue (CQ) at a specific index.
 * This function does not update the SW consumer index nor guarantees the ordering.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_cq_device_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe =
        (struct doca_gpunetio_ib_mlx5_cqe64 *)doca_gpu_dev_verbs_load_const(
            (uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = doca_gpu_dev_verbs_load_const(&cq->cqe_num);
    const uint64_t cqe_rsvd = doca_gpu_dev_verbs_load_const(&cq->cqe_rsvd);
    uint64_t cons_index_in_cq = cons_index + cqe_rsvd;
    uint32_t idx = cons_index_in_cq & (cqe_num - 1);
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 = &cqe[idx];
    uint8_t opown;
    uint8_t opcode;
    uint64_t cqe_ci;
#if __CUDA_ARCH__ >= 900
    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci)
            return 0;
        opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
    } while ((cons_index >= cqe_ci + cqe_num) ||
             ((cqe_ci <= cons_index) &&
              ((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num))));
#else
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci)
            return 0;
        cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global(&cqe64->wqe_counter_sig_op_own_raw);
        cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
        wqe_counter = cqe_chunk >> 16;
        opown = cqe_chunk & 0xff;
    } while ((cons_index >= cqe_ci + cqe_num) ||
             ((cqe_ci <= cons_index) &&
              (((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num)) ||
               (wqe_counter != ((uint32_t)cons_index & 0xffff)))));
#endif

    opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
    if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif
    return (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) * -EIO;
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index. This function waits for the completion
 * to arrive. The polling algorithm is for non-collapsed CQ on GPU only.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_cq_device_at() returns 0. If it is a completion with
 * error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_device_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    int status = doca_priv_gpu_dev_verbs_poll_cq_device_at<resource_sharing_mode>(cq, cons_index);
    if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire_nvidia_nic();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
    }
    return status;
}

/**
 * @brief Poll a host-resident Completion Queue (CQ). CPU proxy progress is responsible
 * for inspecting CQEs and advancing cq->cqe_ci; this device path only spins on cqe_ci and
 * therefore touches none of the CQE buffer.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_cq_host_at() returns 0. If it is a completion with
 * error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_host_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    uint64_t cqe_ci;
    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed_sys_global((uint64_t *)&cq->cqe_ci);
    } while (cons_index >= cqe_ci);
    doca_gpu_dev_verbs_fence_acquire_nvidia_nic();
    return 0;
}

/**
 * @brief [Internal] Poll the Collapsed Completion Queue (CQ) at a specific index. This function
 * waits for the completion to arrive.
 *
 * @param cq - Collapsed Completion Queue (CQ)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @param new_cqe_ci - New CQ Consumer Index
 * @return On success, doca_priv_gpu_dev_verbs_poll_cq_collapsed_at() returns 0. If it is a
 * completion with error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_cq_collapsed_at(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t cons_index, uint64_t *new_cqe_ci) {
    struct doca_gpu_dev_verbs_cq *cq = doca_gpu_dev_verbs_qp_get_cq<qp_type>(qp);
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 =
        (struct doca_gpunetio_ib_mlx5_cqe64 *)doca_gpu_dev_verbs_load_const(
            (uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = doca_gpu_dev_verbs_load_const(&cq->cqe_num);
    uint8_t opown;
    uint8_t opcode;
    uint64_t cqe_ci;
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    // If idx is a lot greater than cons_idx, we might get incorrect result due
    // to wqe_counter wraparound. We need to check prod_idx to be sure that idx
    // has already been submitted.
    while (doca_gpu_dev_verbs_atomic_read<uint64_t, resource_sharing_mode>(&qp->sq_wqe_pi) <
           cons_index);
    doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();

    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci)
            return 0;
        cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global(&cqe64->wqe_counter_sig_op_own_raw);
        cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
        wqe_counter = cqe_chunk >> 16;
        opown = cqe_chunk & 0xff;
        opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
    }
    // NOTE: This while loop is part of do while above.
    // wqe_counter is the HW consumer index. However, we always maintain index
    // + 1 in SW. To be able to compare with idx, we need to use wqe_counter +
    // 1. Because wqe_counter is uint16_t, it may wraparound. Still we know for
    // sure that if idx - wqe_counter - 1 < ncqes, wqe_counter + 1 is less than
    // idx, and thus we need to wait. We don't need to wait when idx ==
    // wqe_counter + 1. That's why we use - (uint16_t)2 here to make this case
    // wraparound.
    while ((opcode == DOCA_GPUNETIO_IB_MLX5_CQE_INVALID) ||
           ((cqe_ci <= cons_index) &&
            ((uint16_t)((uint16_t)cons_index - wqe_counter - (uint16_t)2) < cqe_num)));

    ++wqe_counter;
    *new_cqe_ci = ((cons_index & ~(0xFFFFULL)) | wqe_counter) +
                  (((uint16_t)cons_index > wqe_counter) ? 0x10000ULL : 0x0);

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
    if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif

    return ((opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) * -EIO);
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index. This function waits for the completion
 * to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_cq_collapsed_at() returns 0. If it is a completion
 * with error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_collapsed_at(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t cons_index) {
    uint64_t new_cqe_ci = 0;
    int status = doca_priv_gpu_dev_verbs_poll_cq_collapsed_at<resource_sharing_mode, qp_type>(
        qp, cons_index, &new_cqe_ci);
    if (status == 0) {
        struct doca_gpu_dev_verbs_cq *cq = doca_gpu_dev_verbs_qp_get_cq<qp_type>(qp);
        doca_gpu_dev_verbs_fence_acquire_nvidia_nic();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&(cq->cqe_ci), new_cqe_ci);
    }
    return status;
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ,
          enum doca_gpu_dev_verbs_cq_type cq_type = DOCA_GPUNETIO_VERBS_CQ_UNKNOWN>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_at(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t cons_index) {
    struct doca_gpu_dev_verbs_cq *cq = doca_gpu_dev_verbs_qp_get_cq<qp_type>(qp);
    enum doca_gpu_dev_verbs_cq_type mode = cq_type;
    if (cq_type == DOCA_GPUNETIO_VERBS_CQ_UNKNOWN) {
        mode =
            (enum doca_gpu_dev_verbs_cq_type)doca_gpu_dev_verbs_load_const((uint8_t *)&cq->cq_type);
    }

    if (mode == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST) {
        return doca_gpu_dev_verbs_poll_cq_host_at<resource_sharing_mode>(cq, cons_index);
    } else if (mode == DOCA_GPUNETIO_VERBS_CQ_64B) {
        return doca_gpu_dev_verbs_poll_cq_device_at<resource_sharing_mode>(cq, cons_index);
    } else if (mode == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED) {
        return doca_gpu_dev_verbs_poll_cq_collapsed_at<resource_sharing_mode, qp_type>(qp,
                                                                                       cons_index);
    }
    return EINVAL;
}

/**
 * @brief Poll the Completion Queue (CQ). This function waits for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param count - Number of completions to poll
 * @return On success, doca_gpu_dev_verbs_poll_cq() returns 0. If it is a completion with
 * error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ,
          enum doca_gpu_dev_verbs_cq_type cq_type = DOCA_GPUNETIO_VERBS_CQ_UNKNOWN>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq(struct doca_gpu_dev_verbs_qp *qp,
                                                                 uint32_t count) {
    [[unlikely]] if (count == 0)
        return 0;
    struct doca_gpu_dev_verbs_cq *cq = doca_gpu_dev_verbs_qp_get_cq<qp_type>(qp);
    uint64_t cons_index =
        doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci) + count - 1;
    return doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type, cq_type>(qp, cons_index);
}

/**
 * @brief Increment CQ DBREC
 *
 * @param[in] cq - GPU Completion Queue
 * @param[in] cqe_num - CQE num to increment
 *
 * @return new CQE consumer index
 */
template <bool is_overrun>
__device__ static __forceinline__ uint32_t
doca_gpu_dev_verbs_cq_update_dbrec(struct doca_gpu_dev_verbs_cq *cq, uint32_t cqe_num) {
    uint32_t cqe_ci = DOCA_GPUNETIO_VOLATILE(cq->cqe_ci);

    cqe_ci = (cqe_ci + cqe_num) & DOCA_GPUNETIO_VERBS_CQE_CI_MASK;
    if (is_overrun == false) {
        asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;"
                     :
                     : "l"(cq->dbrec), "r"(doca_gpu_dev_verbs_bswap32(cqe_ci)));
    }

    DOCA_GPUNETIO_VOLATILE(cq->cqe_ci) = cqe_ci;

    return cqe_ci;
}

#endif /* DOCA_GPUNETIO_DEV_VERBS_CQ_H */

/** @} */
