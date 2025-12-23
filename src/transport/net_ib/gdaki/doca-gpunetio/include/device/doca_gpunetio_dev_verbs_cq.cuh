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

/**
 * @brief [Internal] Poll the Completion Queue (CQ) at a specific index.
 * This function does not update the SW consumer index nor guarantees the ordering.
 * It also does not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_one_cq_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    uint8_t *cqe = (uint8_t *)__ldg((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    uint32_t idx = cons_index & (cqe_num - 1);
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
        !((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num));
#else
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&cqe64->wqe_counter);
    cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
    wqe_counter = cqe_chunk >> 16;
    opown = cqe_chunk & 0xff;

    observed_completion =
        !((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num)) &&
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
 * @brief Poll the Completion Queue (CQ) at a specific index. This function does
 * not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_one_cq_at() returns 0. If the completion is
 * not available, returns EBUSY. If it is a completion with error, returns a
 * negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_one_cq_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    int status =
        doca_priv_gpu_dev_verbs_poll_one_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
    if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
    }
    return status;
}

/**
 * @brief [Internal] Poll the Completion Queue (CQ) at a specific index.
 * This function does not update the SW consumer index nor guarantees the ordering.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_cq_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe =
        (struct doca_gpunetio_ib_mlx5_cqe64 *)__ldg((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    uint32_t idx = cons_index & (cqe_num - 1);
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
              ((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num))));
#else
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci)
            return 0;
        cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&cqe64->wqe_counter);
        cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
        wqe_counter = cqe_chunk >> 16;
        opown = cqe_chunk & 0xff;
    } while ((cons_index >= cqe_ci + cqe_num) ||
             ((cqe_ci <= cons_index) &&
              (((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num)) ||
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
 * to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_cq_at() returns 0. If it is a completion with
 * error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    int status = doca_priv_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
    if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
    }
    return status;
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
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq(struct doca_gpu_dev_verbs_cq *cq,
                                                                 uint32_t count) {
    uint64_t cons_index =
        doca_gpu_dev_verbs_atomic_add<uint64_t, resource_sharing_mode>(&cq->cqe_rsvd, count) +
        count - 1;
    return doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
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
