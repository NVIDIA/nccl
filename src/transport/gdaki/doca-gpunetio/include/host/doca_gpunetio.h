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
 * @file doca_gpunetio.h
 * @brief A header file for the doca_gpunetio APIs
 */

#ifndef DOCA_GPUNETIO_H
#define DOCA_GPUNETIO_H

#include "host/doca_error.h"
#include "doca_gpunetio_config.h"
#include "common/doca_gpunetio_verbs_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**********************************************************************************************************************
 * DOCA GPU Lightweight opaque types
 *********************************************************************************************************************/
/**
 * Opaque structure representing a DOCA GPU device handler.
 */
struct doca_gpu;

/**
 * @brief Type of memory the GPUNetIO library can allocate
 *
 */
enum doca_gpu_mem_type {
    /* GPU memory not accessible from CPU. */
    DOCA_GPU_MEM_TYPE_GPU = 0,
    /* GPU memory with direct access from CPU. */
    DOCA_GPU_MEM_TYPE_GPU_CPU = 1,
    /* CPU memory with direct access from GPU. */
    DOCA_GPU_MEM_TYPE_CPU_GPU = 2,
};

/**
 * @brief Forward declaration
 *
 */
struct doca_gpu_dev_verbs_qp;
struct doca_gpu_dev_verbs_cq;

/**
 * @brief GPUNetIO QP handler accessible from CPU
 *
 */
struct doca_gpu_verbs_qp {
    struct doca_gpu *gpu_dev;
    struct doca_verbs_qp *qp;
    uint64_t *cpu_db;
    uint64_t sq_wqe_pi_last;
    uint64_t *sq_db;
    __be32 *sq_dbrec;
    bool cpu_proxy;
    uint32_t sq_num_shift8_be;
    /* CPU handler */
    struct doca_gpu_dev_verbs_qp *qp_cpu;
    /* GPU handler */
    struct doca_gpu_dev_verbs_qp *qp_gpu;
};

/**
 * @brief GPUNetIO QP Error info.
 */
struct doca_gpu_verbs_qp_error_info {
    bool has_error;
    int syndrome;
    int vendor_err_synd;
    int hw_err_synd;
    int hw_synd_type;
    int wqe_counter;
};

typedef void *doca_gpu_verbs_service_t;

/**
 * @brief Create a DOCA GPUNETIO handler.
 *
 * @param [in] gpu_bus_id
 * GPU PCIe address.
 * @param [out] gpu_dev
 * Pointer to the newly created gpu device handler.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - gpu_dev argument is a NULL pointer.
 * - DOCA_ERROR_NOT_FOUND - GPU not found at the input PCIe address
 * - DOCA_ERROR_NO_MEMORY - failed to alloc doca_gpu.
 *
 */
doca_error_t doca_gpu_create(const char *gpu_bus_id, struct doca_gpu **gpu_dev);

/**
 * @brief Destroy a DOCA GPUNETIO handler.
 *
 * @param [in] gpu_dev
 * Pointer to handler to be destroyed.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 */
doca_error_t doca_gpu_destroy(struct doca_gpu *gpu_dev);

/**
 * Allocate a GPU accessible memory buffer. Assumes DPDK has been already attached with
 * doca_gpu_to_dpdk(). According to the memory type specified, the buffer can be allocated in:
 * - DOCA_GPU_MEM_TYPE_GPU memptr_gpu is not NULL while memptr_cpu is NULL.
 * - DOCA_GPU_MEM_TYPE_GPU_CPU both memptr_gpu and memptr_cpu are not NULL.
 * - DOCA_GPU_MEM_TYPE_CPU_GPU both memptr_gpu and memptr_cpu are not NULL.
 *
 * @param [in] gpu_dev
 * DOCA GPUNetIO handler.
 * @param [in] size
 * Buffer size in bytes.
 * @param [in] alignment
 * Buffer memory alignment.
 * If 0, the return is a pointer that is suitably aligned
 * for any kind of variable (in the same manner as malloc()).
 * Otherwise, the return is a pointer that is a multiple of *align*.
 * Alignment value must be a power of two.
 * @param [in] mtype
 * Type of memory buffer. See enum doca_gpu_memtype for reference.
 * @param [out] memptr_gpu
 * GPU memory pointer. Must be used with CUDA API and within CUDA kernels.
 * @param [out] memptr_cpu
 * CPU memory pointer. Must be used for CPU direct access to the memory.
 *
 * @return
 * Non NULL memptr_gpu pointer on success, NULL otherwise.
 * Non NULL memptr_cpu pointer on success in case of DOCA_GPU_MEM_TYPE_CPU_GPU and
 * DOCA_GPU_MEM_TYPE_GPU_CPU, NULL otherwise. DOCA_SUCCESS - in case of success. doca_error code -
 * in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_NO_MEMORY - if an error occurred dealing with GPU memory.
 */
doca_error_t doca_gpu_mem_alloc(struct doca_gpu *gpu_dev, size_t size, size_t alignment,
                                enum doca_gpu_mem_type mtype, void **memptr_gpu, void **memptr_cpu);

/**
 * Free a GPU memory buffer.
 * Only memory allocated with doca_gpu_mem_alloc() can be freed with this function.
 *
 * @param [in] gpu
 * DOCA GPUNetIO handler.
 * @param [in] memptr_gpu
 * GPU memory pointer to be freed.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_mem_free(struct doca_gpu *gpu, void *memptr_gpu);

/**
 * Create a GPU handler for a Verbs QP object
 *
 * @param [in] gpu_dev
 * DOCA GPUNetIO handler.
 * @param [in] qp
 * DOCA network device handler.
 * @param [in] nic_handler
 * Type of NIC handler for this QP.
 * @param [in] gpu_qp_umem_dev_ptr
 * GPU external UMEM.
 * @param [in] cq_sq
 * DOCA Verbs CQ SQ CPU object connected to the QP.
 * @param [out] qp_out
 * DOCA GPUNetIO Verbs QP object.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_export_qp(struct doca_gpu *gpu_dev, struct doca_verbs_qp *qp,
                                      enum doca_gpu_dev_verbs_nic_handler nic_handler,
                                      void *gpu_qp_umem_dev_ptr, struct doca_verbs_cq *cq_sq,
                                      struct doca_gpu_verbs_qp **qp_out);

/**
 * Destroy a GPU handler for a Verbs QP object
 *
 * @param [in] gpu_dev
 * DOCA GPUNetIO handler.
 * @param [in] qp_cpu
 * DOCA Verbs QP CPU object.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_unexport_qp(struct doca_gpu *gpu_dev, struct doca_gpu_verbs_qp *qp);

/**
 * Get a GPUNetIO GPU device handler handler from a GPUNetIO Verbs QP object.
 *
 * @param [in] qp
 * DOCA GPUNetIO Verbs QP object.
 * @param [out] qp_gpu
 * DOCA GPUNetIO Verbs QP GPU object.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_get_qp_dev(struct doca_gpu_verbs_qp *qp,
                                       struct doca_gpu_dev_verbs_qp **qp_gpu);

/**
 * Return a DMABuf file descriptor from a GPU memory address if the GPU device and CUDA installation
 * supports DMABuf.
 *
 * @param [in] gpu_dev
 * DOCA GPUNetIO handler.
 * @param [in] memptr_gpu
 * GPU memory pointer to be freed.
 * @param [in] size
 * Size in bytes to map.
 * @param [out] dmabuf_fd
 * DMABuf file descriptor
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_NOT_SUPPORTED - DMABuf not supported
 */
doca_error_t doca_gpu_dmabuf_fd(struct doca_gpu *gpu_dev, void *memptr_gpu, size_t size,
                                int *dmabuf_fd);

/**
 * Check if UAR can be registered on GPU
 *
 * @param [in] db
 * UAR address
 * @param [out] out_can_register
 * Can register on GPU
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_can_gpu_register_uar(void *db, bool *out_can_register);

/**
 * Export UAR to GPU
 *
 * @param [in] sq_db
 * SQ UAR address
 * @param [out] uar_addr_gpu
 * SQ UAR GPU address
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_DRIVER - if UAR mapping failed
 */
doca_error_t doca_gpu_verbs_export_uar(uint64_t *sq_db, uint64_t **uar_addr_gpu);

/**
 * Unexport UAR from GPU
 *
 * @param [in] uar_addr_gpu
 * SQ UAR GPU address
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_unexport_uar(uint64_t *uar_addr_gpu);

/**
 * Progress QP (ring db) in case of CPU proxy mode
 *
 * @param [in] qp_cpu
 * QP to progress
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_cpu_proxy_progress(struct doca_gpu_verbs_qp *qp_cpu);

/**
 * Create a service object.
 *
 * @param [out] out_service
 * Service handle
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_create_service(doca_gpu_verbs_service_t *out_service);

/**
 * Monitor a QP and make forward progress.
 *
 * @param [in] service
 * Service object
 * @param [in] qp
 * QP to monitor
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_service_monitor_qp(doca_gpu_verbs_service_t service,
                                               struct doca_gpu_verbs_qp *qp);

/**
 * Destroy a service object.
 *
 * @param [in] service
 * Service object to destroy
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_destroy_service(doca_gpu_verbs_service_t service);

/**
 * Query the last error of a GPUNetIO QP
 *
 * @param [in] qp
 * QP to query
 * @param [out] error_info
 * Error info
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_query_last_error(struct doca_gpu_verbs_qp *qp,
                                             struct doca_gpu_verbs_qp_error_info *error_info);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_GPUNETIO_H */
