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

#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <endian.h>
#include <cuda_runtime.h>
#include <string.h>

#include <atomic>
#include <set>
#include <unordered_map>
#include <mutex>

#include "common/doca_gpunetio_verbs_def.h"
#include "host/mlx5_prm.h"
#include "host/mlx5_ifc.h"

#include "doca_verbs_net_wrapper.h"
#include "doca_internal.hpp"
#include "host/doca_gpunetio.h"
#include "doca_gpunetio_gdrcopy.h"
#include "common/doca_gpunetio_verbs_dev.h"
#include "host/doca_verbs.h"
#include "doca_verbs_qp.hpp"
#include "doca_gpunetio_cuda_wrapper.h"
#include "doca_gpunetio_sdk_wrapper.h"
#include "doca_gpunetio.hpp"

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_FULL_ASYNC_STORE_RELEASE_SUPPORT_COMPUTE_CAP_MAJOR 10

#define DOCA_GPUNETIO_FREE_FLOW_RING_DB_THRESHOLD_DEFAULT 64

static doca_error_t normalize_export_cq_type(enum doca_gpu_dev_verbs_cq_type *cq_type) {
    switch (*cq_type) {
        case DOCA_GPUNETIO_VERBS_CQ_UNKNOWN:
            *cq_type = DOCA_GPUNETIO_VERBS_CQ_64B;
            return DOCA_SUCCESS;
        case DOCA_GPUNETIO_VERBS_CQ_64B:
        case DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED:
        case DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST:
            return DOCA_SUCCESS;
        default:
            DOCA_LOG(LOG_ERR, "Invalid CQ type %d", *cq_type);
            return DOCA_ERROR_INVALID_VALUE;
    }
}

struct doca_gpu_mtable {
    uintptr_t base_addr;
    size_t size_orig;
    uintptr_t align_addr_gpu;
    uintptr_t align_addr_cpu;
    size_t size;
    enum doca_gpu_mem_type mtype;
    void *gdr_mh;
};

struct doca_gpu_verbs_service {
    pthread_t service_thread;
    pthread_rwlock_t service_lock;
    bool running;
    std::set<struct doca_gpu_verbs_qp *> *qps;
};

static inline bool priv_query_async_store_release_support(void) {
    int current_device;
    int compute_cap_major;
    cudaError_t status = cudaSuccess;

    status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaGetDevice(&current_device));
    if (status != cudaSuccess) return false;

    status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaDeviceGetAttribute(
        &compute_cap_major, cudaDevAttrComputeCapabilityMajor, current_device));
    if (status != cudaSuccess) return false;

    return (compute_cap_major >= GPU_FULL_ASYNC_STORE_RELEASE_SUPPORT_COMPUTE_CAP_MAJOR);
    return (compute_cap_major >= GPU_FULL_ASYNC_STORE_RELEASE_SUPPORT_COMPUTE_CAP_MAJOR);
}

bool priv_is_power_of_two(uint64_t x) { return x && (x & (x - 1)) == 0; }

static size_t priv_get_page_size() {
    auto ret = sysconf(_SC_PAGESIZE);
    if (ret == -1) return 4096;  // 4KB, default Linux page size

    return (size_t)ret;
}

doca_error_t doca_gpu_create(const char *gpu_bus_id, doca_gpu_t **gpu_dev) {
    doca_gpu_t *gpu_dev_ = nullptr;
    int dmabuf_supported = 0, order = 0;
    CUresult res_drv = CUDA_SUCCESS;
    cudaError_t res_cuda = cudaSuccess;
    doca_sdk_wrapper_error_t err;

    gpu_dev_ = (doca_gpu_t *)calloc(1, sizeof(doca_gpu_t));
    if (gpu_dev_ == nullptr) {
        DOCA_LOG(LOG_ERR, "error in %s: failed to allocate memory for doca_gpu_t", __func__);
        return DOCA_ERROR_NO_MEMORY;
    }

    /* Try with DOCA SDK first */
    err = doca_gpu_sdk_wrapper_create(gpu_bus_id, &(gpu_dev_->sdk));
    if (err == DOCA_SDK_WRAPPER_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Use DOCA GPUNetIO SDK", __func__);
        gpu_dev_->type = DOCA_GPU_LIB_TYPE_SDK;
        (*gpu_dev) = gpu_dev_;
        return DOCA_SUCCESS;
    } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
        DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
        goto exit_error;
    }

    /* In case of DOCA_SDK_WRAPPER_NOT_FOUND or DOCA_SDK_WRAPPER_NOT_SUPPORTED, just rely on open
     * version */
    DOCA_LOG(LOG_INFO, "Use DOCA GPUNetIO open", __func__);

    if (gpu_bus_id == nullptr || gpu_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        goto exit_error;
    }

    gpu_dev_->type = DOCA_GPU_LIB_TYPE_OPEN;
    gpu_dev_->open = (struct doca_gpu_open *)calloc(1, sizeof(struct doca_gpu_open));
    if (gpu_dev_->open == nullptr) {
        DOCA_LOG(LOG_ERR, "error in %s: failed to allocate memory for doca_gpu_open", __func__);
        goto exit_error;
    }

    res_cuda = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaDeviceGetByPCIBusId(&gpu_dev_->open->cuda_dev, gpu_bus_id));
    if (res_cuda != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Invalid GPU bus id provided (ret %d).", res_cuda);
        goto exit_error;
    }

    res_drv = doca_gpu_cuda_wrapper_cuDeviceGetAttribute(
        &(dmabuf_supported), CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, gpu_dev_->open->cuda_dev);
    if (res_drv != CUDA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "cuDeviceGetAttribute CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED returned %d.",
                 res_drv);
        goto exit_error;
    }

    (dmabuf_supported == 1 ? (gpu_dev_->open->support_dmabuf = true)
                           : (gpu_dev_->open->support_dmabuf = false));

    res_drv = doca_gpu_cuda_wrapper_cuDeviceGetAttribute(
        &order, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, gpu_dev_->open->cuda_dev);
    if (res_drv != CUDA_SUCCESS) {
        DOCA_LOG(
            LOG_ERR,
            "cuDeviceGetAttribute CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING returned %d.",
            res_drv);
        goto exit_error;
    }

    gpu_dev_->open->need_mcst = true;
    if (order >= CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER) gpu_dev_->open->need_mcst = false;

    gpu_dev_->open->support_wq_gpumem = true;
    gpu_dev_->open->support_cq_gpumem = true;
    gpu_dev_->open->support_uar_gpumem = true;
    gpu_dev_->open->support_bf_uar = true;
    gpu_dev_->open->support_async_store_release = priv_query_async_store_release_support();
    gpu_dev_->open->support_gdrcopy = doca_gpu_gdrcopy_is_supported();
    gpu_dev_->open->support_gdrcopy_data_direct = doca_gpu_gdrcopy_supports_force_pcie();

    try {
        gpu_dev_->open->mtable = new std::unordered_map<uintptr_t, struct doca_gpu_mtable *>();
    } catch (...) {
        DOCA_LOG(LOG_ERR, "mtable map allocation failed");
        goto exit_error;
    }

    (*gpu_dev) = gpu_dev_;

    return DOCA_SUCCESS;

exit_error:
    if (gpu_dev_ != nullptr) {
        if (gpu_dev_->open) free(gpu_dev_->open);
        free(gpu_dev_);
    }

    return DOCA_ERROR_INITIALIZATION;
}

doca_error_t doca_gpu_destroy(doca_gpu_t *gpu_dev) {
    doca_error_t status = DOCA_SUCCESS;
    doca_sdk_wrapper_error_t err;

    if (gpu_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (gpu_dev->type == DOCA_GPU_LIB_TYPE_SDK) {
        err = doca_gpu_sdk_wrapper_destroy(gpu_dev->sdk);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            gpu_dev->sdk = nullptr;
            goto exit;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            status = DOCA_ERROR_UNEXPECTED;
            goto exit;
        }
    }

    if (gpu_dev->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        status = DOCA_ERROR_INVALID_VALUE;
        goto exit;
    }

    if (gpu_dev->open->mtable != nullptr) {
        if (gpu_dev->open->mtable->size() > 0) {
            DOCA_LOG(LOG_ERR, "mtable map is not empty.");
            status = DOCA_ERROR_INVALID_VALUE;
            goto exit;
        }
        delete gpu_dev->open->mtable;
    }

exit:
    if (gpu_dev->open) free(gpu_dev->open);
    memset(gpu_dev, 0, sizeof(doca_gpu_t));
    free(gpu_dev);

    return status;
}

doca_error_t doca_gpu_mem_alloc(doca_gpu_t *gpu_dev, size_t size, size_t alignment,
                                enum doca_gpu_mem_type mtype, void **memptr_gpu,
                                void **memptr_cpu) {
    cudaError_t res;
    CUresult res_drv;
    int ret;
    void *cudev_memptr_gpu_orig_ = 0;
    void *cudev_memptr_gpu_ = 0;
    struct doca_gpu_mtable *mentry;
    unsigned int flag = 1;
    const char *err_string;
    void *memptr_cpu_ = nullptr;
    void *memptr_cpu_orig_ = 0;
    doca_error_t status = DOCA_SUCCESS;
    doca_sdk_wrapper_error_t err;
    bool is_gpu_cpu_mtype;
    bool is_data_direct_mtype;

    if (gpu_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (gpu_dev->type == DOCA_GPU_LIB_TYPE_SDK) {
        err = doca_gpu_sdk_wrapper_mem_alloc(gpu_dev->sdk, size, alignment, mtype, memptr_gpu,
                                             memptr_cpu);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    is_gpu_cpu_mtype =
        (mtype == DOCA_GPU_MEM_TYPE_GPU_CPU || mtype == DOCA_GPU_MEM_TYPE_GPU_CPU_DATA_DIRECT);
    is_data_direct_mtype = (mtype == DOCA_GPU_MEM_TYPE_GPU_CPU_DATA_DIRECT);

    if (gpu_dev->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA GPUNetIO instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (memptr_gpu == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid memptr_gpu provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (mtype != DOCA_GPU_MEM_TYPE_GPU && !is_gpu_cpu_mtype && mtype != DOCA_GPU_MEM_TYPE_CPU_GPU) {
        DOCA_LOG(LOG_ERR, "Invalid memory type provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (mtype != DOCA_GPU_MEM_TYPE_GPU && memptr_cpu == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid memptr_cpu provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (is_data_direct_mtype) {
        if (!gpu_dev->open->support_gdrcopy_data_direct) {
            DOCA_LOG(LOG_ERR,
                     "DOCA_GPU_MEM_TYPE_GPU_CPU_DATA_DIRECT memory type is not supported on this "
                     "GPU.");
            return DOCA_ERROR_NOT_SUPPORTED;
        }
    }

    if (size == 0) {
        DOCA_LOG(LOG_ERR, "Invalid size provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (alignment == 0) alignment = priv_get_page_size();

    if (priv_is_power_of_two(alignment) == false) {
        DOCA_LOG(LOG_ERR, "alignment %zd has to be power of 2.", alignment);
        return DOCA_ERROR_INVALID_VALUE;
    }

    mentry = (struct doca_gpu_mtable *)calloc(1, sizeof(struct doca_gpu_mtable));
    mentry->mtype = mtype;
    mentry->size = size;

    if (is_gpu_cpu_mtype && alignment != GPU_PAGE_SIZE) alignment = GPU_PAGE_SIZE;

    if (mtype == DOCA_GPU_MEM_TYPE_GPU) {
        mentry->size_orig = mentry->size + alignment;

        res = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
            cudaMalloc(&(cudev_memptr_gpu_orig_), mentry->size_orig));
        if (res != cudaSuccess) {
            err_string = cudaGetErrorString(res);
            DOCA_LOG(LOG_ERR, "cudaMalloc current failed with %s size %zd", err_string,
                     mentry->size_orig);
            status = DOCA_ERROR_DRIVER;
            goto error;
        }

        /* Align memory address */
        cudev_memptr_gpu_ = cudev_memptr_gpu_orig_;
        if (alignment && ((uintptr_t)cudev_memptr_gpu_) % alignment)
            cudev_memptr_gpu_ =
                (void *)((uintptr_t)cudev_memptr_gpu_ +
                         (alignment - (((uintptr_t)cudev_memptr_gpu_) % alignment)));

        /* GPUDirect RDMA attribute required */
        res_drv = doca_gpu_cuda_wrapper_cuPointerSetAttribute(
            &flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)cudev_memptr_gpu_);
        if (res_drv != CUDA_SUCCESS) {
            DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaFree(cudev_memptr_gpu_orig_));
            DOCA_LOG(LOG_ERR, "Could not set SYNC MEMOP attribute for GPU memory at %lx, err %d",
                     (uintptr_t)cudev_memptr_gpu_, res);
            status = DOCA_ERROR_DRIVER;
            goto error;
        }

        mentry->base_addr = (uintptr_t)cudev_memptr_gpu_orig_;
        mentry->align_addr_gpu = (uintptr_t)cudev_memptr_gpu_;
        mentry->align_addr_cpu = 0;
    } else if (is_gpu_cpu_mtype) {
        if (gpu_dev->open->support_gdrcopy == true) {
            bool force_pcie = is_data_direct_mtype;

            mentry->size_orig = mentry->size + alignment;

            res = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
                cudaMalloc(&(cudev_memptr_gpu_orig_), mentry->size_orig));
            if (res != cudaSuccess) {
                err_string = cudaGetErrorString(res);
                DOCA_LOG(LOG_ERR, "cudaMalloc current failed with %s", err_string);
                status = DOCA_ERROR_DRIVER;
                goto error;
            }

            /* Align memory address */
            cudev_memptr_gpu_ = cudev_memptr_gpu_orig_;
            if (alignment && ((uintptr_t)cudev_memptr_gpu_) % alignment)
                cudev_memptr_gpu_ =
                    (void *)((uintptr_t)cudev_memptr_gpu_ +
                             (alignment - (((uintptr_t)cudev_memptr_gpu_) % alignment)));

            /* GPUDirect RDMA attribute required */
            res_drv = doca_gpu_cuda_wrapper_cuPointerSetAttribute(
                &flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)cudev_memptr_gpu_);
            if (res_drv != CUDA_SUCCESS) {
                DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaFree(cudev_memptr_gpu_orig_));
                DOCA_LOG(LOG_ERR,
                         "Could not set SYNC MEMOP attribute for GPU memory at %lx, err %d",
                         (uintptr_t)cudev_memptr_gpu_, res);
                status = DOCA_ERROR_DRIVER;
                goto error;
            }

            mentry->base_addr = (uintptr_t)cudev_memptr_gpu_orig_;
            mentry->align_addr_gpu = (uintptr_t)cudev_memptr_gpu_;
            mentry->align_addr_cpu = 0;

            ret = doca_gpu_gdrcopy_create_mapping((void *)mentry->align_addr_gpu, mentry->size,
                                                  force_pcie, &mentry->gdr_mh,
                                                  (void **)&mentry->align_addr_cpu);
            if (ret) {
                DOCA_LOG(LOG_ERR, "Error mapping GPU memory at %lx to CPU", mentry->align_addr_gpu);
                status = DOCA_ERROR_DRIVER;
                goto error;
            }
        } else {
            DOCA_LOG(LOG_WARNING,
                     "GDRCopy not enabled, can't allocate GPU-CPU memory type. Using "
                     "DOCA_GPU_MEM_TYPE_CPU_GPU mode instead");

            mentry->size_orig = mentry->size;

            memptr_cpu_ = (uint8_t *)calloc(alignment, mentry->size_orig);
            if (memptr_cpu_ == nullptr) {
                DOCA_LOG(LOG_ERR, "Failed to allocate CPU memory.");
                status = DOCA_ERROR_DRIVER;
                goto error;
            }

            res = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostRegister(
                memptr_cpu_, mentry->size_orig, cudaHostRegisterPortable | cudaHostRegisterMapped));
            if (res != cudaSuccess) {
                DOCA_LOG(LOG_ERR, "Could register CPU memory to CUDA %lx, err %d",
                         (uintptr_t)memptr_cpu_, res);
                free(memptr_cpu_);
                status = DOCA_ERROR_DRIVER;
                goto error;
            }

            mentry->base_addr = (uintptr_t)memptr_cpu_;

            res = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
                cudaHostGetDevicePointer(&cudev_memptr_gpu_, memptr_cpu_, 0));
            if (res != cudaSuccess) {
                DOCA_LOG(LOG_ERR, "Could get GPU device ptr for CPU memory %lx, err %d",
                         (uintptr_t)memptr_cpu_, res);
                free(memptr_cpu_);
                status = DOCA_ERROR_DRIVER;
                goto error;
            }

            mentry->align_addr_gpu = (uintptr_t)cudev_memptr_gpu_;
            mentry->align_addr_cpu = (uintptr_t)memptr_cpu_;
        }

    } else if (mtype == DOCA_GPU_MEM_TYPE_CPU_GPU) {
        mentry->size_orig = mentry->size + alignment;

        memptr_cpu_orig_ = (uint8_t *)calloc(mentry->size_orig, 1);
        if (memptr_cpu_orig_ == nullptr) {
            DOCA_LOG(LOG_ERR, "Failed to allocate CPU memory.");
            status = DOCA_ERROR_DRIVER;
            goto error;
        }

        /* Align memory address */
        memptr_cpu_ = memptr_cpu_orig_;
        if (alignment && ((uintptr_t)memptr_cpu_) % alignment)
            memptr_cpu_ = (void *)((uintptr_t)memptr_cpu_ +
                                   (alignment - (((uintptr_t)memptr_cpu_) % alignment)));

        res = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostRegister(
            memptr_cpu_, mentry->size, cudaHostRegisterPortable | cudaHostRegisterMapped));
        if (res != cudaSuccess) {
            DOCA_LOG(LOG_ERR, "Could register CPU memory to CUDA %lx, err %d",
                     (uintptr_t)memptr_cpu_, res);
            free(memptr_cpu_orig_);
            status = DOCA_ERROR_DRIVER;
            goto error;
        }

        mentry->base_addr = (uintptr_t)memptr_cpu_orig_;

        res = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
            cudaHostGetDevicePointer(&cudev_memptr_gpu_, memptr_cpu_, 0));
        if (res != cudaSuccess) {
            DOCA_LOG(LOG_ERR, "Could get GPU device ptr for CPU memory %lx, err %d",
                     (uintptr_t)memptr_cpu_, res);
            cudaHostUnregister(memptr_cpu_);
            free(memptr_cpu_orig_);
            status = DOCA_ERROR_DRIVER;
            goto error;
        }

        mentry->align_addr_gpu = (uintptr_t)cudev_memptr_gpu_;
        mentry->align_addr_cpu = (uintptr_t)memptr_cpu_;
    }

    *memptr_gpu = (void *)mentry->align_addr_gpu;
    if (memptr_cpu) *memptr_cpu = (void *)mentry->align_addr_cpu;

    // DOCA_LOG(LOG_DEBUG, "New memory: Orig %lx GPU %lx CPU %lx type %d size %zd\n",
    // 	      mentry->base_addr,
    // 	      mentry->align_addr_gpu,
    // 	      mentry->align_addr_cpu,
    // 	      mentry->mtype,
    // 	      mentry->size);

    try {
        gpu_dev->open->mtable->insert({mentry->align_addr_gpu, mentry});
    } catch (...) {
        DOCA_LOG(LOG_ERR, "mtable map insert failed");
        status = DOCA_ERROR_DRIVER;
        goto error;
    }

    return DOCA_SUCCESS;

error:
    free(mentry);
    return status;
}

doca_error_t doca_gpu_mem_free(doca_gpu_t *gpu_dev, void *memptr_gpu) {
    struct doca_gpu_mtable *mentry;
    cudaError_t res_cuda;
    doca_sdk_wrapper_error_t err;

    if (gpu_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (gpu_dev->type == DOCA_GPU_LIB_TYPE_SDK) {
        err = doca_gpu_sdk_wrapper_mem_free(gpu_dev->sdk, memptr_gpu);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (gpu_dev->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA GPUNetIO instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (memptr_gpu == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid memptr_gpu provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    std::unordered_map<uint64_t, struct doca_gpu_mtable *>::const_iterator it =
        gpu_dev->open->mtable->find((uintptr_t)memptr_gpu);
    if (it == gpu_dev->open->mtable->end()) {
        DOCA_LOG(LOG_ERR, "memptr_gpu = %p was not allocated by DOCA GPUNetIO.", memptr_gpu);
        return DOCA_ERROR_INVALID_VALUE;
    }

    mentry = it->second;

    if (mentry->mtype == DOCA_GPU_MEM_TYPE_GPU)
        DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaFree((void *)mentry->base_addr));
    else if (mentry->mtype == DOCA_GPU_MEM_TYPE_GPU_CPU ||
             mentry->mtype == DOCA_GPU_MEM_TYPE_GPU_CPU_DATA_DIRECT) {
        if (gpu_dev->open->support_gdrcopy)
            doca_gpu_gdrcopy_destroy_mapping(mentry->gdr_mh, (void *)mentry->align_addr_cpu,
                                             mentry->size);
        DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaFree((void *)mentry->base_addr));
    } else {
        res_cuda =
            DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostUnregister((void *)mentry->align_addr_cpu));
        if (res_cuda != cudaSuccess)
            DOCA_LOG(LOG_ERR, "Error unregistering GPU memory at %p",
                     (void *)mentry->align_addr_cpu);
        free((void *)mentry->base_addr);
    }

    gpu_dev->open->mtable->erase(it);
    free(mentry);

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_get_dmabuf_fd(doca_gpu_t *gpu_dev, void *memptr_gpu, size_t size,
                                    int *dmabuf_fd) {
#if DOCA_GPUNETIO_HAVE_CUDA_DMABUF == 1
    doca_sdk_wrapper_error_t err;
    CUresult res_drv = CUDA_SUCCESS;

    if (gpu_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (gpu_dev->type == DOCA_GPU_LIB_TYPE_SDK) {
        err = doca_gpu_sdk_wrapper_dmabuf_fd(gpu_dev->sdk, memptr_gpu, size, dmabuf_fd);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (gpu_dev->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA GPUNetIO instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (gpu_dev->open->support_dmabuf == false) {
        DOCA_LOG(LOG_ERR, "DMABuf not supported on this system by this CUDA installation.");
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    if (dmabuf_fd == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DMABuf fd pointer provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    res_drv = doca_gpu_cuda_wrapper_cuMemGetHandleForAddressRange(
        dmabuf_fd, (CUdeviceptr)memptr_gpu, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (res_drv != CUDA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "cuMemGetHandleForAddressRange returned %d.", res_drv);
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    return DOCA_SUCCESS;
#else
    return DOCA_ERROR_NOT_SUPPORTED;
#endif
}

static std::mutex registered_uar_mutex;

doca_error_t doca_gpu_verbs_can_gpu_register_uar(void *db, bool *out_can_register) {
    std::lock_guard<std::mutex> lock(registered_uar_mutex);
    cudaError_t cuda_status = cudaSuccess;
    static bool can_register = false;
    static bool registration_checked = false;

    if (out_can_register == nullptr) return DOCA_ERROR_INVALID_VALUE;

    if (!registration_checked) {
        if (db == nullptr) return DOCA_ERROR_INVALID_VALUE;
        cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostRegister(
            db, DOCA_VERBS_DB_UAR_SIZE,
            cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory));

        can_register =
            (cuda_status == cudaSuccess || cuda_status == cudaErrorHostMemoryAlreadyRegistered);

        if (cuda_status == cudaSuccess) DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostUnregister(db));
        registration_checked = true;
    }

    *out_can_register = can_register;

    return DOCA_SUCCESS;
}

static std::unordered_map<void *, unsigned int> registered_uar_refcount;

doca_error_t doca_gpu_verbs_export_uar(uint64_t *sq_db, uint64_t **uar_addr_gpu) {
    std::lock_guard<std::mutex> lock(registered_uar_mutex);

    void *ptr = nullptr;
    cudaError_t cuda_status = cudaSuccess;
    bool registered = false;
    void *uar_key;

    if (sq_db == nullptr || uar_addr_gpu == nullptr) return DOCA_ERROR_INVALID_VALUE;

    uar_key = (void *)sq_db;
    if (registered_uar_refcount.find(uar_key) == registered_uar_refcount.end()) {
        registered_uar_refcount[uar_key] = 0;
    }

    if (registered_uar_refcount[uar_key] == 0) {
        cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostRegister(
            sq_db, DOCA_VERBS_DB_UAR_SIZE,
            cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory));
        if (cuda_status != cudaSuccess) {
            DOCA_LOG(LOG_ERR,
                     "Function cudaHostRegister (err %d) "
                     "failed on addr %p size %d",
                     cuda_status, (void *)sq_db, DOCA_VERBS_DB_UAR_SIZE);
            goto out;
        }
        registered = true;
    }

    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostGetDevicePointer(&ptr, sq_db, 0));
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR,
                 "Function cudaHostGetDevicePointer (err %d) "
                 "failed on addr %p size %d",
                 cuda_status, (void *)sq_db, DOCA_VERBS_DB_UAR_SIZE);
        goto out;
    }

    registered_uar_refcount[uar_key]++;

    *uar_addr_gpu = (uint64_t *)ptr;

out:
    if (cuda_status != cudaSuccess) {
        if (registered) DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostUnregister(sq_db));
        return DOCA_ERROR_DRIVER;
    }

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_unexport_uar(uint64_t *uar_addr_gpu) {
    std::lock_guard<std::mutex> lock(registered_uar_mutex);

    cudaError_t cuda_status = cudaSuccess;
    void *uar_key;

    if (uar_addr_gpu == nullptr) return DOCA_ERROR_INVALID_VALUE;

    uar_key = (void *)uar_addr_gpu;
    if (registered_uar_refcount.find(uar_key) == registered_uar_refcount.end()) {
        DOCA_LOG(LOG_ERR, "UAR address %p not found in registered_uar_refcount", uar_addr_gpu);
        return DOCA_ERROR_INVALID_VALUE;
    }
    registered_uar_refcount[uar_key]--;
    assert(registered_uar_refcount[uar_key] >= 0);
    if (registered_uar_refcount[uar_key] == 0) {
        registered_uar_refcount.erase(uar_key);
        cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostUnregister(uar_addr_gpu));
        if (cuda_status != cudaSuccess) {
            DOCA_LOG(LOG_ERR, "Failed to unregister UAR address %p", uar_addr_gpu);
            return DOCA_ERROR_DRIVER;
        }
    }

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_export_qp(doca_gpu_t *gpu_dev, doca_verbs_qp_t *qp,
                                      enum doca_gpu_dev_verbs_nic_handler nic_handler,
                                      void *gpu_qp_umem_dev_ptr, doca_verbs_cq_t *cq_sq,
                                      enum doca_gpu_verbs_send_dbr_mode_ext send_dbr_mode_ext,
                                      enum doca_gpu_dev_verbs_cq_type cq_type,
                                      bool enable_data_direct, struct doca_gpu_verbs_qp **qp_out) {
    doca_error_t status = DOCA_SUCCESS;
    struct doca_gpu_dev_verbs_qp *qp_cpu_ = nullptr;
    struct doca_gpu_verbs_qp *qp_gverbs = nullptr;
    void *rq_wqe_daddr;
    uint32_t rq_wqe_num;
    uint32_t rcv_wqe_size;
    uint64_t *sq_db;
    uint32_t sq_wqe_num;
    uint64_t *uar_db_reg = NULL;
    uint32_t *arm_dbr = NULL;
    uint32_t *cq_dbrec;
    uint32_t *dbrec;
    doca_verbs_qp_init_attr_t *qp_init_attr_out = nullptr;
    doca_verbs_qp_attr_t *qp_attr_out = nullptr;
    enum doca_verbs_qp_send_dbr_mode send_dbr_mode;
    bool nic_handler_must_be_cpu_proxy = false;

    // Will introduce SDK wrapper once done with DOCA Verbs
    if (gpu_dev->open == nullptr || qp == nullptr || qp == nullptr || cq_sq == nullptr)
        return DOCA_ERROR_INVALID_VALUE;

    status = normalize_export_cq_type(&cq_type);
    if (status != DOCA_SUCCESS) return status;

    if ((nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR) &&
        (send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR)) {
        DOCA_LOG(LOG_ERR,
                 "DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR cannot be used with "
                 "DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR");
        status = DOCA_ERROR_INVALID_VALUE;
        goto out;
    }

    if ((send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) &&
        !gpu_dev->open->support_gdrcopy) {
        DOCA_LOG(LOG_ERR, "SW-emulated no DBR feature is not supported without GDRCopy");
        status = DOCA_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if (cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST) {
        if (!gpu_dev->open->support_gdrcopy) {
            DOCA_LOG(LOG_ERR, "Host-collapsed CQ type is not supported without GDRCopy");
            status = DOCA_ERROR_NOT_SUPPORTED;
            goto out;
        }
        if (enable_data_direct && !gpu_dev->open->support_gdrcopy_data_direct) {
            DOCA_LOG(LOG_ERR,
                     "GDRCopy does not support data-direct, cannot enable data-direct with "
                     "host-collapsed CQ type");
            status = DOCA_ERROR_NOT_SUPPORTED;
            goto out;
        }
    }

    qp_gverbs = (struct doca_gpu_verbs_qp *)calloc(1, sizeof(struct doca_gpu_verbs_qp));
    if (qp_gverbs == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to allocate CPU memory");
        status = DOCA_ERROR_NO_MEMORY;
        goto out;
    }

    qp_gverbs->qp_cpu =
        (struct doca_gpu_dev_verbs_qp *)calloc(1, sizeof(struct doca_gpu_dev_verbs_qp));
    if (qp_gverbs->qp_cpu == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to allocate CPU memory");
        status = DOCA_ERROR_NO_MEMORY;
        goto out;
    }

    qp_cpu_ = qp_gverbs->qp_cpu;

    // Should this be propagated to GPU?
    // Do we need it?
    // if (qp->get_uar_mtype() == DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME)
    //     gpu_dev->open->support_bf_uar = true;

    // Check QP and CQ same size!!!!

    doca_verbs_qp_get_wq(qp,
                         (void **)&(qp_cpu_->sq_wqe_daddr),  // broken for external umem
                         &sq_wqe_num,
                         (void **)&(rq_wqe_daddr),  // broken for external umem
                         &rq_wqe_num, &rcv_wqe_size);

    status = doca_verbs_qp_get_dbr_addr(qp, (void **)&dbrec);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Can't get QP dbr addr.");
        goto out;
    }

    qp_cpu_->sq_wqe_num = (uint16_t)sq_wqe_num;
    qp_cpu_->sq_wqe_mask = qp_cpu_->sq_wqe_num - 1;
    status = doca_verbs_qp_get_qpn(qp, &qp_cpu_->sq_num);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Can't get QP number.");
        goto out;
    }

    qp_cpu_->sq_num_shift8 = qp_cpu_->sq_num << 8;
    qp_cpu_->sq_num_shift8_be = htobe32(qp_cpu_->sq_num_shift8);
    qp_cpu_->sq_num_shift8_be_1ds = htobe32(qp_cpu_->sq_num_shift8 | 1);
    qp_cpu_->sq_num_shift8_be_2ds = htobe32(qp_cpu_->sq_num_shift8 | 2);
    qp_cpu_->sq_num_shift8_be_3ds = htobe32(qp_cpu_->sq_num_shift8 | 3);
    qp_cpu_->sq_num_shift8_be_4ds = htobe32(qp_cpu_->sq_num_shift8 | 4);
    qp_cpu_->sq_num_shift8_be_5ds = htobe32(qp_cpu_->sq_num_shift8 | 5);
    qp_cpu_->sq_wqe_pi = 0;
    qp_cpu_->sq_rsvd_index = 0;
    qp_cpu_->sq_ready_index = 0;
    qp_cpu_->sq_lock = 0;
    qp_cpu_->sq_dbrec = (__be32 *)(dbrec + DOCA_GPUNETIO_IB_MLX5_SND_DBR);
    qp_cpu_->mem_type = DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU;
    qp_gverbs->cpu_db = nullptr;
    qp_gverbs->sq_db = nullptr;
    qp_gverbs->sq_wqe_pi_last = 0;
    qp_gverbs->cpu_proxy = false;
    qp_gverbs->qp_gpu = nullptr;
    qp_gverbs->send_dbr_mode_ext = send_dbr_mode_ext;
    qp_gverbs->qp = qp;
    qp_gverbs->cq_sq = cq_sq;

    status = doca_verbs_qp_init_attr_create(&qp_init_attr_out);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Can't create QP init attr structure.");
        goto out;
    }

    status = doca_verbs_qp_attr_create(&qp_attr_out);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Can't create QP attr structure.");
        goto out;
    }

    status = doca_verbs_qp_query(qp, qp_attr_out, qp_init_attr_out);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Can't query QP.");
        goto out;
    }

    status = doca_verbs_qp_init_attr_get_send_dbr_mode(qp_init_attr_out, &send_dbr_mode);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Can't get_send_dbr_mode.");
        goto out;
    }

    if (((send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR) &&
         (send_dbr_mode != DOCA_VERBS_QP_SEND_DBR_MODE_DBR_VALID)) ||
        ((send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW) &&
         (send_dbr_mode != DOCA_VERBS_QP_SEND_DBR_MODE_NO_DBR_EXT))) {
        DOCA_LOG(LOG_ERR, "Invalid send_dbr_mode_ext.");
        status = DOCA_ERROR_INVALID_VALUE;
        goto out;
    }

    status = doca_verbs_qp_get_uar_addr(qp, (void **)&sq_db);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Can't get QP UAR address.");
        goto out;
    }

    if ((nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) == 0) {
        status = doca_gpu_verbs_export_uar(sq_db, (uint64_t **)&(qp_cpu_->sq_db));
        if (status != DOCA_SUCCESS && nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
            DOCA_LOG(LOG_ERR, "Can't export UAR to GPU.");
            goto out;
        }
    }

    if ((status != DOCA_SUCCESS && nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) ||
        (nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) != 0) {
        DOCA_LOG(LOG_WARNING, "Enabling CPU proxy mode");

        bool use_free_flow = ((nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_FREE_FLOW) != 0);
        uint32_t num_cpu_db_entries = use_free_flow ? sq_wqe_num : 1;
        status = doca_gpu_mem_alloc(gpu_dev, sizeof(uint64_t) * num_cpu_db_entries,
                                    priv_get_page_size(), DOCA_GPU_MEM_TYPE_CPU_GPU,
                                    (void **)&(qp_gverbs->cpu_db), (void **)&(qp_gverbs->cpu_db));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to alloc GPU memory for CPU proxy DB");
            goto out;
        }

        memset(qp_gverbs->cpu_db, 0, sizeof(uint64_t) * num_cpu_db_entries);
        qp_cpu_->sq_db = qp_gverbs->cpu_db;
        qp_gverbs->cpu_proxy = true;
        qp_gverbs->sq_num_shift8_be = qp_cpu_->sq_num_shift8_be;
        qp_gverbs->sq_dbrec = qp_cpu_->sq_dbrec;
        qp_gverbs->sq_db = sq_db;
        qp_cpu_->nic_handler = use_free_flow ? DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW
                                             : DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
        nic_handler_must_be_cpu_proxy = true;
    }

    if (!nic_handler_must_be_cpu_proxy) {
        if (send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) {
            assert(gpu_dev->open->support_gdrcopy);
            qp_gverbs->sq_dbrec = qp_cpu_->sq_dbrec;
            qp_gverbs->sq_db = sq_db;
            qp_gverbs->cpu_proxy = true;
        }
        if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
            if ((send_dbr_mode == DOCA_VERBS_QP_SEND_DBR_MODE_NO_DBR_EXT) ||
                (send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED))
                qp_cpu_->nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR;
            else
                qp_cpu_->nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
        } else
            qp_cpu_->nic_handler = nic_handler;
    }

    doca_verbs_cq_get_wq(cq_sq, (void **)&(qp_cpu_->cq_sq.cqe_daddr), &(qp_cpu_->cq_sq.cqe_num),
                         &(qp_cpu_->cq_sq.cqe_size));

    status = doca_verbs_cq_get_dbr_db_addr(cq_sq, &uar_db_reg, (uint32_t **)&(cq_dbrec), &arm_dbr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to get CQ DBR address, error = %d", status);
        goto out;
    }

    qp_cpu_->cq_sq.dbrec = (__be32 *)cq_dbrec;
    doca_verbs_cq_get_cq_num(cq_sq, &qp_cpu_->cq_sq.cq_num);
    qp_cpu_->cq_sq.cqe_mask = (qp_cpu_->cq_sq.cqe_num - 1);
    qp_cpu_->cq_sq.cqe_ci = 0;
    qp_cpu_->cq_sq.cqe_rsvd = 0;
    qp_cpu_->cq_sq.mem_type = DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU;
    qp_cpu_->cq_sq.cq_type = cq_type;

    qp_gverbs->gpu_dev = gpu_dev;
    qp_gverbs->free_flow_ring_db_threshold = DOCA_GPUNETIO_FREE_FLOW_RING_DB_THRESHOLD_DEFAULT;
    qp_gverbs->enable_data_direct = enable_data_direct;
    qp_gverbs->cq_type = cq_type;

    *qp_out = qp_gverbs;

out:
    if (qp_attr_out) doca_verbs_qp_attr_destroy(qp_attr_out);
    if (qp_init_attr_out) doca_verbs_qp_init_attr_destroy(qp_init_attr_out);

    if (status != DOCA_SUCCESS) {
        if (qp_gverbs) {
            if (qp_gverbs->qp_cpu) {
                if (qp_gverbs->qp_cpu->sq_db &&
                    ((qp_gverbs->qp_cpu->nic_handler &
                      DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) == 0))
                    doca_gpu_verbs_unexport_uar(qp_gverbs->qp_cpu->sq_db);
                free(qp_gverbs->qp_cpu);
            }
            if (qp_gverbs->cpu_db) {
                doca_gpu_mem_free(gpu_dev, qp_gverbs->cpu_db);
            }
            free(qp_gverbs);
        }
    }

    return status;
}

doca_error_t doca_gpu_verbs_get_qp_dev(struct doca_gpu_verbs_qp *qp,
                                       struct doca_gpu_dev_verbs_qp **qp_gpu) {
    doca_error_t status = DOCA_SUCCESS;
    int custatus = 0;

    enum doca_gpu_mem_type mtype;

    if (qp == nullptr) return DOCA_ERROR_INVALID_VALUE;

    assert(qp->qp_cpu);

    if (qp->qp_gpu == nullptr) {
        if ((qp->send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) ||
            (qp->cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST)) {
            mtype = qp->enable_data_direct ? DOCA_GPU_MEM_TYPE_GPU_CPU_DATA_DIRECT
                                           : DOCA_GPU_MEM_TYPE_GPU_CPU;
        } else {
            mtype = DOCA_GPU_MEM_TYPE_GPU;
        }

        status = doca_gpu_mem_alloc(qp->gpu_dev, sizeof(struct doca_gpu_dev_verbs_qp),
                                    priv_get_page_size(), mtype, (void **)&qp->qp_gpu,
                                    (void **)&qp->qp_gpu_h);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to alloc gpu memory for qp_gpu");
            return status;
        }

        custatus = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaMemcpy(
            qp->qp_gpu, qp->qp_cpu, sizeof(struct doca_gpu_dev_verbs_qp), cudaMemcpyHostToDevice));
        if (custatus != cudaSuccess) {
            DOCA_LOG(LOG_ERR, "cuMemcpyHtoD failed");
            doca_gpu_mem_free(qp->gpu_dev, qp->qp_gpu);
            qp->qp_gpu = nullptr;
            qp->qp_gpu_h = nullptr;
            return DOCA_ERROR_DRIVER;
        }

        if ((qp->send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) &&
            ((qp->qp_cpu->nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) == 0)) {
            assert(qp->gpu_dev->open->support_gdrcopy);
            qp->cpu_db = &qp->qp_gpu_h->sq_wqe_pi;
        }
    }

    *qp_gpu = qp->qp_gpu;

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_unexport_qp(doca_gpu_t *gpu_dev, struct doca_gpu_verbs_qp *qp_gverbs) {
    if (gpu_dev == nullptr || qp_gverbs == nullptr) return DOCA_ERROR_INVALID_VALUE;

    if (qp_gverbs->refcount > 0) {
        return DOCA_ERROR_IN_USE;
    }

    if (qp_gverbs->cpu_db &&
        ((qp_gverbs->send_dbr_mode_ext !=
          DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) ||
         (qp_gverbs->qp_cpu &&
          (qp_gverbs->qp_cpu->nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) != 0)))
        doca_gpu_mem_free(gpu_dev, qp_gverbs->cpu_db);

    if (qp_gverbs->qp_cpu) {
        if ((qp_gverbs->qp_cpu->nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) == 0)
            doca_gpu_verbs_unexport_uar(qp_gverbs->qp_cpu->sq_db);
        free(qp_gverbs->qp_cpu);
    }

    if (qp_gverbs->qp_gpu) {
        doca_gpu_mem_free(gpu_dev, qp_gverbs->qp_gpu);
        qp_gverbs->qp_gpu = nullptr;
        qp_gverbs->qp_gpu_h = nullptr;
    }

    free(qp_gverbs);

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_export_multi_qps_dev(doca_gpu_t *gpu_dev,
                                                 struct doca_gpu_verbs_qp **qps,
                                                 unsigned int num_qps,
                                                 struct doca_gpu_dev_verbs_qp **out_qp_gpus) {
    doca_error_t status = DOCA_SUCCESS;
    int custatus = 0;

    enum doca_gpu_mem_type mtype;
    struct doca_gpu_dev_verbs_qp *qp_gpus_d = nullptr;
    struct doca_gpu_dev_verbs_qp *qp_cpus = nullptr;
    struct doca_gpu_dev_verbs_qp *qp_gpus_h = nullptr;
    struct doca_gpu_verbs_qp *qp;
    bool need_cpu_mapping = false;
    bool need_data_direct = false;

    unsigned int qp_idx;

    if (gpu_dev == nullptr || qps == nullptr || num_qps == 0 || out_qp_gpus == nullptr) {
        status = DOCA_ERROR_INVALID_VALUE;
        goto out;
    }

    for (unsigned int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
        qp = qps[qp_idx];
        if (qp == nullptr) continue;
        assert(qp->qp_cpu != nullptr);
        need_cpu_mapping |=
            (qp->send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) ||
            (qp->cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST);
        need_data_direct |= qp->enable_data_direct;
    }

    if (need_cpu_mapping) assert(gpu_dev->open->support_gdrcopy);

    if (need_cpu_mapping)
        mtype =
            need_data_direct ? DOCA_GPU_MEM_TYPE_GPU_CPU_DATA_DIRECT : DOCA_GPU_MEM_TYPE_GPU_CPU;
    else
        mtype = DOCA_GPU_MEM_TYPE_GPU;
    status = doca_gpu_mem_alloc(gpu_dev, sizeof(struct doca_gpu_dev_verbs_qp) * num_qps,
                                GPU_PAGE_SIZE, mtype, (void **)&qp_gpus_d, (void **)&qp_cpus);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to alloc gpu memory for qp_gpus");
        goto out;
    }

    qp_gpus_h =
        (struct doca_gpu_dev_verbs_qp *)calloc(num_qps, sizeof(struct doca_gpu_dev_verbs_qp));
    if (qp_gpus_h == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to allocate memory for qp_gpus_h");
        goto out;
    }

    for (qp_idx = 0; qp_idx < num_qps; qp_idx++) {
        qp = qps[qp_idx];
        if (qp == nullptr) continue;
        assert(qp->qp_cpu != nullptr);
        qp->qp_gpu = &(qp_gpus_d[qp_idx]);
        if (qp->cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST)
            qp->qp_gpu_h = &(qp_cpus[qp_idx]);
        if ((qp->send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) &&
            ((qp->qp_cpu->nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) == 0)) {
            qp->qp_gpu_h = &(qp_cpus[qp_idx]);
            qp->cpu_db = &qp->qp_gpu_h->sq_wqe_pi;
        }
        memcpy(&qp_gpus_h[qp_idx], qp->qp_cpu, sizeof(struct doca_gpu_dev_verbs_qp));
    }

    custatus = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaMemcpy(qp_gpus_d, qp_gpus_h, sizeof(struct doca_gpu_dev_verbs_qp) * num_qps,
                   cudaMemcpyHostToDevice));
    if (custatus != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "cuMemcpyHtoD failed");
        status = DOCA_ERROR_DRIVER;
        goto out;
    }

    *out_qp_gpus = qp_gpus_d;

out:
    if (status != DOCA_SUCCESS) {
        if (qp_gpus_d) doca_gpu_mem_free(gpu_dev, qp_gpus_d);
    }
    if (qp_gpus_h) free(qp_gpus_h);
    return status;
}

doca_error_t doca_gpu_verbs_unexport_multi_qps_dev(doca_gpu_t *gpu_dev,
                                                   struct doca_gpu_verbs_qp **qps,
                                                   unsigned int num_qps,
                                                   struct doca_gpu_dev_verbs_qp *qp_gpus) {
    doca_error_t status = DOCA_SUCCESS;
    if (gpu_dev == nullptr || qps == nullptr || num_qps == 0 || qp_gpus == nullptr)
        return DOCA_ERROR_INVALID_VALUE;

    for (unsigned int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
        struct doca_gpu_verbs_qp *qp = qps[qp_idx];
        if (qp == nullptr) continue;
        qp->qp_gpu = nullptr;
        qp->qp_gpu_h = nullptr;
    }

    status = doca_gpu_mem_free(gpu_dev, qp_gpus);
    return status;
}

static inline void priv_cpu_proxy_progress_full_assisted(struct doca_gpu_verbs_qp *qp,
                                                         bool *out_progressed) {
    __be32 dbr_val;
    bool progressed = false;
    bool use_free_flow =
        (qp->qp_cpu->nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_FREE_FLOW) != 0;
    uint64_t pi_last = qp->sq_wqe_pi_last;
    uint64_t new_pi = pi_last;
    const uint16_t wqe_mask = qp->qp_cpu->sq_wqe_mask;

    if (use_free_flow) {
        while (reinterpret_cast<std::atomic<uint64_t> *>(&qp->cpu_db[new_pi & wqe_mask])
                   ->load(std::memory_order_relaxed) == new_pi + 1) {
            new_pi++;
            if ((qp->free_flow_ring_db_threshold) > 0 &&
                (new_pi - pi_last >= qp->free_flow_ring_db_threshold))
                break;
        }
    } else {
        new_pi = (reinterpret_cast<std::atomic<uint64_t> *>(qp->cpu_db)
                      ->load(std::memory_order_relaxed));
    }

    if (new_pi > pi_last) {
        struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg ctrl_seg = {
            .opmod_idx_opcode = htobe32(new_pi << 8), .qpn_ds = qp->sq_num_shift8_be};

        if (qp->send_dbr_mode_ext != DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW) {
            dbr_val = htobe32(new_pi & 0xffff);

            // Ring the DB ASAP.
            // The second DB ringing happens after the fence. This is used when the NIC enters a
            // recovery state and it needs to read DBR.
            reinterpret_cast<std::atomic<uint64_t> *>(qp->sq_db)->store(
                *reinterpret_cast<uint64_t *>(&ctrl_seg), std::memory_order_relaxed);
            reinterpret_cast<std::atomic<uint32_t> *>(qp->sq_dbrec)
                ->store(dbr_val, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
        }
        reinterpret_cast<std::atomic<uint64_t> *>(qp->sq_db)->store(
            *reinterpret_cast<uint64_t *>(&ctrl_seg), std::memory_order_relaxed);

        qp->sq_wqe_pi_last = new_pi;
        progressed = true;
    }
    *out_progressed = progressed;
}

static inline void priv_cpu_proxy_progress_dbr_assisted(struct doca_gpu_verbs_qp *qp) {
    uint32_t tmp_db = 0;
    __be32 dbr_val;

    tmp_db = (uint32_t)(reinterpret_cast<std::atomic<uint64_t> *>(qp->cpu_db)
                            ->load(std::memory_order_relaxed));
    if (tmp_db != qp->sq_wqe_pi_last) {
        struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg ctrl_seg = {
            .opmod_idx_opcode = htobe32(tmp_db << 8), .qpn_ds = qp->sq_num_shift8_be};

        dbr_val = htobe32(tmp_db & 0xffff);
        reinterpret_cast<std::atomic<uint32_t> *>(qp->sq_dbrec)
            ->store(dbr_val, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<std::atomic<uint64_t> *>(qp->sq_db)->store(
            *reinterpret_cast<uint64_t *>(&ctrl_seg), std::memory_order_relaxed);

        qp->sq_wqe_pi_last = tmp_db;
    }
}

static inline void priv_cpu_proxy_progress_cq(struct doca_gpu_verbs_qp *qp, bool *out_progressed) {
    bool progressed = false;
    struct doca_gpu_dev_verbs_qp *qp_cpu = qp->qp_cpu;
    struct doca_gpu_dev_verbs_cq *cq = &qp_cpu->cq_sq;
    uint64_t old_cqe_ci = 0;
    uint64_t new_cqe_ci = 0;
    uint32_t cqe_chunk = 0;
    uint16_t wqe_counter = 0;
    uint8_t opown = 0;
    uint8_t opcode = 0;
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 =
        reinterpret_cast<struct doca_gpunetio_ib_mlx5_cqe64 *>(cq->cqe_daddr);

    [[unlikely]] if (qp->qp_gpu_h == nullptr)
        goto out;

    old_cqe_ci = cq->cqe_ci;

    cqe_chunk = reinterpret_cast<std::atomic<uint32_t> *>(&cqe64->wqe_counter_sig_op_own_raw)
                    ->load(std::memory_order_relaxed);
    cqe_chunk = be32toh(cqe_chunk);
    wqe_counter = cqe_chunk >> 16;
    opown = cqe_chunk & 0xff;
    opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

    if ((opcode == DOCA_GPUNETIO_IB_MLX5_CQE_INVALID) ||
        ((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(wqe_counter & cq->cqe_num)))
        goto out;

    [[unlikely]] if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) {
        DOCA_LOG(LOG_WARNING, "CQE indicates request error");
        goto out;
    }

    ++wqe_counter;
    new_cqe_ci = ((old_cqe_ci & ~(0xFFFFULL)) | wqe_counter) +
                 (((uint16_t)old_cqe_ci > wqe_counter) ? 0x10000ULL : 0x0);

    if (new_cqe_ci > old_cqe_ci) {
        uint64_t *gpu_cqe_ci = &qp->qp_gpu_h->cq_sq.cqe_ci;

        if (qp->gpu_dev->open->need_mcst) {
            (void)READ_ONCE(*gpu_cqe_ci);
        }

        WRITE_ONCE(*gpu_cqe_ci, new_cqe_ci);
        cq->cqe_ci = new_cqe_ci;
        progressed = true;
    }

out:
    *out_progressed = progressed;
}

doca_error_t doca_gpu_verbs_cpu_proxy_progress(struct doca_gpu_verbs_qp *qp, bool *out_progressed) {
    bool progressed = false;
    bool cq_progressed = false;
    bool qp_progressed = false;
    if (qp == nullptr) return DOCA_ERROR_INVALID_VALUE;

    if (qp->cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST) {
        priv_cpu_proxy_progress_cq(qp, &cq_progressed);
    }

    if (qp->cpu_proxy != true) {
        if (qp->cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST) {
            if (out_progressed) *out_progressed = cq_progressed;
            return DOCA_SUCCESS;
        }
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    if ((qp->send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) &&
        ((qp->qp_cpu->nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) == 0)) {
        priv_cpu_proxy_progress_dbr_assisted(qp);
    } else {
        priv_cpu_proxy_progress_full_assisted(qp, &qp_progressed);
    }

    progressed = cq_progressed || qp_progressed;

    if (out_progressed) *out_progressed = progressed;
    return DOCA_SUCCESS;
}

static void *priv_service_mainloop(void *args) {
    struct doca_gpu_verbs_service *service = (struct doca_gpu_verbs_service *)args;
    bool progressed = false;

    while (service->running) {
        pthread_rwlock_rdlock(&service->service_lock);
        do {
            progressed = false;
            for (auto qp : *service->qps) {
                bool qp_progressed = false;
                doca_gpu_verbs_cpu_proxy_progress(qp, &qp_progressed);
                progressed |= qp_progressed;
            }
        } while (progressed);
        pthread_rwlock_unlock(&service->service_lock);
        sched_yield();
    }

    return nullptr;
}

doca_error_t doca_gpu_verbs_create_service(doca_gpu_verbs_service_t *out_service) {
    int status = 0;
    doca_error_t doca_status = DOCA_SUCCESS;
    struct doca_gpu_verbs_service *service = nullptr;

    if (out_service == nullptr) return DOCA_ERROR_INVALID_VALUE;

    service = (struct doca_gpu_verbs_service *)calloc(1, sizeof(struct doca_gpu_verbs_service));
    if (service == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to allocate memory for service");
        doca_status = DOCA_ERROR_NO_MEMORY;
        goto out;
    }

    status = pthread_rwlock_init(&service->service_lock, nullptr);
    if (status != 0) {
        DOCA_LOG(LOG_ERR, "Failed to initialize service lock");
        doca_status = DOCA_ERROR_DRIVER;
        goto out;
    }

    service->running = true;
    service->qps = new std::set<struct doca_gpu_verbs_qp *>();
    status = pthread_create(&service->service_thread, nullptr, priv_service_mainloop, service);
    if (status != 0) {
        DOCA_LOG(LOG_ERR, "Failed to create service thread");
        doca_status = DOCA_ERROR_DRIVER;
        goto out;
    }

    *out_service = service;

out:
    if (status) {
        if (service->qps) delete service->qps;
        if (service) free(service);
    }
    return doca_status;
}

doca_error_t doca_gpu_verbs_service_monitor_qp(doca_gpu_verbs_service_t service,
                                               struct doca_gpu_verbs_qp *qp) {
    struct doca_gpu_verbs_service *service_ = (struct doca_gpu_verbs_service *)service;
    if (service == nullptr || qp == nullptr) return DOCA_ERROR_INVALID_VALUE;

    pthread_rwlock_wrlock(&service_->service_lock);
    service_->qps->insert(qp);
    pthread_rwlock_unlock(&service_->service_lock);

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_destroy_service(doca_gpu_verbs_service_t service) {
    struct doca_gpu_verbs_service *service_ = (struct doca_gpu_verbs_service *)service;
    if (service == nullptr) return DOCA_ERROR_INVALID_VALUE;

    service_->running = false;
    pthread_join(service_->service_thread, nullptr);
    pthread_rwlock_destroy(&service_->service_lock);
    delete service_->qps;
    free(service_);

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_query_last_error(struct doca_gpu_verbs_qp *qp,
                                             struct doca_gpu_verbs_qp_error_info *error_info) {
    doca_error_t status = DOCA_SUCCESS;
    enum doca_verbs_qp_state current_state;
    doca_verbs_qp_attr_t *qp_attr = nullptr;
    doca_verbs_qp_init_attr_t *qp_init_attr = nullptr;

    if (qp == nullptr || qp->qp == nullptr || error_info == nullptr)
        return DOCA_ERROR_INVALID_VALUE;

    memset(error_info, 0, sizeof(struct doca_gpu_verbs_qp_error_info));

    status = doca_verbs_qp_init_attr_create(&qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs qp attributes");
        goto out;
    }

    status = doca_verbs_qp_attr_create(&qp_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs qp attributes");
        goto out;
    }

    status = doca_verbs_qp_query(qp->qp, qp_attr, qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query QP");
        goto out;
    }

    status = doca_verbs_qp_attr_get_current_state(qp_attr, &current_state);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query QP attr current state");
        goto out;
    }

    error_info->has_error = (current_state == DOCA_VERBS_QP_STATE_ERR);

    status = doca_verbs_qp_init_attr_destroy(qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs qp attributes");
        goto out;
    }

    qp_init_attr = nullptr;

    status = doca_verbs_qp_attr_destroy(qp_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs qp attributes");
        goto out;
    }

    qp_attr = nullptr;

    return DOCA_SUCCESS;

out:
    if (qp_init_attr) doca_verbs_qp_init_attr_destroy(qp_init_attr);
    if (qp_attr) doca_verbs_qp_attr_destroy(qp_attr);

    return status;
}

doca_error_t doca_gpu_verbs_reset_tracking_and_memory(struct doca_gpu_verbs_qp *qp_gverbs) {
    doca_error_t status = DOCA_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;

    struct doca_gpu_dev_verbs_qp qp_gpu_h;

    if (qp_gverbs == nullptr) {
        status = DOCA_ERROR_INVALID_VALUE;
        goto out;
    }

    assert(qp_gverbs->qp_gpu);
    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaMemcpy(&qp_gpu_h, qp_gverbs->qp_gpu,
                                                              sizeof(struct doca_gpu_dev_verbs_qp),
                                                              cudaMemcpyDeviceToHost));
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to copy qp_gpu to qp_gpu_h");
        status = DOCA_ERROR_DRIVER;
        goto out;
    }

    assert(qp_gverbs->qp_cpu);

    qp_gverbs->qp_cpu->sq_wqe_pi = 0;
    qp_gverbs->qp_cpu->sq_rsvd_index = 0;
    qp_gverbs->qp_cpu->sq_ready_index = 0;
    qp_gverbs->qp_cpu->sq_lock = 0;

    qp_gverbs->qp_cpu->cq_sq.cqe_ci = 0;
    qp_gverbs->qp_cpu->cq_sq.cqe_rsvd = qp_gpu_h.sq_rsvd_index;

    qp_gverbs->sq_wqe_pi_last = 0;

    if (qp_gverbs->cpu_proxy) {
        assert(qp_gverbs->sq_dbrec);
        if (qp_gverbs->sq_dbrec) *qp_gverbs->sq_dbrec = 0;
        if (qp_gverbs->cpu_db) {
            uint32_t num_cpu_db_entries =
                qp_gverbs->qp_cpu->nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_FREE_FLOW
                    ? qp_gverbs->qp_cpu->sq_wqe_num
                    : 1;
            memset(qp_gverbs->cpu_db, 0, num_cpu_db_entries * sizeof(uint64_t));
        }
    } else {
        assert(qp_gverbs->qp_cpu->sq_dbrec);
        cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
            cudaMemset(qp_gverbs->qp_cpu->sq_dbrec, 0, sizeof(uint32_t)));
        if (cuda_status != cudaSuccess) {
            DOCA_LOG(LOG_ERR, "Failed to reset sq_dbrec");
            status = DOCA_ERROR_DRIVER;
            goto out;
        }
    }

    assert(qp_gverbs->qp_cpu->cq_sq.cqe_daddr);
    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaMemset(qp_gverbs->qp_cpu->cq_sq.cqe_daddr, 0xff,
                   qp_gverbs->qp_cpu->cq_sq.cqe_num * qp_gverbs->qp_cpu->cq_sq.cqe_size));
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to reset cqe_daddr");
        status = DOCA_ERROR_DRIVER;
        goto out;
    }

    assert(qp_gverbs->qp_cpu->sq_wqe_daddr);
    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaMemset(qp_gverbs->qp_cpu->sq_wqe_daddr, 0,
                   qp_gverbs->qp_cpu->sq_wqe_num * sizeof(struct doca_gpu_dev_verbs_wqe)));
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to reset sq_wqe_daddr");
        status = DOCA_ERROR_DRIVER;
        goto out;
    }

    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaMemcpy(qp_gverbs->qp_gpu, qp_gverbs->qp_cpu,
                                                              sizeof(struct doca_gpu_dev_verbs_qp),
                                                              cudaMemcpyHostToDevice));
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to update qp_gpu");
        status = DOCA_ERROR_DRIVER;
        goto out;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to synchronize");
        status = DOCA_ERROR_DRIVER;
        goto out;
    }

out:
    return status;
}

doca_error_t doca_gpu_verbs_get_library_version(uint32_t *version) {
    if (version == nullptr) return DOCA_ERROR_INVALID_VALUE;
    *version = DOCA_GPUNETIO_VERSION;
    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_check_device_code_compatibility(uint32_t device_code_version) {
    if ((device_code_version < DOCA_GPUNETIO_MIN_COMPAT_DEVICE_CODE_VERSION) ||
        (device_code_version > DOCA_GPUNETIO_VERSION))
        return DOCA_ERROR_NOT_SUPPORTED;
    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_check_host_code_compatibility(uint32_t host_code_version) {
    if ((host_code_version < DOCA_GPUNETIO_MIN_COMPAT_HOST_CODE_VERSION) ||
        (host_code_version > DOCA_GPUNETIO_VERSION))
        return DOCA_ERROR_NOT_SUPPORTED;
    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_req_notify_cq(doca_gpu_t *gpu_dev, doca_verbs_cq_t *verbs_cq) {
    doca_error_t status = DOCA_SUCCESS;
    cudaError_t cuda_status;
    cudaPointerAttributes attributes;
    bool is_gpu_memory = false;
    uint32_t *ci_dbr = nullptr;
    uint32_t *arm_dbr = nullptr;
    uint64_t *uar_db_reg = nullptr;
    uint32_t cqn = 0;
    __be32 dbr_val;
    __be64 db_val;
    cudaStream_t stream = nullptr;

    if (gpu_dev == nullptr || verbs_cq == nullptr) return DOCA_ERROR_INVALID_VALUE;

    if (verbs_cq->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        if (gpu_dev->type != DOCA_GPU_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "Invalid DOCA GPUNetIO SDK handler provided.");
            return DOCA_ERROR_INVALID_VALUE;
        }

        auto err = doca_gpu_sdk_wrapper_verbs_req_notify_cq(gpu_dev->sdk, verbs_cq->sdk);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        } else if (err == DOCA_SDK_WRAPPER_NOT_SUPPORTED) {
            return DOCA_ERROR_NOT_SUPPORTED;
        }
    }

    if (verbs_cq->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ open instance provided at %s line %d.", __func__,
                 __LINE__);
        return DOCA_ERROR_INVALID_VALUE;
    }

    cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
        status = DOCA_ERROR_DRIVER;
        goto out;
    }

    status = doca_verbs_cq_get_dbr_db_addr(verbs_cq, &uar_db_reg, &ci_dbr, &arm_dbr);
    if (status) goto out;

    status = doca_verbs_cq_get_cq_num(verbs_cq, &cqn);
    if (status) goto out;

    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaPointerGetAttributes(&attributes, static_cast<const void *>(arm_dbr)));
    if (cuda_status != cudaSuccess && cuda_status != cudaErrorInvalidValue) {
        // cudaErrorInvalidValue is expected if the pointer was not allocated in, mapped by or
        // registered with context supporting unified addressing. In this case, the pointer is
        // not a device pointer.
        DOCA_LOG(LOG_ERR, "Failed to get CUDA pointer attributes");
        status = DOCA_ERROR_DRIVER;
        goto out;
    } else if (cuda_status == cudaSuccess && attributes.type == cudaMemoryTypeDevice)
        is_gpu_memory = true;

    dbr_val = htobe32(MLX5_CQ_DB_REQ_NOT_SOL);

    if (is_gpu_memory) {
        cudaError_t cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
            cudaMemcpyAsync(arm_dbr, &dbr_val, sizeof(dbr_val), cudaMemcpyHostToDevice, stream));
        if (cuda_status != cudaSuccess) {
            DOCA_LOG(LOG_ERR, "Failed to copy CQ arm DBR to GPU memory");
            return DOCA_ERROR_DRIVER;
        }

        cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaStreamSynchronize(stream));
        if (cuda_status != cudaSuccess) {
            DOCA_LOG(LOG_ERR, "Failed to synchronize GPU CQ arm DBR write");
            status = DOCA_ERROR_DRIVER;
            goto out;
        }
    } else
        *arm_dbr = dbr_val;

    std::atomic_thread_fence(std::memory_order_release);

    db_val = htobe64((static_cast<uint64_t>(MLX5_CQ_DB_REQ_NOT_SOL) << 32) | cqn);
    *reinterpret_cast<volatile __be64 *>(uar_db_reg) = db_val;

out:
    cudaStreamDestroy(stream);

    return status;
}
