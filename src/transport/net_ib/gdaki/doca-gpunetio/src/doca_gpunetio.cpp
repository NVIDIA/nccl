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
#include <cuda_runtime.h>
#include <string.h>

#include <atomic>
#include <set>
#include <unordered_map>
#include <mutex>

#include "host/mlx5_prm.h"
#include "host/mlx5_ifc.h"

#include "doca_verbs_net_wrapper.h"
#include "doca_internal.hpp"
#include "host/doca_gpunetio.h"
#include "doca_gpunetio_gdrcopy.h"
#include "common/doca_gpunetio_verbs_dev.h"
#include "host/doca_verbs.h"
#include "doca_verbs_qp.hpp"
#include "doca_verbs_cuda_wrapper.h"

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_FULL_ASYNC_STORE_RELEASE_SUPPORT_COMPUTE_CAP_MAJOR 10

struct doca_gpu_mtable {
    uintptr_t base_addr;
    size_t size_orig;
    uintptr_t align_addr_gpu;
    uintptr_t align_addr_cpu;
    size_t size;
    enum doca_gpu_mem_type mtype;
    void *gdr_mh;
};

struct doca_gpu {
    CUdevice cuda_dev; /* CUDA device handler */
    std::unordered_map<uintptr_t, struct doca_gpu_mtable *>
        *mtable;                       /* Table of GPU/CPU memory allocated addresses */
    bool support_gdrcopy;              ///< Boolean value that indicates if gdrcopy is
                                       ///< supported
    bool support_dmabuf;               ///< Boolean value that indicates if dmabuf is
                                       ///< supported by the gpu
    bool support_wq_gpumem;            ///< Boolean value that indicates if gpumem is
                                       ///< available and nic-gpu mapping is supported
    bool support_cq_gpumem;            ///< Boolean value that indicates if gpumem is
                                       ///< available and nic-gpu mapping is supported
    bool support_uar_gpumem;           ///< Boolean value that indicates if gpumem is
                                       ///< available and gpu-nic mapping is supported
    bool support_async_store_release;  ///< Boolean value that indicates if
                                       ///< async store release is supported
    bool support_bf_uar;               ///< Boolean value that indicates if BlueFlame
                                       ///< is supported
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

doca_error_t doca_gpu_create(const char *gpu_bus_id, struct doca_gpu **gpu_dev) {
    struct doca_gpu *gpu_dev_;
    int dmabuf_supported;
    CUresult res_drv = CUDA_SUCCESS;
    cudaError_t res_cuda = cudaSuccess;

    if (gpu_bus_id == nullptr || gpu_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    gpu_dev_ = (struct doca_gpu *)calloc(1, sizeof(struct doca_gpu));
    if (gpu_dev_ == nullptr) {
        DOCA_LOG(LOG_ERR, "error in %s: failed to allocate memory for doca_gpu", __func__);
        return DOCA_ERROR_NO_MEMORY;
    }

    res_cuda =
        DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaDeviceGetByPCIBusId(&gpu_dev_->cuda_dev, gpu_bus_id));
    if (res_cuda != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Invalid GPU bus id provided (ret %d).", res_drv);
        goto exit_error;
    }

    res_drv = doca_verbs_wrapper_cuDeviceGetAttribute(
        &(dmabuf_supported), CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, gpu_dev_->cuda_dev);
    if (res_drv != CUDA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "cuDeviceGetAttribute CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED returned %d.",
                 res_drv);
        goto exit_error;
    }

    (dmabuf_supported == 1 ? (gpu_dev_->support_dmabuf = true)
                           : (gpu_dev_->support_dmabuf = false));

    // status = gdaki_map_uar(guar);
    // device_attr->support_uar_gpumem = (status == 0);
    // did_map_uar = (status == 0);

    // TBD
    gpu_dev_->support_wq_gpumem = true;
    gpu_dev_->support_cq_gpumem = true;
    gpu_dev_->support_uar_gpumem = true;
    gpu_dev_->support_bf_uar = true;
    gpu_dev_->support_async_store_release = priv_query_async_store_release_support();
    gpu_dev_->support_gdrcopy = doca_gpu_gdrcopy_is_supported();

    try {
        gpu_dev_->mtable = new std::unordered_map<uintptr_t, struct doca_gpu_mtable *>();
    } catch (...) {
        DOCA_LOG(LOG_ERR, "mtable map allocation failed");
        goto exit_error;
    }

    (*gpu_dev) = gpu_dev_;

    return DOCA_SUCCESS;

exit_error:
    free(gpu_dev_);

    return DOCA_ERROR_INITIALIZATION;
}

doca_error_t doca_gpu_destroy(struct doca_gpu *gpu_dev) {
    if (gpu_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (gpu_dev->mtable != nullptr) {
        if (gpu_dev->mtable->size() > 0) {
            DOCA_LOG(LOG_ERR, "mtable map is not empty.");
            return DOCA_ERROR_INVALID_VALUE;
        }
        delete gpu_dev->mtable;
    }

    free(gpu_dev);

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_mem_alloc(struct doca_gpu *gpu_dev, size_t size, size_t alignment,
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
    doca_error_t status = DOCA_SUCCESS;

    if (gpu_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA GPUNetIO instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (memptr_gpu == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid memptr_gpu provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (mtype != DOCA_GPU_MEM_TYPE_GPU && memptr_cpu == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid memptr_cpu provided.");
        return DOCA_ERROR_INVALID_VALUE;
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

    if (mtype == DOCA_GPU_MEM_TYPE_GPU_CPU && alignment != GPU_PAGE_SIZE) alignment = GPU_PAGE_SIZE;

    if (mtype == DOCA_GPU_MEM_TYPE_GPU) {
        mentry->size_orig = mentry->size + alignment;

        res = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
            cudaMalloc(&(cudev_memptr_gpu_orig_), mentry->size_orig));
        if (res != cudaSuccess) {
            err_string = cudaGetErrorString(res);
            DOCA_LOG(LOG_ERR, "cudaMalloc current failed with %s size %zd", err_string,
                     mentry->size_orig);
            goto error;
        }

        /* Align memory address */
        cudev_memptr_gpu_ = cudev_memptr_gpu_orig_;
        if (alignment && ((uintptr_t)cudev_memptr_gpu_) % alignment)
            cudev_memptr_gpu_ =
                (void *)((uintptr_t)cudev_memptr_gpu_ +
                         (alignment - (((uintptr_t)cudev_memptr_gpu_) % alignment)));

        /* GPUDirect RDMA attribute required */
        res_drv = doca_verbs_wrapper_cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                                           (CUdeviceptr)cudev_memptr_gpu_);
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
    } else if (mtype == DOCA_GPU_MEM_TYPE_GPU_CPU) {
        if (gpu_dev->support_gdrcopy == true) {
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
            res_drv = doca_verbs_wrapper_cuPointerSetAttribute(
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

            ret =
                doca_gpu_gdrcopy_create_mapping((void *)mentry->align_addr_gpu, mentry->size,
                                                &mentry->gdr_mh, (void **)&mentry->align_addr_cpu);
            if (ret) {
                DOCA_LOG(LOG_ERR, "Error mapping GPU memory at %lx to CPU", mentry->align_addr_gpu);
                status = DOCA_ERROR_DRIVER;
                goto error;
            }
        } else {
            DOCA_LOG(LOG_WARNING,
                     "GDRCopy not enabled, can't allocate memory type DOCA_GPU_MEM_TYPE_GPU_CPU. "
                     "Using DOCA_GPU_MEM_TYPE_CPU_GPU mode instead");

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

    *memptr_gpu = (void *)mentry->align_addr_gpu;
    if (memptr_cpu) *memptr_cpu = (void *)mentry->align_addr_cpu;

    // DOCA_LOG(LOG_DEBUG, "New memory: Orig %lx GPU %lx CPU %lx type %d size %zd\n",
    // 	      mentry->base_addr,
    // 	      mentry->align_addr_gpu,
    // 	      mentry->align_addr_cpu,
    // 	      mentry->mtype,
    // 	      mentry->size);

    try {
        gpu_dev->mtable->insert({mentry->align_addr_gpu, mentry});
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

doca_error_t doca_gpu_mem_free(struct doca_gpu *gpu_dev, void *memptr_gpu) {
    struct doca_gpu_mtable *mentry;
    cudaError_t res_cuda;

    if (gpu_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA GPUNetIO instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (memptr_gpu == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid memptr_gpu provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    std::unordered_map<uint64_t, struct doca_gpu_mtable *>::const_iterator it =
        gpu_dev->mtable->find((uintptr_t)memptr_gpu);
    if (it == gpu_dev->mtable->end()) {
        DOCA_LOG(LOG_ERR, "memptr_gpu = %p was not allocated by DOCA GPUNetIO.", memptr_gpu);
        return DOCA_ERROR_INVALID_VALUE;
    }

    mentry = it->second;

    if (mentry->mtype == DOCA_GPU_MEM_TYPE_GPU)
        DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaFree((void *)mentry->base_addr));
    else if (mentry->mtype == DOCA_GPU_MEM_TYPE_GPU_CPU) {
        if (gpu_dev->support_gdrcopy)
            doca_gpu_gdrcopy_destroy_mapping(mentry->gdr_mh, (void *)mentry->align_addr_cpu,
                                             mentry->size);
        DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaFree((void *)mentry->base_addr));
    } else {
        res_cuda = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostUnregister((void *)mentry->base_addr));
        if (res_cuda != cudaSuccess)
            DOCA_LOG(LOG_ERR, "Error unregistering GPU memory at %p", (void *)mentry->base_addr);
        free((void *)mentry->base_addr);
    }

    gpu_dev->mtable->erase(it);
    free(mentry);

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_dmabuf_fd(struct doca_gpu *gpu_dev, void *memptr_gpu, size_t size,
                                int *dmabuf_fd) {
#if DOCA_GPUNETIO_HAVE_CUDA_DMABUF == 1
    CUresult res_drv = CUDA_SUCCESS;

    if (gpu_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA GPUNetIO instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (gpu_dev->support_dmabuf == false) {
        DOCA_LOG(LOG_ERR, "DMABuf not supported on this system by this CUDA installation.");
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    if (dmabuf_fd == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DMABuf fd pointer provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    res_drv = doca_verbs_wrapper_cuMemGetHandleForAddressRange(
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

    if (db == nullptr || out_can_register == nullptr) return DOCA_ERROR_INVALID_VALUE;

    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostRegister(
        db, DOCA_VERBS_DB_UAR_SIZE,
        cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory));

    *out_can_register =
        (cuda_status == cudaSuccess || cuda_status == cudaErrorHostMemoryAlreadyRegistered);

    if (cuda_status == cudaSuccess) DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostUnregister(db));

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

doca_error_t doca_gpu_verbs_export_qp(struct doca_gpu *gpu_dev, struct doca_verbs_qp *qp,
                                      enum doca_gpu_dev_verbs_nic_handler nic_handler,
                                      void *gpu_qp_umem_dev_ptr, struct doca_verbs_cq *cq_sq,
                                      struct doca_gpu_verbs_qp **qp_out) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct doca_gpu_dev_verbs_qp *qp_cpu_ = nullptr;
    void *rq_wqe_daddr;
    uint32_t rq_wqe_num;
    uint32_t rcv_wqe_size;
    uint64_t *sq_db;
    uint32_t sq_wqe_num;
    uint64_t *uar_db_reg = NULL;
    uint32_t *arm_dbr = NULL;
    uint32_t *cq_dbrec;

    if (gpu_dev == nullptr || qp == nullptr || qp == nullptr || cq_sq == nullptr)
        return DOCA_ERROR_INVALID_VALUE;

    *qp_out = (struct doca_gpu_verbs_qp *)calloc(1, sizeof(struct doca_gpu_verbs_qp));
    if (*qp_out == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to allocate CPU memory");
        return DOCA_ERROR_NO_MEMORY;
    }

    (*qp_out)->qp_cpu =
        (struct doca_gpu_dev_verbs_qp *)calloc(1, sizeof(struct doca_gpu_dev_verbs_qp));
    if ((*qp_out)->qp_cpu == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to allocate CPU memory");
        free(*qp_out);
        return DOCA_ERROR_NO_MEMORY;
    }

    qp_cpu_ = (*qp_out)->qp_cpu;

    // Should this be propagated to GPU?
    if (qp->get_uar_mtype() == DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME)
        gpu_dev->support_bf_uar = true;

    // Check QP and CQ same size!!!!

    doca_verbs_qp_get_wq(qp,
                         (void **)&(qp_cpu_->sq_wqe_daddr),  // broken for external umem
                         &sq_wqe_num,
                         (void **)&(rq_wqe_daddr),  // broken for external umem
                         &rq_wqe_num, &rcv_wqe_size);

    uint32_t *dbrec = reinterpret_cast<uint32_t *>(doca_verbs_qp_get_dbr_addr(qp));

    qp_cpu_->sq_wqe_num = (uint16_t)sq_wqe_num;
    qp_cpu_->sq_wqe_mask = qp_cpu_->sq_wqe_num - 1;
    qp_cpu_->sq_num = doca_verbs_qp_get_qpn(qp);
    qp_cpu_->sq_num_shift8 = qp_cpu_->sq_num << 8;
    qp_cpu_->sq_num_shift8_be = htobe32(qp_cpu_->sq_num_shift8);
    qp_cpu_->sq_num_shift8_be_1ds = htobe32(qp_cpu_->sq_num_shift8 | 1);
    qp_cpu_->sq_num_shift8_be_2ds = htobe32(qp_cpu_->sq_num_shift8 | 2);
    qp_cpu_->sq_num_shift8_be_3ds = htobe32(qp_cpu_->sq_num_shift8 | 3);
    qp_cpu_->sq_num_shift8_be_4ds = htobe32(qp_cpu_->sq_num_shift8 | 4);
    qp_cpu_->sq_wqe_pi = 0;
    qp_cpu_->sq_rsvd_index = 0;
    qp_cpu_->sq_ready_index = 0;
    qp_cpu_->sq_lock = 0;
    qp_cpu_->sq_dbrec = (__be32 *)(dbrec + DOCA_GPUNETIO_IB_MLX5_SND_DBR);
    qp_cpu_->mem_type = DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU;
    (*qp_out)->cpu_db = nullptr;
    (*qp_out)->sq_db = nullptr;
    (*qp_out)->sq_wqe_pi_last = 0;
    (*qp_out)->cpu_proxy = false;
    (*qp_out)->qp_gpu = nullptr;
    (*qp_out)->qp = qp;

    sq_db = reinterpret_cast<uint64_t *>(doca_verbs_qp_get_uar_addr(qp));

    if (nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
        status = doca_gpu_verbs_export_uar(sq_db, (uint64_t **)&(qp_cpu_->sq_db));
        if (status != DOCA_SUCCESS && nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
            DOCA_LOG(LOG_ERR, "Can't export UAR to GPU.");
            goto destroy_uar;
        }
    }

    if ((status != DOCA_SUCCESS && nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) ||
        nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
        DOCA_LOG(LOG_WARNING, "Enabling CPU proxy mode");

        status = doca_gpu_mem_alloc(gpu_dev, sizeof(uint64_t), priv_get_page_size(),
                                    DOCA_GPU_MEM_TYPE_CPU_GPU, (void **)&((*qp_out)->cpu_db),
                                    (void **)&((*qp_out)->cpu_db));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to alloc GPU memory for CPU proxy DB");
            goto destroy_uar;
        }

        *((*qp_out)->cpu_db) = 0;
        qp_cpu_->sq_db = (*qp_out)->cpu_db;
        (*qp_out)->sq_dbrec = qp_cpu_->sq_dbrec;
        (*qp_out)->sq_db = reinterpret_cast<uint64_t *>(doca_verbs_qp_get_uar_addr(qp));
        (*qp_out)->cpu_proxy = true;
        (*qp_out)->sq_num_shift8_be = qp_cpu_->sq_num_shift8_be;
        qp_cpu_->nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
    } else {
        qp_cpu_->nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
    }

    doca_verbs_cq_get_wq(cq_sq, (void **)&(qp_cpu_->cq_sq.cqe_daddr), &(qp_cpu_->cq_sq.cqe_num),
                         &(qp_cpu_->cq_sq.cqe_size));

    doca_verbs_cq_get_dbr_addr(cq_sq, &uar_db_reg, (uint32_t **)&(cq_dbrec), &arm_dbr);

    qp_cpu_->cq_sq.dbrec = (__be32 *)cq_dbrec;
    qp_cpu_->cq_sq.cq_num = doca_verbs_cq_get_cqn(cq_sq);
    qp_cpu_->cq_sq.cqe_mask = (qp_cpu_->cq_sq.cqe_num - 1);
    qp_cpu_->cq_sq.cqe_ci = 0;
    qp_cpu_->cq_sq.cqe_rsvd = 0;
    qp_cpu_->cq_sq.mem_type = DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU;

    (*qp_out)->gpu_dev = gpu_dev;

    return DOCA_SUCCESS;

destroy_uar:
    if (nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
        tmp_status = doca_gpu_verbs_unexport_uar((*qp_out)->qp_cpu->sq_db);
        if (tmp_status != DOCA_SUCCESS)
            DOCA_LOG(LOG_ERR, "Failed to destroy GPU descriptor memory");
    }

    free((*qp_out)->qp_cpu);
    free(*qp_out);

    return status;
}

doca_error_t doca_gpu_verbs_get_qp_dev(struct doca_gpu_verbs_qp *qp,
                                       struct doca_gpu_dev_verbs_qp **qp_gpu) {
    doca_error_t status = DOCA_SUCCESS;
    int custatus = 0;

    if (qp == nullptr) return DOCA_ERROR_INVALID_VALUE;

    if (qp->qp_gpu == nullptr) {
        status = doca_gpu_mem_alloc(qp->gpu_dev, sizeof(struct doca_gpu_dev_verbs_qp),
                                    priv_get_page_size(), DOCA_GPU_MEM_TYPE_GPU,
                                    (void **)&qp->qp_gpu, nullptr);
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
            return DOCA_ERROR_DRIVER;
        }
    }

    *qp_gpu = qp->qp_gpu;

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_unexport_qp(struct doca_gpu *gpu_dev,
                                        struct doca_gpu_verbs_qp *qp_gverbs) {
    if (gpu_dev == nullptr || qp_gverbs == nullptr) return DOCA_ERROR_INVALID_VALUE;

    if (qp_gverbs->cpu_db) doca_gpu_mem_free(gpu_dev, qp_gverbs->cpu_db);

    if (qp_gverbs->qp_cpu) {
        if (qp_gverbs->qp_cpu->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY)
            doca_gpu_verbs_unexport_uar(qp_gverbs->qp_cpu->sq_db);
        free(qp_gverbs->qp_cpu);
    }

    if (qp_gverbs->qp_gpu) {
        doca_gpu_mem_free(gpu_dev, qp_gverbs->qp_gpu);
        qp_gverbs->qp_gpu = nullptr;
    }

    free(qp_gverbs);

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_cpu_proxy_progress(struct doca_gpu_verbs_qp *qp_cpu) {
    uint32_t tmp_db = 0;
    __be32 dbr_val;

    if (qp_cpu == nullptr) return DOCA_ERROR_INVALID_VALUE;

    if (qp_cpu->cpu_proxy != true) return DOCA_ERROR_NOT_SUPPORTED;

    tmp_db = (uint32_t) * ((volatile uint64_t *)qp_cpu->cpu_db);
    if (tmp_db != qp_cpu->sq_wqe_pi_last) {
        struct doca_gpu_dev_verbs_wqe_ctrl_seg ctrl_seg = {.opmod_idx_opcode = htobe32(tmp_db << 8),
                                                           .qpn_ds = qp_cpu->sq_num_shift8_be};

        dbr_val = htobe32(tmp_db & 0xffff);

        // Ring the DB ASAP.
        // The second DB ringing happens after the fence. This is used when the NIC enters a
        // recovery state and it needs to read DBR.
        *((volatile uint32_t *)qp_cpu->sq_dbrec) = dbr_val;
        std::atomic_thread_fence(std::memory_order_release);
        *((volatile uint64_t *)qp_cpu->sq_db) = *((volatile uint64_t *)&ctrl_seg);

        // DOCA_LOG(LOG_DEBUG, "CPU proxy ring wqe %d\n", tmp_db);
        qp_cpu->sq_wqe_pi_last = tmp_db;
    }

    return DOCA_SUCCESS;
}

static void *priv_service_mainloop(void *args) {
    struct doca_gpu_verbs_service *service = (struct doca_gpu_verbs_service *)args;
    const unsigned int num_loops = 1000;

    while (service->running) {
        pthread_rwlock_rdlock(&service->service_lock);
        for (unsigned int i = 0; i < num_loops; i++) {
            for (auto qp : *service->qps) {
                doca_gpu_verbs_cpu_proxy_progress(qp);
            }
        }
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

    if (qp == nullptr || qp->qp == nullptr || error_info == nullptr)
        return DOCA_ERROR_INVALID_VALUE;

    memset(error_info, 0, sizeof(struct doca_gpu_verbs_qp_error_info));

    struct doca_verbs_qp_attr qp_attr;
    struct doca_verbs_qp_init_attr qp_init_attr;
    status = doca_verbs_qp_query(qp->qp, &qp_attr, &qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query QP");
        return status;
    }

    error_info->has_error = (qp_attr.current_state == DOCA_VERBS_QP_STATE_ERR);

    return DOCA_SUCCESS;
}
