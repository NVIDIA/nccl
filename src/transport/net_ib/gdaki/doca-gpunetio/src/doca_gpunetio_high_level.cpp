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

#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <atomic>
#include <time.h>
#include <unordered_map>
#include <cuda_runtime.h>

#include "host/mlx5_prm.h"
#include "host/mlx5_ifc.h"

#include "doca_verbs_net_wrapper.h"
#include "doca_internal.hpp"
#include "host/doca_gpunetio_high_level.h"
#include "doca_gpunetio_gdrcopy.h"
#include "host/doca_verbs.h"
#include "doca_verbs_qp.hpp"
#include "common/doca_gpunetio_verbs_dev.h"
#include "doca_gpunetio.hpp"

#define DBR_SIZE (8)
#define MAX_SEND_SEGS (1)
#define MAX_RECEIVE_SEGS (1)

static size_t priv_get_page_size() {
    auto ret = sysconf(_SC_PAGESIZE);
    if (ret == -1) return 4096;  // 4KB, default Linux page size

    return (size_t)ret;
}

static uint32_t align_up_uint32(uint32_t value, uint32_t alignment) {
    uint64_t remainder = (value % alignment);

    if (remainder == 0) return value;

    return (uint32_t)(value + (alignment - remainder));
}

static bool cq_type_is_collapsed(enum doca_gpu_dev_verbs_cq_type cq_type) {
    return (cq_type & DOCA_GPUNETIO_VERBS_CQ_FLAG_COLLAPSED) != 0;
}

static enum doca_gpu_dev_verbs_cq_type resolve_cq_type(
    const struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr) {
    if (qp_init_attr->cq_type != DOCA_GPUNETIO_VERBS_CQ_UNKNOWN) return qp_init_attr->cq_type;

    return qp_init_attr->cq_collapsed ? DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED
                                      : DOCA_GPUNETIO_VERBS_CQ_64B;
}

static doca_error_t validate_cq_type(const struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr) {
    switch (qp_init_attr->cq_type) {
        case DOCA_GPUNETIO_VERBS_CQ_UNKNOWN:
        case DOCA_GPUNETIO_VERBS_CQ_64B:
        case DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED:
        case DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST:
            break;
        default:
            DOCA_LOG(LOG_ERR, "Invalid CQ type %d", qp_init_attr->cq_type);
            return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_type_is_collapsed(qp_init_attr->cq_type) && !qp_init_attr->cq_collapsed) {
        DOCA_LOG(LOG_ERR, "Collapsed CQ type requires cq_collapsed to be true");
        return DOCA_ERROR_INVALID_VALUE;
    }

    return DOCA_SUCCESS;
}

static doca_error_t create_uar(doca_dev_t *net_dev, enum doca_gpu_dev_verbs_nic_handler nic_handler,
                               doca_verbs_uar_t **external_uar) {
    doca_error_t status = DOCA_SUCCESS;

    if (nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
        status = doca_verbs_uar_create(net_dev, DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED,
                                       external_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to doca_verbs_uar_create NC DEDICATED");
            status = doca_verbs_uar_create(net_dev, DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE,
                                           external_uar);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to doca_verbs_uar_create NC");
            } else {
                DOCA_LOG(LOG_INFO, "UAR created with DOCA_UAR_ALLOCATION_TYPE_NONCACHE");
            }
            return DOCA_SUCCESS;
        } else
            return DOCA_SUCCESS;
    }

    if ((nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) ||
        (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO && status != DOCA_SUCCESS)) {
        status =
            doca_verbs_uar_create(net_dev, DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME, external_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to doca_verbs_uar_create NC");
            return status;
        }
    } else
        return DOCA_ERROR_DRIVER;

    return status;
}

static doca_error_t create_gpu_umem(doca_gpu_t *gpu_dev, doca_dev_t *net_dev,
                                    enum doca_gpu_verbs_mem_reg_type mreg_type, size_t umem_sz,
                                    void *umem_ptr, doca_verbs_umem_t **umem) {
    doca_error_t status;
    int dmabuf_fd = DOCA_VERBS_DMABUF_INVALID_FD;

    if (mreg_type == DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT) {
        status = doca_gpu_get_dmabuf_fd(gpu_dev, umem_ptr, umem_sz, &dmabuf_fd);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING,
                     "GPU doesn't support dmabuf, fallback to legacy nvidia-peermem mode");
            dmabuf_fd = DOCA_VERBS_DMABUF_INVALID_FD;
        }

        status = doca_verbs_umem_create(net_dev, gpu_dev, umem_ptr, umem_sz, IBV_ACCESS_LOCAL_WRITE,
                                        dmabuf_fd, 0, umem);
        if (status != DOCA_SUCCESS) {
            if (dmabuf_fd > 0) {
                DOCA_LOG(LOG_WARNING,
                         "Failed to create gpu umem with dmabuf. Fallback to legacy nvidia-peermem "
                         "mode");
                status = doca_verbs_umem_create(net_dev, gpu_dev, umem_ptr, umem_sz,
                                                IBV_ACCESS_LOCAL_WRITE,
                                                DOCA_VERBS_DMABUF_INVALID_FD, 0, umem);
                if (status != DOCA_SUCCESS) {
                    DOCA_LOG(LOG_ERR, "Failed to create gpu umem with nvidia-peermem mode");
                    goto destroy_resources;
                }
            } else {
                DOCA_LOG(LOG_ERR, "Failed to create gpu umem");
                goto destroy_resources;
            }
        }
    } else if (mreg_type == DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_DMABUF) {
        status = doca_gpu_get_dmabuf_fd(gpu_dev, umem_ptr, umem_sz, &dmabuf_fd);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "GPU doesn't support dmabuf.");
            goto destroy_resources;
        }

        status = doca_verbs_umem_create(net_dev, gpu_dev, umem_ptr, umem_sz, IBV_ACCESS_LOCAL_WRITE,
                                        dmabuf_fd, 0, umem);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "GPU doesn't support dmabuf.");
            goto destroy_resources;
        }
    } else if (mreg_type == DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_PEERMEM) {
        status = doca_verbs_umem_create(net_dev, gpu_dev, umem_ptr, umem_sz, IBV_ACCESS_LOCAL_WRITE,
                                        DOCA_VERBS_DMABUF_INVALID_FD, 0, umem);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create gpu umem with nvidia-peermem mode");
            goto destroy_resources;
        }
    }

    // Immediately close dmabuf_fd after registration.
    if (dmabuf_fd > 0 && dmabuf_fd != (int)DOCA_VERBS_DMABUF_INVALID_FD) close(dmabuf_fd);

    return DOCA_SUCCESS;

destroy_resources:
    if (*umem) doca_verbs_umem_destroy(*umem);

    return status;
}

static uint32_t calc_cq_external_umem_size(uint32_t queue_size, uint32_t dbr_size) {
    uint32_t cqe_buf_size = 0;

    if (queue_size != 0)
        cqe_buf_size =
            (uint32_t)(queue_size * sizeof(struct doca_gpunetio_ib_mlx5_cqe64) + dbr_size);

    return align_up_uint32(cqe_buf_size, priv_get_page_size());
}

static void mlx5_init_cqes(struct doca_gpunetio_ib_mlx5_cqe64 *cqes, uint32_t nb_cqes) {
    for (uint32_t cqe_idx = 0; cqe_idx < nb_cqes; cqe_idx++)
        cqes[cqe_idx].op_own =
            (DOCA_GPUNETIO_IB_MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) |
            DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK;
}

static doca_error_t create_umem_hl(doca_gpu_t *gpu_dev, doca_dev_t *net_dev,
                                   enum doca_gpu_verbs_mem_reg_type mreg_type, size_t size,
                                   bool is_host_mem, bool enable_umem_cpu,
                                   struct doca_gpu_verbs_umem_hl **out_umem) {
    doca_error_t status = DOCA_SUCCESS;
    struct doca_gpu_verbs_umem_hl *umem_hl = NULL;
    enum doca_gpu_verbs_mem_reg_type reg_type = mreg_type;
    void *registered_ptr = NULL;

    umem_hl = (struct doca_gpu_verbs_umem_hl *)calloc(1, sizeof(struct doca_gpu_verbs_umem_hl));
    if (umem_hl == NULL) {
        DOCA_LOG(LOG_ERR, "Failed to allocate umem_hl");
        return DOCA_ERROR_NO_MEMORY;
    }

    if (is_host_mem) {
        umem_hl->base_cpu_ptr = calloc(size, sizeof(uint8_t));
        if (umem_hl->base_cpu_ptr == NULL) {
            DOCA_LOG(LOG_ERR, "Failed to allocate host memory for umem_hl slab");
            free(umem_hl);
            return DOCA_ERROR_NO_MEMORY;
        }
        umem_hl->base_gpu_ptr = umem_hl->base_cpu_ptr;
        reg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_PEERMEM;
        registered_ptr = umem_hl->base_cpu_ptr;
    } else if (enable_umem_cpu) {
        status = doca_gpu_mem_alloc(gpu_dev, size, priv_get_page_size(), DOCA_GPU_MEM_TYPE_CPU_GPU,
                                    &umem_hl->base_gpu_ptr, &umem_hl->base_cpu_ptr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to alloc CPU-GPU memory for umem_hl slab");
            free(umem_hl);
            return status;
        }
        reg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_PEERMEM;
        registered_ptr = umem_hl->base_cpu_ptr;
    } else {
        status = doca_gpu_mem_alloc(gpu_dev, size, priv_get_page_size(), DOCA_GPU_MEM_TYPE_GPU,
                                    &umem_hl->base_gpu_ptr, NULL);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to alloc GPU memory for umem_hl slab");
            free(umem_hl);
            return status;
        }
        umem_hl->base_cpu_ptr = NULL;
        registered_ptr = umem_hl->base_gpu_ptr;
    }

    status = create_gpu_umem(gpu_dev, net_dev, reg_type, size, registered_ptr, &umem_hl->umem);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "create_gpu_umem failed for umem_hl slab");
        if (is_host_mem) {
            free(umem_hl->base_cpu_ptr);
        } else {
            doca_gpu_mem_free(gpu_dev, umem_hl->base_gpu_ptr);
        }
        free(umem_hl);
        return status;
    }

    umem_hl->size = size;
    umem_hl->next_offset = 0;
    umem_hl->refcount = 0;
    umem_hl->mreg_type = reg_type;

    *out_umem = umem_hl;
    return DOCA_SUCCESS;
}

static void destroy_umem_hl(doca_gpu_t *gpu_dev, struct doca_gpu_verbs_umem_hl *umem_hl) {
    if (umem_hl == NULL) return;

    if (umem_hl->umem != NULL) {
        doca_verbs_umem_destroy(umem_hl->umem);
    }

    if (umem_hl->base_cpu_ptr != NULL && umem_hl->base_cpu_ptr == umem_hl->base_gpu_ptr) {
        free(umem_hl->base_cpu_ptr);
    } else if (umem_hl->base_gpu_ptr != NULL) {
        doca_gpu_mem_free(gpu_dev, umem_hl->base_gpu_ptr);
    }

    memset(umem_hl, 0, sizeof(*umem_hl));
    free(umem_hl);
}

static doca_error_t suballoc_from_umem_hl(struct doca_gpu_verbs_umem_hl *umem_hl, size_t alloc_size,
                                          void **out_gpu_ptr, void **out_cpu_ptr,
                                          size_t *out_offset) {
    if (umem_hl->next_offset + alloc_size > umem_hl->size) {
        DOCA_LOG(LOG_ERR, "umem_hl slab overflow: next_offset %zu + alloc_size %zu > size %zu",
                 umem_hl->next_offset, alloc_size, umem_hl->size);
        return DOCA_ERROR_NO_MEMORY;
    }

    *out_offset = umem_hl->next_offset;
    *out_gpu_ptr = (void *)((uintptr_t)umem_hl->base_gpu_ptr + umem_hl->next_offset);
    if (out_cpu_ptr != NULL) {
        *out_cpu_ptr = umem_hl->base_cpu_ptr != NULL
                           ? (void *)((uintptr_t)umem_hl->base_cpu_ptr + umem_hl->next_offset)
                           : NULL;
    }

    umem_hl->next_offset += alloc_size;
    umem_hl->refcount++;

    return DOCA_SUCCESS;
}

static doca_error_t create_cq(doca_gpu_t *gpu_dev, doca_dev_t *net_dev, struct ibv_pd *ibpd,
                              enum doca_gpu_verbs_mem_reg_type mreg_type, uint32_t ncqes,
                              void **umem_dev_ptr, doca_verbs_umem_t **gpu_umem,
                              doca_verbs_uar_t *external_uar, bool cq_collapsed,
                              bool enable_umem_cpu, doca_verbs_comp_channel_t *comp_channel,
                              doca_verbs_cq_t **verbs_cq,
                              struct doca_gpu_verbs_umem_hl *shared_cq_umem = NULL,
                              struct doca_gpu_verbs_umem_hl *shared_cq_dbr_umem = NULL) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    cudaError_t status_cuda = cudaSuccess;
    doca_verbs_cq_attr_t *verbs_cq_attr = NULL;
    doca_verbs_cq_t *new_cq = NULL;
    struct doca_gpunetio_ib_mlx5_cqe64 *cq_ring_haddr = NULL;
    uint32_t external_umem_size = 0;
    void *gpu_umem_dev_ptr = NULL;
    void *cpu_umem_dev_ptr = NULL;
    doca_verbs_umem_t *cq_umem_to_use = NULL;
    uint64_t cq_umem_offset = 0;

    *umem_dev_ptr = nullptr;
    *gpu_umem = nullptr;

    status = doca_verbs_cq_attr_create(&verbs_cq_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs cq attributes");
        return status;
    }

    if (shared_cq_umem != NULL) {
        /* Shared slab path: suballocate from pre-allocated, pre-initialized slab */
        void *sub_gpu_ptr = NULL;
        void *sub_cpu_ptr = NULL;
        size_t sub_offset = 0;

        /* Suballocate CQ DBR from the shared CQ DBR slab */
        if (shared_cq_dbr_umem != NULL) {
            void *sub_dbr_ptr = NULL;
            size_t sub_dbr_offset = 0;
            size_t dbr_alloc_sz = align_up_uint32(DBR_SIZE, priv_get_page_size());
            status = suballoc_from_umem_hl(shared_cq_dbr_umem, dbr_alloc_sz, &sub_dbr_ptr, NULL,
                                           &sub_dbr_offset);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to suballocate CQ DBR slice from shared slab");
                goto destroy_resources;
            }

            status = doca_verbs_cq_attr_set_external_dbr_umem(
                verbs_cq_attr, shared_cq_dbr_umem->umem, (uint64_t)sub_dbr_offset);
        }

        // If doca_verbs_cq_attr_set_external_dbr_umem symbol is not present in DOCA SDK
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "Failed to set CQ external DBR umem. Embedding DBR in CQ UMEM");
            external_umem_size = calc_cq_external_umem_size(ncqes, DBR_SIZE);

            status = suballoc_from_umem_hl(shared_cq_umem, external_umem_size, &sub_gpu_ptr,
                                           &sub_cpu_ptr, &sub_offset);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to suballocate CQ slice from shared slab");
                goto destroy_resources;
            }

            shared_cq_dbr_umem = shared_cq_umem;
        } else {
            external_umem_size = calc_cq_external_umem_size(ncqes, 0);

            status = suballoc_from_umem_hl(shared_cq_umem, external_umem_size, &sub_gpu_ptr,
                                           &sub_cpu_ptr, &sub_offset);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to suballocate CQ slice from shared slab");
                goto destroy_resources;
            }
        }

        *umem_dev_ptr = (sub_cpu_ptr != NULL) ? sub_cpu_ptr : sub_gpu_ptr;
        *gpu_umem = NULL; /* slab owns the umem; caller must not free it */
        cq_umem_to_use = shared_cq_umem->umem;
        cq_umem_offset = (uint64_t)sub_offset;

    } else {
        external_umem_size = calc_cq_external_umem_size(ncqes, DBR_SIZE);
        if (enable_umem_cpu) {
            status = doca_gpu_mem_alloc(gpu_dev, external_umem_size, priv_get_page_size(),
                                        DOCA_GPU_MEM_TYPE_CPU_GPU, (void **)(&gpu_umem_dev_ptr),
                                        (void **)(&cpu_umem_dev_ptr));
            *umem_dev_ptr = cpu_umem_dev_ptr;
        } else {
            status = doca_gpu_mem_alloc(gpu_dev, external_umem_size, priv_get_page_size(),
                                        DOCA_GPU_MEM_TYPE_GPU, (void **)(&gpu_umem_dev_ptr), NULL);
            *umem_dev_ptr = gpu_umem_dev_ptr;
        }

        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to alloc gpu memory for external umem cq");
            goto destroy_resources;
        }

        if (enable_umem_cpu) {
            status =
                create_gpu_umem(gpu_dev, net_dev, DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_PEERMEM,
                                external_umem_size, *umem_dev_ptr, gpu_umem);
        } else {
            status = create_gpu_umem(gpu_dev, net_dev, mreg_type, external_umem_size, *umem_dev_ptr,
                                     gpu_umem);
        }
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "create umem for CQ failed with %d", status);
            goto destroy_resources;
        }

        cq_umem_to_use = *gpu_umem;
        cq_umem_offset = 0;
    }

    /* Initialize all CQEs with INVALID value. */
    {
        cq_ring_haddr =
            (struct doca_gpunetio_ib_mlx5_cqe64 *)calloc(external_umem_size, sizeof(uint8_t));
        if (cq_ring_haddr == NULL) {
            status = DOCA_ERROR_NO_MEMORY;
            goto destroy_resources;
        }

        mlx5_init_cqes(cq_ring_haddr, ncqes);

        if (enable_umem_cpu) {
            memcpy(*umem_dev_ptr, (void *)(cq_ring_haddr), external_umem_size * sizeof(uint8_t));
        } else {
            status_cuda = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaMemcpy(
                (*umem_dev_ptr), (void *)(cq_ring_haddr), external_umem_size, cudaMemcpyDefault));
            if (status_cuda != cudaSuccess) {
                DOCA_LOG(LOG_ERR, "Failed to cudaMempy gpu cq cq ring buffer ret %d", status_cuda);
                goto destroy_resources;
            }
        }

        free(cq_ring_haddr);
        cq_ring_haddr = nullptr;

        if (status_cuda != cudaSuccess) {
            status = DOCA_ERROR_DRIVER;
            goto destroy_resources;
        }
    }

    status = doca_verbs_cq_attr_set_external_umem(verbs_cq_attr, cq_umem_to_use, cq_umem_offset);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs cq external umem");
        goto destroy_resources;
    }

    status = doca_verbs_cq_attr_set_cq_size(verbs_cq_attr, ncqes);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs cq size");
        goto destroy_resources;
    }

    status = doca_verbs_cq_attr_set_cq_overrun(verbs_cq_attr, DOCA_VERBS_CQ_ENABLE_OVERRUN);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs cq size");
        goto destroy_resources;
    }

    if (cq_collapsed == true) {
        status = doca_verbs_cq_attr_set_cq_collapsed(verbs_cq_attr, 1);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to set doca verbs cq collapsed");
            goto destroy_resources;
        }
    }

    if (external_uar != NULL) {
        status = doca_verbs_cq_attr_set_external_uar(verbs_cq_attr, external_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to set doca verbs cq external uar");
            goto destroy_resources;
        }
    }

    if (comp_channel) {
        status = doca_verbs_cq_attr_set_comp_channel(verbs_cq_attr, comp_channel);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to set doca verbs cq comp_channel");
            goto destroy_resources;
        }

        status = doca_verbs_cq_attr_set_st(verbs_cq_attr, DOCA_VERBS_CQ_ST_NO_ACTION);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to set doca verbs cq st");
            goto destroy_resources;
        }
    }

    status = doca_verbs_cq_create(net_dev, verbs_cq_attr, &new_cq);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs cq");
        goto destroy_resources;
    }

    status = doca_verbs_cq_attr_destroy(verbs_cq_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs cq attributes");
        goto destroy_resources;
    }

    *verbs_cq = new_cq;

    return DOCA_SUCCESS;

destroy_resources:
    if (new_cq != NULL) {
        tmp_status = doca_verbs_cq_destroy(new_cq);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs cq");
    }

    if (verbs_cq_attr != NULL) {
        tmp_status = doca_verbs_cq_attr_destroy(verbs_cq_attr);
        if (tmp_status != DOCA_SUCCESS)
            DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs cq attributes");
    }

    if (shared_cq_umem == NULL) {
        /* Only free per-QP resources; shared slab is freed by the list destructor */
        if (*gpu_umem != NULL) {
            tmp_status = doca_verbs_umem_destroy(*gpu_umem);
            if (tmp_status != DOCA_SUCCESS)
                DOCA_LOG(LOG_ERR, "Failed to destroy gpu ring buffer umem");
        }

        if (cq_ring_haddr) {
            free(cq_ring_haddr);
        }

        if (gpu_umem_dev_ptr != 0) {
            tmp_status = doca_gpu_mem_free(gpu_dev, gpu_umem_dev_ptr);
            if (tmp_status != DOCA_SUCCESS)
                DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of cq umem buffer");
            *umem_dev_ptr = 0;
        }
    }

    if (cq_ring_haddr) free(cq_ring_haddr);

    return status;
}

static uint32_t calc_qp_external_umem_size(uint32_t sq_nwqes) {
    uint32_t sq_ring_size = 0;

    if (sq_nwqes != 0) sq_ring_size = (uint32_t)(sq_nwqes * sizeof(struct doca_gpu_dev_verbs_wqe));

    return align_up_uint32(sq_ring_size, priv_get_page_size());
}

static doca_error_t create_qp(doca_gpu_t *gpu_dev, doca_dev_t *net_dev, struct ibv_pd *ibpd,
                              enum doca_gpu_verbs_mem_reg_type mreg_type, doca_verbs_cq_t *cq_sq,
                              uint32_t sq_nwqe, void **umem_dev_ptr, doca_verbs_umem_t **gpu_umem,
                              void **umem_dbr_dev_ptr, doca_verbs_umem_t **gpu_umem_dbr,
                              doca_verbs_uar_t *external_uar,
                              enum doca_gpu_dev_verbs_nic_handler req_nic_handler,
                              bool set_core_direct,
                              enum doca_gpu_verbs_send_dbr_mode_ext send_dbr_mode_ext,
                              enum doca_verbs_qp_ordering_semantic ordering_semantic,
                              bool enable_umem_cpu, doca_verbs_qp_t **verbs_qp,
                              enum doca_gpu_dev_verbs_nic_handler *out_nic_handler,
                              struct doca_gpu_verbs_umem_hl *shared_sq_umem = NULL,
                              struct doca_gpu_verbs_umem_hl *shared_sq_dbr_umem = NULL) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    doca_verbs_qp_init_attr_t *qp_init_attr = nullptr;
    doca_verbs_qp_t *new_qp = nullptr;
    uint32_t external_umem_size = 0;
    size_t dbr_umem_align_sz = align_up_uint32(DBR_SIZE, priv_get_page_size());
    enum doca_gpu_dev_verbs_nic_handler nic_handler = req_nic_handler;
    void *gpu_umem_dev_ptr = NULL;
    void *cpu_umem_dev_ptr = NULL;
    void *gpu_umem_dbr_dev_ptr = NULL;
    void *cpu_umem_dbr_dev_ptr = NULL;

    *umem_dev_ptr = nullptr;
    *umem_dbr_dev_ptr = nullptr;
    *gpu_umem = nullptr;
    *gpu_umem_dbr = nullptr;

    status = doca_verbs_qp_init_attr_create(&qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs qp attributes");
        return status;
    }

    status = doca_verbs_qp_init_attr_set_external_uar(qp_init_attr, external_uar);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set receive_max_sges");
        goto destroy_resources;
    }

    if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
        bool can_register = false;
        status =
            doca_gpu_verbs_can_gpu_register_uar(external_uar->open->get_reg_addr(), &can_register);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to check if UAR can be registered on GPU");
            goto destroy_resources;
        }

        nic_handler = can_register ? DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB
                                   : DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
    }

    if (send_dbr_mode_ext > DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR &&
        nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
        nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR;
    }

    external_umem_size = calc_qp_external_umem_size(sq_nwqe);

    if (shared_sq_umem != NULL) {
        void *sub_gpu_ptr = NULL;
        void *sub_cpu_ptr = NULL;
        size_t sub_offset = 0;
        status = suballoc_from_umem_hl(shared_sq_umem, external_umem_size, &sub_gpu_ptr,
                                       &sub_cpu_ptr, &sub_offset);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to suballocate SQ slice from shared slab");
            goto destroy_resources;
        }
        *umem_dev_ptr = (sub_cpu_ptr != NULL) ? sub_cpu_ptr : sub_gpu_ptr;
        *gpu_umem = NULL;
        status = doca_verbs_qp_init_attr_set_external_umem(qp_init_attr, shared_sq_umem->umem,
                                                           (uint64_t)sub_offset);
    } else {
        if (enable_umem_cpu) {
            status = doca_gpu_mem_alloc(gpu_dev, external_umem_size, priv_get_page_size(),
                                        DOCA_GPU_MEM_TYPE_CPU_GPU, (void **)(&gpu_umem_dev_ptr),
                                        (void **)(&cpu_umem_dev_ptr));
            *umem_dev_ptr = cpu_umem_dev_ptr;
        } else {
            status = doca_gpu_mem_alloc(gpu_dev, external_umem_size, priv_get_page_size(),
                                        DOCA_GPU_MEM_TYPE_GPU, (void **)(&gpu_umem_dev_ptr), NULL);
            *umem_dev_ptr = gpu_umem_dev_ptr;
        }

        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to alloc external umem qp");
            goto destroy_resources;
        }

        if (enable_umem_cpu) {
            status =
                create_gpu_umem(gpu_dev, net_dev, DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_PEERMEM,
                                external_umem_size, *umem_dev_ptr, gpu_umem);
        } else {
            status = create_gpu_umem(gpu_dev, net_dev, mreg_type, external_umem_size, *umem_dev_ptr,
                                     gpu_umem);
        }
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "create umem for QP failed with %d", status);
            goto destroy_resources;
        }

        status = doca_verbs_qp_init_attr_set_external_umem(qp_init_attr, *gpu_umem, 0);
    }
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs qp external umem");
        goto destroy_resources;
    }

    if (shared_sq_dbr_umem != NULL) {
        void *sub_dbr_ptr = NULL;
        void *sub_dbr_cpu_ptr = NULL;
        size_t sub_dbr_offset = 0;
        status = suballoc_from_umem_hl(shared_sq_dbr_umem, dbr_umem_align_sz, &sub_dbr_ptr,
                                       &sub_dbr_cpu_ptr, &sub_dbr_offset);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to suballocate DBR slice from shared slab");
            goto destroy_resources;
        }
        *umem_dbr_dev_ptr = (sub_dbr_cpu_ptr != NULL) ? sub_dbr_cpu_ptr : sub_dbr_ptr;
        *gpu_umem_dbr = NULL;
        status = doca_verbs_qp_init_attr_set_external_umem_dbr(
            qp_init_attr, shared_sq_dbr_umem->umem, (uint64_t)sub_dbr_offset);
    } else {
        if (((nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) != 0) ||
            (send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED)) {
            *umem_dbr_dev_ptr = calloc(dbr_umem_align_sz, sizeof(uint8_t));
            if (*umem_dbr_dev_ptr == nullptr) {
                DOCA_LOG(LOG_ERR, "Failed to alloc gpu memory for external umem qp");
                goto destroy_resources;
            }
        } else {
            if (enable_umem_cpu) {
                status = doca_gpu_mem_alloc(
                    gpu_dev, dbr_umem_align_sz, priv_get_page_size(), DOCA_GPU_MEM_TYPE_CPU_GPU,
                    (void **)(&gpu_umem_dbr_dev_ptr), (void **)(&cpu_umem_dbr_dev_ptr));
                *umem_dbr_dev_ptr = cpu_umem_dbr_dev_ptr;
            } else {
                status = doca_gpu_mem_alloc(gpu_dev, dbr_umem_align_sz, priv_get_page_size(),
                                            DOCA_GPU_MEM_TYPE_GPU, (void **)(&gpu_umem_dbr_dev_ptr),
                                            NULL);
                *umem_dbr_dev_ptr = gpu_umem_dbr_dev_ptr;
            }

            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to alloc gpu memory for external umem qp");
                goto destroy_resources;
            }
        }

        /* DBR is host-allocated in CPU Proxy path; use PEERMEM for host memory. */
        enum doca_gpu_verbs_mem_reg_type dbr_mreg_type =
            (((nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) != 0) ||
             (send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) ||
             enable_umem_cpu)
                ? DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_PEERMEM
                : mreg_type;
        status = create_gpu_umem(gpu_dev, net_dev, dbr_mreg_type, dbr_umem_align_sz,
                                 *umem_dbr_dev_ptr, gpu_umem_dbr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Create UMEM for DBREC failed with %d", status);
            goto destroy_resources;
        }

        status = doca_verbs_qp_init_attr_set_external_umem_dbr(qp_init_attr, *gpu_umem_dbr, 0);
    }
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs qp external dbr umem");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_pd(qp_init_attr, net_dev);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs PD");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_sq_wr(qp_init_attr, sq_nwqe);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set SQ size");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_rq_wr(qp_init_attr, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set RQ size");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_qp_type(qp_init_attr, DOCA_VERBS_QP_TYPE_RC);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set QP type");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_send_cq(qp_init_attr, cq_sq);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs CQ");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_send_max_sges(qp_init_attr, MAX_SEND_SEGS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set send_max_sges");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_receive_max_sges(qp_init_attr, MAX_RECEIVE_SEGS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set receive_max_sges");
        goto destroy_resources;
    }

    if (set_core_direct) {
        status = doca_verbs_qp_init_attr_set_core_direct_master(qp_init_attr, 1);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to set core_direct");
            goto destroy_resources;
        }
    }

    if (send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW) {
        status = doca_verbs_qp_init_attr_set_send_dbr_mode(qp_init_attr,
                                                           DOCA_VERBS_QP_SEND_DBR_MODE_NO_DBR_EXT);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to set send_dbr_mode");
            goto destroy_resources;
        }
    }

    if (ordering_semantic != DOCA_VERBS_QP_ORDERING_SEMANTIC_IBTA) {
        status = doca_verbs_qp_init_attr_set_ordering_semantic(qp_init_attr, ordering_semantic);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to set ordering_semantic");
            goto destroy_resources;
        }
    }

    status = doca_verbs_qp_create(net_dev, qp_init_attr, &new_qp);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs QP");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_destroy(qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs QP attributes");
        goto destroy_resources;
    }

    *verbs_qp = new_qp;
    *out_nic_handler = nic_handler;

    return DOCA_SUCCESS;

destroy_resources:
    if (new_qp != NULL) {
        tmp_status = doca_verbs_qp_destroy(new_qp);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs QP");
    }

    if (qp_init_attr != NULL) {
        tmp_status = doca_verbs_qp_init_attr_destroy(qp_init_attr);
        if (tmp_status != DOCA_SUCCESS)
            DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs QP attributes");
    }

    if (*gpu_umem != NULL) {
        tmp_status = doca_verbs_umem_destroy(*gpu_umem);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy gpu umem");
        *gpu_umem = nullptr;
    }

    if (shared_sq_umem == NULL && gpu_umem_dev_ptr != NULL) {
        tmp_status = doca_gpu_mem_free(gpu_dev, gpu_umem_dev_ptr);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of umem");
        *umem_dev_ptr = 0;
    }

    if (*gpu_umem_dbr != NULL) {
        tmp_status = doca_verbs_umem_destroy(*gpu_umem_dbr);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy gpu umem");
        *gpu_umem_dbr = nullptr;
    }

    if (shared_sq_dbr_umem == NULL && (*umem_dbr_dev_ptr) != 0) {
        if (((nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) != 0) ||
            send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) {
            free(*umem_dbr_dev_ptr);
        } else {
            tmp_status = doca_gpu_mem_free(gpu_dev, gpu_umem_dbr_dev_ptr);
            if (tmp_status != DOCA_SUCCESS)
                DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of umem");
        }
        *umem_dbr_dev_ptr = 0;
    }

    return status;
}

static doca_error_t doca_gpu_verbs_destroy_qp_hl_internal(struct doca_gpu_verbs_qp_hl *qp) {
    doca_error_t status;

    if (qp == nullptr) return DOCA_ERROR_INVALID_VALUE;

    const uint8_t emulate_no_dbr_ext =
        (qp->send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED);

    if (qp->qp_gverbs != nullptr) {
        status = doca_gpu_verbs_unexport_qp(qp->gpu_dev, qp->qp_gverbs);
        if (status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to unexport qp gverbs resources");
    }

    if (qp->qp != nullptr) {
        status = doca_verbs_qp_destroy(qp->qp);
        if (status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs QP");
    }

    if (qp->qp_umem != NULL) {
        status = doca_verbs_umem_destroy(qp->qp_umem);
        if (status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy gpu qp umem");
    }

    if (qp->qp_umem_gpu_ptr != 0) {
        status = doca_gpu_mem_free(qp->gpu_dev, qp->qp_umem_gpu_ptr);
        if (status != DOCA_SUCCESS)
            DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of qp ring buffer");
    }

    if (qp->qp_umem_dbr != NULL) {
        status = doca_verbs_umem_destroy(qp->qp_umem_dbr);
        if (status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy gpu qp umem dbr");
    }

    if (qp->qp_umem_dbr_gpu_ptr != NULL) {
        if (((qp->nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) != 0) ||
            emulate_no_dbr_ext) {
            free(qp->qp_umem_dbr_gpu_ptr);
        } else {
            status = doca_gpu_mem_free(qp->gpu_dev, qp->qp_umem_dbr_gpu_ptr);
            if (status != DOCA_SUCCESS)
                DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of qp ring buffer dbr");
        }
    }

    if (qp->external_uar != NULL) {
        status = doca_verbs_uar_destroy(qp->external_uar);
        if (status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs UAR");
    }

    if (qp->cq_sq) {
        status = doca_verbs_cq_destroy(qp->cq_sq);
        if (status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs CQ");

        if (qp->cq_sq_umem != NULL) {
            status = doca_verbs_umem_destroy(qp->cq_sq_umem);
            if (status != DOCA_SUCCESS)
                DOCA_LOG(LOG_ERR, "Failed to destroy gpu sq cq ring buffer umem");
        }

        if (qp->cq_sq_umem_gpu_ptr != 0) {
            status = doca_gpu_mem_free(qp->gpu_dev, qp->cq_sq_umem_gpu_ptr);
            if (status != DOCA_SUCCESS)
                DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of sq cq ring buffer");
        }
    }

    memset(qp, 0, sizeof(*qp));

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_create_qp_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                                         struct doca_gpu_verbs_qp_hl **qp) {
    doca_error_t status = DOCA_SUCCESS;

    if (qp_init_attr == nullptr || qp == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input value: qp_init_attr %p qp %p", (void *)qp_init_attr,
                 (void *)*qp);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (qp_init_attr->gpu_dev == nullptr || qp_init_attr->ibpd == nullptr ||
        qp_init_attr->net_dev == nullptr || qp_init_attr->sq_nwqe == 0) {
        DOCA_LOG(LOG_ERR, "Invalid input value: gpu_dev %p ibpd %p net_dev %p sq_nwqe %d",
                 (void *)qp_init_attr->gpu_dev, (void *)qp_init_attr->ibpd,
                 (void *)qp_init_attr->net_dev, qp_init_attr->sq_nwqe);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if ((qp_init_attr->send_dbr_mode_ext ==
         DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) &&
        !qp_init_attr->gpu_dev->open->support_gdrcopy) {
        DOCA_LOG(LOG_ERR, "SW-emulated no DBR feature is not supported without GDRCopy");
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    if (qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW) {
        DOCA_LOG(LOG_ERR,
                 "nic_handler (%d) must be AUTO %d, CPU_PROXY %d, GPU_SM_DB %d, GPU_SM_BF %d, or "
                 "CPU_PROXY_FREE_FLOW %d",
                 qp_init_attr->nic_handler, DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW);
        return DOCA_ERROR_INVALID_VALUE;
    }

    status = validate_cq_type(qp_init_attr);
    if (status != DOCA_SUCCESS) return status;

    enum doca_gpu_dev_verbs_cq_type cq_type = resolve_cq_type(qp_init_attr);

    bool enable_data_direct =
        qp_init_attr->flags & DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_SUPPORT_DATA_DIRECT;

    struct doca_gpu_verbs_qp_hl *qp_ =
        (struct doca_gpu_verbs_qp_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_hl));
    if (qp_ == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed alloc memory for high-level qp");
        return DOCA_ERROR_NO_MEMORY;
    }

    bool enable_cq_umem_cpu =
        qp_init_attr->enable_umem_cpu || (cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST);

    qp_->gpu_dev = qp_init_attr->gpu_dev;
    qp_->send_dbr_mode_ext = qp_init_attr->send_dbr_mode_ext;

    if (qp_init_attr->sq_nwqe > 0) {
        qp_init_attr->sq_nwqe =
            (uint32_t)doca_internal_utils_next_power_of_two(qp_init_attr->sq_nwqe);

        status = create_cq(qp_->gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd,
                           qp_init_attr->mreg_type, qp_init_attr->sq_nwqe, &qp_->cq_sq_umem_gpu_ptr,
                           &qp_->cq_sq_umem, NULL, qp_init_attr->cq_collapsed, enable_cq_umem_cpu,
                           qp_init_attr->comp_channel, &qp_->cq_sq, NULL, NULL);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create doca verbs cq");
            goto exit_error;
        }
    }

    status = create_uar(qp_init_attr->net_dev, qp_init_attr->nic_handler, &qp_->external_uar);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs uar");
        goto exit_error;
    }

    status = create_qp(qp_->gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd,
                       qp_init_attr->mreg_type, qp_->cq_sq, qp_init_attr->sq_nwqe,
                       &qp_->qp_umem_gpu_ptr, &qp_->qp_umem, &qp_->qp_umem_dbr_gpu_ptr,
                       &qp_->qp_umem_dbr, qp_->external_uar, qp_init_attr->nic_handler, false,
                       qp_init_attr->send_dbr_mode_ext, qp_init_attr->ordering_semantic,
                       qp_init_attr->enable_umem_cpu, &qp_->qp, &qp_->nic_handler, NULL, NULL);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs qp");
        goto exit_error;
    }

    /* This is still ok even in case of enable_umem_cpu == true as it summarize in the CPU structure
     * qp_gverbs all the data path required elements
     */
    status = doca_gpu_verbs_export_qp(qp_->gpu_dev, qp_->qp, qp_->nic_handler, qp_->qp_umem_gpu_ptr,
                                      qp_->cq_sq, qp_->send_dbr_mode_ext, cq_type,
                                      enable_data_direct, &qp_->qp_gverbs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create GPU verbs QP");
        return status;
    }

    if (qp_init_attr->comp_channel) {
        status = doca_verbs_cq_set_cq_context(qp_->cq_sq, qp_->qp_gverbs);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed doca_verbs_cq_set_cq_context with %d", status);
            goto exit_error;
        }

        status = doca_gpu_verbs_req_notify_cq(qp_->gpu_dev, qp_->cq_sq);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed doca_gpu_verbs_req_notify_cq with %d", status);
            goto exit_error;
        }

        DOCA_LOG(LOG_INFO, "doca_verbs_cq_set_cq_context set with context %p",
                 (void *)qp_->qp_gverbs);
    }

    *qp = qp_;

    return DOCA_SUCCESS;

exit_error:
    if (qp_) {
        doca_gpu_verbs_destroy_qp_hl_internal(qp_);
    }

    free(qp_);
    return status;
}

doca_error_t doca_gpu_verbs_destroy_qp_hl(struct doca_gpu_verbs_qp_hl *qp) {
    if (qp == nullptr) return DOCA_ERROR_INVALID_VALUE;

    doca_gpu_verbs_destroy_qp_hl_internal(qp);
    free(qp);

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_create_qp_group_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                                               struct doca_gpu_verbs_qp_group_hl **qpg) {
    doca_error_t status = DOCA_SUCCESS;

    if (qp_init_attr == nullptr || qpg == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input value: qp_init_attr %p qp %p", (void *)qp_init_attr,
                 (void *)*qpg);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (qp_init_attr->gpu_dev == nullptr || qp_init_attr->ibpd == nullptr ||
        qp_init_attr->sq_nwqe == 0) {
        DOCA_LOG(LOG_ERR, "Invalid input value: gpu_dev %p ibpd %p sq_nwqe %d",
                 (void *)qp_init_attr->gpu_dev, (void *)qp_init_attr->ibpd, qp_init_attr->sq_nwqe);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW) {
        DOCA_LOG(LOG_ERR,
                 "nic_handler (%d) must be AUTO %d, CPU_PROXY %d, GPU_SM_DB %d, GPU_SM_BF %d, or "
                 "CPU_PROXY_FREE_FLOW %d",
                 qp_init_attr->nic_handler, DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if ((qp_init_attr->send_dbr_mode_ext ==
         DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) &&
        !qp_init_attr->gpu_dev->open->support_gdrcopy) {
        DOCA_LOG(LOG_ERR, "SW-emulated no DBR feature is not supported without GDRCopy");
        return DOCA_ERROR_INVALID_VALUE;
    }

    status = validate_cq_type(qp_init_attr);
    if (status != DOCA_SUCCESS) return status;

    struct doca_gpu_verbs_qp_group_hl *qpg_ =
        (struct doca_gpu_verbs_qp_group_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_group_hl));
    if (qpg_ == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed alloc memory for high-level qp");
        return DOCA_ERROR_NO_MEMORY;
    }

    enum doca_gpu_dev_verbs_cq_type cq_type = resolve_cq_type(qp_init_attr);

    bool enable_data_direct =
        qp_init_attr->flags & DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_SUPPORT_DATA_DIRECT;

    bool enable_cq_umem_cpu =
        qp_init_attr->enable_umem_cpu || (cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST);

    /********** Create main QP **********/

    qpg_->qp_main.gpu_dev = qp_init_attr->gpu_dev;
    qpg_->qp_main.send_dbr_mode_ext = qp_init_attr->send_dbr_mode_ext;

    status =
        create_uar(qp_init_attr->net_dev, qp_init_attr->nic_handler, &qpg_->qp_main.external_uar);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs uar");
        goto exit_error;
    }

    if (qp_init_attr->sq_nwqe > 0) {
        qp_init_attr->sq_nwqe =
            (uint32_t)doca_internal_utils_next_power_of_two(qp_init_attr->sq_nwqe);

        status = create_cq(
            qpg_->qp_main.gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd,
            qp_init_attr->mreg_type, qp_init_attr->sq_nwqe, &qpg_->qp_main.cq_sq_umem_gpu_ptr,
            &qpg_->qp_main.cq_sq_umem, qpg_->qp_main.external_uar, qp_init_attr->cq_collapsed,
            enable_cq_umem_cpu, qp_init_attr->comp_channel, &qpg_->qp_main.cq_sq, NULL, NULL);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create doca verbs cq");
            goto exit_error;
        }
    }

    status = create_qp(
        qpg_->qp_main.gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd, qp_init_attr->mreg_type,
        qpg_->qp_main.cq_sq, qp_init_attr->sq_nwqe, &qpg_->qp_main.qp_umem_gpu_ptr,
        &qpg_->qp_main.qp_umem, &qpg_->qp_main.qp_umem_dbr_gpu_ptr, &qpg_->qp_main.qp_umem_dbr,
        qpg_->qp_main.external_uar, qp_init_attr->nic_handler, false,
        qp_init_attr->send_dbr_mode_ext, qp_init_attr->ordering_semantic,
        qp_init_attr->enable_umem_cpu, &qpg_->qp_main.qp, &qpg_->qp_main.nic_handler, NULL, NULL);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs qp");
        goto exit_error;
    }

    status = doca_gpu_verbs_export_qp(qpg_->qp_main.gpu_dev, qpg_->qp_main.qp,
                                      qpg_->qp_main.nic_handler, qpg_->qp_main.qp_umem_gpu_ptr,
                                      qpg_->qp_main.cq_sq, qpg_->qp_main.send_dbr_mode_ext, cq_type,
                                      enable_data_direct, &qpg_->qp_main.qp_gverbs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create GPU verbs QP");
        return status;
    }

    if (qp_init_attr->comp_channel) {
        status = doca_verbs_cq_set_cq_context(qpg_->qp_main.cq_sq, qpg_->qp_main.qp_gverbs);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed doca_verbs_cq_set_cq_context with %d", status);
            goto exit_error;
        }

        status = doca_gpu_verbs_req_notify_cq(qpg_->qp_main.gpu_dev, qpg_->qp_main.cq_sq);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed doca_gpu_verbs_req_notify_cq with %d", status);
            goto exit_error;
        }

        DOCA_LOG(LOG_INFO, "doca_verbs_cq_set_cq_context set with context %p",
                 (void *)qpg_->qp_main.qp_gverbs);
    }

    /********** Create companion QP **********/

    qpg_->qp_companion.gpu_dev = qp_init_attr->gpu_dev;
    qpg_->qp_companion.external_uar = qpg_->qp_main.external_uar;
    qpg_->qp_companion.send_dbr_mode_ext = qpg_->qp_main.send_dbr_mode_ext;

    if (qp_init_attr->sq_nwqe > 0) {
        qp_init_attr->sq_nwqe =
            (uint32_t)doca_internal_utils_next_power_of_two(qp_init_attr->sq_nwqe);

        status = create_cq(qpg_->qp_companion.gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd,
                           qp_init_attr->mreg_type, qp_init_attr->sq_nwqe,
                           &qpg_->qp_companion.cq_sq_umem_gpu_ptr, &qpg_->qp_companion.cq_sq_umem,
                           qpg_->qp_companion.external_uar, qp_init_attr->cq_collapsed,
                           enable_cq_umem_cpu, qp_init_attr->comp_channel,
                           &qpg_->qp_companion.cq_sq, NULL, NULL);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create doca verbs cq");
            goto exit_error;
        }
    }

    status = create_qp(qpg_->qp_companion.gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd,
                       qp_init_attr->mreg_type, qpg_->qp_companion.cq_sq, qp_init_attr->sq_nwqe,
                       &qpg_->qp_companion.qp_umem_gpu_ptr, &qpg_->qp_companion.qp_umem,
                       &qpg_->qp_companion.qp_umem_dbr_gpu_ptr, &qpg_->qp_companion.qp_umem_dbr,
                       qpg_->qp_companion.external_uar, qp_init_attr->nic_handler, true,
                       qp_init_attr->send_dbr_mode_ext, qp_init_attr->ordering_semantic,
                       qp_init_attr->enable_umem_cpu, &qpg_->qp_companion.qp,
                       &qpg_->qp_companion.nic_handler, NULL, NULL);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs qp");
        goto exit_error;
    }

    status =
        doca_gpu_verbs_export_qp(qpg_->qp_companion.gpu_dev, qpg_->qp_companion.qp,
                                 qpg_->qp_companion.nic_handler, qpg_->qp_companion.qp_umem_gpu_ptr,
                                 qpg_->qp_companion.cq_sq, qpg_->qp_companion.send_dbr_mode_ext,
                                 cq_type, enable_data_direct, &qpg_->qp_companion.qp_gverbs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create GPU verbs QP");
        return status;
    }

    if (qp_init_attr->comp_channel) {
        status =
            doca_verbs_cq_set_cq_context(qpg_->qp_companion.cq_sq, qpg_->qp_companion.qp_gverbs);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed doca_verbs_cq_set_cq_context with %d", status);
            goto exit_error;
        }

        status = doca_gpu_verbs_req_notify_cq(qpg_->qp_companion.gpu_dev, qpg_->qp_companion.cq_sq);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed doca_gpu_verbs_req_notify_cq with %d", status);
            goto exit_error;
        }

        DOCA_LOG(LOG_INFO, "doca_verbs_cq_set_cq_context set with context %p",
                 (void *)qpg_->qp_companion.qp_gverbs);
    }

    *qpg = qpg_;

    return DOCA_SUCCESS;

exit_error:
    if (qpg_) {
        doca_gpu_verbs_destroy_qp_hl_internal(&qpg_->qp_main);
        qpg_->qp_companion.external_uar = nullptr;
        doca_gpu_verbs_destroy_qp_hl_internal(&qpg_->qp_companion);
    }

    free(qpg_);
    return status;
}

doca_error_t doca_gpu_verbs_destroy_qp_group_hl(struct doca_gpu_verbs_qp_group_hl *qpg) {
    if (qpg == nullptr) return DOCA_ERROR_INVALID_VALUE;

    doca_gpu_verbs_destroy_qp_hl_internal(&qpg->qp_main);
    qpg->qp_companion.external_uar = nullptr;
    doca_gpu_verbs_destroy_qp_hl_internal(&qpg->qp_companion);

    memset(qpg, 0, sizeof(*qpg));

    free(qpg);

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_qp_flat_list_create_hl(struct doca_gpu_verbs_qp_hl **qp_list,
                                                   uint32_t num_elems,
                                                   struct doca_gpu_dev_verbs_qp **qp_gpu) {
    doca_error_t status = DOCA_SUCCESS;
    cudaError_t error;
    struct doca_gpu_dev_verbs_qp *qp_gpu_;

    if (num_elems == 0 || qp_list == nullptr || qp_gpu == nullptr) return DOCA_ERROR_INVALID_VALUE;

    error = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaMalloc((void **)&qp_gpu_, sizeof(struct doca_gpu_dev_verbs_qp) * num_elems));
    if (error != cudaSuccess) return DOCA_ERROR_NO_MEMORY;

    for (uint32_t i = 0; i < num_elems; i++) {
        error = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
            cudaMemcpy(qp_gpu_ + i, qp_list[i]->qp_gverbs->qp_cpu,
                       sizeof(struct doca_gpu_dev_verbs_qp), cudaMemcpyDefault));
        if (error != cudaSuccess) goto exit_error;
    }

    *qp_gpu = qp_gpu_;

    return status;

exit_error:
    DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaFree(qp_gpu));
    return status;
}

doca_error_t doca_gpu_verbs_qp_flat_list_destroy_hl(struct doca_gpu_dev_verbs_qp *qp_gpu) {
    if (qp_gpu == nullptr) return DOCA_ERROR_INVALID_VALUE;

    DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaFree(qp_gpu));
    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_create_qp_list_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                                              uint32_t num_qps,
                                              struct doca_gpu_verbs_qp_list_hl **qp_list) {
    doca_error_t status = DOCA_SUCCESS;
    struct doca_gpu_verbs_qp_list_hl *list = NULL;
    struct doca_gpu_verbs_umem_hl *cq_umem = NULL, *cq_dbr_umem = NULL;
    struct doca_gpu_verbs_umem_hl *sq_umem = NULL, *sq_dbr_umem = NULL;
    uint32_t cq_size_per_qp, sq_size_per_qp, dbr_size_per_qp;

    if (qp_init_attr == NULL || qp_list == NULL || num_qps == 0) return DOCA_ERROR_INVALID_VALUE;
    if (qp_init_attr->gpu_dev == NULL || qp_init_attr->net_dev == NULL ||
        qp_init_attr->ibpd == NULL || qp_init_attr->sq_nwqe == 0)
        return DOCA_ERROR_INVALID_VALUE;

    if ((qp_init_attr->send_dbr_mode_ext ==
         DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) &&
        !qp_init_attr->gpu_dev->open->support_gdrcopy) {
        DOCA_LOG(LOG_ERR, "SW-emulated no DBR feature is not supported without GDRCopy");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW) {
        DOCA_LOG(LOG_ERR,
                 "nic_handler (%d) must be AUTO %d, CPU_PROXY %d, GPU_SM_DB %d, GPU_SM_BF %d, or "
                 "CPU_PROXY_FREE_FLOW %d",
                 qp_init_attr->nic_handler, DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW);
        return DOCA_ERROR_INVALID_VALUE;
    }

    status = validate_cq_type(qp_init_attr);
    if (status != DOCA_SUCCESS) return status;

    enum doca_gpu_dev_verbs_cq_type cq_type = resolve_cq_type(qp_init_attr);

    bool enable_data_direct =
        qp_init_attr->flags & DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_SUPPORT_DATA_DIRECT;

    bool enable_cq_umem_cpu =
        qp_init_attr->enable_umem_cpu || (cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST);

    uint32_t sq_nwqe = (uint32_t)doca_internal_utils_next_power_of_two(qp_init_attr->sq_nwqe);

    cq_size_per_qp = calc_cq_external_umem_size(sq_nwqe, DBR_SIZE);
    sq_size_per_qp = calc_qp_external_umem_size(sq_nwqe);
    dbr_size_per_qp = align_up_uint32(DBR_SIZE, priv_get_page_size());

    /* Pre-resolve AUTO nic_handler so that shared slab memory type is correct */
    enum doca_gpu_dev_verbs_nic_handler resolved_nic_handler;
    if (qp_init_attr->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
        doca_verbs_uar_t *probe_uar = NULL;
        status = create_uar(qp_init_attr->net_dev, qp_init_attr->nic_handler, &probe_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create probe UAR for AUTO resolution");
            return status;
        }
        bool can_register = false;
        status =
            doca_gpu_verbs_can_gpu_register_uar(probe_uar->open->get_reg_addr(), &can_register);
        doca_verbs_uar_destroy(probe_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to check if UAR can be registered on GPU");
            return status;
        }
        resolved_nic_handler = can_register ? DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB
                                            : DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
    } else {
        resolved_nic_handler = qp_init_attr->nic_handler;
    }

    list = (struct doca_gpu_verbs_qp_list_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_list_hl));
    if (list == NULL) return DOCA_ERROR_NO_MEMORY;

    list->qps = (struct doca_gpu_verbs_qp_hl *)calloc(num_qps, sizeof(struct doca_gpu_verbs_qp_hl));
    if (list->qps == NULL) {
        free(list);
        return DOCA_ERROR_NO_MEMORY;
    }
    list->num_qps = num_qps;

    status = create_umem_hl(qp_init_attr->gpu_dev, qp_init_attr->net_dev, qp_init_attr->mreg_type,
                            (size_t)cq_size_per_qp * num_qps, false, enable_cq_umem_cpu, &cq_umem);
    if (status != DOCA_SUCCESS) goto exit_error;

    status =
        create_umem_hl(qp_init_attr->gpu_dev, qp_init_attr->net_dev, qp_init_attr->mreg_type,
                       (size_t)dbr_size_per_qp * num_qps, false, enable_cq_umem_cpu, &cq_dbr_umem);
    if (status != DOCA_SUCCESS) goto exit_error;

    status = create_umem_hl(qp_init_attr->gpu_dev, qp_init_attr->net_dev, qp_init_attr->mreg_type,
                            (size_t)sq_size_per_qp * num_qps, false, qp_init_attr->enable_umem_cpu,
                            &sq_umem);
    if (status != DOCA_SUCCESS) goto exit_error;

    {
        bool dbr_is_host_mem =
            ((resolved_nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) != 0 ||
             qp_init_attr->send_dbr_mode_ext ==
                 DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED);
        status = create_umem_hl(qp_init_attr->gpu_dev, qp_init_attr->net_dev,
                                qp_init_attr->mreg_type, (size_t)dbr_size_per_qp * num_qps,
                                dbr_is_host_mem, qp_init_attr->enable_umem_cpu, &sq_dbr_umem);
        if (status != DOCA_SUCCESS) goto exit_error;
    }

    for (uint32_t i = 0; i < num_qps; i++) {
        struct doca_gpu_verbs_qp_hl *qp_ = &list->qps[i];
        qp_->gpu_dev = qp_init_attr->gpu_dev;
        qp_->send_dbr_mode_ext = qp_init_attr->send_dbr_mode_ext;
        qp_->nic_handler = resolved_nic_handler;

        status = create_cq(qp_->gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd,
                           qp_init_attr->mreg_type, sq_nwqe, &qp_->cq_sq_umem_gpu_ptr,
                           &qp_->cq_sq_umem, NULL, qp_init_attr->cq_collapsed, enable_cq_umem_cpu,
                           qp_init_attr->comp_channel, &qp_->cq_sq, cq_umem, cq_dbr_umem);
        if (status != DOCA_SUCCESS) goto exit_error;

        status = create_uar(qp_init_attr->net_dev, resolved_nic_handler, &qp_->external_uar);
        if (status != DOCA_SUCCESS) goto exit_error;

        status = create_qp(
            qp_->gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd, qp_init_attr->mreg_type,
            qp_->cq_sq, sq_nwqe, &qp_->qp_umem_gpu_ptr, &qp_->qp_umem, &qp_->qp_umem_dbr_gpu_ptr,
            &qp_->qp_umem_dbr, qp_->external_uar, resolved_nic_handler, false,
            qp_init_attr->send_dbr_mode_ext, qp_init_attr->ordering_semantic,
            qp_init_attr->enable_umem_cpu, &qp_->qp, &qp_->nic_handler, sq_umem, sq_dbr_umem);
        if (status != DOCA_SUCCESS) goto exit_error;

        status = doca_gpu_verbs_export_qp(qp_->gpu_dev, qp_->qp, qp_->nic_handler,
                                          qp_->qp_umem_gpu_ptr, qp_->cq_sq, qp_->send_dbr_mode_ext,
                                          cq_type, enable_data_direct, &qp_->qp_gverbs);
        if (status != DOCA_SUCCESS) goto exit_error;

        if (qp_init_attr->comp_channel) {
            status = doca_verbs_cq_set_cq_context(qp_->cq_sq, qp_->qp_gverbs);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed doca_verbs_cq_set_cq_context with %d", status);
                goto exit_error;
            }

            status = doca_gpu_verbs_req_notify_cq(qp_->gpu_dev, qp_->cq_sq);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed doca_gpu_verbs_req_notify_cq with %d", status);
                goto exit_error;
            }

            DOCA_LOG(LOG_INFO, "doca_verbs_cq_set_cq_context set with context %p",
                     (void *)qp_->qp_gverbs);
        }
    }

    list->cq_umem = cq_umem;
    list->cq_dbr_umem = cq_dbr_umem;
    list->sq_umem = sq_umem;
    list->sq_dbr_umem = sq_dbr_umem;
    *qp_list = list;
    return DOCA_SUCCESS;

exit_error:
    for (uint32_t i = 0; i < num_qps; i++) {
        struct doca_gpu_verbs_qp_hl *qp_ = &list->qps[i];
        /* Null out suballocated pointers so destroy_internal skips doca_gpu_mem_free on them */
        qp_->qp_umem_gpu_ptr = NULL;
        qp_->qp_umem_dbr_gpu_ptr = NULL;
        qp_->cq_sq_umem_gpu_ptr = NULL;
        doca_gpu_verbs_destroy_qp_hl_internal(qp_);
    }
    if (cq_umem) destroy_umem_hl(qp_init_attr->gpu_dev, cq_umem);
    if (cq_dbr_umem) destroy_umem_hl(qp_init_attr->gpu_dev, cq_dbr_umem);
    if (sq_umem) destroy_umem_hl(qp_init_attr->gpu_dev, sq_umem);
    if (sq_dbr_umem) destroy_umem_hl(qp_init_attr->gpu_dev, sq_dbr_umem);
    free(list->qps);
    free(list);
    return status;
}

doca_error_t doca_gpu_verbs_destroy_qp_list_hl(struct doca_gpu_verbs_qp_list_hl *qp_list) {
    if (qp_list == NULL) return DOCA_ERROR_INVALID_VALUE;

    doca_gpu_t *gpu_dev = NULL;
    if (qp_list->num_qps > 0 && qp_list->qps != NULL) gpu_dev = qp_list->qps[0].gpu_dev;

    for (uint32_t i = 0; i < qp_list->num_qps; i++) {
        struct doca_gpu_verbs_qp_hl *qp_ = &qp_list->qps[i];
        /* Null out suballocated pointers so destroy_internal skips doca_gpu_mem_free on them */
        qp_->qp_umem_gpu_ptr = NULL;
        qp_->qp_umem_dbr_gpu_ptr = NULL;
        qp_->cq_sq_umem_gpu_ptr = NULL;
        doca_gpu_verbs_destroy_qp_hl_internal(qp_);
    }

    if (qp_list->cq_umem) destroy_umem_hl(gpu_dev, qp_list->cq_umem);
    if (qp_list->cq_dbr_umem) destroy_umem_hl(gpu_dev, qp_list->cq_dbr_umem);
    if (qp_list->sq_umem) destroy_umem_hl(gpu_dev, qp_list->sq_umem);
    if (qp_list->sq_dbr_umem) destroy_umem_hl(gpu_dev, qp_list->sq_dbr_umem);

    free(qp_list->qps);
    free(qp_list);
    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_create_qp_group_list_hl(
    struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr, uint32_t num_qp_groups,
    struct doca_gpu_verbs_qp_group_list_hl **qpg_list) {
    doca_error_t status = DOCA_SUCCESS;
    struct doca_gpu_verbs_qp_group_list_hl *list = NULL;
    struct doca_gpu_verbs_umem_hl *cq_umem = NULL, *cq_dbr_umem = NULL;
    struct doca_gpu_verbs_umem_hl *sq_umem = NULL, *sq_dbr_umem = NULL;
    uint32_t cq_size_per_qp, sq_size_per_qp, dbr_size_per_qp;
    uint32_t total_qps;

    if (qp_init_attr == NULL || qpg_list == NULL || num_qp_groups == 0)
        return DOCA_ERROR_INVALID_VALUE;
    if (qp_init_attr->gpu_dev == NULL || qp_init_attr->net_dev == NULL ||
        qp_init_attr->ibpd == NULL || qp_init_attr->sq_nwqe == 0)
        return DOCA_ERROR_INVALID_VALUE;

    if (qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF &&
        qp_init_attr->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW) {
        DOCA_LOG(LOG_ERR,
                 "nic_handler (%d) must be AUTO %d, CPU_PROXY %d, GPU_SM_DB %d, GPU_SM_BF %d, or "
                 "CPU_PROXY_FREE_FLOW %d",
                 qp_init_attr->nic_handler, DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF,
                 DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if ((qp_init_attr->send_dbr_mode_ext ==
         DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) &&
        !qp_init_attr->gpu_dev->open->support_gdrcopy) {
        DOCA_LOG(LOG_ERR, "SW-emulated no DBR feature is not supported without GDRCopy");
        return DOCA_ERROR_INVALID_VALUE;
    }

    status = validate_cq_type(qp_init_attr);
    if (status != DOCA_SUCCESS) return status;

    enum doca_gpu_dev_verbs_cq_type cq_type = resolve_cq_type(qp_init_attr);

    bool enable_cq_umem_cpu =
        qp_init_attr->enable_umem_cpu || (cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST);

    bool enable_data_direct =
        qp_init_attr->flags & DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_SUPPORT_DATA_DIRECT;

    uint32_t sq_nwqe = (uint32_t)doca_internal_utils_next_power_of_two(qp_init_attr->sq_nwqe);
    total_qps = 2 * num_qp_groups; /* main + companion per group */

    cq_size_per_qp = calc_cq_external_umem_size(sq_nwqe, DBR_SIZE);
    sq_size_per_qp = calc_qp_external_umem_size(sq_nwqe);
    dbr_size_per_qp = align_up_uint32(DBR_SIZE, priv_get_page_size());

    /* Pre-resolve AUTO nic_handler so that shared slab memory type is correct */
    enum doca_gpu_dev_verbs_nic_handler resolved_nic_handler;
    if (qp_init_attr->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
        doca_verbs_uar_t *probe_uar = NULL;
        status = create_uar(qp_init_attr->net_dev, qp_init_attr->nic_handler, &probe_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create probe UAR for AUTO resolution");
            return status;
        }
        bool can_register = false;
        status =
            doca_gpu_verbs_can_gpu_register_uar(probe_uar->open->get_reg_addr(), &can_register);
        doca_verbs_uar_destroy(probe_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to check if UAR can be registered on GPU");
            return status;
        }
        resolved_nic_handler = can_register ? DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB
                                            : DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
    } else {
        resolved_nic_handler = qp_init_attr->nic_handler;
    }

    list = (struct doca_gpu_verbs_qp_group_list_hl *)calloc(
        1, sizeof(struct doca_gpu_verbs_qp_group_list_hl));
    if (list == NULL) return DOCA_ERROR_NO_MEMORY;

    list->qpgs = (struct doca_gpu_verbs_qp_group_hl *)calloc(
        num_qp_groups, sizeof(struct doca_gpu_verbs_qp_group_hl));
    if (list->qpgs == NULL) {
        free(list);
        return DOCA_ERROR_NO_MEMORY;
    }
    list->num_qp_groups = num_qp_groups;

    status =
        create_umem_hl(qp_init_attr->gpu_dev, qp_init_attr->net_dev, qp_init_attr->mreg_type,
                       (size_t)cq_size_per_qp * total_qps, false, enable_cq_umem_cpu, &cq_umem);
    if (status != DOCA_SUCCESS) goto exit_error;

    status = create_umem_hl(qp_init_attr->gpu_dev, qp_init_attr->net_dev, qp_init_attr->mreg_type,
                            (size_t)dbr_size_per_qp * total_qps, false, enable_cq_umem_cpu,
                            &cq_dbr_umem);
    if (status != DOCA_SUCCESS) goto exit_error;

    status = create_umem_hl(qp_init_attr->gpu_dev, qp_init_attr->net_dev, qp_init_attr->mreg_type,
                            (size_t)sq_size_per_qp * total_qps, false,
                            qp_init_attr->enable_umem_cpu, &sq_umem);
    if (status != DOCA_SUCCESS) goto exit_error;

    {
        bool dbr_is_host_mem =
            ((resolved_nic_handler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) != 0 ||
             qp_init_attr->send_dbr_mode_ext ==
                 DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED);
        status = create_umem_hl(qp_init_attr->gpu_dev, qp_init_attr->net_dev,
                                qp_init_attr->mreg_type, (size_t)dbr_size_per_qp * total_qps,
                                dbr_is_host_mem, qp_init_attr->enable_umem_cpu, &sq_dbr_umem);
        if (status != DOCA_SUCCESS) goto exit_error;
    }

    for (uint32_t i = 0; i < num_qp_groups; i++) {
        struct doca_gpu_verbs_qp_hl *main_ = &list->qpgs[i].qp_main;
        struct doca_gpu_verbs_qp_hl *comp_ = &list->qpgs[i].qp_companion;

        main_->gpu_dev = qp_init_attr->gpu_dev;
        main_->send_dbr_mode_ext = qp_init_attr->send_dbr_mode_ext;
        main_->nic_handler = resolved_nic_handler;

        /* One UAR shared between main and companion */
        status = create_uar(qp_init_attr->net_dev, resolved_nic_handler, &main_->external_uar);
        if (status != DOCA_SUCCESS) goto exit_error;

        status = create_cq(main_->gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd,
                           qp_init_attr->mreg_type, sq_nwqe, &main_->cq_sq_umem_gpu_ptr,
                           &main_->cq_sq_umem, main_->external_uar, qp_init_attr->cq_collapsed,
                           qp_init_attr->enable_umem_cpu, qp_init_attr->comp_channel, &main_->cq_sq,
                           cq_umem, cq_dbr_umem);
        if (status != DOCA_SUCCESS) goto exit_error;

        status = create_qp(main_->gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd,
                           qp_init_attr->mreg_type, main_->cq_sq, sq_nwqe, &main_->qp_umem_gpu_ptr,
                           &main_->qp_umem, &main_->qp_umem_dbr_gpu_ptr, &main_->qp_umem_dbr,
                           main_->external_uar, resolved_nic_handler, false,
                           qp_init_attr->send_dbr_mode_ext, qp_init_attr->ordering_semantic,
                           qp_init_attr->enable_umem_cpu, &main_->qp, &main_->nic_handler, sq_umem,
                           sq_dbr_umem);
        if (status != DOCA_SUCCESS) goto exit_error;

        status = doca_gpu_verbs_export_qp(
            main_->gpu_dev, main_->qp, main_->nic_handler, main_->qp_umem_gpu_ptr, main_->cq_sq,
            main_->send_dbr_mode_ext, cq_type, enable_data_direct, &main_->qp_gverbs);
        if (status != DOCA_SUCCESS) goto exit_error;

        if (qp_init_attr->comp_channel) {
            status = doca_verbs_cq_set_cq_context(main_->cq_sq, main_->qp_gverbs);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed doca_verbs_cq_set_cq_context with %d", status);
                goto exit_error;
            }

            status = doca_gpu_verbs_req_notify_cq(main_->gpu_dev, main_->cq_sq);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed doca_gpu_verbs_req_notify_cq with %d", status);
                goto exit_error;
            }

            DOCA_LOG(LOG_INFO, "doca_verbs_cq_set_cq_context set with context %p",
                     (void *)main_->qp_gverbs);
        }

        comp_->gpu_dev = qp_init_attr->gpu_dev;
        comp_->send_dbr_mode_ext = qp_init_attr->send_dbr_mode_ext;
        comp_->nic_handler = resolved_nic_handler;
        comp_->external_uar = main_->external_uar; /* shared UAR */

        status = create_cq(comp_->gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd,
                           qp_init_attr->mreg_type, sq_nwqe, &comp_->cq_sq_umem_gpu_ptr,
                           &comp_->cq_sq_umem, comp_->external_uar, qp_init_attr->cq_collapsed,
                           qp_init_attr->enable_umem_cpu, qp_init_attr->comp_channel, &comp_->cq_sq,
                           cq_umem, cq_dbr_umem);
        if (status != DOCA_SUCCESS) goto exit_error;

        status = create_qp(comp_->gpu_dev, qp_init_attr->net_dev, qp_init_attr->ibpd,
                           qp_init_attr->mreg_type, comp_->cq_sq, sq_nwqe, &comp_->qp_umem_gpu_ptr,
                           &comp_->qp_umem, &comp_->qp_umem_dbr_gpu_ptr, &comp_->qp_umem_dbr,
                           comp_->external_uar, resolved_nic_handler, true,
                           qp_init_attr->send_dbr_mode_ext, qp_init_attr->ordering_semantic,
                           qp_init_attr->enable_umem_cpu, &comp_->qp, &comp_->nic_handler, sq_umem,
                           sq_dbr_umem);
        if (status != DOCA_SUCCESS) goto exit_error;

        status = doca_gpu_verbs_export_qp(
            comp_->gpu_dev, comp_->qp, comp_->nic_handler, comp_->qp_umem_gpu_ptr, comp_->cq_sq,
            comp_->send_dbr_mode_ext, cq_type, enable_data_direct, &comp_->qp_gverbs);
        if (status != DOCA_SUCCESS) goto exit_error;

        if (qp_init_attr->comp_channel) {
            status = doca_verbs_cq_set_cq_context(comp_->cq_sq, comp_->qp_gverbs);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed doca_verbs_cq_set_cq_context with %d", status);
                goto exit_error;
            }

            status = doca_gpu_verbs_req_notify_cq(comp_->gpu_dev, comp_->cq_sq);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed doca_gpu_verbs_req_notify_cq with %d", status);
                goto exit_error;
            }

            DOCA_LOG(LOG_INFO, "doca_verbs_cq_set_cq_context set with context %p",
                     (void *)comp_->qp_gverbs);
        }
    }

    list->cq_umem = cq_umem;
    list->cq_dbr_umem = cq_dbr_umem;
    list->sq_umem = sq_umem;
    list->sq_dbr_umem = sq_dbr_umem;
    *qpg_list = list;
    return DOCA_SUCCESS;

exit_error:
    for (uint32_t i = 0; i < num_qp_groups; i++) {
        struct doca_gpu_verbs_qp_hl *main_ = &list->qpgs[i].qp_main;
        struct doca_gpu_verbs_qp_hl *comp_ = &list->qpgs[i].qp_companion;
        /* Null out suballocated ptrs so destroy_internal skips doca_gpu_mem_free */
        comp_->qp_umem_gpu_ptr = NULL;
        comp_->qp_umem_dbr_gpu_ptr = NULL;
        comp_->cq_sq_umem_gpu_ptr = NULL;
        comp_->external_uar = NULL; /* owned by main; don't double-free */
        doca_gpu_verbs_destroy_qp_hl_internal(comp_);
        main_->qp_umem_gpu_ptr = NULL;
        main_->qp_umem_dbr_gpu_ptr = NULL;
        main_->cq_sq_umem_gpu_ptr = NULL;
        doca_gpu_verbs_destroy_qp_hl_internal(main_);
    }
    if (cq_umem) destroy_umem_hl(qp_init_attr->gpu_dev, cq_umem);
    if (cq_dbr_umem) destroy_umem_hl(qp_init_attr->gpu_dev, cq_dbr_umem);
    if (sq_umem) destroy_umem_hl(qp_init_attr->gpu_dev, sq_umem);
    if (sq_dbr_umem) destroy_umem_hl(qp_init_attr->gpu_dev, sq_dbr_umem);
    free(list->qpgs);
    free(list);
    return status;
}

doca_error_t doca_gpu_verbs_destroy_qp_group_list_hl(
    struct doca_gpu_verbs_qp_group_list_hl *qpg_list) {
    if (qpg_list == NULL) return DOCA_ERROR_INVALID_VALUE;

    doca_gpu_t *gpu_dev = NULL;
    if (qpg_list->num_qp_groups > 0 && qpg_list->qpgs != NULL)
        gpu_dev = qpg_list->qpgs[0].qp_main.gpu_dev;

    for (uint32_t i = 0; i < qpg_list->num_qp_groups; i++) {
        struct doca_gpu_verbs_qp_hl *main_ = &qpg_list->qpgs[i].qp_main;
        struct doca_gpu_verbs_qp_hl *comp_ = &qpg_list->qpgs[i].qp_companion;
        /* Null out suballocated ptrs so destroy_internal skips doca_gpu_mem_free */
        comp_->qp_umem_gpu_ptr = NULL;
        comp_->qp_umem_dbr_gpu_ptr = NULL;
        comp_->cq_sq_umem_gpu_ptr = NULL;
        comp_->external_uar = NULL; /* owned by main; don't double-free */
        doca_gpu_verbs_destroy_qp_hl_internal(comp_);
        main_->qp_umem_gpu_ptr = NULL;
        main_->qp_umem_dbr_gpu_ptr = NULL;
        main_->cq_sq_umem_gpu_ptr = NULL;
        doca_gpu_verbs_destroy_qp_hl_internal(main_);
    }

    if (qpg_list->cq_umem) destroy_umem_hl(gpu_dev, qpg_list->cq_umem);
    if (qpg_list->cq_dbr_umem) destroy_umem_hl(gpu_dev, qpg_list->cq_dbr_umem);
    if (qpg_list->sq_umem) destroy_umem_hl(gpu_dev, qpg_list->sq_umem);
    if (qpg_list->sq_dbr_umem) destroy_umem_hl(gpu_dev, qpg_list->sq_dbr_umem);

    free(qpg_list->qpgs);
    free(qpg_list);
    return DOCA_SUCCESS;
}
