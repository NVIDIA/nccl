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

static doca_error_t create_uar(struct ibv_context *ibctx,
                               enum doca_gpu_dev_verbs_nic_handler nic_handler,
                               struct doca_verbs_uar **external_uar, bool bf_supported) {
    doca_error_t status = DOCA_SUCCESS;

    if (nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
        status = doca_verbs_uar_create(ibctx, DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED,
                                       external_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to doca_verbs_uar_create NC DEDICATED");
            status =
                doca_verbs_uar_create(ibctx, DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE, external_uar);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to doca_verbs_uar_create NC");
            } else {
                DOCA_LOG(LOG_INFO, "UAR created with DOCA_UAR_ALLOCATION_TYPE_NONCACHE");
            }
            return DOCA_SUCCESS;
        } else
            return DOCA_SUCCESS;
    }

    if (bf_supported &&
        (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF ||
         (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO && status != DOCA_SUCCESS))) {
        status =
            doca_verbs_uar_create(ibctx, DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME, external_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to doca_verbs_uar_create NC");
            return status;
        }
    } else
        return DOCA_ERROR_DRIVER;

    return status;
}

static doca_error_t create_gpu_umem(struct doca_gpu *gpu_dev, struct ibv_pd *ibpd,
                                    enum doca_gpu_verbs_mem_reg_type mreg_type, uint32_t umem_sz,
                                    void *umem_ptr, struct doca_verbs_umem **umem) {
    doca_error_t status;
    int dmabuf_fd;
    struct ibv_context *ibctx = ibpd->context;

    if (mreg_type == DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT) {
        status = doca_gpu_dmabuf_fd(gpu_dev, umem_ptr, umem_sz, &dmabuf_fd);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING,
                     "GPU doesn't support dmabuf, fallback to legacy nvidia-peermem mode");
            dmabuf_fd = DOCA_VERBS_DMABUF_INVALID_FD;
        }

        status = doca_verbs_umem_create(ibctx, umem_ptr, umem_sz, IBV_ACCESS_LOCAL_WRITE, dmabuf_fd,
                                        0, umem);
        if (status != DOCA_SUCCESS) {
            if (dmabuf_fd > 0) {
                DOCA_LOG(LOG_WARNING,
                         "Failed to create gpu umem with dmabuf. Fallback to legacy nvidia-peermem "
                         "mode");
                status = doca_verbs_umem_create(ibctx, umem_ptr, umem_sz, IBV_ACCESS_LOCAL_WRITE,
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
        status = doca_gpu_dmabuf_fd(gpu_dev, umem_ptr, umem_sz, &dmabuf_fd);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "GPU doesn't support dmabuf.");
            goto destroy_resources;
        }

        status = doca_verbs_umem_create(ibctx, umem_ptr, umem_sz, IBV_ACCESS_LOCAL_WRITE, dmabuf_fd,
                                        0, umem);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "GPU doesn't support dmabuf.");
            goto destroy_resources;
        }
    } else if (mreg_type == DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_PEERMEM) {
        status = doca_verbs_umem_create(ibctx, umem_ptr, umem_sz, IBV_ACCESS_LOCAL_WRITE,
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

static uint32_t calc_cq_external_umem_size(uint32_t queue_size) {
    uint32_t cqe_buf_size = 0;

    if (queue_size != 0)
        cqe_buf_size = (uint32_t)(queue_size * sizeof(struct doca_gpunetio_ib_mlx5_cqe64));

    return align_up_uint32(cqe_buf_size, priv_get_page_size());
}

static void mlx5_init_cqes(struct doca_gpunetio_ib_mlx5_cqe64 *cqes, uint32_t nb_cqes) {
    for (uint32_t cqe_idx = 0; cqe_idx < nb_cqes; cqe_idx++)
        cqes[cqe_idx].op_own =
            (DOCA_GPUNETIO_IB_MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) |
            DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK;
}

static doca_error_t create_cq(struct doca_gpu *gpu_dev, struct ibv_pd *ibpd,
                              enum doca_gpu_verbs_mem_reg_type mreg_type, uint32_t ncqes,
                              void **gpu_umem_dev_ptr, struct doca_verbs_umem **gpu_umem,
                              void **gpu_umem_dbr_dev_ptr, struct doca_verbs_umem **gpu_umem_dbr,
                              struct doca_verbs_uar *external_uar,
                              struct doca_verbs_cq **verbs_cq) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    cudaError_t status_cuda = cudaSuccess;
    struct doca_verbs_cq_attr *verbs_cq_attr = NULL;
    struct doca_verbs_cq *new_cq = NULL;
    struct doca_gpunetio_ib_mlx5_cqe64 *cq_ring_haddr = NULL;
    uint32_t external_umem_size = 0;
    size_t dbr_umem_align_sz;
    struct ibv_context *ibctx = ibpd->context;

    status = doca_verbs_cq_attr_create(&verbs_cq_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs cq attributes");
        return status;
    }

    external_umem_size = calc_cq_external_umem_size(ncqes);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to calc external umem size");
        goto destroy_resources;
    }

    status = doca_gpu_mem_alloc(gpu_dev, external_umem_size, priv_get_page_size(),
                                DOCA_GPU_MEM_TYPE_GPU, (void **)gpu_umem_dev_ptr, NULL);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to alloc gpu memory for external umem cq");
        goto destroy_resources;
    }

    cq_ring_haddr =
        (struct doca_gpunetio_ib_mlx5_cqe64 *)(calloc(external_umem_size, sizeof(uint8_t)));
    if (cq_ring_haddr == NULL) {
        DOCA_LOG(LOG_ERR, "Failed to allocate cq host ring buffer memory for initialization");
        status = DOCA_ERROR_NO_MEMORY;
        goto destroy_resources;
    }

    mlx5_init_cqes(cq_ring_haddr, ncqes);

    DOCA_LOG(LOG_DEBUG, "Create CQ memcpy cq_ring_haddr %p into gpu_umem_dev_ptr %p size %d\n",
             (void *)(cq_ring_haddr), (*gpu_umem_dev_ptr), external_umem_size);

    status_cuda = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaMemcpy(
        (*gpu_umem_dev_ptr), (void *)(cq_ring_haddr), external_umem_size, cudaMemcpyDefault));
    if (status_cuda != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to cudaMempy gpu cq cq ring buffer ret %d", status_cuda);
        goto destroy_resources;
    }

    free(cq_ring_haddr);
    cq_ring_haddr = nullptr;

    status =
        create_gpu_umem(gpu_dev, ibpd, mreg_type, external_umem_size, *gpu_umem_dev_ptr, gpu_umem);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "create_gpu_umem failed with %d", status);
        goto destroy_resources;
    }

    status = doca_verbs_cq_attr_set_external_umem(verbs_cq_attr, *gpu_umem, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs cq external umem");
        goto destroy_resources;
    }

    dbr_umem_align_sz = CUDA_ROUND_UP(DBR_SIZE, priv_get_page_size());
    status = doca_gpu_mem_alloc(gpu_dev, dbr_umem_align_sz, priv_get_page_size(),
                                DOCA_GPU_MEM_TYPE_GPU, (void **)gpu_umem_dbr_dev_ptr, nullptr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to alloc gpu memory for external umem qp");
        goto destroy_resources;
    }

    status = create_gpu_umem(gpu_dev, ibpd, mreg_type, dbr_umem_align_sz, *gpu_umem_dbr_dev_ptr,
                             gpu_umem_dbr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "create_gpu_umem failed with %d", status);
        goto destroy_resources;
    }

    status = doca_verbs_cq_attr_set_external_dbr_umem(verbs_cq_attr, *gpu_umem_dbr, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs cq external dbr umem");
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

    if (external_uar != NULL) {
        status = doca_verbs_cq_attr_set_external_uar(verbs_cq_attr, external_uar);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to set doca verbs cq external uar");
            goto destroy_resources;
        }
    }

    status = doca_verbs_cq_create(ibctx, verbs_cq_attr, &new_cq);
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

    if (*gpu_umem != NULL) {
        tmp_status = doca_verbs_umem_destroy(*gpu_umem);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy gpu ring buffer umem");
    }

    if (*gpu_umem_dbr != NULL) {
        tmp_status = doca_verbs_umem_destroy(*gpu_umem_dbr);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy gpu ring buffer umem");
    }

    if (cq_ring_haddr) {
        free(cq_ring_haddr);
    }

    if ((*gpu_umem_dev_ptr) != 0) {
        tmp_status = doca_gpu_mem_free(gpu_dev, (*gpu_umem_dev_ptr));
        if (tmp_status != DOCA_SUCCESS)
            DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of cq umem buffer");
    }

    if ((*gpu_umem_dbr_dev_ptr) != 0) {
        tmp_status = doca_gpu_mem_free(gpu_dev, (*gpu_umem_dbr_dev_ptr));
        if (tmp_status != DOCA_SUCCESS)
            DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of cq umem dbr buffer");
    }

    return status;
}

static uint32_t calc_qp_external_umem_size(uint32_t sq_nwqes) {
    uint32_t sq_ring_size = 0;

    if (sq_nwqes != 0) sq_ring_size = (uint32_t)(sq_nwqes * sizeof(struct doca_gpu_dev_verbs_wqe));

    return align_up_uint32(sq_ring_size, priv_get_page_size());
}

static doca_error_t create_qp(struct doca_gpu *gpu_dev, struct ibv_pd *ibpd,
                              enum doca_gpu_verbs_mem_reg_type mreg_type,
                              struct doca_verbs_cq *cq_sq, uint32_t sq_nwqe,
                              void **gpu_umem_dev_ptr, struct doca_verbs_umem **gpu_umem,
                              void **gpu_umem_dbr_dev_ptr, struct doca_verbs_umem **gpu_umem_dbr,
                              struct doca_verbs_uar *external_uar,
                              enum doca_gpu_dev_verbs_nic_handler req_nic_handler,
                              bool set_core_direct, struct doca_verbs_qp **verbs_qp,
                              enum doca_gpu_dev_verbs_nic_handler *out_nic_handler) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr = NULL;
    struct doca_verbs_qp *new_qp = NULL;
    uint32_t external_umem_size = 0;
    size_t dbr_umem_align_sz = align_up_uint32(DBR_SIZE, priv_get_page_size());
    struct ibv_context *ibctx = ibpd->context;
    enum doca_gpu_dev_verbs_nic_handler nic_handler = req_nic_handler;

    status = doca_verbs_qp_init_attr_create(&verbs_qp_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs qp attributes");
        return status;
    }

    status = doca_verbs_qp_init_attr_set_external_uar(verbs_qp_init_attr, external_uar);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set receive_max_sges");
        goto destroy_resources;
    }

    if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
        bool can_register = false;
        status = doca_gpu_verbs_can_gpu_register_uar(external_uar->get_reg_addr(), &can_register);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to check if UAR can be registered on GPU");
            goto destroy_resources;
        }

        nic_handler = can_register ? DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB
                                   : DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
    }

    external_umem_size = calc_qp_external_umem_size(sq_nwqe);

    status = doca_gpu_mem_alloc(gpu_dev, external_umem_size, priv_get_page_size(),
                                DOCA_GPU_MEM_TYPE_GPU, gpu_umem_dev_ptr, NULL);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to alloc gpu memory for external umem qp");
        goto destroy_resources;
    }

    status =
        create_gpu_umem(gpu_dev, ibpd, mreg_type, external_umem_size, *gpu_umem_dev_ptr, gpu_umem);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "create_gpu_umem failed with %d", status);
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_external_umem(verbs_qp_init_attr, *gpu_umem, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs qp external umem");
        goto destroy_resources;
    }

    if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
        *gpu_umem_dbr_dev_ptr = calloc(dbr_umem_align_sz, sizeof(uint8_t));
        if (*gpu_umem_dbr_dev_ptr == nullptr) {
            DOCA_LOG(LOG_ERR, "Failed to alloc gpu memory for external umem qp");
            goto destroy_resources;
        }
    } else {
        status = doca_gpu_mem_alloc(gpu_dev, dbr_umem_align_sz, priv_get_page_size(),
                                    DOCA_GPU_MEM_TYPE_GPU, gpu_umem_dbr_dev_ptr, NULL);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to alloc gpu memory for external umem qp");
            goto destroy_resources;
        }
    }

    status = create_gpu_umem(gpu_dev, ibpd, mreg_type, dbr_umem_align_sz, *gpu_umem_dbr_dev_ptr,
                             gpu_umem_dbr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "create_gpu_umem failed with %d", status);
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_external_dbr_umem(verbs_qp_init_attr, *gpu_umem_dbr, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs qp external dbr umem");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_pd(verbs_qp_init_attr, ibpd);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs PD");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_sq_wr(verbs_qp_init_attr, sq_nwqe);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set SQ size");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_rq_wr(verbs_qp_init_attr, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set RQ size");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_qp_type(verbs_qp_init_attr, DOCA_VERBS_QP_TYPE_RC);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set QP type");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_send_cq(verbs_qp_init_attr, cq_sq);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set doca verbs CQ");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_send_max_sges(verbs_qp_init_attr, MAX_SEND_SEGS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set send_max_sges");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_set_receive_max_sges(verbs_qp_init_attr, MAX_RECEIVE_SEGS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set receive_max_sges");
        goto destroy_resources;
    }

    if (set_core_direct) {
        status = doca_verbs_qp_init_attr_set_core_direct_master(verbs_qp_init_attr, 1);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to set core_direct");
            goto destroy_resources;
        }
    }

    status = doca_verbs_qp_create(ibctx, verbs_qp_init_attr, &new_qp);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs QP");
        goto destroy_resources;
    }

    status = doca_verbs_qp_init_attr_destroy(verbs_qp_init_attr);
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

    if (verbs_qp_init_attr != NULL) {
        tmp_status = doca_verbs_qp_init_attr_destroy(verbs_qp_init_attr);
        if (tmp_status != DOCA_SUCCESS)
            DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs QP attributes");
    }

    if (*gpu_umem != NULL) {
        tmp_status = doca_verbs_umem_destroy(*gpu_umem);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy gpu umem");
    }

    if ((*gpu_umem_dev_ptr) != 0) {
        tmp_status = doca_gpu_mem_free(gpu_dev, (*gpu_umem_dev_ptr));
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of umem");
    }

    if (*gpu_umem_dbr != NULL) {
        tmp_status = doca_verbs_umem_destroy(*gpu_umem_dbr);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy gpu umem");
    }

    if ((*gpu_umem_dbr_dev_ptr) != 0) {
        if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
            free(*gpu_umem_dbr_dev_ptr);
        } else {
            tmp_status = doca_gpu_mem_free(gpu_dev, (*gpu_umem_dbr_dev_ptr));
            if (tmp_status != DOCA_SUCCESS)
                DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of umem");
        }
    }

    // Immediately close dmabuf_fd after registration.
    // if (dmabuf_fd > 0) close(dmabuf_fd);

    return status;
}

doca_error_t doca_gpu_verbs_create_qp_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                                         struct doca_gpu_verbs_qp_hl **qp) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;

    if (qp_init_attr == nullptr || qp == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input value: qp_init_attr %p qp %p", (void *)qp_init_attr,
                 (void *)*qp);
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (qp_init_attr->gpu_dev == nullptr || qp_init_attr->ibpd == nullptr ||
        qp_init_attr->sq_nwqe == 0) {
        DOCA_LOG(LOG_ERR, "Invalid input value: gpu_dev %p ibpd %p sq_nwqe %d",
                 (void *)qp_init_attr->gpu_dev, (void *)qp_init_attr->ibpd, qp_init_attr->sq_nwqe);
        return DOCA_ERROR_INVALID_VALUE;
    }

    struct doca_gpu_verbs_qp_hl *qp_ =
        (struct doca_gpu_verbs_qp_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_hl));
    if (qp_ == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed alloc memory for high-level qp");
        return DOCA_ERROR_NO_MEMORY;
    }

    qp_->gpu_dev = qp_init_attr->gpu_dev;

    if (qp_init_attr->sq_nwqe > 0) {
        qp_init_attr->sq_nwqe =
            (uint32_t)doca_internal_utils_next_power_of_two(qp_init_attr->sq_nwqe);

        status = create_cq(qp_->gpu_dev, qp_init_attr->ibpd, qp_init_attr->mreg_type,
                           qp_init_attr->sq_nwqe, &qp_->cq_sq_umem_gpu_ptr, &qp_->cq_sq_umem,
                           &qp_->cq_sq_umem_dbr_gpu_ptr, &qp_->cq_sq_umem_dbr, NULL, &qp_->cq_sq);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create doca verbs cq");
            goto exit_error;
        }
    }

    qp_->nic_handler = qp_init_attr->nic_handler;

    status = create_uar(qp_init_attr->ibpd->context, qp_->nic_handler, &qp_->external_uar, true);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs uar");
        goto exit_error;
    }

    status = create_qp(qp_->gpu_dev, qp_init_attr->ibpd, qp_init_attr->mreg_type, qp_->cq_sq,
                       qp_init_attr->sq_nwqe, &qp_->qp_umem_gpu_ptr, &qp_->qp_umem,
                       &qp_->qp_umem_dbr_gpu_ptr, &qp_->qp_umem_dbr, qp_->external_uar,
                       qp_init_attr->nic_handler, false, &qp_->qp, &qp_->nic_handler);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs qp");
        goto exit_error;
    }

    status = doca_gpu_verbs_export_qp(qp_->gpu_dev, qp_->qp, qp_->nic_handler, qp_->qp_umem_gpu_ptr,
                                      qp_->cq_sq, &qp_->qp_gverbs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create GPU verbs QP");
        return status;
    }

    *qp = qp_;

    return DOCA_SUCCESS;

exit_error:
    if (qp_->external_uar != NULL) {
        tmp_status = doca_verbs_uar_destroy(qp_->external_uar);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs UAR");
    }

    free(qp_);
    return status;
}

static doca_error_t doca_gpu_verbs_destroy_qp_hl_internal(struct doca_gpu_verbs_qp_hl *qp) {
    doca_error_t status;

    if (qp == nullptr) return DOCA_ERROR_INVALID_VALUE;

    status = doca_gpu_verbs_unexport_qp(qp->gpu_dev, qp->qp_gverbs);
    if (status != DOCA_SUCCESS)
        DOCA_LOG(LOG_ERR, "Failed to destroy doca gpu thread argument cq memory");

    status = doca_verbs_qp_destroy(qp->qp);
    if (status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs QP");

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
        if (qp->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
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

        if (qp->cq_sq_umem_dbr != NULL) {
            status = doca_verbs_umem_destroy(qp->cq_sq_umem_dbr);
            if (status != DOCA_SUCCESS)
                DOCA_LOG(LOG_ERR, "Failed to destroy gpu sq cq ring buffer umem");
        }

        if (qp->cq_sq_umem_dbr_gpu_ptr != 0) {
            status = doca_gpu_mem_free(qp->gpu_dev, qp->cq_sq_umem_dbr_gpu_ptr);
            if (status != DOCA_SUCCESS)
                DOCA_LOG(LOG_ERR, "Failed to destroy gpu memory of sq cq umem dbr buffer");
        }
    }

    memset(qp, 0, sizeof(*qp));

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_destroy_qp_hl(struct doca_gpu_verbs_qp_hl *qp) {
    if (qp == nullptr) return DOCA_ERROR_INVALID_VALUE;

    doca_gpu_verbs_destroy_qp_hl_internal(qp);
    free(qp);

    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_create_qp_group_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                                               struct doca_gpu_verbs_qp_group_hl **qpg) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;

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

    if (qp_init_attr->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
        DOCA_LOG(LOG_ERR, "BlueFlame not supported with QP group");
        return DOCA_ERROR_INVALID_VALUE;
    }

    struct doca_gpu_verbs_qp_group_hl *qpg_ =
        (struct doca_gpu_verbs_qp_group_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_group_hl));
    if (qpg_ == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed alloc memory for high-level qp");
        return DOCA_ERROR_NO_MEMORY;
    }

    /********** Create main QP **********/

    qpg_->qp_main.gpu_dev = qp_init_attr->gpu_dev;

    status = create_uar(qp_init_attr->ibpd->context, qpg_->qp_main.nic_handler,
                        &qpg_->qp_main.external_uar, true);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs uar");
        goto exit_error;
    }

    if (qp_init_attr->sq_nwqe > 0) {
        qp_init_attr->sq_nwqe =
            (uint32_t)doca_internal_utils_next_power_of_two(qp_init_attr->sq_nwqe);

        status = create_cq(qpg_->qp_main.gpu_dev, qp_init_attr->ibpd, qp_init_attr->mreg_type,
                           qp_init_attr->sq_nwqe, &qpg_->qp_main.cq_sq_umem_gpu_ptr,
                           &qpg_->qp_main.cq_sq_umem, &qpg_->qp_main.cq_sq_umem_dbr_gpu_ptr,
                           &qpg_->qp_main.cq_sq_umem_dbr, qpg_->qp_main.external_uar,
                           &qpg_->qp_main.cq_sq);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create doca verbs cq");
            goto exit_error;
        }
    }

    status = create_qp(
        qpg_->qp_main.gpu_dev, qp_init_attr->ibpd, qp_init_attr->mreg_type, qpg_->qp_main.cq_sq,
        qp_init_attr->sq_nwqe, &qpg_->qp_main.qp_umem_gpu_ptr, &qpg_->qp_main.qp_umem,
        &qpg_->qp_main.qp_umem_dbr_gpu_ptr, &qpg_->qp_main.qp_umem_dbr, qpg_->qp_main.external_uar,
        qp_init_attr->nic_handler, false, &qpg_->qp_main.qp, &qpg_->qp_main.nic_handler);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs qp");
        goto exit_error;
    }

    status = doca_gpu_verbs_export_qp(qpg_->qp_main.gpu_dev, qpg_->qp_main.qp,
                                      qpg_->qp_main.nic_handler, qpg_->qp_main.qp_umem_gpu_ptr,
                                      qpg_->qp_main.cq_sq, &qpg_->qp_main.qp_gverbs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create GPU verbs QP");
        return status;
    }

    /********** Create companion QP **********/

    qpg_->qp_companion.gpu_dev = qp_init_attr->gpu_dev;
    qpg_->qp_companion.external_uar = qpg_->qp_main.external_uar;

    if (qp_init_attr->sq_nwqe > 0) {
        qp_init_attr->sq_nwqe =
            (uint32_t)doca_internal_utils_next_power_of_two(qp_init_attr->sq_nwqe);

        status =
            create_cq(qpg_->qp_companion.gpu_dev, qp_init_attr->ibpd, qp_init_attr->mreg_type,
                      qp_init_attr->sq_nwqe, &qpg_->qp_companion.cq_sq_umem_gpu_ptr,
                      &qpg_->qp_companion.cq_sq_umem, &qpg_->qp_companion.cq_sq_umem_dbr_gpu_ptr,
                      &qpg_->qp_companion.cq_sq_umem_dbr, qpg_->qp_companion.external_uar,
                      &qpg_->qp_companion.cq_sq);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create doca verbs cq");
            goto exit_error;
        }
    }

    status = create_qp(qpg_->qp_companion.gpu_dev, qp_init_attr->ibpd, qp_init_attr->mreg_type,
                       qpg_->qp_companion.cq_sq, qp_init_attr->sq_nwqe,
                       &qpg_->qp_companion.qp_umem_gpu_ptr, &qpg_->qp_companion.qp_umem,
                       &qpg_->qp_companion.qp_umem_dbr_gpu_ptr, &qpg_->qp_companion.qp_umem_dbr,
                       qpg_->qp_companion.external_uar, qp_init_attr->nic_handler, true,
                       &qpg_->qp_companion.qp, &qpg_->qp_companion.nic_handler);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs qp");
        goto exit_error;
    }

    status =
        doca_gpu_verbs_export_qp(qpg_->qp_companion.gpu_dev, qpg_->qp_companion.qp,
                                 qpg_->qp_companion.nic_handler, qpg_->qp_companion.qp_umem_gpu_ptr,
                                 qpg_->qp_companion.cq_sq, &qpg_->qp_companion.qp_gverbs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create GPU verbs QP");
        return status;
    }

    *qpg = qpg_;

    return DOCA_SUCCESS;

exit_error:
    if (qpg_->qp_main.external_uar != NULL) {
        tmp_status = doca_verbs_uar_destroy(qpg_->qp_main.external_uar);
        if (tmp_status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs UAR");
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
