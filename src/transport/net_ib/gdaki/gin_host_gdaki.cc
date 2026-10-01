/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <alloc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <mutex>

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"
#include "mlx5/mlx5prm.h"
#include "gin/gin_host.h"
#include "gin_host_gdaki.h"
#include "plugin/nccl_net.h"
#include "param.h"

#include "doca_gpunetio_host.h"
#include "nccl_device/gin/gdaki/gin_gdaki_device_host_common.h"
#include "gpucontext/gpucontext.h"
#include "../gin.h"
#include "../common.h"
#include "utils.h"

#define DOCACHECK(call) \
  do { \
    doca_error_t err = call; \
    if (err != DOCA_SUCCESS) { \
      /* Print the back trace*/ \
      WARN("DOCA failure %d", err); \
      return ncclSystemError; \
    } \
  } while (0)

#define DOCACHECKGOTO(call, RES, label) \
  do { \
    doca_error_t err = call; \
    if (err != DOCA_SUCCESS) { \
      /* Print the back trace*/ \
      WARN("DOCA failure %d", err); \
      RES = ncclSystemError; \
      goto label; \
    } \
  } while (0)

#define VERBS_TEST_DBR_SIZE (8)
#define MAX_PCI_ADDRESS_LEN 32U
#define DEFAULT_HOP_LIMIT (255)

NCCL_PARAM(GinGdakiNicHandler, "GIN_GDAKI_NIC_HANDLER", 0);
NCCL_PARAM(GinGdakiQpDepth, "GIN_GDAKI_QP_DEPTH", 128);
NCCL_PARAM(GinGdakiMaxDestRdAtomic, "GIN_GDAKI_MAX_DEST_RD_ATOMIC", -2);
NCCL_PARAM(GinGdakiMaxQpRdAtomic, "GIN_GDAKI_MAX_QP_RD_ATOMIC", -2);
NCCL_PARAM(GinGdakiLAGAwareDisable, "GIN_GDAKI_LAG_AWARE_DISABLE", 0);
NCCL_PARAM(GinGdakiCqType, "GIN_GDAKI_CQ_TYPE", 0);
NCCL_PARAM(GinErrorQuerySec, "GIN_ERROR_QUERY_SEC", 10);
NCCL_PARAM(GinIbOooAll, "GIN_IB_OOO_OPT", 0);
// Allocate a dedicated GIN-owned Q counter set and attach every GIN QP to it (diagnostic-only).
// Set to 0 to disable, leaving GIN QPs on the device default counter set (behavior unchanged).
NCCL_PARAM(GinGdakiQCounter, "GIN_GDAKI_Q_COUNTER", 1);
USE_NCCL_PARAM(ncclParamIbTimeout, uint8_t);
USE_NCCL_PARAM(ncclParamIbRetryCnt, uint8_t);
extern ncclResult_t ncclIbGetPkeyIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                                       int* pkeyIndex);
USE_NCCL_PARAM(ncclParamIbSl, int32_t);
extern int64_t ncclParamIbPciRelaxedOrdering();
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamDmaBufEnable();

static const int NCCL_IB_SL_DEFAULT = 0;
static const int NCCL_IB_TC_DEFAULT = 0;

static bool gdakiQpNeedsProxyProgress(const struct doca_gpu_verbs_qp* qp) {
  return qp->cpu_proxy || qp->cq_type == DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST;
}

static enum doca_verbs_qp_ordering_semantic gdakiOrderingSematic() {
  if (ncclParamGinIbOooAll() == 1) return DOCA_VERBS_QP_ORDERING_SEMANTIC_OOO_ALL;

  return DOCA_VERBS_QP_ORDERING_SEMANTIC_IBTA;
}

static inline bool gdakiRelaxedOrderingEnabled() {
  static bool hasCheckedRelaxedOrdering = false;
  static bool relaxedOrderingEnabled = false;

  static std::mutex lockMutex;
  std::lock_guard<std::mutex> lock(lockMutex);

  if (!hasCheckedRelaxedOrdering) {
    int roMode = ncclParamIbPciRelaxedOrdering();
    ncclResult_t r = ncclInternalError;
    if (roMode == 1 || roMode == 2) {
      // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
      r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
    }

    relaxedOrderingEnabled = (r != ncclInternalError);
    hasCheckedRelaxedOrdering = true;
  }
  return relaxedOrderingEnabled;
}

static ncclResult_t gdakiRegMrDmaBuf(struct ibv_mr** mr, struct ibv_pd* pd, void* addr, size_t length, int access) {
  int status = 0;
  int dmabuf_fd = -1;

  if (ncclParamDmaBufEnable() == 0) return ncclInvalidUsage;

#if CUDA_VERSION >= 11070
  static size_t host_page_size = sysconf(_SC_PAGESIZE);
  size_t aligned_size = length;
  ALIGN_SIZE(aligned_size, host_page_size);

#if CUDA_VERSION >= 12080
  if (ncclParamIbDataDirect()) {
    status = pfn_cuMemGetHandleForAddressRange((void*)&dmabuf_fd, (CUdeviceptr)addr, aligned_size,
                                               CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                               CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);
    if (status) {
      INFO(NCCL_NET,
           "Failed to get DMA-BUF handle for address range with type PCIE, error=%d. Trying a "
           "different method.",
           status);
      goto try_legacy;
    }
    status =
      wrap_mlx5dv_reg_dmabuf_mr(mr, pd, 0, aligned_size, 0, dmabuf_fd, access, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
    if (status) {
      INFO(NCCL_NET,
           "Failed to register memory with DMA-BUF and data direct, error=%d. Trying a different "
           "method.",
           status);
      close(dmabuf_fd);
      dmabuf_fd = -1;
    } else goto out;
  }
try_legacy:

#endif

  CUCHECK(cuMemGetHandleForAddressRange((void*)&dmabuf_fd, (CUdeviceptr)addr, aligned_size,
                                        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
  status = wrap_ibv_reg_dmabuf_mr(mr, pd, 0, aligned_size, 0, dmabuf_fd, access);
  if (status) INFO(NCCL_NET, "Failed to register memory with DMA-BUF, error=%d. Trying a different method.", status);
#else
  status = ncclInvalidUsage;
#endif

#if CUDA_VERSION >= 12080
out:
#endif
  if (dmabuf_fd >= 0) {
    close(dmabuf_fd);
  }
  return (ncclResult_t)status;
}

static ncclResult_t gdakiRegMr(struct ibv_mr** mr, struct ibv_pd* pd, void* addr, size_t length, int access,
                               bool force_strict_ordering = false) {
  int status = 0;

  if (!force_strict_ordering && gdakiRelaxedOrderingEnabled()) access |= IBV_ACCESS_RELAXED_ORDERING;

  NOWARN(status = gdakiRegMrDmaBuf(mr, pd, addr, length, access), NCCL_NET);
  if (status == ncclSuccess) return ncclSuccess;

  NCCLCHECK(wrap_ibv_reg_mr_iova2(mr, pd, addr, length, 0, access));
  return ncclSuccess;
}

template <typename T>
class GdakiHostGPUMemHandle {
private:
  CUmemGenericAllocationHandle cumemhandle;
  unsigned int num_elements;

public:
  T* host_buf;
  T* gpu_buf;

  ncclResult_t allocate(unsigned int num_elements) {
    this->host_buf = (T*)calloc(num_elements, sizeof(T));
    EQCHECK(this->host_buf, nullptr);

    NCCLCHECK(ncclCuMemAlloc((void**)&this->gpu_buf, &this->cumemhandle, CU_MEM_HANDLE_TYPE_NONE,
                             num_elements * sizeof(T), nullptr));

    this->num_elements = num_elements;

    return ncclSuccess;
  }

  ncclResult_t deallocate() {
    if (this->host_buf != nullptr) {
      free(this->host_buf);
      this->host_buf = nullptr;
    }
    if (this->gpu_buf != nullptr) {
      NCCLCHECK(ncclCuMemFree(this->gpu_buf, nullptr));
      this->gpu_buf = nullptr;
    }
    return ncclSuccess;
  }

  ncclResult_t copy_h_to_d() {
    NCCLCHECK(ncclCudaMemcpy<T>(this->gpu_buf, this->host_buf, this->num_elements));
    return ncclSuccess;
  }

  ncclResult_t copy_d_to_h() {
    NCCLCHECK(ncclCudaMemcpy<T>(this->host_buf, this->gpu_buf, this->num_elements));
    return ncclSuccess;
  }

  GdakiHostGPUMemHandle() : cumemhandle(0), num_elements(0), host_buf(nullptr), gpu_buf(nullptr) {};

  ~GdakiHostGPUMemHandle() {
     // Should only be used in error cleanup path as it ignores return code
    this->deallocate();
  }
};

template <typename T>
class GdakiGlobalGPUBufferTable {
private:
  CUmemGenericAllocationHandle cumemhandle;
  unsigned int num_elements;
  unsigned int next_unused_idx;
  unsigned int num_ranks;
  GdakiHostGPUMemHandle<__be32> rkeys_hd_mhandle;

public:
  T* gpu_ptr;
  struct ibv_mr* mr;

  ncclResult_t allocate(unsigned int num_elements, unsigned int num_ranks) {
    this->num_elements = num_elements;
    this->num_ranks = num_ranks;
    this->next_unused_idx = 0;
    if (num_elements == 0) return ncclSuccess;

    NCCLCHECK(ncclCuMemAlloc((void**)&this->gpu_ptr, &this->cumemhandle, CU_MEM_HANDLE_TYPE_NONE,
                             num_elements * sizeof(T), nullptr));
    CUDACHECK(cudaMemset(this->gpu_ptr, 0, num_elements * sizeof(T)));
    NCCLCHECK(this->rkeys_hd_mhandle.allocate(num_ranks));
    return ncclSuccess;
  }

  ncclResult_t deallocate() {
    if (this->gpu_ptr != nullptr) {
      NCCLCHECK(ncclCuMemFree(this->gpu_ptr, nullptr));
      this->gpu_ptr = nullptr;
    }
    return ncclSuccess;
  }

  ncclResult_t register_mr(struct ibv_pd* ib_pd, bool force_strict_ordering = false) {
    if (this->num_elements == 0) return ncclSuccess;
    NCCLCHECK(gdakiRegMr(&this->mr, ib_pd, this->gpu_ptr, this->num_elements * sizeof(T),
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_ATOMIC,
                         force_strict_ordering));
    return ncclSuccess;
  }

  ncclResult_t deregister_mr() {
    if (this->mr != nullptr) {
      NCCLCHECK(wrap_ibv_dereg_mr(this->mr));
      this->mr = nullptr;
    }
    return ncclSuccess;
  }

  ncclResult_t exchange_info(struct ncclGinIbCollComm* cComm) {
    if (this->num_elements == 0) return ncclSuccess;
    __be32 rkey = htobe32(this->mr->rkey);
    NCCLCHECK(cComm->allGather(cComm, &rkey, this->rkeys_hd_mhandle.host_buf, sizeof(__be32)));
    NCCLCHECK(this->rkeys_hd_mhandle.copy_h_to_d());
    return ncclSuccess;
  }

  ncclResult_t allocate_elements(unsigned int num_elements, unsigned int* out_start_idx) {
    if (this->next_unused_idx + num_elements > this->num_elements) {
      WARN("Not enough space to get elements");
      return ncclInvalidUsage;
    }

    *out_start_idx = this->next_unused_idx;
    this->next_unused_idx += num_elements;

    return ncclSuccess;
  }

  void free_elements(unsigned int start_idx, unsigned int num_elements) {
    // No op for now as we don't allow reusing elements.
  }

  uint32_t* get_rkeys_d() {
    return this->rkeys_hd_mhandle.gpu_buf;
  }

  GdakiGlobalGPUBufferTable() : cumemhandle(0), num_elements(0), next_unused_idx(0), gpu_ptr(nullptr), mr(nullptr) {};
  ~GdakiGlobalGPUBufferTable() {
     // Should only be used in error cleanup path as it ignores return codes
    this->deregister_mr();
    this->deallocate();
  }
};

struct gdaki_mem_handle {
  int type;
  struct ibv_mr* mr;
  GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle>* gdaki_mhandle_hd_mhandle;
  GdakiHostGPUMemHandle<uint32_t>* rkeys_hd_mhandle;
};

struct gdaki_exch_info {
  int lid;
  int qpn;
  union ibv_gid gid;
  struct doca_verbs_gid vgid;
  int gid_index;

  char hostname[HOST_NAME_MAX];
  char ib_dev_name[IBV_SYSFS_NAME_MAX];
  int cuda_id;
  int rank;
  enum ibv_mtu active_mtu;
  enum doca_verbs_qp_ordering_semantic ordering_semantic;
};

struct gdaki_context {
  int cuda_id;
  doca_gpu_t* gdev;
  struct ibv_device* ib_dev;
  doca_dev_t* ndev;
  doca_verbs_ah_attr_t* ah; /* DOCA Verbs address handle */
  struct ibv_device_attr ib_dev_attr;
  struct doca_verbs_gid gid;

  char hostname[HOST_NAME_MAX];
  char ib_dev_name[IBV_SYSFS_NAME_MAX];
  int rank;

  union ibv_gid rgid;
  struct ibv_port_attr port_attr;
  uint8_t port_num;
  int gid_index;
  int pkey_index;

  bool needCompanion;
  uint32_t qp_rq_size;
  uint32_t qp_sq_size;
  struct doca_gpu_verbs_qp_group_list_hl* gqp_group_list;
  struct doca_gpu_verbs_qp_list_hl* gqp_list;
  struct doca_gpu_verbs_qp_list_hl* self_gqp_list;
  struct doca_gpu_verbs_qp_list_hl* self_companion_gqp_list;
  struct doca_gpu_verbs_qp_group_hl** gqp_groups;
  struct doca_gpu_verbs_qp_hl** gqps;
  struct doca_gpu_verbs_qp_hl** companion_gqps;

  /* Dedicated GIN Q counter set. ginCounterSetId is non-zero only while GIN QPs are attached to it. The DevX
   * object outlives every GIN QP since a Q counter set cannot be deallocated while a QP still references it. */
  struct mlx5dv_devx_obj* ginQCounter;
  uint32_t ginCounterSetId;

  GdakiGlobalGPUBufferTable<uint64_t>* counters_table;
  GdakiGlobalGPUBufferTable<uint64_t>* signals_table;
  struct ncclGinGdakiGPUContext* gin_gdaki_gpu_ctx_host_staging; // formatted according to current version
  GdakiHostGPUMemHandle<char>* gin_gdaki_gpu_ctx_hd_mhandle; // formatted according to backendVersion
  int backendVersion;
  struct {
    void* addr;
    struct ibv_mr* mr;
    CUmemGenericAllocationHandle mhandle;
  } sink_buffer;
  uint64_t last_error_query_time;

  uint64_t* last_issued_get;
  uint64_t* last_visible_get;

  struct ncclGinIbCollComm* collComm;
  ncclNetDeviceHandle_t* devHandle;
  int nContexts;

  doca_verbs_comp_channel_t* docaEvent;
};

static const struct {
  const char* name;
  int dw;
} gdakiQCounterFields[] = {
  {"rx_write_requests", NCCL_MLX5_Q_COUNTER_RX_WRITE_REQUESTS_DW},
  {"rx_read_requests", NCCL_MLX5_Q_COUNTER_RX_READ_REQUESTS_DW},
  {"rx_atomic_requests", NCCL_MLX5_Q_COUNTER_RX_ATOMIC_REQUESTS_DW},
  {"out_of_buffer", NCCL_MLX5_Q_COUNTER_OUT_OF_BUFFER_DW},
  {"out_of_sequence", NCCL_MLX5_Q_COUNTER_OUT_OF_SEQUENCE_DW},
  {"duplicate_request", NCCL_MLX5_Q_COUNTER_DUPLICATE_REQUEST_DW},
  {"rnr_nak_retry_err", NCCL_MLX5_Q_COUNTER_RNR_NAK_RETRY_ERR_DW},
  {"packet_seq_err", NCCL_MLX5_Q_COUNTER_PACKET_SEQ_ERR_DW},
  {"implied_nak_seq_err", NCCL_MLX5_Q_COUNTER_IMPLIED_NAK_SEQ_ERR_DW},
  {"local_ack_timeout_err", NCCL_MLX5_Q_COUNTER_LOCAL_ACK_TIMEOUT_ERR_DW},
  {"resp_cqe_error", NCCL_MLX5_Q_COUNTER_RESP_CQE_ERROR_DW},
  {"req_cqe_error", NCCL_MLX5_Q_COUNTER_REQ_CQE_ERROR_DW},
  {"req_transport_retries_exceeded", NCCL_MLX5_Q_COUNTER_REQ_TRANSPORT_RETRIES_EXCEEDED_DW},
  {"resp_cqe_flush_error", NCCL_MLX5_Q_COUNTER_RESP_CQE_FLUSH_ERROR_DW},
  {"req_cqe_flush_error", NCCL_MLX5_Q_COUNTER_REQ_CQE_FLUSH_ERROR_DW},
};

// Best-effort: on failure GIN QPs stay on the device default counter set.
static void gdakiQCounterCreate(struct gdaki_context* ctx, struct ibv_context* context) {
  uint32_t in[NCCL_MLX5_ALLOC_Q_COUNTER_IN_SIZE_DW] = {};
  uint32_t out[NCCL_MLX5_ALLOC_Q_COUNTER_OUT_SIZE_DW] = {};
  in[NCCL_MLX5_CMD_IN_OPCODE_DW] = htobe32(NCCL_MLX5_CMD_OP_ALLOC_Q_COUNTER << 16);
  if (wrap_mlx5dv_devx_obj_create(&ctx->ginQCounter, context, in, sizeof(in), out, sizeof(out)) != ncclSuccess) {
    INFO(NCCL_NET,
         "[%d] GIN GDAKI Q counter not available (status=%#x syndrome=%#x); GIN QPs will use the default counter set",
         ctx->rank, be32toh(out[NCCL_MLX5_CMD_OUT_STATUS_DW]) >> 24, be32toh(out[NCCL_MLX5_CMD_OUT_SYNDROME_DW]));
    return;
  }
  ctx->ginCounterSetId = be32toh(out[NCCL_MLX5_ALLOC_Q_COUNTER_OUT_COUNTER_SET_ID_DW]) & 0xff;
  INFO(NCCL_NET, "GIN GDAKI counter set: rank=%d dev=%s port=%u counterSetId=%u", ctx->rank, ctx->ib_dev_name,
       ctx->port_num, ctx->ginCounterSetId);
}

// Returns the totals of the GIN Q counter set as " counterSetTotals: name=value ...", or an empty string when GIN QPs
// are not attached to it or it cannot be queried.
static const char* gdakiQCounterString(struct gdaki_context* ctx, char* buf, size_t size) {
  buf[0] = '\0';
  if (ctx->ginCounterSetId == 0) return buf;

  uint32_t in[NCCL_MLX5_QUERY_Q_COUNTER_IN_SIZE_DW] = {};
  uint32_t out[NCCL_MLX5_QUERY_Q_COUNTER_OUT_SIZE_DW] = {};
  in[NCCL_MLX5_CMD_IN_OPCODE_DW] = htobe32(NCCL_MLX5_CMD_OP_QUERY_Q_COUNTER << 16);
  in[NCCL_MLX5_QUERY_Q_COUNTER_IN_COUNTER_SET_ID_DW] = htobe32(ctx->ginCounterSetId & 0xff);
  if (wrap_mlx5dv_devx_obj_query(ctx->ginQCounter, in, sizeof(in), out, sizeof(out)) != ncclSuccess) {
    INFO(NCCL_NET, "[%d] GIN GDAKI could not query counterSetId=%u (status=%#x syndrome=%#x)", ctx->rank,
         ctx->ginCounterSetId, be32toh(out[NCCL_MLX5_CMD_OUT_STATUS_DW]) >> 24,
         be32toh(out[NCCL_MLX5_CMD_OUT_SYNDROME_DW]));
    return buf;
  }

  size_t len = snprintf(buf, size, " counterSetTotals:");
  for (const auto& field : gdakiQCounterFields) {
    if (len >= size) break;
    len += snprintf(buf + len, size - len, " %s=%u", field.name, be32toh(out[field.dw]));
  }
  return buf;
}

static void gdakiFillExchInfo(struct gdaki_exch_info* exch_info, struct gdaki_context* gdaki_ctx,
                              struct doca_gpu_verbs_qp_hl* gqp) {
  exch_info->lid = gdaki_ctx->port_attr.lid;
  doca_verbs_qp_get_qpn(gqp->qp, (uint32_t*)&exch_info->qpn);
  memcpy(exch_info->gid.raw, gdaki_ctx->rgid.raw, sizeof(union ibv_gid));
  memcpy(exch_info->vgid.raw, gdaki_ctx->rgid.raw, sizeof(union ibv_gid));
  exch_info->gid_index = gdaki_ctx->gid_index;
  snprintf(exch_info->hostname, IBV_SYSFS_NAME_MAX, "%s", gdaki_ctx->hostname);
  snprintf(exch_info->ib_dev_name, IBV_SYSFS_NAME_MAX, "%s", gdaki_ctx->ib_dev_name);
  exch_info->cuda_id = gdaki_ctx->cuda_id;
  exch_info->rank = gdaki_ctx->rank;
  exch_info->active_mtu = gdaki_ctx->port_attr.active_mtu;
  exch_info->ordering_semantic = gdakiOrderingSematic();
}

static ncclResult_t gdakiCreateVerbsAh(struct gdaki_context* ctx, int ib_sl, int ib_tc, int ib_gid_index) {
  ncclResult_t status = ncclSuccess;

  DOCACHECK(doca_verbs_ah_attr_create(ctx->ndev, &ctx->ah));

  if (ctx->port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_sl(ctx->ah, ib_sl), status, destroy_verbs_ah);
  } else {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_traffic_class(ctx->ah, ib_tc), status, destroy_verbs_ah);
    DOCACHECKGOTO(doca_verbs_ah_attr_set_addr_type(ctx->ah, DOCA_VERBS_ADDR_TYPE_IPv4), status, destroy_verbs_ah);
  }

  // set_port_num?
  DOCACHECKGOTO(doca_verbs_ah_attr_set_sgid_index(ctx->ah, ib_gid_index), status, destroy_verbs_ah);

  return ncclSuccess;

destroy_verbs_ah:
  DOCACHECK(doca_verbs_ah_attr_destroy(ctx->ah));
  return status;
}

static bool gdakiIsValidActiveMtu(enum ibv_mtu active_mtu) {
  switch (active_mtu) {
  case IBV_MTU_256:
  case IBV_MTU_512:
  case IBV_MTU_1024:
  case IBV_MTU_2048:
  case IBV_MTU_4096:
    return true;
  default:
    return false;
  }
}

static ncclResult_t gdakiGetPathMtu(enum ibv_mtu local_active_mtu, enum ibv_mtu remote_active_mtu,
                                    enum doca_verbs_mtu_size* path_mtu) {
  if (!gdakiIsValidActiveMtu(local_active_mtu) || !gdakiIsValidActiveMtu(remote_active_mtu)) {
    WARN("Unexpected active_mtu value (local %d, remote %d)", local_active_mtu, remote_active_mtu);
    return ncclInternalError;
  }

  enum ibv_mtu active_mtu = local_active_mtu < remote_active_mtu ? local_active_mtu : remote_active_mtu;
  switch (active_mtu) {
  case IBV_MTU_256:
    *path_mtu = DOCA_VERBS_MTU_SIZE_256_BYTES;
    break;
  case IBV_MTU_512:
    *path_mtu = DOCA_VERBS_MTU_SIZE_512_BYTES;
    break;
  case IBV_MTU_1024:
    *path_mtu = DOCA_VERBS_MTU_SIZE_1K_BYTES;
    break;
  case IBV_MTU_2048:
    *path_mtu = DOCA_VERBS_MTU_SIZE_2K_BYTES;
    break;
  case IBV_MTU_4096:
    *path_mtu = DOCA_VERBS_MTU_SIZE_4K_BYTES;
    break;
  default:
    WARN("Unexpected active_mtu value %d", active_mtu);
    return ncclInternalError;
  }

  return ncclSuccess;
}

static ncclResult_t gdakiConnectQp(struct gdaki_context* ctx, struct doca_gpu_verbs_qp_hl* gqp,
                                   struct gdaki_exch_info* exch_info, uint8_t lagTxPortAffinity = 0) {
  ncclResult_t status = ncclSuccess;
  doca_verbs_qp_attr_t* verbs_qp_attr = nullptr;
  int max_dest_rd_atomic =
    ncclParamGinGdakiMaxDestRdAtomic() > 0 ? ncclParamGinGdakiMaxDestRdAtomic() : ctx->ib_dev_attr.max_qp_rd_atom;
  int max_qp_rd_atomic =
    ncclParamGinGdakiMaxQpRdAtomic() > 0 ? ncclParamGinGdakiMaxQpRdAtomic() : ctx->ib_dev_attr.max_qp_rd_atom;
  enum doca_verbs_mtu_size path_mtu;
  int dlid = exch_info->lid;

  int attrMask = DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                 DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM;

  if (ctx->port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    bool sameSubnet = (ncclIbExtractLocalSubnetPrefix(ctx->rgid.global.subnet_prefix) ==
                       ncclIbExtractLocalSubnetPrefix(exch_info->gid.global.subnet_prefix));
    bool needGlobal = !sameSubnet || (ctx->port_attr.flags & IBV_QPF_GRH_REQUIRED);
    if (needGlobal) {
      if (!sameSubnet) {
        uint16_t flid = ncclIbExtractFlid(&exch_info->gid);
        if (flid == 0) {
          WARN("Warning: remote FLID configured as zero even when endpoints are on different subnets, using dlid as "
               "fallback");
          // Note: We set dlid = exch_info->lid above.
        } else {
          dlid = flid;
        }
      }
      DOCACHECK(doca_verbs_ah_attr_set_addr_type(ctx->ah, DOCA_VERBS_ADDR_TYPE_IB_GRH));
      DOCACHECK(doca_verbs_ah_attr_set_hop_limit(ctx->ah, DEFAULT_HOP_LIMIT));
    } else {
      DOCACHECK(doca_verbs_ah_attr_set_addr_type(ctx->ah, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH));
      DOCACHECK(doca_verbs_ah_attr_set_hop_limit(ctx->ah, 0));
    }
  } else {
    DOCACHECK(doca_verbs_ah_attr_set_hop_limit(ctx->ah, DEFAULT_HOP_LIMIT));
  }

  NCCLCHECK(gdakiGetPathMtu(ctx->port_attr.active_mtu, exch_info->active_mtu, &path_mtu));
  enum doca_verbs_qp_ordering_semantic ordering_semantic = gdakiOrderingSematic();

  if (ordering_semantic != exch_info->ordering_semantic) {
    uint32_t qpn;
    DOCACHECK(doca_verbs_qp_get_qpn(gqp->qp, &qpn));
    WARN("Can't connect local QP %x ordering_semantic %x with remote QP %x "
         "ordering_semantic %x are ordering_semantic value is different",
         qpn, ordering_semantic, exch_info->qpn, exch_info->ordering_semantic);
    return ncclInvalidArgument;
  }

  DOCACHECK(doca_verbs_ah_attr_set_gid(ctx->ah, exch_info->vgid));
  DOCACHECK(doca_verbs_ah_attr_set_dlid(ctx->ah, dlid));
  DOCACHECK(doca_verbs_qp_attr_create(&verbs_qp_attr));
  if (ctx->ginCounterSetId != 0) {
    doca_error_t qcErr = doca_verbs_qp_attr_set_counter_set_id(verbs_qp_attr, ctx->ginCounterSetId);
    if (qcErr != DOCA_SUCCESS) {
      // Best-effort: stop attaching the remaining GIN QPs, but keep the Q counter set allocated until
      // teardown since QPs connected earlier may already reference it.
      INFO(NCCL_NET,
           "[%d] GIN GDAKI could not program counter_set_id=%u on a QP (err=%d); remaining GIN QPs will use the "
           "default counter set",
           ctx->rank, ctx->ginCounterSetId, qcErr);
      ctx->ginCounterSetId = 0;
    }
  }
  DOCACHECKGOTO(doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, path_mtu), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, 0), status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, 0), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_port_num(verbs_qp_attr, ctx->port_num), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, ncclParamIbTimeout()), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, ncclParamIbRetryCnt()), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, 7), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, 12), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_INIT), status,
                destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, 1), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, 1), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_atomic_mode(verbs_qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC), status,
                destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, ctx->ah), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, exch_info->qpn), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_pkey_index(verbs_qp_attr, ctx->pkey_index), status, destroy_verbs_qp_attr);

  if (lagTxPortAffinity > 0) {
    DOCACHECKGOTO(doca_verbs_qp_attr_set_lag_tx_port_affinity(verbs_qp_attr, lagTxPortAffinity), status,
                  destroy_verbs_qp_attr);
    attrMask |= DOCA_VERBS_QP_ATTR_LAG_TX_PORT_AFFINITY;
  }

  DOCACHECKGOTO(doca_verbs_qp_modify(gqp->qp, verbs_qp_attr, attrMask), status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_attr_set_max_dest_rd_atomic(verbs_qp_attr, max_dest_rd_atomic), status,
                destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTR), status,
                destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(gqp->qp, verbs_qp_attr,
                                     DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                       DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                       DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
                                       DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC | DOCA_VERBS_QP_ATTR_ATOMIC_MODE),
                status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_attr_set_max_rd_atomic(verbs_qp_attr, max_qp_rd_atomic), status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTS), status,
                destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(gqp->qp, verbs_qp_attr,
                                     DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                       DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                                       DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC),
                status, destroy_verbs_qp_attr);

  DOCACHECK(doca_verbs_qp_attr_destroy(verbs_qp_attr));

  return ncclSuccess;

destroy_verbs_qp_attr:
  WARN("[%d] Failed to connect GDAKI QP: local_node_name=%s local_gpu_id=%d local_nic_name=%s "
       "remote_rank=%d remote_node_name=%s remote_gpu_id=%d remote_nic_name=%s",
       ctx->rank, ctx->hostname, ctx->cuda_id, ctx->ib_dev_name, exch_info->rank, exch_info->hostname,
       exch_info->cuda_id, exch_info->ib_dev_name);
  DOCACHECK(doca_verbs_qp_attr_destroy(verbs_qp_attr));
  return status;
}

NCCL_PARAM(GinGdakiUseReliableDB, "GDAKI_USE_RELIABLE_DB", 0);
NCCL_PARAM(GinGdakiForceMcst, "GDAKI_FORCE_MCST", 0);

static ncclResult_t gdakiCheckDeviceSupport(doca_gpu_t* gpuDev, doca_dev_t* netDev,
                                            enum doca_gpu_dev_verbs_nic_handler nicHandler, int rank) {
  doca_error_t docaStatus = doca_gpu_nic_cap_is_gpu_mem_umem_supported(gpuDev, netDev);
  if (docaStatus != DOCA_SUCCESS &&
      !(gpuDev->type == DOCA_GPU_LIB_TYPE_SDK && netDev->type == DOCA_VERBS_SDK_LIB_TYPE_SDK &&
        docaStatus == DOCA_ERROR_NOT_SUPPORTED)) {
    WARN("[%d] GIN/GDAKI: GPU memory UMEM is required but is not supported by the selected GPU/NIC pair: DOCA error %d",
         rank, docaStatus);
    return ncclInvalidUsage;
  }

  if (nicHandler & DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_GPU_SM) {
    docaStatus = doca_gpu_nic_cap_is_nic_handler_gpu_sm_db_supported(gpuDev, netDev);
    if (docaStatus != DOCA_SUCCESS &&
        !(gpuDev->type == DOCA_GPU_LIB_TYPE_SDK && netDev->type == DOCA_VERBS_SDK_LIB_TYPE_SDK &&
          docaStatus == DOCA_ERROR_NOT_SUPPORTED)) {
      WARN(
        "[%d] GIN/GDAKI: the requested GPU SM NIC handler is not supported by the selected GPU/NIC pair: DOCA error %d",
        rank, docaStatus);
      return ncclInvalidUsage;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiCheckDeviceSupport(void* collComm) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  ncclResult_t status = ncclSuccess;
  doca_gpu_t* gpuDev = nullptr;
  doca_dev_t* netDev = nullptr;
  char pciBusId[MAX_PCI_ADDRESS_LEN];
  int cudaDev;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUDACHECK(cudaDeviceGetPCIBusId(pciBusId, MAX_PCI_ADDRESS_LEN, cudaDev));
  DOCACHECKGOTO(doca_gpu_create(pciBusId, &gpuDev), status, out);
  DOCACHECKGOTO(doca_verbs_dev_open(cComm->ib.pd, &netDev), status, out);
  NCCLCHECKGOTO(gdakiCheckDeviceSupport(
                  gpuDev, netDev, (enum doca_gpu_dev_verbs_nic_handler)ncclParamGinGdakiNicHandler(), cComm->rank),
                status, out);

out:
  if (netDev) {
    doca_error_t docaStatus = doca_verbs_dev_close(netDev);
    if (docaStatus != DOCA_SUCCESS) {
      WARN("[%d] GIN/GDAKI: failed to close the temporary NIC device after capability validation: DOCA error %d",
           cComm->rank, docaStatus);
      if (status == ncclSuccess) status = ncclSystemError;
    }
  }
  if (gpuDev) {
    doca_error_t docaStatus = doca_gpu_destroy(gpuDev);
    if (docaStatus != DOCA_SUCCESS) {
      WARN("[%d] GIN/GDAKI: failed to destroy the temporary GPU device after capability validation: DOCA error %d",
           cComm->rank, docaStatus);
      if (status == ncclSuccess) status = ncclSystemError;
    }
  }
  return status;
}

ncclResult_t ncclGinGdakiCreateContext(void* collComm, ncclGinConfig_t* config, void** outGinCtx,
                                       ncclNetDeviceHandle_t** outDevHandle) {
  ncclResult_t status = ncclSuccess;

  const int nSignals = config->nSignals;
  const int nCounters = config->nCounters;
  const int nContexts = config->nContexts;
  const int queueDepth = config->queueDepth;
  const int trafficClass = (config->trafficClass < 0 || config->trafficClass > UINT8_MAX) ?
                             NCCL_NET_TRAFFIC_CLASS_UNDEF :
                             config->trafficClass;
  const int backendVersion = config->backendVersion;

  if (backendVersion < 0 || backendVersion > NCCL_GIN_GDAKI_GPU_CONTEXT_VERSION) {
    WARN("Invalid GIN GDAKI backend version %d", backendVersion);
    return ncclInternalError;
  }

  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  char pciBusId[MAX_PCI_ADDRESS_LEN];

  const int rank = cComm->rank;
  const int nranks = cComm->nranks;

  int const* peerArray = config->peerArray;
  const int peerArrayCount = config->peerArrayCount;

  if (peerArrayCount < 0 || (peerArrayCount > 0 && peerArray == nullptr)) {
    WARN("GIN GDAKI create context: invalid peerArrayCount %d and peerArray %p", peerArrayCount, peerArray);
    return ncclInternalError;
  }
  const int connectedNRanks = peerArrayCount;

  const int ncontexts = nContexts;
  const int nqps_per_rank = ncontexts;
  const int nqps_for_comm = nqps_per_rank * nranks;  // Number of QPs for communication
  const int nqps_for_comm_this_rank = nqps_per_rank * connectedNRanks;
  const bool needCompanion = (nCounters > 0);
  const int ncompanion_qps = needCompanion ? nqps_for_comm * 2 : 0;  // Number of companion QPs for communication
                                                                      // Double because we connect to self.
  const int nqps = nqps_per_rank * (nranks + 1);  // +1 for the local rank.
                                   // The last group is the responder of the local rank.

  // TODO: Take these config parameters from the environment variables or users.
  const int num_counters = nCounters;
  const int num_signals = nSignals;
  ncclNetProperties_t props;
  ncclNetDeviceHandle_t* devHandle = nullptr;
  struct gdaki_context* gdaki_ctx = nullptr;
  struct gdaki_exch_info* local_exch_info = nullptr;
  struct gdaki_exch_info* remote_exch_info = nullptr;

  struct doca_gpu_verbs_qp_init_attr_hl qp_init_attr;

  uint64_t* sink_buffer = nullptr;
  struct ibv_mr* sink_buffer_mr = nullptr;
  CUmemGenericAllocationHandle sink_buffer_mhandle;

  bool need_cpu_proxy = false;
  int rdmaWritesOrder = 0;
  bool preHopper = true;
  bool dataDirectNic = false;
  char dataDirectPath[PATH_MAX];

  struct doca_gpu_verbs_qp** gverbs_qps = nullptr;
  struct doca_gpu_verbs_qp** contiguous_gverbs_qps = nullptr;

  struct doca_verbs_device_attr* devAttr = nullptr;

  uint8_t isLagTxPortAffinitySupported = 0;
  uint8_t numLagPorts = 0;

  GdakiHostGPUMemHandle<char>* gin_gdaki_gpu_ctx_hd_mhandle = new GdakiHostGPUMemHandle<char>();
  GdakiGlobalGPUBufferTable<uint64_t>* counters_table = new GdakiGlobalGPUBufferTable<uint64_t>();
  GdakiGlobalGPUBufferTable<uint64_t>* signals_table = new GdakiGlobalGPUBufferTable<uint64_t>();

  const int ib_sl = (ncclParamIbSl() != NCCL_PARAM_VAL_AUTO)       ? ncclParamIbSl() :
                    (trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? trafficClass :
                                                                     NCCL_IB_SL_DEFAULT;
  int ib_tc = (trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? trafficClass : NCCL_IB_TC_DEFAULT;
  uint8_t globalTc;
  int ib_gid_index = 0;
  uint32_t qpn, qpn_companion;

  NCCLCHECK(cComm->getProperties(cComm->dev, &props));

  NCCLCHECKGOTO(gin_gdaki_gpu_ctx_hd_mhandle->allocate(ncontexts * NCCL_GIN_GDAKI_GPU_CONTEXT_MAX_SIZE), status, out);
  NCCLCHECKGOTO(counters_table->allocate(num_counters * ncontexts, nranks), status, out);
  NCCLCHECKGOTO(signals_table->allocate(num_signals * ncontexts, nranks), status, out);

  gdaki_ctx = (struct gdaki_context*)calloc(1, sizeof(*gdaki_ctx));
  EQCHECKGOTO(gdaki_ctx, nullptr, status, out);
  gdaki_ctx->needCompanion = needCompanion;
  gdaki_ctx->rank = rank;

  gdaki_ctx->gin_gdaki_gpu_ctx_host_staging =
    (struct ncclGinGdakiGPUContext*)calloc(ncontexts, sizeof(struct ncclGinGdakiGPUContext));
  EQCHECKGOTO(gdaki_ctx->gin_gdaki_gpu_ctx_host_staging, nullptr, status, out);

  devHandle = (ncclNetDeviceHandle_t*)calloc(1, sizeof(*devHandle));
  EQCHECKGOTO(devHandle, nullptr, status, out);

  if (needCompanion) {
    gdaki_ctx->gqp_groups = (struct doca_gpu_verbs_qp_group_hl**)calloc(nqps_for_comm, sizeof(*gdaki_ctx->gqp_groups));
    EQCHECKGOTO(gdaki_ctx->gqp_groups, nullptr, status, out);
  }

  // Main QP
  gdaki_ctx->gqps = (struct doca_gpu_verbs_qp_hl**)calloc(nqps, sizeof(*gdaki_ctx->gqps));
  EQCHECKGOTO(gdaki_ctx->gqps, nullptr, status, out);

  // Companion QP
  if (needCompanion) {
    gdaki_ctx->companion_gqps =
      (struct doca_gpu_verbs_qp_hl**)calloc(ncompanion_qps, sizeof(*gdaki_ctx->companion_gqps));
    EQCHECKGOTO(gdaki_ctx->companion_gqps, nullptr, status, out);
  }

  local_exch_info = (struct gdaki_exch_info*)calloc(nranks, sizeof(*local_exch_info));
  EQCHECKGOTO(local_exch_info, nullptr, status, out);

  remote_exch_info = (struct gdaki_exch_info*)calloc(ncontexts * nranks, sizeof(*remote_exch_info));
  EQCHECKGOTO(remote_exch_info, nullptr, status, out);

  gethostname(gdaki_ctx->hostname, HOST_NAME_MAX);
  snprintf(gdaki_ctx->ib_dev_name, IBV_SYSFS_NAME_MAX, "%s", props.name);

  CUDACHECK(cudaGetDevice(&gdaki_ctx->cuda_id));
  CUDACHECK(cudaDeviceGetPCIBusId(pciBusId, MAX_PCI_ADDRESS_LEN, gdaki_ctx->cuda_id));

  CUDACHECK(cudaDeviceGetAttribute(&rdmaWritesOrder, cudaDevAttrGPUDirectRDMAWritesOrdering, gdaki_ctx->cuda_id));
  preHopper = (rdmaWritesOrder < CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
  if (ncclParamIbDataDirect() > 0) {
    ncclResult_t result =
      wrap_mlx5dv_get_data_direct_sysfs_path(cComm->ib.context, dataDirectPath, sizeof(dataDirectPath));
    if (result == ncclSuccess) {
      dataDirectNic = true;
      INFO(NCCL_NET, "GIN/GDAKI: Data Direct DMA Interface is detected for device %s (%s)", gdaki_ctx->ib_dev_name,
           dataDirectPath);
    } else if (result == ncclInvalidArgument) {
      TRACE(NCCL_NET, "GIN/GDAKI: Device %s does not support Data Direct DMA.", gdaki_ctx->ib_dev_name);
    } else {
      // Query unvailable for older driver versions
      INFO(NCCL_NET, "GIN/GDAKI: mlx5dv_get_data_direct_sysfs_path unavailable for device %s, assuming no Data Direct",
           gdaki_ctx->ib_dev_name);
    }
  }
  INFO(NCCL_NET | NCCL_INIT, "GIN/GDAKI: device %s preHopper=%d dataDirectNic=%d mcst=%d", gdaki_ctx->ib_dev_name,
       (int)preHopper, (int)dataDirectNic, (int)(preHopper || dataDirectNic));

  DOCACHECKGOTO(doca_gpu_create(pciBusId, &gdaki_ctx->gdev), status, out);

  NCCLCHECKGOTO(wrap_ibv_query_device(cComm->ib.context, &gdaki_ctx->ib_dev_attr), status, out);

  // Exchange counters and signals with peers
  NCCLCHECKGOTO(counters_table->register_mr(cComm->ib.pd, true), status, out);
  NCCLCHECKGOTO(signals_table->register_mr(cComm->ib.pd, true), status, out);

  NCCLCHECKGOTO(counters_table->exchange_info(cComm), status, out);
  NCCLCHECKGOTO(signals_table->exchange_info(cComm), status, out);

  gdaki_ctx->port_num = 1; // assume 1 for mlx5 devices
  NCCLCHECKGOTO(wrap_ibv_query_port(cComm->ib.context, gdaki_ctx->port_num, &gdaki_ctx->port_attr), status, out);

  // Get the GID index
  NCCLCHECKGOTO(cComm->getGidIndex(cComm->ib.context, gdaki_ctx->port_num, &gdaki_ctx->port_attr, &ib_gid_index),
                status, out);
  gdaki_ctx->gid_index = ib_gid_index;

  NCCLCHECKGOTO(ncclIbGetPkeyIndex(cComm->ib.context, gdaki_ctx->port_num, &gdaki_ctx->port_attr,
                                   &gdaki_ctx->pkey_index),
                status, out);

  NCCLCHECKGOTO(wrap_ibv_query_gid(cComm->ib.context, 1, ib_gid_index, &gdaki_ctx->rgid), status, out);

  DOCACHECKGOTO(doca_verbs_dev_open(cComm->ib.pd, &gdaki_ctx->ndev), status, out);

  if ((ncclParamGinGdakiLAGAwareDisable() == 0) && (gdaki_ctx->gdev->type == DOCA_GPU_LIB_TYPE_OPEN)) {
    DOCACHECKGOTO(doca_verbs_query_device(cComm->ib.context, &devAttr), status, out);
    isLagTxPortAffinitySupported = doca_verbs_device_attr_get_is_lag_tx_port_affinity_supported(devAttr);

    if (isLagTxPortAffinitySupported) {
      numLagPorts = doca_verbs_device_attr_get_num_lag_ports(devAttr);
    }
  }

  {
    doca_error_t docaStatus = doca_verbs_comp_channel_create(gdaki_ctx->ndev, &gdaki_ctx->docaEvent);
    if (docaStatus == DOCA_ERROR_NOT_SUPPORTED) {
      INFO(NCCL_NET, "doca_verbs_comp_channel_create not supported, falling back to polling-based errors");
      gdaki_ctx->docaEvent = nullptr;
      docaStatus = DOCA_SUCCESS;
    }
    DOCACHECKGOTO(docaStatus, status, out);
  }

  // Match the behavior of libibverbs: global traffic class overrides the user's config and environment variable.
  if (doca_verbs_query_global_traffic_class(gdaki_ctx->ndev, gdaki_ctx->port_num, &globalTc) == DOCA_SUCCESS) {
    ib_tc = globalTc;
  }
  NCCLCHECKGOTO(gdakiCreateVerbsAh(gdaki_ctx, ib_sl, ib_tc, ib_gid_index), status, out);

  if (ncclParamGinGdakiQCounter() && gdaki_ctx->gdev->type == DOCA_GPU_LIB_TYPE_OPEN) {
    gdakiQCounterCreate(gdaki_ctx, cComm->ib.context);
  }

  gdaki_ctx->qp_rq_size = 0;
  gdaki_ctx->qp_sq_size = queueDepth > 0 ? queueDepth : ncclParamGinGdakiQpDepth();

  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.gpu_dev = gdaki_ctx->gdev;
  qp_init_attr.net_dev = gdaki_ctx->ndev;
  qp_init_attr.ibpd = cComm->ib.pd;
  qp_init_attr.sq_nwqe = gdaki_ctx->qp_sq_size;
  qp_init_attr.nic_handler = (enum doca_gpu_dev_verbs_nic_handler)ncclParamGinGdakiNicHandler();
  qp_init_attr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
  if (ncclParamGinGdakiUseReliableDB())
    qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW;
  else qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
  qp_init_attr.comp_channel = gdaki_ctx->docaEvent;
  switch (ncclParamGinGdakiCqType()) {
  case 0:
    qp_init_attr.cq_collapsed = false;
    qp_init_attr.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B;
    INFO(NCCL_NET, "GDAKI CQ type configured as device (0)");
    break;
  case 1:
    qp_init_attr.cq_collapsed = true;
    qp_init_attr.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED;
    INFO(NCCL_NET, "GDAKI CQ type configured as device-collapsed (1)");
    break;
  case 2:
    qp_init_attr.cq_collapsed = true;
    qp_init_attr.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST;
    if ((ncclParamIbDataDirect() > 0) && true == dataDirectNic) {
      qp_init_attr.flags |= DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_SUPPORT_DATA_DIRECT;
    }
    INFO(NCCL_NET, "GDAKI CQ type configured as host-collapsed (2)");
    break;
  default:
    WARN("Invalid NCCL_GIN_GDAKI_CQ_TYPE=%ld; expected 0 (device), 1 (device-collapsed), or 2 (host-collapsed)",
         (long)ncclParamGinGdakiCqType());
    status = ncclInvalidArgument;
    goto out;
  }

  if (nqps_for_comm_this_rank > 0) {
  // SPC-X Ordering Semantic. 0 by default.
    qp_init_attr.ordering_semantic = gdakiOrderingSematic();
    if (needCompanion) {
    retry_create_qp_group_list_hl:
      doca_error_t docaStatus =
        doca_gpu_verbs_create_qp_group_list_hl(&qp_init_attr, nqps_for_comm_this_rank, &gdaki_ctx->gqp_group_list);
      if (docaStatus != DOCA_SUCCESS) {
        if (qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW) {
          qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED;
          goto retry_create_qp_group_list_hl;
        }

        if ((qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) &&
            ncclParamGinGdakiUseReliableDB() == 2) {
          qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
          goto retry_create_qp_group_list_hl;
        }

        WARN("DOCA Error %d", docaStatus);
        status = ncclSystemError;
        goto out;
      }

      const char* dbr_opt_str =
        qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW          ? "HW" :
        qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED ? "SW emulation" :
                                                                                                     "disabled";

      for (int list_idx = 0; list_idx < nqps_for_comm_this_rank; list_idx++) {
        const int qp_idx = peerArray[list_idx % connectedNRanks] + (list_idx / connectedNRanks) * nranks;
        gdaki_ctx->gqp_groups[qp_idx] = &gdaki_ctx->gqp_group_list->qpgs[list_idx];
        gdaki_ctx->gqps[qp_idx] = &gdaki_ctx->gqp_groups[qp_idx]->qp_main;
        gdaki_ctx->companion_gqps[qp_idx] = &gdaki_ctx->gqp_groups[qp_idx]->qp_companion;

        DOCACHECKGOTO(doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp, &qpn), status, out);
        DOCACHECKGOTO(doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp, &qpn_companion), status, out);
        INFO(NCCL_NET, "[%d] Created a QP group: qp_idx=%d, main_qpn=%#x, companion_qpn=%#x, reliable_db=%s", rank,
             qp_idx, qpn, qpn_companion, dbr_opt_str);
      }
    } else {
    retry_create_qp_list_hl:
      doca_error_t docaStatus =
        doca_gpu_verbs_create_qp_list_hl(&qp_init_attr, nqps_for_comm_this_rank, &gdaki_ctx->gqp_list);
      if (docaStatus != DOCA_SUCCESS) {
        if (qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW) {
          qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED;
          goto retry_create_qp_list_hl;
        }

        if ((qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED) &&
            ncclParamGinGdakiUseReliableDB() == 2) {
          qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
          goto retry_create_qp_list_hl;
        }

        WARN("DOCA Error %d", docaStatus);
        status = ncclSystemError;
        goto out;
      }

      const char* dbr_opt_str =
        qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW          ? "HW" :
        qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED ? "SW emulation" :
                                                                                                     "disabled";
      for (int list_idx = 0; list_idx < nqps_for_comm_this_rank; list_idx++) {
        const int qp_idx = peerArray[list_idx % connectedNRanks] + (list_idx / connectedNRanks) * nranks;
        gdaki_ctx->gqps[qp_idx] = &gdaki_ctx->gqp_list->qps[list_idx];

        DOCACHECKGOTO(doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp, &qpn), status, out);
        INFO(NCCL_NET, "[%d] Created a QP: qp_idx=%d, qpn=%#x, reliable_db=%s", rank, qp_idx, qpn, dbr_opt_str);
      }
    }
  }

  qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
  if (nqps > nqps_for_comm) {
    DOCACHECKGOTO(doca_gpu_verbs_create_qp_list_hl(&qp_init_attr, nqps - nqps_for_comm, &gdaki_ctx->self_gqp_list),
                  status, out);
    for (int qp_idx = nqps_for_comm; qp_idx < nqps; qp_idx++) {
      gdaki_ctx->gqps[qp_idx] = &gdaki_ctx->self_gqp_list->qps[qp_idx - nqps_for_comm];

      DOCACHECKGOTO(doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp, &qpn), status, out);
      INFO(NCCL_NET, "[%d] Created a self-loop peer QP: qp_idx=%d, qpn=%#x", rank, qp_idx, qpn);
    }
  }

  if (ncompanion_qps > nqps_for_comm && nqps_for_comm_this_rank > 0) {
    DOCACHECKGOTO(doca_gpu_verbs_create_qp_list_hl(&qp_init_attr, nqps_for_comm_this_rank,
                                                   &gdaki_ctx->self_companion_gqp_list),
                  status, out);
    for (int list_idx = 0; list_idx < nqps_for_comm_this_rank; list_idx++) {
      const int qp_idx = nqps_for_comm + peerArray[list_idx % connectedNRanks] + (list_idx / connectedNRanks) * nranks;
      gdaki_ctx->companion_gqps[qp_idx] = &gdaki_ctx->self_companion_gqp_list->qps[list_idx];

      DOCACHECKGOTO(doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp, &qpn_companion), status, out);
      INFO(NCCL_NET, "[%d] Created a self-loop peer companion QP: qp_idx=%d, qpn=%#x", rank, qp_idx, qpn_companion);
    }
  }

  for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
    // Prepare information for exchange with peers
    for (int i = 0; i < peerArrayCount; i++) {
      const int rank_idx = peerArray[i];
      int qp_idx = rank_idx + ctx_idx * nranks;
      gdakiFillExchInfo(&local_exch_info[rank_idx], gdaki_ctx, gdaki_ctx->gqps[qp_idx]);
    }

    // Exchange information with peers
    NCCLCHECKGOTO(cComm->allToAll(cComm, local_exch_info, &remote_exch_info[ctx_idx * nranks],
                                  sizeof(struct gdaki_exch_info)),
                  status, out);
  }

  for (int i = 0; i < peerArrayCount; i++) {
    const int rank_idx = peerArray[i];
    if (rank_idx == rank) continue;
    for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
      int qp_idx = rank_idx + ctx_idx * nranks;
      struct gdaki_exch_info* peer_info = &remote_exch_info[ctx_idx * nranks + rank_idx];

      uint8_t lagTxPortAffinity = (numLagPorts > 0) ? (ctx_idx % numLagPorts) + 1 : 0;

      NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->gqps[qp_idx], peer_info, lagTxPortAffinity), status, out);
      DOCACHECKGOTO(doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp, &qpn), status, out);
      INFO(NCCL_NET,
           "[%d] Connected main QP: qp_idx=%d, main_qpn=%#x, remote_rank=%d, remote_qpn=%#x, lagTxPortAffinity=%u%s, "
           "counterSetId=%u",
           rank, qp_idx, qpn, rank_idx, peer_info->qpn, lagTxPortAffinity, (numLagPorts > 0) ? " (LAG enabled)" : "",
           gdaki_ctx->ginCounterSetId);
    }
  }

  for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
    int qp_idx = rank + ctx_idx * nranks;
    if (gdaki_ctx->gqps[qp_idx] == nullptr) continue;
    struct gdaki_exch_info exch_info;
    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->gqps[nqps_for_comm + ctx_idx]);
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->gqps[qp_idx], &exch_info), status, out);
    DOCACHECKGOTO(doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp, &qpn), status, out);
    INFO(NCCL_NET, "[%d] Connected self-loop QP: qp_idx=%d, main_qpn=%#x, peer_qpn=%#x, counterSetId=%u", rank, qp_idx,
         qpn, exch_info.qpn, gdaki_ctx->ginCounterSetId);
  }

  for (int qp_idx = 0; qp_idx < nqps_per_rank; qp_idx++) {
    int peer_qp_idx = nqps_for_comm + qp_idx;
    int local_qp_idx = qp_idx * nranks + rank;
    if (gdaki_ctx->gqps[local_qp_idx] == nullptr) continue;
    struct gdaki_exch_info exch_info;
    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->gqps[local_qp_idx]);
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->gqps[peer_qp_idx], &exch_info), status, out);
    DOCACHECKGOTO(doca_verbs_qp_get_qpn(gdaki_ctx->gqps[peer_qp_idx]->qp, &qpn), status, out);
    INFO(NCCL_NET, "[%d] Connected self-loop peer QP: qp_idx=%d, qpn=%#x, main_qpn=%#x, counterSetId=%u", rank,
         peer_qp_idx, qpn, exch_info.qpn, gdaki_ctx->ginCounterSetId);
  }

  if (needCompanion) {
    for (int i = 0; i < nqps_for_comm_this_rank; i++) {
      const int qp_idx = peerArray[i % connectedNRanks] + (i / connectedNRanks) * nranks;
      int peer_qp_idx = nqps_for_comm + qp_idx;
      struct gdaki_exch_info exch_info;
      gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->companion_gqps[peer_qp_idx]);
      NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->companion_gqps[qp_idx], &exch_info), status, out);
      DOCACHECKGOTO(doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp, &qpn_companion), status, out);
      INFO(NCCL_NET,
           "[%d] Connected companion QP: qp_idx=%d, companion_qpn=%#x, peer_companion_qpn=%#x, counterSetId=%u", rank,
           qp_idx, qpn_companion, exch_info.qpn, gdaki_ctx->ginCounterSetId);

      gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->companion_gqps[qp_idx]);
      NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->companion_gqps[peer_qp_idx], &exch_info), status, out);
      DOCACHECKGOTO(doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp, &qpn_companion), status, out);
      INFO(NCCL_NET,
           "[%d] Connected self-loop peer companion QP: qp_idx=%d, peer_companion_qpn=%#x, "
           "companion_qpn=%#x, counterSetId=%u",
           rank, peer_qp_idx, qpn_companion, exch_info.qpn, gdaki_ctx->ginCounterSetId);
    }
  }

  NCCLCHECKGOTO(ncclCuMemAlloc((void**)&sink_buffer, &sink_buffer_mhandle, CU_MEM_HANDLE_TYPE_NONE, sizeof(uint64_t),
                               nullptr),
                status, out);

  NCCLCHECKGOTO(gdakiRegMr(&sink_buffer_mr, cComm->ib.pd, sink_buffer, sizeof(uint64_t),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                             IBV_ACCESS_REMOTE_ATOMIC),
                status, out);

  NCCLCHECKGOTO(ncclCudaCalloc(&gdaki_ctx->last_issued_get, ncontexts * nranks, NULL), status, out);
  NCCLCHECKGOTO(ncclCudaCalloc(&gdaki_ctx->last_visible_get, ncontexts * nranks, NULL), status, out);

  gverbs_qps = (struct doca_gpu_verbs_qp**)calloc(nranks, sizeof(struct doca_gpu_verbs_qp*));
  EQCHECKGOTO(gverbs_qps, nullptr, status, out);

  // calloc(0) may return NULL, which is not an error when no peers are connected.
  contiguous_gverbs_qps =
    (struct doca_gpu_verbs_qp**)calloc(connectedNRanks > 0 ? connectedNRanks : 1, sizeof(struct doca_gpu_verbs_qp*));
  EQCHECKGOTO(contiguous_gverbs_qps, nullptr, status, out);

  for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
    struct ncclGinGdakiGPUContext* gin_gdaki_gpu_ctx = &gdaki_ctx->gin_gdaki_gpu_ctx_host_staging[ctx_idx];

    unsigned int buffer_start;
    unsigned int contiguous_qp_idx = 0;
    for (int i = 0; i < peerArrayCount; i++) {
      const int qp_idx = peerArray[i];
      gverbs_qps[qp_idx] = gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
      contiguous_gverbs_qps[contiguous_qp_idx] = gverbs_qps[qp_idx];
      ++contiguous_qp_idx;
      need_cpu_proxy |= gdakiQpNeedsProxyProgress(gverbs_qps[qp_idx]);
    }
    DOCACHECKGOTO(doca_gpu_verbs_export_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks, &gin_gdaki_gpu_ctx->gdqp),
                  status, out);

    if (needCompanion) {
      contiguous_qp_idx = 0;
      for (int i = 0; i < peerArrayCount; i++) {
        const int qp_idx = peerArray[i];
        gverbs_qps[qp_idx] = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
        contiguous_gverbs_qps[contiguous_qp_idx] = gverbs_qps[qp_idx];
        ++contiguous_qp_idx;
        need_cpu_proxy |= gdakiQpNeedsProxyProgress(gverbs_qps[qp_idx]);
      }
      DOCACHECKGOTO(doca_gpu_verbs_export_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks,
                                                        &gin_gdaki_gpu_ctx->companion_gdqp),
                    status, out);
    } else {
      gin_gdaki_gpu_ctx->companion_gdqp = nullptr;
    }

    if (nCounters) {
      NCCLCHECKGOTO(counters_table->allocate_elements(num_counters, &buffer_start), status, out);
      gin_gdaki_gpu_ctx->counters_table.buffer = counters_table->gpu_ptr + buffer_start;
      gin_gdaki_gpu_ctx->counters_table.rkeys = counters_table->get_rkeys_d();
      gin_gdaki_gpu_ctx->counters_table.lkey = htobe32(counters_table->mr->lkey);
      gin_gdaki_gpu_ctx->counters_table.offset = buffer_start;
    }
    if (nSignals) {
      NCCLCHECKGOTO(signals_table->allocate_elements(num_signals, &buffer_start), status, out);
      gin_gdaki_gpu_ctx->signals_table.buffer = signals_table->gpu_ptr + buffer_start;
      gin_gdaki_gpu_ctx->signals_table.rkeys = signals_table->get_rkeys_d();
      gin_gdaki_gpu_ctx->signals_table.lkey = htobe32(signals_table->mr->lkey);
      gin_gdaki_gpu_ctx->signals_table.offset = buffer_start;
    }
    gin_gdaki_gpu_ctx->sink_buffer_lkey = htobe32(sink_buffer_mr->lkey);
    gin_gdaki_gpu_ctx->last_issued_get = gdaki_ctx->last_issued_get + ctx_idx * nranks;
    gin_gdaki_gpu_ctx->last_visible_get = gdaki_ctx->last_visible_get + ctx_idx * nranks;

    // MCST is needed on pre-Hopper or on post-Hopper with Data Direct
    bool use_mcst = ncclParamGinGdakiForceMcst() ? true : (preHopper || dataDirectNic);
    NCCLCHECKGOTO(ncclGinGdakiGPUContext_init(backendVersion, gin_gdaki_gpu_ctx_hd_mhandle->host_buf, ctx_idx,
                                              gin_gdaki_gpu_ctx->gdqp, gin_gdaki_gpu_ctx->companion_gdqp,
                                              gin_gdaki_gpu_ctx->counters_table, gin_gdaki_gpu_ctx->signals_table,
                                              gin_gdaki_gpu_ctx->sink_buffer_lkey, gin_gdaki_gpu_ctx->last_issued_get,
                                              gin_gdaki_gpu_ctx->last_visible_get, use_mcst),
                  status, out);
  }

  NCCLCHECKGOTO(gin_gdaki_gpu_ctx_hd_mhandle->copy_h_to_d(), status, out);

  devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  devHandle->netDeviceVersion = NCCL_GIN_GDAKI_VERSION;
  devHandle->handle = (void*)gin_gdaki_gpu_ctx_hd_mhandle->gpu_buf;
  devHandle->size = 0;
  devHandle->needsProxyProgress = need_cpu_proxy;

  gdaki_ctx->counters_table = counters_table;
  gdaki_ctx->signals_table = signals_table;
  gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle = gin_gdaki_gpu_ctx_hd_mhandle;
  gdaki_ctx->backendVersion = backendVersion;
  gdaki_ctx->sink_buffer.addr = sink_buffer;
  gdaki_ctx->sink_buffer.mr = sink_buffer_mr;
  gdaki_ctx->sink_buffer.mhandle = sink_buffer_mhandle;
  gdaki_ctx->collComm = cComm;
  gdaki_ctx->devHandle = devHandle;
  gdaki_ctx->nContexts = ncontexts;

  *outDevHandle = devHandle;
  *outGinCtx = gdaki_ctx;

out:
  if (status != ncclSuccess) {
    if (gdaki_ctx) {
      if (gdaki_ctx->docaEvent) {
        doca_verbs_comp_channel_destroy(gdaki_ctx->docaEvent);
        gdaki_ctx->docaEvent = nullptr;
      }
      // Clean up any allocated GPU memory
      if (gdaki_ctx->gin_gdaki_gpu_ctx_host_staging) {
        for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
          struct ncclGinGdakiGPUContext* gin_gdaki_gpu_ctx = &gdaki_ctx->gin_gdaki_gpu_ctx_host_staging[ctx_idx];
          if (gin_gdaki_gpu_ctx->gdqp) {
            for (int i = 0; i < peerArrayCount; i++) {
              const int qp_idx = peerArray[i];
              gverbs_qps[qp_idx] = gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
            }
            doca_gpu_verbs_unexport_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks, gin_gdaki_gpu_ctx->gdqp);
            gin_gdaki_gpu_ctx->gdqp = nullptr;
          }
          if (gin_gdaki_gpu_ctx->companion_gdqp) {
            for (int i = 0; i < peerArrayCount; i++) {
              const int qp_idx = peerArray[i];
              gverbs_qps[qp_idx] = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
            }
            doca_gpu_verbs_unexport_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks,
                                                  gin_gdaki_gpu_ctx->companion_gdqp);
            gin_gdaki_gpu_ctx->companion_gdqp = nullptr;
          }
        }
      }

      if (gdaki_ctx->gqp_group_list) {
        doca_gpu_verbs_destroy_qp_group_list_hl(gdaki_ctx->gqp_group_list);
        gdaki_ctx->gqp_group_list = nullptr;
      }
      if (gdaki_ctx->gqp_list) {
        doca_gpu_verbs_destroy_qp_list_hl(gdaki_ctx->gqp_list);
        gdaki_ctx->gqp_list = nullptr;
      }
      if (gdaki_ctx->self_gqp_list) {
        doca_gpu_verbs_destroy_qp_list_hl(gdaki_ctx->self_gqp_list);
        gdaki_ctx->self_gqp_list = nullptr;
      }
      if (gdaki_ctx->self_companion_gqp_list) {
        doca_gpu_verbs_destroy_qp_list_hl(gdaki_ctx->self_companion_gqp_list);
        gdaki_ctx->self_companion_gqp_list = nullptr;
      }

      if (gdaki_ctx->gqp_groups) free(gdaki_ctx->gqp_groups);
      if (gdaki_ctx->gqps) free(gdaki_ctx->gqps);
      if (gdaki_ctx->companion_gqps) free(gdaki_ctx->companion_gqps);

      if (gdaki_ctx->ginQCounter) wrap_mlx5dv_devx_obj_destroy(gdaki_ctx->ginQCounter);

      if (gdaki_ctx->ndev) {
        doca_verbs_dev_close(gdaki_ctx->ndev);
        gdaki_ctx->ndev = nullptr;
      }
      if (gdaki_ctx->gdev) doca_gpu_destroy(gdaki_ctx->gdev);
      free(gdaki_ctx->gin_gdaki_gpu_ctx_host_staging);
    }

    if (devHandle) free(devHandle);

    if (sink_buffer_mr) wrap_ibv_dereg_mr(sink_buffer_mr);
    if (sink_buffer) ncclCuMemFree(sink_buffer, nullptr);

    if (gin_gdaki_gpu_ctx_hd_mhandle) delete gin_gdaki_gpu_ctx_hd_mhandle;
    if (counters_table) delete counters_table;
    if (signals_table) delete signals_table;

    if (gdaki_ctx) {
      if (gdaki_ctx->last_issued_get) NCCLCHECK(ncclCudaFree(gdaki_ctx->last_issued_get, NULL));
      if (gdaki_ctx->last_visible_get) NCCLCHECK(ncclCudaFree(gdaki_ctx->last_visible_get, NULL));
      memset(gdaki_ctx, 0, sizeof(*gdaki_ctx));
      free(gdaki_ctx);
    }
  }

  if (devAttr) doca_verbs_device_attr_free(devAttr);

  if (local_exch_info) free(local_exch_info);

  if (remote_exch_info) free(remote_exch_info);

  if (gverbs_qps) free(gverbs_qps);
  if (contiguous_gverbs_qps) free(contiguous_gverbs_qps);

  return status;
}

ncclResult_t ncclGinGdakiDestroyContext(void* ginCtx) {
  if (!ginCtx) return ncclInvalidArgument;

  struct gdaki_context* gdaki_ctx = (struct gdaki_context*)ginCtx;
  struct ncclGinIbCollComm* cComm = gdaki_ctx->collComm;
  const int nranks = cComm->nranks;
  const int ncontexts = gdaki_ctx->nContexts;

  if (gdaki_ctx->docaEvent) {
    doca_verbs_comp_channel_destroy(gdaki_ctx->docaEvent);
    gdaki_ctx->docaEvent = nullptr;
  }

  if (gdaki_ctx->gin_gdaki_gpu_ctx_host_staging) {
    struct doca_gpu_verbs_qp** gverbs_qps =
      (struct doca_gpu_verbs_qp**)calloc(nranks, sizeof(struct doca_gpu_verbs_qp*));
    for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
      struct ncclGinGdakiGPUContext* gin_gdaki_gpu_ctx = &gdaki_ctx->gin_gdaki_gpu_ctx_host_staging[ctx_idx];
      if (gin_gdaki_gpu_ctx->gdqp) {
        for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
          if (gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx] == nullptr) continue;
          gverbs_qps[qp_idx] = gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
        }
        DOCACHECK(doca_gpu_verbs_unexport_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks, gin_gdaki_gpu_ctx->gdqp));
        gin_gdaki_gpu_ctx->gdqp = nullptr;
      }
      if (gin_gdaki_gpu_ctx->companion_gdqp) {
        for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
          if (gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx] == nullptr) continue;
          gverbs_qps[qp_idx] = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
        }
        DOCACHECK(doca_gpu_verbs_unexport_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks,
                                                        gin_gdaki_gpu_ctx->companion_gdqp));
        gin_gdaki_gpu_ctx->companion_gdqp = nullptr;
      }
    }
    free(gverbs_qps);
    free(gdaki_ctx->gin_gdaki_gpu_ctx_host_staging);
    gdaki_ctx->gin_gdaki_gpu_ctx_host_staging = nullptr;
  }
  if (gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle) {
    NCCLCHECK(gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle->deallocate());
    delete gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle;
    gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle = nullptr;
  }

  if (gdaki_ctx->gqp_group_list) {
    DOCACHECK(doca_gpu_verbs_destroy_qp_group_list_hl(gdaki_ctx->gqp_group_list));
    gdaki_ctx->gqp_group_list = nullptr;
  }
  if (gdaki_ctx->gqp_list) {
    DOCACHECK(doca_gpu_verbs_destroy_qp_list_hl(gdaki_ctx->gqp_list));
    gdaki_ctx->gqp_list = nullptr;
  }
  if (gdaki_ctx->self_gqp_list) {
    DOCACHECK(doca_gpu_verbs_destroy_qp_list_hl(gdaki_ctx->self_gqp_list));
    gdaki_ctx->self_gqp_list = nullptr;
  }
  if (gdaki_ctx->self_companion_gqp_list) {
    DOCACHECK(doca_gpu_verbs_destroy_qp_list_hl(gdaki_ctx->self_companion_gqp_list));
    gdaki_ctx->self_companion_gqp_list = nullptr;
  }

  if (gdaki_ctx->gqp_groups) free(gdaki_ctx->gqp_groups);
  if (gdaki_ctx->gqps) free(gdaki_ctx->gqps);
  if (gdaki_ctx->companion_gqps) free(gdaki_ctx->companion_gqps);

  if (gdaki_ctx->ginQCounter) wrap_mlx5dv_devx_obj_destroy(gdaki_ctx->ginQCounter);

  if (gdaki_ctx->counters_table) {
    NCCLCHECK(gdaki_ctx->counters_table->deregister_mr());
    NCCLCHECK(gdaki_ctx->counters_table->deallocate());
    delete gdaki_ctx->counters_table;
  }
  if (gdaki_ctx->signals_table) {
    NCCLCHECK(gdaki_ctx->signals_table->deregister_mr());
    NCCLCHECK(gdaki_ctx->signals_table->deallocate());
    delete gdaki_ctx->signals_table;
  }

  if (gdaki_ctx->last_issued_get) NCCLCHECK(ncclCudaFree(gdaki_ctx->last_issued_get, NULL));
  if (gdaki_ctx->last_visible_get) NCCLCHECK(ncclCudaFree(gdaki_ctx->last_visible_get, NULL));

  if (gdaki_ctx->sink_buffer.mr) NCCLCHECK(wrap_ibv_dereg_mr(gdaki_ctx->sink_buffer.mr));
  if (gdaki_ctx->sink_buffer.addr) NCCLCHECK(ncclCuMemFree(gdaki_ctx->sink_buffer.addr, nullptr));

  if (gdaki_ctx->ah) {
    DOCACHECK(doca_verbs_ah_attr_destroy(gdaki_ctx->ah));
  }

  if (gdaki_ctx->gdev) {
    DOCACHECK(doca_gpu_destroy(gdaki_ctx->gdev));
  }
  if (gdaki_ctx->devHandle) free(gdaki_ctx->devHandle);

  memset(gdaki_ctx, 0, sizeof(*gdaki_ctx));
  free(gdaki_ctx);

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle,
                                  void** ginHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  ncclResult_t status = ncclSuccess;

  struct ibv_mr* mr = nullptr;
  GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle>* gdaki_mhandle_hd_mhandle =
    new GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle>();
  GdakiHostGPUMemHandle<__be32>* rkeys_hd_mhandle = new GdakiHostGPUMemHandle<__be32>();
  __be32 rkey;

  struct gdaki_mem_handle* gdaki_mhandle = nullptr;
  bool force_strict_ordering = (mr_flags & NCCL_NET_MR_FLAG_FORCE_SO);

  gdaki_mhandle = (struct gdaki_mem_handle*)calloc(1, sizeof(*gdaki_mhandle));
  EQCHECKGOTO(gdaki_mhandle, nullptr, status, out);

  NCCLCHECKGOTO(gdakiRegMr(&mr, cComm->ib.pd, data, size,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                             IBV_ACCESS_REMOTE_ATOMIC,
                           force_strict_ordering),
                status, out);

  rkey = htobe32(mr->rkey);
  NCCLCHECKGOTO(rkeys_hd_mhandle->allocate(cComm->nranks), status, out);
  NCCLCHECKGOTO(cComm->allGather(cComm, &rkey, rkeys_hd_mhandle->host_buf, sizeof(__be32)), status, out);
  NCCLCHECKGOTO(rkeys_hd_mhandle->copy_h_to_d(), status, out);

  NCCLCHECKGOTO(gdaki_mhandle_hd_mhandle->allocate(1), status, out);
  gdaki_mhandle_hd_mhandle->host_buf->rkeys = rkeys_hd_mhandle->gpu_buf;
  gdaki_mhandle_hd_mhandle->host_buf->lkey = htobe32(mr->lkey);
  NCCLCHECKGOTO(gdaki_mhandle_hd_mhandle->copy_h_to_d(), status, out);

  gdaki_mhandle->type = type;
  gdaki_mhandle->mr = mr;
  gdaki_mhandle->gdaki_mhandle_hd_mhandle = gdaki_mhandle_hd_mhandle;
  gdaki_mhandle->rkeys_hd_mhandle = rkeys_hd_mhandle;

  INFO(NCCL_NET, "[%d] Registered MR: data=%p, size=%zu, lkey(be32)=%#x, rkey(be32)=%#x", cComm->rank, data, size,
       htobe32(mr->lkey), htobe32(mr->rkey));

  *mhandle = (void*)gdaki_mhandle;
  *ginHandle = (void*)gdaki_mhandle_hd_mhandle->gpu_buf;

out:
  if (status != ncclSuccess) {
    if (mr) wrap_ibv_dereg_mr(mr);
    free(gdaki_mhandle);
    delete gdaki_mhandle_hd_mhandle;
    delete rkeys_hd_mhandle;
  }
  return status;
}

ncclResult_t ncclGinGdakiDeregMrSym(void* collComm, void* mhandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  struct gdaki_mem_handle* gdaki_mhandle = (struct gdaki_mem_handle*)mhandle;
  struct ibv_mr* mr = gdaki_mhandle->mr;

  INFO(NCCL_NET, "[%d] Unregistering MR: lkey(be32)=%#x, rkey(be32)=%#x", cComm->rank, htobe32(mr->lkey),
       htobe32(mr->rkey));

  NCCLCHECK(wrap_ibv_dereg_mr(mr));

  NCCLCHECK(gdaki_mhandle->gdaki_mhandle_hd_mhandle->deallocate());
  delete gdaki_mhandle->gdaki_mhandle_hd_mhandle;
  NCCLCHECK(gdaki_mhandle->rkeys_hd_mhandle->deallocate());
  delete gdaki_mhandle->rkeys_hd_mhandle;

  memset(gdaki_mhandle, 0, sizeof(*gdaki_mhandle));

  free(gdaki_mhandle);

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiProgress(void* ctx) {
  struct gdaki_context* gdakiCtx = (struct gdaki_context*)ctx;
  const int ncontexts = gdakiCtx->nContexts;
  const int nranks = gdakiCtx->collComm->nranks;
  const int nqpsPerRank = ncontexts;
  const int nqpsForComm = nqpsPerRank * nranks;  // Number of QPs for communication
  bool has_progressed = true;
  bool progressed;

  while (has_progressed) {
    has_progressed = false;
    for (int qpIdx = 0; qpIdx < nqpsForComm; qpIdx++) {
      if (gdakiCtx->gqps[qpIdx] == nullptr) continue;
      struct doca_gpu_verbs_qp* qp = gdakiCtx->gqps[qpIdx]->qp_gverbs;
      if (gdakiQpNeedsProxyProgress(qp)) {
        DOCACHECK(doca_gpu_verbs_cpu_proxy_progress(qp, &progressed));
        has_progressed |= progressed;
      }

      if (gdakiCtx->companion_gqps) {
        qp = gdakiCtx->companion_gqps[qpIdx]->qp_gverbs;
        if (gdakiQpNeedsProxyProgress(qp)) {
          DOCACHECK(doca_gpu_verbs_cpu_proxy_progress(qp, &progressed));
          has_progressed |= progressed;
        }
      }
    }
  }

  return ncclSuccess;
}

// Logs a GIN QP error. When GIN QPs are attached to the GIN Q counter set, the same line carries their cumulative
// hardware counters, so triage can tell whether GIN traffic as a whole saw retries, drops, sequence errors, etc.
static void gdakiReportQpError(struct gdaki_context* ctx, struct doca_gpu_verbs_qp* qp,
                               const struct doca_gpu_verbs_qp_error_info* errorInfo) {
  const int nranks = ctx->collComm->nranks;
  const int nqpsForComm = ctx->nContexts * nranks;
  const char* type = "self-loop peer";
  int qpIdx = -1;
  for (int idx = 0; idx < nqpsForComm && qpIdx < 0; idx++) {
    if (ctx->gqps[idx] == nullptr) continue;
    if (ctx->gqps[idx]->qp_gverbs == qp) {
      type = "main";
    } else if (ctx->companion_gqps && ctx->companion_gqps[idx]->qp_gverbs == qp) {
      type = "companion";
    } else {
      continue;
    }
    qpIdx = idx;
  }

  uint32_t qpn = 0;
  doca_verbs_qp_get_qpn(qp->qp, &qpn);

  char counters[1024];
  // Communication QPs are indexed qpIdx = remoteRank + contextId * nranks.
  WARN("GDAKI QP error on qpIdx %d/%d (%s): rank=%d remoteRank=%d contextId=%d nranks=%d ncontexts=%d qpn=%#x "
       "counterSetId=%u syndrome=%#x vendor_err=%#x hw_err=%#x hw_type=%#x wqe_counter=%d%s",
       qpIdx, nqpsForComm, type, ctx->rank, qpIdx < 0 ? -1 : qpIdx % nranks, qpIdx < 0 ? -1 : qpIdx / nranks, nranks,
       ctx->nContexts, qpn, ctx->ginCounterSetId, (unsigned)errorInfo->syndrome, (unsigned)errorInfo->vendor_err_synd,
       (unsigned)errorInfo->hw_err_synd, (unsigned)errorInfo->hw_synd_type, errorInfo->wqe_counter,
       gdakiQCounterString(ctx, counters, sizeof(counters)));
}

static ncclResult_t ncclGinGdakiQueryLastErrorPolling(struct gdaki_context* gdakiCtx, bool* hasError) {
  bool hasError_ = false;
  const int ncontexts = gdakiCtx->nContexts;
  const int nranks = gdakiCtx->collComm->nranks;
  const int nqpsPerRank = ncontexts;
  const int nqpsForComm = nqpsPerRank * nranks;  // Number of QPs for communication

  for (int qpIdx = 0; qpIdx < nqpsForComm; qpIdx++) {
    if (gdakiCtx->gqps[qpIdx] == nullptr) continue;
    struct doca_gpu_verbs_qp* qp = gdakiCtx->gqps[qpIdx]->qp_gverbs;
    struct doca_gpu_verbs_qp_error_info errorInfo;

    DOCACHECK(doca_gpu_verbs_query_last_error(qp, &errorInfo));
    if (errorInfo.has_error) {
      gdakiReportQpError(gdakiCtx, qp, &errorInfo);
      hasError_ = true;
      break;
    }

    if (gdakiCtx->companion_gqps) {
      qp = gdakiCtx->companion_gqps[qpIdx]->qp_gverbs;
      DOCACHECK(doca_gpu_verbs_query_last_error(qp, &errorInfo));
      if (errorInfo.has_error) {
        gdakiReportQpError(gdakiCtx, qp, &errorInfo);
        hasError_ = true;
        break;
      }
    }
  }

  *hasError = hasError_;
  return ncclSuccess;
}

static ncclResult_t ncclGinGdakiQueryLastErrorEvent(struct gdaki_context* gdakiCtx, bool* hasError) {
  bool hasError_ = false;
  struct doca_gpu_verbs_qp* eventQp = nullptr;
  void* cq_context;
  doca_error_t status;

  if (gdakiCtx->docaEvent) {
    status = doca_verbs_get_cq_comp_channel_event(gdakiCtx->docaEvent, &cq_context);
    if (status == DOCA_SUCCESS && cq_context != nullptr) {
      eventQp = (struct doca_gpu_verbs_qp*)cq_context;

      struct doca_gpu_verbs_qp_error_info errorInfo;
      DOCACHECK(doca_gpu_verbs_query_last_error(eventQp, &errorInfo));
      hasError_ = errorInfo.has_error;
      if (hasError_) gdakiReportQpError(gdakiCtx, eventQp, &errorInfo);

      status = doca_verbs_ack_cq_events(eventQp->cq_sq, 1);
      if (status != DOCA_SUCCESS) return ncclInternalError;
    } else if (status == DOCA_SUCCESS && cq_context == nullptr) {
      WARN("doca_verbs_get_cq_comp_channel_event failure: cq_context not set on success.");
    } else if (status != DOCA_SUCCESS && status != DOCA_ERROR_AGAIN) return ncclInternalError;
  }

  *hasError = hasError_;
  return ncclSuccess;
}

ncclResult_t ncclGinGdakiQueryLastError(void* ginCtx, bool* hasError) {
  ncclResult_t status = ncclSuccess;
  struct gdaki_context* gdakiCtx = (struct gdaki_context*)ginCtx;
  bool hasError_ = false;

  // We throttle the frequency of these queries since they can easily take 250us.
  uint64_t now = clockNano();
  if ((now - gdakiCtx->last_error_query_time) / 1e9 < ncclParamGinErrorQuerySec()) {
    goto exit;
  }
  gdakiCtx->last_error_query_time = now;

  if (gdakiCtx->docaEvent) {
    NCCLCHECKGOTO(ncclGinGdakiQueryLastErrorEvent(gdakiCtx, &hasError_), status, exit);
  } else {
    NCCLCHECKGOTO(ncclGinGdakiQueryLastErrorPolling(gdakiCtx, &hasError_), status, exit);
  }

exit:
  *hasError = hasError_;
  return status;
}
