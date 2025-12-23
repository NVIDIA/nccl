/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <alloc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <mutex>

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"
#include "gin/gin_host.h"
#include "gin_host_gdaki.h"
#include "plugin/nccl_net.h"
#include "param.h"

#include "doca_gpunetio_host.h"
#include "nccl_device/gin/gdaki/gin_gdaki_device_host_common.h"
#include "../gin.h"

#define DOCACHECK(call)                                       \
  do {                                                        \
    doca_error_t RES = call;                                  \
    if (RES != DOCA_SUCCESS) {                           \
      /* Print the back trace*/                               \
      INFO(NCCL_NET, "%s:%d -> %d", __FILE__, __LINE__, RES); \
      return ncclSystemError;                                 \
    }                                                         \
  } while (0)

#define DOCACHECKGOTO(call, DOCA_RES, NCCL_RES, label)             \
  do {                                                             \
    DOCA_RES = call;                                               \
    if (DOCA_RES != DOCA_SUCCESS) {                           \
      /* Print the back trace*/                                    \
      INFO(NCCL_NET, "%s:%d -> %d", __FILE__, __LINE__, DOCA_RES); \
      NCCL_RES = ncclSystemError;                                  \
      goto label;                                                  \
    }                                                              \
  } while (0)

#define VERBS_TEST_DBR_SIZE (8)
#define MAX_PCI_ADDRESS_LEN 32U

NCCL_PARAM(GinGdakiNicHandler, "GIN_GDAKI_NIC_HANDLER", 0);
NCCL_PARAM(GinGdakiQpDepth, "GIN_GDAKI_QP_DEPTH", 128);
NCCL_PARAM(GinErrorQuerySec, "GIN_ERROR_QUERY_SEC", 10);
extern int64_t ncclParamIbTimeout();
extern int64_t ncclParamIbRetryCnt();
extern int64_t ncclParamIbPkey();
extern int64_t ncclParamIbSl();
extern int64_t ncclParamIbTc();
extern int64_t ncclParamIbPciRelaxedOrdering();
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamDmaBufEnable();

static const int NCCL_IB_SL_DEFAULT = 0;
static const int NCCL_IB_TC_DEFAULT = 0;

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

static ncclResult_t gdakiRegMrDmaBuf(struct ibv_mr **mr, struct ibv_pd *pd, void *addr,
                                     size_t length, int access) {
  int status = 0;
  int dmabuf_fd = -1;

  if (ncclParamDmaBufEnable() == 0) return ncclInvalidUsage;

#if CUDA_VERSION >= 11070
  static size_t host_page_size = sysconf(_SC_PAGESIZE);
  size_t aligned_size = length;
  ALIGN_SIZE(aligned_size, host_page_size);

#if CUDA_VERSION >= 12080
  if (ncclParamIbDataDirect()) {
    status = pfn_cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)addr, aligned_size,
                                               CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                               CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);
    if (status) {
      INFO(NCCL_NET,
           "Failed to get DMA-BUF handle for address range with type PCIE, error=%d. Trying a "
           "different method.",
           status);
      goto try_legacy;
    }
    status = wrap_mlx5dv_reg_dmabuf_mr(mr, pd, 0, aligned_size, 0, dmabuf_fd, access,
                                       MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
    if (status) {
      INFO(NCCL_NET,
           "Failed to register memory with DMA-BUF and data direct, error=%d. Trying a different "
           "method.",
           status);
      close(dmabuf_fd);
      dmabuf_fd = -1;
    } else
      goto out;
  }
try_legacy:

#endif

  CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)addr, aligned_size,
                                        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
  status = wrap_ibv_reg_dmabuf_mr(mr, pd, 0, aligned_size, 0, dmabuf_fd, access);
  if (status)
    INFO(NCCL_NET, "Failed to register memory with DMA-BUF, error=%d. Trying a different method.",
         status);
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

static ncclResult_t gdakiRegMr(struct ibv_mr **mr, struct ibv_pd *pd, void *addr, size_t length,
                               int access, bool force_strict_ordering = false) {
  int status = 0;

  if (!force_strict_ordering && gdakiRelaxedOrderingEnabled())
    access |= IBV_ACCESS_RELAXED_ORDERING;

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
  T *host_buf;
  T *gpu_buf;

  ncclResult_t allocate(unsigned int num_elements) {
    this->host_buf = (T *)calloc(num_elements, sizeof(T));
    EQCHECK(this->host_buf, nullptr);

    NCCLCHECK(ncclCuMemAlloc((void **)&this->gpu_buf, &this->cumemhandle, CU_MEM_HANDLE_TYPE_NONE,
                             num_elements * sizeof(T)));

    this->num_elements = num_elements;

    return ncclSuccess;
  }

  void deallocate() {
    if (this->host_buf != nullptr) {
      free(this->host_buf);
    }
    if (this->gpu_buf != nullptr) {
      ncclCuMemFree(this->gpu_buf);
    }
  }

  ncclResult_t copy_h_to_d() {
    NCCLCHECK(ncclCudaMemcpy<T>(this->gpu_buf, this->host_buf, this->num_elements));
    return ncclSuccess;
  }

  ncclResult_t copy_d_to_h() {
    NCCLCHECK(ncclCudaMemcpy<T>(this->host_buf, this->gpu_buf, this->num_elements));
    return ncclSuccess;
  }

  GdakiHostGPUMemHandle() : cumemhandle(0), num_elements(0), host_buf(nullptr), gpu_buf(nullptr){};
  GdakiHostGPUMemHandle(unsigned int num_elements) {
    ncclResult_t status = this->allocate(num_elements);
    if (status != ncclSuccess) {
      throw status;
    }
  };

  ~GdakiHostGPUMemHandle() { this->deallocate(); }
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
  T *gpu_ptr;
  struct ibv_mr *mr;

  ncclResult_t allocate(unsigned int num_elements, unsigned int num_ranks) {
    NCCLCHECK(ncclCuMemAlloc((void **)&this->gpu_ptr, &this->cumemhandle, CU_MEM_HANDLE_TYPE_NONE,
                             num_elements * sizeof(T)));
    CUDACHECK(cudaMemset(this->gpu_ptr, 0, num_elements * sizeof(T)));
    NCCLCHECK(this->rkeys_hd_mhandle.allocate(num_ranks));

    this->num_elements = num_elements;
    this->num_ranks = num_ranks;
    this->next_unused_idx = 0;

    return ncclSuccess;
  }

  void deallocate() {
    if (this->gpu_ptr != nullptr) {
      ncclCuMemFree(this->gpu_ptr);
    }
  }

  ncclResult_t register_mr(struct ibv_pd *ib_pd, bool force_strict_ordering = false) {
    NCCLCHECK(gdakiRegMr(&this->mr, ib_pd, this->gpu_ptr, this->num_elements * sizeof(T),
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_ATOMIC,
                         force_strict_ordering));
    return ncclSuccess;
  }

  void deregister_mr() {
    if (this->mr != nullptr) {
      wrap_ibv_dereg_mr(this->mr);
      this->mr = nullptr;
    }
  }

  ncclResult_t exchange_info(struct ncclGinIbCollComm *cComm) {
    __be32 rkey = htobe32(this->mr->rkey);
    NCCLCHECK(cComm->allGather(cComm, &rkey, this->rkeys_hd_mhandle.host_buf, sizeof(__be32)));
    NCCLCHECK(this->rkeys_hd_mhandle.copy_h_to_d());
    return ncclSuccess;
  }

  ncclResult_t allocate_elements(unsigned int num_elements, unsigned int *out_start_idx) {
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

  uint32_t *get_rkeys_d() { return this->rkeys_hd_mhandle.gpu_buf; }

  GdakiGlobalGPUBufferTable()
    : gpu_ptr(nullptr), mr(nullptr), cumemhandle(nullptr), num_elements(0), next_unused_idx(0){};
  GdakiGlobalGPUBufferTable(unsigned int num_elements, unsigned int num_ranks) {
    this->allocate(num_elements, num_ranks);
  };
  ~GdakiGlobalGPUBufferTable() { this->deallocate(); }
};

struct gdaki_mem_handle {
  int type;
  struct ibv_mr *mr;
  GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle> *gdaki_mhandle_hd_mhandle;
  GdakiHostGPUMemHandle<uint32_t> *rkeys_hd_mhandle;
};

struct gdaki_exch_info {
  int lid;
  int qpn;
  union ibv_gid gid;
  struct doca_verbs_gid vgid;
  int gid_index;
};

struct gdaki_context {
  int cuda_id;
  struct doca_gpu *gdev;
  struct ibv_device *ib_dev;
  struct ibv_context *ib_ctx;    /* DOCA Verbs Context */
  struct ibv_pd *ib_pd;          /* local protection domain */
  struct doca_verbs_ah_attr *ah; /* DOCA Verbs address handle */
  struct doca_verbs_gid gid;

  union ibv_gid rgid;
  struct ibv_port_attr port_attr;
  uint8_t port_num;
  int gid_index;

  uint32_t qp_rq_size;
  uint32_t qp_sq_size;
  struct doca_gpu_verbs_qp_group_hl **gqp_groups;
  struct doca_gpu_verbs_qp_hl **gqps;
  struct doca_gpu_verbs_qp_hl **companion_gqps;

  GdakiGlobalGPUBufferTable<uint64_t> *counters_table;
  GdakiGlobalGPUBufferTable<uint64_t> *signals_table;
  GdakiHostGPUMemHandle<struct ncclGinGdakiGPUContext> *gin_gdaki_gpu_ctx_hd_mhandle;
  struct {
    void *addr;
    struct ibv_mr *mr;
    CUmemGenericAllocationHandle mhandle;
  } sink_buffer;
  struct timespec last_error_query_time;

  struct ncclGinIbCollComm *collComm;
  ncclNetDeviceHandle_v11_t *devHandle;
};

template <typename T>
static inline T gdaki_round_up(T x, T y) {
  return ((x + y - 1) / y) * y;
}

static void gdakiFillExchInfo(struct gdaki_exch_info *exch_info, struct gdaki_context *gdaki_ctx,
                              struct doca_gpu_verbs_qp_hl *gqp) {
  exch_info->lid = gdaki_ctx->port_attr.lid;
  exch_info->qpn = doca_verbs_qp_get_qpn(gqp->qp);
  memcpy(exch_info->gid.raw, gdaki_ctx->rgid.raw, sizeof(union ibv_gid));
  memcpy(exch_info->vgid.raw, gdaki_ctx->rgid.raw, sizeof(union ibv_gid));
  exch_info->gid_index = gdaki_ctx->gid_index;
}

static ncclResult_t gdakiCreateVerbsAh(struct gdaki_context *ctx, int ib_sl, int ib_tc,
                                       int ib_gid_index) {
  ncclResult_t status = ncclSuccess;
  doca_error_t docaStatus = DOCA_SUCCESS;

  DOCACHECK(doca_verbs_ah_attr_create(ctx->ib_ctx, &ctx->ah));
  DOCACHECK(doca_verbs_ah_attr_set_sl(ctx->ah, ib_sl));
  DOCACHECK(doca_verbs_ah_attr_set_traffic_class(ctx->ah, ib_tc));

  if (ctx->port_attr.link_layer == 1) {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_addr_type(ctx->ah, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH),
                  docaStatus, status, destroy_verbs_ah);
  } else {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_addr_type(ctx->ah, DOCA_VERBS_ADDR_TYPE_IPv4),
                  docaStatus, status, destroy_verbs_ah);
  }

  // set_port_num?
  DOCACHECKGOTO(doca_verbs_ah_attr_set_sgid_index(ctx->ah, ib_gid_index), docaStatus, status,
                destroy_verbs_ah);
  DOCACHECKGOTO(doca_verbs_ah_attr_set_hop_limit(ctx->ah, 255), docaStatus, status, destroy_verbs_ah);

  return ncclSuccess;

destroy_verbs_ah:
  DOCACHECK(doca_verbs_ah_attr_destroy(ctx->ah));
  return status;
}

static ncclResult_t gdakiConnectQp(struct gdaki_context *ctx, struct doca_gpu_verbs_qp_hl *gqp,
                                   struct gdaki_exch_info *exch_info) {
  ncclResult_t status = ncclSuccess;
  doca_error_t docaStatus = DOCA_SUCCESS;
  struct doca_verbs_qp_attr *verbs_qp_attr = nullptr;

  DOCACHECK(doca_verbs_ah_attr_set_gid(ctx->ah, exch_info->vgid));
  DOCACHECK(doca_verbs_ah_attr_set_dlid(ctx->ah, exch_info->lid));
  DOCACHECK(doca_verbs_qp_attr_create(&verbs_qp_attr));
  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, DOCA_VERBS_MTU_SIZE_4K_BYTES),
    docaStatus, status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, 0), docaStatus, status,
                destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, 0), docaStatus, status,
                destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_port_num(verbs_qp_attr, ctx->port_num), docaStatus,
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, ncclParamIbTimeout()), docaStatus,
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, ncclParamIbRetryCnt()), docaStatus,
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, 7), docaStatus, status,
                destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, 12), docaStatus,
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_INIT),
    docaStatus, status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, 1), docaStatus,
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, 1), docaStatus,
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_atomic(
                  verbs_qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC),
                docaStatus, status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, ctx->ah), docaStatus,
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, exch_info->qpn),
                docaStatus, status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_pkey_index(verbs_qp_attr, ncclParamIbPkey()), docaStatus,
                status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(
                  gqp->qp, verbs_qp_attr,
                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                    DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PKEY_INDEX |
                    DOCA_VERBS_QP_ATTR_PORT_NUM),
                docaStatus, status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTR),
    docaStatus, status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(
                  gqp->qp, verbs_qp_attr,
                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                    DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                    DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER),
                docaStatus, status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTS),
    docaStatus, status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(
                  gqp->qp, verbs_qp_attr,
                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                    DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                    DOCA_VERBS_QP_ATTR_RNR_RETRY),
                docaStatus, status, destroy_verbs_qp_attr);

  DOCACHECK(doca_verbs_qp_attr_destroy(verbs_qp_attr));

  return ncclSuccess;

destroy_verbs_qp_attr:
  DOCACHECK(doca_verbs_qp_attr_destroy(verbs_qp_attr));
  return status;
}

ncclResult_t ncclGinGdakiCreateContext(void *collComm, int nSignals, int nCounters,
                                       void **outGinCtx, ncclNetDeviceHandle_v11_t **outDevHandle) {
  int status = ncclSuccess;
  doca_error_t docaStatus;

  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;

  char pciBusId[MAX_PCI_ADDRESS_LEN];

  const int rank = cComm->rank;
  const int nranks = cComm->nranks;
  const int ncontexts = 1;
  const int nqps_per_rank = ncontexts;
  const int nqps_for_comm = nqps_per_rank * nranks;  // Number of QPs for communication
  const int ncompanion_qps = nqps_for_comm * 2;      // Number of companion QPs for communication
                                                     // Double because we connect to self.
  const int nqps =
    nqps_per_rank * (nranks + 1);  // +1 for the local rank.
                                   // The last group is the responder of the local rank.

  // TODO: Take these config parameters from the environment variables or users.
  const int num_counters = nCounters;
  const int num_signals = nSignals;
  ncclNetProperties_t props;
  ncclNetDeviceHandle_v11_t *devHandle = nullptr;
  struct gdaki_context *gdaki_ctx = nullptr;
  struct gdaki_exch_info *local_exch_info = nullptr;
  struct gdaki_exch_info *remote_exch_info = nullptr;

  struct doca_gpu_verbs_qp_init_attr_hl qp_init_attr;

  uint64_t *sink_buffer = nullptr;
  struct ibv_mr *sink_buffer_mr = nullptr;
  CUmemGenericAllocationHandle sink_buffer_mhandle;

  bool need_cpu_proxy = false;

  GdakiHostGPUMemHandle<struct ncclGinGdakiGPUContext> *gin_gdaki_gpu_ctx_hd_mhandle =
    new GdakiHostGPUMemHandle<struct ncclGinGdakiGPUContext>(ncontexts);

  GdakiGlobalGPUBufferTable<uint64_t> *counters_table =
    new GdakiGlobalGPUBufferTable<uint64_t>(num_counters, nranks);
  GdakiGlobalGPUBufferTable<uint64_t> *signals_table =
    new GdakiGlobalGPUBufferTable<uint64_t>(num_signals, nranks);

  const int ib_sl = (ncclParamIbSl() != -1) ? ncclParamIbSl() : NCCL_IB_SL_DEFAULT;
  const int ib_tc = (ncclParamIbTc() != -1) ? ncclParamIbTc() : NCCL_IB_TC_DEFAULT;
  int ib_gid_index = 0;

  NCCLCHECK(cComm->getProperties(cComm->dev, &props));

  const size_t host_page_size = sysconf(_SC_PAGESIZE);
  gdaki_ctx = (struct gdaki_context *)calloc(1, sizeof(*gdaki_ctx));
  EQCHECKGOTO(gdaki_ctx, nullptr, status, out);

  devHandle = (ncclNetDeviceHandle_v11_t *)calloc(1, sizeof(*devHandle));
  EQCHECKGOTO(devHandle, nullptr, status, out);

  gdaki_ctx->gqp_groups = (struct doca_gpu_verbs_qp_group_hl **)calloc(
    nqps_for_comm, sizeof(*gdaki_ctx->gqp_groups));
  EQCHECKGOTO(gdaki_ctx->gqp_groups, nullptr, status, out);

  // Main QP
  gdaki_ctx->gqps = (struct doca_gpu_verbs_qp_hl **)calloc(nqps, sizeof(*gdaki_ctx->gqps));
  EQCHECKGOTO(gdaki_ctx->gqps, nullptr, status, out);

  // Companion QP
  gdaki_ctx->companion_gqps =
    (struct doca_gpu_verbs_qp_hl **)calloc(ncompanion_qps, sizeof(*gdaki_ctx->companion_gqps));
  EQCHECKGOTO(gdaki_ctx->companion_gqps, nullptr, status, out);

  local_exch_info = (struct gdaki_exch_info *)calloc(nranks, sizeof(*local_exch_info));
  EQCHECKGOTO(local_exch_info, nullptr, status, out);

  remote_exch_info = (struct gdaki_exch_info *)calloc(nranks, sizeof(*remote_exch_info));
  EQCHECKGOTO(remote_exch_info, nullptr, status, out);

  CUDACHECK(cudaGetDevice(&gdaki_ctx->cuda_id));
  CUDACHECK(cudaDeviceGetPCIBusId(pciBusId, MAX_PCI_ADDRESS_LEN, gdaki_ctx->cuda_id));

  DOCACHECKGOTO(doca_gpu_create(pciBusId, &gdaki_ctx->gdev), docaStatus, status, out);

  gdaki_ctx->ib_ctx = (struct ibv_context *)cComm->ibvCtx;

  // Allocate the protection domain
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&gdaki_ctx->ib_pd, gdaki_ctx->ib_ctx), status, out);

  // Exchange counters and signals with peers
  NCCLCHECKGOTO(counters_table->register_mr(gdaki_ctx->ib_pd, true), status, out);
  NCCLCHECKGOTO(signals_table->register_mr(gdaki_ctx->ib_pd, true), status, out);

  NCCLCHECKGOTO(counters_table->exchange_info(cComm), status, out);
  NCCLCHECKGOTO(signals_table->exchange_info(cComm), status, out);

  gdaki_ctx->port_num = 1; // assume 1 for mlx5 devices
  NCCLCHECKGOTO(wrap_ibv_query_port(gdaki_ctx->ib_ctx, gdaki_ctx->port_num, &gdaki_ctx->port_attr),
                status, out);

  // Get the GID index
  NCCLCHECKGOTO(cComm->getGidIndex(gdaki_ctx->ib_ctx, gdaki_ctx->port_num, &gdaki_ctx->port_attr, &ib_gid_index), status, out);
  gdaki_ctx->gid_index = ib_gid_index;

  NCCLCHECKGOTO(wrap_ibv_query_gid(gdaki_ctx->ib_ctx, 1, ib_gid_index, &gdaki_ctx->rgid), status,
                out);

  NCCLCHECKGOTO(gdakiCreateVerbsAh(gdaki_ctx, ib_sl, ib_tc, ib_gid_index), status, out);

  gdaki_ctx->qp_rq_size = 0;
  gdaki_ctx->qp_sq_size = ncclParamGinGdakiQpDepth();

  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.gpu_dev = gdaki_ctx->gdev;
  qp_init_attr.ibpd = gdaki_ctx->ib_pd;
  qp_init_attr.sq_nwqe = gdaki_ctx->qp_sq_size;
  qp_init_attr.nic_handler =
    (enum doca_gpu_dev_verbs_nic_handler)ncclParamGinGdakiNicHandler();
  qp_init_attr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

  for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
    DOCACHECKGOTO(
      doca_gpu_verbs_create_qp_group_hl(&qp_init_attr, &gdaki_ctx->gqp_groups[qp_idx]),
      docaStatus, status, out);

    gdaki_ctx->gqps[qp_idx] = &gdaki_ctx->gqp_groups[qp_idx]->qp_main;
    gdaki_ctx->companion_gqps[qp_idx] = &gdaki_ctx->gqp_groups[qp_idx]->qp_companion;

    INFO(NCCL_NET, "[%d] Created a QP group: qp_idx=%d, main_qpn=%#x, companion_qpn=%#x", rank,
         qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp),
         doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp));
  }

  for (int qp_idx = nqps_for_comm; qp_idx < nqps; qp_idx++) {
    DOCACHECKGOTO(doca_gpu_verbs_create_qp_hl(&qp_init_attr, &gdaki_ctx->gqps[qp_idx]),
                  docaStatus, status, out);
    INFO(NCCL_NET, "[%d] Created a self-loop peer QP: qp_idx=%d, qpn=%#x", rank, qp_idx,
         doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp));
  }

  for (int qp_idx = nqps_for_comm; qp_idx < ncompanion_qps; qp_idx++) {
    DOCACHECKGOTO(
      doca_gpu_verbs_create_qp_hl(&qp_init_attr, &gdaki_ctx->companion_gqps[qp_idx]),
      docaStatus, status, out);
    INFO(NCCL_NET, "[%d] Created a self-loop peer companion QP: qp_idx=%d, qpn=%#x", rank, qp_idx,
         doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp));
  }

  for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
    // Prepare information for exchange with peers
    for (int rank_idx = 0; rank_idx < nranks; rank_idx++) {
      int qp_idx = rank_idx + ctx_idx * nranks;
      gdakiFillExchInfo(&local_exch_info[rank_idx], gdaki_ctx, gdaki_ctx->gqps[qp_idx]);
    }

    // Exchange information with peers
    NCCLCHECKGOTO(
      cComm->allToAll(cComm, local_exch_info, remote_exch_info, sizeof(struct gdaki_exch_info)),
      status, out);

    for (int rank_idx = 0; rank_idx < nranks; rank_idx++) {
      int qp_idx = rank_idx + ctx_idx * nranks;
      if (rank_idx == rank)
        gdakiFillExchInfo(&remote_exch_info[rank_idx], gdaki_ctx,
                          gdaki_ctx->gqps[nqps_for_comm + ctx_idx]);

      NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->gqps[qp_idx], &remote_exch_info[rank_idx]),
                    status, out);

      INFO(NCCL_NET,
           "[%d] Connected main QP: qp_idx=%d, main_qpn=%#x, remote_rank=%d, remote_qpn=%#x", rank,
           qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp), rank_idx,
           remote_exch_info[rank_idx].qpn);
    }
  }

  for (int qp_idx = 0; qp_idx < nqps_per_rank; qp_idx++) {
    int peer_qp_idx = nqps_for_comm + qp_idx;
    struct gdaki_exch_info exch_info;
    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->gqps[qp_idx * nqps_per_rank + rank]);
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->gqps[peer_qp_idx], &exch_info), status, out);
    INFO(NCCL_NET, "[%d] Connected self-loop peer QP: qp_idx=%d, qpn=%#x, main_qpn=%#x", rank,
         peer_qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->gqps[peer_qp_idx]->qp), exch_info.qpn);
  }

  for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
    int peer_qp_idx = nqps_for_comm + qp_idx;
    struct gdaki_exch_info exch_info;
    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->companion_gqps[peer_qp_idx]);
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->companion_gqps[qp_idx], &exch_info), status,
                  out);
    INFO(NCCL_NET,
         "[%d] Connected companion QP: qp_idx=%d, companion_qpn=%#x, peer_companion_qpn=%#x", rank,
         qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp), exch_info.qpn);

    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->companion_gqps[qp_idx]);
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->companion_gqps[peer_qp_idx], &exch_info),
                  status, out);
    INFO(NCCL_NET,
         "[%d] Connected self-loop peer companion QP: qp_idx=%d, peer_companion_qpn=%#x, "
         "companion_qpn=%#x",
         rank, peer_qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[peer_qp_idx]->qp),
         exch_info.qpn);
  }

  NCCLCHECKGOTO(ncclCuMemAlloc((void **)&sink_buffer, &sink_buffer_mhandle, CU_MEM_HANDLE_TYPE_NONE,
                               sizeof(uint64_t)),
                status, out);

  NCCLCHECKGOTO(gdakiRegMr(&sink_buffer_mr, gdaki_ctx->ib_pd, sink_buffer, sizeof(uint64_t),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                             IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC),
                status, out);

  for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
    struct ncclGinGdakiGPUContext *gin_gdaki_gpu_ctx =
      &gin_gdaki_gpu_ctx_hd_mhandle->host_buf[ctx_idx];

    struct doca_gpu_dev_verbs_qp *tmp_qp;
    struct doca_gpu_dev_verbs_qp *tmp_qp_companion;

    tmp_qp = (struct doca_gpu_dev_verbs_qp *)calloc(nranks,
                                                         sizeof(struct doca_gpu_dev_verbs_qp));
    tmp_qp_companion = (struct doca_gpu_dev_verbs_qp *)calloc(
      nranks, sizeof(struct doca_gpu_dev_verbs_qp));
    for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
      struct doca_gpu_dev_verbs_qp *qp_cpu =
        gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs->qp_cpu;
      memcpy(&tmp_qp[qp_idx], qp_cpu, sizeof(struct doca_gpu_dev_verbs_qp));
      need_cpu_proxy |= (qp_cpu->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY);

      qp_cpu = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs->qp_cpu;
      memcpy(&tmp_qp_companion[qp_idx], qp_cpu, sizeof(struct doca_gpu_dev_verbs_qp));
      need_cpu_proxy |= (qp_cpu->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY);
    }

    DOCACHECKGOTO(
      doca_gpu_mem_alloc(gdaki_ctx->gdev, sizeof(struct doca_gpu_dev_verbs_qp) * nranks,
                              host_page_size, DOCA_GPU_MEM_TYPE_GPU,
                              (void **)&gin_gdaki_gpu_ctx->gdqp, nullptr);
      , docaStatus, status, out);

    NCCLCHECKGOTO(
      ncclCudaMemcpy<struct doca_gpu_dev_verbs_qp>(gin_gdaki_gpu_ctx->gdqp, tmp_qp, nranks),
      status, out);

    DOCACHECKGOTO(
      doca_gpu_mem_alloc(gdaki_ctx->gdev, sizeof(struct doca_gpu_dev_verbs_qp) * nranks,
                              host_page_size, DOCA_GPU_MEM_TYPE_GPU,
                              (void **)&gin_gdaki_gpu_ctx->companion_gdqp, nullptr);
      , docaStatus, status, out);

    NCCLCHECKGOTO(ncclCudaMemcpy<struct doca_gpu_dev_verbs_qp>(
                    gin_gdaki_gpu_ctx->companion_gdqp, tmp_qp_companion, nranks),
                  status, out);

    gin_gdaki_gpu_ctx->counters_table.buffer = counters_table->gpu_ptr;
    gin_gdaki_gpu_ctx->counters_table.rkeys = counters_table->get_rkeys_d();
    gin_gdaki_gpu_ctx->counters_table.lkey = htobe32(counters_table->mr->lkey);
    gin_gdaki_gpu_ctx->signals_table.buffer = signals_table->gpu_ptr;
    gin_gdaki_gpu_ctx->signals_table.rkeys = signals_table->get_rkeys_d();
    gin_gdaki_gpu_ctx->signals_table.lkey = htobe32(signals_table->mr->lkey);
    gin_gdaki_gpu_ctx->sink_buffer_lkey = htobe32(sink_buffer_mr->lkey);

    free(tmp_qp);
    free(tmp_qp_companion);
  }

  NCCLCHECKGOTO(gin_gdaki_gpu_ctx_hd_mhandle->copy_h_to_d(), status, out);

  devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  devHandle->netDeviceVersion = NCCL_GIN_GDAKI_VERSION;
  devHandle->handle = (void *)gin_gdaki_gpu_ctx_hd_mhandle->gpu_buf;
  devHandle->size = 0;
  devHandle->needsProxyProgress = need_cpu_proxy;

  gdaki_ctx->ib_pd = gdaki_ctx->ib_pd;
  gdaki_ctx->counters_table = counters_table;
  gdaki_ctx->signals_table = signals_table;
  gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle = gin_gdaki_gpu_ctx_hd_mhandle;
  gdaki_ctx->sink_buffer.addr = sink_buffer;
  gdaki_ctx->sink_buffer.mr = sink_buffer_mr;
  gdaki_ctx->sink_buffer.mhandle = sink_buffer_mhandle;
  gdaki_ctx->collComm = cComm;
  gdaki_ctx->devHandle = devHandle;

  cComm->ginCtx = gdaki_ctx;

  *outDevHandle = devHandle;
  *outGinCtx = gdaki_ctx;

out:
  if (status != ncclSuccess) {
    if (gdaki_ctx) {
      // Clean up any allocated GPU memory
      if (gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle) {
        for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
          struct ncclGinGdakiGPUContext *gin_gdaki_gpu_ctx =
            &gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle->host_buf[ctx_idx];
          if (gin_gdaki_gpu_ctx->gdqp) {
            doca_gpu_mem_free(gdaki_ctx->gdev, gin_gdaki_gpu_ctx->gdqp);
            gin_gdaki_gpu_ctx->gdqp = nullptr;
          }
          if (gin_gdaki_gpu_ctx->companion_gdqp) {
            doca_gpu_mem_free(gdaki_ctx->gdev, gin_gdaki_gpu_ctx->companion_gdqp);
            gin_gdaki_gpu_ctx->companion_gdqp = nullptr;
          }
        }
      }

      for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
        doca_gpu_verbs_destroy_qp_group_hl(gdaki_ctx->gqp_groups[qp_idx]);
        gdaki_ctx->gqp_groups[qp_idx] = nullptr;
      }
      for (int qp_idx = nqps_for_comm; qp_idx < nqps; qp_idx++) {
        doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->gqps[qp_idx]);
        gdaki_ctx->gqps[qp_idx] = nullptr;
      }
      for (int qp_idx = nqps_for_comm; qp_idx < ncompanion_qps; qp_idx++) {
        doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->companion_gqps[qp_idx]);
        gdaki_ctx->companion_gqps[qp_idx] = nullptr;
      }

      if (gdaki_ctx->gqp_groups) free(gdaki_ctx->gqp_groups);
      if (gdaki_ctx->gqps) free(gdaki_ctx->gqps);
      if (gdaki_ctx->companion_gqps) free(gdaki_ctx->companion_gqps);

      if (gdaki_ctx->gdev) doca_gpu_destroy(gdaki_ctx->gdev);
    }

    if (devHandle) free(devHandle);

    if (sink_buffer_mr) NCCLCHECK(wrap_ibv_dereg_mr(sink_buffer_mr));
    if (sink_buffer) ncclCuMemFree(sink_buffer);

    delete gin_gdaki_gpu_ctx_hd_mhandle;

    if (counters_table) {
      counters_table->deregister_mr();
      delete counters_table;
    }

    if (signals_table) {
      signals_table->deregister_mr();
      delete signals_table;
    }

    if (gdaki_ctx) {
      if (gdaki_ctx->ib_pd) NCCLCHECK(wrap_ibv_dealloc_pd(gdaki_ctx->ib_pd));

      memset(gdaki_ctx, 0, sizeof(*gdaki_ctx));
      free(gdaki_ctx);
    }
  }

  if (local_exch_info) free(local_exch_info);

  if (remote_exch_info) free(remote_exch_info);

  return (ncclResult_t)status;
}

ncclResult_t ncclGinGdakiDestroyContext(void *ginCtx) {
  if (!ginCtx) return ncclInvalidArgument;

  struct gdaki_context *gdaki_ctx = (struct gdaki_context *)ginCtx;
  struct ncclGinIbCollComm *cComm = gdaki_ctx->collComm;
  const int nranks = cComm->nranks;
  const int ncontexts = 1;
  const int nqps_per_rank = ncontexts;
  const int nqps_for_comm = nqps_per_rank * nranks;  // Number of QPs for communication
  const int ncompanion_qps = nqps_for_comm * 2;      // Number of companion QPs for communication
                                                     // Double because we connect to self.
  const int nqps =
    nqps_per_rank * (nranks + 1);  // +1 for the local rank.
                                   // The last group is the responder of the local rank.

  for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
    doca_gpu_verbs_destroy_qp_group_hl(gdaki_ctx->gqp_groups[qp_idx]);
    gdaki_ctx->gqp_groups[qp_idx] = nullptr;
  }
  for (int qp_idx = nqps_for_comm; qp_idx < nqps; qp_idx++) {
    doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->gqps[qp_idx]);
    gdaki_ctx->gqps[qp_idx] = nullptr;
  }
  for (int qp_idx = nqps_for_comm; qp_idx < ncompanion_qps; qp_idx++) {
    doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->companion_gqps[qp_idx]);
    gdaki_ctx->companion_gqps[qp_idx] = nullptr;
  }

  if (gdaki_ctx->gqp_groups) free(gdaki_ctx->gqp_groups);
  if (gdaki_ctx->gqps) free(gdaki_ctx->gqps);
  if (gdaki_ctx->companion_gqps) free(gdaki_ctx->companion_gqps);

  if (gdaki_ctx->counters_table) {
    gdaki_ctx->counters_table->deregister_mr();
    delete gdaki_ctx->counters_table;
  }
  if (gdaki_ctx->signals_table) {
    gdaki_ctx->signals_table->deregister_mr();
    delete gdaki_ctx->signals_table;
  }

  if (gdaki_ctx->sink_buffer.mr) NCCLCHECK(wrap_ibv_dereg_mr(gdaki_ctx->sink_buffer.mr));
  if (gdaki_ctx->sink_buffer.addr) NCCLCHECK(ncclCuMemFree(gdaki_ctx->sink_buffer.addr));

  if (gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle) {
    for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
      struct ncclGinGdakiGPUContext *gin_gdaki_gpu_ctx =
        &gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle->host_buf[ctx_idx];
      if (gin_gdaki_gpu_ctx->gdqp) {
        DOCACHECK(doca_gpu_mem_free(gdaki_ctx->gdev, gin_gdaki_gpu_ctx->gdqp));
      }
      if (gin_gdaki_gpu_ctx->companion_gdqp) {
        DOCACHECK(doca_gpu_mem_free(gdaki_ctx->gdev, gin_gdaki_gpu_ctx->companion_gdqp));
      }
    }
    delete gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle;
  }

  if (gdaki_ctx->ah) {
    DOCACHECK(doca_verbs_ah_attr_destroy(gdaki_ctx->ah));
  }

  if (gdaki_ctx->gdev) {
    DOCACHECK(doca_gpu_destroy(gdaki_ctx->gdev));
  }
  if (gdaki_ctx->ib_pd) NCCLCHECK(wrap_ibv_dealloc_pd(gdaki_ctx->ib_pd));

  if (gdaki_ctx->devHandle) free(gdaki_ctx->devHandle);

  memset(gdaki_ctx, 0, sizeof(*gdaki_ctx));
  free(gdaki_ctx);

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiRegMrSym(void *collComm, void *data, size_t size, int type, void **mhandle,
                                  void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;

  struct gdaki_context *gdaki_ctx = (struct gdaki_context *)cComm->ginCtx;
  struct ibv_mr *mr = nullptr;
  GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle> *gdaki_mhandle_hd_mhandle =
    new GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle>(1);
  GdakiHostGPUMemHandle<__be32> *rkeys_hd_mhandle =
    new GdakiHostGPUMemHandle<__be32>(cComm->nranks);
  __be32 rkey;

  struct gdaki_mem_handle *gdaki_mhandle = nullptr;
  gdaki_mhandle = (struct gdaki_mem_handle *)calloc(1, sizeof(*gdaki_mhandle));
  EQCHECK(gdaki_mhandle, nullptr);

  NCCLCHECK(gdakiRegMr(&mr, gdaki_ctx->ib_pd, data, size,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_ATOMIC));

  rkey = htobe32(mr->rkey);
  NCCLCHECK(cComm->allGather(cComm, &rkey, rkeys_hd_mhandle->host_buf, sizeof(__be32)));
  NCCLCHECK(rkeys_hd_mhandle->copy_h_to_d());

  gdaki_mhandle_hd_mhandle->host_buf->rkeys = rkeys_hd_mhandle->gpu_buf;
  gdaki_mhandle_hd_mhandle->host_buf->lkey = htobe32(mr->lkey);
  NCCLCHECK(gdaki_mhandle_hd_mhandle->copy_h_to_d());

  gdaki_mhandle->type = type;
  gdaki_mhandle->mr = mr;
  gdaki_mhandle->gdaki_mhandle_hd_mhandle = gdaki_mhandle_hd_mhandle;
  gdaki_mhandle->rkeys_hd_mhandle = rkeys_hd_mhandle;

  INFO(NCCL_NET, "[%d] Registered MR: data=%p, size=%zu, lkey(be32)=%#x, rkey(be32)=%#x",
       cComm->rank, data, size, htobe32(mr->lkey), htobe32(mr->rkey));

  *mhandle = (void *)gdaki_mhandle;
  *ginHandle = (void *)gdaki_mhandle_hd_mhandle->gpu_buf;

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiDeregMrSym(void *collComm, void *mhandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct gdaki_mem_handle *gdaki_mhandle = (struct gdaki_mem_handle *)mhandle;
  struct ibv_mr *mr = gdaki_mhandle->mr;

  INFO(NCCL_NET, "[%d] Unregistering MR: lkey(be32)=%#x, rkey(be32)=%#x", cComm->rank,
       htobe32(mr->lkey), htobe32(mr->rkey));

  NCCLCHECK(wrap_ibv_dereg_mr(mr));

  delete gdaki_mhandle->gdaki_mhandle_hd_mhandle;
  delete gdaki_mhandle->rkeys_hd_mhandle;

  memset(gdaki_mhandle, 0, sizeof(*gdaki_mhandle));

  free(gdaki_mhandle);

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiProgress(void *collComm) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct gdaki_context *gdakiCtx = (struct gdaki_context *)cComm->ginCtx;
  const int ncontexts = 1;
  const int nranks = gdakiCtx->collComm->nranks;
  const int nqpsPerRank = ncontexts;
  const int nqpsForComm = nqpsPerRank * nranks;  // Number of QPs for communication

  for (int qpIdx = 0; qpIdx < nqpsForComm; qpIdx++) {
    struct doca_gpu_verbs_qp *qp = gdakiCtx->gqps[qpIdx]->qp_gverbs;
    if (qp->cpu_proxy) {
      DOCACHECK(doca_gpu_verbs_cpu_proxy_progress(qp));
    }

    qp = gdakiCtx->companion_gqps[qpIdx]->qp_gverbs;
    if (qp->cpu_proxy) {
      DOCACHECK(doca_gpu_verbs_cpu_proxy_progress(qp));
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiQueryLastError(void *ginCtx, bool *hasError) {
  struct gdaki_context *gdakiCtx = (struct gdaki_context *)ginCtx;
  bool hasError_ = false;
  const int ncontexts = 1;
  const int nranks = gdakiCtx->collComm->nranks;
  const int nqpsPerRank = ncontexts;
  const int nqpsForComm = nqpsPerRank * nranks;  // Number of QPs for communication

  // We throttle the frequency of these queries since they can easily take 250us.
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
    if (ts.tv_sec - gdakiCtx->last_error_query_time.tv_sec +
          (ts.tv_nsec - gdakiCtx->last_error_query_time.tv_nsec) / 1e9 <
        ncclParamGinErrorQuerySec()) {
      goto exit;
    }
    gdakiCtx->last_error_query_time = ts;
  }

  for (int qpIdx = 0; qpIdx < nqpsForComm; qpIdx++) {
    struct doca_gpu_verbs_qp *qp = gdakiCtx->gqps[qpIdx]->qp_gverbs;
    struct doca_gpu_verbs_qp_error_info errorInfo;
    DOCACHECK(doca_gpu_verbs_query_last_error(qp, &errorInfo));
    hasError_ |= errorInfo.has_error;
    if (hasError_) break;

    qp = gdakiCtx->companion_gqps[qpIdx]->qp_gverbs;
    DOCACHECK(doca_gpu_verbs_query_last_error(qp, &errorInfo));
    hasError_ |= errorInfo.has_error;
    if (hasError_) break;
  }
exit:
  *hasError = hasError_;
  return ncclSuccess;
}
