#include <cstdint>
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"

typedef struct {
  int nranks;
  uint32_t queueSize;
  ncclGinProxyGfd_t* queues;
  uint32_t* pis;
  // The consumer indices will reside in CPU or GPU memory depending on the availability of GDR
  uint32_t* cis;

  uint64_t* counters;
  uint64_t* signals;
} ncclGinProxyGpuCtx_v1_t;

static_assert(sizeof(ncclGinProxyGpuCtx_v1_t) == 48);

void ncclGinProxyGpuCtx_v1_init(void* ctxArray, int idx, int nRanks, uint32_t queueSize, ncclGinProxyGfd_t* queues,
                                uint32_t* pis, uint32_t* cis, uint64_t* counters, uint64_t* signals) {
  ncclGinProxyGpuCtx_v1_t* ctx = (ncclGinProxyGpuCtx_v1_t*)ctxArray + idx;
  ctx->nranks = nRanks;
  ctx->queueSize = queueSize;
  ctx->queues = queues;
  ctx->pis = pis;
  ctx->cis = cis;
  ctx->counters = counters;
  ctx->signals = signals;
}
