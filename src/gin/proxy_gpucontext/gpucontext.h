#include "gpucontext_v1.h"
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"
#include <algorithm>
#include "alloc.h"

constexpr size_t NCCL_GIN_PROXY_GPU_CONTEXT_MAX_SIZE =
  std::max({sizeof(ncclGinProxyGpuCtx_v1_t), sizeof(ncclGinProxyGpuCtx_t)});

const int NCCL_GIN_PROXY_GPU_CONTEXT_VERSION = 2;

void ncclGinProxyGpuCtx_initCurrent(void* ctxArray, int idx, int nranks, uint32_t queueSize, ncclGinProxyGfd_t* queues,
                                    uint32_t* pis, uint32_t* cis, uint64_t* counters, uint64_t* signals,
                                    uint64_t* signalOffsets, uint32_t* lastIssuedGet, uint32_t* lastVisibleGet) {
  ncclGinProxyGpuCtx_t* ctx = (ncclGinProxyGpuCtx_t*)ctxArray + idx;
  ctx->nranks = nranks;
  ctx->queueSize = queueSize;
  ctx->queues = queues;
  ctx->pis = pis;
  ctx->cis = cis;
  ctx->counters = counters;
  ctx->signals = signals;
  ctx->signalOffsets = signalOffsets;
  ctx->lastIssuedGet = lastIssuedGet;
  ctx->lastVisibleGet = lastVisibleGet;
}

ncclResult_t ncclGinProxyGpuCtx_init(int version, void* ctxArray, int idx, int nranks, uint32_t queueSize,
                                     ncclGinProxyGfd_t* queues, uint32_t* pis, uint32_t* cis, uint64_t* counters,
                                     uint64_t* signals, uint64_t* signalOffsets, uint32_t* lastIssuedGet,
                                     uint32_t* lastVisibleGet) {
  switch (version) {
  case 1:
    ncclGinProxyGpuCtx_v1_init(ctxArray, idx, nranks, queueSize, queues, pis, cis, counters, signals);
    break;
  case NCCL_GIN_PROXY_GPU_CONTEXT_VERSION:
    ncclGinProxyGpuCtx_initCurrent(ctxArray, idx, nranks, queueSize, queues, pis, cis, counters, signals, signalOffsets,
                                   lastIssuedGet, lastVisibleGet);
    break;
  default:
    WARN("Invalid GIN proxy backend version %d", version);
    return ncclInternalError;
  }
  return ncclSuccess;
}
