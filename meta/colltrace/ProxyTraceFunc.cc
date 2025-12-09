// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/colltrace/ProxyTraceFunc.h"

#include "meta/colltrace/ProxyTrace.h"
#include "meta/cvars/nccl_cvars.h"

namespace ncclx::colltrace {
void proxyTraceInfoCopy(ncclProxyOp& proxyOp, ncclComm* comm) {
  proxyOp.traceArgs.collInfo.commHash = comm->commHash;
  proxyOp.traceArgs.collInfo.opCount = comm->opCount;
  proxyOp.traceArgs.rank = comm->rank;

  proxyOp.traceArgs.remoteRank = proxyOp.root;
}

void proxyTraceAddBasicInfo(
    ncclProxyOp& proxyOp,
    int nChannels,
    ncclFunc_t coll) {
  proxyOp.traceArgs.collInfo.nChannels = nChannels;
  proxyOp.traceArgs.collInfo.coll = coll;
}

ncclResult_t proxyTraceInit(
    struct ncclProxyState* state,
    struct ncclComm* comm) {
  if (NCCL_PROXYTRACE.empty()) {
    return ncclSuccess;
  }
  try {
    state->trace = std::make_unique<ProxyTrace>();
  } catch (const std::exception& e) {
    WARN(
        "PROXYTRACE: failed to initialize ProxyTrace, comm %p commDesc %s: %s",
        comm,
        comm->config.commDesc,
        e.what());
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t proxyTraceDestroy(struct ncclProxyState* state) {
  if (state->trace) {
    state->trace.reset();
  }
  return ncclSuccess;
}
} // namespace ncclx::colltrace
