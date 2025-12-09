// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comm.h"
#include "proxy.h"

#include "meta/colltrace/ProxyTrace.h"

namespace ncclx::colltrace {
void proxyTraceInfoCopy(ncclProxyOp& proxyOp, ncclComm* comm);

void proxyTraceAddBasicInfo(
    ncclProxyOp& proxyOp,
    int nChannels,
    ncclFunc_t coll);

ncclResult_t proxyTraceInit(
    struct ncclProxyState* state,
    struct ncclComm* comm);

ncclResult_t proxyTraceDestroy(struct ncclProxyState* state);
} // namespace ncclx::colltrace
