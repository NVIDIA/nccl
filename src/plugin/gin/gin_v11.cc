/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v11_t* ncclGin_v11;
static ncclGin_t ncclGin;

static ncclResult_t ncclGin_getProperties(int dev, ncclNetProperties_t* props) {
  return ncclGin_v11->getProperties(dev, (ncclNetProperties_v11_t*)props);
}

static ncclResult_t ncclGin_connect(void* ctx, void* handles[], int nranks, int rank,
                                    int nConnections, int queueDepth,
                                    void* listenComm, void** collComm) {
  if (nConnections > 1) {
    WARN("GIN plugin v11 does not support multiple connections");
    return ncclInvalidUsage;
  }
  if (queueDepth != 0) {
    WARN("GIN plugin v11 does not support specifying queue depth");
    return ncclInvalidUsage;
  }
  return ncclGin_v11->connect(ctx, handles, nranks, rank, listenComm, collComm);
}

static ncclResult_t ncclGin_createContext(void* collComm, int nSignals, int nCounters, int nContexts, void** ginCtx, ncclNetDeviceHandle_t** devHandle) {
  if (nContexts > 1) {
    WARN("GIN plugin v11 does not support multiple contexts");
    return ncclInvalidUsage;
  }
  return ncclGin_v11->createContext(collComm, nSignals, nCounters, ginCtx, devHandle);
}

static ncclResult_t ncclGin_iput(void* collComm, uint64_t srcOff, void* srcMhandle, size_t size,
    uint64_t dstOff, void* dstMhandle, uint32_t rank, int connectionId, void** request) {
  if (connectionId != 0) {
    WARN("GIN plugin v11 does not support multiple connections");
    return ncclInvalidUsage;
  }
  return ncclGin_v11->iput(collComm, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, request);
}

static ncclResult_t ncclGin_iputSignal(void* collComm, uint64_t srcOff, void* srcMhandle,
    size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff, void *signalMhandle,
    uint64_t signalValue, uint32_t signalOp, int connectionId, void** request) {
  if (connectionId != 0) {
    WARN("GIN plugin v11 does not support multiple connections");
    return ncclInvalidUsage;
  }
  return ncclGin_v11->iputSignal(collComm, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, signalOff, signalMhandle, signalValue, signalOp, request);
}
ncclGin_t* getNcclGin_v11(void* lib) {
  ncclGin_v11 = (ncclGin_v11_t*)dlsym(lib, "ncclGinPlugin_v11");
  if (ncclGin_v11) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded gin plugin %s (v11)", ncclGin_v11->name);
    ncclGin.name = ncclGin_v11->name;
    ncclGin.init = ncclGin_v11->init;
    ncclGin.devices = ncclGin_v11->devices;
    ncclGin.getProperties = ncclGin_getProperties;
    ncclGin.listen = ncclGin_v11->listen;
    ncclGin.connect = ncclGin_connect;
    ncclGin.createContext = ncclGin_createContext;
    ncclGin.regMrSym = ncclGin_v11->regMrSym;
    ncclGin.regMrSymDmaBuf = ncclGin_v11->regMrSymDmaBuf;
    ncclGin.deregMrSym = ncclGin_v11->deregMrSym;
    ncclGin.destroyContext = ncclGin_v11->destroyContext;
    ncclGin.closeColl = ncclGin_v11->closeColl;
    ncclGin.closeListen = ncclGin_v11->closeListen;
    ncclGin.iput = ncclGin_iput;
    ncclGin.iputSignal = ncclGin_iputSignal;
    ncclGin.test = ncclGin_v11->test;
    ncclGin.ginProgress = ncclGin_v11->ginProgress;
    ncclGin.queryLastError = ncclGin_v11->queryLastError;
    ncclGin.finalize = ncclGin_v11->finalize;
    return &ncclGin;
  }
  return nullptr;
}
