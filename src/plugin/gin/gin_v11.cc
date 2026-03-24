/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
#include "proxy.h"
#include "os.h"

static ncclGin_v11_t* ncclGin_v11;
static ncclGin_t ncclGin;

static ncclResult_t ncclGin_createContext(void* collComm, ncclGinConfig_t* config,
    void** ginCtx, ncclNetDeviceHandle_t** devHandle) {
  if (config->nContexts > 1) {
    WARN("GIN plugin v11 does not support multiple contexts");
    return ncclInvalidUsage;
  }
  if (ncclGin_v11->createContext == NULL) {
    *ginCtx = collComm;
    return ncclSuccess;
  } else {
    NCCLCHECK(ncclGin_v11->createContext(collComm, config->nSignals, config->nCounters, ginCtx, devHandle));
  }
  return ncclSuccess;
}

static ncclResult_t ncclGin_destroyContext(void* ginCtx) {
  if (ncclGin_v11->destroyContext) {
    NCCLCHECK(ncclGin_v11->destroyContext(ginCtx));
  }
  return ncclSuccess;
}

static ncclResult_t ncclGin_iput(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
    uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request) {
  if (context != 0) {
    WARN("GIN plugin v11 does not support multiple contexts");
    return ncclInvalidUsage;
  }
  return ncclGin_v11->iput(ginCtx, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, request);
}

static ncclResult_t ncclGin_iputSignal(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
    size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff, void *signalMhandle,
    uint64_t signalValue, uint32_t signalOp, void** request) {
  if (context != 0) {
    WARN("GIN plugin v11 does not support multiple connections");
    return ncclInvalidUsage;
  }
  return ncclGin_v11->iputSignal(ginCtx, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, signalOff, signalMhandle, signalValue, signalOp, request);
}

static ncclResult_t ncclGin_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v11_t props_v11;
  NCCLCHECK(ncclGin_v11->getProperties(dev, &props_v11));
  props->name = props_v11.name;
  props->pciPath = props_v11.pciPath;
  props->guid = props_v11.guid;
  props->ptrSupport = props_v11.ptrSupport;
  props->regIsGlobal = props_v11.regIsGlobal;
  props->forceFlush = props_v11.forceFlush;
  props->speed = props_v11.speed;
  props->port = props_v11.port;
  props->latency = props_v11.latency;
  props->maxComms = props_v11.maxComms;
  props->maxRecvs = props_v11.maxRecvs;
  props->netDeviceType = props_v11.netDeviceType;
  props->netDeviceVersion = props_v11.netDeviceVersion;
  props->vProps.ndevs = props_v11.vProps.ndevs;
  for (int i = 0; i < props_v11.vProps.ndevs; i++)
    props->vProps.devs[i] = props_v11.vProps.devs[i];
  props->maxP2pBytes = props_v11.maxP2pBytes;
  props->maxCollBytes = props_v11.maxCollBytes;
  props->maxMultiRequestSize = props_v11.maxMultiRequestSize;
  // Undefined to be ignore in NCCL core
  props->railId = NCCL_NET_ID_UNDEF;
  props->planeId = NCCL_NET_ID_UNDEF;
  return ncclSuccess;
}

ncclGin_t* getNcclGin_v11(void* lib) {
  ncclGin_v11 = (ncclGin_v11_t*)ncclOsDlsym(lib, "ncclGinPlugin_v11");
  if (ncclGin_v11) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded gin plugin %s (v11)", ncclGin_v11->name);
    ncclGin.name = ncclGin_v11->name;
    ncclGin.init = ncclGin_v11->init;
    ncclGin.devices = ncclGin_v11->devices;
    ncclGin.getProperties = ncclGin_getProperties;
    ncclGin.listen = ncclGin_v11->listen;
    ncclGin.connect = ncclGin_v11->connect;
    ncclGin.createContext = ncclGin_createContext;
    ncclGin.regMrSym = ncclGin_v11->regMrSym;
    ncclGin.regMrSymDmaBuf = ncclGin_v11->regMrSymDmaBuf;
    ncclGin.deregMrSym = ncclGin_v11->deregMrSym;
    ncclGin.destroyContext = ncclGin_destroyContext;
    ncclGin.closeColl = ncclGin_v11->closeColl;
    ncclGin.closeListen = ncclGin_v11->closeListen;
    ncclGin.iput = ncclGin_iput;
    ncclGin.iputSignal = ncclGin_iputSignal;
    ncclGin.iget = NULL;
    ncclGin.iflush = NULL;
    ncclGin.test = ncclGin_v11->test;
    ncclGin.ginProgress = ncclGin_v11->ginProgress;
    ncclGin.queryLastError = ncclGin_v11->queryLastError;
    ncclGin.finalize = ncclGin_v11->finalize;
    return &ncclGin;
  }
  return nullptr;
}
