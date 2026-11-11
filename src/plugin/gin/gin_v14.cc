/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
#include "proxy.h"
#include <dlfcn.h>
#include <string.h>

static ncclGin_v14_t* ncclGin_v14;
static ncclGin_t ncclGin;

static ncclResult_t ncclGin_v14_connect(void* ctx, void* handles[], int nranks, int rank, void* listenComm,
                                        void** collComm, volatile uint32_t* abortFlag) {
  (void)abortFlag;
  return ncclGin_v14->connect(ctx, handles, nranks, rank, listenComm, collComm);
}

static ncclResult_t ncclGin_v14_getGinProperties(ncclGinProperties_t* ginProps) {
  ncclGinProperties_v14_t ginProps_v14;
  memset(&ginProps_v14, 0, sizeof(ginProps_v14));
  NCCLCHECK(ncclGin_v14->getGinProperties(&ginProps_v14));
  ginProps->supportsStrongSignals = ginProps_v14.supportsStrongSignals;
  ginProps->supportsVASignals = ginProps_v14.supportsVASignals;
  // v14 plugins predate the barrier preference; NCCL's default barrier is what they have always had.
  ginProps->barrierOptions = NCCL_GIN_BARRIER_DEFAULT;
  return ncclSuccess;
}

static ncclResult_t ncclGin_v14_createContext(void* collComm, ncclGinConfig_v15_t* config, void** ginCtx,
                                              ncclNetDeviceHandle_v11_t** devHandle) {
  ncclGinConfig_v14_t config_v14;
  memset(&config_v14, 0, sizeof(config_v14));
  config_v14.nSignals = config->nSignals;
  config_v14.nCounters = config->nCounters;
  config_v14.nContexts = config->nContexts;
  config_v14.queueDepth = config->queueDepth;
  config_v14.trafficClass = config->trafficClass;
  config_v14.backendVersion = config->backendVersion;
  config_v14.rankStride = 1;
  return ncclGin_v14->createContext(collComm, &config_v14, ginCtx, devHandle);
}

ncclGin_t* getNcclGin_v14(void* lib) {
  ncclGin_v14 = (ncclGin_v14_t*)dlsym(lib, "ncclGinPlugin_v14");
  if (ncclGin_v14) {
    INFO(NCCL_INIT | NCCL_NET, "GIN/Plugin: Loaded gin plugin %s (v14)", ncclGin_v14->name);
    ncclGin.name = ncclGin_v14->name;
    ncclGin.init = ncclGin_v14->init;
    ncclGin.devices = ncclGin_v14->devices;
    ncclGin.getGinProperties = ncclGin_v14_getGinProperties;
    ncclGin.getProperties = ncclGin_v14->getProperties;
    ncclGin.listen = ncclGin_v14->listen;
    ncclGin.connect = ncclGin_v14_connect;
    ncclGin.createContext = ncclGin_v14_createContext;
    ncclGin.regMrSym = ncclGin_v14->regMrSym;
    ncclGin.regMrSymDmaBuf = ncclGin_v14->regMrSymDmaBuf;
    ncclGin.deregMrSym = ncclGin_v14->deregMrSym;
    ncclGin.destroyContext = ncclGin_v14->destroyContext;
    ncclGin.closeColl = ncclGin_v14->closeColl;
    ncclGin.closeListen = ncclGin_v14->closeListen;
    ncclGin.ginProgress = ncclGin_v14->ginProgress;
    ncclGin.queryLastError = ncclGin_v14->queryLastError;
    ncclGin.finalize = ncclGin_v14->finalize;
    return &ncclGin;
  }
  return nullptr;
}
