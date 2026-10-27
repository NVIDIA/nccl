/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v15_t* ncclGin_v15;
static ncclGin_t ncclGin;

static ncclResult_t ncclGin_v15_createContext(void* collComm, ncclGinConfig_v16_t* config, void** ginCtx,
                                              ncclNetDeviceHandle_v11_t** devHandle) {
  ncclGinConfig_v15_t config_v15;
  memset(&config_v15, 0, sizeof(config_v15));
  config_v15.nSignals = config->nSignals;
  config_v15.nCounters = config->nCounters;
  config_v15.nContexts = config->nContexts;
  config_v15.queueDepth = config->queueDepth;
  config_v15.trafficClass = config->trafficClass;
  config_v15.backendVersion = config->backendVersion;
  config_v15.peerArray = config->peerArray;
  config_v15.peerArrayCount = config->peerArrayCount;
  return ncclGin_v15->createContext(collComm, &config_v15, ginCtx, devHandle);
}

ncclGin_t* getNcclGin_v15(void* lib) {
  ncclGin_v15 = (ncclGin_v15_t*)dlsym(lib, "ncclGinPlugin_v15");
  if (ncclGin_v15) {
    INFO(NCCL_INIT | NCCL_NET, "GIN/Plugin: Loaded gin plugin %s (v15)", ncclGin_v15->name);
    ncclGin.name = ncclGin_v15->name;
    ncclGin.init = ncclGin_v15->init;
    ncclGin.devices = ncclGin_v15->devices;
    ncclGin.getGinProperties = ncclGin_v15->getGinProperties;
    ncclGin.getProperties = ncclGin_v15->getProperties;
    ncclGin.listen = ncclGin_v15->listen;
    ncclGin.connect = ncclGin_v15->connect;
    ncclGin.createContext = ncclGin_v15_createContext;
    ncclGin.regMrSym = ncclGin_v15->regMrSym;
    ncclGin.regMrSymDmaBuf = ncclGin_v15->regMrSymDmaBuf;
    ncclGin.deregMrSym = ncclGin_v15->deregMrSym;
    ncclGin.destroyContext = ncclGin_v15->destroyContext;
    ncclGin.closeColl = ncclGin_v15->closeColl;
    ncclGin.closeListen = ncclGin_v15->closeListen;
    ncclGin.ginProgress = ncclGin_v15->ginProgress;
    ncclGin.queryLastError = ncclGin_v15->queryLastError;
    ncclGin.finalize = ncclGin_v15->finalize;
    return &ncclGin;
  }
  return nullptr;
}
