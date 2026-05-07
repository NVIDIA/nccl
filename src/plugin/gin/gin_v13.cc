/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v13_t* ncclGin_v13;
static ncclGin_t ncclGin;

ncclResult_t ncclGin_getGinProperties(ncclGinProperties_t* ginProps) {
  ginProps->supportsStrongSignals = true;
  ginProps->supportsVASignals = false;
  return ncclSuccess;
}

ncclGin_t* getNcclGin_v13(void* lib) {
  ncclGin_v13 = (ncclGin_v13_t*)ncclOsDlsym(lib, "ncclGinPlugin_v13");
  if (ncclGin_v13) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded gin plugin %s (v13)", ncclGin_v13->name);
    ncclGin.name = ncclGin_v13->name;
    ncclGin.init = ncclGin_v13->init;
    ncclGin.devices = ncclGin_v13->devices;
    ncclGin.getGinProperties = ncclGin_getGinProperties;
    ncclGin.getProperties = ncclGin_v13->getProperties;
    ncclGin.listen = ncclGin_v13->listen;
    ncclGin.connect = ncclGin_v13->connect;
    ncclGin.createContext = ncclGin_v13->createContext;
    ncclGin.regMrSym = ncclGin_v13->regMrSym;
    ncclGin.regMrSymDmaBuf = ncclGin_v13->regMrSymDmaBuf;
    ncclGin.deregMrSym = ncclGin_v13->deregMrSym;
    ncclGin.destroyContext = ncclGin_v13->destroyContext;
    ncclGin.closeColl = ncclGin_v13->closeColl;
    ncclGin.closeListen = ncclGin_v13->closeListen;
    ncclGin.ginProgress = ncclGin_v13->ginProgress;
    ncclGin.queryLastError = ncclGin_v13->queryLastError;
    ncclGin.finalize = ncclGin_v13->finalize;
    return &ncclGin;
  }
  return nullptr;
}
