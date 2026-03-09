/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_net.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v12_t* ncclGin_v12;
static ncclGin_t ncclGin;

static ncclResult_t ncclGin_updateContextQosParams(void* collComm, const int contextIndex, const int trafficClass) {
  WARN("GIN plugin v12 does not support updateContextQosParams");
  return ncclInvalidUsage;
}

ncclGin_t* getNcclGin_v12(void* lib) {
  ncclGin_v12 = (ncclGin_v12_t*)dlsym(lib, "ncclGinPlugin_v12");
  if (ncclGin_v12) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded gin plugin %s (v12)", ncclGin_v12->name);
    ncclGin.name = ncclGin_v12->name;
    ncclGin.init = ncclGin_v12->init;
    ncclGin.devices = ncclGin_v12->devices;
    ncclGin.getProperties = ncclGin_v12->getProperties;
    ncclGin.listen = ncclGin_v12->listen;
    ncclGin.connect = ncclGin_v12->connect;
    ncclGin.createContext = ncclGin_v12->createContext;
    ncclGin.regMrSym = ncclGin_v12->regMrSym;
    ncclGin.regMrSymDmaBuf = ncclGin_v12->regMrSymDmaBuf;
    ncclGin.deregMrSym = ncclGin_v12->deregMrSym;
    ncclGin.destroyContext = ncclGin_v12->destroyContext;
    ncclGin.closeColl = ncclGin_v12->closeColl;
    ncclGin.closeListen = ncclGin_v12->closeListen;
    ncclGin.iput = ncclGin_v12->iput;
    ncclGin.iputSignal = ncclGin_v12->iputSignal;
    ncclGin.test = ncclGin_v12->test;
    ncclGin.ginProgress = ncclGin_v12->ginProgress;
    ncclGin.queryLastError = ncclGin_v12->queryLastError;
    ncclGin.finalize = ncclGin_v12->finalize;
    ncclGin.updateContextQosParams = ncclGin_updateContextQosParams;
    return &ncclGin;
  }
  return nullptr;
}
