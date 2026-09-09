/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_rma.h"
#include "checks.h"
#include "os.h"
#include <dlfcn.h>

static ncclRma_v14_t* ncclRma_v14;
static ncclRma_t ncclRma;

static ncclResult_t ncclRma_v14_iput(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                     uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags,
                                     void** request) {
  (void)optFlags;
  return ncclRma_v14->iput(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, request);
}

// Drop optFlags (v14 has no aggregate hint); isStrongSignal is passed through.
static ncclResult_t ncclRma_v14_iputSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                           uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                           void* signalMhandle, uint64_t signalValue, uint32_t signalOp,
                                           bool isStrongSignal, uint32_t optFlags, void** request) {
  (void)optFlags;
  return ncclRma_v14->iputSignal(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, signalOff,
                                 signalMhandle, signalValue, signalOp, isStrongSignal, request);
}

static ncclResult_t ncclRma_v14_iget(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
                                     uint64_t localOff, void* localMhandle, uint32_t rank, uint32_t optFlags,
                                     void** request) {
  (void)optFlags;
  return ncclRma_v14->iget(rmaCtx, context, remoteOff, remoteMhandle, size, localOff, localMhandle, rank, request);
}

ncclRma_t* getNcclRma_v14(void* lib) {
  ncclRma_v14 = (ncclRma_v14_t*)dlsym(lib, "ncclRmaPlugin_v14");
  if (ncclRma_v14) {
    INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Loaded rma plugin %s (v14)", ncclRma_v14->name);
    ncclRma.name = ncclRma_v14->name;
    ncclRma.init = ncclRma_v14->init;
    ncclRma.devices = ncclRma_v14->devices;
    ncclRma.getProperties = ncclRma_v14->getProperties;
    ncclRma.listen = ncclRma_v14->listen;
    ncclRma.connect = ncclRma_v14->connect;
    ncclRma.createContext = ncclRma_v14->createContext;
    ncclRma.regMrSym = ncclRma_v14->regMrSym;
    ncclRma.regMrSymDmaBuf = ncclRma_v14->regMrSymDmaBuf;
    ncclRma.deregMrSym = ncclRma_v14->deregMrSym;
    ncclRma.destroyContext = ncclRma_v14->destroyContext;
    ncclRma.closeColl = ncclRma_v14->closeColl;
    ncclRma.closeListen = ncclRma_v14->closeListen;
    ncclRma.iput = ncclRma_v14_iput;
    ncclRma.iputSignal = ncclRma_v14_iputSignal;
    ncclRma.iget = ncclRma_v14_iget;
    ncclRma.iflush = ncclRma_v14->iflush;
    ncclRma.test = ncclRma_v14->test;
    ncclRma.rmaProgress = ncclRma_v14->rmaProgress;
    ncclRma.queryLastError = ncclRma_v14->queryLastError;
    ncclRma.finalize = ncclRma_v14->finalize;
    return &ncclRma;
  }
  return nullptr;
}
