/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>
#include <string.h>
#include "nccl/rma.h"

#define __hidden __attribute__((visibility("hidden")))

#define NCCL_MAX_NET_SIZE_BYTES (1*1024*1024*1024*1024L) // 1TB

/* Opaque data structures */

struct rmaContext {
  uint64_t commId;
};

struct rmaListenComm {
  int dev;
};

struct rmaCollComm {
  int nranks;
  int rank;
};

struct rmaCtx {
  int nContexts;
  int trafficClass;
  int rankStride;
};

struct rmaMemHandle {
  void* data;
  size_t size;
};

struct rmaRequest {
  int done;
};

/* Shared functions */

__hidden ncclResult_t rmaInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  struct rmaContext* c = (struct rmaContext*)calloc(1, sizeof(*c));
  if (c == NULL) return ncclSystemError;
  c->commId = commId;
  *ctx = c;
  return ncclSuccess;
}

__hidden ncclResult_t rmaDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

__hidden ncclResult_t rmaGetProperties(int dev, ncclNetProperties_v12_t* props) {
  props->name = (char*)"RMA Example";
  props->pciPath = NULL;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_CUDA;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = 100000;
  props->port = 0;
  props->latency = 0;
  props->maxComms = 1024 * 1024;
  props->maxRecvs = 1;
  props->netDeviceType = 0; // NCCL_NET_DEVICE_HOST
  props->netDeviceVersion = 0;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxMultiRequestSize = 1;
  props->railId = 0;
  props->planeId = 0;
  return ncclSuccess;
}

__hidden ncclResult_t rmaListen(void* ctx, int dev, void* handle, void** listenComm) {
  struct rmaListenComm* comm = (struct rmaListenComm*)calloc(1, sizeof(*comm));
  if (comm == NULL) return ncclSystemError;
  comm->dev = dev;
  memset(handle, 0, NCCL_RMA_HANDLE_MAXSIZE);
  *listenComm = comm;
  return ncclSuccess;
}

__hidden ncclResult_t rmaConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  struct rmaCollComm* comm = (struct rmaCollComm*)calloc(1, sizeof(*comm));
  if (comm == NULL) return ncclSystemError;
  comm->nranks = nranks;
  comm->rank = rank;
  *collComm = comm;
  return ncclSuccess;
}

__hidden ncclResult_t rmaCreateContext(void* collComm, ncclRmaConfig_v14_t* config, void** rmaCtxOut) {
  struct rmaCtx* ctx = (struct rmaCtx*)calloc(1, sizeof(*ctx));
  if (ctx == NULL) return ncclSystemError;
  ctx->nContexts = config->nContexts;
  ctx->trafficClass = config->trafficClass;
  ctx->rankStride = config->rankStride;
  *rmaCtxOut = ctx;
  return ncclSuccess;
}

__hidden ncclResult_t rmaRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle) {
  struct rmaMemHandle* m = (struct rmaMemHandle*)calloc(1, sizeof(*m));
  if (m == NULL) return ncclSystemError;
  m->data = data;
  m->size = size;
  *mhandle = m;
  return ncclSuccess;
}

__hidden ncclResult_t rmaRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle) {
  return rmaRegMrSym(collComm, data, size, type, mrFlags, mhandle);
}

__hidden ncclResult_t rmaDeregMrSym(void* collComm, void* mhandle) {
  free(mhandle);
  return ncclSuccess;
}

__hidden ncclResult_t rmaDestroyContext(void* rmaCtx) {
  free(rmaCtx);
  return ncclSuccess;
}

__hidden ncclResult_t rmaCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

__hidden ncclResult_t rmaCloseListen(void* listenComm) {
  free(listenComm);
  return ncclSuccess;
}

__hidden ncclResult_t rmaQueryLastError(void* rmaCtx, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

__hidden ncclResult_t rmaFinalize(void* ctx) {
  free(ctx);
  return ncclSuccess;
}

__hidden struct rmaRequest* rmaAllocRequest(void) {
  return (struct rmaRequest*)calloc(1, sizeof(struct rmaRequest));
}

/* Data operations */

__hidden ncclResult_t rmaIput(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
    uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request) {
  struct rmaRequest* r = rmaAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t rmaIputSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle,
    size_t size, uint64_t dstOff, void* dstMhandle,
    uint32_t rank, uint64_t signalOff, void* signalMhandle,
    uint64_t signalValue, uint32_t signalOp, bool isStrongSignal, void** request) {
  struct rmaRequest* r = rmaAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t rmaIget(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
    uint64_t localOff, void* localMhandle, uint32_t rank, void** request) {
  struct rmaRequest* r = rmaAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t rmaIflush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request) {
  struct rmaRequest* r = rmaAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t rmaTest(void* collComm, void* request, int* done) {
  *done = 1;
  free(request);
  return ncclSuccess;
}

__hidden ncclResult_t rmaProgress(void* rmaCtx) {
  return ncclSuccess;
}

/* Exported plugin struct */

const ncclRma_v14_t ncclRmaPlugin_v14 = {
  .name = "Example",
  .init = rmaInit,
  .devices = rmaDevices,
  .getProperties = rmaGetProperties,
  .listen = rmaListen,
  .connect = rmaConnect,
  .createContext = rmaCreateContext,
  .regMrSym = rmaRegMrSym,
  .regMrSymDmaBuf = rmaRegMrSymDmaBuf,
  .deregMrSym = rmaDeregMrSym,
  .destroyContext = rmaDestroyContext,
  .closeColl = rmaCloseColl,
  .closeListen = rmaCloseListen,
  .iput = rmaIput,
  .iputSignal = rmaIputSignal,
  .iget = rmaIget,
  .iflush = rmaIflush,
  .test = rmaTest,
  .rmaProgress = rmaProgress,
  .queryLastError = rmaQueryLastError,
  .finalize = rmaFinalize,
};
