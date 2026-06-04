/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>
#include <string.h>
#include "nccl/gin.h"

#define __hidden __attribute__((visibility("hidden")))

#define NCCL_MAX_NET_SIZE_BYTES (1*1024*1024*1024*1024L) // 1TB

/* Opaque data structures */

struct ginContext {
  uint64_t commId;
};

struct ginListenComm {
  int dev;
};

struct ginCollComm {
  int nranks;
  int rank;
};

struct ginCtx {
  int nSignals;
  int nCounters;
};

struct ginMemHandle {
  void* data;
  size_t size;
};

struct ginRequest {
  int done;
};

/* Shared functions (identical across v11 and v13) */

__hidden ncclResult_t ginInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  struct ginContext* c = (struct ginContext*)calloc(1, sizeof(*c));
  if (c == NULL) return ncclSystemError;
  c->commId = commId;
  *ctx = c;
  return ncclSuccess;
}

__hidden ncclResult_t ginDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

__hidden ncclResult_t ginListen(void* ctx, int dev, void* handle, void** listenComm) {
  struct ginListenComm* comm = (struct ginListenComm*)calloc(1, sizeof(*comm));
  if (comm == NULL) return ncclSystemError;
  comm->dev = dev;
  memset(handle, 0, NCCL_GIN_HANDLE_MAXSIZE);
  *listenComm = comm;
  return ncclSuccess;
}

__hidden ncclResult_t ginConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  struct ginCollComm* comm = (struct ginCollComm*)calloc(1, sizeof(*comm));
  if (comm == NULL) return ncclSystemError;
  comm->nranks = nranks;
  comm->rank = rank;
  *collComm = comm;
  return ncclSuccess;
}

__hidden ncclResult_t ginRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle, void** ginHandle) {
  struct ginMemHandle* m = (struct ginMemHandle*)calloc(1, sizeof(*m));
  if (m == NULL) return ncclSystemError;
  m->data = data;
  m->size = size;
  *mhandle = m;
  *ginHandle = m;
  return ncclSuccess;
}

__hidden ncclResult_t ginRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ginRegMrSym(collComm, data, size, type, mrFlags, mhandle, ginHandle);
}

__hidden ncclResult_t ginDeregMrSym(void* collComm, void* mhandle) {
  free(mhandle);
  return ncclSuccess;
}

__hidden ncclResult_t ginDestroyContext(void* ginCtx) {
  free(ginCtx);
  return ncclSuccess;
}

__hidden ncclResult_t ginCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

__hidden ncclResult_t ginCloseListen(void* listenComm) {
  free(listenComm);
  return ncclSuccess;
}

__hidden ncclResult_t ginTest(void* collComm, void* request, int* done) {
  struct ginRequest* r = (struct ginRequest*)request;
  *done = 1;
  free(r);
  return ncclSuccess;
}

__hidden ncclResult_t ginQueryLastError(void* ginCtx, bool* hasError) {
  *hasError = false;
  return ncclSuccess;
}

__hidden ncclResult_t ginFinalize(void* ctx) {
  free(ctx);
  return ncclSuccess;
}

__hidden struct ginRequest* ginAllocRequest(void) {
  struct ginRequest* r = (struct ginRequest*)calloc(1, sizeof(*r));
  return r;
}

/* v11-specific functions */

__hidden ncclResult_t ginGetProperties_v11(int dev, ncclNetProperties_v11_t* props) {
  props->name = (char*)"GIN Example";
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
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

__hidden ncclResult_t ginCreateContext_v11(void* collComm, int nSignals, int nCounters, void** ginCtxOut, ncclNetDeviceHandle_v11_t** devHandle) {
  struct ginCtx* gc = (struct ginCtx*)calloc(1, sizeof(*gc));
  if (gc == NULL) return ncclSystemError;
  gc->nSignals = nSignals;
  gc->nCounters = nCounters;

  ncclNetDeviceHandle_v11_t* dh = (ncclNetDeviceHandle_v11_t*)calloc(1, sizeof(*dh));
  if (dh == NULL) { free(gc); return ncclSystemError; }
  dh->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  dh->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  dh->handle = NULL;
  dh->size = 0;
  dh->needsProxyProgress = 0;

  *ginCtxOut = gc;
  *devHandle = dh;
  return ncclSuccess;
}

__hidden ncclResult_t ginIput_v11(void* collComm, uint64_t srcOff, void* srcMhandle, size_t size,
    uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request) {
  struct ginRequest* r = ginAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t ginIputSignal_v11(void* collComm, uint64_t srcOff, void* srcMhandle,
    size_t size, uint64_t dstOff, void* dstMhandle,
    uint32_t rank, uint64_t signalOff, void* signalMhandle,
    uint64_t signalValue, uint32_t signalOp, void** request) {
  struct ginRequest* r = ginAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t ginProgress_v11(void* collComm) {
  return ncclSuccess;
}

/* v12-specific functions */

__hidden ncclResult_t ginConnect_v12(void* ctx, void* handles[], int nranks, int rank, int nConnections,
    int queueDepth, void* listenComm, void** collComm) {
  struct ginCollComm* comm = (struct ginCollComm*)calloc(1, sizeof(*comm));
  if (comm == NULL) return ncclSystemError;
  comm->nranks = nranks;
  comm->rank = rank;
  *collComm = comm;
  return ncclSuccess;
}

__hidden ncclResult_t ginCreateContext_v12(void* collComm, int nSignals, int nCounters, int nContexts, void** ginCtxOut, ncclNetDeviceHandle_v11_t** devHandle) {
  struct ginCtx* gc = (struct ginCtx*)calloc(1, sizeof(*gc));
  if (gc == NULL) return ncclSystemError;
  gc->nSignals = nSignals;
  gc->nCounters = nCounters;

  ncclNetDeviceHandle_v11_t* dh = (ncclNetDeviceHandle_v11_t*)calloc(1, sizeof(*dh));
  if (dh == NULL) { free(gc); return ncclSystemError; }
  dh->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  dh->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  dh->handle = NULL;
  dh->size = 0;
  dh->needsProxyProgress = 0;

  *ginCtxOut = gc;
  *devHandle = dh;
  return ncclSuccess;
}

__hidden ncclResult_t ginIput_v12(void* collComm, uint64_t srcOff, void* srcMhandle, size_t size,
    uint64_t dstOff, void* dstMhandle, uint32_t rank, int connectionId, void** request) {
  struct ginRequest* r = ginAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t ginIputSignal_v12(void* collComm, uint64_t srcOff, void* srcMhandle,
    size_t size, uint64_t dstOff, void* dstMhandle,
    uint32_t rank, uint64_t signalOff, void* signalMhandle,
    uint64_t signalValue, uint32_t signalOp, int connectionId, void** request) {
  struct ginRequest* r = ginAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

/* v13-specific functions */

__hidden ncclResult_t ginGetProperties_v13(int dev, ncclNetProperties_v12_t* props) {
  props->name = (char*)"GIN Example";
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
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxMultiRequestSize = 1;
  props->railId = 0;
  props->planeId = 0;
  return ncclSuccess;
}

__hidden ncclResult_t ginCreateContext_v13(void* collComm, ncclGinConfig_v13_t* config, void** ginCtxOut, ncclNetDeviceHandle_v11_t** devHandle) {
  struct ginCtx* gc = (struct ginCtx*)calloc(1, sizeof(*gc));
  if (gc == NULL) return ncclSystemError;
  gc->nSignals = config->nSignals;
  gc->nCounters = config->nCounters;

  ncclNetDeviceHandle_v11_t* dh = (ncclNetDeviceHandle_v11_t*)calloc(1, sizeof(*dh));
  if (dh == NULL) { free(gc); return ncclSystemError; }
  dh->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  dh->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  dh->handle = NULL;
  dh->size = 0;
  dh->needsProxyProgress = 0;

  *ginCtxOut = gc;
  *devHandle = dh;
  return ncclSuccess;
}

__hidden ncclResult_t ginIput_v13(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
    uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request) {
  struct ginRequest* r = ginAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t ginIputSignal_v13(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
    size_t size, uint64_t dstOff, void* dstMhandle,
    uint32_t rank, uint64_t signalOff, void* signalMhandle,
    uint64_t signalValue, uint32_t signalOp, void** request) {
  struct ginRequest* r = ginAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t ginIget_v13(void* ginCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
    uint64_t localOff, void* localMhandle, uint32_t rank, void** request) {
  struct ginRequest* r = ginAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t ginIflush_v13(void* ginCtx, int context, void* mhandle, uint32_t rank, void** request) {
  struct ginRequest* r = ginAllocRequest();
  if (r == NULL) return ncclSystemError;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t ginProgress_v13(void* ginCtx) {
  return ncclSuccess;
}

/* Exported plugin structs */

const ncclGin_v11_t ncclGinPlugin_v11 = {
  .name = "Example",
  .init = ginInit,
  .devices = ginDevices,
  .getProperties = ginGetProperties_v11,
  .listen = ginListen,
  .connect = ginConnect,
  .createContext = ginCreateContext_v11,
  .regMrSym = ginRegMrSym,
  .regMrSymDmaBuf = ginRegMrSymDmaBuf,
  .deregMrSym = ginDeregMrSym,
  .destroyContext = ginDestroyContext,
  .closeColl = ginCloseColl,
  .closeListen = ginCloseListen,
  .iput = ginIput_v11,
  .iputSignal = ginIputSignal_v11,
  .test = ginTest,
  .ginProgress = ginProgress_v11,
  .queryLastError = ginQueryLastError,
  .finalize = ginFinalize,
};

const ncclGin_v12_t ncclGinPlugin_v12 = {
  .name = "Example",
  .init = ginInit,
  .devices = ginDevices,
  .getProperties = ginGetProperties_v11,
  .listen = ginListen,
  .connect = ginConnect_v12,
  .createContext = ginCreateContext_v12,
  .regMrSym = ginRegMrSym,
  .regMrSymDmaBuf = ginRegMrSymDmaBuf,
  .deregMrSym = ginDeregMrSym,
  .destroyContext = ginDestroyContext,
  .closeColl = ginCloseColl,
  .closeListen = ginCloseListen,
  .iput = ginIput_v12,
  .iputSignal = ginIputSignal_v12,
  .test = ginTest,
  .ginProgress = ginProgress_v11,
  .queryLastError = ginQueryLastError,
  .finalize = ginFinalize,
};

const ncclGin_v13_t ncclGinPlugin_v13 = {
  .name = "Example",
  .init = ginInit,
  .devices = ginDevices,
  .getProperties = ginGetProperties_v13,
  .listen = ginListen,
  .connect = ginConnect,
  .createContext = ginCreateContext_v13,
  .regMrSym = ginRegMrSym,
  .regMrSymDmaBuf = ginRegMrSymDmaBuf,
  .deregMrSym = ginDeregMrSym,
  .destroyContext = ginDestroyContext,
  .closeColl = ginCloseColl,
  .closeListen = ginCloseListen,
  .iput = ginIput_v13,
  .iputSignal = ginIputSignal_v13,
  .iget = ginIget_v13,
  .iflush = ginIflush_v13,
  .test = ginTest,
  .ginProgress = ginProgress_v13,
  .queryLastError = ginQueryLastError,
  .finalize = ginFinalize,
};
