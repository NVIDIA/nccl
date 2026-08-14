/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "rma_socket.h"
#include "net.h"
#include "net_socket/common.h"
#include "nccl_device/utility.h"

#include <algorithm>
#include <limits.h>
#include <mutex>
#include <string.h>

static int ncclRmaSocketProxyRefCount;
static std::mutex ncclRmaSocketProxyMutex;

ncclResult_t ncclRmaSocketProxyInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  (void)commId;
  (void)logFunction;
  if (ctx) *ctx = NULL;
  std::lock_guard<std::mutex> lock(ncclRmaSocketProxyMutex);
  if (ncclRmaSocketProxyRefCount) {
    ncclRmaSocketProxyRefCount++;
    return ncclSuccess;
  }
  if (ncclGdrCopy == NULL) ncclGdrCopy = ncclGdrInit();
  if (ncclGdrCopy == NULL) {
    WARN("RMA/Socket : rma_socket requires GDRCopy; and GDR Copy is not available");
    return ncclInternalError;
  }
  NCCLCHECK(ncclNetSocketInitDevices("RMA/Socket"));
  ncclRmaSocketProxyRefCount++;
  return ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyDevices(int* ndev) {
  *ndev = ncclNetIfs;
  return ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyGetProperties(int dev, ncclNetProperties_t* props) {
  NCCLCHECK(ncclNetSocket.getProperties(dev, props));
  // props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  (void)ctx;
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaSocketProxyHandle* handle = (struct ncclRmaSocketProxyHandle*)opaqueHandle;
  memset(handle, 0, sizeof(struct ncclRmaSocketProxyHandle));
  static_assert(sizeof(struct ncclRmaSocketProxyHandle) <= NCCL_NET_HANDLE_MAXSIZE,
                "ncclRmaSocketProxyHandle size too large");
  struct ncclSocket* sock;
  NCCLCHECK(ncclCalloc(&sock, 1));
  NCCLCHECKGOTO(ncclNetSocketCreateListener("RMA/Socket", dev, sock, &handle->connectAddr, &handle->magic), ret, fail);
  *listenComm = sock;
exit:
  return ret;
fail:
  NCCLCHECKIGNORE(ncclSocketClose(sock), ret);
  free(sock);
  goto exit;
}

ncclResult_t ncclRmaSocketProxyCloseListen(void* opaqueListenComm) {
  struct ncclSocket* sock = (struct ncclSocket*)opaqueListenComm;
  ncclResult_t ret = ncclSuccess;
  if (sock) {
    NCCLCHECKIGNORE(ncclSocketClose(sock), ret);
    free(sock);
  }
  return ret;
}

static ncclResult_t ncclRmaSocketProxyRecordError(struct ncclRmaSocketProxyCollComm* comm, ncclResult_t result) {
  if (comm != NULL && result != ncclSuccess && result != ncclInProgress) comm->lastError = result;
  return result;
}

static void ncclRmaSocketProxyCtxAddActiveOp(struct ncclRmaSocketProxyCtx* ctx) {
  if (ctx) ctx->activeOps++;
}

static void ncclRmaSocketProxyCtxRemoveActiveOp(struct ncclRmaSocketProxyCtx* ctx) {
  if (ctx) {
    ctx->activeOps--;
    if (ctx->activeOps < 0) {
      WARN("RMA/Socket : context active operation count became %d", ctx->activeOps);
      ctx->activeOps = 0;
    }
  }
}

static void ncclRmaSocketProxyReleaseRequest(struct ncclRmaSocketProxyRequest* req) {
  if (req == NULL) return;
  if (req->type != ncclRmaSocketProxyRequestTypeFree && req->ctx) ncclRmaSocketProxyCtxRemoveActiveOp(req->ctx);
  memset(req, 0, sizeof(*req));
}

static ncclResult_t ncclRmaSocketProxyGetRequest(struct ncclRmaSocketProxyCtx* ctx, int peer,
                                                 enum ncclRmaSocketProxyRequestType type,
                                                 struct ncclRmaSocketProxyRequest** reqOut) {
  if (ctx == NULL || ctx->collComm == NULL || reqOut == NULL || peer < 0 || peer >= ctx->collComm->nranks ||
      type == ncclRmaSocketProxyRequestTypeFree) {
    WARN("RMA/Socket : invalid request allocation ctx=%p peer=%d type=%d", ctx, peer, type);
    return ncclInvalidArgument;
  }
  *reqOut = NULL;
  for (int i = 0; i < ctx->requestCount; i++) {
    if (ctx->requests[i].type == ncclRmaSocketProxyRequestTypeFree) {
      struct ncclRmaSocketProxyRequest* req = &(ctx->requests[i]);
      memset(req, 0, sizeof(*req));
      req->type = type;
      req->ctx = ctx;
      req->peer = peer;
      req->globalRequestId = ctx->collComm->nextGlobalRequestId++;
      ncclRmaSocketProxyCtxAddActiveOp(ctx);
      *reqOut = req;
      return ncclSuccess;
    }
  }
  WARN("RMA/Socket : unable to allocate socket RMA request");
  return ncclInternalError;
}

static ncclResult_t ncclRmaSocketProxyBuildHeader(struct ncclRmaSocketProxyHeader* header, uint16_t type,
                                                  uint32_t status, uint64_t requestId, uint32_t opFlags, uint64_t mrId,
                                                  uint64_t dstOff, size_t size, uint64_t signalMrId, uint64_t signalOff,
                                                  uint64_t signalValue, uint32_t signalOp) {
  if (size > INT_MAX) {
    WARN("RMA/Socket : message size %zu exceeds socket API limit", size);
    return ncclInvalidArgument;
  }
  const uint32_t allowedFlags = type == ncclRmaSocketProxyMsgTypeIput ? NCCL_RMA_SOCKET_OP_HAS_SIGNAL : 0;
  if ((opFlags & ~allowedFlags) != 0) {
    WARN("RMA/Socket : invalid opFlags 0x%x for message type %u", opFlags, (unsigned)type);
    return ncclInvalidArgument;
  }
  memset(header, 0, sizeof(*header));
  header->type = type;
  header->status = status;
  header->requestId = requestId;
  header->opFlags = opFlags;
  header->signalOp = signalOp;
  header->mrId = mrId;
  header->dstOff = dstOff;
  header->size = size;
  header->signalMrId = signalMrId;
  header->signalOff = signalOff;
  header->signalValue = signalValue;
  return ncclSuccess;
}

static void ncclRmaSocketProxyPopAndCompleteSendTask(struct ncclRmaSocketProxyPeerSender* sender) {
  if (sender == NULL || sender->count == 0) return;
  struct ncclRmaSocketProxySendTask* task = &sender->tasks[sender->head];
  if (task->controlMsg.type == ncclRmaSocketProxyMsgTypeIput) {
    sender->lastCompletedRequestGlobalId = task->controlMsg.requestId;
  }
  memset(task, 0, sizeof(*task));
  sender->head = (sender->head + 1) % sender->capacity;
  sender->count--;
}

static ncclResult_t ncclRmaSocketProxyProgressSendTask(struct ncclRmaSocketProxyCollComm* comm, int peer,
                                                       struct ncclRmaSocketProxySendTask* task) {
  struct ncclRmaSocketProxyPeerSender* sender = &comm->peerSender[peer];
  int closed = 0;

  if (task->controlMsgOffset < (int)sizeof(task->controlMsg)) {
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &sender->sock, &task->controlMsg, sizeof(task->controlMsg),
                                 &task->controlMsgOffset, &closed));
    if (closed) {
      WARN("RMA/Socket : send socket to peer=%d closed while sending control message", peer);
      return ncclRemoteError;
    }
    if (task->controlMsgOffset < (int)sizeof(task->controlMsg)) return ncclSuccess;
  }

  if (task->size == 0) {
    ncclRmaSocketProxyPopAndCompleteSendTask(sender);
    return ncclSuccess;
  }

  /* Bound each progress call to one payload chunk. */
  if (task->chunkSize == 0) {
    if (sender->buffer == NULL) {
      WARN("RMA/Socket : missing send staging buffer");
      return ncclInternalError;
    }
    size_t remaining = task->size - task->payloadOffset;
    int chunkSize = (int)std::min(remaining, (size_t)NCCL_RMA_SOCKET_CHUNK_SIZE);
    const void* src = (const char*)task->srcPtr + task->payloadOffset;
    NCCLCHECK(ncclRmaSocketProxyCopy(sender->buffer, NCCL_PTR_HOST, NULL, src, task->srcType, task->srcHandle,
                                     chunkSize));
    task->chunkSize = chunkSize;
    task->chunkOffset = 0;
  }
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &sender->sock, sender->buffer, task->chunkSize, &task->chunkOffset,
                               &closed));
  if (closed) {
    WARN("RMA/Socket : send socket to peer=%d closed while sending payload", peer);
    return ncclRemoteError;
  }
  if (task->chunkOffset < task->chunkSize) return ncclSuccess;

  task->payloadOffset += task->chunkSize;
  task->chunkSize = 0;
  task->chunkOffset = 0;
  if (task->payloadOffset == task->size) ncclRmaSocketProxyPopAndCompleteSendTask(sender);

  return ncclSuccess;
}

static ncclResult_t ncclRmaSocketProxyProgressComm(struct ncclRmaSocketProxyCollComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm == NULL) return ncclInvalidArgument;

  for (int peer = 0; peer < comm->nranks; peer++) {
    NCCLCHECKGOTO(ncclRmaSocketProxyProgressPeerRecv(comm, peer), ret, fail);
  }
  for (int peer = 0; peer < comm->nranks; peer++) {
    struct ncclRmaSocketProxyPeerSender* sender = &comm->peerSender[peer];
    if (sender->count == 0) continue;
    struct ncclRmaSocketProxySendTask* task = &sender->tasks[sender->head];
    NCCLCHECKGOTO(ncclRmaSocketProxyProgressSendTask(comm, peer, task), ret, fail);
  }
  return ncclSuccess;
fail:
  return ncclRmaSocketProxyRecordError(comm, ret);
}

struct ncclRmaSocketProxyPutArgs {
  struct ncclRmaSocketProxyCtx* ctx;
  uint32_t opFlags;
  uint64_t dstMrId;
  uint64_t dstOff;
  uint64_t signalMrId;
  void* srcPtr;
  struct ncclRmaSocketProxyMrHandle* srcHandle;
  int srcType;
};

/* Validate iput/iputSignal arguments and derive the header fields. Allocates nothing. */
static ncclResult_t ncclRmaSocketProxyValidatePutArgs(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle,
                                                      size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank,
                                                      bool hasSignal, uint64_t signalOff, void* signalMhandle,
                                                      uint32_t signalOp, struct ncclRmaSocketProxyPutArgs* args) {
  if (size > 0 && (srcMhandle == NULL || dstMhandle == NULL)) {
    WARN("RMA/Socket : put with data requires src/dst handles srcMhandle=%p dstMhandle=%p", srcMhandle, dstMhandle);
    return ncclInvalidArgument;
  }
  if (hasSignal && signalMhandle == NULL) {
    WARN("RMA/Socket : putSignal requires a signal handle");
    return ncclInvalidArgument;
  }
  if (hasSignal && signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("RMA/Socket : unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }
  struct ncclRmaSocketProxyCtx* contexts = (struct ncclRmaSocketProxyCtx*)rmaCtx;
  if (context < 0 || context >= contexts[0].nContexts) {
    WARN("RMA/Socket : invalid put context=%d nContexts=%d", context, contexts[0].nContexts);
    return ncclInvalidArgument;
  }
  struct ncclRmaSocketProxyCtx* ctx = &contexts[context];
  struct ncclRmaSocketProxyCollComm* comm = ctx->collComm;
  if (rank >= (uint32_t)comm->nranks) {
    WARN("RMA/Socket : invalid put rank=%u nranks=%d", rank, comm->nranks);
    return ncclInvalidArgument;
  }
  if (size > INT_MAX) {
    WARN("RMA/Socket : put size %zu exceeds socket API limit", size);
    return ncclInvalidArgument;
  }

  struct ncclRmaSocketProxyMrHandle* srcHandle = (struct ncclRmaSocketProxyMrHandle*)srcMhandle;
  struct ncclRmaSocketProxyMrHandle* dstHandle = (struct ncclRmaSocketProxyMrHandle*)dstMhandle;
  struct ncclRmaSocketProxyMrHandle* signalHandle = (struct ncclRmaSocketProxyMrHandle*)signalMhandle;

  memset(args, 0, sizeof(*args));
  args->ctx = ctx;
  args->srcType = NCCL_PTR_HOST;  // Default for signal-only requests

  if (size > 0) {
    if (!ncclRmaSocketProxySupportedPtrType(srcHandle->type) || !ncclRmaSocketProxySupportedPtrType(dstHandle->type)) {
      WARN("RMA/Socket : unsupported put pointer types src=%d dst=%d", srcHandle->type, dstHandle->type);
      return ncclInvalidArgument;
    }
    NCCLCHECK(ncclRmaSocketProxyValidateRange(srcHandle->size, srcOff, size, "source MR"));
    NCCLCHECK(ncclRmaSocketProxyValidateRange(dstHandle->size, dstOff, size, "destination MR"));
    args->dstMrId = dstHandle->mrId;
    args->dstOff = dstOff;
    args->srcPtr = (void*)((uintptr_t)srcHandle->data + srcOff);
    args->srcHandle = srcHandle;
    args->srcType = srcHandle->type;
  }
  if (hasSignal) {
    if (signalHandle->type != NCCL_PTR_CUDA) {
      WARN("RMA/Socket : host VA signals are not supported");
      return ncclInvalidUsage;
    }
    if ((signalOff % sizeof(uint64_t)) != 0) {
      WARN("RMA/Socket : signal offset %lu is not %zu-byte aligned", signalOff, sizeof(uint64_t));
      return ncclInvalidArgument;
    }
    NCCLCHECK(ncclRmaSocketProxyValidateRange(signalHandle->size, signalOff, sizeof(uint64_t), "signal MR"));
    args->opFlags |= NCCL_RMA_SOCKET_OP_HAS_SIGNAL;
    args->signalMrId = signalHandle->mrId;
  }
  return ncclSuccess;
}

static ncclResult_t ncclRmaSocketProxySubmitPut(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle,
                                                size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank,
                                                bool hasSignal, uint64_t signalOff, void* signalMhandle,
                                                uint64_t signalValue, uint32_t signalOp, bool isStrongSignal,
                                                void** request) {
  /* One proxy CPU thread serializes payload visibility and the signal update. */
  (void)isStrongSignal;

  if (rmaCtx == NULL || request == NULL) {
    WARN("RMA/Socket : invalid put arguments rmaCtx=%p request=%p", rmaCtx, request);
    return ncclInvalidArgument;
  }
  *request = NULL;

  struct ncclRmaSocketProxyPutArgs args;
  NCCLCHECK(ncclRmaSocketProxyValidatePutArgs(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank,
                                              hasSignal, signalOff, signalMhandle, signalOp, &args));
  struct ncclRmaSocketProxyCtx* ctx = args.ctx;
  struct ncclRmaSocketProxyCollComm* comm = ctx->collComm;

  ncclResult_t ret = ncclSuccess;
  struct ncclRmaSocketProxyRequest* req = NULL;
  struct ncclRmaSocketProxyPeerSender* sender = &comm->peerSender[rank];
  struct ncclRmaSocketProxySendTask* task = NULL;
  NCCLCHECKGOTO(ncclRmaSocketProxyGetRequest(ctx, rank, ncclRmaSocketProxyRequestTypePut, &req), ret, fail);

  if (sender->tasks == NULL || sender->capacity <= 0 || sender->count < 0 || sender->count > sender->capacity) {
    WARN("RMA/Socket : invalid send queue state count=%d capacity=%d", sender->count, sender->capacity);
    ret = ncclInternalError;
    goto fail;
  }
  if (sender->count == sender->capacity) {
    WARN("RMA/Socket : send queue is full with %d tasks", sender->count);
    ret = ncclInternalError;
    goto fail;
  }

  task = &sender->tasks[sender->tail];
  memset(task, 0, sizeof(*task));
  task->srcPtr = args.srcPtr;
  task->srcHandle = args.srcHandle;
  task->size = size;
  task->srcType = args.srcType;
  NCCLCHECKGOTO(ncclRmaSocketProxyBuildHeader(&task->controlMsg, ncclRmaSocketProxyMsgTypeIput, ncclSuccess,
                                              req->globalRequestId, args.opFlags, args.dstMrId, args.dstOff, size,
                                              args.signalMrId, signalOff, signalValue, hasSignal ? signalOp : 0),
                ret, fail);
  sender->tail = (sender->tail + 1) % sender->capacity;
  sender->count++;

  *request = req;
  return ncclSuccess;
fail:
  ncclRmaSocketProxyReleaseRequest(req);
  return ret;
}

ncclResult_t ncclRmaSocketProxyIPut(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                    uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags,
                                    void** request) {
  (void)optFlags;
  return ncclRmaSocketProxySubmitPut(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, false, 0,
                                     NULL, 0, 0, false, request);
}

ncclResult_t ncclRmaSocketProxyIPutSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                          uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                          void* signalMhandle, uint64_t signalValue, uint32_t signalOp,
                                          bool isStrongSignal, uint32_t optFlags, void** request) {
  (void)optFlags;
  return ncclRmaSocketProxySubmitPut(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, true,
                                     signalOff, signalMhandle, signalValue, signalOp, isStrongSignal, request);
}

/* GET and flush are deferred; a NULL request represents an immediately completed no-op. */
ncclResult_t ncclRmaSocketProxyIGet(void*, int, uint64_t, void*, size_t, uint64_t, void*, uint32_t, uint32_t,
                                    void** request) {
  if (request == NULL) return ncclInvalidArgument;
  *request = NULL;
  return ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyIFlush(void*, int, void*, uint32_t, void** request) {
  if (request == NULL) return ncclInvalidArgument;
  *request = NULL;
  return ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyTest(void* collComm, void* request, int* done) {
  ncclResult_t ret = ncclSuccess;
  if (done == NULL) {
    WARN("RMA/Socket : test called with NULL done pointer");
    return ncclInvalidArgument;
  }
  *done = 0;
  struct ncclRmaSocketProxyRequest* req = (struct ncclRmaSocketProxyRequest*)request;
  if (req == NULL) {
    WARN("RMA/Socket : test called with NULL request");
    return ncclInternalError;
  }
  struct ncclRmaSocketProxyCollComm* comm = (struct ncclRmaSocketProxyCollComm*)collComm;
  if (comm == NULL || req->type == ncclRmaSocketProxyRequestTypeFree || req->ctx == NULL ||
      req->ctx->collComm != comm) {
    WARN("RMA/Socket : test called with an invalid request or a request from a different collComm");
    return ncclInvalidArgument;
  }

  NCCLCHECKGOTO(ncclRmaSocketProxyProgressComm(comm), ret, fail);

  if (req->type != ncclRmaSocketProxyRequestTypePut) {
    WARN("RMA/Socket : test called with invalid request type=%d", req->type);
    ret = ncclInternalError;
    goto fail;
  }

  if (nccl::utility::rollingLessEq<uint64_t>(req->globalRequestId,
                                             comm->peerSender[req->peer].lastCompletedRequestGlobalId)) {
    *done = 1;
    ncclRmaSocketProxyReleaseRequest(req);
  }
  return ncclSuccess;
fail:
  ncclRmaSocketProxyReleaseRequest(req);
  return ret;
}

ncclResult_t ncclRmaSocketProxyProgress(void* rmaCtx) {
  if (rmaCtx == NULL) return ncclSuccess;
  struct ncclRmaSocketProxyCtx* ctx = (struct ncclRmaSocketProxyCtx*)rmaCtx;
  return ncclRmaSocketProxyProgressComm(ctx[0].collComm);
}

ncclResult_t ncclRmaSocketProxyQueryLastError(void* rmaCtx, bool* hasError) {
  if (hasError == NULL) {
    WARN("RMA/Socket : queryLastError called with NULL hasError pointer");
    return ncclInvalidArgument;
  }
  *hasError = false;
  if (rmaCtx == NULL) return ncclSuccess;
  struct ncclRmaSocketProxyCtx* ctx = (struct ncclRmaSocketProxyCtx*)rmaCtx;
  struct ncclRmaSocketProxyCollComm* comm = ctx[0].collComm;
  *hasError = (comm != NULL && comm->lastError != ncclSuccess);
  return ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyFinalize(void* ctx) {
  (void)ctx;
  std::lock_guard<std::mutex> lock(ncclRmaSocketProxyMutex);
  if (ncclRmaSocketProxyRefCount == 0) {
    WARN("RMA/Socket : finalize called with zero reference count");
    return ncclInternalError;
  }
  ncclRmaSocketProxyRefCount--;
  return ncclSuccess;
}

/* rma_v14 callback table exported to the built-in plugin registry. */

ncclRma_t ncclRmaSocketProxy = {
  "RMA_SOCKET_PROXY",
  ncclRmaSocketProxyInit,               // rma_socket.cc
  ncclRmaSocketProxyDevices,            // rma_socket.cc
  ncclRmaSocketProxyGetProperties,      // rma_socket.cc
  ncclRmaSocketProxyListen,             // rma_socket.cc
  ncclRmaSocketProxyConnect,            // rma_socket_setup.cc
  ncclRmaSocketProxyCreateContext,      // rma_socket_setup.cc
  ncclRmaSocketProxyRegMrSym,           // rma_socket_mem.cc
  ncclRmaSocketProxyRegMrSymDmaBuf,     // rma_socket_mem.cc
  ncclRmaSocketProxyDeregMrSym,         // rma_socket_mem.cc
  ncclRmaSocketProxyDestroyContext,     // rma_socket_setup.cc
  ncclRmaSocketProxyCloseColl,          // rma_socket_setup.cc
  ncclRmaSocketProxyCloseListen,        // rma_socket.cc
  ncclRmaSocketProxyIPut,               // rma_socket.cc
  ncclRmaSocketProxyIPutSignal,         // rma_socket.cc
  ncclRmaSocketProxyIGet,               // no-op
  ncclRmaSocketProxyIFlush,             // no-op
  ncclRmaSocketProxyTest,               // rma_socket.cc
  ncclRmaSocketProxyProgress,           // rma_socket.cc
  ncclRmaSocketProxyQueryLastError,     // rma_socket.cc
  ncclRmaSocketProxyFinalize            // rma_socket.cc
};
