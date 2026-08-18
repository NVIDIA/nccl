/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "rma_socket.h"

#include <limits.h>

static ncclResult_t ncclRmaSocketProxyFreeCtx(struct ncclRmaSocketProxyCtx* ctx) {
  if (ctx) free(ctx->requests);
  return ncclSuccess;
}

static ncclResult_t ncclRmaSocketProxyFreeCollComm(struct ncclRmaSocketProxyCollComm* comm) {
  if (comm == NULL) return ncclSuccess;
  if (comm->mrHead != NULL) {
    WARN("RMA/Socket : freeing collComm with registered memory handles");
    return ncclInvalidUsage;
  }

  ncclResult_t ret = ncclSuccess;
  for (int peer = 0; peer < comm->nranks; peer++) {
    if (comm->peerSender) {
      struct ncclRmaSocketProxyPeerSender* sender = &comm->peerSender[peer];
      NCCLCHECKIGNORE(ncclSocketClose(&sender->sock), ret);
      NCCLCHECKIGNORE(ncclRmaSocketProxyFreeChunkBuffer(sender->buffer), ret);
      free(sender->tasks);
    }
    if (comm->peerReceiver) {
      struct ncclRmaSocketProxyPeerReceiver* receiver = &comm->peerReceiver[peer];
      NCCLCHECKIGNORE(ncclSocketClose(&receiver->sock), ret);
      NCCLCHECKIGNORE(ncclRmaSocketProxyFreeChunkBuffer(receiver->buffer), ret);
    }
  }
  free(comm->peerSender);
  free(comm->peerReceiver);

  free(comm);
  return ret;
}

static ncclResult_t ncclRmaSocketProxySetSendQueueCapacity(struct ncclRmaSocketProxyCollComm* comm, int nContexts) {
  const int tasksPerContext = 2 * NCCL_RMA_SOCKET_MAX_REQUESTS_PER_PEER;
  if (comm == NULL || nContexts < 0 || nContexts > INT_MAX / tasksPerContext) {
    WARN("RMA/Socket : invalid send queue context count %d", nContexts);
    return ncclInvalidArgument;
  }

  int requiredCapacity = tasksPerContext * nContexts;
  for (int peer = 0; peer < comm->nranks; peer++) {
    struct ncclRmaSocketProxyPeerSender* sender = &comm->peerSender[peer];
    if (requiredCapacity > sender->capacity) {
      if (sender->count < 0 || sender->count > sender->capacity) {
        WARN("RMA/Socket : invalid send queue count=%d allocated capacity=%d", sender->count, sender->capacity);
        return ncclInternalError;
      }

      struct ncclRmaSocketProxySendTask* tasks = NULL;
      NCCLCHECK(ncclCalloc(&tasks, requiredCapacity));
      for (int i = 0; i < sender->count; i++) {
        int oldIndex = (sender->head + i) % sender->capacity;
        tasks[i] = sender->tasks[oldIndex];
      }
      free(sender->tasks);
      sender->tasks = tasks;
      sender->capacity = requiredCapacity;
      sender->head = 0;
      sender->tail = sender->count;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclRmaSocketProxyConnectPeers(struct ncclRmaSocketProxyCollComm* comm,
                                                   struct ncclSocket* listenSock, void* handles[]) {
  int nranks = comm->nranks;
  int* accepted = NULL;
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket acceptedSock;
  bool acceptedSockInitialized = false;

  NCCLCHECKGOTO(ncclCalloc(&comm->peerSender, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->peerReceiver, nranks), ret, fail);
  for (int peer = 0; peer < nranks; peer++) {
    comm->peerSender[peer].lastCompletedRequestGlobalId = UINT64_MAX;
    comm->peerReceiver[peer].lastCompletedRequestGlobalId = UINT64_MAX;
    NCCLCHECKGOTO(ncclRmaSocketProxyAllocChunkBuffer(&comm->peerSender[peer].buffer, NCCL_RMA_SOCKET_CHUNK_SIZE), ret,
                  fail);
    NCCLCHECKGOTO(ncclRmaSocketProxyAllocChunkBuffer(&comm->peerReceiver[peer].buffer, NCCL_RMA_SOCKET_CHUNK_SIZE), ret,
                  fail);
  }
  NCCLCHECKGOTO(ncclCalloc(&accepted, nranks), ret, fail);

  /* Create one outgoing socket to every peer. */
  for (int peer = 0; peer < nranks; peer++) {
    if (handles[peer] == NULL) {
      WARN("RMA/Socket : connect missing handle for peer %d", peer);
      ret = ncclInvalidArgument;
      goto fail;
    }
    struct ncclRmaSocketProxyHandle* handle = (struct ncclRmaSocketProxyHandle*)handles[peer];
    struct ncclRmaSocketProxyPeerSender* sender = &comm->peerSender[peer];
    NCCLCHECKGOTO(ncclSocketInit(&sender->sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetSocket, NULL, 0),
                  ret, fail);
    NCCLCHECKGOTO(ncclSocketConnect(&sender->sock), ret, fail);
    NCCLCHECKGOTO(ncclSocketSend(&sender->sock, &comm->rank, sizeof(comm->rank)), ret, fail);
  }

  /* Accept one incoming socket from every peer. */
  for (int i = 0; i < nranks; i++) {
    int peer = -1;
    int ready = 0;
    NCCLCHECKGOTO(ncclSocketInit(&acceptedSock), ret, fail);
    acceptedSockInitialized = true;
    NCCLCHECKGOTO(ncclSocketAccept(&acceptedSock, listenSock), ret, fail);
    do {
      NCCLCHECKGOTO(ncclSocketReady(&acceptedSock, &ready), ret, fail);
    } while (!ready);
    NCCLCHECKGOTO(ncclSocketRecv(&acceptedSock, &peer, sizeof(peer)), ret, fail);
    if (peer < 0 || peer >= nranks || accepted[peer]) {
      WARN("RMA/Socket : accepted invalid peer rank %d local rank %d nranks %d", peer, comm->rank, nranks);
      ret = ncclInternalError;
      goto fail;
    }
    comm->peerReceiver[peer].sock = acceptedSock;
    acceptedSockInitialized = false;
    accepted[peer] = 1;
  }

  free(accepted);
  return ret;
fail:
  if (acceptedSockInitialized) NCCLCHECKIGNORE(ncclSocketClose(&acceptedSock), ret);
  free(accepted);
  return ret;
}

ncclResult_t ncclRmaSocketProxyConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm,
                                       void** collComm) {
  if (handles == NULL || listenComm == NULL || collComm == NULL || rank < 0 || rank >= nranks) {
    WARN("RMA/Socket : invalid connect arguments handles=%p listenComm=%p collComm=%p rank=%d nranks=%d", handles,
         listenComm, collComm, rank, nranks);
    return ncclInvalidArgument;
  }

  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = (struct ncclSocket*)listenComm;
  struct ncclRmaSocketProxyCollComm* comm = NULL;
  *collComm = NULL;

  (void)ctx;
  NCCLCHECKGOTO(ncclCalloc(&comm, 1), ret, fail);
  comm->rank = rank;
  comm->nranks = nranks;
  comm->nextGlobalRequestId = 0;
  comm->lastError = ncclSuccess;
  NCCLCHECKGOTO(ncclRmaSocketProxyConnectPeers(comm, listenSock, handles), ret, fail);

  *collComm = comm;
  return ncclSuccess;
fail:
  NCCLCHECKIGNORE(ncclRmaSocketProxyFreeCollComm(comm), ret);
  return ret;
}

ncclResult_t ncclRmaSocketProxyCreateContext(void* collComm, ncclRmaConfig_t* config, void** rmaCtx) {
  if (collComm == NULL || config == NULL || rmaCtx == NULL) {
    WARN("RMA/Socket : invalid createContext arguments collComm=%p config=%p rmaCtx=%p", collComm, config, rmaCtx);
    return ncclInvalidArgument;
  }
  struct ncclRmaSocketProxyCollComm* comm = (struct ncclRmaSocketProxyCollComm*)collComm;
  if (config->nContexts <= 0) {
    WARN("RMA/Socket : invalid context count %d", config->nContexts);
    return ncclInvalidArgument;
  }
  if (config->rankStride <= 0 || (comm->nranks % config->rankStride) != 0) {
    WARN("RMA/Socket : invalid rank stride %d for nranks %d", config->rankStride, comm->nranks);
    return ncclInvalidArgument;
  }
  if (config->nContexts > INT_MAX - comm->nContexts) {
    WARN("RMA/Socket : context count overflow current=%d additional=%d", comm->nContexts, config->nContexts);
    return ncclInvalidArgument;
  }

  struct ncclRmaSocketProxyCtx* ctx = NULL;
  ncclResult_t ret = ncclSuccess;
  int newContextCount = comm->nContexts + config->nContexts;

  NCCLCHECKGOTO(ncclCalloc(&ctx, config->nContexts), ret, fail);

  /* Initialize each context object. */
  for (int c = 0; c < config->nContexts; c++) {
    ctx[c].collComm = comm;
    ctx[c].nContexts = config->nContexts;
    ctx[c].requestCount = NCCL_RMA_SOCKET_MAX_REQUESTS_PER_PEER * comm->nranks;
    NCCLCHECKGOTO(ncclCalloc(&ctx[c].requests, ctx[c].requestCount), ret, fail);
  }
  NCCLCHECKGOTO(ncclRmaSocketProxySetSendQueueCapacity(comm, newContextCount), ret, fail);

  comm->nContexts = newContextCount;
  *rmaCtx = ctx;
  return ncclSuccess;
fail:
  if (ctx) {
    for (int c = 0; c < config->nContexts; c++) NCCLCHECKIGNORE(ncclRmaSocketProxyFreeCtx(&ctx[c]), ret);
    free(ctx);
  }
  return ret;
}

ncclResult_t ncclRmaSocketProxyDestroyContext(void* rmaCtx) {
  struct ncclRmaSocketProxyCtx* ctx = (struct ncclRmaSocketProxyCtx*)rmaCtx;
  if (ctx) {
    int nContexts = ctx[0].nContexts;
    struct ncclRmaSocketProxyCollComm* comm = ctx[0].collComm;
    if (comm == NULL || nContexts <= 0 || nContexts > comm->nContexts) {
      WARN("RMA/Socket : destroyContext called with invalid context count=%d live contexts=%d", nContexts,
           comm ? comm->nContexts : 0);
      return ncclInvalidUsage;
    }

    /* Requests retain their owning context until test() releases them. */
    for (int c = 0; c < nContexts; c++) {
      if (ctx[c].activeOps != 0) {
        WARN("RMA/Socket : destroyContext called with %d active operations on context index=%d", ctx[c].activeOps, c);
        return ncclInvalidUsage;
      }
    }

    comm->nContexts -= nContexts;

    /* No context-owned request remains, so release the context storage. */
    for (int c = 0; c < nContexts; c++) NCCLCHECK(ncclRmaSocketProxyFreeCtx(&ctx[c]));
    free(ctx);
  }
  return ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyCloseColl(void* collComm) {
  struct ncclRmaSocketProxyCollComm* comm = (struct ncclRmaSocketProxyCollComm*)collComm;
  if (comm == NULL) return ncclSuccess;

  if (comm->nContexts != 0) {
    WARN("RMA/Socket : closeColl called with %d live contexts", comm->nContexts);
    return ncclInvalidUsage;
  }
  if (comm->mrHead != NULL) {
    WARN("RMA/Socket : closeColl called with registered memory handles");
    return ncclInvalidUsage;
  }
  for (int peer = 0; peer < comm->nranks; peer++) {
    if (comm->peerSender && comm->peerSender[peer].count != 0) {
      WARN("RMA/Socket : closeColl called with %d queued sends for peer=%d", comm->peerSender[peer].count, peer);
      return ncclInvalidUsage;
    }
    if (comm->peerReceiver) {
      struct ncclRmaSocketProxyPeerReceiver* receiver = &comm->peerReceiver[peer];
      if (receiver->recvState == ncclRmaSocketProxyRecvStatePayload || receiver->controlMsgOffset != 0) {
        WARN("RMA/Socket : closeColl called with an incomplete receive from peer=%d state=%d controlOffset=%d", peer,
             receiver->recvState, receiver->controlMsgOffset);
        return ncclInvalidUsage;
      }
    }
  }
  return ncclRmaSocketProxyFreeCollComm(comm);
}
