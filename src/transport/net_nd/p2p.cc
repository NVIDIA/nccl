/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include <cuda_runtime_api.h>
#include <limits.h>

enum ncclNdCompletionOp {
  ncclNdCompletionOpCts = 1,
  ncclNdCompletionOpData = 2,
  ncclNdCompletionOpRecord = 3,
};

// Completion contexts are opaque to the provider. Encode the request-pool
// slot, request generation, replay attempt, and operation instead of passing a
// request pointer directly. A flushed QP can report more than one error for a
// send (the silent data write and its ordered completion-record write), and
// those CQEs can arrive after the request was replayed or its pool slot reused.
// The generation and attempt make those stale CQEs safe to ignore.
static constexpr unsigned kNdContextOpBits = 2;
static constexpr unsigned kNdContextSlotBits = 8;
static constexpr unsigned kNdContextAttemptBits = 4;
static constexpr unsigned kNdContextGenerationBits = 24;
static constexpr int kNdMaxReplayAttempts = 1;
static constexpr unsigned kNdContextSlotShift = kNdContextOpBits;
static constexpr unsigned kNdContextAttemptShift = kNdContextSlotShift + kNdContextSlotBits;
static constexpr unsigned kNdContextGenerationShift = kNdContextAttemptShift + kNdContextAttemptBits;
static constexpr unsigned kNdContextMagicShift = kNdContextGenerationShift + kNdContextGenerationBits;
static constexpr ULONG_PTR kNdContextMagic = 0xa5;
static constexpr ULONG_PTR kNdContextGenerationMask = (1ULL << kNdContextGenerationBits) - 1;
static_assert(NET_ND_MAX_REQUESTS <= (1U << kNdContextSlotBits), "ND completion context needs more request-slot bits");
static_assert(kNdMaxReplayAttempts < (1U << kNdContextAttemptBits),
              "ND completion context needs more replay-attempt bits");

struct ncclNdDecodedCompletionContext {
  int slot;
  int attempt;
  int op;
  ULONG_PTR generation;
};

// Encode request identity and operation into an opaque completion context.
static void* ncclNdMakeCompletionContext(struct ncclNdRequest* req, enum ncclNdCompletionOp op) {
  ULONG_PTR slot = (ULONG_PTR)(req - req->base->reqs);
  ULONG_PTR generation = (ULONG_PTR)(req->id + 1) & kNdContextGenerationMask;
  ULONG_PTR encoded = (ULONG_PTR)op | (slot << kNdContextSlotShift) |
                      ((ULONG_PTR)req->replayCount << kNdContextAttemptShift) |
                      (generation << kNdContextGenerationShift) | (kNdContextMagic << kNdContextMagicShift);
  return (void*)encoded;
}

// Decode and validate an opaque provider completion context.
static bool ncclNdDecodeCompletionContext(void* context, struct ncclNdDecodedCompletionContext* decoded) {
  ULONG_PTR encoded = (ULONG_PTR)context;
  ULONG_PTR magic = encoded >> kNdContextMagicShift;
  if (magic != kNdContextMagic) return false;
  decoded->op = (int)(encoded & ((1U << kNdContextOpBits) - 1));
  decoded->slot = (int)((encoded >> kNdContextSlotShift) & ((1U << kNdContextSlotBits) - 1));
  decoded->attempt = (int)((encoded >> kNdContextAttemptShift) & ((1U << kNdContextAttemptBits) - 1));
  decoded->generation = (encoded >> kNdContextGenerationShift) & kNdContextGenerationMask;
  return decoded->op >= ncclNdCompletionOpCts && decoded->op <= ncclNdCompletionOpRecord && decoded->slot >= 0 &&
         decoded->slot < NET_ND_MAX_REQUESTS;
}

// Classify provider failures that are safe to retry on another rail.
static bool ncclNdCompletionCanFailover(HRESULT status) {
  // ND_CANCELED is produced for requests flushed after a QP failure, while
  // ND_IO_TIMEOUT reports a CQ request timeout caused by a connection or
  // remote-QP failure. Keep ND_TIMEOUT for providers that report an immediate
  // operation timeout. Local length, access, formatting, and remote-access
  // errors are programming/protocol failures and must remain fatal instead of
  // being retried on another rail.
  return status == ND_CANCELED || status == ND_IO_TIMEOUT || status == ND_TIMEOUT;
}

// Return the CTS group associated with one request-ring slot.
static inline struct ncclNdSendFifo* ncclNdGetCtsSlot(struct ncclNdNetCommBase* base, int slot) {
  return base->localCtsFifo + (slot * NCCL_NET_ND_MAX_RECVS);
}

// Derive a nonzero completion generation from a request ID.
static inline uint32_t ncclNdCompletionGeneration(uint64_t id) {
  uint32_t generation = (uint32_t)(id + 1);
  // Zero means "not completed", including for a zero-byte transfer.
  return generation == 0 ? 1 : generation;
}

// Pack a request generation and transfer size into one completion record.
static inline uint64_t ncclNdCompletionValue(uint64_t id, size_t size) {
  return ((uint64_t)ncclNdCompletionGeneration(id) << 32) | (uint32_t)size;
}

// Check whether the peer acknowledged a completed send generation.
static bool ncclNdSendWasAcknowledged(struct ncclNdRequest* req) {
  if (req->type != NCCL_NET_ND_REQ_SEND) return false;
  volatile uint64_t* acknowledgement =
    &req->base->remCompletionRecords.acknowledgements[req->send.slot][req->send.index];
  uint64_t value = *acknowledgement;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  return value == ncclNdCompletionValue(req->id, (size_t)req->send.size);
}

// Select inline posting when the payload fits the QP limit.
static inline ULONG ncclNdInlineWriteFlag(struct ncclNdNetCommBase* base, struct ncclNdQp* qp, ULONG size) {
  if (size == 0 || qp->inlineDataSize == 0 || size > qp->inlineDataSize) return 0;
  if (base->stats != NULL) base->stats->inlineWrites.fetch_add(1, std::memory_order_relaxed);
  return ND_OP_FLAG_INLINE;
}

// Prefer the least-loaded QP while rotating the starting point so equally
// loaded QPs and adapters continue to receive traffic fairly.
static int ncclNdSelectQp(struct ncclNdNetCommBase* base) {
  if (base->nqps <= 0) return -1;
  int start = base->qpIndex;
  int selected = -1;
  LONG64 leastBytes = LLONG_MAX;
  LONG64 leastOutstanding = LLONG_MAX;
  LONG failedMask = InterlockedCompareExchange(&base->failedDevicesMask, 0, 0);
  for (int offset = 0; offset < base->nqps; offset++) {
    int candidate = (start + offset) % base->nqps;
    struct ncclNdQp* qp = &base->qps[candidate];
    if (qp->qp == NULL || qp->devIndex < 0 || qp->devIndex >= base->ndevs || (failedMask & (1L << qp->devIndex)) != 0)
      continue;
    LONG64 bytes = InterlockedCompareExchange64(&qp->outstandingBytes, 0, 0);
    LONG64 outstanding = InterlockedCompareExchange64(&qp->outstanding, 0, 0);
    if (bytes < leastBytes || (bytes == leastBytes && outstanding < leastOutstanding)) {
      selected = candidate;
      leastBytes = bytes;
      leastOutstanding = outstanding;
    }
  }
  if (selected < 0) return -1;
  base->qpIndex = (selected + 1) % base->nqps;
  return selected;
}

// Compute an overflow-safe deadline for data-path progress.
static ULONGLONG ncclNdDataDeadline(void) {
  int64_t timeoutSeconds = NCCL_ND_DATA_TIMEOUT_SECONDS;
  ULONGLONG now = GetTickCount64();
  ULONGLONG maxSeconds = (ULLONG_MAX - now) / 1000;
  return now + (ULONGLONG)(timeoutSeconds > (int64_t)maxSeconds ? maxSeconds : timeoutSeconds) * 1000;
}

enum ncclNdDataFailureKind {
  ncclNdDataFailureOther,
  ncclNdDataFailureTimeout,
  ncclNdDataFailureDisconnect,
};

// Publish the first fatal data-path error and flush every QP.
static ncclResult_t ncclNdFailDataPath(struct ncclNdNetCommBase* base, ncclResult_t error, const char* reason,
                                       enum ncclNdDataFailureKind kind = ncclNdDataFailureOther) {
  bool first = false;
  ncclResult_t sticky = ncclNdSetFatalError(base, error, &first);
  if (first) {
    if (base->stats != NULL) {
      struct ncclNdStats* stats = base->stats;
      if (kind == ncclNdDataFailureTimeout) {
        stats->dataTimeouts.fetch_add(1, std::memory_order_relaxed);
      } else if (kind == ncclNdDataFailureDisconnect) {
        stats->peerDisconnects.fetch_add(1, std::memory_order_relaxed);
      }
    }
    WARN("NET/ND : Comm %llu dev %d data path failed: %s", (unsigned long long)base->commId, base->primaryDev, reason);
    for (int q = 0; q < base->nqps; q++) {
      if (base->qps[q].qp != NULL) (void)base->qps[q].qp->Flush();
    }
  }
  return sticky;
}

// Retire one failed rail or fail the communicator when failover is unavailable.
ncclResult_t ncclNdHandleRailFailure(struct ncclNdNetCommBase* base, int devIndex, const char* reason) {
  if (base == NULL || devIndex < 0 || devIndex >= base->ndevs) return ncclInvalidArgument;
  LONG bit = 1L << devIndex;
  LONG failedMask = InterlockedCompareExchange(&base->failedDevicesMask, 0, 0);
  if ((failedMask & bit) != 0) return ncclSuccess;
  if (ncclParamNdResiliencyPortFailover() == 0 || base->ndevs < 2 || failedMask != 0) {
    if (base->stats != NULL) base->stats->failoverFailures.fetch_add(1, std::memory_order_relaxed);
    return ncclNdFailDataPath(base, ncclSystemError, reason ? reason : "NetworkDirect rail failed");
  }
  LONG previous = InterlockedCompareExchange(&base->failedDevicesMask, failedMask | bit, failedMask);
  if (previous != failedMask) return ncclNdHandleRailFailure(base, devIndex, reason);
  if (base->stats != NULL) {
    base->stats->railFailures.fetch_add(1, std::memory_order_relaxed);
    base->stats->railFailovers.fetch_add(1, std::memory_order_relaxed);
  }
  WARN("NET/ND : Comm %llu retiring rail %d after %s; continuing on remaining adapters",
       (unsigned long long)base->commId, devIndex, reason ? reason : "transport failure");
  for (int q = 0; q < base->nqps; q++) {
    if (base->qps[q].devIndex == devIndex && base->qps[q].qp != NULL) (void)base->qps[q].qp->Flush();
  }
  return ncclSuccess;
}

// Poll link generations, provider errors, and adapter health.
static ncclResult_t ncclNdCheckProvider(struct ncclNdNetCommBase* base, bool force = false) {
  const ULONGLONG providerCheckIntervalMs = NCCL_ND_PROVIDER_CHECK_INTERVAL_MS;
  const ULONGLONG linkCheckIntervalMs = NCCL_ND_LINK_CHECK_INTERVAL_MS;
  ULONGLONG now = GetTickCount64();

  // Consume link-event generations immediately, independent of polling cadence.
  for (int i = 0; i < base->vProps.ndevs; i++) {
    struct ncclNdDev* dev = &ncclNdDevs[base->vProps.devs[i]];
    LONG64 generation = InterlockedCompareExchange64(&dev->linkFailureGeneration, 0, 0);
    if (generation != base->linkFailureGenerations[i]) {
      base->linkFailureGenerations[i] = generation;
      if (base->stats != NULL) base->stats->adapterHealthFailures.fetch_add(1, std::memory_order_relaxed);
      char reason[128];
      snprintf(reason, sizeof(reason), "NetworkDirect adapter %d reported an interface link event", dev->device);
      NCCLCHECK(ncclNdHandleRailFailure(base, i, reason));
    }
  }
  // Poll CQ error notifications at the configured provider cadence.
  ULONGLONG nextProvider = (ULONGLONG)InterlockedCompareExchange64(&base->nextProviderCheckMs, 0, 0);
  if (!force && now < nextProvider) return ncclSuccess;
  InterlockedExchange64(&base->nextProviderCheckMs, (LONG64)(now + providerCheckIntervalMs));

  for (int i = 0; i < base->ndevs; i++) {
    struct ncclNdNetCommDevBase* devBase = ncclNdGetNetCommDevBase(base, i);
    if (!devBase->cqErrorArmed || devBase->cqErrorOv.hEvent == NULL) continue;
    DWORD waitResult = WaitForSingleObject(devBase->cqErrorOv.hEvent, 0);
    if (waitResult == WAIT_TIMEOUT) continue;
    if (waitResult == WAIT_FAILED) {
      if (base->stats != NULL) base->stats->providerNotifications.fetch_add(1, std::memory_order_relaxed);
      return ncclNdFailDataPath(base, ncclSystemError, "could not query provider error notification");
    }
    int done = 0;
    HRESULT status = ND_PENDING;
    ncclResult_t queryResult = wrap_nd_cq_get_status(devBase->cq, &devBase->cqErrorOv, &done, &status);
    if (queryResult != ncclSuccess || done) {
      if (base->stats != NULL) {
        base->stats->providerNotifications.fetch_add(1, std::memory_order_relaxed);
        base->stats->cqErrors.fetch_add(1, std::memory_order_relaxed);
      }
      char reason[128];
      snprintf(reason, sizeof(reason), "provider reported a completion-queue error on adapter %d (status 0x%08x)",
               devBase->ndDevN, (unsigned int)status);
      return ncclNdFailDataPath(base, ncclSystemError, reason);
    }
  }

  // Revalidate provider objects and Windows link state at the slower link cadence.
  ULONGLONG nextLink = (ULONGLONG)InterlockedCompareExchange64(&base->nextLinkCheckMs, 0, 0);
  if (!force && now < nextLink) return ncclSuccess;
  InterlockedExchange64(&base->nextLinkCheckMs, (LONG64)(now + linkCheckIntervalMs));
  for (int i = 0; i < base->vProps.ndevs; i++) {
    int ndDevN = base->vProps.devs[i];
    bool healthy = false;
    ncclResult_t healthResult = ncclNdCheckAdapterHealth(ndDevN, &healthy);
    if (healthResult != ncclSuccess || !healthy) {
      if (base->stats != NULL) base->stats->adapterHealthFailures.fetch_add(1, std::memory_order_relaxed);
      char reason[128];
      snprintf(reason, sizeof(reason), "NetworkDirect adapter %d reset, disappeared, or lost link", ndDevN);
      NCCLCHECK(ncclNdHandleRailFailure(base, i, reason));
    }
  }
  return ncclSuccess;
}

// The TCP socket remains connected after metadata exchange. A non-consuming
// peek makes graceful close/reset visible even when no ND completion arrives.
static ncclResult_t ncclNdCheckPeer(struct ncclNdNetCommBase* base, bool passive = false) {
  if (InterlockedCompareExchange(&base->peerClosing, 0, 0) != 0) {
    return passive ?
             ncclSuccess :
             ncclNdFailDataPath(base, ncclRemoteError, "peer began orderly shutdown", ncclNdDataFailureDisconnect);
  }
  // NCCL polls test() aggressively. Keep peer-close detection responsive
  // without adding a Winsock call to every unsuccessful progress iteration.
  static constexpr ULONGLONG kPeerCheckIntervalMs = 100;
  ULONGLONG now = GetTickCount64();
  ULONGLONG nextCheck = (ULONGLONG)InterlockedCompareExchange64(&base->nextPeerCheckMs, 0, 0);
  if (now < nextCheck) return ncclSuccess;
  InterlockedExchange64(&base->nextPeerCheckMs, (LONG64)(now + kPeerCheckIntervalMs));
  if (base->stats != NULL) base->stats->healthChecks.fetch_add(1, std::memory_order_relaxed);

  struct ncclSocket* sock =
    base->isSend ? &((struct ncclNdSendComm*)base)->sock : &((struct ncclNdRecvComm*)base)->sock;
  if (sock->state != ncclSocketStateReady || sock->socketDescriptor == NCCL_INVALID_SOCKET) {
    return ncclNdFailDataPath(base, ncclRemoteError, "peer control socket is not connected",
                              ncclNdDataFailureDisconnect);
  }

  char byte = 0;
  int received = recv(sock->socketDescriptor, &byte, 1, MSG_PEEK);
  if (received == 0) {
    return ncclNdFailDataPath(base, ncclRemoteError, "peer closed the control connection", ncclNdDataFailureDisconnect);
  }
  if (received == SOCKET_ERROR) {
    int error = WSAGetLastError();
    if (error == WSAEWOULDBLOCK || error == WSAEINTR) return ncclSuccess;
    switch (error) {
    case WSAECONNABORTED:
    case WSAECONNRESET:
    case WSAENETRESET:
    case WSAENOTCONN:
    case WSAESHUTDOWN:
    case WSAETIMEDOUT:
      return ncclNdFailDataPath(base, ncclRemoteError, "peer control connection failed", ncclNdDataFailureDisconnect);
    default:
      return ncclNdFailDataPath(base, ncclSystemError, "control socket liveness check failed");
    }
  }
  int consumed = recv(sock->socketDescriptor, &byte, 1, 0);
  if (consumed == 1 && byte == NCCL_ND_CONTROL_CLOSE) {
    InterlockedExchange(&base->peerClosing, 1);
    INFO(NCCL_NET, "NET/ND : Comm %llu dev %d observed orderly peer shutdown", (unsigned long long)base->commId,
         base->primaryDev);
    return passive ?
             ncclSuccess :
             ncclNdFailDataPath(base, ncclRemoteError, "peer began orderly shutdown", ncclNdDataFailureDisconnect);
  }
  return ncclNdFailDataPath(base, ncclRemoteError, "unexpected peer control data", ncclNdDataFailureDisconnect);
}

// Send the best-effort orderly-shutdown marker to the peer.
void ncclNdNotifyPeerClosing(struct ncclNdNetCommBase* base) {
  struct ncclSocket* sock =
    base->isSend ? &((struct ncclNdSendComm*)base)->sock : &((struct ncclNdRecvComm*)base)->sock;
  if (sock->state != ncclSocketStateReady || sock->socketDescriptor == NCCL_INVALID_SOCKET) return;
  char marker = NCCL_ND_CONTROL_CLOSE;
  int sent = send(sock->socketDescriptor, &marker, sizeof(marker), 0);
  if (sent == sizeof(marker)) {
    TRACE(NCCL_NET, "NET/ND : Comm %llu dev %d sent orderly shutdown marker", (unsigned long long)base->commId,
          base->primaryDev);
  }
}

// Translate one request stage into ordered ND writes on a selected QP.
static ncclResult_t ncclNdPostRequestOnQp(struct ncclNdRequest* req, int qpIndex, HRESULT* postStatus) {
  struct ncclNdNetCommBase* base = req->base;
  if (postStatus != NULL) *postStatus = ND_SUCCESS;
  if (qpIndex < 0 || qpIndex >= base->nqps) return ncclInternalError;
  struct ncclNdQp* ndQp = &base->qps[qpIndex];
  int devIdx = ndQp->devIndex;
  int remDevIdx = ndQp->remDevIdx;
  if (ndQp->qp == NULL || devIdx < 0 || devIdx >= base->ndevs || remDevIdx < 0 || remDevIdx >= base->nRemDevs)
    return ncclInternalError;

  // Receives publish either CTS descriptors or failover acknowledgements.
  if (req->type == NCCL_NET_ND_REQ_RECV) {
    struct ncclNdRecvComm* rComm = (struct ncclNdRecvComm*)base;
    struct ncclNdRecvCommDev* devComm = &rComm->devs[devIdx];
    ND2_SGE sge = {};
    UINT64 remoteAddr = 0;
    UINT32 remoteToken = 0;
    enum ncclNdCompletionOp op = ncclNdCompletionOpCts;
    if (req->recv.stage == NCCL_ND_RECV_STAGE_CTS) {
      struct ncclNdSendFifo* cts = ncclNdGetCtsSlot(base, req->recv.slot);
      sge.Buffer = cts;
      sge.BufferLength = (ULONG)(sizeof(*cts) * req->nreqs);
      sge.MemoryRegionToken = wrap_nd_get_local_token(devComm->ctsFifoMr);
      remoteAddr = (UINT64)base->remCtsFifo + (req->recv.slot * NCCL_NET_ND_MAX_RECVS * sizeof(*cts));
      remoteToken = base->remCtsFifoTokens[remDevIdx];
    } else {
      sge.Buffer = &base->completionRecords[req->recv.slot][0];
      sge.BufferLength = (ULONG)(sizeof(uint64_t) * req->nreqs);
      sge.MemoryRegionToken = wrap_nd_get_local_token(devComm->cmplsRecordsMr);
      remoteAddr = base->remCompletionRecords.addr + offsetof(struct ncclNdRemCompletionRecords, acknowledgements) +
                   ((req->recv.slot * NCCL_NET_ND_MAX_RECVS) * sizeof(uint64_t));
      remoteToken = base->remCompletionRecords.tokens[remDevIdx];
      op = ncclNdCompletionOpRecord;
    }
    ULONG flags = ncclNdInlineWriteFlag(base, ndQp, sge.BufferLength);
    void* context = ncclNdMakeCompletionContext(req, op);
    NCCLCHECK(wrap_nd_write(ndQp->qp, context, &sge, 1, remoteAddr, remoteToken, flags, postStatus));
    req->postedBytes = sge.BufferLength;
  } else if (req->type == NCCL_NET_ND_REQ_SEND) {
    // Order the completion record after data, or publish a release record.
    struct ncclNdSendComm* sComm = (struct ncclNdSendComm*)base;
    if (req->send.stage == NCCL_ND_SEND_STAGE_DATA && req->send.size > 0) {
      ND2_SGE dataSge = {};
      dataSge.Buffer = req->send.data;
      dataSge.BufferLength = (ULONG)req->send.size;
      dataSge.MemoryRegionToken = req->send.localTokens[devIdx];
      ULONG flags = ND_OP_FLAG_SILENT_SUCCESS | ncclNdInlineWriteFlag(base, ndQp, dataSge.BufferLength);
      void* context = ncclNdMakeCompletionContext(req, ncclNdCompletionOpData);
      NCCLCHECK(wrap_nd_write(ndQp->qp, context, &dataSge, 1, req->send.remoteAddr, req->send.remoteTokens[remDevIdx],
                              flags, postStatus));
    }
    uint64_t* completion = &base->remCompletionRecords.elems[req->send.slot][req->send.index];
    *completion = ncclNdCompletionValue(req->id, (size_t)req->send.size);
    struct ncclNdSendCommDev* devComm = &sComm->devs[devIdx];
    ND2_SGE completionSge = {};
    completionSge.Buffer = completion;
    completionSge.BufferLength = sizeof(*completion);
    completionSge.MemoryRegionToken = wrap_nd_get_local_token(devComm->cmplsRecordsMr);
    UINT64 completionAddr = base->remCompletionRecords.addr;
    if (req->send.stage == NCCL_ND_SEND_STAGE_RELEASE)
      completionAddr +=
        offsetof(struct ncclNdNetCommBase, releaseRecords) - offsetof(struct ncclNdNetCommBase, completionRecords);
    completionAddr += ((req->send.slot * NCCL_NET_ND_MAX_RECVS + req->send.index) * sizeof(*completion));
    ULONG flags = ncclNdInlineWriteFlag(base, ndQp, completionSge.BufferLength);
    void* context = ncclNdMakeCompletionContext(req, ncclNdCompletionOpRecord);
    NCCLCHECK(wrap_nd_write(ndQp->qp, context, &completionSge, 1, completionAddr,
                            base->remCompletionRecords.tokens[remDevIdx], flags, postStatus));
    req->postedBytes = (req->send.stage == NCCL_ND_SEND_STAGE_DATA ? req->send.size : 0) + sizeof(*completion);
  } else {
    return ncclInternalError;
  }

  // Track the post on its QP and CQ so shared progress can dispatch completion.
  req->qpIndex = qpIndex;
  ncclNdAddEvent(req, devIdx, ncclNdGetNetCommDevBase(base, devIdx));
  InterlockedIncrement64(&ndQp->outstanding);
  InterlockedAdd64(&ndQp->outstandingBytes, req->postedBytes);
  InterlockedIncrement64(&ndQp->postedOps);
  return ncclSuccess;
}

// Post a request and retry retryable failures on a surviving rail.
static ncclResult_t ncclNdPostRequestWithFailover(struct ncclNdRequest* req) {
  while (true) {
    int qpIndex = ncclNdSelectQp(req->base);
    if (qpIndex < 0) {
      if (req->base->stats != NULL) req->base->stats->failoverFailures.fetch_add(1, std::memory_order_relaxed);
      return ncclNdFailDataPath(req->base, ncclSystemError, "no operational NetworkDirect rail remains");
    }
    HRESULT postStatus = ND_SUCCESS;
    ncclResult_t result = ncclNdPostRequestOnQp(req, qpIndex, &postStatus);
    if (result == ncclSuccess) return ncclSuccess;
    if (!ncclNdCompletionCanFailover(postStatus)) {
      char reason[160];
      snprintf(reason, sizeof(reason), "posting request on QP %d has non-retryable status 0x%08x", qpIndex,
               (unsigned int)postStatus);
      return ncclNdFailDataPath(req->base, result, reason);
    }
    if (req->replayCount >= kNdMaxReplayAttempts) {
      if (req->base->stats != NULL) req->base->stats->failoverFailures.fetch_add(1, std::memory_order_relaxed);
      return ncclNdFailDataPath(req->base, result, "NetworkDirect replay budget exhausted");
    }
    char reason[128];
    snprintf(reason, sizeof(reason), "posting request on QP %d failed", qpIndex);
    NCCLCHECK(ncclNdHandleRailFailure(req->base, req->base->qps[qpIndex].devIndex, reason));
    req->replayCount++;
    if (req->base->stats != NULL) req->base->stats->replayedRequests.fetch_add(1, std::memory_order_relaxed);
  }
}

// Publish a tagged receive group and return its asynchronous request.
ncclResult_t ncclNdIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles,
                         void** request) {
  struct ncclNdRecvComm* rComm = (struct ncclNdRecvComm*)recvComm;
  struct ncclNdNetCommBase* base = &rComm->base;
  ncclNdScopedSrwLock progressLock(&base->progressLock);
  *request = NULL;
  ncclResult_t fatalError = ncclNdGetFatalError(base);
  if (fatalError != ncclSuccess) return fatalError;
  NCCLCHECK(ncclNdCheckProvider(base));
  NCCLCHECK(ncclNdCheckPeer(base));

  // Validate the complete receive group before consuming a request slot.
  if (n <= 0 || n > NCCL_NET_ND_MAX_RECVS) {
    WARN("NET/ND : Too many receives %d (max %d)", n, NCCL_NET_ND_MAX_RECVS);
    return ncclInternalError;
  }
  for (int r = 0; r < n; r++) {
    if (data[r] == NULL || mhandles[r] == NULL || sizes[r] > INT_MAX) {
      WARN("NET/ND : Invalid receive buffer %d (data=%p mhandle=%p size=%zu)", r, data[r], mhandles[r], sizes[r]);
      return ncclInternalError;
    }
  }

  // Reserve request and FIFO-generation state before publishing CTS.
  struct ncclNdRequest* req;
  NCCLCHECK(ncclNdGetRequest(base, NCCL_NET_ND_REQ_RECV, &req));
  if (req == NULL) return ncclSuccess;
  req->recv.sizes = (int*)malloc(sizeof(int) * n);
  if (req->recv.sizes == NULL) {
    ncclNdFreeRequest(req);
    return ncclSystemError;
  }
  for (int r = 0; r < n; r++) req->recv.sizes[r] = 0;

  int slot = (int)(base->fifoHead % NET_ND_MAX_REQUESTS);
  if (rComm->recvReqs[slot] != NULL) {
    WARN("NET/ND : Receive FIFO slot %d is still in use", slot);
    free(req->recv.sizes);
    req->recv.sizes = NULL;
    ncclNdFreeRequest(req);
    return ncclNdSetFatalError(base, ncclInternalError);
  }

  req->id = base->fifoHead;
  req->nreqs = n;
  req->recv.slot = slot;
  req->recv.stage = NCCL_ND_RECV_STAGE_CTS;
  rComm->recvReqs[slot] = req;
  TRACE(NCCL_NET, "NET/ND : Irecv group id=%llu slot=%d n=%d", (unsigned long long)req->id, slot, n);

  // Encode buffer addresses, tags, and per-rail tokens into one CTS group.
  struct ncclNdSendFifo* cts = ncclNdGetCtsSlot(base, slot);
  memset(cts, 0, sizeof(*cts) * NCCL_NET_ND_MAX_RECVS);
  memset(base->completionRecords[slot], 0, sizeof(base->completionRecords[slot]));
  memset(base->releaseRecords[slot], 0, sizeof(base->releaseRecords[slot]));

  for (int r = 0; r < n; r++) {
    struct ncclNdMrHandle* mrHandle = (struct ncclNdMrHandle*)mhandles[r];
    cts[r].addr = (uint64_t)data[r];
    cts[r].size = (uint64_t)sizes[r];
    cts[r].nreqs = n;
    cts[r].tag = tags ? (uint32_t)tags[r] : 0;
    for (int devIdx = 0; devIdx < base->ndevs; devIdx++) {
      cts[r].tokens[devIdx] = wrap_nd_get_remote_token(mrHandle->mrs[devIdx]);
    }
    // Publish idx last in the local staging buffer. The single ordered RDMA
    // write below transfers the whole receive group to the sender.
    cts[r].idx = req->id + 1;
  }

  if (base->remCtsFifo == NULL || base->nqps <= 0 || base->qps[0].qp == NULL) {
    WARN("NET/ND : Cannot publish receive CTS without a connected QP");
    rComm->recvReqs[slot] = NULL;
    free(req->recv.sizes);
    req->recv.sizes = NULL;
    ncclNdFreeRequest(req);
    return ncclNdSetFatalError(base, ncclInternalError);
  }

  ncclResult_t result = ncclNdPostRequestWithFailover(req);
  if (result != ncclSuccess) {
    rComm->recvReqs[slot] = NULL;
    free(req->recv.sizes);
    req->recv.sizes = NULL;
    ncclNdFreeRequest(req);
    return ncclNdSetFatalError(base, result);
  }

  base->fifoHead++;
  *request = req;
  return ncclSuccess;
}

// Match a published receive and post its tagged send.
ncclResult_t ncclNdIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                         void** request) {
  struct ncclNdSendComm* sComm = (struct ncclNdSendComm*)sendComm;
  struct ncclNdNetCommBase* base = &sComm->base;
  ncclNdScopedSrwLock progressLock(&base->progressLock);
  *request = NULL;
  ncclResult_t fatalError = ncclNdGetFatalError(base);
  if (fatalError != ncclSuccess) return fatalError;
  NCCLCHECK(ncclNdCheckProvider(base));
  NCCLCHECK(ncclNdCheckPeer(base));

  // Wait non-blockingly for the receiver to publish a complete CTS group.
  if (sComm->ctsDeadlineMs == 0) sComm->ctsDeadlineMs = ncclNdDataDeadline();
  if (sComm->ctsDeadlineMs != 0 && GetTickCount64() >= sComm->ctsDeadlineMs) {
    return ncclNdFailDataPath(base, ncclRemoteError, "timed out waiting for receive CTS", ncclNdDataFailureTimeout);
  }

  int slot = (int)(base->fifoHead % NET_ND_MAX_REQUESTS);
  volatile struct ncclNdSendFifo* cts = ncclNdGetCtsSlot(base, slot);
  uint64_t expectedIdx = base->fifoHead + 1;
  if (cts[0].idx != expectedIdx) return ncclSuccess;
  int nreqs = (int)cts[0].nreqs;
  if (nreqs <= 0 || nreqs > NCCL_NET_ND_MAX_RECVS) {
    WARN("NET/ND : Invalid CTS receive count %d", nreqs);
    return ncclNdSetFatalError(base, ncclInternalError);
  }
  for (int r = 1; r < nreqs; r++) {
    if (cts[r].idx != expectedIdx) return ncclSuccess;
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  for (int r = 1; r < nreqs; r++) {
    if (cts[r].nreqs != (uint32_t)nreqs) {
      WARN("NET/ND : Inconsistent CTS receive count at index %d", r);
      return ncclNdSetFatalError(base, ncclInternalError);
    }
  }

  // Match the tag to one unclaimed receive in the published group.
  int recvIndex = -1;
  for (int r = 0; r < nreqs; r++) {
    if (sComm->sendReqs[slot][r] == NULL && cts[r].tag == (uint32_t)tag) {
      recvIndex = r;
      break;
    }
  }
  if (recvIndex == -1) return ncclSuccess;
  TRACE(NCCL_NET, "NET/ND : Isend matched id=%llu slot=%d tag=%d index=%d/%d", (unsigned long long)base->fifoHead, slot,
        tag, recvIndex, nreqs);

  struct ncclNdMrHandle* mrHandle = (struct ncclNdMrHandle*)mhandle;
  if (mrHandle == NULL || data == NULL || size > INT_MAX) {
    WARN("NET/ND : Invalid send buffer (data=%p mhandle=%p size=%zu)", data, mhandle, size);
    return ncclInternalError;
  }

  if (base->nqps <= 0) {
    WARN("NET/ND : No QP available for isend");
    return ncclNdSetFatalError(base, ncclInternalError);
  }

  if (base->remCompletionRecords.addr == 0) {
    WARN("NET/ND : Missing remote completion-record metadata");
    return ncclNdSetFatalError(base, ncclInternalError);
  }
  // Validate the peer address and size before allocating a send request.
  uint64_t remoteSize = cts[recvIndex].size;
  if (remoteSize > INT_MAX || cts[recvIndex].addr == 0) {
    WARN("NET/ND : Invalid CTS entry %d (addr=%llu size=%llu)", recvIndex, (unsigned long long)cts[recvIndex].addr,
         (unsigned long long)remoteSize);
    return ncclNdSetFatalError(base, ncclInternalError);
  }

  // Capture local and remote MR tokens for the ordered data and record writes.
  struct ncclNdRequest* req;
  NCCLCHECK(ncclNdGetRequest(base, NCCL_NET_ND_REQ_SEND, &req));
  if (req == NULL) return ncclSuccess;
  req->id = base->fifoHead;
  req->nreqs = 1;
  req->send.data = data;
  req->send.size = (int)(size < (size_t)remoteSize ? size : (size_t)remoteSize);
  req->send.offset = 0;
  req->send.slot = slot;
  req->send.index = recvIndex;
  req->send.stage = NCCL_ND_SEND_STAGE_DATA;
  req->send.remoteAddr = cts[recvIndex].addr;
  base->remCompletionRecords.acknowledgements[slot][recvIndex] = 0;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  for (int i = 0; i < base->ndevs; i++) req->send.localTokens[i] = wrap_nd_get_local_token(mrHandle->mrs[i]);
  for (int i = 0; i < base->nRemDevs; i++) req->send.remoteTokens[i] = cts[recvIndex].tokens[i];

  // Data writes are idempotent and the generation-tagged completion record is
  // written last on the same QP. This permits a failed request to be replayed
  // on a surviving rail without exposing a partial transfer as complete.
  ncclResult_t result = ncclNdPostRequestWithFailover(req);
  if (result != ncclSuccess) {
    NCCLCHECK(ncclNdFreeRequest(req));
    return ncclNdSetFatalError(base, result);
  }

  sComm->sendReqs[slot][recvIndex] = req;
  sComm->sendReqsCnt[slot]++;
  if (sComm->sendReqsCnt[slot] == nreqs) {
    memset((void*)cts, 0, sizeof(*cts) * NCCL_NET_ND_MAX_RECVS);
    memset(sComm->sendReqs[slot], 0, sizeof(sComm->sendReqs[slot]));
    sComm->sendReqsCnt[slot] = 0;
    sComm->ctsDeadlineMs = 0;
    base->fifoHead++;
  }

  *request = req;
  return ncclSuccess;
}

// Flush received GPUDirect writes into CUDA-visible memory.
ncclResult_t ncclNdIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclNdRecvComm* rComm = (struct ncclNdRecvComm*)recvComm;
  struct ncclNdNetCommBase* base = &rComm->base;
  ncclNdScopedSrwLock progressLock(&base->progressLock);
  (void)data;
  (void)mhandles;
  *request = NULL;
  ncclResult_t fatalError = ncclNdGetFatalError(base);
  if (fatalError != ncclSuccess) return fatalError;
  NCCLCHECK(ncclNdCheckProvider(base));

  int last = -1;
  for (int i = 0; i < n; i++)
    if (sizes[i] > 0) last = i;
  if (last == -1) return ncclSuccess;

#if CUDART_VERSION >= 11030
  // NetworkDirect has no verbs-style local loopback QP for flushing GPU
  // visibility. Use CUDA's host-side GPUDirect RDMA write flush instead. The
  // proxy thread has the receiving GPU's context current, and this call does
  // not return until prior remote writes are visible to that context.
  ncclResult_t result = ncclSuccess;
  CUDACHECKGOTO(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                   cudaFlushGPUDirectRDMAWritesToOwner),
                result, fail);
  TRACE(NCCL_NET, "NET/ND : Flushed GPUDirect RDMA writes for %d receives", n);
  return ncclSuccess;

fail:
  return ncclNdSetFatalError(base, result);
#else
  WARN("NET/ND : GPUDirect RDMA write flush requires CUDA 11.3 or newer");
  return ncclNdSetFatalError(base, ncclInternalError);
#endif
}

// Progress shared CQs and report completion for one asynchronous request.
ncclResult_t ncclNdTest(void* request, int* done, int* sizes) {
  struct ncclNdRequest* req = (struct ncclNdRequest*)request;
  *done = 0;

  if (!req) {
    *done = 1;
    return ncclSuccess;
  }
  ncclNdScopedSrwLock progressLock(&req->base->progressLock);
  ncclResult_t fatalError = ncclNdGetFatalError(req->base);
  if (fatalError != ncclSuccess) return fatalError;
  NCCLCHECK(ncclNdCheckProvider(req->base));

  // Poll each CQ needed by this request. Results can belong to any in-flight
  // request sharing that CQ, so dispatch by RequestContext instead of dropping
  // completions that do not match the request currently being tested.
  for (int i = 0; i < req->base->ndevs; i++) {
    if (req->events[i] > 0) {
      // Poll completion queue
      struct ncclNdNetCommDevBase* devBase = req->devBases[i];
      ND2_RESULT results[NCCL_ND_MAX_CQ_POLL_BATCH];
      ULONG nResults = wrap_nd_get_results(devBase->cq, results, NCCL_ND_CQ_POLL_BATCH);

      // Process completions
      for (ULONG j = 0; j < nResults; j++) {
        ND2_RESULT* result = &results[j];

        struct ncclNdDecodedCompletionContext context = {};
        if (!ncclNdDecodeCompletionContext(result->RequestContext, &context)) {
          WARN("NET/ND : Completion has unknown request context %p", result->RequestContext);
          return ncclNdSetFatalError(req->base, ncclInternalError);
        }
        struct ncclNdRequest* owner = &req->base->reqs[context.slot];
        ULONG_PTR ownerGeneration = (ULONG_PTR)(owner->id + 1) & kNdContextGenerationMask;
        if (owner->base != req->base || owner->type == NCCL_NET_ND_REQ_UNUSED ||
            ownerGeneration != context.generation) {
          TRACE(NCCL_NET, "NET/ND : Ignoring stale completion context %p", result->RequestContext);
          continue;
        }
        if (context.attempt != owner->replayCount) {
          TRACE(NCCL_NET, "NET/ND : Ignoring completion from replay attempt %d (current %d) for request %p",
                context.attempt, owner->replayCount, owner);
          continue;
        }
        if (FAILED(result->Status)) {
          if (!ncclNdCompletionCanFailover(result->Status)) {
            if (req->base->stats != NULL) req->base->stats->cqErrors.fetch_add(1, std::memory_order_relaxed);
            char reason[160];
            snprintf(reason, sizeof(reason), "QP completion has non-retryable status 0x%08x",
                     (unsigned int)result->Status);
            return ncclNdFailDataPath(req->base, ncclSystemError, reason);
          }
          if (owner->events[i] <= 0 || owner->qpIndex < 0 || owner->qpIndex >= req->base->nqps ||
              req->base->qps[owner->qpIndex].devIndex != i) {
            WARN("NET/ND : Invalid failed completion context %p on device %d", owner, i);
            return ncclNdSetFatalError(req->base, ncclInternalError);
          }
          owner->events[i]--;
          InterlockedDecrement64(&req->base->qps[owner->qpIndex].outstanding);
          InterlockedAdd64(&req->base->qps[owner->qpIndex].outstandingBytes, -owner->postedBytes);
          char reason[128];
          snprintf(reason, sizeof(reason), "QP %d completion failed with status 0x%08x", owner->qpIndex,
                   (unsigned int)result->Status);
          NCCLCHECK(ncclNdHandleRailFailure(req->base, i, reason));
          if (owner->type == NCCL_NET_ND_REQ_SEND && owner->send.stage == NCCL_ND_SEND_STAGE_DATA &&
              ncclNdSendWasAcknowledged(owner)) {
            // The receiver does not return its buffer until this acknowledgement
            // write completes. If it is visible here, the failed local CQE was
            // an acknowledgement-loss ambiguity and the send must not be replayed.
            owner->replayCount++;
            if (req->base->stats != NULL) req->base->stats->cqErrors.fetch_add(1, std::memory_order_relaxed);
            TRACE(NCCL_NET, "NET/ND : Request %p was acknowledged before its rail failed", owner);
            continue;
          }
          if (owner->replayCount >= kNdMaxReplayAttempts) {
            if (req->base->stats != NULL) req->base->stats->failoverFailures.fetch_add(1, std::memory_order_relaxed);
            return ncclNdFailDataPath(req->base, ncclSystemError,
                                      "NetworkDirect replay budget exhausted after CQ error");
          }
          owner->replayCount++;
          if (req->base->stats != NULL) {
            req->base->stats->cqErrors.fetch_add(1, std::memory_order_relaxed);
            req->base->stats->replayedRequests.fetch_add(1, std::memory_order_relaxed);
          }
          NCCLCHECK(ncclNdPostRequestWithFailover(owner));
          continue;
        }
        bool expectedCompletion =
          (owner->type == NCCL_NET_ND_REQ_RECV &&
           context.op ==
             (owner->recv.stage == NCCL_ND_RECV_STAGE_CTS ? ncclNdCompletionOpCts : ncclNdCompletionOpRecord)) ||
          (owner->type == NCCL_NET_ND_REQ_SEND && context.op == ncclNdCompletionOpRecord);
        if (!expectedCompletion) {
          if (owner->type == NCCL_NET_ND_REQ_SEND && context.op == ncclNdCompletionOpData) {
            TRACE(NCCL_NET, "NET/ND : Ignoring successful silent data completion for request %p", owner);
            continue;
          }
          WARN("NET/ND : Completion operation %d does not match request type %d", context.op, owner->type);
          return ncclNdSetFatalError(req->base, ncclInternalError);
        }
        if (owner->events[i] <= 0) {
          WARN("NET/ND : Unexpected completion for request %p on device %d", owner, i);
          return ncclNdSetFatalError(req->base, ncclInternalError);
        }
        owner->events[i]--;
        if (owner->qpIndex >= 0 && owner->qpIndex < req->base->nqps) {
          InterlockedDecrement64(&req->base->qps[owner->qpIndex].outstanding);
          InterlockedAdd64(&req->base->qps[owner->qpIndex].outstandingBytes, -owner->postedBytes);
        }
      }
    }
  }

  int allComplete = 1;
  for (int i = 0; i < req->base->ndevs; i++) {
    if (req->events[i] > 0) allComplete = 0;
  }

  // In failover mode, wait for peer acknowledgement before releasing buffer ownership.
  if (allComplete && req->type == NCCL_NET_ND_REQ_SEND && ncclParamNdResiliencyPortFailover() != 0) {
    if (req->send.stage == NCCL_ND_SEND_STAGE_DATA) req->send.stage = NCCL_ND_SEND_STAGE_WAIT_ACK;
    if (req->send.stage == NCCL_ND_SEND_STAGE_WAIT_ACK) {
      if (ncclNdSendWasAcknowledged(req)) {
        req->send.stage = NCCL_ND_SEND_STAGE_RELEASE;
        NCCLCHECK(ncclNdPostRequestWithFailover(req));
      }
      allComplete = 0;
    }
  }

  // A receive completes only after every tag-selected data write has been
  // followed by its generation-tagged completion record. The upper 32 bits
  // distinguish a new FIFO generation from stale data after slot reuse; the
  // lower 32 bits carry the transferred size and therefore support size zero.
  if (allComplete && req->type == NCCL_NET_ND_REQ_RECV) {
    const uint32_t expectedGeneration = ncclNdCompletionGeneration(req->id);
    const int slot = req->recv.slot;
    for (int r = 0; r < req->nreqs; r++) {
      volatile uint64_t* record = &req->base->completionRecords[slot][r];
      uint64_t value = *record;
      if ((uint32_t)(value >> 32) != expectedGeneration) {
        allComplete = 0;
        break;
      }
    }
    if (allComplete) {
      std::atomic_thread_fence(std::memory_order_seq_cst);
      for (int r = 0; r < req->nreqs; r++) {
        uint64_t value = req->base->completionRecords[slot][r];
        if ((uint32_t)(value >> 32) != expectedGeneration) {
          allComplete = 0;
          break;
        }
        req->recv.sizes[r] = (int)(uint32_t)value;
      }
    }
    if (allComplete && ncclParamNdResiliencyPortFailover() != 0) {
      if (req->recv.stage == NCCL_ND_RECV_STAGE_CTS) {
        // A send completion can fail locally after its writes reached this
        // peer. Acknowledge the generation, but keep ownership of the receive
        // buffers until the sender releases them after resolving any replay.
        req->recv.stage = NCCL_ND_RECV_STAGE_ACK;
        NCCLCHECK(ncclNdPostRequestWithFailover(req));
        allComplete = 0;
      } else if (req->recv.stage == NCCL_ND_RECV_STAGE_ACK) {
        req->recv.stage = NCCL_ND_RECV_STAGE_WAIT_RELEASE;
        allComplete = 0;
      } else {
        for (int r = 0; r < req->nreqs; r++) {
          volatile uint64_t* release = &req->base->releaseRecords[slot][r];
          uint64_t expected = ncclNdCompletionValue(req->id, (size_t)req->recv.sizes[r]);
          if (*release != expected) {
            allComplete = 0;
            break;
          }
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
      }
    }
  }

  if (allComplete) {
    *done = 1;
    // Return receive sizes and release the request only after protocol completion.
    if (req->type == NCCL_NET_ND_REQ_RECV && req->recv.sizes) {
      if (sizes) {
        for (int r = 0; r < req->nreqs; r++) {
          sizes[r] = req->recv.sizes[r];
        }
      }
      free(req->recv.sizes);
      req->recv.sizes = NULL;
      struct ncclNdRecvComm* rComm = (struct ncclNdRecvComm*)req->base;
      rComm->recvReqs[req->recv.slot] = NULL;
    }
    NCCLCHECK(ncclNdFreeRequest(req));
  } else {
    ncclResult_t peerResult = ncclNdCheckPeer(req->base);
    if (peerResult != ncclSuccess) return peerResult;
    if (req->deadlineMs != 0 && GetTickCount64() >= req->deadlineMs) {
      return ncclNdFailDataPath(req->base, ncclRemoteError, "timed out waiting for request completion",
                                ncclNdDataFailureTimeout);
    }
  }

  return ncclSuccess;
}
