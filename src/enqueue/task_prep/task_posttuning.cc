/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "argcheck.h"
#include "sym_kernels.h"
#include "ce_coll.h"
#include "channel.h"
#include "checks.h"
#include "collectives.h"
#include "comm.h"
#include "config/collconfig.h"
#include "dev_runtime.h"
#include "device.h"
#include "enqueue.h"
#include "enqueue/task_posttuning.h"
#include "graph.h"
#include "group.h"
#include "register.h"
#include "scheduler.h"
#include "transport.h"
#include "profiler.h"
#include "compiler.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>

// Config-option env overrides, defined via NCCL_PARAM in init.cc (external linkage).
int64_t ncclParamMinCTAs();
int64_t ncclParamMaxCTAs();
int64_t ncclParamNvlsChannels();
int64_t ncclParamCGAClusterSize();

static inline int ncclFuncTrafficPerByte(ncclFunc_t func, int nRanks) {
  switch (func) {
  case ncclFuncAllReduce:
    return 2;
  case ncclFuncAllGather:
    return nRanks;
  case ncclFuncReduceScatter:
    return nRanks;
  default:
    return 1;
  }
}

static inline int postTuningFuncTrafficPerByte(ncclFunc_t func, int nRanks) {
  switch (func) {
  case ncclFuncAllReduce:
    return 2;
  case ncclFuncAllGather:
    return nRanks;
  case ncclFuncReduceScatter:
    return nRanks;
  default:
    return 1;
  }
}

static void postTuningSetChunkSteps(struct ncclTaskColl* task) {
  switch (task->func) {
  case ncclFuncAllReduce:
    task->chunkSteps = ALLREDUCE_CHUNKSTEPS;
    task->sliceSteps = ALLREDUCE_SLICESTEPS;
    break;
  case ncclFuncAllGather:
    task->chunkSteps = ALLGATHER_CHUNKSTEPS;
    task->sliceSteps = ALLGATHER_SLICESTEPS;
    break;
  case ncclFuncReduceScatter:
    task->chunkSteps = REDUCESCATTER_CHUNKSTEPS;
    task->sliceSteps = REDUCESCATTER_SLICESTEPS;
    break;
  case ncclFuncBroadcast:
    task->chunkSteps = BROADCAST_CHUNKSTEPS;
    task->sliceSteps = BROADCAST_SLICESTEPS;
    break;
  default:
    task->chunkSteps = 1;
    task->sliceSteps = 1;
    break;
  }
}

static ncclResult_t fillCollTaskFromRaw(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo,
                                        struct ncclTaskColl* task) {
  const struct ncclRawTaskColl* raw = &tInfo->raw->coll;

  memset(task, 0, sizeof(*task));
  task->func = raw->func;
  task->sendbuff = raw->sendbuff;
  task->recvbuff = raw->recvbuff;
  task->count = raw->count;
  task->root = raw->root;
  task->datatype = raw->datatype;
  size_t elementSize = ncclTypeSize(task->datatype);
  if (task->func == ncclFuncAllGather || task->func == ncclFuncBroadcast) {
    task->count *= elementSize;
    task->datatype = ncclInt8;
    elementSize = 1;
  }
  task->trafficBytes = task->count * elementSize * postTuningFuncTrafficPerByte(task->func, comm->nRanks);
  task->opHost = raw->opHost;
  task->opDev = raw->opDev;
  postTuningSetChunkSteps(task);
  task->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
  task->groupApiEventHandle = nullptr;
  task->collApiEventHandle = nullptr;

  const ncclCollConfig_t* cfg = &raw->collConfig;
  bool ctaPolicyEnvOverridden = ncclGetEnvCtaPolicy() != NCCL_CONFIG_UNDEF_INT;
  task->CTAPolicy = ncclCollConfigResolveCTAPolicy(cfg->CTAPolicy, comm->config.CTAPolicy, ctaPolicyEnvOverridden);
  task->aggIsolate = ncclCollConfigNeedAggIsolate(cfg) || task->CTAPolicy != comm->config.CTAPolicy;
  NCCL_CONFIG_SET(task, minCTAs, ncclParamMinCTAs(), cfg->minCTAs, comm->config.minCTAs, 1, MAXCHANNELS);
  NCCL_CONFIG_SET(task, maxCTAs, ncclParamMaxCTAs(),
                  (std::min(cfg->maxCTAs, comm->config.maxCTAs)) /* clamp config's maxCTAs by comm's */,
                  comm->config.maxCTAs, 1, MAXCHANNELS);
  if (task->minCTAs > task->maxCTAs) {
    INFO(NCCL_COLL, "Task minCTAs(%d) is larger than maxCTAs(%d), reset task minCTAs to 1", task->minCTAs,
         task->maxCTAs);
    task->minCTAs = 1;
  }
  NCCL_CONFIG_SET(task, nvlsCTAs, ncclParamNvlsChannels(), cfg->nvlsCTAs, comm->config.nvlsCTAs, 1, MAXCHANNELS);
  NCCL_CONFIG_SET(task, cgaClusterSize, ncclParamCGAClusterSize(), cfg->cgaClusterSize, comm->config.cgaClusterSize, 0,
                  NCCL_MAX_CGA_CLUSTER_SIZE);
  NCCLCHECK(ncclCollConfigGetAlgMask(cfg, task->func, &task->algMask));
  task->forceAlgSelection = cfg->forceAlgSelection;

  return ncclSuccess;
}

static ncclResult_t postTuneSymFreeTuningInfoRaw(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo) {
  if (tInfo->raw != nullptr) {
    ncclMemoryPoolFree(&comm->memPool_ncclRawTask, tInfo->raw);
    tInfo->raw = nullptr;
  }
  return ncclSuccess;
}

static ncclResult_t postTuneP2pChannelBase(struct ncclComm* comm, int peer, bool isSendNotRecv, uint8_t* baseOut) {
  int round = 0;
  while (peer != (isSendNotRecv ? comm->p2pSchedule[round].sendRank : comm->p2pSchedule[round].recvRank)) {
    round += 1;
  }
  *baseOut = ncclP2pChannelBaseForRound(comm, round);
  return ncclSuccess;
}

static ncclResult_t postTuneP2pRecordPreconnect(struct ncclComm* comm, int peer, bool isSendNotRecv,
                                                bool* needPreconnect) {
  struct ncclKernelPlanner* planner = &comm->planner;
  uint8_t base;

  if (peer < 0 || peer >= comm->nRanks) return ncclInvalidArgument;
  if (comm->rank == peer) return ncclSuccess;
  if (isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen) return ncclSuccess;

  NCCLCHECK(postTuneP2pChannelBase(comm, peer, isSendNotRecv, &base));

  // Mark channels that need pre-connect. planner->peers[peer].send/recvSeen is
  // private to each comm, so we need to set it anyway.
  (isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen) = true;
  for (int c = 0; c < comm->p2pnChannelsPerPeer; c++) {
    int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, c);

    // P2P uses only 1 connector. The send/recv connector is shared among split
    // shared comms, so set hasSeen to avoid duplicate connection setup if user
    // groups sendrecv ops with split shared comms together.
    if (isSendNotRecv) {
      if (comm->channels[channelId].peers[peer]->send[1].hasSeen == 0) {
        comm->channels[channelId].peers[peer]->send[1].hasSeen = 1;
        comm->channels[channelId].peers[peer]->send[1].p2pOnly = 1;
        comm->connectSend[peer] |= (1ULL << channelId);
        *needPreconnect = true;
      }
    } else {
      if (comm->channels[channelId].peers[peer]->recv[1].hasSeen == 0) {
        comm->channels[channelId].peers[peer]->recv[1].hasSeen = 1;
        comm->channels[channelId].peers[peer]->recv[1].p2pOnly = 1;
        comm->connectRecv[peer] |= (1ULL << channelId);
        *needPreconnect = true;
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t fillP2pTaskFromRaw(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo,
                                       struct ncclTaskP2p* task) {
  struct ncclRawTaskSendRecv* raw = &tInfo->raw->sendRecv;

  if (raw->peer < 0 || raw->peer >= comm->nRanks) return ncclInvalidArgument;
  if (raw->func != ncclFuncSend && raw->func != ncclFuncRecv) return ncclInternalError;

  task->func = raw->func;
  task->collAPI = raw->collAPI;
  task->buff = raw->buff;
  task->count = raw->count;
  task->datatype = raw->datatype;
  task->root = raw->peer;
  task->bytes = raw->bytes;
  if (task->collAPI == ncclFuncAlltoAll || task->collAPI == ncclFuncScatter || task->collAPI == ncclFuncGather) {
    task->allowUB = false;
  } else {
    task->allowUB = true;
  }
  task->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
  task->groupApiEventHandle = nullptr;
  task->p2pApiEventHandle = nullptr;
  return ncclSuccess;
}

static ncclResult_t postTuneP2pRegisterBuffer(struct ncclComm* comm, struct ncclTaskP2p* task, bool isSendNotRecv,
                                              int protocol) {
  constexpr int connIndex = 1;
  struct ncclKernelPlanner* planner = &comm->planner;
  int peer = task->root;
  uint8_t base;
  bool network;
  bool proxySameProcess;

  if (protocol != NCCL_PROTO_SIMPLE) return ncclSuccess;
  if (!task->allowUB || task->bytes == 0 || task->buff == nullptr || peer == comm->rank) return ncclSuccess;

  NCCLCHECK(postTuneP2pChannelBase(comm, peer, isSendNotRecv, &base));

  int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, 0);
  struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
  struct ncclConnector* conn =
    isSendNotRecv ? &channelPeers[peer]->send[connIndex] : &channelPeers[peer]->recv[connIndex];

  network = conn->transportComm == (isSendNotRecv ? &netTransport.send : &netTransport.recv);
  proxySameProcess = conn->proxyConn.sameProcess;

  if (network) {
    bool pxnUsed = !ncclPxnDisable(comm) && comm->isAllNvlink && comm->maxLocalRanks > 1;
    if (proxySameProcess && !pxnUsed && (conn->conn.flags & NCCL_DIRECT_NIC)) {
      for (int part = 0; part < comm->p2pnChannelsPerPeer; part++) {
        int partChannelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
        struct ncclConnector* partConn = isSendNotRecv ? &comm->channels[partChannelId].peers[peer]->send[connIndex] :
                                                         &comm->channels[partChannelId].peers[peer]->recv[connIndex];
        int regFlag = 0;
        void* handle = nullptr;
        NCCLCHECK(ncclRegisterP2pNetBuffer(comm, task->buff, task->bytes, partConn, &regFlag, &handle,
                                           &planner->collCleanupQueue));
        if (!regFlag) break;
      }
    }
  } else if (conn->conn.flags & (NCCL_P2P_WRITE | NCCL_P2P_READ)) {
    int regFlag = 0;
    void* regAddr = nullptr;
    NCCLCHECK(ncclRegisterP2pIpcBuffer(comm, task->buff, task->bytes, peer, &regFlag, &regAddr,
                                       &planner->collCleanupQueue));
  }
  return ncclSuccess;
}

static ncclResult_t postTuneTasksDebug(struct ncclComm* comm) {
  while (!ncclIntruQueueEmpty(&comm->argsInfoQueue)) {
    struct ncclArgsInfo* argsInfo = ncclIntruQueueDequeue(&comm->argsInfoQueue);
    ncclResult_t ret = ncclArgsGlobalCheck(argsInfo);
    free(argsInfo);
    NCCLCHECK(ret);
  }
  return ncclSuccess;
}

static ncclResult_t postTuneSymTasksLazyInit(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* symTaskQueue) {
  if (ncclIntruQueueEmpty(symTaskQueue)) return ncclSuccess;
  return ncclSymkInitOnce(comm);
}

static ncclResult_t postTuneRmaFreeTuningInfoRaw(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo) {
  if (tInfo->raw != nullptr) {
    ncclMemoryPoolFree(&comm->memPool_ncclRawTask, tInfo->raw);
    tInfo->raw = nullptr;
  }
  return ncclSuccess;
}

static ncclResult_t postTuneRmaTaskAppend(struct ncclComm* comm, const struct ncclRawTaskRma* raw) {
  struct ncclKernelPlanner* planner = &comm->planner;

  ncclFunc_t func = raw->func;
  void const* srcBuff = nullptr;
  size_t count = 0;
  ncclDataType_t datatype = ncclInt8;
  int peer = 0;
  ncclWindow_t peerWin = nullptr;
  size_t peerWinOffset = 0;
  int sigIdx = 0;
  int ctx = 0;
  unsigned int flags = 0;
  int nDesc = 0;
  const ncclWaitSignalDesc_t* signalDescs = nullptr;

  switch (func) {
  case ncclFuncPutSignal:
    srcBuff = raw->rmaOp.putSignal.localbuff;
    count = raw->rmaOp.putSignal.count;
    datatype = raw->rmaOp.putSignal.datatype;
    peer = raw->rmaOp.putSignal.peer;
    peerWin = raw->rmaOp.putSignal.peerWin;
    peerWinOffset = raw->rmaOp.putSignal.peerWinOffset;
    sigIdx = raw->rmaOp.putSignal.sigIdx;
    ctx = raw->rmaOp.putSignal.ctx;
    flags = raw->rmaOp.putSignal.flags;
    break;
  case ncclFuncSignal:
    peer = raw->rmaOp.signal.peer;
    sigIdx = raw->rmaOp.signal.sigIdx;
    ctx = raw->rmaOp.signal.ctx;
    flags = raw->rmaOp.signal.flags;
    break;
  case ncclFuncWaitSignal:
    datatype = ncclInt32;
    nDesc = raw->rmaOp.waitSignal.nDesc;
    signalDescs = raw->rmaOp.waitSignal.signalDescs;
    break;
  default:
    return ncclInternalError;
  }

  if (!comm->hostRmaSupport) {
    WARN("One sided RMA: host RMA is not supported in this communicator.");
    return ncclInvalidArgument;
  }

  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
  if (driverVersion < 12050) {
    WARN("One-sided RMA requires CUDA driver 12.5 or later (found %d.%d).", driverVersion / 1000,
         (driverVersion % 1000) / 10);
    return ncclInvalidUsage;
  }

  // Check if signal index is valid
  if (sigIdx < 0 || sigIdx >= comm->config.numRmaSig) {
    WARN("Signal index %d is invalid (must be in [0, %d))", sigIdx, comm->config.numRmaSig);
    return ncclInvalidArgument;
  }

  // Check if flags is valid
  if (flags != 0) {
    WARN("Flags %u is invalid (must be 0)", flags);
    return ncclInvalidArgument;
  }

  // ncclSignal / ncclWaitSignal take no window, so they cannot trigger the collective RMA init that
  // runs at the first window registration. Initializing here for a subset of ranks would deadlock,
  // so instead require the user to register a symmetric window first or opt into eager init.
  if ((func == ncclFuncSignal || func == ncclFuncWaitSignal) && !ncclRmaInitialized(comm)) {
    WARN("ncclSignal/ncclWaitSignal called before RMA is initialized. Register a symmetric window "
         "first, or set NCCL_RMA_EAGER_INIT=1 to initialize RMA at communicator creation.");
    return ncclInvalidUsage;
  }

  // Initialize window pointers - only needed for Put and Signal
  struct ncclDevrWindow* peerWinHost = NULL;
  struct ncclDevrWindow* srcWinHost = NULL;
  size_t srcWinOffset = 0;

  if (func == ncclFuncPutSignal) {
    // Validate peer window with detailed debugging
    if (peerWin == NULL) {
      WARN("ncclPutSignal: peerWin is NULL");
      return ncclInvalidArgument;
    }

    struct ncclWindow_vidmem* peerWinDevHost = NULL;
    NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, peerWin, &peerWinDevHost));
    peerWinHost = (struct ncclDevrWindow*)peerWinDevHost->winHost;

    // Validate source buffer and window
    if (srcBuff == NULL) {
      WARN("ncclPutSignal: srcBuff is NULL");
      return ncclInvalidArgument;
    }
    NCCLCHECK(ncclDevrFindWindow(comm, srcBuff, &srcWinHost));
    if (srcWinHost == NULL || !(srcWinHost->winFlags & NCCL_WIN_COLL_SYMMETRIC)) {
      WARN("ncclPutSignal: srcWinHost is not in a valid symmetric window");
      return ncclInvalidArgument;
    }
    srcWinOffset = (char*)srcBuff - (char*)srcWinHost->userPtr;

    bool isMultiSegment = ncclDevrWindowIsMultiSegment(srcWinHost) || ncclDevrWindowIsMultiSegment(peerWinHost);
    bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(srcWinHost) || ncclDevrWindowHasSysmemSegment(peerWinHost);

    if (isMultiSegment) {
      WARN("ncclPutSignal currently does not support VAs backed by multiple physical cuMem segments");
      return ncclInvalidArgument;
    }
    if (hasSysmemSegment) {
      WARN("ncclPutSignal currently does not support VAs with host-backed cuMem segments");
      return ncclInvalidArgument;
    }
  } else if (func == ncclFuncWaitSignal) {
    // Check if signalDescs is valid
    if (signalDescs == NULL || nDesc == 0) {
      WARN("ncclWaitSignal: invalid arguments");
      return ncclInvalidArgument;
    }
    // Validate each descriptor
    for (int i = 0; i < nDesc; i++) {
      if (signalDescs[i].opCnt <= 0) {
        WARN("ncclWaitSignal: descriptor %d has invalid opCnt %d", i, signalDescs[i].opCnt);
        return ncclInvalidArgument;
      }
      if (signalDescs[i].sigIdx < 0 || signalDescs[i].sigIdx >= comm->config.numRmaSig) {
        WARN("ncclWaitSignal: descriptor %d has invalid sigIdx %d (must be in [0, %d))", i, signalDescs[i].sigIdx,
             comm->config.numRmaSig);
        return ncclInvalidArgument;
      }
    }
  }

  // Handle WaitSignal separately
  if (func == ncclFuncWaitSignal) {
    // A single ncclWaitSignal may span multiple contexts (ctx is per-descriptor).
    // Group descriptors by ctx and emit one task per distinct ctx, filed into that
    // ctx's queue. numRmaCtx is small, so a numRmaCtx x nDesc scan is fine; every
    // descriptor ctx is already validated to be in [0, numRmaCtx) above.
    for (int c = 0; c < comm->config.numRmaCtx; c++) {
      // Count descriptors targeting context c.
      int nForCtx = 0;
      for (int i = 0; i < nDesc; i++) {
        if (signalDescs[i].ctx == c) nForCtx++;
      }
      if (nForCtx == 0) continue;

      struct ncclTaskRma* t = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);

      t->func = ncclFuncWaitSignal;
      t->ctx = c;
      t->count = 0;
      t->bytes = 0;
      t->srcBuff = NULL;
      t->srcWinOffset = 0;
      t->srcWinHost = NULL;
      t->peer = 0;
      t->peerWinOffset = 0;
      t->peerWinHost = NULL;
      t->signalMode = NCCL_SIGNAL;
      t->signalIdx = 0;

      // Convert the descriptors targeting ctx c into peers, nsignals and signalIdxs arrays.
      t->npeers = nForCtx;
      t->peers = ncclMemoryStackAlloc<int>(&comm->memScoped, nForCtx);
      t->nsignals = ncclMemoryStackAlloc<int>(&comm->memScoped, nForCtx);
      t->signalIdxs = ncclMemoryStackAlloc<int>(&comm->memScoped, nForCtx);

      int k = 0;
      for (int i = 0; i < nDesc; i++) {
        if (signalDescs[i].ctx != c) continue;
        t->peers[k] = signalDescs[i].peer;
        t->nsignals[k] = signalDescs[i].opCnt;
        t->signalIdxs[k] = signalDescs[i].sigIdx;
        k++;
      }

      t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
      planner->nTasksRma++;
      ncclIntruQueueEnqueue(&planner->rmaTaskQueues[t->ctx], t);
    }

  } else if (func == ncclFuncPutSignal || func == ncclFuncSignal) {
    // Calculate total bytes for the operation
    size_t totalBytes = count * ncclTypeSize(datatype);

    // Define 1GB chunk size for splitting large put operations
    const size_t chunkSize = 1ULL << 30; // 1GB = 1073741824 bytes

    // Determine if we need to split the operation
    int numChunks = 1;
    if (func == ncclFuncPutSignal && totalBytes > chunkSize) {
      numChunks = (totalBytes + chunkSize - 1) / chunkSize;
    }

    // Create tasks for each chunk
    for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
      struct ncclTaskRma* t = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);

      // Calculate chunk-specific size and offsets
      size_t chunkBytes = (chunkIdx == numChunks - 1) ? (totalBytes - chunkIdx * chunkSize) : chunkSize;

      size_t chunkOffset = chunkIdx * chunkSize;

      t->func = func;
      t->srcBuff = (const char*)srcBuff + chunkOffset;
      t->srcWinOffset = srcWinOffset + chunkOffset;
      t->srcWinHost = srcWinHost;
      t->count = chunkBytes / ncclTypeSize(datatype);
      t->datatype = datatype;
      t->bytes = chunkBytes;
      t->ctx = ctx;
      t->signalIdx = sigIdx;
      t->peer = peer;
      t->peerWinOffset = peerWinOffset + chunkOffset;
      t->peerWinHost = peerWinHost;

      // Signal handling: only the last chunk gets the signal
      bool isLastChunk = (chunkIdx == numChunks - 1);
      if (isLastChunk) {
        t->signalMode = NCCL_SIGNAL;
      } else {
        // Earlier chunks: no signal
        t->signalMode = NCCL_SIGNAL_NONE;
      }
      t->peers = NULL;
      t->nsignals = NULL;
      t->signalIdxs = NULL;
      t->npeers = 0;

      t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);

      planner->nTasksRma++;
      // Enqueue the task into the appropriate context queue
      ncclIntruQueueEnqueue(&planner->rmaTaskQueues[t->ctx], t);
    }
  }

  return ncclSuccess;
}

static ncclResult_t postTuneRmaTasks(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* rmaTaskQueue) {
  while (!ncclIntruQueueEmpty(rmaTaskQueue)) {
    struct ncclTaskTuningInfo* tInfo = ncclIntruQueueDequeue(rmaTaskQueue);
    NCCLCHECK(postTuneRmaTaskAppend(comm, &tInfo->raw->rma));
    NCCLCHECK(postTuneRmaFreeTuningInfoRaw(comm, tInfo));
  }
  return ncclSuccess;
}

static ncclResult_t postTuneCeTasksLazyInit(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* ceTaskQueue) {
  if (ncclIntruQueueEmpty(ceTaskQueue)) return ncclSuccess;
  if (comm->ceColl.initialized) return ncclSuccess;
  return ncclCeInit(comm);
}

static ncclResult_t postTuneSymTasks(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* symTaskQueue) {
  struct ncclKernelPlanner* planner = &comm->planner;

  NCCLCHECK(postTuneSymTasksLazyInit(comm, symTaskQueue));
  NCCLCHECK(postTuneTasksDebug(comm));
  if (ncclIntruQueueEmpty(symTaskQueue)) return ncclSuccess;

  while (!ncclIntruQueueEmpty(symTaskQueue)) {
    struct ncclTaskTuningInfo* tInfo = ncclIntruQueueDequeue(symTaskQueue);
    struct ncclTaskColl* task =
      ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);

    NCCLCHECK(fillCollTaskFromRaw(comm, tInfo, task));
    if (!tInfo->tuningOut.valid) return ncclInternalError;
    if (tInfo->tuningOut.symKernelId < 0 || tInfo->tuningOut.symKernelId >= ncclSymkKernelId_Count) {
      return ncclInternalError;
    }

    NCCLCHECK(ncclDevrFindWindow(comm, task->sendbuff, &task->sendWin));
    NCCLCHECK(ncclDevrFindWindow(comm, task->recvbuff, &task->recvWin));
    NCCLCHECK(ncclGetSymRegType(task->sendWin, task->recvWin, &task->winRegType));

    task->devFuncId = (uint32_t)tInfo->tuningOut.symKernelId;
    task->nMaxChannels = tInfo->tuningOut.nChannels;
    task->nWarps = tInfo->tuningOut.nWarps;
    task->isSymLast = 1;
    convertSymTaskDevOp(comm, task);

    ncclIntruQueueEnqueue(&planner->collSymTaskQueue, task);
    NCCLCHECK(postTuneSymFreeTuningInfoRaw(comm, tInfo));
  }

  return ncclSuccess;
}

static ncclResult_t applyTuningToCollTask(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo,
                                          struct ncclTaskColl* task) {
  NCCLCHECK(fillCollTaskFromRaw(comm, tInfo, task));
  if (!tInfo->tuningOut.valid) return ncclInternalError;

  task->algorithm = tInfo->tuningOut.algo;
  task->protocol = tInfo->tuningOut.proto;
  task->nMaxChannels = tInfo->tuningOut.maxChannels;
  task->nWarps = tInfo->tuningOut.nWarps;
  task->devFuncId = ncclDevFuncId(task->func, task->opDev.op, task->datatype, task->algorithm, task->protocol);
  task->isCollnet = 0;
  task->isNvls = 0;
  switch (task->algorithm) {
  case NCCL_ALGO_NVLS:
  case NCCL_ALGO_NVLS_TREE:
    task->isNvls = 1;
    task->isCollnet = (task->algorithm == NCCL_ALGO_NVLS && comm->nNodes > 1) ? 1 : 0;
    break;
  case NCCL_ALGO_PAT:
    // Multi-RPN PAT uses NVLS for intra-node transfers.
    task->isNvls = !comm->isOneRPN;
    break;
  case NCCL_ALGO_COLLNET_CHAIN:
  case NCCL_ALGO_COLLNET_DIRECT:
    task->isCollnet = 1;
    break;
  default:
    break;
  }
  if (task->protocol == NCCL_PROTO_LL) task->trafficBytes *= 4;
  return ncclSuccess;
}

static ncclResult_t postTuneLegacyRegisterCollBuffers(struct ncclComm* comm, struct ncclTaskColl* task,
                                                      void* regBufSend[NCCL_MAX_LOCAL_RANKS],
                                                      void* regBufRecv[NCCL_MAX_LOCAL_RANKS], bool* regNeedConnect) {
  struct ncclKernelPlanner* planner = &comm->planner;

  *regNeedConnect = true;
  NCCLCHECK(ncclRegisterCollBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, regNeedConnect));
  return ncclSuccess;
}

static ncclResult_t postTuneLegacyRegisterNvlsCollBuffers(struct ncclComm* comm, struct ncclTaskColl* task,
                                                          void* regBufSend[NCCL_MAX_LOCAL_RANKS],
                                                          void* regBufRecv[NCCL_MAX_LOCAL_RANKS],
                                                          bool* regNeedConnect) {
  struct ncclKernelPlanner* planner = &comm->planner;

  *regNeedConnect = true;
  NCCLCHECK(ncclRegisterCollNvlsBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue,
                                        regNeedConnect));
  return ncclSuccess;
}

static ncclResult_t postTuneLegacyEnqueueCollWork(
  struct ncclComm* comm, struct ncclKernelPlanner* planner, struct ncclTaskColl* task,
  void* regBufSend[NCCL_MAX_LOCAL_RANKS], void* regBufRecv[NCCL_MAX_LOCAL_RANKS],
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next>* workQueue) {
  struct ncclDevWorkColl devWork = {};
  struct ncclWorkList* workNode = nullptr;

  devWork.sendbuff = (void*)task->sendbuff;
  devWork.recvbuff = (void*)task->recvbuff;
  devWork.sendbuffOffset = task->sendbuffOffset;
  devWork.recvbuffOffset = task->recvbuffOffset;
  devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs;
  devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs;
  devWork.root = task->root;
  devWork.nWarps = task->nWarps;
  devWork.redOpArg = task->opDev.scalarArg;
  devWork.redOpArgIsPtr = task->opDev.scalarArgIsPtr;
  devWork.oneNode = (comm->nNodes == 1);
  devWork.netRegUsed = devWork.regUsed = 0;
  devWork.profilerEnabled = ncclProfilerPluginLoaded() && (task->eActivationMask & ncclProfileKernelCh);
  if (task->algorithm != NCCL_ALGO_NVLS && task->algorithm != NCCL_ALGO_NVLS_TREE) {
    devWork.isOneRPN = comm->isOneRPN;
  }
  if (task->regBufType & NCCL_NET_REG_BUFFER) devWork.netRegUsed = 1;
  if (task->regBufType & (NCCL_IPC_REG_BUFFER | NCCL_NVLS_REG_BUFFER)) devWork.regUsed = 1;

  if (task->regBufType & NCCL_NVLS_REG_BUFFER) {
    struct ncclDevWorkCollReg workReg = {};
    workReg.coll = devWork;
    workReg.dnInputs[0] = regBufSend[0];
    workReg.dnOutputs[0] = regBufRecv[0];
    workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
    workNode->workType = ncclDevWorkTypeCollReg;
    workNode->size = sizeof(struct ncclDevWorkCollReg);
    memcpy((void*)(workNode + 1), (void*)&workReg, workNode->size);
  } else {
    workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
    workNode->workType = ncclDevWorkTypeColl;
    workNode->size = sizeof(struct ncclDevWorkColl);
    memcpy((void*)(workNode + 1), (void*)&devWork, workNode->size);
  }

  ncclIntruQueueEnqueue(workQueue, workNode);
  return ncclSuccess;
}

static ncclResult_t postTuneLegacyRecordAlgoNeedConnect(struct ncclComm* comm, int algorithm, bool regNeedConnect,
                                                        bool* algoNeedConnect, bool* needConnect) {
  if (!comm->runtimeConn || comm->initAlgoChannels[algorithm]) return ncclSuccess;

  if (algorithm == NCCL_ALGO_NVLS_TREE && comm->initAlgoChannels[NCCL_ALGO_NVLS] == false && regNeedConnect == true) {
    comm->initAlgoChannels[NCCL_ALGO_NVLS] = true;
    algoNeedConnect[NCCL_ALGO_NVLS] = true;
  }
  if (algorithm != NCCL_ALGO_NVLS || regNeedConnect == true) {
    comm->initAlgoChannels[algorithm] = true;
    algoNeedConnect[algorithm] = true;
    *needConnect = true;
  }
  return ncclSuccess;
}

static ncclResult_t postTuneLegacyTasks(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* legacyTaskQueue) {
  bool needConnect = false;
  bool algoNeedConnect[NCCL_NUM_ALGORITHMS];
  struct ncclKernelPlanner* planner = &comm->planner;

  memset(algoNeedConnect, 0, sizeof(algoNeedConnect));

  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(legacyTaskQueue); tInfo != nullptr; tInfo = tInfo->next) {
    if (tInfo->tuningOut.algo == NCCL_ALGO_NVLS || tInfo->tuningOut.algo == NCCL_ALGO_NVLS_TREE) {
      struct ncclTaskColl* task =
        ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
      void* regBufSend[NCCL_MAX_LOCAL_RANKS];
      void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
      bool regNeedConnect = true;

      NCCLCHECK(applyTuningToCollTask(comm, tInfo, task));
      NCCLCHECK(postTuneLegacyRegisterNvlsCollBuffers(comm, task, regBufSend, regBufRecv, &regNeedConnect));
      NCCLCHECK(postTuneLegacyRecordAlgoNeedConnect(comm, tInfo->tuningOut.algo, regNeedConnect, algoNeedConnect,
                                                    &needConnect));
      NCCLCHECK(postTuneLegacyEnqueueCollWork(comm, planner, task, regBufSend, regBufRecv, &planner->collWorkQueue));
      planner->nTasksColl += 1;
      ncclIntruQueueEnqueue(&planner->collTaskQueue, task);
    } else {
      bool regNeedConnect = true;
      NCCLCHECK(postTuneLegacyRecordAlgoNeedConnect(comm, tInfo->tuningOut.algo, regNeedConnect, algoNeedConnect,
                                                    &needConnect));
    }
  }

  if (needConnect) {
    NCCLCHECK(ncclCollPreconnect(comm, algoNeedConnect));
  }

  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(legacyTaskQueue); tInfo != nullptr; tInfo = tInfo->next) {
    if (tInfo->tuningOut.algo == NCCL_ALGO_NVLS || tInfo->tuningOut.algo == NCCL_ALGO_NVLS_TREE) {
      continue;
    }

    struct ncclTaskColl* task =
      ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
    bool regNeedConnect = true;

    NCCLCHECK(applyTuningToCollTask(comm, tInfo, task));
    NCCLCHECK(postTuneLegacyRegisterCollBuffers(comm, task, regBufSend, regBufRecv, &regNeedConnect));
    NCCLCHECK(postTuneLegacyEnqueueCollWork(comm, planner, task, regBufSend, regBufRecv, &planner->collWorkQueue));
    planner->nTasksColl += 1;
    ncclIntruQueueEnqueue(&planner->collTaskQueue, task);
  }

  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(legacyTaskQueue); tInfo != nullptr; tInfo = tInfo->next) {
    NCCLCHECK(postTuneSymFreeTuningInfoRaw(comm, tInfo));
  }
  return ncclSuccess;
}

static ncclResult_t postTuneAllGatherVEnsureRingConnected(struct ncclComm* comm) {
  bool algoNeedConnect[NCCL_NUM_ALGORITHMS];

  if (!comm->runtimeConn || comm->initAlgoChannels[NCCL_ALGO_RING]) return ncclSuccess;

  memset(algoNeedConnect, 0, sizeof(algoNeedConnect));
  comm->initAlgoChannels[NCCL_ALGO_RING] = true;
  algoNeedConnect[NCCL_ALGO_RING] = true;
  NCCLCHECK(ncclCollPreconnect(comm, algoNeedConnect));
  return ncclSuccess;
}

static ncclResult_t postTuneAllGatherVEnqueueBroadcastTask(struct ncclComm* comm, struct ncclRawTaskAllGatherV* agv,
                                                           int root) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* task =
    ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
  void* regBufSend[NCCL_MAX_LOCAL_RANKS];
  void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
  bool regNeedConnect = true;
  int collNetSupport = 0;
  int nvlsSupport = 0;

  memset(task, 0, sizeof(*task));
  task->func = ncclFuncBroadcast;
  task->sendbuff = (root == comm->rank) ? agv->sendbuff : nullptr;
  task->recvbuff = agv->recvbuff[root];
  task->count = agv->counts[root];
  task->root = root;
  task->datatype = ncclInt8;
  task->trafficBytes = task->count * postTuningFuncTrafficPerByte(task->func, comm->nRanks);
  postTuningSetChunkSteps(task);
  task->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
  task->opHost = ncclSum;

  NCCLCHECK(ncclGetCollNetSupport(comm, task, &collNetSupport));
  nvlsSupport =
    comm->nvlsSupport && (ncclNvlsSupported(task->opDev.op, task->datatype) || task->func == ncclFuncAllGather);
  NCCLCHECK(ncclGetAlgoInfo(comm, task, collNetSupport, nvlsSupport, 1, nullptr));
  task->devFuncId = ncclDevFuncId(task->func, task->opDev.op, task->datatype, task->algorithm, task->protocol);
  switch (task->algorithm) {
  case NCCL_ALGO_NVLS:
  case NCCL_ALGO_NVLS_TREE:
    task->isNvls = 1;
    task->isCollnet = (task->algorithm == NCCL_ALGO_NVLS && comm->nNodes > 1) ? 1 : 0;
    break;
  case NCCL_ALGO_COLLNET_CHAIN:
  case NCCL_ALGO_COLLNET_DIRECT:
    task->isCollnet = 1;
    break;
  default:
    break;
  }
  if (task->protocol == NCCL_PROTO_LL) task->trafficBytes *= 4;

  NCCLCHECK(postTuneLegacyRegisterCollBuffers(comm, task, regBufSend, regBufRecv, &regNeedConnect));
  NCCLCHECK(postTuneLegacyEnqueueCollWork(comm, planner, task, regBufSend, regBufRecv, &planner->collWorkQueue));
  planner->nTasksColl += 1;
  ncclIntruQueueEnqueue(&planner->collTaskQueue, task);
  return ncclSuccess;
}

static ncclResult_t postTuneAllGatherVEnqueueBcastTasks(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclRawTaskAllGatherV* agv = &tInfo->raw->allGatherV;
  int nRoots = 0;
  int soleRoot = -1;

  if (tInfo->raw->kind != ncclTaskKindAllGatherV) return ncclInternalError;
  if (agv->func != ncclFuncAllGatherV) return ncclInternalError;

  for (int root = 0; root < agv->nRanks; root++) {
    if (agv->counts[root] == 0) continue;
    nRoots += 1;
    soleRoot = root;
  }
  if (nRoots == 0) goto cleanup;

  NCCLCHECK(postTuneAllGatherVEnsureRingConnected(comm));

  if (nRoots == 1) {
    NCCLCHECK(postTuneAllGatherVEnqueueBroadcastTask(comm, agv, soleRoot));
    if (tInfo->raw != nullptr) {
      ncclMemoryPoolFree(&comm->memPool_ncclRawTask, tInfo->raw);
      tInfo->raw = nullptr;
    }
    return ncclSuccess;
  }

  for (int root = 0; root < agv->nRanks; root++) {
    if (agv->counts[root] == 0) continue;

    struct ncclTaskBcast* t =
      ncclMemoryPoolAlloc<struct ncclTaskBcast>(&comm->memPool_ncclTaskBcast, &comm->memPermanent);
    t->func = ncclFuncAllGatherV;
    t->sendbuff = (root == comm->rank) ? agv->sendbuff : nullptr;
    t->recvbuff = agv->recvbuff[root];
    t->count = agv->counts[root];
    t->datatype = ncclInt8;
    t->root = root;
    t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
    t->groupApiEventHandle = nullptr;
    t->collApiEventHandle = nullptr;
    t->eventHandle = nullptr;

    planner->bcast_info.minBcastPeer = std::min(planner->bcast_info.minBcastPeer, root);
    planner->bcast_info.maxBcastPeer = std::max(planner->bcast_info.maxBcastPeer, root);
    if (ncclIntruQueueEmpty(&planner->peers[root].bcastQueue)) {
      planner->bcast_info.BcastPeers += 1;
    }

    ncclIntruQueueEnqueue(&planner->peers[root].bcastQueue, t);
    planner->nTasksBcast += 1;
  }

cleanup:
  if (tInfo->raw != nullptr) {
    ncclMemoryPoolFree(&comm->memPool_ncclRawTask, tInfo->raw);
    tInfo->raw = nullptr;
  }
  return ncclSuccess;
}

static ncclResult_t postTuneAllGatherVTasks(
  struct ncclComm* comm,
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* allgathervTaskQueue) {
  while (!ncclIntruQueueEmpty(allgathervTaskQueue)) {
    struct ncclTaskTuningInfo* tInfo = ncclIntruQueueDequeue(allgathervTaskQueue);
    NCCLCHECK(postTuneAllGatherVEnqueueBcastTasks(comm, tInfo));
  }
  return ncclSuccess;
}

static ncclResult_t postTuneP2pTasks(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* p2pTaskQueue) {
  bool needPreconnect = false;
  struct ncclKernelPlanner* planner = &comm->planner;

  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(p2pTaskQueue); tInfo != nullptr; tInfo = tInfo->next) {
    struct ncclRawTaskSendRecv* raw = &tInfo->raw->sendRecv;
    bool isSendNotRecv = raw->func == ncclFuncSend;

    NCCLCHECK(postTuneP2pRecordPreconnect(comm, raw->peer, isSendNotRecv, &needPreconnect));
  }

  if (needPreconnect) {
    NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
  }

  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(p2pTaskQueue); tInfo != nullptr; tInfo = tInfo->next) {
    struct ncclTaskP2p* task = ncclMemoryPoolAlloc<struct ncclTaskP2p>(&comm->memPool_ncclTaskP2p, &comm->memPermanent);

    NCCLCHECK(fillP2pTaskFromRaw(comm, tInfo, task));

    bool isSendNotRecv = task->func == ncclFuncSend;

    NCCLCHECK(postTuneP2pRegisterBuffer(comm, task, isSendNotRecv, tInfo->tuningOut.proto));

    ncclIntruQueueEnqueue(isSendNotRecv ? &planner->peers[task->root].sendQueue : &planner->peers[task->root].recvQueue,
                          task);
    planner->nTasksP2p += 1;
    if (isSendNotRecv) planner->nTasksP2pSend += 1;
    else planner->nTasksP2pRecv += 1;
    NCCLCHECK(postTuneSymFreeTuningInfoRaw(comm, tInfo));
  }
  return ncclSuccess;
}

static ncclResult_t postTuneCeFreeTuningInfoRaw(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo) {
  if (tInfo->raw != nullptr) {
    ncclMemoryPoolFree(&comm->memPool_ncclRawTask, tInfo->raw);
    tInfo->raw = nullptr;
  }
  return ncclSuccess;
}

static ncclResult_t postTuneCeCollTaskAppend(struct ncclComm* comm, const struct ncclRawTaskColl* raw,
                                             struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin) {
  struct ncclKernelPlanner* planner = &comm->planner;

  struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);

  t->func = raw->func;
  t->sendbuff = raw->sendbuff;
  t->recvbuff = raw->recvbuff;
  t->count = raw->count;
  t->root = raw->root;
  t->datatype = raw->datatype;
  size_t elementSize = ncclTypeSize(t->datatype);
  if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) {
    t->count *= elementSize;
    t->datatype = ncclInt8;
    elementSize = 1;
  }
  t->trafficBytes = t->count * elementSize * ncclFuncTrafficPerByte(t->func, comm->nRanks);
  t->opHost = raw->opHost;
  t->opDev = raw->opDev; // C++ struct assignment
  t->chunkSteps = 1;
  t->sliceSteps = 1;
  t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
  t->groupApiEventHandle = nullptr;
  t->collApiEventHandle = nullptr;
  t->sendWin = sendWin;
  t->recvWin = recvWin;

  ncclIntruQueueEnqueue(&planner->collCeTaskQueue, t);

  return ncclSuccess;
}

/*
 * Materialize tuned copy-engine collectives into CE scheduler inputs.
 */
static ncclResult_t postTuneCeTasks(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* ceTaskQueue) {
  NCCLCHECK(postTuneCeTasksLazyInit(comm, ceTaskQueue));

  while (!ncclIntruQueueEmpty(ceTaskQueue)) {
    struct ncclTaskTuningInfo* tInfo = ncclIntruQueueDequeue(ceTaskQueue);
    struct ncclDevrWindow* sendWin = nullptr;
    struct ncclDevrWindow* recvWin = nullptr;
    struct ncclRawTaskColl* raw = &tInfo->raw->coll;

    NCCLCHECK(ncclDevrFindWindow(comm, raw->sendbuff, &sendWin));
    NCCLCHECK(ncclDevrFindWindow(comm, raw->recvbuff, &recvWin));
    NCCLCHECK(postTuneCeCollTaskAppend(comm, raw, sendWin, recvWin));
    NCCLCHECK(postTuneCeFreeTuningInfoRaw(comm, tInfo));
  }
  return ncclSuccess;
}

static void postTuningAccumulateTime(struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                     float* totalTimeUs) {
  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(queue); tInfo != nullptr; tInfo = tInfo->next) {
    if (tInfo->tuningOut.valid) *totalTimeUs += tInfo->tuningOut.timeUs;
  }
}

static ncclResult_t postTuningSimulation(struct ncclComm* comm, struct ncclClassifiedTaskQueues* ctq,
                                         ncclSimInfo_t* simInfo) {
  float totalTimeUs = 0.0f;

  (void)comm;
  postTuningAccumulateTime(&ctq->symTaskQueue, &totalTimeUs);
  postTuningAccumulateTime(&ctq->legacyTaskQueue, &totalTimeUs);
  postTuningAccumulateTime(&ctq->p2pTaskQueue, &totalTimeUs);
  postTuningAccumulateTime(&ctq->rmaTaskQueue, &totalTimeUs);
  postTuningAccumulateTime(&ctq->ceTaskQueue, &totalTimeUs);

  simInfo->estimatedTime = totalTimeUs;
  return ncclSuccess;
}

ncclResult_t ncclTaskPostTuning(struct ncclComm* comm, struct ncclClassifiedTaskQueues* ctq, ncclSimInfo_t* simInfo) {
  if (comm == nullptr || ctq == nullptr) return ncclInvalidArgument;
  if (simInfo != nullptr) {
    NCCLCHECK(postTuningSimulation(comm, ctq, simInfo));
    return ncclSuccess;
  }

  // Traverse all task families in the order expected by launch preparation.
  NCCLCHECK(postTuneSymTasks(comm, &ctq->symTaskQueue));
  NCCLCHECK(postTuneLegacyTasks(comm, &ctq->legacyTaskQueue));
  NCCLCHECK(postTuneAllGatherVTasks(comm, &ctq->allgathervTaskQueue));
  NCCLCHECK(postTuneP2pTasks(comm, &ctq->p2pTaskQueue));
  NCCLCHECK(postTuneRmaTasks(comm, &ctq->rmaTaskQueue));
  NCCLCHECK(postTuneCeTasks(comm, &ctq->ceTaskQueue));

  return ncclSuccess;
}
