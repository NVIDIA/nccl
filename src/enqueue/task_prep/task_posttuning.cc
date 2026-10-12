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

static void setChunkSteps(struct ncclTaskColl* task) {
  task->chunkSteps = ncclDefaultChunkStep(task->func);
  task->sliceSteps = ncclDefaultSliceStep(task->func);
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
  if (task->func == ncclFuncAllGather || task->func == ncclFuncBroadcast || task->func == ncclFuncAlltoAll) {
    task->count *= elementSize;
    task->datatype = ncclInt8;
    elementSize = 1;
  }
  task->trafficBytes = task->count * elementSize * ncclFuncTrafficPerByte(task->func, comm->nRanks);
  task->opHost = raw->opHost;
  task->opDev = raw->opDev;
  setChunkSteps(task);
  task->launchCompletionEvent = ncclCollConfigGetLaunchCompletionEvent(&tInfo->raw->collConfig);
  ncclTaskCaptureProfilerCtx(task, tInfo->raw->profiler);

  const ncclCollConfig_t* cfg = &tInfo->raw->collConfig;
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
  task->algMask = raw->algMask;
  task->forceAlgSelection = cfg->forceAlgSelection;

  return ncclSuccess;
}

static ncclResult_t freeRaw(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo) {
  if (tInfo->raw != nullptr) {
    // An AllGatherV owns the table of broadcasts it was merged from.
    if (tInfo->raw->kind == ncclTaskKindAllGatherV) {
      delete tInfo->raw->allGatherV.bcastTasks;
      tInfo->raw->allGatherV.bcastTasks = nullptr;
    }
    ncclMemoryPoolFree(&comm->memPool_ncclRawTask, tInfo->raw);
    tInfo->raw = nullptr;
  }
  return ncclSuccess;
}

static ncclResult_t p2pChannelBase(struct ncclComm* comm, int peer, bool isSendNotRecv, uint8_t* baseOut) {
  int round = 0;
  while (peer != (isSendNotRecv ? comm->p2pSchedule[round].sendRank : comm->p2pSchedule[round].recvRank)) {
    round += 1;
  }
  *baseOut = ncclP2pChannelBaseForRound(comm, round);
  return ncclSuccess;
}

static ncclResult_t recordP2pPreconnect(struct ncclComm* comm, int peer, bool isSendNotRecv, bool* needPreconnect) {
  struct ncclKernelPlanner* planner = &comm->planner;
  uint8_t base;

  if (peer < 0 || peer >= comm->nRanks) return ncclInvalidArgument;
  if (comm->rank == peer) return ncclSuccess;
  if (isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen) return ncclSuccess;

  NCCLCHECK(p2pChannelBase(comm, peer, isSendNotRecv, &base));

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
  task->launchCompletionEvent = raw->launchCompletionEvent;
  if (task->collAPI == ncclFuncAlltoAll || task->collAPI == ncclFuncScatter || task->collAPI == ncclFuncGather) {
    task->allowUB = false;
  } else {
    task->allowUB = true;
  }
  ncclTaskCaptureProfilerCtx(task, tInfo->raw->profiler);
  return ncclSuccess;
}

static ncclResult_t registerP2pBuffer(struct ncclComm* comm, struct ncclTaskP2p* task, bool isSendNotRecv,
                                      int protocol) {
  constexpr int connIndex = 1;
  struct ncclKernelPlanner* planner = &comm->planner;
  int peer = task->root;
  uint8_t base;
  bool network;
  bool proxySameProcess;

  if (protocol != NCCL_PROTO_SIMPLE) return ncclSuccess;
  if (!task->allowUB || task->bytes == 0 || task->buff == nullptr || peer == comm->rank) return ncclSuccess;

  NCCLCHECK(p2pChannelBase(comm, peer, isSendNotRecv, &base));

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

static ncclResult_t debugTasks(struct ncclComm* comm) {
  while (!ncclIntruQueueEmpty(&comm->argsInfoQueue)) {
    struct ncclArgsInfo* argsInfo = ncclIntruQueueDequeue(&comm->argsInfoQueue);
    ncclResult_t ret = ncclArgsGlobalCheck(argsInfo);
    free(argsInfo);
    NCCLCHECK(ret);
  }
  return ncclSuccess;
}

static ncclResult_t initSym(struct ncclComm* comm,
                            struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* symTaskQueue) {
  if (ncclIntruQueueEmpty(symTaskQueue)) return ncclSuccess;
  return ncclSymkInitOnce(comm);
}

static ncclResult_t initCe(struct ncclComm* comm,
                           struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* ceTaskQueue) {
  if (ncclIntruQueueEmpty(ceTaskQueue)) return ncclSuccess;
  if (comm->ceColl.initialized) return ncclSuccess;
  return ncclCeInit(comm);
}

static ncclResult_t preSchedSymTasks(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* symTaskQueue) {
  NCCLCHECK(initSym(comm, symTaskQueue));
  NCCLCHECK(debugTasks(comm));
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

static ncclResult_t registerCollBuffers(struct ncclComm* comm, struct ncclTaskColl* task,
                                        void* regBufSend[NCCL_MAX_LOCAL_RANKS], void* regBufRecv[NCCL_MAX_LOCAL_RANKS],
                                        bool* regNeedConnect) {
  struct ncclKernelPlanner* planner = &comm->planner;

  *regNeedConnect = true;
  NCCLCHECK(ncclRegisterCollBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, regNeedConnect));
  return ncclSuccess;
}

static ncclResult_t registerNvlsCollBuffers(struct ncclComm* comm, struct ncclTaskColl* task,
                                            void* regBufSend[NCCL_MAX_LOCAL_RANKS],
                                            void* regBufRecv[NCCL_MAX_LOCAL_RANKS], bool* regNeedConnect) {
  struct ncclKernelPlanner* planner = &comm->planner;

  *regNeedConnect = true;
  NCCLCHECK(ncclRegisterCollNvlsBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue,
                                        regNeedConnect));
  return ncclSuccess;
}

static ncclResult_t enqueueCollWork(struct ncclComm* comm, struct ncclKernelPlanner* planner, struct ncclTaskColl* task,
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

static ncclResult_t recordAlgoNeedConnect(struct ncclComm* comm, int algorithm, bool regNeedConnect,
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

static ncclResult_t preSchedLegacyTasks(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* legacyTaskQueue) {
  bool needConnect = false;
  bool algoNeedConnect[NCCL_NUM_ALGORITHMS];
  struct ncclKernelPlanner* planner = &comm->planner;

  // Buffer registration below gates graph registration on planner->persistent, so establish it
  // here; the legacy launcher only sets it later in ncclLaunchPrepare.
  planner->persistent = ncclCudaGraphValid(planner->capturingGraph);

  // Bin tasks (and their work) by scheduling kind = 2*isCollnet + isNvls, then concatenate into
  // the planner queues in ascending kind order. scheduleCollTasksToPlan restarts channel assignment
  // on every kind change, so interleaved kinds (e.g. standard and collnet) would imbalance channels.
  // Mirrors legacy ncclPrepareTasks (collBins[isCollnet][isNvls]). collTaskQueue and collWorkQueue
  // are consumed in lockstep, so both are binned and concatenated identically.
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> taskBins[4] = {};
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workBins[4] = {};

  memset(algoNeedConnect, 0, sizeof(algoNeedConnect));

  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(legacyTaskQueue); tInfo != nullptr; tInfo = tInfo->next) {
    if (tInfo->tuningOut.algo == NCCL_ALGO_NVLS || tInfo->tuningOut.algo == NCCL_ALGO_NVLS_TREE) {
      struct ncclTaskColl* task =
        ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
      void* regBufSend[NCCL_MAX_LOCAL_RANKS];
      void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
      bool regNeedConnect = true;

      NCCLCHECK(applyTuningToCollTask(comm, tInfo, task));
      NCCLCHECK(registerNvlsCollBuffers(comm, task, regBufSend, regBufRecv, &regNeedConnect));
      NCCLCHECK(recordAlgoNeedConnect(comm, tInfo->tuningOut.algo, regNeedConnect, algoNeedConnect, &needConnect));
      int kind = 2 * task->isCollnet + task->isNvls;
      NCCLCHECK(enqueueCollWork(comm, planner, task, regBufSend, regBufRecv, &workBins[kind]));
      planner->nTasksColl += 1;
      ncclIntruQueueEnqueue(&taskBins[kind], task);
    } else {
      bool regNeedConnect = true;
      NCCLCHECK(recordAlgoNeedConnect(comm, tInfo->tuningOut.algo, regNeedConnect, algoNeedConnect, &needConnect));
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
    NCCLCHECK(registerCollBuffers(comm, task, regBufSend, regBufRecv, &regNeedConnect));
    int kind = 2 * task->isCollnet + task->isNvls;
    NCCLCHECK(enqueueCollWork(comm, planner, task, regBufSend, regBufRecv, &workBins[kind]));
    planner->nTasksColl += 1;
    ncclIntruQueueEnqueue(&taskBins[kind], task);
  }

  // Concatenate the per-kind bins in ascending kind order (isCollnet outer, isNvls inner) so that
  // same-kind tasks -- and their matching work nodes -- stay contiguous and in lockstep.
  for (int kind = 0; kind < 4; kind++) {
    ncclIntruQueueTransfer(&planner->collTaskQueue, &taskBins[kind]);
    ncclIntruQueueTransfer(&planner->collWorkQueue, &workBins[kind]);
  }

  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(legacyTaskQueue); tInfo != nullptr; tInfo = tInfo->next) {
    NCCLCHECK(freeRaw(comm, tInfo));
  }
  return ncclSuccess;
}

static ncclResult_t ensureRingConnected(struct ncclComm* comm) {
  bool algoNeedConnect[NCCL_NUM_ALGORITHMS];

  if (!comm->runtimeConn || comm->initAlgoChannels[NCCL_ALGO_RING]) return ncclSuccess;

  memset(algoNeedConnect, 0, sizeof(algoNeedConnect));
  comm->initAlgoChannels[NCCL_ALGO_RING] = true;
  algoNeedConnect[NCCL_ALGO_RING] = true;
  NCCLCHECK(ncclCollPreconnect(comm, algoNeedConnect));
  return ncclSuccess;
}

static ncclResult_t enqueueBroadcastTask(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo, int root,
                                         struct ncclRawTaskAllGatherVBcast* bcast) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* task =
    ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
  void* regBufSend[NCCL_MAX_LOCAL_RANKS];
  void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
  bool regNeedConnect = true;
  int collNetSupport = 0;
  int nvlsTransportEnabled = 0;

  memset(task, 0, sizeof(*task));
  task->func = ncclFuncBroadcast;
  task->sendbuff = bcast->sendbuff;
  task->recvbuff = bcast->recvbuff;
  task->count = bcast->count;
  task->root = root;
  task->datatype = ncclInt8;
  task->trafficBytes = task->count * ncclFuncTrafficPerByte(task->func, comm->nRanks);
  setChunkSteps(task);
  ncclTaskCaptureProfilerCtx(task, bcast->profiler);
  task->opHost = ncclSum;
  // Carry over the AllGatherV's resolved channel budget. scheduleCollTasksToPlan clamps
  // nMaxChannels by these caps unconditionally, so memset-zeroed caps would clamp it to 0.
  // ncclGetAlgoInfo below keeps the seeded nMaxChannels when its own tuning yields none.
  task->nMaxChannels = tInfo->tuningOut.maxChannels;
  task->minCTAs = tInfo->tuningIn.minCTAs;
  task->maxCTAs = tInfo->tuningIn.maxCTAs;
  task->nvlsCTAs = comm->config.nvlsCTAs;
  task->cgaClusterSize = comm->config.cgaClusterSize;

  NCCLCHECK(ncclGetCollNetSupport(comm, task, &collNetSupport));
  nvlsTransportEnabled = ncclNvlsTransportEnabled(comm) &&
                         (ncclNvlsSupported(task->opDev.op, task->datatype) || task->func == ncclFuncAllGather);
  NCCLCHECK(ncclGetAlgoInfo(comm, task, collNetSupport, nvlsTransportEnabled, 1, nullptr));
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

  NCCLCHECK(registerCollBuffers(comm, task, regBufSend, regBufRecv, &regNeedConnect));
  NCCLCHECK(enqueueCollWork(comm, planner, task, regBufSend, regBufRecv, &planner->collWorkQueue));
  planner->nTasksColl += 1;
  ncclIntruQueueEnqueue(&planner->collTaskQueue, task);
  return ncclSuccess;
}

static ncclResult_t preSchedAllGatherVTasks(
  struct ncclComm* comm,
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* allgathervTaskQueue) {
  struct ncclKernelPlanner* planner = &comm->planner;

  while (!ncclIntruQueueEmpty(allgathervTaskQueue)) {
    struct ncclTaskTuningInfo* tInfo = ncclIntruQueueDequeue(allgathervTaskQueue);
    struct ncclRawTaskAllGatherV* agv;
    int nRoots;

    if (tInfo->raw == nullptr || tInfo->raw->kind != ncclTaskKindAllGatherV) return ncclInternalError;
    agv = &tInfo->raw->allGatherV;
    nRoots = (agv->bcastTasks != nullptr) ? (int)agv->bcastTasks->size() : 0;
    if (nRoots == 0) goto cleanup;

    NCCLCHECK(ensureRingConnected(comm));

    if (nRoots == 1) {
      auto it = agv->bcastTasks->begin();
      NCCLCHECK(enqueueBroadcastTask(comm, tInfo, it->first, &it->second));
      goto cleanup;
    }

    for (auto& kv : *agv->bcastTasks) {
      int root = kv.first;
      struct ncclRawTaskAllGatherVBcast* bcast = &kv.second;
      struct ncclTaskBcast* t =
        ncclMemoryPoolAlloc<struct ncclTaskBcast>(&comm->memPool_ncclTaskBcast, &comm->memPermanent);
      t->func = ncclFuncAllGatherV;
      t->sendbuff = bcast->sendbuff;
      t->recvbuff = bcast->recvbuff;
      t->count = bcast->count;
      t->datatype = ncclInt8;
      t->root = root;
      ncclTaskCaptureProfilerCtx(t, bcast->profiler);
      t->eventHandle = {};

      planner->bcast_info.minBcastPeer = std::min(planner->bcast_info.minBcastPeer, root);
      planner->bcast_info.maxBcastPeer = std::max(planner->bcast_info.maxBcastPeer, root);
      if (ncclIntruQueueEmpty(&planner->peers[root].bcastQueue)) {
        planner->bcast_info.BcastPeers += 1;
      }

      ncclIntruQueueEnqueue(&planner->peers[root].bcastQueue, t);
      planner->nTasksBcast += 1;
    }

  cleanup:
    NCCLCHECK(freeRaw(comm, tInfo));
  }
  return ncclSuccess;
}

static ncclResult_t preSchedP2pTasks(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* p2pTaskQueue) {
  bool needPreconnect = false;
  struct ncclKernelPlanner* planner = &comm->planner;

  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(p2pTaskQueue); tInfo != nullptr; tInfo = tInfo->next) {
    struct ncclRawTaskSendRecv* raw = &tInfo->raw->sendRecv;
    bool isSendNotRecv = raw->func == ncclFuncSend;

    NCCLCHECK(recordP2pPreconnect(comm, raw->peer, isSendNotRecv, &needPreconnect));
  }

  if (needPreconnect) {
    NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
  }

  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(p2pTaskQueue); tInfo != nullptr; tInfo = tInfo->next) {
    struct ncclTaskP2p* task = ncclMemoryPoolAlloc<struct ncclTaskP2p>(&comm->memPool_ncclTaskP2p, &comm->memPermanent);

    NCCLCHECK(fillP2pTaskFromRaw(comm, tInfo, task));

    bool isSendNotRecv = task->func == ncclFuncSend;

    NCCLCHECK(registerP2pBuffer(comm, task, isSendNotRecv, tInfo->tuningOut.proto));

    ncclIntruQueueEnqueue(isSendNotRecv ? &planner->peers[task->root].sendQueue : &planner->peers[task->root].recvQueue,
                          task);
    planner->nTasksP2p += 1;
    if (isSendNotRecv) planner->nTasksP2pSend += 1;
    else planner->nTasksP2pRecv += 1;
    NCCLCHECK(freeRaw(comm, tInfo));
  }
  return ncclSuccess;
}

/*
 * Initialize copy-engine collectives when any CE tasks are present.
 */
static ncclResult_t preSchedCeTasks(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* ceTaskQueue) {
  NCCLCHECK(initCe(comm, ceTaskQueue));
  return ncclSuccess;
}

static void accumulateTime(struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                           float* totalTimeUs) {
  for (struct ncclTaskTuningInfo* tInfo = ncclIntruQueueHead(queue); tInfo != nullptr; tInfo = tInfo->next) {
    if (tInfo->tuningOut.valid) *totalTimeUs += tInfo->tuningOut.timeUs;
  }
}

// The group estimate is the sum of the per-task estimates, assuming the tasks run one after
// another, so it covers every operation in the group, as the ncclSimInfo_t documentation
// describes.
static ncclResult_t taskSimulate(struct ncclClassifiedTaskQueues* ctq, ncclSimInfo_t* simInfo) {
  float totalTimeUs = 0.0f;

  accumulateTime(&ctq->queues[ncclTaskQueueSym], &totalTimeUs);
  accumulateTime(&ctq->queues[ncclTaskQueueLegacy], &totalTimeUs);
  accumulateTime(&ctq->queues[ncclTaskQueueP2p], &totalTimeUs);
  accumulateTime(&ctq->queues[ncclTaskQueueRma], &totalTimeUs);
  accumulateTime(&ctq->queues[ncclTaskQueueCe], &totalTimeUs);

  simInfo->estimatedTime = totalTimeUs;
  return ncclSuccess;
}

// A simulation stops before pre-scheduling, which is where raw tasks are normally released.
static ncclResult_t releaseSimulatedTasks(struct ncclComm* comm, struct ncclClassifiedTaskQueues* ctq) {
  for (int type = 0; type < ncclTaskQueueNum; type++) {
    while (!ncclIntruQueueEmpty(&ctq->queues[type])) {
      NCCLCHECK(freeRaw(comm, ncclIntruQueueDequeue(&ctq->queues[type])));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTaskPostTuning(struct ncclComm* comm, struct ncclClassifiedTaskQueues* ctq, ncclSimInfo_t* simInfo) {
  if (comm == nullptr || ctq == nullptr) return ncclInvalidArgument;
  if (simInfo != nullptr) {
    if (simInfo != reinterpret_cast<ncclSimInfo_t*>(NCCL_COMM_SIMULATION_SKIP)) {
      NCCLCHECK(taskSimulate(ctq, simInfo));
    }
    NCCLCHECK(releaseSimulatedTasks(comm, ctq));
    return ncclSuccess;
  }

  // Traverse all task families in the order expected by launch preparation.
  NCCLCHECK(preSchedSymTasks(comm, &ctq->queues[ncclTaskQueueSym]));
  NCCLCHECK(preSchedCeTasks(comm, &ctq->queues[ncclTaskQueueCe]));
  NCCLCHECK(preSchedLegacyTasks(comm, &ctq->queues[ncclTaskQueueLegacy]));
  NCCLCHECK(preSchedAllGatherVTasks(comm, &ctq->queues[ncclTaskQueueAllGatherV]));
  NCCLCHECK(preSchedP2pTasks(comm, &ctq->queues[ncclTaskQueueP2p]));

  return ncclSuccess;
}
