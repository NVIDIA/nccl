/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "gin.h"
#include "param.h"
#include "graph.h"
#include "transport.h"
#include "register_inline.h"
#include "gin/gin_host.h"
#include "gin/gin_host_proxy.h"
#include "compiler.h"
#include <cmath>

NCCL_PARAM(GinEnable, "GIN_ENABLE", 1);
NCCL_PARAM(GinSignalPoolSize, "GIN_SIGNAL_POOL_SIZE", 512 << 10);
NCCL_PARAM(GinCounterPoolSize, "GIN_COUNTER_POOL_SIZE", 512 << 10);

ncclResult_t getGlobalGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) {
    return ncclInternalError;
  }

  if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) {
    *ginType = NCCL_GIN_TYPE_NONE;
    return ncclSuccess;
  }

  *ginType = comm->sharedRes->ginState.ginType;
  return ncclSuccess;
}

ncclResult_t getGlobalRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) {
    return ncclInternalError;
  }

  if (comm->globalGinSupport == NCCL_GIN_CONNECTION_NONE) {
    *ginType = NCCL_GIN_TYPE_NONE;
    return ncclSuccess;
  }
  *ginType = comm->sharedRes->ginState.ginType;
  return ncclSuccess;
}

ncclResult_t setLocalGinType(struct ncclComm* comm) {
  if (comm == nullptr || comm->sharedRes->ginState.ncclGin == nullptr) {
    return ncclInternalError;
  }
  ncclGinState& ginState = comm->sharedRes->ginState;
  ginState.ginType = NCCL_GIN_TYPE_NONE;

  if (!ncclParamGinEnable()) {
    return ncclSuccess;
  }

  ncclNetProperties_t props;
  NCCLCHECK(ginState.ncclGin->getProperties(0, &props));
  if (props.netDeviceType == NCCL_NET_DEVICE_GIN_PROXY ||
      props.netDeviceType == NCCL_NET_DEVICE_GIN_GDAKI) {
    // NOTE: The following cast is valid because ncclGinType_t variant values
    // should match NCCL_NET_DEVICE_GIN_* values from `enum ncclNetDeviceType`.
    ginState.ginType = static_cast<ncclGinType_t>(props.netDeviceType);
    return ncclSuccess;
  }
  WARN("Cannot get gin type: ncclGin is not null but net device type (%d) is not a gin type",
       props.netDeviceType);
  return ncclInternalError;
}

void* ncclGinProgress(struct ncclGinState* ginState_) {
  struct ncclGinState* ginState = (struct ncclGinState*)ginState_;
  while (1) {
    std::unique_lock<std::mutex> lock(ginState->mutex);
    if (ginState->ginProgress == 1) {
      lock.unlock();
      for (int n=0; n<ginState->ginCommCount; n++) {
        ncclResult_t ret;
        if (ginState->ginType == NCCL_GIN_TYPE_PROXY) {
          ret = ncclGinProxyProgress(ginState->ncclGin, ginState->ginCtx[n]);
        } else {
          ret = ginState->ncclGin->ginProgress(ginState->ginComms[n]);
        }
        if (ret != ncclSuccess) {
          COMPILER_ATOMIC_STORE(&ginState->asyncResult, ret, std::memory_order_release);
          INFO(NCCL_ALL,"%s:%d -> %d [GIN Progress Thread]", __FILE__, __LINE__, ret);
          ginState->ginProgress = -2;
          return NULL;
        }
      }
      std::this_thread::yield();
    } else if (ginState->ginProgress == -1) {
      return NULL;
    } else if (ginState->ginProgress == 0) {
      ginState->cond.wait(lock);
    } else {
      INFO(NCCL_ALL,"%s:%d -> [GIN Progress Thread] state unknown %d", __FILE__, __LINE__, ginState->ginProgress);
      ginState->ginProgress = -2;
      return NULL;
    }
  }
}

static ncclRequirementFlagOptions_t parseRequirementFlagOption(int reqFlagOption) {
  if (reqFlagOption >= NCCL_REQUIREMENT_FLAG_OPTION_NOT_REQUIRED && reqFlagOption <= NCCL_REQUIREMENT_FLAG_OPTION_REQUIRED) {
    return (ncclRequirementFlagOptions_t)reqFlagOption;
  }
  return NCCL_REQUIREMENT_FLAG_OPTION_NOT_REQUIRED;
}

NCCL_PARAM(GinNconnections, "GIN_NCONNECTIONS", -2);
NCCL_PARAM(GinNcontexts, "GIN_NCONTEXTS", -1);

ncclResult_t ncclGinConnectOnce(struct ncclComm* comm, ncclGinConnectionType_t requestedConnectionType, int reqGinContextCount, int reqGinQueueDepth, int reqGinUseReliableDB, int reqGinUseExpertControl) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (ginState->connected) return ncclSuccess;

  ncclResult_t ret = ncclSuccess;
  if (ncclParamGinEnable() == 0) {
    WARN("GIN is disabled.");
    return ncclInternalError;
  }

  // Load plugin
  if (ginState->ncclGin == NULL) {
    WARN("GIN not supported.");
    return ncclInvalidUsage;
  }

  ginState->ginConnectionType = requestedConnectionType;
  ginState->ginInstance = comm->ginContext;

  int ndev = 0;
  NCCLCHECK(ginState->ncclGin->devices(&ndev));
  if (ndev <= 0) {
    WARN("No GIN-capable devices found.");
    return ncclInternalError;
  }

  if (!comm->symmetricSupport) {
    WARN("Communicator does not support symmetric memory!");
    return ncclInternalError;
  }

  int nLocalGinDevs;
  int localGinDevs[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(ncclTopoGetLocalGinDevs(comm, localGinDevs, &nLocalGinDevs));

  void** handles = NULL;
  char* allHandles = NULL;

  int* ginCommCountHandles = NULL;
  int nContextsTotal;
  int nContextsPerComm;

  if (reqGinQueueDepth == 0)
    reqGinQueueDepth = ginState->ginQueueDepth;
  ginState->ginQueueDepth = reqGinQueueDepth;

  if (reqGinUseReliableDB < NCCL_REQUIREMENT_FLAG_OPTION_NOT_REQUIRED)
    reqGinUseReliableDB = ginState->ginUseReliableDB;
  ginState->ginUseReliableDB = parseRequirementFlagOption(reqGinUseReliableDB);

  if (reqGinUseExpertControl < NCCL_REQUIREMENT_FLAG_OPTION_NOT_REQUIRED)
    reqGinUseExpertControl = ginState->ginUseExpertControl;
  ginState->ginUseExpertControl = parseRequirementFlagOption(reqGinUseExpertControl);

  NCCLCHECKGOTO(ncclCalloc(&ginCommCountHandles, comm->nRanks), ret, fail);

  ginState->ginCommCount = nLocalGinDevs;
  if (ncclParamGinNconnections() != -2) ginState->ginCommCount = ncclParamGinNconnections();
  ginState->ginCommCount = std::min<int>(NCCL_GIN_MAX_CONNECTIONS, ginState->ginCommCount);

  ginCommCountHandles[comm->rank] = ginState->ginCommCount;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
  for (int r = 0; r < comm->nRanks; r++) {
    ginState->ginCommCount = std::min(ginState->ginCommCount, ginCommCountHandles[r]);
  }

  nContextsTotal = ncclParamGinNcontexts();
  if (nContextsTotal <= 0) {
    nContextsTotal = std::max(reqGinContextCount, NCCL_GIN_MAX_CONNECTIONS);
  }
  nContextsTotal = ROUNDUP(nContextsTotal, ginState->ginCommCount);
  nContextsPerComm = nContextsTotal / ginState->ginCommCount;
  ginState->ginContextCount = nContextsTotal;
  ginState->ctxFirstAvailable = 0;
  ginState->ctxLastExclusive = nContextsTotal;
  INFO(NCCL_INIT, "devCommCreate: %d Local NET, creating %d GIN connections with %d contexts each (%d contexts total requested)", nLocalGinDevs, ginState->ginCommCount, nContextsPerComm, reqGinContextCount);

  NCCLCHECKGOTO(ncclCalloc(&allHandles, (size_t)comm->nRanks * NCCL_NET_HANDLE_MAXSIZE), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&handles, comm->nRanks), ret, fail);

  int nGinRanks;
  int myGinRank;
  if (requestedConnectionType == NCCL_GIN_CONNECTION_FULL) {
    nGinRanks = comm->nRanks;
    myGinRank = comm->rank;
    for (int r = 0; r < nGinRanks; r++) {
      handles[r] = allHandles + r * NCCL_NET_HANDLE_MAXSIZE;
    }
  } else {
    ncclTeam_t railTeam = ncclTeamRail(comm);
    nGinRanks = railTeam.nRanks;
    myGinRank = railTeam.rank;
    for (int r = 0; r < nGinRanks; r++) {
      int worldRank = ncclTeamRankToWorld(comm, railTeam, r);
      handles[r] = allHandles + worldRank * NCCL_NET_HANDLE_MAXSIZE;
    }
  }

  ginState->signalSpaceSize = ncclParamGinSignalPoolSize();
  if (ginState->signalSpaceSize < 0 || (1 << 30) <= ginState->signalSpaceSize) {
    WARN("NCCL_GIN_SIGNAL_POOL_SIZE has invalid value.");
    ginState->signalSpaceSize = 64 << 10;
  }
  ginState->counterSpaceSize = ncclParamGinCounterPoolSize();
  if (ginState->counterSpaceSize < 0 || (1 << 30) <= ginState->counterSpaceSize) {
    WARN("NCCL_GIN_COUNTER_POOL_SIZE has invalid value.");
    ginState->counterSpaceSize = 64 << 10;
  }

  for (int n = 0; n < ginState->ginCommCount; n++) {
    void* listenComm;
    NCCLCHECKGOTO(
      ginState->ncclGin->listen(ginState->ginInstance, localGinDevs[n%nLocalGinDevs],
                                allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm),
      ret, fail);
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret,
                  fail);
    NCCLCHECKGOTO(
            ginState->ncclGin->connect(comm->ginContext, handles, nGinRanks, myGinRank,
                nContextsPerComm, ginState->ginQueueDepth, static_cast<ncclGinRequirementFlagOptions_v12_t>(ginState->ginUseReliableDB),
                static_cast<ncclGinRequirementFlagOptions_v12_t>(ginState->ginUseExpertControl),
                listenComm, ginState->ginComms + n),
      ret, fail);
    if (ginState->ginType == NCCL_GIN_TYPE_PROXY) {
      NCCLCHECKGOTO(ncclGinProxyCreateContext(comm, ginState->ginComms[n],
                                              localGinDevs[n % nLocalGinDevs], ginState->signalSpaceSize,
                                              ginState->counterSpaceSize, nContextsPerComm,
                                              &ginState->ginCtx[n], &ginState->ginDevHandles[n]),
                    ret, fail);
    } else {
      NCCLCHECKGOTO(ginState->ncclGin->createContext(
                      ginState->ginComms[n], ginState->signalSpaceSize, ginState->counterSpaceSize,
                      nContextsPerComm, &ginState->ginCtx[n], &ginState->ginDevHandles[n]),
                    ret, fail);
    }
    NCCLCHECKGOTO(ginState->ncclGin->closeListen(listenComm), ret, fail);
  }
  free(handles);
  handles = NULL;
  free(allHandles);
  allHandles = NULL;
  free(ginCommCountHandles);
  ginCommCountHandles = NULL;

  // Check whether we need proxy progress and if so, start / wake up the progress thread.
  ginState->needsProxyProgress = 0;
  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginState->ginDevHandles[n]->needsProxyProgress) ginState->needsProxyProgress = 1;
  }
  if (ginState->needsProxyProgress) {
    ginState->ginProgress = 1;
    ginState->thread = std::thread(ncclGinProgress, ginState);
    ncclSetThreadName(ginState->thread, "NCCL GIN Progress%2d", comm->cudaDev);
  }

  ncclSpaceConstruct(&ginState->counterSpace);
  ncclSpaceConstruct(&ginState->signalSpace);

exit:
  if (ret == ncclSuccess) ginState->connected = true;
  return ret;
fail:
  if (allHandles)
    free(allHandles);
  if (handles)
    free(handles);
  if (ginCommCountHandles)
    free(ginCommCountHandles);
  goto exit;
}

ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (!ginState->connected) return ncclSuccess;

  if (ginState->needsProxyProgress) {
    {
      std::lock_guard<std::mutex> lock(ginState->mutex);
      comm->sharedRes->ginState.ginProgress = -1;
      ginState->cond.notify_one();
    }
    ginState->thread.join();
  }

  if (ginState->ginType == NCCL_GIN_TYPE_PROXY) {
    for (int n = 0; n < ginState->ginCommCount; n++) {
      if (ginState->ginCtx[n] != NULL) {
        NCCLCHECK(ncclGinProxyDestroyContext(ginState->ncclGin, ginState->ginCtx[n]));
        ginState->ginCtx[n] = NULL;
      }
    }
  }

  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginState->ginCtx[n] != NULL) {
      NCCLCHECK(ginState->ncclGin->destroyContext(ginState->ginCtx[n]));
      ginState->ginCtx[n] = NULL;
    }
    if (ginState->ginComms[n] != NULL) {
      NCCLCHECK(ginState->ncclGin->closeColl(ginState->ginComms[n]));
      ginState->ginComms[n] = NULL;
    }
  }
  memset((void*)ginState, 0, sizeof(*ginState));
  return ncclSuccess;
}

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  int mrFlags = (winFlags & NCCL_WIN_STRICT_ORDERING) ? NCCL_NET_MR_FLAG_FORCE_SO : 0;
  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginState->ginType == NCCL_GIN_TYPE_PROXY) {
      NCCLCHECK(ncclGinProxyRegister(ginState->ncclGin, ginState->ginCtx[n], address, size,
                                     NCCL_PTR_CUDA, mrFlags, &ginHostWins[n], &ginDevWins[n]));
    } else {
      NCCLCHECK(ginState->ncclGin->regMrSym(ginState->ginComms[n], address, size, NCCL_PTR_CUDA, mrFlags,
                                            &ginHostWins[n], &ginDevWins[n]));
    }
    if (ginHostWins[n] == NULL) {
      WARN("rank %d - GIN Symmetric register failed: buff %p, size %ld", comm->rank, address, size);
      return ncclSystemError;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginState->ginType == NCCL_GIN_TYPE_PROXY) {
      NCCLCHECK(ncclGinProxyDeregister(ginState->ncclGin, ginState->ginCtx[n], ginHostWins[n]));
    } else {
      NCCLCHECK(ginState->ncclGin->deregMrSym(ginState->ginComms[n], ginHostWins[n]));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinAllocSignalsCounters(struct ncclComm* comm, int nSignals, uint32_t* outSignal0,
                                         int nCounters, uint32_t* outCounter0) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  int64_t start;
  if (nSignals != 0) {
    NCCLCHECKGOTO(
      ncclSpaceAlloc(&ginState->signalSpace, ginState->signalSpaceSize, nSignals, 1, &start), ret,
      fail);
    *outSignal0 = (uint32_t)start;
  }
  if (nCounters != 0) {
    NCCLCHECKGOTO(
      ncclSpaceAlloc(&ginState->counterSpace, ginState->counterSpaceSize, nCounters, 1, &start),
      ret, fail_signals);
    *outCounter0 = (uint32_t)start;
  }
  return ncclSuccess;
fail_signals:
  if (nSignals != 0) ncclSpaceFree(&ginState->signalSpace, *outSignal0, nSignals);
fail:
  return ret;
}

ncclResult_t ncclGinFreeSignalsCounters(struct ncclComm* comm, uint32_t signal0, int nSignals,
                                        uint32_t counter0, int nCounters) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (nSignals != 0) ncclSpaceFree(&ginState->signalSpace, signal0, nSignals);
  if (nCounters != 0) ncclSpaceFree(&ginState->counterSpace, counter0, nCounters);
  return ncclSuccess;
}

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError) {
  bool hasError_ = false;
  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginState->ginType == NCCL_GIN_TYPE_PROXY)
      NCCLCHECK(ncclGinProxyQueryLastError(ginState->ncclGin, ginState->ginCtx[n], &hasError_));
    else
      NCCLCHECK(ginState->ncclGin->queryLastError(ginState->ginCtx[n], &hasError_));
    if (hasError_) break;
  }
  *hasError = hasError_;
  return ncclSuccess;
}
