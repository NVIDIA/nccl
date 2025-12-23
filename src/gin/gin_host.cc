/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "param.h"
#include "graph.h"
#include "transport.h"
#include "register_inline.h"
#include "gin/gin_host.h"
#include "gin/gin_host_proxy.h"
#include "compiler.h"

NCCL_PARAM(GinEnable, "GIN_ENABLE", 1);
NCCL_PARAM(GinType, "GIN_TYPE", -1);
NCCL_PARAM(GinSignalPoolSize, "GIN_SIGNAL_POOL_SIZE", 64 << 10);
NCCL_PARAM(GinCounterPoolSize, "GIN_COUNTER_POOL_SIZE", 64 << 10);

ncclResult_t getGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) {
    return ncclInternalError;
  }
  if (!comm->ginSupport) {
    *ginType = NCCL_GIN_TYPE_NONE;
    return ncclSuccess;
  }
  ncclNetProperties_t props;
  NCCLCHECK(comm->sharedRes->ginState.ncclGin->getProperties(0, &props));
  if (props.netDeviceType == NCCL_NET_DEVICE_GIN_PROXY) {
    *ginType = NCCL_GIN_TYPE_PROXY;
    return ncclSuccess;
  }
  if (props.netDeviceType == NCCL_NET_DEVICE_GIN_GDAKI) {
    *ginType = NCCL_GIN_TYPE_GDAKI;
    return ncclSuccess;
  }
  WARN("Cannot get gin type: ncclGin is not null but net device type (%d) is not a gin type", props.netDeviceType);
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

NCCL_PARAM(GinNcontexts, "GIN_NCONTEXTS", NCCL_GIN_MAX_CONTEXTS);

ncclResult_t ncclGinConnectOnce(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (ginState->ncclGin == NULL) {
    WARN("GIN not supported.");
    return ncclInvalidUsage;
  }
  if (ncclParamGinEnable() == 0) {
    WARN("GIN is disabled.");
    return ncclInternalError;
  }
  if (ginState->connected) return ncclSuccess;

  NCCLCHECK(ginState->ncclGin->init(&ginState->ginInstance, comm->commHash, ncclDebugLog));

  int ndev = 0;
  NCCLCHECK(ginState->ncclGin->devices(&ndev));
  if (ndev <= 0) {
    WARN("No GIN-capable devices found.");
    return ncclInternalError;
  }

  NCCLCHECK(getGinType(comm, &ginState->ginType));
  if ((ncclParamGinType() != -1) && (ginState->ginType != ncclParamGinType())) {
    WARN("GIN-capable device type mismatch.");
    return ncclInternalError;
  }

  int nLocalNets;
  int64_t localNets[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(ncclTopoGetLocalNets(comm->topo, comm->rank, localNets, &nLocalNets));

  void** handles = NULL;
  char* allHandles = NULL;

  ginState->ginCommCount = std::min<int>(NCCL_GIN_MAX_CONTEXTS, ncclParamGinNcontexts());

  NCCLCHECKGOTO(ncclCalloc(&allHandles, (size_t)comm->nRanks * NCCL_NET_HANDLE_MAXSIZE), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&handles, comm->nRanks), ret, fail);
  for (int r = 0; r < comm->nRanks; r++) handles[r] = allHandles + r * NCCL_NET_HANDLE_MAXSIZE;

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
      ginState->ncclGin->listen(ginState->ginInstance, localNets[n%nLocalNets],
                                allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm),
      ret, fail);
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret,
                  fail);
    NCCLCHECKGOTO(ginState->ncclGin->connect(comm->ginContext, handles, comm->nRanks, comm->rank,
                                             listenComm, ginState->ginComms + n),
                  ret, fail);
    if (ginState->ginType == NCCL_GIN_TYPE_PROXY) {
      NCCLCHECKGOTO(ncclGinProxyCreateContext(comm, ginState->ginComms[n], localNets[n%nLocalNets],
                                              ginState->signalSpaceSize, ginState->counterSpaceSize,
                                              &ginState->ginCtx[n], &ginState->ginDevHandles[n]),
                    ret, fail);
    } else {
      NCCLCHECKGOTO(ginState->ncclGin->createContext(
                      ginState->ginComms[n], ginState->signalSpaceSize, ginState->counterSpaceSize,
                      &ginState->ginCtx[n], &ginState->ginDevHandles[n]),
                    ret, fail);
    }
    NCCLCHECKGOTO(ginState->ncclGin->closeListen(listenComm), ret, fail);
  }
  free(handles);
  handles = NULL;
  free(allHandles);
  allHandles = NULL;

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
  free(allHandles);
  free(handles);
  goto exit;
}

ncclResult_t ncclGinFinalize(struct ncclComm* comm) {
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
  NCCLCHECK(ginState->ncclGin->finalize(ginState->ginInstance));
  memset((void*)ginState, 0, sizeof(*ginState));
  return ncclSuccess;
}

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONTEXTS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONTEXTS]) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginState->ginType == NCCL_GIN_TYPE_PROXY) {
      NCCLCHECK(ncclGinProxyRegister(ginState->ncclGin, ginState->ginCtx[n], address, size,
                                     NCCL_PTR_CUDA, 0, &ginHostWins[n], &ginDevWins[n]));
    } else {
      NCCLCHECK(ginState->ncclGin->regMrSym(ginState->ginComms[n], address, size, NCCL_PTR_CUDA, 0,
                                            &ginHostWins[n], &ginDevWins[n]));
    }
    if (ginHostWins[n] == NULL) {
      WARN("rank %d - GIN Symmetric register failed: buff %p, size %ld", comm->rank, address, size);
      return ncclSystemError;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONTEXTS]) {
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
