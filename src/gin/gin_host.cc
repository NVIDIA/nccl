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

NCCL_PARAM(GinEnable, "GIN_ENABLE", 1);
NCCL_PARAM(GinType, "GIN_TYPE", -1);
NCCL_PARAM(GinSignalPoolSize, "GIN_SIGNAL_POOL_SIZE", 64 << 10);
NCCL_PARAM(GinCounterPoolSize, "GIN_COUNTER_POOL_SIZE", 64 << 10);

void* ncclGinProgress(void* ginState_) {
  struct ncclGinState* ginState = (struct ncclGinState*)ginState_;
  while (1) {
    pthread_mutex_lock(&ginState->threadLock);
    if (ginState->ginProgress == 1) {
      pthread_mutex_unlock(&ginState->threadLock);
      for (int n=0; n<ginState->ginCommCount; n++) {
        ncclResult_t ret;
        if (ginState->ginType == NCCL_NET_DEVICE_GIN_PROXY) {
          ret = ncclGinProxyProgress(ginState->ncclGin, (struct ncclGinCollComm*)ginState->ginCtx[n]);
        } else {
          ret = ginState->ncclGin->ginProgress(ginState->ginCtx[n]);
        }
        if (ret != ncclSuccess) {
          __atomic_store_n(&ginState->asyncResult, ret, __ATOMIC_RELEASE);
          INFO(NCCL_ALL,"%s:%d -> %d [GIN Progress Thread]", __FILE__, __LINE__, ret);
          ginState->ginProgress = -2;
          return NULL;
        }
      }
      sched_yield();
    } else if (ginState->ginProgress == -1) {
      pthread_mutex_unlock(&ginState->threadLock);
      return NULL;
    } else if (ginState->ginProgress == 0) {
      pthread_cond_wait(&ginState->threadCond, &ginState->threadLock);
      pthread_mutex_unlock(&ginState->threadLock);
    } else {
      pthread_mutex_unlock(&ginState->threadLock);
      INFO(NCCL_ALL,"%s:%d -> [GIN Progress Thread] state unknown %d", __FILE__, __LINE__, ginState->ginProgress);
      ginState->ginProgress = -2;
      return NULL;
    }
  }
}

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

  ncclNetProperties_t props;
  NCCLCHECK(ginState->ncclGin->getProperties(0, &props));
  ginState->ginType = props.netDeviceType;
  if ((ncclParamGinType() != -1) && (ginState->ginType != ncclParamGinType())) {
    WARN("GIN-capable device type mismatch.");
    return ncclInternalError;
  }

  int ginCommCount;
  int64_t localNets[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(ncclTopoGetLocalNets(comm->topo, comm->rank, localNets, &ginState->ginCommCount));
  ginCommCount = std::min<int>(ginState->ginCommCount, NCCL_GIN_MAX_CONTEXTS);
  ginCommCount = std::min<int>(ginCommCount, ndev);

  int* allCommCounts = NULL;
  void** handles = NULL;
  char* allHandles = NULL;

  // Get the min local net count from all ranks
  NCCLCHECK(ncclCalloc(&allCommCounts, comm->nRanks));
  allCommCounts[comm->rank] = ginCommCount;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allCommCounts, sizeof(int)), ret, fail);
  for (int i = 0; i < comm->nRanks; i++) {
    ginCommCount = std::min<int>(ginCommCount, allCommCounts[i]);
  }
  free(allCommCounts);
  allCommCounts = NULL;

  if (ginCommCount == 0) {
    WARN("Gin connect : min local net count is zero");
    ret = ncclSystemError;
    goto fail;
  }
  ginState->ginCommCount = ginCommCount;

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

  for (int n = 0; n < ginCommCount; n++) {
    void* listenComm;
    NCCLCHECKGOTO(
      ginState->ncclGin->listen(ginState->ginInstance, localNets[n],
                                allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm),
      ret, fail);
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret,
                  fail);
    NCCLCHECKGOTO(ginState->ncclGin->connect(comm->netContext, handles, comm->nRanks, comm->rank,
                                             listenComm, ginState->ginComms + n),
                  ret, fail);
    if (ginState->ginType == NCCL_NET_DEVICE_GIN_PROXY) {
      NCCLCHECKGOTO(ncclGinProxyCreateContext(comm, ginState->ginComms[n], localNets[n],
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
  for (int n = 0; n < ginCommCount; n++) {
    if (ginState->ginDevHandles[n]->needsProxyProgress) ginState->needsProxyProgress = 1;
  }
  if (ginState->needsProxyProgress) {
    ginState->ginProgress = 1;
    pthread_mutex_init(&ginState->threadLock, NULL);
    pthread_cond_init(&ginState->threadCond, NULL);
    PTHREADCHECK(pthread_create(&ginState->thread, NULL, ncclGinProgress, ginState), "pthread_create");
    ncclSetThreadName(ginState->thread, "NCCL GIN Progress%2d", comm->cudaDev);
  }

  ncclSpaceConstruct(&ginState->counterSpace);
  ncclSpaceConstruct(&ginState->signalSpace);

exit:
  if (ret == ncclSuccess) ginState->connected = true;
  return ret;
fail:
  free(allCommCounts);
  free(allHandles);
  free(handles);
  goto exit;
}

ncclResult_t ncclGinFinalize(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (!ginState->connected) return ncclSuccess;

  if (ginState->needsProxyProgress) {
    pthread_mutex_lock(&ginState->threadLock);
    comm->sharedRes->ginState.ginProgress = -1;
    pthread_cond_signal(&ginState->threadCond);
    pthread_mutex_unlock(&ginState->threadLock);
    PTHREADCHECK(pthread_join(ginState->thread, NULL), "pthread_join");
  }

  if (ginState->ginType == NCCL_NET_DEVICE_GIN_PROXY) {
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
  memset(ginState, 0, sizeof(*ginState));
  return ncclSuccess;
}

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONTEXTS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONTEXTS]) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginState->ginType == NCCL_NET_DEVICE_GIN_PROXY) {
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
    if (ginState->ginType == NCCL_NET_DEVICE_GIN_PROXY) {
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
    if (ginState->ginType == NCCL_NET_DEVICE_GIN_PROXY)
      NCCLCHECK(ncclGinProxyQueryLastError(ginState->ncclGin, ginState->ginCtx[n], &hasError_));
    else
      NCCLCHECK(ginState->ncclGin->queryLastError(ginState->ginCtx[n], &hasError_));
    if (hasError_) break;
  }
  *hasError = hasError_;
  return ncclSuccess;
}
