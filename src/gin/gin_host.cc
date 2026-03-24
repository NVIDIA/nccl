/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

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

  if (comm->compCap < 70) {
    /* GIN only supported for Volta and later */
    INFO(NCCL_INIT, "Compute Capability (%d) is not sufficient to enable GIN.  Require Volta (70) or newer.",comm->compCap);
    return ncclSuccess;
  }

  ncclNetProperties_t props;
  NCCLCHECK(ginState.ncclGin->getProperties(0, &props));
  if (props.netDeviceType == NCCL_NET_DEVICE_GIN_PROXY ||
      props.netDeviceType == NCCL_NET_DEVICE_GIN_GDAKI) {
    // NOTE: The following cast is valid because ncclGinType_t variant values
    // should match NCCL_NET_DEVICE_GIN_* values from `enum ncclNetDeviceType`.
    ginState.ginType = static_cast<ncclGinType_t>(props.netDeviceType);

    if (ginState.ginType == NCCL_GIN_TYPE_PROXY) {
      // Replace ginState->ncclGin by a layer adding host queues
      NCCLCHECK(ncclGinProxyInit(&ginState.ncclGin));
    }
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
      struct ncclGinStateDevComm* dc = ginState->devComms;
      while (dc) {
        for (int n=0; n<ginState->ginCommCount; n++) {
          ncclResult_t ret = ginState->ncclGin->ginProgress(dc->ginCtx[n]);
          if (ret != ncclSuccess) {
            COMPILER_ATOMIC_STORE(&ginState->asyncResult, ret, std::memory_order_release);
            INFO(NCCL_ALL,"%s:%d -> %d [GIN Progress Thread]", __FILE__, __LINE__, ret);
            ginState->ginProgress = -2;
            return NULL;
          }
        }
        dc = dc->next;
      }
      lock.unlock();
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

NCCL_PARAM(GinNconnections, "GIN_NCONNECTIONS", -2);

ncclResult_t ncclGinConnectOnce(struct ncclComm* comm) {
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

  ginState->ginConnectionType = comm->globalGinSupport;
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
  NCCLCHECKGOTO(ncclCalloc(&ginCommCountHandles, comm->nRanks), ret, fail);

  ginState->ginCommCount = nLocalGinDevs;
  if (ginState->ginVersion < 13) {
    // We only support one context per connection, so we better create as many connections as possible.
    ginState->ginCommCount = NCCL_GIN_MAX_CONNECTIONS;
  }

  if (ncclParamGinNconnections() != -2) ginState->ginCommCount = ncclParamGinNconnections();
  ginState->ginCommCount = std::min<int>(NCCL_GIN_MAX_CONNECTIONS, ginState->ginCommCount);

  ginCommCountHandles[comm->rank] = ginState->ginCommCount;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
  for (int r = 0; r < comm->nRanks; r++) {
    ginState->ginCommCount = std::min(ginState->ginCommCount, ginCommCountHandles[r]);
  }

  NCCLCHECKGOTO(ncclCalloc(&allHandles, (size_t)comm->nRanks * NCCL_NET_HANDLE_MAXSIZE), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&handles, comm->nRanks), ret, fail);

  int nGinRanks;
  int myGinRank;
  if (ginState->ginConnectionType == NCCL_GIN_CONNECTION_FULL) {
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

  for (int n = 0; n < ginState->ginCommCount; n++) {
    void* listenComm;
    NCCLCHECKGOTO(
      ginState->ncclGin->listen(ginState->ginInstance, localGinDevs[n%nLocalGinDevs],
                                allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm),
      ret, fail);

    NCCLCHECKGOTO(ginState->ncclGin->getProperties(localGinDevs[n%nLocalGinDevs], ginState->ginProps+n),
      ret, fail);

    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret,
                  fail);

    NCCLCHECKGOTO(ginState->ncclGin->connect(comm->ginContext, handles, nGinRanks, myGinRank,
          listenComm, ginState->ginComms + n),
        ret, fail);

    NCCLCHECKGOTO(ginState->ncclGin->closeListen(listenComm), ret, fail);
  }
  free(handles);
  handles = NULL;
  free(allHandles);
  allHandles = NULL;
  free(ginCommCountHandles);
  ginCommCountHandles = NULL;

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

ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
    struct ncclDevComm* devComm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  devComm->ginSignalCount = reqs->ginSignalCount;
  devComm->ginCounterCount = reqs->ginCounterCount;

  // Allocate contexts
  int nContextsTotal = reqs->ginContextCount;
  if (ginState->ginVersion < 13) {
    nContextsTotal = ginState->ginCommCount;
  }
  devComm->ginContextCount = nContextsTotal;
  devComm->ginConnectionCount = ginState->ginCommCount;

  if (!reqs->ginExclusiveContexts) {
    // TODO: check if a shared devComm in the list could match our requirements.
  }

  nContextsTotal = ROUNDUP(nContextsTotal, ginState->ginCommCount);
  int nContextsPerComm = nContextsTotal / ginState->ginCommCount;
  INFO(NCCL_INIT, "devCommCreate: creating %d contexts: %d GIN connections with %d contexts each (%d contexts total requested)",
      nContextsTotal, ginState->ginCommCount, nContextsPerComm, reqs->ginContextCount);

  struct ncclGinStateDevComm* ginStateDevComm = NULL;
  NCCLCHECK(ncclCalloc(&ginStateDevComm, 1));
  ginStateDevComm->contextCount = nContextsTotal;
  ncclResult_t ret = ncclSuccess;

  ncclGinConfig_t ginConfig = {
    reqs->ginSignalCount,
    reqs->ginCounterCount,
    nContextsPerComm,
    reqs->ginQueueDepth,
    0
  };

  for (int n = 0; n < ginState->ginCommCount; n++) {
    NCCLCHECKGOTO(ginState->ncclGin->createContext(
                    ginState->ginComms[n], &ginConfig, &ginStateDevComm->ginCtx[n], &ginStateDevComm->devHandles[n]),
                  ret, end);
    devComm->ginNetDeviceTypes[n] = ginStateDevComm->devHandles[n]->netDeviceType;
    devComm->ginHandles[n] = ginStateDevComm->devHandles[n]->handle;
    if (ginStateDevComm->devHandles[n]->needsProxyProgress) ginState->needsProxyProgress = 1;
  }

  if (ginState->needsProxyProgress && ginState->ginProgress == 0) {
    ginState->ginProgress = 1;
    ginState->thread = std::thread(ncclGinProgress, ginState);
    ncclSetThreadName(ginState->thread, "NCCL GIN Progress%2d", comm->cudaDev);
  }

  // Add devComm context to the list
  {
    std::unique_lock<std::mutex> lock(ginState->mutex);
    struct ncclGinStateDevComm* last = ginState->devComms;
    if (last) {
      while (last->next) last = last->next;
      last->next = ginStateDevComm;
     } else {
      ginState->devComms = ginStateDevComm;
    }
  }

end:
  if (ret != ncclSuccess) {
    for (int n=0; n<ginState->ginCommCount; n++) {
      if (ginStateDevComm->ginCtx[n])
        ginState->ncclGin->destroyContext(ginStateDevComm->ginCtx[n]);
    }
    free(ginStateDevComm);
  }
  return ret;
}

ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm) {
  // Find the resource associated with this devComm. Use the gin handle as key.
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  struct ncclGinStateDevComm* dc = ginState->devComms, *prevDc = NULL;
  while (1) {
    if (dc == NULL) {
      WARN("Dev comm not found\n");
      return ncclInternalError;
    }
    if (dc->devHandles[0]->handle == devComm->ginHandles[0]) break;
    prevDc = dc;
    dc = dc->next;
  }

  std::unique_lock<std::mutex> lock(ginState->mutex);
  // Remove from linked list
  if (prevDc) prevDc->next = dc->next;
  else ginState->devComms = dc->next;
  lock.unlock();

  // Free GIN contexts
  for (int n = 0; n < ginState->ginCommCount; n++) {
    NCCLCHECK(ginState->ncclGin->destroyContext(dc->ginCtx[n]));
  }
  free(dc);
  return ncclSuccess;
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

  for (int n = 0; n < ginState->ginCommCount; n++) {
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
    NCCLCHECK(ginState->ncclGin->regMrSym(ginState->ginComms[n], address, size, NCCL_PTR_CUDA, mrFlags,
                                          &ginHostWins[n], &ginDevWins[n]));
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
    NCCLCHECK(ginState->ncclGin->deregMrSym(ginState->ginComms[n], ginHostWins[n]));
  }
  return ncclSuccess;
}

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError) {
  *hasError = false;
  struct ncclGinStateDevComm* dc = ginState->devComms;
  while (dc) {
    for (int n = 0; n < ginState->ginCommCount; n++) {
      NCCLCHECK(ginState->ncclGin->queryLastError(dc->ginCtx[n], hasError));
      if (*hasError) return ncclSuccess;
    }
    dc = dc->next;
  }
  return ncclSuccess;
}
