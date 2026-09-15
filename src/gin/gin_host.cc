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

// Backend version compatibility. Index: backend version. Value: min compatible NCCL version
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5), NCCL_VERSION(2, 32, 0)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
constexpr int efaGdaBackendMinVersions[] = {0, NCCL_VERSION(2, 31, 0), NCCL_VERSION(2, 32, 0)};

ncclResult_t ncclGetGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) return ncclInternalError;

  *ginType = comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL ? NCCL_GIN_TYPE_NONE :
                                                                  comm->sharedRes->ginState.backends[0].ginType;
  return ncclSuccess;
}

ncclResult_t ncclGetRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) return ncclInternalError;

  *ginType = comm->globalGinSupport == NCCL_GIN_CONNECTION_NONE ? NCCL_GIN_TYPE_NONE :
                                                                  comm->sharedRes->ginState.backends[0].ginType;
  return ncclSuccess;
}

static void ginProgressWriteLock(struct ncclGinState* ginState) {
  // This logic assumes just 1 writer. That's okay for this use case.
  ginState->writePending.store(true);
  ginState->devCommRwMutex.lock();
}

static void ginProgressWriteUnlock(struct ncclGinState* ginState) {
  ginState->devCommRwMutex.unlock();
  ginState->writePending.store(false);
}

// Per-thread progress worker. Thread t owns GIN connections t, t+proxyNthreads, t+2*proxyNthreads, ...
// across all devComms.
void* ncclGinProgress(struct ncclGinState* ginState, int threadIdx) {
  if (ncclOsCpuCount(ginState->cpuAffinity)) {
    ncclOsSetAffinity(ginState->cpuAffinity);
  }
  while (1) {
    if (ginState->proxyThreadStopSignal.load()) return NULL;
    // Back off while the main thread needs to modify the devComms list.
    if (ginState->writePending.load()) {
      std::this_thread::yield();
      continue;
    }
    {
      std::shared_lock<std::shared_timed_mutex> rlock(ginState->devCommRwMutex);
      struct ncclGinStateDevComm* dc = ginState->devComms;
      while (dc) {
        struct ncclGinBackendState* backend = &ginState->backends[dc->backendIndex];
        for (int commIdx = threadIdx; commIdx < backend->ginCommCount; commIdx += ginState->proxyNthreads) {
          if (dc->devHandles[commIdx]->needsProxyProgress) {
            ncclResult_t ret = backend->ncclGin->ginProgress(dc->ginCtx[commIdx]);
            if (ret != ncclSuccess) {
              COMPILER_ATOMIC_STORE(&ginState->asyncResult, ret, std::memory_order_release);
              INFO_LOC(NCCL_ALL, "-> %d [GIN Progress Thread %d]", ret, threadIdx);
              return NULL;
            }
          }
        }
        dc = dc->next;
      }
    }
    std::this_thread::yield();
  }
}

NCCL_PARAM(GinNconnections, "GIN_NCONNECTIONS", -2);
NCCL_PARAM(GinProxyNthreads, "GIN_PROXY_NTHREADS", 1);

ncclResult_t ncclGinConnectOnce(struct ncclComm* comm) {
  ncclTeam_t ginTeam;
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  if (ginState->connected) return ncclSuccess;

  ncclResult_t ret = ncclSuccess;
  if (ncclParamGinEnable() == 0) {
    WARN("GIN is disabled.");
    return ncclInternalError;
  }

  if (!ginState->supported) {
    WARN("GIN not supported.");
    return ncclInvalidUsage;
  }

  ginState->ginConnectionType = comm->globalGinSupport;

  if (!comm->symmetricSupport) {
    WARN("Communicator does not support symmetric memory!");
    return ncclInternalError;
  }

  int nLocalGinDevs;
  int localGinDevs[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(ncclTopoGetLocalGinDevs(comm, localGinDevs, &nLocalGinDevs));
  if (nLocalGinDevs > NCCL_GIN_MAX_CONNECTIONS) {
    ATTN("Found %d local devices, but GIN supports at most %d connections. Using the first %d connections.",
         nLocalGinDevs, NCCL_GIN_MAX_CONNECTIONS, NCCL_GIN_MAX_CONNECTIONS);
  }

  void** handles = NULL;
  char* allHandles = NULL;
  void** listenComms = NULL;
  int ndev = 0;
  struct ncclGinBackendState* backend = NULL;
  int* ginCommCountHandles = NULL;

  NCCLCHECKGOTO(ncclCalloc(&listenComms, ginState->numActiveBackends), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ginCommCountHandles, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&allHandles, (size_t)comm->nRanks * NCCL_NET_HANDLE_MAXSIZE), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&handles, comm->nRanks), ret, fail);

  // Connect the maximum supported connection type. Any future devComm may request
  // up to this connection type.
  ginTeam = ncclTeamWorld(comm);
  if (ginState->ginConnectionType != NCCL_GIN_CONNECTION_FULL) {
    ginTeam = {
      .nRanks = comm->nRanks / comm->contiguousRanksPerHost,
      .rank = comm->rank / comm->contiguousRanksPerHost,
      .stride = comm->contiguousRanksPerHost,
    };
  }
  for (int r = 0; r < ginTeam.nRanks; r++) {
    int worldRank = ncclTeamRankToWorld(comm, ginTeam, r);
    handles[r] = allHandles + worldRank * NCCL_NET_HANDLE_MAXSIZE;
  }

  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    backend = &ginState->backends[backendIdx];

    NCCLCHECKGOTO(backend->ncclGin->devices(&ndev), ret, fail);
    if (ndev <= 0) {
      WARN("No GIN-capable devices found.");
      ret = ncclInternalError;
      goto fail;
    }

    backend->ginCommCount = nLocalGinDevs;

    // Resolve the number of GIN progress threads. Default 1; Max NCCL_GIN_MAX_CONNECTIONS.
    ginState->proxyNthreads = 1;
    if (ncclParamGinProxyNthreads() > 1) ginState->proxyNthreads = ncclParamGinProxyNthreads();
    ginState->proxyNthreads = std::min<int>(NCCL_GIN_MAX_CONNECTIONS, ginState->proxyNthreads);

    if (ncclParamGinNconnections() != -2) backend->ginCommCount = ncclParamGinNconnections();
    backend->ginCommCount = std::min<int>(NCCL_GIN_MAX_CONNECTIONS, backend->ginCommCount);
    // Ensure ginCommCount >= proxyNthreads before AllGather.
    if (backend->ginCommCount < ginState->proxyNthreads) {
      backend->ginCommCount = ginState->proxyNthreads;
      INFO(NCCL_INIT, "GIN: increased ginCommCount to %d to match GIN_PROXY_NTHREADS", backend->ginCommCount);
    }

    ginCommCountHandles[comm->rank] = backend->ginCommCount;
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
    for (int r = 0; r < comm->nRanks; r++) {
      backend->ginCommCount = std::min(backend->ginCommCount, ginCommCountHandles[r]);
    }
    // After cross-rank min, proxyNthreads may exceed ginCommCount if ranks disagree
    // on NCCL_GIN_PROXY_NTHREADS (atypical — env vars are normally uniform across a job).
    // Extra threads simply idle in the stride loop; no correctness issue.

    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      NCCLCHECKGOTO(backend->ncclGin->listen(backend->ginInstance, localGinDevs[commIdx % nLocalGinDevs],
                                             allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank,
                                             &listenComms[backendIdx]),
                    ret, fail);

      NCCLCHECKGOTO(backend->ncclGin->getProperties(localGinDevs[commIdx % nLocalGinDevs], backend->ginProps + commIdx),
                    ret, fail);

      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret, fail);

      NCCLCHECKGOTO(backend->ncclGin->connect(backend->ginInstance, handles, ginTeam.nRanks, ginTeam.rank,
                                              listenComms[backendIdx], backend->ginComms + commIdx),
                    ret, fail);

      NCCLCHECKGOTO(backend->ncclGin->closeListen(listenComms[backendIdx]), ret, fail);
      listenComms[backendIdx] = NULL;
    }
  }

exit:
  free(handles);
  free(allHandles);
  free(ginCommCountHandles);
  free(listenComms);
  if (ret == ncclSuccess) ginState->connected = true;
  return ret;
fail:
  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    backend = &ginState->backends[backendIdx];

    if (listenComms[backendIdx] != NULL) {
      NCCLCHECKIGNORE(backend->ncclGin->closeListen(listenComms[backendIdx]), ret);
    }

    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      if (backend->ginComms[commIdx] != NULL) {
        NCCLCHECKIGNORE(backend->ncclGin->closeColl(backend->ginComms[commIdx]), ret);
        backend->ginComms[commIdx] = NULL;
      }
    }
  }
  goto exit;
}

ncclResult_t ncclGinValidateSignalRequest(struct ncclDevCommRequirements const* reqs,
                                          struct ncclGinBackendState* backend) {
  if (reqs->ginStrongSignalsRequired && !backend->supportsStrongSignals) {
    WARN("GIN strong signals are required, but the GIN plugin does not support them.");
    return ncclInvalidUsage;
  }

  if (reqs->ginVaSignalsRequired && !backend->supportsVASignals) {
    WARN("GIN VA signals are required, but the GIN plugin does not support them.");
    return ncclInvalidUsage;
  }

  return ncclSuccess;
}

static ncclResult_t ginDevCommSetupWithBackend(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                               struct ncclDevComm* devComm, uint32_t deviceCodeVersion,
                                               struct ncclGinBackendState* backend) {
  ncclGinConfig_t ginConfig;
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  devComm->backendIndex = (uint8_t)(backend - ginState->backends);
  devComm->ginSignalCount = reqs->ginSignalCount;
  devComm->ginCounterCount = reqs->ginCounterCount;
  // Legacy signals default to what is specified in DevCommRequirements
  devComm->ginStrongLegacySignals = reqs->ginStrongSignalsRequired;

  // Allocate contexts
  int nContextsTotal = reqs->ginContextCount;
  devComm->ginConnectionCount = backend->ginCommCount;
  if (!reqs->ginExclusiveContexts) {
    // TODO: check if a shared devComm in the list could match our requirements.
  }

  nContextsTotal = ROUNDUP(nContextsTotal, backend->ginCommCount);
  devComm->ginContextCount = nContextsTotal;
  int nContextsPerComm = nContextsTotal / backend->ginCommCount;
  INFO(NCCL_INIT,
       "devCommCreate: creating %d contexts: %d GIN connections with %d contexts each (%d contexts total requested)",
       nContextsTotal, backend->ginCommCount, nContextsPerComm, reqs->ginContextCount);

  struct ncclGinStateDevComm* ginStateDevComm = NULL;
  NCCLCHECK(ncclCalloc(&ginStateDevComm, 1));
  ginStateDevComm->contextCount = nContextsTotal;
  ginStateDevComm->backendIndex = (int)(backend - ginState->backends);

  const int* backendVersionArray;
  int nVersions;
  switch (backend->ginType) {
  case NCCL_GIN_TYPE_PROXY:
    backendVersionArray = proxyBackendMinVersions;
    nVersions = sizeof(proxyBackendMinVersions) / sizeof(int);
    break;
  case NCCL_GIN_TYPE_GDAKI:
    backendVersionArray = gdakiBackendMinVersions;
    nVersions = sizeof(gdakiBackendMinVersions) / sizeof(int);
    break;
  case NCCL_GIN_TYPE_GPI:
    backendVersionArray = gpiBackendMinVersions;
    nVersions = sizeof(gpiBackendMinVersions) / sizeof(int);
    break;
  case NCCL_GIN_TYPE_EFA_GDA:
    backendVersionArray = efaGdaBackendMinVersions;
    nVersions = sizeof(efaGdaBackendMinVersions) / sizeof(int);
    break;
  default:
    WARN("Cannot get backend version for unsupported GIN type %d", backend->ginType);
    return ncclInternalError;
  }

  int backendVersion = 0;
  for (int i = 0; i < nVersions; i++) {
    if (deviceCodeVersion < backendVersionArray[i]) break;
    backendVersion = i;
  }

  ncclResult_t ret = ncclSuccess;
  bool needsProxyProgress = false;

  int connectedStride =
    comm->sharedRes->ginState.ginConnectionType == NCCL_GIN_CONNECTION_FULL ? 1 : comm->contiguousRanksPerHost;
  int requestedStride = 1;
  if (reqs->ginConnectionType == NCCL_GIN_CONNECTION_CUSTOM_STRIDE) {
    requestedStride = reqs->ginCustomStride;
  } else if (reqs->ginConnectionType == NCCL_GIN_CONNECTION_RAIL) {
    requestedStride = ncclTeamRail(comm).stride;
  }

  if (requestedStride == 0) {
    WARN("Cannot create DevComm with a GIN rank stride of 0. To disable GIN, set reqs->ginConnectionType to "
         "NCCL_GIN_CONNECTION_NONE.");
    ret = ncclInvalidUsage;
    goto end;
  }
  if (requestedStride > ncclTeamRail(comm).stride) {
    // Hierarchical barriers assume GIN is at least RAIL connected.
    WARN("Cannot create DevComm with a GIN rank stride %d greater than the rail team stride %d", requestedStride,
         ncclTeamRail(comm).stride);
    ret = ncclInvalidUsage;
    goto end;
  }
  if (requestedStride % connectedStride != 0) {
    WARN("Cannot create DevComm with the requested GIN rank stride %d, this comm only supports strides that are "
         "multiples of %d",
         requestedStride, connectedStride);
    ret = ncclInvalidUsage;
    goto end;
  }

  devComm->ginConnectionStride = connectedStride;
  devComm->ginConnectionStride_rcp32 = idivRcp32(connectedStride);
  devComm->ginContextStride = requestedStride;
  ginConfig = {
    reqs->ginSignalCount,
    reqs->ginCounterCount,
    nContextsPerComm,
    reqs->ginQueueDepth,
    reqs->ginTrafficClass != NCCL_CONFIG_UNDEF_INT ? reqs->ginTrafficClass : comm->config.trafficClass,
    backendVersion,
    /*rankStride*/ requestedStride / connectedStride,
  };

  for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
    NCCLCHECKGOTO(backend->ncclGin->createContext(backend->ginComms[commIdx], &ginConfig,
                                                  &ginStateDevComm->ginCtx[commIdx],
                                                  &ginStateDevComm->devHandles[commIdx]),
                  ret, end);
    if (ginStateDevComm->ginCtx[commIdx] == NULL || ginStateDevComm->devHandles[commIdx] == NULL ||
        ginStateDevComm->devHandles[commIdx]->handle == NULL) {
      WARN("GIN plugin %s returned invalid context for connection %d: ginCtx=%p devHandle=%p handle=%p",
           backend->ncclGin->name, commIdx, ginStateDevComm->ginCtx[commIdx], ginStateDevComm->devHandles[commIdx],
           ginStateDevComm->devHandles[commIdx] ? ginStateDevComm->devHandles[commIdx]->handle : NULL);
      ret = ncclInternalError;
      goto end;
    }
    devComm->ginNetDeviceTypes[commIdx] = ginStateDevComm->devHandles[commIdx]->netDeviceType;
    devComm->ginHandles[commIdx] = ginStateDevComm->devHandles[commIdx]->handle;
    if (ginStateDevComm->devHandles[commIdx]->needsProxyProgress) needsProxyProgress = true;
  }

  // Add devComm and (re)start progress threads as needed.
  {
    bool needsStart = needsProxyProgress && !ginState->proxyThreadsCreated;

    if (ginState->proxyThreadsCreated) ginProgressWriteLock(ginState);
    struct ncclGinStateDevComm* last = ginState->devComms;
    if (last) {
      while (last->next) last = last->next;
      last->next = ginStateDevComm;
    } else {
      ginState->devComms = ginStateDevComm;
    }
    if (ginState->proxyThreadsCreated) ginProgressWriteUnlock(ginState);

    if (needsStart) {
      ginState->cpuAffinity = comm->cpuAffinity;
      ginState->proxyThreadsCreated = true;
      for (int t = 0; t < ginState->proxyNthreads; t++) {
        ginState->thread[t] = std::thread([ginState, t] { ncclGinProgress(ginState, t); });
        ncclSetThreadName(ginState->thread[t], "NCCL GIN P%d-%d", comm->cudaDev, t);
      }
    }
  }

end:
  if (ret != ncclSuccess) {
    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      if (ginStateDevComm->ginCtx[commIdx]) backend->ncclGin->destroyContext(ginStateDevComm->ginCtx[commIdx]);
    }
    free(ginStateDevComm);
    devComm->backendIndex = 0;
    devComm->ginConnectionCount = 0;
    devComm->ginContextCount = 0;
    devComm->ginConnectionStride = 0;
    devComm->ginConnectionStride_rcp32 = 0;
    devComm->ginContextStride = 0;
    memset(devComm->ginNetDeviceTypes, 0, sizeof(devComm->ginNetDeviceTypes));
    memset(devComm->ginHandles, 0, sizeof(devComm->ginHandles));
  }
  return ret;
}

ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                 struct ncclDevComm* devComm, uint32_t deviceCodeVersion) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  ncclGinType_t reqGinType = reqs->ginType;
  int64_t envGinType = ncclParamGinType();

  if (reqGinType == NCCL_GIN_TYPE_NONE && envGinType > NCCL_GIN_TYPE_NONE) {
    reqGinType = (ncclGinType_t)envGinType;
  }

  if (reqGinType >= NCCL_GIN_MAX_TYPES) {
    WARN("Invalid GIN type requested (%d)", reqGinType);
    return ncclInvalidUsage;
  }

  for (int i = 0; i < ginState->numActiveBackends; i++) {
    struct ncclGinBackendState* candidate = &ginState->backends[i];

    if (reqGinType != NCCL_GIN_TYPE_NONE && candidate->ginType != reqGinType) {
      continue;
    }
    if (ncclSuccess != ncclGinValidateSignalRequest(reqs, candidate)) {
      continue;
    }

    if (ncclSuccess == ginDevCommSetupWithBackend(comm, reqs, devComm, deviceCodeVersion, candidate)) {
      return ncclSuccess;
    }

    INFO(NCCL_INIT, "GIN: DevComm setup failed with backend type %d", candidate->ginType);
  }

  WARN("GIN: DevComm setup failed on all available backends");
  return ncclInternalError;
}

// Called from main thread.
ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm) {
  // Find the resource associated with this devComm. Use the gin handle as key.
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  struct ncclGinStateDevComm *dc = ginState->devComms, *prevDc = NULL;
  while (1) {
    if (dc == NULL) {
      WARN("Dev comm not found\n");
      return ncclInternalError;
    }
    if (dc->devHandles[0]->handle == devComm->ginHandles[0]) break;
    prevDc = dc;
    dc = dc->next;
  }

  ginProgressWriteLock(ginState);
  // Remove from linked list
  if (prevDc) prevDc->next = dc->next;
  else ginState->devComms = dc->next;
  ginProgressWriteUnlock(ginState);

  struct ncclGinBackendState* backend = &ginState->backends[dc->backendIndex];
  // The devComm is now unreachable by any progress thread; safe to destroy
  // its contexts while the workers keep progressing the rest of the list.
  for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
    NCCLCHECK(backend->ncclGin->destroyContext(dc->ginCtx[commIdx]));
  }
  free(dc);
  return ncclSuccess;
}

ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (!ginState->connected) return ncclSuccess;

  if (ginState->proxyThreadsCreated) {
    ginState->proxyThreadStopSignal.store(true);
    for (int t = 0; t < ginState->proxyNthreads; t++) {
      if (ginState->thread[t].joinable()) ginState->thread[t].join();
    }
  }

  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    struct ncclGinBackendState* backend = &ginState->backends[backendIdx];
    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      if (backend->ginComms[commIdx] != NULL) {
        NCCLCHECK(backend->ncclGin->closeColl(backend->ginComms[commIdx]));
        backend->ginComms[commIdx] = NULL;
      }
    }
  }
  memset((void*)ginState, 0, sizeof(*ginState));
  return ncclSuccess;
}

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS],
                             int winFlags, bool multiSegment, int memType) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  int mrFlags = (winFlags & NCCL_WIN_STRICT_ORDERING) ? NCCL_NET_MR_FLAG_FORCE_SO : 0;
  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    struct ncclGinBackendState* backend = &ginState->backends[backendIdx];
    if (multiSegment) {
      // Multi-segment GIN registration requires DMABUF support on all GIN connections
      for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
        if (!(backend->ginProps[commIdx].ptrSupport & NCCL_PTR_DMABUF)) {
          WARN(
            "Window registration of addresses that span multiple physical segments requires DMABUF support with GIN.");
          return ncclInvalidArgument;
        }
      }
    }
    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      int slot = backendIdx * NCCL_GIN_MAX_CONNECTIONS + commIdx;
      NCCLCHECK(backend->ncclGin->regMrSym(backend->ginComms[commIdx], address, size, memType, mrFlags,
                                           &ginHostWins[slot], &ginDevWins[slot]));
      if (ginHostWins[slot] == NULL) {
        WARN("rank %d - GIN Symmetric register failed: buff %p, size %ld", comm->rank, address, size);
        return ncclSystemError;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinDeregister(struct ncclComm* comm,
                               void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS]) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  for (int backendIdx = 0; backendIdx < ginState->numActiveBackends; backendIdx++) {
    struct ncclGinBackendState* backend = &ginState->backends[backendIdx];
    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      int slot = backendIdx * NCCL_GIN_MAX_CONNECTIONS + commIdx;
      NCCLCHECK(backend->ncclGin->deregMrSym(backend->ginComms[commIdx], ginHostWins[slot]));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError) {
  *hasError = false;
  std::shared_lock<std::shared_timed_mutex> rlock(ginState->devCommRwMutex);
  struct ncclGinStateDevComm* dc = ginState->devComms;
  while (dc) {
    struct ncclGinBackendState* backend = &ginState->backends[dc->backendIndex];
    for (int commIdx = 0; commIdx < backend->ginCommCount; commIdx++) {
      NCCLCHECK(backend->ncclGin->queryLastError(dc->ginCtx[commIdx], hasError));
      if (*hasError) return ncclSuccess;
    }
    dc = dc->next;
  }
  return ncclSuccess;
}
