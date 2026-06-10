#include <nccl.h>
#include "shim_core.h"
#include "kv_store_client.h"
#include <dlfcn.h>
#include <cstdio>

using namespace nccl_checkpoint;

static int g_CommCheckpointCount = 0;

static ncclResult_t replayRegistration(ncclComm_t synthComm, void* synthMR, RegConfig* config) {
  using real_t = ncclResult_t (*)(const ncclComm_t, void*, size_t, void**);
  static real_t real_ncclCommRegister = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommRegister", &real_ncclCommRegister));

  ncclComm_t realComm = synthComm;
  NCCLCHECK(g_commHandles.toReal(synthComm, &realComm));
  RegHandleEntry& regEntry = g_regHandles[synthMR];
  regEntry.synthComm = synthComm;
  regEntry.realHandle = nullptr;
  NCCLCHECK(real_ncclCommRegister(realComm, config->base, config->sz, &regEntry.realHandle));
  return ncclSuccess;
}

static ncclResult_t replayWindow(ncclComm_t synthComm, ncclWindow_t synthWin, WindowConfig* config) {
  using real_t = ncclResult_t (*)(ncclComm_t, void*, size_t, ncclWindow_t*, int);
  static real_t real_ncclCommWindowRegister = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommWindowRegister", &real_ncclCommWindowRegister));

  ncclComm_t realComm = synthComm;
  NCCLCHECK(g_commHandles.toReal(synthComm, &realComm));
  WindowHandleEntry& windowEntry = g_windowHandles[synthWin];
  windowEntry.synthComm = synthComm;
  windowEntry.realHandle = nullptr;
  NCCLCHECK(real_ncclCommWindowRegister(realComm, config->base, config->sz, &windowEntry.realHandle, config->flags));
  return ncclSuccess;
}

static ncclResult_t isCommBlocking(CommInitParams* params, bool* isBlocking) {
  if (params->commParent != nullptr) {
    const CommHandleEntry* parentEntry = nullptr;
    NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
    if (parentEntry->config == nullptr) return ncclInternalError;
    return isCommBlocking(parentEntry->config, isBlocking);
  }
  *isBlocking = !params->configProvided || params->config.blocking == NCCL_CONFIG_UNDEF_INT || params->config.blocking;
  return ncclSuccess;
}

static uintptr_t commKey(ncclComm_t comm) {
  return reinterpret_cast<uintptr_t>(comm);
}

static const char* creationKindString(CommCreationKind creation) {
  switch (creation) {
  case create_via_init:
    return "init";
  case create_via_split:
    return "split";
  case create_via_shrink:
    return "shrink";
  case create_via_grow:
    return "grow";
  }
  return "unknown";
}

static ncclResult_t replayCommResources(ncclComm_t synthComm) {
  CommHandleEntry& commEntry = g_commHandles[synthComm];

  for (const auto& [_, synthMR] : commEntry.registrations) {
    const RegHandleEntry* entry = nullptr;
    NCCLCHECK(g_regHandles.find(synthMR, &entry));
    if (entry->config == nullptr) return ncclInternalError;
    NCCLCHECK(replayRegistration(synthComm, synthMR, entry->config));
  }
  for (const auto& [_, synthWin] : commEntry.windows) {
    const WindowHandleEntry* entry = nullptr;
    NCCLCHECK(g_windowHandles.find(synthWin, &entry));
    if (entry->config == nullptr) return ncclInternalError;
    NCCLCHECK(replayWindow(synthComm, synthWin, entry->config));
  }
  return ncclSuccess;
}

static ncclResult_t commGetHash(ncclComm_t realComm, uint64_t* commHash) {
  if (realComm->endMagic != NCCL_MAGIC) {
    int runtime_version;
    NCCLCHECK(ncclGetVersion(&runtime_version));
    WARN("Detected memory sentinel mismatch!  Likely the NCCL runtime version "
         "(%d) is not compatible with the libnccl-checkpoint-shim.so version (%d), "
         "or the communicator has experienced memory corruption."
         "  Please recompile with the same versions.",
         runtime_version, NCCL_VERSION_CODE);
    return ncclInternalError;
  }
  *commHash = realComm->commHash;
  return ncclSuccess;
}

static bool commIsActiveForUser(const CommHandleEntry* entry) {
  assert(entry != nullptr);
  return entry->userState == comm_user_active;
}

static bool commNeedsRestore(const CommHandleEntry* entry) {
  assert(entry != nullptr);
  return entry->userState == comm_user_active || entry->userState == comm_user_nocolor || entry->liveChildCommCount > 0;
}

static ncclResult_t restoreCommViaInit(KVStoreClient& kv, ncclComm_t synthComm, CommInitParams* params) {
  using config_fn_t = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
  static config_fn_t real_ncclCommInitRankConfig = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommInitRankConfig", &real_ncclCommInitRankConfig));

  uint64_t commHash = params->commHash;
  if (commHash == 0) return ncclInternalError;
  char uidKey[64];
  snprintf(uidKey, sizeof(uidKey), "restore/%016llx/uid", (unsigned long long)commHash);

  ncclUniqueId newUid;
  if (params->rank == 0) {
    ncclGetUniqueId(&newUid);
    if (!kv.set(uidKey, &newUid, sizeof(newUid))) return ncclInternalError;
    TRACE(NCCL_CHECKPOINT, "set restore uid for comm %p key %s", synthComm, uidKey);
  }

  if (params->rank != 0) {
    TRACE(NCCL_CHECKPOINT, "waiting for restore uid for comm %p key %s", synthComm, uidKey);
    size_t outLen;
    if (!kv.get(uidKey, &newUid, sizeof(newUid), &outLen) || outLen != sizeof(newUid)) {
      WARN("ncclCheckpointRestore: failed to get uid %s", uidKey);
      return ncclInternalError;
    }
  }
  TRACE(NCCL_CHECKPOINT, "got restore uid for comm %p key %s", synthComm, uidKey);
  ncclComm_t* newReal = &(g_commHandles[synthComm].realHandle);
  *newReal = nullptr;
  ncclConfig_t* cfg = params->getConfig();
  NCCLCHECK(real_ncclCommInitRankConfig(newReal, params->nranks, newUid, params->rank, cfg));
  TRACE(NCCL_CHECKPOINT, "queued init restore for comm %p into real slot %p", synthComm, newReal);
  return ncclSuccess;
}

static ncclResult_t restoreCommViaSplit(ncclComm_t synthComm, CommInitParams* params) {
  using split_fn_t = ncclResult_t (*)(ncclComm_t, int, int, ncclComm_t*, ncclConfig_t*);
  static split_fn_t real_ncclCommSplit = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommSplit", &real_ncclCommSplit));

  const CommHandleEntry* parentEntry = nullptr;
  NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
  if (parentEntry->realHandle == nullptr) {
    WARN("ncclCheckpointRestore: split parent %p has no real communicator", params->commParent);
    return ncclInternalError;
  }
  ncclComm_t realParent = parentEntry->realHandle;

  ncclComm_t* realChild = &(g_commHandles[synthComm].realHandle);
  *realChild = nullptr;
  ncclConfig_t* cfgPtr = params->getConfig();
  NCCLCHECK(real_ncclCommSplit(realParent, params->splitColor, params->splitKey, realChild, cfgPtr));
  return ncclSuccess;
}

static ncclResult_t restoreCommViaShrink(ncclComm_t synthComm, CommInitParams* params) {
  using shrink_fn_t = ncclResult_t (*)(ncclComm_t, int*, int, ncclComm_t*, ncclConfig_t*, int);
  static shrink_fn_t real_ncclCommShrink = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommShrink", &real_ncclCommShrink));

  const CommHandleEntry* parentEntry = nullptr;
  NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
  if (parentEntry->realHandle == nullptr) {
    WARN("ncclCheckpointRestore: shrink parent %p has no real communicator", params->commParent);
    return ncclInternalError;
  }
  ncclComm_t realParent = parentEntry->realHandle;

  ncclComm_t* realChild = &(g_commHandles[synthComm].realHandle);
  *realChild = nullptr;
  int* excludeRanks = params->shrinkExcludeRanks.empty() ? nullptr : params->shrinkExcludeRanks.data();
  int excludeRanksCount = static_cast<int>(params->shrinkExcludeRanks.size());
  ncclConfig_t* cfgPtr = params->getConfig();
  NCCLCHECK(real_ncclCommShrink(realParent, excludeRanks, excludeRanksCount, realChild, cfgPtr, params->shrinkFlags));
  return ncclSuccess;
}

static ncclResult_t restoreCommViaGrow(KVStoreClient& kv, ncclComm_t synthComm, CommInitParams* params) {
  using grow_fn_t = ncclResult_t (*)(ncclComm_t, int, const ncclUniqueId*, int, ncclComm_t*, ncclConfig_t*);
  using uid_fn_t = ncclResult_t (*)(ncclComm_t, ncclUniqueId*);
  static grow_fn_t real_ncclCommGrow = nullptr;
  static uid_fn_t real_ncclCommGetUniqueId = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommGrow", &real_ncclCommGrow));

  uint64_t commHash = params->commHash;
  if (commHash == 0) return ncclInternalError;
  char uidKey[64];
  snprintf(uidKey, sizeof(uidKey), "restore/%016llx/grow_uid", (unsigned long long)commHash);

  ncclComm_t realParent = nullptr;
  ncclUniqueId growUid;
  const ncclUniqueId* growUidPtr = nullptr;

  if (params->commParent != nullptr) {
    const CommHandleEntry* parentEntry = nullptr;
    NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
    if (parentEntry->realHandle == nullptr) {
      WARN("ncclCheckpointRestore: grow parent %p has no real communicator", params->commParent);
      return ncclInternalError;
    }
    realParent = parentEntry->realHandle;
    if (params->growUniqueIdProvided) {
      NCCLCHECK(resolveRealFunction("ncclCommGetUniqueId", &real_ncclCommGetUniqueId));
      NCCLCHECK(real_ncclCommGetUniqueId(realParent, &growUid));
      if (!kv.set(uidKey, &growUid, sizeof(growUid))) return ncclInternalError;
      growUidPtr = &growUid;
    }
  } else {
    size_t outLen;
    if (!kv.get(uidKey, &growUid, sizeof(growUid), &outLen) || outLen != sizeof(growUid)) {
      WARN("ncclCheckpointRestore: failed to get grow uid %s", uidKey);
      return ncclInternalError;
    }
    growUidPtr = &growUid;
  }

  ncclComm_t* realChild = &(g_commHandles[synthComm].realHandle);
  *realChild = nullptr;
  ncclConfig_t* cfgPtr = params->getConfig();
  NCCLCHECK(real_ncclCommGrow(realParent, params->nranks, growUidPtr, params->growRankArg, realChild, cfgPtr));
  return ncclSuccess;
}

static ncclResult_t waitRestoredComm(ncclComm_t synthComm, CommInitParams* params) {
  if (params->creation == create_via_split) {
    const CommHandleEntry* parentEntry = nullptr;
    NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
    if (parentEntry->realHandle == nullptr) {
      WARN("ncclCheckpointRestore: split parent %p has no real communicator", params->commParent);
      return ncclInternalError;
    }
    NCCLCHECK_WAIT(ncclInProgress, parentEntry->realHandle);
    if (params->splitColor == NCCL_SPLIT_NOCOLOR) return ncclSuccess;
  }
  if (params->creation == create_via_shrink) {
    const CommHandleEntry* parentEntry = nullptr;
    NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
    if (parentEntry->realHandle == nullptr) {
      WARN("ncclCheckpointRestore: shrink parent %p has no real communicator", params->commParent);
      return ncclInternalError;
    }
    NCCLCHECK_WAIT(ncclInProgress, parentEntry->realHandle);
  }
  if (params->creation == create_via_grow && params->commParent != nullptr) {
    const CommHandleEntry* parentEntry = nullptr;
    NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
    if (parentEntry->realHandle == nullptr) {
      WARN("ncclCheckpointRestore: grow parent %p has no real communicator", params->commParent);
      return ncclInternalError;
    }
    NCCLCHECK_WAIT(ncclInProgress, parentEntry->realHandle);
  }

  const CommHandleEntry* entry = nullptr;
  NCCLCHECK(g_commHandles.find(synthComm, &entry));
  if (entry->realHandle == nullptr) {
    WARN("ncclCheckpointRestore: restored communicator %p has no real communicator", synthComm);
    return ncclInternalError;
  }
  NCCLCHECK_WAIT(ncclInProgress, entry->realHandle);
  return ncclSuccess;
}

static ncclResult_t restoreComm(KVStoreClient& kv, ncclComm_t synthComm, CommInitParams* params) {
  CUDACHECK(cudaSetDevice(params->cudaDev));
  if (params->creation == create_via_split) {
    // ensure parent is ready before splitting
    const CommHandleEntry* parentEntry = nullptr;
    NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
    if (parentEntry->config == nullptr) {
      WARN("ncclCheckpointRestore: split parent %p has no init params", params->commParent);
      return ncclInternalError;
    }
    NCCLCHECK(waitRestoredComm(params->commParent, parentEntry->config));
    return restoreCommViaSplit(synthComm, params);
  }
  if (params->creation == create_via_shrink) {
    const CommHandleEntry* parentEntry = nullptr;
    NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
    if (parentEntry->config == nullptr) {
      WARN("ncclCheckpointRestore: shrink parent %p has no init params", params->commParent);
      return ncclInternalError;
    }
    NCCLCHECK(waitRestoredComm(params->commParent, parentEntry->config));
    return restoreCommViaShrink(synthComm, params);
  }
  if (params->creation == create_via_grow) {
    if (params->commParent != nullptr) {
      const CommHandleEntry* parentEntry = nullptr;
      NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
      if (parentEntry->config == nullptr) {
        WARN("ncclCheckpointRestore: grow parent %p has no init params", params->commParent);
        return ncclInternalError;
      }
      NCCLCHECK(waitRestoredComm(params->commParent, parentEntry->config));
    }
    return restoreCommViaGrow(kv, synthComm, params);
  }
  if (params->creation == create_via_init) {
    return restoreCommViaInit(kv, synthComm, params);
  }
  return ncclInternalError;
}

static ncclResult_t waitAllRestoredComms() {
  return g_commHandles.forEachHandle([&](ncclComm_t synthComm, const CommHandleEntry* entry) {
    CommInitParams* params = entry->config;
    if (!commNeedsRestore(entry)) return ncclSuccess;
    return waitRestoredComm(synthComm, params);
  });
}

static ncclResult_t restoreAllComms(KVStoreClient& kv) {
  uintptr_t group_id_start = 0;
  bool group_is_blocking = true;
  int devSave = 0;
  CUDACHECK(cudaGetDevice(&devSave));

  NCCLCHECK(ncclGroupStart());
  TRACE(NCCL_CHECKPOINT, "restore ncclGroupStart");
  ncclResult_t ret = g_commHandles.forEachHandle([&](ncclComm_t synthComm, const CommHandleEntry* entry) {
    CommInitParams* params = entry->config;
    if (!commNeedsRestore(entry)) return ncclSuccess;
    bool comm_is_blocking = true;
    NCCLCHECK(isCommBlocking(params, &comm_is_blocking));
    INFO(NCCL_CHECKPOINT, "restoring comm synth=0x%lx kind=%s rank=%d/%d parent=0x%lx cudaDev=%d blocking=%d",
         commKey(synthComm), creationKindString(params->creation), params->rank, params->nranks,
         commKey(params->commParent), params->cudaDev, comm_is_blocking ? 1 : 0);
    if (group_id_start == 0) {
      group_is_blocking = comm_is_blocking;
      group_id_start = commKey(synthComm);
    }
    ncclComm_t parent = nullptr;
    if (params->commParent != nullptr) {
      const CommHandleEntry* parentEntry = nullptr;
      NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
      parent = params->commParent;
    }
    bool need_new_group;
    need_new_group = group_is_blocking != comm_is_blocking;
    need_new_group |= parent != nullptr && commKey(parent) >= group_id_start;
    if (need_new_group) {
      TRACE(NCCL_CHECKPOINT,
            "restore ncclGroupEnd/ncclGroupStart blocking=%d nextBlocking=%d parent=0x%lx groupStart=0x%lx",
            group_is_blocking, comm_is_blocking, commKey(parent), group_id_start);
      NCCLCHECK(ncclGroupEnd());
      NCCLCHECK(ncclGroupStart());
      group_is_blocking = comm_is_blocking;
      group_id_start = commKey(synthComm);
    }
    NCCLCHECK(restoreComm(kv, synthComm, params));
    return ncclSuccess;
  });
  TRACE(NCCL_CHECKPOINT, "restore ncclGroupEnd");
  NCCLCHECK(ncclGroupEnd());
  NCCLCHECK(ret);
  CUDACHECK(cudaSetDevice(devSave));
  return ncclSuccess;
}

static ncclResult_t replayAllCommResources() {
  ncclResult_t ret = g_commHandles.forEachHandle([&](ncclComm_t synthComm, const CommHandleEntry* entry) {
    CommInitParams* params = entry->config;
    if (!commNeedsRestore(entry)) return ncclSuccess;
    if (!commIsActiveForUser(entry)) return ncclSuccess;
    bool comm_is_blocking = true;
    NCCLCHECK(isCommBlocking(params, &comm_is_blocking));
    if (!comm_is_blocking) {
      if (params->commParent) {
        const CommHandleEntry* parentEntry = nullptr;
        NCCLCHECK(g_commHandles.find(params->commParent, &parentEntry));
        NCCLCHECK_WAIT(ncclInProgress, params->commParent);
      }
    }
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(replayCommResources(synthComm));
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
  });
  NCCLCHECK(ret);
  ret = g_commHandles.forEachHandle([&](ncclComm_t synthComm, const CommHandleEntry* entry) {
    if (!commNeedsRestore(entry)) return ncclSuccess;
    if (!commIsActiveForUser(entry)) return ncclSuccess;
    NCCLCHECK_WAIT(ncclInProgress, synthComm);
    return ncclSuccess;
  });
  NCCLCHECK(ret);
  return ncclSuccess;
}

static ncclResult_t destroyRestoredUserDestroyedComms() {
  using destroy_t = ncclResult_t (*)(ncclComm_t);
  static destroy_t real_destroy = nullptr;

  uintptr_t maxHandle = g_commHandles.peekNextHandle();
  for (uintptr_t handleValue = maxHandle; handleValue > 1; handleValue--) {
    ncclComm_t synthComm = reinterpret_cast<ncclComm_t>(handleValue - 1);
    if (!g_commHandles.checkHandle(synthComm)) continue;
    const CommHandleEntry* entry = nullptr;
    NCCLCHECK(g_commHandles.find(synthComm, &entry));
    if (entry->config == nullptr) continue;
    if (!commNeedsRestore(entry)) continue;
    if (entry->userState != comm_user_destroyed || entry->realHandle == nullptr) continue;

    ncclResult_t ret;
    NCCLCHECK(ncclCommGetAsyncError(entry->realHandle, &ret));
    NCCLCHECK_WAIT(ret, entry->realHandle);
    NCCLCHECK(resolveRealFunction("ncclCommDestroy", &real_destroy));
    NCCLCHECK(real_destroy(entry->realHandle));
    g_commHandles.remap(synthComm, nullptr);
  }
  return ncclSuccess;
}

extern "C" ncclResult_t ncclCheckpointPrepare(void) {
  using finalize_t = ncclResult_t (*)(ncclComm_t);
  using destroy_t = ncclResult_t (*)(ncclComm_t);
  static finalize_t real_finalize = nullptr;
  static destroy_t real_destroy = nullptr;
  ncclResult_t ret;

  g_CommCheckpointCount = 0;

  NCCLCHECK(g_commHandles.forEachHandle([&](ncclComm_t synthComm, const CommHandleEntry* entry) {
    if (!entry->restoreUnsafe) return ncclSuccess;
    WARN("ncclCheckpointPrepare: communicator %p cannot be checkpointed: %s", synthComm,
         entry->restoreUnsafeReason ? entry->restoreUnsafeReason : "restore-unsafe API was used");
    return ncclInvalidUsage;
  }));

  NCCLCHECK(g_commHandles.forEachHandle([&](ncclComm_t synthComm, const CommHandleEntry* entry) {
    CommInitParams* params = entry->config;
    if (!commNeedsRestore(entry)) return ncclSuccess;
    g_CommCheckpointCount++;
    bool isActive = commIsActiveForUser(entry);
    if (!isActive && entry->userState != comm_user_finalized) {
      return ncclSuccess;
    }
    ncclComm_t realComm;
    NCCLCHECK(g_commHandles.toReal(synthComm, &realComm));
    if (!realComm) {
      return ncclSuccess;
    }
    ncclResult_t ret;
    NCCLCHECK(ncclCommGetAsyncError(realComm, &ret));
    NCCLCHECK_WAIT(ret, realComm);
    // extract hash at prepare time rather than blocking during init time.
    NCCLCHECK(commGetHash(realComm, &params->commHash));
    params->cudaDev = realComm->cudaDev;
    TRACE(NCCL_CHECKPOINT, "prepare finalize comm %p real=%p hash=0x%016lx cudaDev=%d", synthComm, realComm,
          params->commHash, params->cudaDev);
    if (isActive) {
      NCCLCHECK(resolveRealFunction("ncclCommFinalize", &real_finalize));
      NCCLCHECK(real_finalize(realComm));
    }
    TRACE(NCCL_CHECKPOINT, "prepare waited for comm %p status=%d", synthComm, ret);
    return ret;
  }));

  NCCLCHECK(g_commHandles.forEachHandle([&](ncclComm_t synthComm, const CommHandleEntry* entry) {
    if (!commNeedsRestore(entry)) return ncclSuccess;
    bool isActive = commIsActiveForUser(entry);
    if (!isActive && entry->userState != comm_user_finalized) {
      return ncclSuccess;
    }
    ncclResult_t ret;
    NCCLCHECK(ncclCommGetAsyncError(synthComm, &ret));
    NCCLCHECK_WAIT(ret, synthComm);
    return ncclSuccess;
  }));

  TRACE(NCCL_CHECKPOINT, "prepare ready for destroys");
  NCCLCHECK(g_commHandles.forEachHandle([&](ncclComm_t synthComm, const CommHandleEntry* entry) {
    if (!commNeedsRestore(entry)) return ncclSuccess;
    bool isActive = commIsActiveForUser(entry);
    if (!isActive && entry->userState != comm_user_finalized) {
      return ncclSuccess;
    }
    ncclComm_t realComm = synthComm;
    NCCLCHECK(g_commHandles.toReal(synthComm, &realComm));
    TRACE(NCCL_CHECKPOINT, "prepare destroy comm %p real=%p", synthComm, realComm);
    NCCLCHECK(resolveRealFunction("ncclCommDestroy", &real_destroy));
    ret = real_destroy(realComm);
    assert(ret != ncclInProgress);
    if (ret == ncclSuccess) g_commHandles.remap(synthComm, nullptr);
    NCCLCHECK(ret);
    return ncclSuccess;
  }));
  markCheckpointPrepared();
  return ncclSuccess;
}

extern "C" ncclResult_t ncclCheckpointRestore(void) {
  clearCheckpointPrepared();
  if (g_CommCheckpointCount == 0) {
    return ncclSuccess;
  }

  INFO(NCCL_CHECKPOINT, "starting restore for %d communicator(s)", g_CommCheckpointCount);
  KVStoreClient kv;
  if (!kv.connect_from_env()) {
    WARN("ncclCheckpointRestore: failed to connect to KVS; set NCCL_CHECKPOINT_KVS_PATH");
    return ncclInvalidArgument;
  }

  NCCLCHECK(restoreAllComms(kv));
  NCCLCHECK(replayAllCommResources());
  NCCLCHECK(destroyRestoredUserDestroyedComms());
  INFO(NCCL_CHECKPOINT, "restore complete");
  return ncclSuccess;
}
