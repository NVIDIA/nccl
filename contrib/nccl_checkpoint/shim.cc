#include <nccl.h>
#include "shim_core.h"
#include <dlfcn.h>
#include <cstdio>
#include <vector>

using namespace nccl_checkpoint;

// include our auto-generated shims
#include "shim_auto.cc"

extern "C" ncclResult_t ncclCheckpointGetVersion(int* checkpointVersion, int* ncclVersion) {
  if (checkpointVersion == nullptr || ncclVersion == nullptr) return ncclInvalidArgument;
  *checkpointVersion = NCCL_CHECKPOINT_VERSION_CODE;
  *ncclVersion = NCCL_VERSION_CODE;
  return ncclSuccess;
}

static CommInitParams* allocInitParams(int nranks, int rank, const ncclConfig_t* configPtr) {
  CommInitParams* params = new CommInitParams{};
  params->creation = create_via_init;
  params->nranks = nranks;
  params->rank = rank;
  params->commHash = 0; // we'll fill this later during checkpointPrepare()
  params->configProvided = configPtr != nullptr;
  params->net_name[0] = '\0';
  params->comm_name[0] = '\0';
  if (configPtr) {
    params->config = *configPtr;
    if (configPtr->netName) {
      strncpy(params->net_name, configPtr->netName, sizeof(params->net_name) - 1);
    }
    if (configPtr->commName) {
      strncpy(params->comm_name, configPtr->commName, sizeof(params->comm_name) - 1);
    }
  } else {
    params->config = NCCL_CONFIG_INITIALIZER;
  }
  params->config.size = sizeof(ncclConfig_t);
  params->config.magic = NCCL_API_MAGIC;
  params->config.version = NCCL_VERSION_CODE;
  params->config.netName = params->net_name[0] ? params->net_name : nullptr;
  params->config.commName = params->comm_name[0] ? params->comm_name : nullptr;
  params->commParent = nullptr;
  return params;
}

static CommInitParams* allocSplitInitParams(ncclComm_t parentSynthComm, int color, int key,
                                            const ncclConfig_t* configPtr) {
  CommInitParams* params = allocInitParams(0, 0, configPtr);
  params->creation = create_via_split;
  params->commParent = parentSynthComm;
  params->splitColor = color;
  params->splitKey = key;
  params->configProvided = configPtr != nullptr;
  return params;
}

static CommInitParams* allocShrinkInitParams(ncclComm_t parentSynthComm, const int* excludeRanksList,
                                             int excludeRanksCount, const ncclConfig_t* configPtr, int shrinkFlags) {
  CommInitParams* params = allocInitParams(0, 0, configPtr);
  params->creation = create_via_shrink;
  params->commParent = parentSynthComm;
  params->shrinkFlags = shrinkFlags;
  if (excludeRanksList != nullptr && excludeRanksCount > 0) {
    params->shrinkExcludeRanks.assign(excludeRanksList, excludeRanksList + excludeRanksCount);
  }
  params->configProvided = configPtr != nullptr;
  return params;
}

static CommInitParams* allocGrowInitParams(ncclComm_t parentSynthComm, int nRanks, const ncclUniqueId* uniqueId,
                                           int rank, const ncclConfig_t* configPtr) {
  CommInitParams* params = allocInitParams(nRanks, rank, configPtr);
  params->creation = create_via_grow;
  params->commParent = parentSynthComm;
  params->growRankArg = rank;
  params->growUniqueIdProvided = uniqueId != nullptr;
  params->configProvided = configPtr != nullptr;
  return params;
}

static RegConfig* allocRegConfig(void* buff, size_t size) {
  RegConfig* config = new RegConfig{};
  config->base = buff;
  config->sz = size;
  config->sequence = nextRegistrationSequence();
  return config;
}

static WindowConfig* allocWindowConfig(void* buff, size_t size, int flags) {
  WindowConfig* config = new WindowConfig{};
  config->base = buff;
  config->sz = size;
  config->flags = flags;
  config->sequence = nextRegistrationSequence();
  return config;
}

static ncclResult_t registerNewChild(ncclComm_t parent) {
  if (parent == nullptr) return ncclSuccess;
  CommHandleEntry* entry = nullptr;
  NCCLCHECK(g_commHandles.find(parent, &entry));
  entry->liveChildCommCount++;
  return ncclSuccess;
}

// A finalized/destroyed communicator may still be needed later as the parent
// for restoring a child split/shrink/grow, so capture the fields needed for
// recreation while the real communicator is still available.
static ncclResult_t captureCommRuntimeState(ncclComm_t synthComm, ncclComm_t realComm) {
  if (!g_commHandles.checkHandle(synthComm) || realComm == nullptr) return ncclSuccess;
  const CommHandleEntry* entry = nullptr;
  NCCLCHECK(g_commHandles.find(synthComm, &entry));
  if (entry->config == nullptr) return ncclSuccess;
  if (realComm->endMagic != NCCL_MAGIC) {
    WARN("Compile and runtime version mismatch (Magic Mismatch).  libnccl-checkpoint-shim.so detected"
         " that NCCL runtime is a different version.  Please recompile with the same versions.");
    return ncclInternalError;
  }

  CommInitParams* params = g_commHandles[synthComm].config;
  params->commHash = realComm->commHash;
  params->cudaDev = realComm->cudaDev;
  return ncclSuccess;
}

// CREATE:comm — records init params for restore
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
  using real_t = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int);
  static real_t real_ncclCommInitRank = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommInitRank", &real_ncclCommInitRank));
  ncclComm_t realComm = nullptr;
  ncclResult_t ret = real_ncclCommInitRank(&realComm, nranks, commId, rank);
  if (realComm != nullptr) {
    *comm = g_commHandles.makeSynthetic(realComm, allocInitParams(nranks, rank, nullptr));
    if (*comm == nullptr) {
      ret = ncclSystemError;
    }
  }
  return ret;
}

// CREATE:comm — records init params for restore
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) {
  using real_t = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
  static real_t real_ncclCommInitRankConfig = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommInitRankConfig", &real_ncclCommInitRankConfig));
  ncclComm_t realComm = nullptr;
  ncclResult_t ret = real_ncclCommInitRankConfig(&realComm, nranks, commId, rank, config);
  if (realComm != nullptr) {
    *comm = g_commHandles.makeSynthetic(realComm, allocInitParams(nranks, rank, config));
    if (*comm == nullptr) {
      ret = ncclSystemError;
    }
  }
  return ret;
}

// CREATE:newcomm — records init params for restore
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds,
                                      ncclConfig_t* config) {
  using real_t = ncclResult_t (*)(ncclComm_t*, int, int, int, ncclUniqueId*, ncclConfig_t*);
  static real_t real_ncclCommInitRankScalable = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommInitRankScalable", &real_ncclCommInitRankScalable));
  ncclComm_t realComm = nullptr;
  ncclResult_t ret = real_ncclCommInitRankScalable(&realComm, nranks, myrank, nId, commIds, config);
  if (realComm != nullptr) {
    *newcomm = g_commHandles.makeSynthetic(realComm, allocInitParams(nranks, myrank, config));
    if (*newcomm == nullptr) {
      ret = ncclSystemError;
    }
  }
  return ret;
}

// CREATE:newcomm — records split parent/color/key for restore
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config) {
  using real_t = ncclResult_t (*)(ncclComm_t, int, int, ncclComm_t*, ncclConfig_t*);
  static real_t real_ncclCommSplit = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommSplit", &real_ncclCommSplit));
  ncclResult_t ret = ncclSuccess;

  ncclComm_t realComm = comm;
  NCCLCHECK(g_commHandles.toReal(comm, &realComm));

  CommInitParams* params = allocSplitInitParams(comm, color, key, config);
  ncclComm_t synthComm = g_commHandles.makeSynthetic(nullptr, params);
  if (synthComm == nullptr) {
    delete params;
    *newcomm = nullptr;
    return ncclSystemError;
  }

  ncclComm_t* realNewComm = &(g_commHandles[synthComm].realHandle);
  if (color != NCCL_SPLIT_NOCOLOR) {
    *newcomm = synthComm;
  } else {
    g_commHandles[synthComm].userState = comm_user_nocolor;
    *newcomm = nullptr;
  }
  NCCLCHECKGOTO(real_ncclCommSplit(realComm, color, key, realNewComm, config), ret, fail);
  NCCLCHECKGOTO(registerNewChild(comm), ret, fail);
  return ret;

fail:
  *newcomm = nullptr;
  g_commHandles.remove(synthComm);
  return ret;
}

// CREATE:newcomm — records shrink parent/excluded ranks/flags for restore
ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm,
                            ncclConfig_t* config, int shrinkFlags) {
  using real_t = ncclResult_t (*)(ncclComm_t, int*, int, ncclComm_t*, ncclConfig_t*, int);
  static real_t real_ncclCommShrink = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommShrink", &real_ncclCommShrink));
  ncclResult_t ret = ncclSuccess;

  ncclComm_t realComm = comm;
  NCCLCHECK(g_commHandles.toReal(comm, &realComm));

  CommInitParams* params = allocShrinkInitParams(comm, excludeRanksList, excludeRanksCount, config, shrinkFlags);
  ncclComm_t synthComm = g_commHandles.makeSynthetic(nullptr, params);
  if (synthComm == nullptr) {
    delete params;
    *newcomm = nullptr;
    return ncclSystemError;
  }

  ncclComm_t* realNewComm = &(g_commHandles[synthComm].realHandle);
  *newcomm = synthComm;
  NCCLCHECKGOTO(real_ncclCommShrink(realComm, excludeRanksList, excludeRanksCount, realNewComm, config, shrinkFlags),
                ret, fail);
  NCCLCHECKGOTO(registerNewChild(comm), ret, fail);
  return ret;

fail:
  *newcomm = nullptr;
  g_commHandles.remove(synthComm);
  return ret;
}

// CREATE:newcomm — records grow parent/role for restore
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm,
                          ncclConfig_t* config) {
  using real_t = ncclResult_t (*)(ncclComm_t, int, const ncclUniqueId*, int, ncclComm_t*, ncclConfig_t*);
  static real_t real_ncclCommGrow = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommGrow", &real_ncclCommGrow));
  ncclResult_t ret = ncclSuccess;

  ncclComm_t realComm = comm;
  if (comm != nullptr) {
    NCCLCHECK(g_commHandles.toReal(comm, &realComm));
  }

  if (newcomm == nullptr) {
    return real_ncclCommGrow(realComm, nRanks, uniqueId, rank, nullptr, config);
  }

  CommInitParams* params = allocGrowInitParams(comm, nRanks, uniqueId, rank, config);
  ncclComm_t synthComm = g_commHandles.makeSynthetic(nullptr, params);
  if (synthComm == nullptr) {
    delete params;
    *newcomm = nullptr;
    return ncclSystemError;
  }

  ncclComm_t* realNewComm = &(g_commHandles[synthComm].realHandle);
  *newcomm = synthComm;
  NCCLCHECKGOTO(real_ncclCommGrow(realComm, nRanks, uniqueId, rank, realNewComm, config), ret, fail);
  NCCLCHECKGOTO(registerNewChild(comm), ret, fail);
  return ret;

fail:
  *newcomm = nullptr;
  g_commHandles.remove(synthComm);
  return ret;
}

static ncclResult_t forgetCommResources(ncclComm_t comm) {
  CommHandleEntry* entry = nullptr;
  NCCLCHECK(g_commHandles.find(comm, &entry));
  for (const auto& [_, synthMR] : entry->registrations) {
    g_regHandles.remove(synthMR);
  }
  for (const auto& [_, synthWin] : entry->windows) {
    g_windowHandles.remove(synthWin);
  }
  entry->registrations.clear();
  entry->windows.clear();
  return ncclSuccess;
}

static ncclResult_t releaseNocolorChildren(ncclComm_t parent) {
  CommHandleEntry* parentEntry = nullptr;
  NCCLCHECK(g_commHandles.find(parent, &parentEntry));
  NCCLCHECK(g_commHandles.forEachHandle([&](ncclComm_t synthComm, const CommHandleEntry* entry) {
    CommInitParams* params = entry->config;
    if (params->commParent == parent && entry->userState == comm_user_nocolor) {
      assert(entry->realHandle == nullptr);
      g_commHandles.remove(synthComm);
      parentEntry->liveChildCommCount--;
    }
    return ncclSuccess;
  }));
  assert(parentEntry->liveChildCommCount >= 0);
  return ncclSuccess;
}

static ncclResult_t releaseCommEntry(ncclComm_t comm) {
  ncclComm_t current = comm;
  int generation = 0;
  NCCLCHECK(forgetCommResources(comm));
  NCCLCHECK(releaseNocolorChildren(comm));

  while (current != nullptr) {
    CommHandleEntry* entry = nullptr;
    NCCLCHECK(g_commHandles.find(current, &entry));
    if (!entry->config) return ncclInternalError;
    if (generation > 0) entry->liveChildCommCount--;
    if (entry->userState == comm_user_active) return ncclSuccess;
    if (entry->liveChildCommCount > 0) return ncclSuccess;

    ncclComm_t parent = entry->config->commParent;
    g_commHandles.remove(current);
    generation++;
    current = parent;
  }
  return ncclSuccess;
}

static ncclResult_t markCommUserState(ncclComm_t comm, CommUserState userState) {
  CommHandleEntry* entry = nullptr;
  NCCLCHECK(g_commHandles.find(comm, &entry));
  entry->userState = userState;
  return ncclSuccess;
}

// DESTROY:comm - release tracking when no child restore depends on it.
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  using real_t = ncclResult_t (*)(ncclComm_t);
  static real_t real_ncclCommAbort = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommAbort", &real_ncclCommAbort));
  ncclComm_t realComm = comm;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(g_commHandles.toReal(comm, &realComm), ret, exit);
  NCCLCHECKGOTO(captureCommRuntimeState(comm, realComm), ret, exit);
  NCCLCHECKGOTO(real_ncclCommAbort(realComm), ret, exit);
  NCCLCHECKGOTO(markCommUserState(comm, comm_user_destroyed), ret, exit);
  g_commHandles.remap(comm, nullptr);
  NCCLCHECK(releaseCommEntry(comm));

exit:
  return ret;
}

// FINALIZE:comm - keep tracking until Destroy or checkpoint prepare.
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  using real_t = ncclResult_t (*)(ncclComm_t);
  static real_t real_ncclCommFinalize = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommFinalize", &real_ncclCommFinalize));
  ncclComm_t realComm = comm;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(g_commHandles.toReal(comm, &realComm), ret, exit);
  NCCLCHECKGOTO(captureCommRuntimeState(comm, realComm), ret, exit);
  NCCLCHECKGOTO(real_ncclCommFinalize(realComm), ret, exit);
  NCCLCHECKGOTO(markCommUserState(comm, comm_user_finalized), ret, exit);

exit:
  return ret;
}

// DESTROY:comm - release tracking when no child restore depends on it.
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  using real_t = ncclResult_t (*)(ncclComm_t);
  static real_t real_ncclCommDestroy = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommDestroy", &real_ncclCommDestroy));
  ncclComm_t realComm = comm;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(g_commHandles.toReal(comm, &realComm), ret, exit);
  NCCLCHECKGOTO(captureCommRuntimeState(comm, realComm), ret, exit);
  NCCLCHECKGOTO(real_ncclCommDestroy(realComm), ret, exit);
  NCCLCHECKGOTO(markCommUserState(comm, comm_user_destroyed), ret, exit);
  g_commHandles.remap(comm, nullptr);
  NCCLCHECK(releaseCommEntry(comm));

exit:
  return ret;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
  using real_t = ncclResult_t (*)(ncclComm_t*, int, const int*);
  static real_t real_ncclCommInitAll = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommInitAll", &real_ncclCommInitAll));

  std::vector<ncclComm_t> realComms(ndev);
  ncclResult_t ret = real_ncclCommInitAll(realComms.data(), ndev, devlist);
  for (int i = 0; i < ndev; i++) {
    if (realComms[i] != nullptr) {
      CommInitParams* params = allocInitParams(ndev, i, nullptr);
      params->cudaDev = devlist != nullptr ? devlist[i] : i;
      comm[i] = g_commHandles.makeSynthetic(realComms[i], params);
      if (comm[i] == nullptr) {
        delete params;
        ret = ncclSystemError;
        return ret;
      }
    }
  }
  return ret;
}

ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  using real_t = ncclResult_t (*)(const ncclComm_t, void*, size_t, void**);
  static real_t real_ncclCommRegister = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommRegister", &real_ncclCommRegister));

  ncclComm_t realComm = const_cast<ncclComm_t>(comm);
  ncclResult_t ret = g_commHandles.toReal(const_cast<ncclComm_t>(comm), &realComm);
  if (ret != ncclSuccess) return ret;
  void* realHandle = nullptr;
  ret = real_ncclCommRegister(realComm, buff, size, &realHandle);
  if (ret != ncclSuccess || realHandle == nullptr) {
    if (handle) *handle = realHandle;
    return ret;
  }

  RegConfig* params = allocRegConfig(buff, size);
  void* synthHandle = g_regHandles.makeSynthetic(realHandle, params);
  if (synthHandle == nullptr) {
    delete params;
    return ncclSystemError;
  }
  g_regHandles[synthHandle].synthComm = const_cast<ncclComm_t>(comm);
  g_commHandles[const_cast<ncclComm_t>(comm)].registrations[params->sequence] = synthHandle;
  *handle = synthHandle;
  return ret;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  using real_t = ncclResult_t (*)(const ncclComm_t, void*);
  static real_t real_ncclCommDeregister = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommDeregister", &real_ncclCommDeregister));

  ncclComm_t realComm = const_cast<ncclComm_t>(comm);
  ncclResult_t ret = g_commHandles.toReal(const_cast<ncclComm_t>(comm), &realComm);
  if (ret != ncclSuccess) return ret;
  void* realHandle = handle;
  const RegHandleEntry* entry = nullptr;
  if (g_regHandles.checkHandle(handle)) {
    NCCLCHECK(g_regHandles.find(handle, &entry));
  }
  if (entry != nullptr && entry->synthComm != const_cast<ncclComm_t>(comm)) return ncclInvalidArgument;
  uint64_t sequence = (entry != nullptr && entry->config != nullptr) ? entry->config->sequence : 0;
  ret = g_regHandles.toReal(handle, &realHandle);
  if (ret != ncclSuccess) return ret;
  ret = real_ncclCommDeregister(realComm, realHandle);
  if (ret == ncclSuccess && realHandle != nullptr) {
    if (sequence != 0) g_commHandles[const_cast<ncclComm_t>(comm)].registrations.erase(sequence);
    g_regHandles.remove(handle);
  }
  return ret;
}

ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags) {
  using real_t = ncclResult_t (*)(ncclComm_t, void*, size_t, ncclWindow_t*, int);
  static real_t real_ncclCommWindowRegister = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommWindowRegister", &real_ncclCommWindowRegister));

  ncclComm_t realComm = comm;
  ncclResult_t ret = g_commHandles.toReal(comm, &realComm);
  if (ret != ncclSuccess) return ret;

  WindowConfig* params = allocWindowConfig(buff, size, winFlags);
  ncclWindow_t synthWin = g_windowHandles.makeSynthetic(nullptr, params);
  *win = synthWin;
  if (synthWin == nullptr) {
    delete params;
    return ncclSystemError;
  }
  g_windowHandles[synthWin].synthComm = comm;
  g_commHandles[comm].windows[params->sequence] = synthWin;
  ncclWindow_t* realWin = &(g_windowHandles[synthWin].realHandle);
  ret = real_ncclCommWindowRegister(realComm, buff, size, realWin, winFlags);
  if (ret != ncclSuccess && ret != ncclInProgress) {
    g_commHandles[comm].windows.erase(params->sequence);
    g_windowHandles.remove(synthWin);
    return ret;
  }
  return ret;
}

ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  using real_t = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
  static real_t real_ncclCommWindowDeregister = nullptr;
  NCCLCHECK(resolveRealFunction("ncclCommWindowDeregister", &real_ncclCommWindowDeregister));

  ncclComm_t realComm = comm;
  ncclResult_t ret = g_commHandles.toReal(comm, &realComm);
  if (ret != ncclSuccess) return ret;
  ncclWindow_t realWin = win;
  const WindowHandleEntry* entry = nullptr;
  if (g_windowHandles.checkHandle(win)) {
    NCCLCHECK(g_windowHandles.find(win, &entry));
  }
  if (entry != nullptr && entry->synthComm != comm) return ncclInvalidArgument;
  uint64_t sequence = (entry != nullptr && entry->config != nullptr) ? entry->config->sequence : 0;
  ret = g_windowHandles.toReal(win, &realWin);
  if (ret != ncclSuccess) return ret;
  ret = real_ncclCommWindowDeregister(realComm, realWin);
  if (ret == ncclSuccess && realWin != nullptr) {
    if (sequence != 0) g_commHandles[comm].windows.erase(sequence);
    g_windowHandles.remove(win);
  }
  return ret;
}
