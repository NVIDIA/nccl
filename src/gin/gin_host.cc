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
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime_api.h>

NCCL_PARAM(GinEnable, "GIN_ENABLE", 1);
NCCL_PARAM(DevApiJit, "DEV_API_JIT", 0);

// Backend version compatibility. Index: backend version. Value: min compatible NCCL version
const int proxyBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gdakiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 3), NCCL_VERSION(2, 30, 5)};
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};

ncclResult_t ncclGetGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) return ncclInternalError;

  *ginType =
    comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL ? NCCL_GIN_TYPE_NONE : comm->sharedRes->ginState.ginType;
  return ncclSuccess;
}

ncclResult_t ncclGetRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) return ncclInternalError;

  *ginType =
    comm->globalGinSupport == NCCL_GIN_CONNECTION_NONE ? NCCL_GIN_TYPE_NONE : comm->sharedRes->ginState.ginType;
  return ncclSuccess;
}

void* ncclGinProgress(struct ncclGinState* ginState_) {
  struct ncclGinState* ginState = (struct ncclGinState*)ginState_;
  if (ncclOsCpuCount(ginState->cpuAffinity)) {
    ncclOsSetAffinity(ginState->cpuAffinity);
  }
  while (1) {
    std::unique_lock<std::mutex> lock(ginState->mutex);
    if (ginState->ginProgress == 1) {
      struct ncclGinStateDevComm* dc = ginState->devComms;
      while (dc) {
        for (int n = 0; n < ginState->ginCommCount; n++) {
          ncclResult_t ret = ginState->ncclGin->ginProgress(dc->ginCtx[n]);
          if (ret != ncclSuccess) {
            COMPILER_ATOMIC_STORE(&ginState->asyncResult, ret, std::memory_order_release);
            INFO_LOC(NCCL_ALL, "-> %d [GIN Progress Thread]", ret);
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
      INFO_LOC(NCCL_ALL, "[GIN Progress Thread] state unknown %d", ginState->ginProgress);
      ginState->ginProgress = -2;
      return NULL;
    }
  }
}

NCCL_PARAM(GinNconnections, "GIN_NCONNECTIONS", -2);

static bool envIsSet(const char* envName) {
  const char* env = ncclGetEnv(envName);
  return env != NULL && env[0] != '\0';
}

struct ncclGinConnInfo {
  int ginDev;
  uint64_t guid;
  int16_t planeId;
  char name[NCCL_GIN_NIC_NAME_MAX];
};

struct ncclGinRankInfo {
  int connCount;
  ncclGinConnInfo conns[NCCL_GIN_MAX_CONNECTIONS];
};

struct ncclGinDevName {
  int dev;
  char name[NCCL_GIN_NIC_NAME_MAX];
};

static std::string trimString(std::string const& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
  return s.substr(b, e - b);
}

static ncclResult_t parseAffinityPairs(const char* envName, std::vector<std::pair<std::string, std::string>>& pairs) {
  const char* env = ncclGetEnv(envName);
  if (env == NULL || env[0] == '\0') return ncclSuccess;

  std::string text(env);
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t end = text.find(';', pos);
    std::string entry = trimString(text.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
    if (!entry.empty()) {
      size_t eq = entry.find('=');
      if (eq == std::string::npos) {
        WARN("%s entry '%s' is missing '='", envName, entry.c_str());
        return ncclInvalidUsage;
      }
      std::string key = trimString(entry.substr(0, eq));
      std::string value = trimString(entry.substr(eq + 1));
      if (key.empty() || value.empty()) {
        WARN("%s entry '%s' has empty key or value", envName, entry.c_str());
        return ncclInvalidUsage;
      }
      pairs.push_back(std::make_pair(key, value));
    }
    if (end == std::string::npos) break;
    pos = end + 1;
  }
  if (pairs.empty()) {
    WARN("%s is set but contains no entries", envName);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

static ncclResult_t validateUniqueAffinityKeys(const char* envName,
                                               std::vector<std::pair<std::string, std::string>> const& pairs) {
  for (size_t i = 0; i < pairs.size(); i++) {
    for (size_t j = i + 1; j < pairs.size(); j++) {
      if (pairs[i].first == pairs[j].first) {
        WARN("%s has multiple entries for key '%s'", envName, pairs[i].first.c_str());
        return ncclInvalidUsage;
      }
    }
  }
  return ncclSuccess;
}

static bool parseCudaAffinityKey(std::string const& key, int* cudaDev) {
  const std::string prefix = "cuda:";
  if (key.compare(0, prefix.size(), prefix) != 0 || key.size() == prefix.size()) return false;

  long value = 0;
  for (size_t i = prefix.size(); i < key.size(); i++) {
    if (!std::isdigit(static_cast<unsigned char>(key[i]))) return false;
    value = value * 10 + (key[i] - '0');
    if (value > INT_MAX) return false;
  }
  *cudaDev = (int)value;
  return true;
}

static ncclResult_t validateGpuAffinityEntries(std::vector<std::pair<std::string, std::string>> const& entries) {
  int cudaDevCount = 0;
  CUDACHECK(cudaGetDeviceCount(&cudaDevCount));
  for (size_t i = 0; i < entries.size(); i++) {
    int cudaDev = -1;
    if (!parseCudaAffinityKey(entries[i].first, &cudaDev)) {
      WARN("NCCL_GPU_NIC_AFFINITY has invalid key '%s'; expected cuda:<logical-device-id>",
           entries[i].first.c_str());
      return ncclInvalidUsage;
    }
    if (cudaDev < 0 || cudaDev >= cudaDevCount) {
      WARN("NCCL_GPU_NIC_AFFINITY references unknown CUDA logical device cuda:%d; visible device count is %d",
           cudaDev, cudaDevCount);
      return ncclInvalidUsage;
    }
  }
  return ncclSuccess;
}

static ncclResult_t splitNicList(const char* envName, std::string const& list, std::vector<std::string>& names) {
  size_t pos = 0;
  while (pos <= list.size()) {
    size_t end = list.find(',', pos);
    std::string name = trimString(list.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
    if (name.empty()) {
      WARN("%s contains an empty NIC name in '%s'", envName, list.c_str());
      return ncclInvalidUsage;
    }
    if (std::find(names.begin(), names.end(), name) != names.end()) {
      WARN("%s contains duplicate NIC name '%s'", envName, name.c_str());
      return ncclInvalidUsage;
    }
    names.push_back(name);
    if (end == std::string::npos) break;
    pos = end + 1;
  }
  return names.empty() ? ncclInvalidUsage : ncclSuccess;
}

static ncclResult_t getGinDevNames(struct ncclGinState* ginState, int ndev, std::vector<ncclGinDevName>& ginDevNames) {
  for (int d = 0; d < ndev; d++) {
    ncclNetProperties_t props;
    NCCLCHECK(ginState->ncclGin->getProperties(d, &props));
    if (props.name == NULL || props.name[0] == '\0') {
      WARN("GIN device %d did not report a valid name", d);
      return ncclInvalidUsage;
    }
    ncclGinDevName devName;
    devName.dev = d;
    snprintf(devName.name, sizeof(devName.name), "%s", props.name);
    ginDevNames.push_back(devName);
  }
  return ncclSuccess;
}

static int findGinDevByName(std::vector<ncclGinDevName> const& ginDevNames, std::string const& name) {
  for (size_t i = 0; i < ginDevNames.size(); i++) {
    if (name == ginDevNames[i].name) return ginDevNames[i].dev;
  }
  return -1;
}

static ncclResult_t validateGpuAffinityNicLists(
  std::vector<std::pair<std::string, std::string>> const& entries,
  std::vector<ncclGinDevName> const& ginDevNames) {
  for (size_t e = 0; e < entries.size(); e++) {
    std::vector<std::string> nicNames;
    NCCLCHECK(splitNicList("NCCL_GPU_NIC_AFFINITY", entries[e].second, nicNames));
    if (nicNames.size() > NCCL_TOPO_MAX_NODES) {
      WARN("NCCL_GPU_NIC_AFFINITY selects too many NICs for %s", entries[e].first.c_str());
      return ncclInvalidUsage;
    }
    for (size_t i = 0; i < nicNames.size(); i++) {
      if (findGinDevByName(ginDevNames, nicNames[i]) < 0) {
        WARN("NCCL_GPU_NIC_AFFINITY selects unknown or non-GIN NIC '%s' for %s", nicNames[i].c_str(),
             entries[e].first.c_str());
        return ncclInvalidUsage;
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t applyGpuNicAffinity(struct ncclComm* comm, int ndev, int* localGinDevs, int* nLocalGinDevs) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  ginState->gpuNicAffinityEnabled = false;

  const char* env = ncclGetEnv("NCCL_GPU_NIC_AFFINITY");
  if (env == NULL || env[0] == '\0') return ncclSuccess;

  std::vector<std::pair<std::string, std::string>> entries;
  std::vector<ncclGinDevName> ginDevNames;
  std::string cudaKey = "cuda:" + std::to_string(comm->cudaDev);
  std::string selectedList;

  NCCLCHECK(parseAffinityPairs("NCCL_GPU_NIC_AFFINITY", entries));
  NCCLCHECK(validateUniqueAffinityKeys("NCCL_GPU_NIC_AFFINITY", entries));
  NCCLCHECK(validateGpuAffinityEntries(entries));
  NCCLCHECK(getGinDevNames(ginState, ndev, ginDevNames));
  NCCLCHECK(validateGpuAffinityNicLists(entries, ginDevNames));

  for (size_t i = 0; i < entries.size(); i++) {
    if (entries[i].first == cudaKey) {
      selectedList = entries[i].second;
    }
  }
  if (selectedList.empty()) return ncclSuccess;

  std::vector<std::string> nicNames;
  NCCLCHECK(splitNicList("NCCL_GPU_NIC_AFFINITY", selectedList, nicNames));
  if (nicNames.size() > NCCL_TOPO_MAX_NODES) {
    WARN("NCCL_GPU_NIC_AFFINITY selects too many NICs for %s", cudaKey.c_str());
    return ncclInvalidUsage;
  }
  for (size_t i = 0; i < nicNames.size(); i++) {
    int dev = findGinDevByName(ginDevNames, nicNames[i]);
    if (dev < 0) {
      WARN("NCCL_GPU_NIC_AFFINITY selects unknown or non-GIN NIC '%s' for %s", nicNames[i].c_str(), cudaKey.c_str());
      return ncclInvalidUsage;
    }
    localGinDevs[i] = dev;
  }
  *nLocalGinDevs = nicNames.size();
  ginState->gpuNicAffinityEnabled = true;
  INFO(NCCL_INIT | NCCL_NET, "GIN affinity: %s selected %d NICs from NCCL_GPU_NIC_AFFINITY", cudaKey.c_str(),
       *nLocalGinDevs);
  return ncclSuccess;
}

static void fillLocalGinRankInfo(struct ncclGinState* ginState, ncclGinRankInfo* info) {
  memset(info, 0, sizeof(*info));
  info->connCount = ginState->ginCommCount;
  for (int n = 0; n < ginState->ginCommCount; n++) {
    info->conns[n].ginDev = ginState->localGinDevs[n];
    info->conns[n].guid = ginState->ginProps[n].guid;
    info->conns[n].planeId = ginState->ginProps[n].planeId;
    snprintf(info->conns[n].name, sizeof(info->conns[n].name), "%s", ginState->localGinNames[n]);
  }
}

static int ginRankToWorldRank(struct ncclComm* comm, ncclGinConnectionType_t connectionType, int ginRank) {
  if (connectionType == NCCL_GIN_CONNECTION_FULL) return ginRank;
  ncclTeam_t railTeam = ncclTeamRail(comm);
  return ncclTeamRankToWorld(comm, railTeam, ginRank);
}

static bool ginPeerOnSameNode(struct ncclComm* comm, int peerWorldRank) {
  if (peerWorldRank == comm->rank) return true;
  if (comm->rankToNode != NULL) return comm->rankToNode[peerWorldRank] == comm->node;
  return comm->peerInfo[peerWorldRank].hostHash == comm->peerInfo[comm->rank].hostHash;
}

static int findConnByName(ncclGinRankInfo const* info, char const* name, int connCount) {
  for (int c = 0; c < connCount; c++) {
    if (strcmp(info->conns[c].name, name) == 0) return c;
  }
  return -1;
}

static std::string joinNicList(std::vector<std::string> const& names) {
  std::string joined;
  for (size_t i = 0; i < names.size(); i++) {
    if (i != 0) joined += ",";
    joined += names[i];
  }
  return joined;
}

static ncclResult_t getPeerAffinityTargets(std::vector<std::pair<std::string, std::string>> const& entries,
                                           char const* localName, std::vector<std::string>* targets) {
  bool found = false;
  for (size_t i = 0; i < entries.size(); i++) {
    if (entries[i].first == localName) {
      if (found) {
        WARN("NCCL_NIC_PEER_AFFINITY has multiple entries for NIC '%s'", localName);
        return ncclInvalidUsage;
      }
      NCCLCHECK(splitNicList("NCCL_NIC_PEER_AFFINITY", entries[i].second, *targets));
      found = true;
    }
  }
  if (!found) {
    WARN("NCCL_NIC_PEER_AFFINITY is missing an entry for selected NIC '%s'", localName);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

static ncclResult_t selectPeerAffinityConn(std::vector<std::pair<std::string, std::string>> const& entries,
                                           char const* localName, ncclGinRankInfo const* peerInfo, int connCount,
                                           int peerWorld, int* remoteConn, std::string* targetName) {
  std::vector<std::string> targets;
  NCCLCHECK(getPeerAffinityTargets(entries, localName, &targets));
  for (size_t i = 0; i < targets.size(); i++) {
    int conn = findConnByName(peerInfo, targets[i].c_str(), connCount);
    if (conn >= 0) {
      *remoteConn = conn;
      *targetName = targets[i];
      return ncclSuccess;
    }
  }
  std::string targetList = joinNicList(targets);
  WARN("NCCL_NIC_PEER_AFFINITY maps local NIC '%s' to peer NIC candidates '%s', but peer world rank %d did not select "
       "any of them.",
       localName, targetList.c_str(), peerWorld);
  return ncclInvalidUsage;
}

static ncclResult_t validatePeerAffinitySelectedGinNames(
  struct ncclComm* comm, struct ncclGinState* ginState, int nGinRanks, ncclGinRankInfo* allRankInfo,
  std::vector<std::pair<std::string, std::string>> const& peerEntries) {
  for (int peer = 0; peer < nGinRanks; peer++) {
    int peerWorld = ginRankToWorldRank(comm, ginState->ginConnectionType, peer);
    ncclGinRankInfo* peerInfo = allRankInfo + peerWorld;
    if (peerInfo->connCount < ginState->ginCommCount) {
      WARN("GIN affinity metadata mismatch for peer world rank %d: peer connCount=%d local connCount=%d", peerWorld,
           peerInfo->connCount, ginState->ginCommCount);
      return ncclInternalError;
    }
    for (int a = 0; a < ginState->ginCommCount; a++) {
      for (int b = a + 1; b < ginState->ginCommCount; b++) {
        if (strcmp(peerInfo->conns[a].name, peerInfo->conns[b].name) == 0) {
          WARN("NCCL_NIC_PEER_AFFINITY requires unique selected GIN NICs, but peer world rank %d connection %d and %d "
               "both use '%s'",
               peerWorld, a, b, peerInfo->conns[a].name);
          return ncclInvalidUsage;
        }
      }
    }
    for (int c = 0; c < ginState->ginCommCount; c++) {
      std::vector<std::string> unusedTargets;
      NCCLCHECK(getPeerAffinityTargets(peerEntries, peerInfo->conns[c].name, &unusedTargets));
    }
  }
  return ncclSuccess;
}

static ncclResult_t buildRemoteConnByPeer(struct ncclComm* comm, int nGinRanks, int myGinRank,
                                          ncclGinRankInfo* allRankInfo) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  const char* peerEnv = ncclGetEnv("NCCL_NIC_PEER_AFFINITY");
  const bool peerAffinityEnabled = (peerEnv != NULL && peerEnv[0] != '\0');
  std::vector<std::pair<std::string, std::string>> peerEntries;

  if (ginState->remoteConnByPeer) {
    free(ginState->remoteConnByPeer);
    ginState->remoteConnByPeer = NULL;
  }
  ginState->nicPeerAffinityEnabled = false;
  ginState->ginRankCount = nGinRanks;
  NCCLCHECK(ncclCalloc(&ginState->remoteConnByPeer, ginState->ginCommCount * nGinRanks));
  for (int lc = 0; lc < ginState->ginCommCount; lc++) {
    for (int peer = 0; peer < nGinRanks; peer++) {
      ginState->remoteConnByPeer[lc * nGinRanks + peer] = lc;
    }
  }

  if (peerAffinityEnabled) {
    if (ginState->ncclGin != &ncclGinIbGdaki) {
      WARN("NCCL_NIC_PEER_AFFINITY is supported only by the internal GIN_IB_GDAKI backend.");
      return ncclInvalidUsage;
    }
    NCCLCHECK(parseAffinityPairs("NCCL_NIC_PEER_AFFINITY", peerEntries));
    NCCLCHECK(validateUniqueAffinityKeys("NCCL_NIC_PEER_AFFINITY", peerEntries));
    NCCLCHECK(validatePeerAffinitySelectedGinNames(comm, ginState, nGinRanks, allRankInfo, peerEntries));
    ginState->nicPeerAffinityEnabled = true;

    for (int peer = 0; peer < nGinRanks; peer++) {
      int peerWorld = ginRankToWorldRank(comm, ginState->ginConnectionType, peer);
      ncclGinRankInfo* peerInfo = allRankInfo + peerWorld;
      if (peerInfo->connCount < ginState->ginCommCount) {
        WARN("GIN affinity metadata mismatch for peer world rank %d: peer connCount=%d local connCount=%d", peerWorld,
             peerInfo->connCount, ginState->ginCommCount);
        return ncclInternalError;
      }
      for (int lc = 0; lc < ginState->ginCommCount; lc++) {
        if (ginPeerOnSameNode(comm, peerWorld)) {
          ginState->remoteConnByPeer[lc * nGinRanks + peer] = lc;
          continue;
        }
        std::string targetName;
        int remoteConn = -1;
        NCCLCHECK(selectPeerAffinityConn(peerEntries, ginState->localGinNames[lc], peerInfo, ginState->ginCommCount,
                                         peerWorld, &remoteConn, &targetName));

        std::string reciprocalTarget;
        int reciprocalConn = -1;
        ncclGinRankInfo* localInfo = allRankInfo + comm->rank;
        NCCLCHECK(selectPeerAffinityConn(peerEntries, peerInfo->conns[remoteConn].name, localInfo,
                                         ginState->ginCommCount, comm->rank, &reciprocalConn, &reciprocalTarget));
        if (reciprocalConn != lc) {
          WARN("NCCL_NIC_PEER_AFFINITY must be reciprocal: local '%s' -> peer '%s', but peer '%s' selects local '%s'",
               ginState->localGinNames[lc], targetName.c_str(), peerInfo->conns[remoteConn].name,
               reciprocalTarget.c_str());
          return ncclInvalidUsage;
        }
        ginState->remoteConnByPeer[lc * nGinRanks + peer] = remoteConn;
      }
    }
  }

  if (ginState->gpuNicAffinityEnabled || ginState->nicPeerAffinityEnabled) {
    for (int lc = 0; lc < ginState->ginCommCount; lc++) {
      INFO(NCCL_INIT | NCCL_NET, "GIN affinity: rank %d ginRank %d localConn %d localNic %s dev %d plane %d",
           comm->rank, myGinRank, lc, ginState->localGinNames[lc], ginState->localGinDevs[lc],
           ginState->ginProps[lc].planeId);
      for (int peer = 0; peer < nGinRanks; peer++) {
        int peerWorld = ginRankToWorldRank(comm, ginState->ginConnectionType, peer);
        int remoteConn = ginState->remoteConnByPeer[lc * nGinRanks + peer];
        INFO(NCCL_INIT | NCCL_NET,
             "GIN affinity: rank %d localConn %d localNic %s -> peerWorld %d peerGinRank %d remoteConn %d remoteNic %s",
             comm->rank, lc, ginState->localGinNames[lc], peerWorld, peer, remoteConn,
             allRankInfo[peerWorld].conns[remoteConn].name);
      }
    }
  }
  return ncclSuccess;
}

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
  NCCLCHECK(applyGpuNicAffinity(comm, ndev, localGinDevs, &nLocalGinDevs));
  ginState->nicPeerAffinityEnabled = false;
  ginState->ginRankCount = 0;

  void** handles = NULL;
  char* allHandles = NULL;
  ncclGinRankInfo* allRankInfo = NULL;

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
  if (ginState->ginCommCount <= 0) {
    WARN("No GIN connections can be created.");
    ret = ncclInternalError;
    goto fail;
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
    int localDev = localGinDevs[n % nLocalGinDevs];
    ginState->localGinDevs[n] = localDev;
    NCCLCHECKGOTO(ginState->ncclGin->getProperties(localDev, ginState->ginProps + n), ret, fail);
    if (ginState->ginProps[n].name == NULL || ginState->ginProps[n].name[0] == '\0') {
      WARN("GIN device %d did not report a valid name", localDev);
      ret = ncclInvalidUsage;
      goto fail;
    }
    snprintf(ginState->localGinNames[n], sizeof(ginState->localGinNames[n]), "%s", ginState->ginProps[n].name);
  }

  if (envIsSet("NCCL_GPU_NIC_AFFINITY") || envIsSet("NCCL_NIC_PEER_AFFINITY")) {
    NCCLCHECKGOTO(ncclCalloc(&allRankInfo, comm->nRanks), ret, fail);
    fillLocalGinRankInfo(ginState, allRankInfo + comm->rank);
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allRankInfo, sizeof(ncclGinRankInfo)), ret, fail);
    NCCLCHECKGOTO(buildRemoteConnByPeer(comm, nGinRanks, myGinRank, allRankInfo), ret, fail);
  }

  for (int n = 0; n < ginState->ginCommCount; n++) {
    void* listenComm;
    NCCLCHECKGOTO(ginState->ncclGin->listen(ginState->ginInstance, ginState->localGinDevs[n],
                                            allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm),
                  ret, fail);

    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret, fail);

    NCCLCHECKGOTO(ginState->ncclGin->connect(comm->ginContext, handles, nGinRanks, myGinRank, listenComm,
                                             ginState->ginComms + n),
                  ret, fail);

    NCCLCHECKGOTO(ginState->ncclGin->closeListen(listenComm), ret, fail);
  }
  free(handles);
  handles = NULL;
  free(allHandles);
  allHandles = NULL;
  free(ginCommCountHandles);
  ginCommCountHandles = NULL;
  free(allRankInfo);
  allRankInfo = NULL;

exit:
  if (ret == ncclSuccess) ginState->connected = true;
  return ret;
fail:
  if (allHandles) free(allHandles);
  if (handles) free(handles);
  if (ginCommCountHandles) free(ginCommCountHandles);
  if (allRankInfo) free(allRankInfo);
  if (ginState->remoteConnByPeer) {
    free(ginState->remoteConnByPeer);
    ginState->remoteConnByPeer = NULL;
  }
  goto exit;
}

ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                 struct ncclDevComm* devComm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  if (reqs->ginStrongSignalsRequired && !ginState->supportsStrongSignals) {
    WARN("GIN strong signals are required, but the GIN plugin does not support them.");
    return ncclInvalidUsage;
  }

  if (reqs->ginVaSignalsRequired && !ginState->supportsVASignals) {
    WARN("GIN VA signals are required, but the GIN plugin does not support them.");
    return ncclInvalidUsage;
  }

  devComm->ginSignalCount = reqs->ginSignalCount;
  devComm->ginCounterCount = reqs->ginCounterCount;
  // Legacy signals default to what is specified in DevCommRequirements
  devComm->ginStrongLegacySignals = reqs->ginStrongSignalsRequired;

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
  INFO(NCCL_INIT,
       "devCommCreate: creating %d contexts: %d GIN connections with %d contexts each (%d contexts total requested)",
       nContextsTotal, ginState->ginCommCount, nContextsPerComm, reqs->ginContextCount);

  struct ncclGinStateDevComm* ginStateDevComm = NULL;
  NCCLCHECK(ncclCalloc(&ginStateDevComm, 1));
  ginStateDevComm->contextCount = nContextsTotal;

  const int* backendVersionArray;
  int nVersions;
  switch (ginState->ginType) {
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
  default:
    WARN("Cannot get backend version for invalid GIN type %d", ginState->ginType);
    return ncclInternalError;
  }

  int backendVersion = 0;
  if (ncclParamDevApiJit() == 1) {
    // JIT: device code version is the latest version.
    backendVersion = nVersions - 1;
  } else {
    // Non-JIT: device code version matches reqs->version.
    for (int i = 0; i < nVersions; i++) {
      if (reqs->version >= backendVersionArray[i]) backendVersion = i;
      else break;
    }
  }

  ncclResult_t ret = ncclSuccess;

  ncclGinConfig_t ginConfig = {
    reqs->ginSignalCount,
    reqs->ginCounterCount,
    nContextsPerComm,
    reqs->ginQueueDepth,
    reqs->ginTrafficClass != NCCL_CONFIG_UNDEF_INT ? reqs->ginTrafficClass : comm->config.trafficClass,
    backendVersion,
    reqs->ginConnectionType == NCCL_GIN_CONNECTION_RAIL && ginState->ginConnectionType == NCCL_GIN_CONNECTION_FULL ?
      comm->devrState.lsaSize :
      1
  };

  if (ginState->nicPeerAffinityEnabled) {
    if (ginState->ncclGin != &ncclGinIbGdaki) {
      WARN("NCCL_NIC_PEER_AFFINITY is supported only by the internal GIN_IB_GDAKI backend.");
      ret = ncclInvalidUsage;
      goto end;
    }
    NCCLCHECKGOTO(ncclGinIbGdakiCreateContextGroup(ginState->ginComms, ginState->ginCommCount, &ginConfig,
                                                   ginState->remoteConnByPeer, ginStateDevComm->ginCtx,
                                                   ginStateDevComm->devHandles),
                  ret, end);
  } else {
    for (int n = 0; n < ginState->ginCommCount; n++) {
      NCCLCHECKGOTO(ginState->ncclGin->createContext(ginState->ginComms[n], &ginConfig, &ginStateDevComm->ginCtx[n],
                                                     &ginStateDevComm->devHandles[n]),
                    ret, end);
    }
  }

  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginStateDevComm->ginCtx[n] == NULL || ginStateDevComm->devHandles[n] == NULL ||
        ginStateDevComm->devHandles[n]->handle == NULL) {
      WARN("GIN plugin %s returned invalid context for connection %d: ginCtx=%p devHandle=%p handle=%p",
           ginState->ncclGin->name, n, ginStateDevComm->ginCtx[n], ginStateDevComm->devHandles[n],
           ginStateDevComm->devHandles[n] ? ginStateDevComm->devHandles[n]->handle : NULL);
      ret = ncclInternalError;
      goto end;
    }
    devComm->ginNetDeviceTypes[n] = ginStateDevComm->devHandles[n]->netDeviceType;
    devComm->ginHandles[n] = ginStateDevComm->devHandles[n]->handle;
    if (ginStateDevComm->devHandles[n]->needsProxyProgress) ginState->needsProxyProgress = 1;
  }

  if (ginState->needsProxyProgress && ginState->ginProgress == 0) {
    ginState->cpuAffinity = comm->cpuAffinity;
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
    for (int n = 0; n < ginState->ginCommCount; n++) {
      if (ginStateDevComm->ginCtx[n]) ginState->ncclGin->destroyContext(ginStateDevComm->ginCtx[n]);
    }
    free(ginStateDevComm);
  }
  return ret;
}

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
  if (ginState->remoteConnByPeer) {
    free(ginState->remoteConnByPeer);
    ginState->remoteConnByPeer = NULL;
  }
  memset((void*)ginState, 0, sizeof(*ginState));
  return ncclSuccess;
}

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags, bool multiSegment,
                             int memType) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (multiSegment) {
    // Multi-segment GIN registration requires DMABUF support on all GIN connections
    for (int n = 0; n < ginState->ginCommCount; n++) {
      if (!(ginState->ginProps[n].ptrSupport & NCCL_PTR_DMABUF)) {
        WARN("Window registration of addresses that span multiple physical segments requires DMABUF support with GIN.");
        return ncclInvalidArgument;
      }
    }
  }
  int mrFlags = (winFlags & NCCL_WIN_STRICT_ORDERING) ? NCCL_NET_MR_FLAG_FORCE_SO : 0;
  if (ginState->nicPeerAffinityEnabled) {
    if (ginState->ncclGin != &ncclGinIbGdaki) {
      WARN("NCCL_NIC_PEER_AFFINITY is supported only by the internal GIN_IB_GDAKI backend.");
      return ncclInvalidUsage;
    }
    NCCLCHECK(ncclGinIbGdakiRegMrSymGroup(ginState->ginComms, ginState->ginCommCount, ginState->remoteConnByPeer,
                                          address, size, memType, mrFlags, ginHostWins, (void**)ginDevWins));
    for (int n = 0; n < ginState->ginCommCount; n++) {
      if (ginHostWins[n] == NULL) {
        WARN("rank %d - GIN Symmetric register failed: buff %p, size %ld", comm->rank, address, size);
        return ncclSystemError;
      }
    }
    return ncclSuccess;
  }

  for (int n = 0; n < ginState->ginCommCount; n++) {
    NCCLCHECK(ginState->ncclGin->regMrSym(ginState->ginComms[n], address, size, memType, mrFlags, &ginHostWins[n],
                                          &ginDevWins[n]));
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
