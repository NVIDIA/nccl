/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "debug.h"
#include "os.h"

#if defined(NCCL_OS_LINUX)

#include "alloc.h"
#include "bootstrap.h"
#include "checks.h"
#include "diagnostics_log.h"
#include "p2p.h"
#include "graph.h"
#include "graph/topo.h"
#include "../include/p2p.h"
#include "transport.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

enum ncclDiagP2pHandlePath {
  ncclDiagP2pHandleDirect = 1,
  ncclDiagP2pHandleLegacyIpc = 2,
  ncclDiagP2pHandleCuMemPosixFd = 3,
  ncclDiagP2pHandleCuMemFabric = 4,
  ncclDiagP2pHandleCuMemOther = 5,
};

enum ncclDiagP2pReason {
  ncclDiagP2pReasonNone = 0,
  ncclDiagP2pReasonIndirect = 1,
  ncclDiagP2pReasonNoDescriptor = 2,
  ncclDiagP2pReasonImport = 3,
  ncclDiagP2pReasonWriteLaunch = 4,
  ncclDiagP2pReasonWriteMismatch = 5,
  ncclDiagP2pReasonReadLaunch = 6,
  ncclDiagP2pReasonReadMismatch = 7,
  ncclDiagP2pReasonTopo = 8,
  ncclDiagP2pReasonLocalCuda = 9,
};

static constexpr int kDiagP2pBarrierWrote = 0xd1a601;
static constexpr int kDiagP2pBarrierReported = 0xd1a603;
static constexpr int kDiagP2pBarrierImportsFreed = 0xd1a604;

struct ncclDiagP2pEdgeInfo {
  int p2p;
  int read;
  int pathType;
  int sameProcess;
  int handleType;
};

struct ncclDiagP2pMemDesc {
  int valid;
  size_t bytes;
  uintptr_t directPtr;
  ncclIpcDesc ipcDesc;
};

struct ncclDiagP2pEdgeResult {
  int tested;
  int reason;
  uint64_t writeGot;
  uint64_t verifyGot;
  uint64_t readGot;
};

struct ncclDiagP2pWriteObs {
  uint64_t writeValue;
  uint64_t verifyValue;
};

struct ncclDiagP2pSummary {
  uint64_t tested;
  uint64_t passed;
  uint64_t skipped;
};

struct ncclDiagP2pMapping {
  void* ptr;
  int active;
  int sameProcessCuMem;
  int crossProcessCuMem;
  int legacyIpc;
  int tracked;
  int peerAccessEnabled;
  int peerAccessDev;
};

static ncclResult_t ncclDiagP2pBuildRankSet(struct ncclComm* comm, int** ranks, int* rank, int* nRanks) {
  // The topology is fused over the local ranks normally and over the MNNVL clique when MNNVL is active. Cross-clique
  // P2P expands that set to every rank in the same NVLink fabric domain.
  if (!comm->p2pCrossClique) {
    *ranks = comm->MNNVL ? comm->clique.ranks : comm->localRankToRank;
    *rank = comm->MNNVL ? comm->cliqueRank : comm->localRank;
    *nRanks = comm->MNNVL ? comm->clique.size : comm->localRanks;
    return ncclSuccess;
  }

  if (comm->nvlDomainSize <= 0) return ncclInternalError;
  NCCLCHECK(ncclCalloc(ranks, comm->nvlDomainSize));
  *rank = -1;
  *nRanks = comm->nvlDomainSize;
  int slot = 0;
  for (int peer = 0; peer < comm->nRanks; peer++) {
    if (memcmp(comm->peerInfo[comm->rank].fabricInfo.clusterUuid, comm->peerInfo[peer].fabricInfo.clusterUuid,
               sizeof(comm->peerInfo[comm->rank].fabricInfo.clusterUuid)) != 0)
      continue;
    if (slot == *nRanks) return ncclInternalError;
    (*ranks)[slot] = peer;
    if (peer == comm->rank) *rank = slot;
    slot++;
  }
  return *rank >= 0 && slot == *nRanks ? ncclSuccess : ncclInternalError;
}

static int ncclDiagP2pRankToSlot(const int* ranks, int nRanks, int rank) {
  for (int slot = 0; slot < nRanks; slot++) {
    if (ranks[slot] == rank) return slot;
  }
  return -1;
}

static bool ncclDiagP2pSameProcess(struct ncclComm* comm, int srcRank, int dstRank) {
  const struct ncclPeerInfo* a = comm->peerInfo + srcRank;
  const struct ncclPeerInfo* b = comm->peerInfo + dstRank;
  return a->hostHash == b->hostHash && a->pidHash == b->pidHash;
}

static int ncclDiagP2pHandleType(struct ncclComm* comm, int srcRank, int dstRank) {
  if (ncclDiagP2pSameProcess(comm, srcRank, dstRank)) return ncclDiagP2pHandleDirect;
  if (!ncclCuMemEnable()) return ncclDiagP2pHandleLegacyIpc;
#if CUDART_VERSION >= 11030
  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) return ncclDiagP2pHandleCuMemPosixFd;
  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_FABRIC) return ncclDiagP2pHandleCuMemFabric;
#endif
  return ncclDiagP2pHandleCuMemOther;
}

static const char* ncclDiagP2pHandleName(int handleType) {
  switch (handleType) {
  case ncclDiagP2pHandleDirect:
    return "DIRECT";
  case ncclDiagP2pHandleLegacyIpc:
    return "LEGACY_CUDA_IPC";
  case ncclDiagP2pHandleCuMemPosixFd:
    return "CUMEM_POSIX_FD";
  case ncclDiagP2pHandleCuMemFabric:
    return "CUMEM_FABRIC";
  case ncclDiagP2pHandleCuMemOther:
    return "CUMEM_OTHER";
  default:
    return "NONE";
  }
}

static const char* ncclDiagP2pReasonName(int reason) {
  switch (reason) {
  case ncclDiagP2pReasonIndirect:
    return "indirect";
  case ncclDiagP2pReasonNoDescriptor:
    return "noDescriptor";
  case ncclDiagP2pReasonImport:
    return "import";
  case ncclDiagP2pReasonWriteLaunch:
    return "writeLaunch";
  case ncclDiagP2pReasonWriteMismatch:
    return "writeMismatch";
  case ncclDiagP2pReasonReadLaunch:
    return "readLaunch";
  case ncclDiagP2pReasonReadMismatch:
    return "readMismatch";
  case ncclDiagP2pReasonTopo:
    return "topo";
  case ncclDiagP2pReasonLocalCuda:
    return "localCuda";
  default:
    return "none";
  }
}

static const char* ncclDiagP2pPathName(int pathType) {
  return (pathType >= PATH_LOC && pathType <= PATH_DIS) ? topoPathTypeStr[pathType] : "UNK";
}

static bool ncclDiagP2pIsFabricEdge(const struct ncclDiagP2pEdgeInfo* edge) {
  // MNNVL edges use a fabric handle even when their topology path is reported as NVLink.
  return edge->handleType == ncclDiagP2pHandleCuMemFabric || edge->pathType == PATH_NET;
}

static const char* ncclDiagP2pEdgeAdvice(const struct ncclDiagP2pEdgeInfo* edge) {
  if (ncclDiagP2pIsFabricEdge(edge)) {
    return "check the IMEX domain with 'nvidia-imex-ctl -H -N' (nodes READY, connectivity C) and verify access to "
           "/dev/nvidia-caps-imex-channels/channel*";
  }

  switch (edge->pathType) {
  case PATH_NVL:
  case PATH_NVB:
    return "check the single-node NVLink topology and peer-access state with 'nvidia-smi topo -m' and "
           "'nvidia-smi topo -p2p n'";
  case PATH_PIX:
  case PATH_PXB:
  case PATH_PHB:
  case PATH_SYS:
    return "check the affected pair with 'nvidia-smi topo -p2p p', then check Linux bare-metal IOMMU mode and PCIe "
           "ACS settings";
  default:
    return "inspect the affected GPU pair with 'nvidia-smi topo -m' and the applicable 'nvidia-smi topo -p2p' check";
  }
}

static const char* ncclDiagP2pImportAdvice(const struct ncclDiagP2pEdgeInfo* edge) {
  if (ncclDiagP2pIsFabricEdge(edge)) return ncclDiagP2pEdgeAdvice(edge);

  switch (edge->handleType) {
  case ncclDiagP2pHandleDirect:
    return "inspect preceding CUDA peer-access or virtual-memory mapping errors on the source rank";
  case ncclDiagP2pHandleLegacyIpc:
    return "check CUDA IPC support, GPU visibility, and process or container isolation";
  case ncclDiagP2pHandleCuMemPosixFd:
    return "check cuMem POSIX-FD sharing support and process or container permissions";
  default:
    return "check CUDA virtual-memory handle support and permissions between the processes";
  }
}

static int ncclDiagP2pPathType(struct ncclComm* comm, int srcRank, int dstRank) {
  int srcGpuIdx = -1;
  int dstGpuIdx = -1;
  if (comm->topo == nullptr) return PATH_DIS;
  if (ncclTopoRankToIndex(comm->topo, srcRank, &srcGpuIdx, false) != ncclSuccess ||
      ncclTopoRankToIndex(comm->topo, dstRank, &dstGpuIdx, false) != ncclSuccess) {
    return PATH_DIS;
  }
  int pathType = comm->topo->nodes[GPU].nodes[srcGpuIdx].paths[GPU][dstGpuIdx].type;
  return (pathType >= 0 && pathType <= PATH_DIS) ? pathType : PATH_DIS;
}

static void ncclDiagP2pSetReason(struct ncclDiagP2pEdgeResult* result, int reason) {
  if (result->reason == ncclDiagP2pReasonNone) result->reason = reason;
}

static void ncclDiagP2pSetLocalReason(int rank, int nRanks, struct ncclDiagP2pEdgeResult* allResults, int reason) {
  for (int dst = 0; dst < nRanks; dst++) {
    struct ncclDiagP2pEdgeResult* result = allResults + rank * nRanks + dst;
    if (!result->tested) continue;
    ncclDiagP2pSetReason(result, reason);
  }
}

static bool ncclDiagP2pCudaSuccess(cudaError_t err, const char* phase) {
  if (err == cudaSuccess) return true;
  WARN("Diagnostics P2P %s CUDA failure: %s", phase, cudaGetErrorString(err));
  (void)cudaGetLastError();
  return false;
}

static bool ncclDiagP2pNcclSuccess(ncclResult_t result, const char* phase) {
  if (result == ncclSuccess) return true;
  WARN("Diagnostics P2P %s returned %d", phase, result);
  return false;
}

static void ncclDiagP2pLogEdge(struct ncclComm* comm, const char* phase, int srcRank, int dstRank,
                               const struct ncclDiagP2pEdgeInfo* edge, const char* extra) {
  const struct ncclPeerInfo* srcInfo = comm->peerInfo + srcRank;
  const struct ncclPeerInfo* dstInfo = comm->peerInfo + dstRank;
  INFO(NCCL_INIT,
       "Diagnostics P2P %s srcRank=%d srcCudaDev=%d srcNvmlDev=%d dstRank=%d dstCudaDev=%d dstNvmlDev=%d path=%s "
       "handle=%s topoRead=%d%s%s",
       phase, srcRank, srcInfo->cudaDev, srcInfo->nvmlDev, dstRank, dstInfo->cudaDev, dstInfo->nvmlDev,
       ncclDiagP2pPathName(edge->pathType), ncclDiagP2pHandleName(edge->handleType), edge->read,
       extra == nullptr ? "" : " ", extra == nullptr ? "" : extra);
}

static void ncclDiagP2pFormatPeerFields(struct ncclComm* comm, int srcRank, int dstRank,
                                        const struct ncclDiagP2pEdgeInfo* edge, char* buf, size_t size) {
  const struct ncclPeerInfo* srcInfo = comm->peerInfo + srcRank;
  const struct ncclPeerInfo* dstInfo = comm->peerInfo + dstRank;
  snprintf(buf, size, "srcRank=%d srcCudaDev=%d srcNvmlDev=%d dstRank=%d dstCudaDev=%d dstNvmlDev=%d path=%s handle=%s",
           srcRank, srcInfo->cudaDev, srcInfo->nvmlDev, dstRank, dstInfo->cudaDev, dstInfo->nvmlDev,
           ncclDiagP2pPathName(edge->pathType), ncclDiagP2pHandleName(edge->handleType));
}

static void ncclDiagP2pReport(struct ncclComm* comm, int srcRank, int dstRank, const struct ncclDiagP2pEdgeInfo* edge,
                              const struct ncclDiagP2pEdgeResult* result) {
  char edgeFields[256];
  ncclDiagP2pFormatPeerFields(comm, srcRank, dstRank, edge, edgeFields, sizeof(edgeFields));

  if (result->reason == ncclDiagP2pReasonNoDescriptor) {
    DIAG_PRINT("NCCL DIAG [INFO] p2p: destination buffer unavailable %s reason=%s; inspect earlier allocation, export, "
               "or initialization errors on the destination rank, then %s",
               edgeFields, ncclDiagP2pReasonName(result->reason), ncclDiagP2pEdgeAdvice(edge));
    return;
  }

  if (result->reason == ncclDiagP2pReasonLocalCuda) {
    DIAG_PRINT("NCCL DIAG [INFO] p2p: local CUDA setup failed %s reason=%s; inspect preceding device, stream, "
               "allocation, or initialization errors on the source rank",
               edgeFields, ncclDiagP2pReasonName(result->reason));
    return;
  }

  if (result->reason == ncclDiagP2pReasonImport) {
    DIAG_PRINT("NCCL DIAG [INFO] p2p: peer-memory import failed %s reason=%s; %s", edgeFields,
               ncclDiagP2pReasonName(result->reason), ncclDiagP2pImportAdvice(edge));
    return;
  }

  if (result->reason == ncclDiagP2pReasonWriteMismatch) {
    DIAG_PRINT("NCCL DIAG [INFO] p2p: write mismatch %s expected=0x%016llx got=0x%016llx verify=0x%016llx; %s",
               edgeFields, (unsigned long long)ncclDiagP2pWritePattern(srcRank, dstRank),
               (unsigned long long)result->writeGot, (unsigned long long)result->verifyGot,
               ncclDiagP2pEdgeAdvice(edge));
    return;
  }

  if (result->reason == ncclDiagP2pReasonReadMismatch) {
    DIAG_PRINT("NCCL DIAG [INFO] p2p: read mismatch %s expected=0x%016llx got=0x%016llx; %s", edgeFields,
               (unsigned long long)ncclDiagP2pReadPattern(dstRank, srcRank), (unsigned long long)result->readGot,
               ncclDiagP2pEdgeAdvice(edge));
    return;
  }

  if (result->reason == ncclDiagP2pReasonTopo) {
    DIAG_PRINT("NCCL DIAG [INFO] p2p: topology check failed %s reason=%s; inspect preceding topology records, then %s",
               edgeFields, ncclDiagP2pReasonName(result->reason), ncclDiagP2pEdgeAdvice(edge));
    return;
  }

  DIAG_PRINT("NCCL DIAG [INFO] p2p: launch/check failed %s reason=%s; inspect preceding CUDA or NCCL warnings, then %s",
             edgeFields, ncclDiagP2pReasonName(result->reason), ncclDiagP2pEdgeAdvice(edge));
}

static void ncclDiagP2pBuildGroupSummary(int nRanks, const struct ncclDiagP2pEdgeResult* allResults,
                                         struct ncclDiagP2pSummary* summary) {
  *summary = {};
  for (int src = 0; src < nRanks; src++) {
    for (int dst = 0; dst < nRanks; dst++) {
      const struct ncclDiagP2pEdgeResult* result = allResults + src * nRanks + dst;
      if (result->reason == ncclDiagP2pReasonIndirect) summary->skipped++;
      if (!result->tested) continue;
      summary->tested++;
      if (result->reason == ncclDiagP2pReasonNone) summary->passed++;
    }
  }
}

static void ncclDiagP2pReportSummary(struct ncclComm* comm, const struct ncclDiagP2pSummary* summaries) {
  if (comm->rank != 0) return;

  uint64_t tested = 0;
  uint64_t passed = 0;
  uint64_t skipped = 0;
  for (int rank = 0; rank < comm->nRanks; rank++) {
    tested += summaries[rank].tested;
    passed += summaries[rank].passed;
    skipped += summaries[rank].skipped;
  }

  if (tested == 0) return;

  // This check covers topology-eligible directed GPU P2P pairs. A given collective and algorithm may use only a subset
  // of those pairs, and may use multiple channels for the same pair.
  const char* level = passed == tested ? "[OK]   " : "[INFO] ";
  char counts[64];
  if (passed == tested) snprintf(counts, sizeof(counts), "all %llu", (unsigned long long)passed);
  else snprintf(counts, sizeof(counts), "%llu/%llu", (unsigned long long)passed, (unsigned long long)tested);
  if (skipped == 0) DIAG_PRINT("NCCL DIAG %sp2p: %s directed GPU P2P edges verified", level, counts);
  else
    DIAG_PRINT("NCCL DIAG %sp2p: %s directed GPU P2P edges verified (skipped indirect=%llu)", level, counts,
               (unsigned long long)skipped);
}

static void ncclDiagP2pReportGroupFailures(struct ncclComm* comm, const int* ranks, int nRanks,
                                           const struct ncclDiagP2pEdgeInfo* edgeMatrix,
                                           struct ncclDiagP2pEdgeResult* allResults) {
  for (int src = 0; src < nRanks; src++) {
    for (int dst = 0; dst < nRanks; dst++) {
      struct ncclDiagP2pEdgeResult* result = allResults + src * nRanks + dst;
      if (!result->tested || result->reason == ncclDiagP2pReasonNone) continue;
      const struct ncclDiagP2pEdgeInfo* edge = edgeMatrix + src * nRanks + dst;
      ncclDiagP2pReport(comm, ranks[src], ranks[dst], edge, result);
    }
  }
}

static void ncclDiagP2pDiscoverLocalEdges(struct ncclComm* comm, const int* ranks, int rank, int nRanks,
                                          struct ncclDiagP2pEdgeInfo* edgeMatrix,
                                          struct ncclDiagP2pEdgeResult* allResults, int* outPeers, int* outPeerCount) {
  *outPeerCount = 0;
  for (int dstSlot = 0; dstSlot < nRanks; dstSlot++) {
    int dst = ranks[dstSlot];
    struct ncclDiagP2pEdgeInfo* edge = edgeMatrix + rank * nRanks + dstSlot;
    struct ncclDiagP2pEdgeResult* result = allResults + rank * nRanks + dstSlot;
    edge->p2p = 0;
    edge->read = 0;
    edge->pathType = (dst == comm->rank) ? PATH_LOC : ncclDiagP2pPathType(comm, comm->rank, dst);
    edge->sameProcess = ncclDiagP2pSameProcess(comm, comm->rank, dst) ? 1 : 0;
    edge->handleType = ncclDiagP2pHandleType(comm, comm->rank, dst);
    if (dst == comm->rank) continue;

    int intermediateRank = -1;
    ncclResult_t topoRet =
      ncclTopoCheckP2p(comm, comm->topo, comm->rank, dst, &edge->p2p, &edge->read, &intermediateRank, nullptr);
    if (topoRet != ncclSuccess) {
      edge->p2p = 0;
      edge->read = 0;
      result->tested = 1;
      ncclDiagP2pSetReason(result, ncclDiagP2pReasonTopo);
      INFO(NCCL_INIT, "Diagnostics P2P topo check failed srcRank=%d dstRank=%d result=%d", comm->rank, dst, topoRet);
      continue;
    }
    if (edge->p2p && intermediateRank != -1) {
      edge->p2p = 0;
      result->reason = ncclDiagP2pReasonIndirect;
      ncclDiagP2pLogEdge(comm, "skip", comm->rank, dst, edge, "reason=indirect");
    } else if (edge->p2p) {
      outPeers[(*outPeerCount)++] = dstSlot;
      result->tested = 1;
    }
  }
}

static void ncclDiagP2pBuildInboundPeers(struct ncclComm* comm, const int* ranks, int rank, int nRanks,
                                         const struct ncclDiagP2pEdgeInfo* edgeMatrix, int* inPeers, int* inPeerCount,
                                         bool* needsLocalHandle) {
  *inPeerCount = 0;
  *needsLocalHandle = false;
  for (int src = 0; src < nRanks; src++) {
    const struct ncclDiagP2pEdgeInfo* edge = edgeMatrix + src * nRanks + rank;
    if (!edge->p2p) continue;

    inPeers[(*inPeerCount)++] = src;
    if (ncclCuMemEnable() && edge->sameProcess &&
        comm->peerInfo[ranks[src]].cudaDev != comm->peerInfo[comm->rank].cudaDev) {
      *needsLocalHandle = true;
    }
  }
}

static ncclResult_t ncclDiagP2pReleaseLocalHandle(const struct ncclDiagP2pMemDesc* desc) {
#if CUDART_VERSION >= 11030
  CUCHECK(cuMemRelease(desc->ipcDesc.memHandle));
  return ncclSuccess;
#else
  return ncclInternalError;
#endif
}

static ncclResult_t ncclDiagP2pMapSameProcess(struct ncclComm* comm, int dstRank, const struct ncclDiagP2pMemDesc* desc,
                                              struct ncclDiagP2pMapping* mapping) {
  const struct ncclPeerInfo* dstInfo = comm->peerInfo + dstRank;

  if (dstInfo->cudaDev == comm->cudaDev) {
    mapping->ptr = (void*)desc->directPtr;
    mapping->active = 1;
    return ncclSuccess;
  }

  if (ncclCuMemEnable()) {
#if CUDART_VERSION >= 11030
    // VMM access is scoped to this mapping by ncclCuMemAllocAddr/cuMemSetAccess, so it does not require context-wide
    // peer access and cannot interfere with another user of the CUDA context.
    CUmemGenericAllocationHandle handle = desc->ipcDesc.memHandle;
    NCCLCHECK(ncclCuMemAllocAddr(&mapping->ptr, &handle, desc->bytes));
    mapping->active = 1;
    mapping->sameProcessCuMem = 1;
#else
    return ncclInternalError;
#endif
    return ncclSuccess;
  }

  // Legacy same-process direct pointers require context-wide peer access.
  cudaError_t err = cudaDeviceEnablePeerAccess(dstInfo->cudaDev, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    (void)cudaGetLastError();
  } else if (err != cudaSuccess) {
    INFO(NCCL_INIT, "Diagnostics: failed to enable peer access to dev %d: %s", dstInfo->cudaDev,
         cudaGetErrorString(err));
    (void)cudaGetLastError();
    return ncclUnhandledCudaError;
  } else {
    mapping->peerAccessEnabled = 1;
    mapping->peerAccessDev = dstInfo->cudaDev;
  }

  mapping->ptr = (void*)desc->directPtr;
  mapping->active = 1;
  return ncclSuccess;
}

static ncclResult_t ncclDiagP2pFreeMapping(struct ncclComm* comm, struct ncclDiagP2pMapping* mapping) {
  if (mapping == nullptr) return ncclSuccess;

  ncclResult_t ret = ncclSuccess;
  if (mapping->active && mapping->ptr != nullptr) {
    if (mapping->sameProcessCuMem) {
      NCCLCHECKIGNORE(ncclCuMemFreeAddr(mapping->ptr, nullptr), ret);
    } else if (mapping->crossProcessCuMem) {
      NCCLCHECKIGNORE(ncclCudaFree(mapping->ptr, mapping->tracked ? comm->memManager : nullptr), ret);
    } else if (mapping->legacyIpc) {
      cudaError_t err = cudaIpcCloseMemHandle(mapping->ptr);
      if (err != cudaSuccess) {
        if (ret == ncclSuccess) ret = ncclUnhandledCudaError;
        (void)cudaGetLastError();
      }
    }
  }

  if (mapping->peerAccessEnabled) {
    cudaError_t err = cudaDeviceDisablePeerAccess(mapping->peerAccessDev);
    if (err == cudaErrorPeerAccessNotEnabled) {
      (void)cudaGetLastError();
    } else if (err != cudaSuccess) {
      if (ret == ncclSuccess) ret = ncclUnhandledCudaError;
      (void)cudaGetLastError();
    }
  }

  mapping->ptr = nullptr;
  mapping->active = 0;
  mapping->peerAccessEnabled = 0;
  return ret;
}

static void ncclDiagP2pImportMappings(struct ncclComm* comm, const int* ranks, int rank, int nRanks, int outPeerCount,
                                      const int* outPeers, const struct ncclDiagP2pEdgeInfo* edgeMatrix,
                                      struct ncclDiagP2pMemDesc* memDescs, struct ncclDiagP2pMapping* mappings,
                                      struct ncclDiagP2pEdgeResult* allResults, bool cudaUsable, cudaStream_t stream) {
  for (int i = 0; i < outPeerCount; i++) {
    int dstSlot = outPeers[i];
    int dst = ranks[dstSlot];
    const struct ncclDiagP2pEdgeInfo* edge = edgeMatrix + rank * nRanks + dstSlot;
    struct ncclDiagP2pEdgeResult* result = allResults + rank * nRanks + dstSlot;
    if (!cudaUsable || stream == nullptr) {
      ncclDiagP2pSetReason(result, ncclDiagP2pReasonLocalCuda);
      ncclDiagP2pLogEdge(comm, "import", comm->rank, dst, edge, "import=0 reason=localCuda");
      continue;
    }
    if (!memDescs[dstSlot].valid) {
      ncclDiagP2pSetReason(result, ncclDiagP2pReasonNoDescriptor);
      ncclDiagP2pLogEdge(comm, "import", comm->rank, dst, edge, "import=0 reason=noDescriptor");
      continue;
    }

    ncclResult_t importRet = ncclSuccess;
    if (edge->sameProcess) {
      importRet = ncclDiagP2pMapSameProcess(comm, dst, memDescs + dstSlot, mappings + dstSlot);
    } else {
      importRet =
        ncclP2pImportShareableBuffer(comm, dst, memDescs[dstSlot].bytes, &memDescs[dstSlot].ipcDesc,
                                     &mappings[dstSlot].ptr, (void*)memDescs[dstSlot].directPtr, ncclMemScratch);
      if (mappings[dstSlot].ptr != nullptr) {
        mappings[dstSlot].active = 1;
        if (ncclCuMemEnable()) {
          mappings[dstSlot].crossProcessCuMem = 1;
          mappings[dstSlot].tracked = importRet == ncclSuccess ? 1 : 0;
        } else {
          mappings[dstSlot].legacyIpc = 1;
        }
      }
    }

    if (importRet == ncclSuccess && mappings[dstSlot].ptr != nullptr) {
      ncclDiagP2pLogEdge(comm, "import", comm->rank, dst, edge, "import=1");
    } else {
      ncclDiagP2pSetReason(result, ncclDiagP2pReasonImport);
      ncclDiagP2pLogEdge(comm, "import", comm->rank, dst, edge, "import=0 reason=import");
    }
  }
}

static bool ncclDiagP2pPrepareRemoteOps(struct ncclComm* comm, const int* ranks, int rank, int nRanks, int outPeerCount,
                                        const int* outPeers, const struct ncclDiagP2pEdgeResult* allResults,
                                        const struct ncclDiagP2pMapping* mappings,
                                        struct ncclDiagP2pRemoteOp** remoteOpsHost,
                                        struct ncclDiagP2pRemoteOp** remoteOpsDev, int* remoteOpCount,
                                        cudaStream_t stream) {
  int importedCount = 0;
  *remoteOpCount = 0;
  for (int i = 0; i < outPeerCount; i++) {
    int dstSlot = outPeers[i];
    const struct ncclDiagP2pEdgeResult* result = allResults + rank * nRanks + dstSlot;
    if (result->reason == ncclDiagP2pReasonNone) importedCount++;
  }
  if (importedCount == 0) return true;

  ncclResult_t hostAllocRet = ncclCalloc(remoteOpsHost, importedCount);
  if (!ncclDiagP2pNcclSuccess(hostAllocRet, "allocate remote op descriptors")) return false;

  ncclResult_t devAllocRet = ncclCudaCalloc(remoteOpsDev, importedCount, comm->memManager, ncclMemScratch);
  if (!ncclDiagP2pNcclSuccess(devAllocRet, "allocate remote op descriptors on device")) return false;

  for (int i = 0; i < outPeerCount; i++) {
    int dstSlot = outPeers[i];
    int dst = ranks[dstSlot];
    const struct ncclDiagP2pEdgeResult* result = allResults + rank * nRanks + dstSlot;
    if (result->reason != ncclDiagP2pReasonNone) continue;

    struct ncclDiagP2pRemoteOp* op = *remoteOpsHost + (*remoteOpCount)++;
    op->remoteSlots = (struct ncclDiagP2pSlot*)mappings[dstSlot].ptr;
    op->srcRank = comm->rank;
    op->dstRank = dst;
    op->srcSlot = rank;
  }

  cudaError_t cudaErr = cudaMemcpyAsync(
    *remoteOpsDev, *remoteOpsHost, *remoteOpCount * sizeof(struct ncclDiagP2pRemoteOp), cudaMemcpyHostToDevice, stream);
  return ncclDiagP2pCudaSuccess(cudaErr, "copy remote op descriptors");
}

ncclResult_t ncclDiagP2pRun(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  int* p2pRanks = nullptr;
  int p2pRank = -1;
  int p2pNRanks = 0;
  struct ncclDiagP2pEdgeInfo* edgeMatrix = nullptr;
  ncclResult_t* setupResults = nullptr;
  struct ncclDiagP2pSummary* summaries = nullptr;
  struct ncclDiagP2pMemDesc* memDescs = nullptr;
  struct ncclDiagP2pMapping* mappings = nullptr;
  struct ncclDiagP2pEdgeResult* allResults = nullptr;
  int* outPeers = nullptr;
  int* inPeers = nullptr;
  struct ncclDiagP2pRemoteOp* remoteOpsHost = nullptr;
  struct ncclDiagP2pRemoteOp* remoteOpsDev = nullptr;
  struct ncclDiagP2pSlot* localSlots = nullptr;
  struct ncclDiagP2pSlot* hostSlots = nullptr;
  int* p2pRanksDev = nullptr;
  uint64_t* readbackDev = nullptr;
  uint64_t* readbackHost = nullptr;
  struct ncclDiagP2pWriteObs* writeObsMatrix = nullptr;
  cudaStream_t stream = nullptr;
  int outPeerCount = 0;
  int inPeerCount = 0;
  int remoteOpCount = 0;
  int savedDev = -1;
  size_t localBytes = 0;
  bool savedDevValid = false;
  bool cudaUsable = true;
  bool writePhaseOk = false;
  bool readPhaseOk = false;
  bool needsLocalHandle = false;
  bool localHandleRetained = false;
  bool peerAccessChanged = false;
  ncclResult_t setupRet = ncclSuccess;
  ncclResult_t runRet = ncclSuccess;
  ncclResult_t cleanupRet = ncclSuccess;

  cudaError_t cudaErr = cudaGetDevice(&savedDev);
  if (ncclDiagP2pCudaSuccess(cudaErr, "getDevice")) {
    savedDevValid = true;
    if (savedDev != comm->cudaDev) {
      cudaErr = cudaSetDevice(comm->cudaDev);
      cudaUsable = ncclDiagP2pCudaSuccess(cudaErr, "setDevice");
    }
  } else {
    cudaUsable = false;
  }
  if (cudaUsable) {
    cudaErr = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaUsable = ncclDiagP2pCudaSuccess(cudaErr, "createStream");
  }

  NCCLCHECKGOTO(ncclCalloc(&setupResults, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclDiagP2pBuildRankSet(comm, &p2pRanks, &p2pRank, &p2pNRanks), setupRet, setup_complete);
  NCCLCHECKGOTO(ncclCalloc(&summaries, comm->nRanks), setupRet, setup_complete);
  NCCLCHECKGOTO(ncclCalloc(&edgeMatrix, (size_t)p2pNRanks * p2pNRanks), setupRet, setup_complete);
  NCCLCHECKGOTO(ncclCalloc(&memDescs, p2pNRanks), setupRet, setup_complete);
  NCCLCHECKGOTO(ncclCalloc(&mappings, p2pNRanks), setupRet, setup_complete);
  NCCLCHECKGOTO(ncclCalloc(&allResults, (size_t)p2pNRanks * p2pNRanks), setupRet, setup_complete);
  NCCLCHECKGOTO(ncclCalloc(&writeObsMatrix, (size_t)p2pNRanks * p2pNRanks), setupRet, setup_complete);
  NCCLCHECKGOTO(ncclCalloc(&outPeers, p2pNRanks), setupRet, setup_complete);
  NCCLCHECKGOTO(ncclCalloc(&inPeers, p2pNRanks), setupRet, setup_complete);

setup_complete:
  setupResults[comm->rank] = setupRet;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, setupResults, sizeof(ncclResult_t)), ret, fail);

  for (int rank = 0; rank < comm->nRanks; rank++) {
    if (setupResults[rank] == ncclSuccess) continue;
    setupRet = setupResults[rank];
    if (comm->rank == 0) {
      DIAG_PRINT("NCCL DIAG [INFO] p2p: setup failed on rank %d result=%d", rank, setupResults[rank]);
    }
  }
  NCCLCHECKGOTO(setupRet, ret, fail);

  ncclDiagP2pDiscoverLocalEdges(comm, p2pRanks, p2pRank, p2pNRanks, edgeMatrix, allResults, outPeers, &outPeerCount);

  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, p2pRanks, p2pRank, p2pNRanks, edgeMatrix,
                                            p2pNRanks * sizeof(struct ncclDiagP2pEdgeInfo)),
                ret, fail);

  ncclDiagP2pBuildInboundPeers(comm, p2pRanks, p2pRank, p2pNRanks, edgeMatrix, inPeers, &inPeerCount,
                               &needsLocalHandle);

  localBytes = (size_t)p2pNRanks * sizeof(struct ncclDiagP2pSlot);
  memDescs[p2pRank].bytes = inPeerCount == 0 ? 0 : localBytes;
  if (inPeerCount > 0 && cudaUsable && stream != nullptr) {
    // A single owner-held reference is shared by all same-process mappings.
    ncclResult_t allocRet =
      ncclP2pAllocateShareableBuffer(localBytes, needsLocalHandle ? 1 : 0, &memDescs[p2pRank].ipcDesc,
                                     (void**)&localSlots, -1, comm->memManager, ncclMemScratch);
    if (ncclDiagP2pNcclSuccess(allocRet, "allocate local slots")) {
      localHandleRetained = needsLocalHandle;
      ncclResult_t ranksAllocRet = ncclCudaCalloc(&p2pRanksDev, p2pNRanks, comm->memManager, ncclMemScratch);
      if (ncclDiagP2pNcclSuccess(ranksAllocRet, "allocate P2P rank map")) {
        cudaErr = cudaMemcpyAsync(p2pRanksDev, p2pRanks, p2pNRanks * sizeof(int), cudaMemcpyHostToDevice, stream);
        ncclResult_t initRet = ncclSuccess;
        if (ncclDiagP2pCudaSuccess(cudaErr, "copy P2P rank map")) {
          initRet = ncclDiagP2pInitSlots(localSlots, p2pRanksDev, p2pNRanks, comm->rank, stream);
          cudaErr = initRet == ncclSuccess ? cudaStreamSynchronize(stream) : cudaSuccess;
        } else {
          initRet = ncclUnhandledCudaError;
        }
        if (initRet == ncclSuccess && ncclDiagP2pCudaSuccess(cudaErr, "initialize local slots")) {
          memDescs[p2pRank].valid = 1;
          memDescs[p2pRank].directPtr = (uintptr_t)localSlots;
        } else {
          (void)ncclDiagP2pNcclSuccess(initRet, "initialize local slots");
          cudaUsable = false;
        }
      } else {
        cudaUsable = false;
      }
    }
  }

  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, p2pRanks, p2pRank, p2pNRanks, memDescs,
                                            sizeof(struct ncclDiagP2pMemDesc)),
                ret, fail);

  ncclDiagP2pImportMappings(comm, p2pRanks, p2pRank, p2pNRanks, outPeerCount, outPeers, edgeMatrix, memDescs, mappings,
                            allResults, cudaUsable, stream);

  for (int i = 0; i < outPeerCount; i++) {
    if (!mappings[outPeers[i]].peerAccessEnabled) continue;
    peerAccessChanged = true;
    break;
  }
  if (peerAccessChanged) {
    DIAG_PRINT("NCCL DIAG [INFO] p2p: temporarily enabled context-wide CUDA peer access rank=%d cudaDev=%d; "
               "avoid concurrent CUDA use on this context until diagnostics completes",
               comm->rank, comm->cudaDev);
  }

  if (outPeerCount > 0 && cudaUsable && stream != nullptr) {
    cudaUsable = ncclDiagP2pPrepareRemoteOps(comm, p2pRanks, p2pRank, p2pNRanks, outPeerCount, outPeers, allResults,
                                             mappings, &remoteOpsHost, &remoteOpsDev, &remoteOpCount, stream);
  }

  writePhaseOk = cudaUsable && stream != nullptr;
  if (writePhaseOk && remoteOpCount > 0) {
    for (int i = 0; i < remoteOpCount; i++) {
      struct ncclDiagP2pRemoteOp* op = remoteOpsHost + i;
      int dstSlot = ncclDiagP2pRankToSlot(p2pRanks, p2pNRanks, op->dstRank);
      struct ncclDiagP2pEdgeInfo* edge = edgeMatrix + p2pRank * p2pNRanks + dstSlot;
      ncclDiagP2pLogEdge(comm, "write", op->srcRank, op->dstRank, edge, nullptr);
    }
    ncclResult_t launchRet = ncclDiagP2pRemoteWrite(remoteOpsDev, remoteOpCount, stream);
    if (launchRet != ncclSuccess) writePhaseOk = false;
  }
  if (stream != nullptr) {
    cudaErr = cudaStreamSynchronize(stream);
    if (!ncclDiagP2pCudaSuccess(cudaErr, "write phase")) {
      writePhaseOk = false;
      cudaUsable = false;
    }
  }
  if (!writePhaseOk) {
    ncclDiagP2pSetLocalReason(p2pRank, p2pNRanks, allResults, ncclDiagP2pReasonWriteLaunch);
  }
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, p2pRanks, p2pRank, p2pNRanks, kDiagP2pBarrierWrote), ret,
                fail);

  if (inPeerCount > 0 && memDescs[p2pRank].valid && cudaUsable && stream != nullptr) {
    bool localVerifyOk = false;
    ncclResult_t hostAllocRet = ncclCalloc(&hostSlots, p2pNRanks);
    if (ncclDiagP2pNcclSuccess(hostAllocRet, "allocate write observations")) {
      ncclResult_t verifyRet = ncclDiagP2pVerifyWrites(localSlots, p2pNRanks, stream);
      if (verifyRet == ncclSuccess) {
        cudaErr = cudaMemcpyAsync(hostSlots, localSlots, localBytes, cudaMemcpyDeviceToHost, stream);
        if (ncclDiagP2pCudaSuccess(cudaErr, "copy write observations")) {
          cudaErr = cudaStreamSynchronize(stream);
          localVerifyOk = ncclDiagP2pCudaSuccess(cudaErr, "verify write observations");
        }
      } else {
        (void)ncclDiagP2pNcclSuccess(verifyRet, "verify write observations");
      }
    }

    if (localVerifyOk) {
      for (int i = 0; i < inPeerCount; i++) {
        int srcSlot = inPeers[i];
        struct ncclDiagP2pWriteObs* obs = writeObsMatrix + p2pRank * p2pNRanks + srcSlot;
        obs->writeValue = hostSlots[srcSlot].writeValue;
        obs->verifyValue = hostSlots[srcSlot].verifyValue;
      }
    } else {
      cudaUsable = false;
    }
  }

  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, p2pRanks, p2pRank, p2pNRanks, writeObsMatrix,
                                            p2pNRanks * sizeof(struct ncclDiagP2pWriteObs)),
                ret, fail);
  for (int i = 0; i < outPeerCount; i++) {
    int dstSlot = outPeers[i];
    int dst = p2pRanks[dstSlot];
    struct ncclDiagP2pEdgeResult* result = allResults + p2pRank * p2pNRanks + dstSlot;
    struct ncclDiagP2pWriteObs* obs = writeObsMatrix + dstSlot * p2pNRanks + p2pRank;
    uint64_t expected = ncclDiagP2pWritePattern(comm->rank, dst);
    result->writeGot = obs->writeValue;
    result->verifyGot = obs->verifyValue;
    if (obs->writeValue != expected || obs->verifyValue != expected) {
      ncclDiagP2pSetReason(result, ncclDiagP2pReasonWriteMismatch);
    }
  }

  readPhaseOk = cudaUsable && stream != nullptr;
  if (readPhaseOk && remoteOpCount > 0) {
    ncclResult_t readAllocRet = ncclCudaCalloc(&readbackDev, remoteOpCount, comm->memManager, ncclMemScratch);
    if (!ncclDiagP2pNcclSuccess(readAllocRet, "allocate readback")) {
      readPhaseOk = false;
      cudaUsable = false;
    }
  }
  if (readPhaseOk && remoteOpCount > 0) {
    ncclResult_t hostAllocRet = ncclCalloc(&readbackHost, remoteOpCount);
    if (!ncclDiagP2pNcclSuccess(hostAllocRet, "allocate read observations")) {
      readPhaseOk = false;
      cudaUsable = false;
    }
  }
  if (readPhaseOk && remoteOpCount > 0) {
    for (int i = 0; i < remoteOpCount; i++) {
      struct ncclDiagP2pRemoteOp* op = remoteOpsHost + i;
      int dstSlot = ncclDiagP2pRankToSlot(p2pRanks, p2pNRanks, op->dstRank);
      struct ncclDiagP2pEdgeInfo* edge = edgeMatrix + p2pRank * p2pNRanks + dstSlot;
      ncclDiagP2pLogEdge(comm, "read", op->srcRank, op->dstRank, edge, nullptr);
    }
    ncclResult_t launchRet = ncclDiagP2pRemoteRead(remoteOpsDev, remoteOpCount, readbackDev, stream);
    if (launchRet != ncclSuccess) readPhaseOk = false;
  }
  if (readPhaseOk && remoteOpCount > 0) {
    cudaErr =
      cudaMemcpyAsync(readbackHost, readbackDev, remoteOpCount * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    if (ncclDiagP2pCudaSuccess(cudaErr, "copy read observations")) {
      cudaErr = cudaStreamSynchronize(stream);
      readPhaseOk = ncclDiagP2pCudaSuccess(cudaErr, "read phase");
    } else {
      readPhaseOk = false;
    }
  }
  if (!readPhaseOk) {
    ncclDiagP2pSetLocalReason(p2pRank, p2pNRanks, allResults, ncclDiagP2pReasonReadLaunch);
  }

  for (int i = 0; i < remoteOpCount; i++) {
    int dst = remoteOpsHost[i].dstRank;
    int dstSlot = ncclDiagP2pRankToSlot(p2pRanks, p2pNRanks, dst);
    struct ncclDiagP2pEdgeResult* result = allResults + p2pRank * p2pNRanks + dstSlot;
    if (!readPhaseOk) continue;
    uint64_t expected = ncclDiagP2pReadPattern(dst, comm->rank);
    result->readGot = readbackHost[i];
    if (readbackHost[i] != expected) ncclDiagP2pSetReason(result, ncclDiagP2pReasonReadMismatch);
  }

  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, p2pRanks, p2pRank, p2pNRanks, allResults,
                                            p2pNRanks * sizeof(struct ncclDiagP2pEdgeResult)),
                ret, fail);

  if (p2pRank == 0) ncclDiagP2pBuildGroupSummary(p2pNRanks, allResults, summaries + comm->rank);
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, summaries, sizeof(struct ncclDiagP2pSummary)), ret, fail);
  ncclDiagP2pReportSummary(comm, summaries);
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, kDiagP2pBarrierReported), ret, fail);
  if (p2pRank == 0) ncclDiagP2pReportGroupFailures(comm, p2pRanks, p2pNRanks, edgeMatrix, allResults);

fail:
  runRet = ret;
  cleanupRet = ncclSuccess;
  if (stream != nullptr) {
    if (!ncclDiagP2pCudaSuccess(cudaStreamSynchronize(stream), "synchronize before cleanup") &&
        cleanupRet == ncclSuccess) {
      cleanupRet = ncclUnhandledCudaError;
    }
  }

  for (int dst = 0; mappings != nullptr && dst < p2pNRanks; dst++) {
    NCCLCHECKIGNORE(ncclDiagP2pFreeMapping(comm, mappings + dst), cleanupRet);
  }

  if (runRet == ncclSuccess && p2pRanks != nullptr) {
    NCCLCHECKIGNORE(bootstrapIntraNodeBarrier(comm->bootstrap, p2pRanks, p2pRank, p2pNRanks,
                                              kDiagP2pBarrierImportsFreed),
                    cleanupRet);
  }
  if (localHandleRetained) {
    NCCLCHECKIGNORE(ncclDiagP2pReleaseLocalHandle(memDescs + p2pRank), cleanupRet);
    localHandleRetained = false;
  }
  if (readbackDev != nullptr) {
    NCCLCHECKIGNORE(ncclCudaFree(readbackDev, comm->memManager), cleanupRet);
  }
  if (remoteOpsDev != nullptr) {
    NCCLCHECKIGNORE(ncclCudaFree(remoteOpsDev, comm->memManager), cleanupRet);
  }
  if (p2pRanksDev != nullptr) {
    NCCLCHECKIGNORE(ncclCudaFree(p2pRanksDev, comm->memManager), cleanupRet);
  }
  if (localSlots != nullptr) {
    NCCLCHECKIGNORE(ncclCudaFree(localSlots, comm->memManager), cleanupRet);
  }
  if (stream != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
      if (cleanupRet == ncclSuccess) cleanupRet = ncclUnhandledCudaError;
      (void)cudaGetLastError();
    }
  }
  if (savedDevValid && savedDev != comm->cudaDev) {
    cudaError_t err = cudaSetDevice(savedDev);
    if (err != cudaSuccess) {
      if (cleanupRet == ncclSuccess) cleanupRet = ncclUnhandledCudaError;
      (void)cudaGetLastError();
    }
  }
  if (cleanupRet != ncclSuccess) {
    DIAG_PRINT("NCCL DIAG [INFO] p2p: resource cleanup failed rank=%d result=%d; some temporary resources may remain",
               comm->rank, cleanupRet);
  }
  ret = runRet == ncclSuccess ? cleanupRet : runRet;
  if (comm->p2pCrossClique) free(p2pRanks);
  free(edgeMatrix);
  free(setupResults);
  free(summaries);
  free(memDescs);
  free(mappings);
  free(allResults);
  free(outPeers);
  free(inPeers);
  free(remoteOpsHost);
  free(writeObsMatrix);
  free(hostSlots);
  free(readbackHost);
  return ret;
}

#endif
