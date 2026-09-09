/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "dev_runtime.h"
#include "comm.h"

typedef struct ncclResourceWindow_vidmem_v23000 {
  char reserved1[8];
  char* lsaFlatBase;
  char reserved2[8];
  uint32_t stride4G;
  uint32_t mcOffset4K;
  char reserved3[32];  // NOTE: shrunk from 40 in 2.30u1 to reclaim 8 bytes
} ncclResourceWindow_vidmem_v23000_t;

static_assert(offsetof(ncclResourceWindow_vidmem_v23000, lsaFlatBase) == offsetof(ncclWindow_vidmem, lsaFlatBase));
static_assert(offsetof(ncclResourceWindow_vidmem_v23000, stride4G) == offsetof(ncclWindow_vidmem, stride4G));
static_assert(offsetof(ncclResourceWindow_vidmem_v23000, mcOffset4K) == offsetof(ncclWindow_vidmem, mcOffset4K));
static_assert(sizeof(ncclResourceWindow_vidmem_v23000) == 64);

struct ncclDevComm_v23000 {
  unsigned int magic;
  unsigned int version;

  int rank, nRanks;
  uint32_t nRanks_rcp32;
  int lsaRank, lsaSize;
  uint32_t lsaSize_rcp32;

  ncclDevCommWindowTable_t windowTable;

  ncclWindow_t resourceWindow;
  ncclResourceWindow_vidmem_v23000_t resourceWindow_inlined;
  // 2.30u1 reclaimed 8 bytes from resourceWindow_inlined's reserved3 (now 32
  // bytes) and placed hybridWorldGinBarrier in ncclDevComm in that space.
  ncclGinBarrierHandle_t hybridWorldGinBarrier;

  ncclMultimemHandle_t lsaMultimem;
  ncclLsaBarrierHandle_t lsaBarrier;
  ncclGinBarrierHandle_t railGinBarrier;

  uint8_t ginConnectionCount;
  uint8_t ginNetDeviceTypes[NCCL_GIN_MAX_CONNECTIONS];
  void* ginHandles[NCCL_GIN_MAX_CONNECTIONS];
  int ginSignalCount;
  int ginCounterCount;
  uint64_t* ginSignalShadows;
  uint32_t ginContextCount;
  bool ginConnectionsRailed;
  bool ginStrongLegacySignals;
  bool ginContextsRailed;
  uint32_t* abortFlag;

  ncclLsaBarrierHandle_t hybridLsaBarrier;
  ncclGinBarrierHandle_t hybridRailGinBarrier;

  ncclGinBarrierHandle_t worldGinBarrier;
};

static_assert(offsetof(struct ncclDevComm_v23000, magic) == 0);
static_assert(offsetof(struct ncclDevComm_v23000, version) == 4);
static_assert(offsetof(struct ncclDevComm_v23000, rank) == 8);
static_assert(offsetof(struct ncclDevComm_v23000, nRanks) == 12);
static_assert(offsetof(struct ncclDevComm_v23000, nRanks_rcp32) == 16);
static_assert(offsetof(struct ncclDevComm_v23000, lsaRank) == 20);
static_assert(offsetof(struct ncclDevComm_v23000, lsaSize) == 24);
static_assert(offsetof(struct ncclDevComm_v23000, lsaSize_rcp32) == 28);
static_assert(offsetof(struct ncclDevComm_v23000, windowTable) == 32);
static_assert(offsetof(struct ncclDevComm_v23000, resourceWindow) == 40);
static_assert(offsetof(struct ncclDevComm_v23000, resourceWindow_inlined) == 48);
static_assert(offsetof(struct ncclDevComm_v23000, hybridWorldGinBarrier) == 112);
static_assert(offsetof(struct ncclDevComm_v23000, lsaMultimem) == 120);
static_assert(offsetof(struct ncclDevComm_v23000, lsaBarrier) == 128);
static_assert(offsetof(struct ncclDevComm_v23000, railGinBarrier) == 136);
static_assert(offsetof(struct ncclDevComm_v23000, ginConnectionCount) == 144);
static_assert(offsetof(struct ncclDevComm_v23000, ginNetDeviceTypes) == 145);
static_assert(offsetof(struct ncclDevComm_v23000, ginHandles) == 152);
static_assert(offsetof(struct ncclDevComm_v23000, ginSignalCount) == 184);
static_assert(offsetof(struct ncclDevComm_v23000, ginCounterCount) == 188);
static_assert(offsetof(struct ncclDevComm_v23000, ginSignalShadows) == 192);
static_assert(offsetof(struct ncclDevComm_v23000, ginContextCount) == 200);
static_assert(offsetof(struct ncclDevComm_v23000, ginConnectionsRailed) == 204);
static_assert(offsetof(struct ncclDevComm_v23000, ginStrongLegacySignals) == 205);
static_assert(offsetof(struct ncclDevComm_v23000, ginContextsRailed) == 206);
static_assert(offsetof(struct ncclDevComm_v23000, abortFlag) == 208);
static_assert(offsetof(struct ncclDevComm_v23000, hybridLsaBarrier) == 216);
static_assert(offsetof(struct ncclDevComm_v23000, hybridRailGinBarrier) == 224);
static_assert(offsetof(struct ncclDevComm_v23000, worldGinBarrier) == 232);
static_assert(sizeof(struct ncclDevComm_v23000) == 240);

static ncclResult_t ncclDevCommRequirementsFilter_v23000(ncclComm_t comm, ncclDevCommRequirements_t* reqs) {
  reqs->ginType = comm->sharedRes->ginState.backends[0].ginType;
  return ncclSuccess;
}

static void ncclDevCommCopyResourceWindowNewToOld_v23000(ncclResourceWindow_vidmem_v23000_t* old,
                                                         ncclResourceWindow_vidmem_t const& nw) {
  old->lsaFlatBase = nw.lsaFlatBase;
  old->stride4G = nw.stride4G;
  old->mcOffset4K = nw.mcOffset4K;
}

static void ncclDevCommCopyResourceWindowOldToNew_v23000(ncclResourceWindow_vidmem_t* nw,
                                                         ncclResourceWindow_vidmem_v23000_t const& old) {
  nw->lsaFlatBase = old.lsaFlatBase;
  nw->stride4G = old.stride4G;
  nw->mcOffset4K = old.mcOffset4K;
}

static ncclResult_t ncclDevCommCopyNewToOld_v23000(ncclComm_t comm, void* oldDevComm,
                                                   struct ncclDevComm const* newDevComm) {
  struct ncclDevComm_v23000* old = (struct ncclDevComm_v23000*)oldDevComm;

  memset(old, '\0', sizeof(*old));
  old->magic = newDevComm->magic;
  old->version = newDevComm->version;
  old->rank = newDevComm->rank;
  old->nRanks = newDevComm->nRanks;
  old->nRanks_rcp32 = newDevComm->nRanks_rcp32;
  old->lsaRank = newDevComm->lsaRank;
  old->lsaSize = newDevComm->lsaSize;
  old->lsaSize_rcp32 = newDevComm->lsaSize_rcp32;
  old->windowTable = newDevComm->windowTable;
  old->resourceWindow = newDevComm->resourceWindow;
  ncclDevCommCopyResourceWindowNewToOld_v23000(&old->resourceWindow_inlined, newDevComm->resourceWindow_inlined);
  old->hybridWorldGinBarrier = newDevComm->hybridDenseGinBarrier;
  old->lsaMultimem = newDevComm->lsaMultimem;
  old->lsaBarrier = newDevComm->lsaBarrier;
  old->railGinBarrier = newDevComm->railGinBarrier;
  old->ginConnectionCount = newDevComm->ginConnectionCount;
  memcpy(old->ginNetDeviceTypes, newDevComm->ginNetDeviceTypes, sizeof(old->ginNetDeviceTypes));
  memcpy(old->ginHandles, newDevComm->ginHandles, sizeof(old->ginHandles));
  old->ginSignalCount = newDevComm->ginSignalCount;
  old->ginCounterCount = newDevComm->ginCounterCount;
  old->ginSignalShadows = newDevComm->ginSignalShadows;
  old->ginContextCount = newDevComm->ginContextCount;

  old->ginConnectionsRailed = (newDevComm->ginConnectionStride > 1);
  old->ginStrongLegacySignals = newDevComm->ginStrongLegacySignals;
  old->ginContextsRailed = (newDevComm->ginContextStride > 1);

  old->abortFlag = newDevComm->abortFlag;
  old->hybridLsaBarrier = newDevComm->hybridLsaBarrier;
  old->hybridRailGinBarrier = newDevComm->hybridRailGinBarrier;
  old->worldGinBarrier = newDevComm->worldGinBarrier;

  return ncclSuccess;
}

static ncclResult_t ncclDevCommCopyOldToNew_v23000(ncclComm_t comm, struct ncclDevComm* newDevComm,
                                                   void const* oldDevComm) {
  struct ncclDevComm_v23000 const* old = (struct ncclDevComm_v23000 const*)oldDevComm;

  newDevComm->magic = old->magic;
  newDevComm->version = old->version;
  newDevComm->rank = old->rank;
  newDevComm->nRanks = old->nRanks;
  newDevComm->nRanks_rcp32 = old->nRanks_rcp32;
  newDevComm->lsaRank = old->lsaRank;
  newDevComm->lsaSize = old->lsaSize;
  newDevComm->lsaSize_rcp32 = old->lsaSize_rcp32;
  newDevComm->windowTable = old->windowTable;
  newDevComm->resourceWindow = old->resourceWindow;
  ncclDevCommCopyResourceWindowOldToNew_v23000(&newDevComm->resourceWindow_inlined, old->resourceWindow_inlined);
  newDevComm->hybridDenseGinBarrier = old->hybridWorldGinBarrier;
  newDevComm->lsaMultimem = old->lsaMultimem;
  newDevComm->lsaBarrier = old->lsaBarrier;
  newDevComm->railGinBarrier = old->railGinBarrier;
  newDevComm->ginConnectionCount = old->ginConnectionCount;
  memcpy(newDevComm->ginNetDeviceTypes, old->ginNetDeviceTypes, sizeof(newDevComm->ginNetDeviceTypes));
  memcpy(newDevComm->ginHandles, old->ginHandles, sizeof(newDevComm->ginHandles));
  newDevComm->ginSignalCount = old->ginSignalCount;
  newDevComm->ginCounterCount = old->ginCounterCount;
  newDevComm->ginSignalShadows = old->ginSignalShadows;
  newDevComm->ginContextCount = old->ginContextCount;
  newDevComm->ginConnectionStride = old->ginConnectionsRailed ? old->lsaSize : 1;
  newDevComm->ginContextStride = old->ginContextsRailed ? old->lsaSize : 1;
  newDevComm->ginStrongLegacySignals = old->ginStrongLegacySignals;

  newDevComm->abortFlag = old->abortFlag;
  newDevComm->hybridLsaBarrier = old->hybridLsaBarrier;
  newDevComm->hybridRailGinBarrier = old->hybridRailGinBarrier;
  newDevComm->worldGinBarrier = old->worldGinBarrier;

  return ncclSuccess;
}

struct ncclDevCommCompat ncclDevCommCompat_v23000 = {
  NCCL_VERSION(2, 30, 0),               // minVersion
  NCCL_VERSION(2, 30, 7),               // maxVersion
  nullptr,                              // commPropertiesFilter
  ncclDevCommRequirementsFilter_v23000, // devCommRequirementsFilter
  ncclDevCommCopyNewToOld_v23000,       // devCommCopyNewToOld
  ncclDevCommCopyOldToNew_v23000,       // devCommCopyOldToNew
};
