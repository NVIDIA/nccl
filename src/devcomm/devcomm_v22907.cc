/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "dev_runtime.h"
#include "utils.h"

struct ncclDevComm_v22907 {
  int rank, nRanks;
  uint32_t nRanks_rcp32;
  int lsaRank, lsaSize;
  uint32_t lsaSize_rcp32;

  struct ncclDevCommWindowTable* windowTable;

  ncclWindow_t resourceWindow;
  struct ncclWindow_vidmem resourceWindow_inlined;

  ncclMultimemHandle_t lsaMultimem;
  ncclLsaBarrierHandle_t lsaBarrier;
  ncclGinBarrierHandle_t railGinBarrier;

  uint8_t ginConnectionCount;
  uint8_t ginNetDeviceTypes[NCCL_GIN_MAX_CONNECTIONS];
  void* ginHandles[NCCL_GIN_MAX_CONNECTIONS];
  uint32_t ginSignalBase;
  int ginSignalCount;
  uint32_t ginCounterBase;
  int ginCounterCount;
  uint64_t* ginSignalShadows;
  uint32_t ginContextCount;
  uint32_t ginContextBase;
  bool ginIsRailed;

  uint32_t* abortFlag;
};

static_assert(offsetof(struct ncclDevComm_v22907, rank) == 0);
static_assert(offsetof(struct ncclDevComm_v22907, nRanks) == 4);
static_assert(offsetof(struct ncclDevComm_v22907, nRanks_rcp32) == 8);
static_assert(offsetof(struct ncclDevComm_v22907, lsaRank) == 12);
static_assert(offsetof(struct ncclDevComm_v22907, lsaSize) == 16);
static_assert(offsetof(struct ncclDevComm_v22907, lsaSize_rcp32) == 20);
static_assert(offsetof(struct ncclDevComm_v22907, windowTable) == 24);
static_assert(offsetof(struct ncclDevComm_v22907, resourceWindow) == 32);
static_assert(offsetof(struct ncclDevComm_v22907, resourceWindow_inlined) == 40);
static_assert(offsetof(struct ncclDevComm_v22907, lsaMultimem) == 112);
static_assert(offsetof(struct ncclDevComm_v22907, lsaBarrier) == 120);
static_assert(offsetof(struct ncclDevComm_v22907, railGinBarrier) == 128);
static_assert(offsetof(struct ncclDevComm_v22907, ginConnectionCount) == 136);
static_assert(offsetof(struct ncclDevComm_v22907, ginNetDeviceTypes) == 137);
static_assert(offsetof(struct ncclDevComm_v22907, ginHandles) == 144);
static_assert(offsetof(struct ncclDevComm_v22907, ginSignalBase) == 176);
static_assert(offsetof(struct ncclDevComm_v22907, ginSignalCount) == 180);
static_assert(offsetof(struct ncclDevComm_v22907, ginCounterBase) == 184);
static_assert(offsetof(struct ncclDevComm_v22907, ginCounterCount) == 188);
static_assert(offsetof(struct ncclDevComm_v22907, ginSignalShadows) == 192);
static_assert(offsetof(struct ncclDevComm_v22907, ginContextCount) == 200);
static_assert(offsetof(struct ncclDevComm_v22907, ginContextBase) == 204);
static_assert(offsetof(struct ncclDevComm_v22907, ginIsRailed) == 208);
static_assert(offsetof(struct ncclDevComm_v22907, abortFlag) == 216);
static_assert(sizeof(struct ncclDevComm_v22907) == 224);


static ncclResult_t ncclCommPropertiesFilter_v22907(ncclComm_t comm, struct ncclCommProperties* props) {
  // We don't provide backwards compatibility for GIN with 2.29.7.  If a communicator needs it, we indicate that
  // the Device API is not available.
  props->deviceApiSupport = (props->deviceApiSupport && ncclTeamLsa(comm).nRanks == comm->nRanks);
  props->ginType = NCCL_GIN_TYPE_NONE;
  props->railedGinType = NCCL_GIN_TYPE_NONE;

  return ncclSuccess;
}

static ncclResult_t ncclDevCommRequirementsFilter_v22907(ncclComm_t comm, ncclDevCommRequirements_t* reqs) {
  bool requestedGinResources = reqs->ginSignalCount > 0 || reqs->ginCounterCount > 0 ||
                               reqs->barrierCount > 0 || reqs->railGinBarrierCount > 0;
  struct ncclDevResourceRequirements* node = reqs->resourceRequirementsList;
  while (!requestedGinResources && node != nullptr) {
    requestedGinResources = node->ginSignalCount > 0 || node->ginCounterCount > 0;
    node = node->next;
  }
  if (requestedGinResources && (reqs->ginConnectionType != NCCL_GIN_CONNECTION_NONE || reqs->ginForceEnable)) {
    char compiledBuf[16], runtimeBuf[16];
    WARN("The application was compiled with too old version of NCCL. It was compiled with NCCL version %s, but is running with NCCL library version %s. Because of its use of GIN device kernels, it needs to be recompiled, preferably with the same NCCL version that it will be running with.",
         ncclVersionToString(reqs->version, compiledBuf, sizeof(compiledBuf)),
         ncclVersionToString(NCCL_VERSION_CODE, runtimeBuf, sizeof(runtimeBuf)));
    return ncclInvalidUsage;
  }

  return ncclSuccess;
}

static ncclResult_t ncclDevCommCopyNewToOld_v22907(ncclComm_t comm, void* oldDevComm,
                                                   struct ncclDevComm const* newDevComm) {
  struct ncclDevComm_v22907* old = (struct ncclDevComm_v22907*)oldDevComm;

  memset(old, '\0', sizeof(*old));
  ncclDevCommCopyLsaData(&old->rank, &newDevComm->rank);
  // No need to copy GIN-specific fields since we don't provide backwards compatibility for GIN with 2.29.7.
  old->abortFlag = newDevComm->abortFlag;

  return ncclSuccess;
}

struct ncclDevCommCompat ncclDevCommCompat_v22907 = {
  NCCL_VERSION(2, 29, 5), NCCL_VERSION(2, 29, 7), // minVersion, maxVersion
  ncclCommPropertiesFilter_v22907,                // commPropertiesFilter
  ncclDevCommRequirementsFilter_v22907,           // devCommRequirementsFilter
  ncclDevCommCopyNewToOld_v22907,                 // devCommCopyNewToOld
  nullptr,                                        // devCommCopyOldToNew -- we'll use the v22902 variant
};
