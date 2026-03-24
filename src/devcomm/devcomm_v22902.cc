/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "dev_runtime.h"
#include "utils.h"

typedef enum : uint8_t {
  NCCL_GIN_TYPE_NONE_v22902 = 0,
  NCCL_GIN_TYPE_PROXY_v22902 = 2,
  NCCL_GIN_TYPE_GDAKI_v22902 = 3,
} ncclGinType_t_v22902;

static_assert(sizeof(ncclGinType_t_v22902) == 1);

struct ncclCommProperties_v22902 {
  size_t size;
  unsigned int magic;
  unsigned int version;

  int rank;
  int nRanks;
  int cudaDev;
  int nvmlDev;
  bool deviceApiSupport;
  bool multimemSupport;
  ncclGinType_t_v22902 ginType;
};

static_assert(offsetof(struct ncclCommProperties_v22902, ginType) == 34);
static_assert(sizeof(struct ncclCommProperties_v22902) == 40);


struct ncclDevComm_v22902 {
  int rank, nRanks;
  uint32_t nRanks_rcp32;
  int lsaRank, lsaSize;
  uint32_t lsaSize_rcp32;

  struct ncclDevCommWindowTable* windowTable;

  // ncclWindow_t is just a (device) pointer.
  ncclWindow_t resourceWindow;
  struct ncclWindow_vidmem resourceWindow_inlined;

  ncclMultimemHandle_t lsaMultimem;
  ncclLsaBarrierHandle_t lsaBarrier;
  ncclGinBarrierHandle_t railGinBarrier;

  uint8_t ginContextCount;
  uint8_t ginNetDeviceTypes[4];
  void* ginHandles[4];
  uint32_t ginSignalBase;
  int ginSignalCount;
  uint32_t ginCounterBase;
  int ginCounterCount;
  uint64_t* ginSignalShadows;
};

static_assert(offsetof(struct ncclDevComm_v22902, rank) == 0);
static_assert(offsetof(struct ncclDevComm_v22902, nRanks) == 4);
static_assert(offsetof(struct ncclDevComm_v22902, nRanks_rcp32) == 8);
static_assert(offsetof(struct ncclDevComm_v22902, lsaRank) == 12);
static_assert(offsetof(struct ncclDevComm_v22902, lsaSize) == 16);
static_assert(offsetof(struct ncclDevComm_v22902, lsaSize_rcp32) == 20);
static_assert(offsetof(struct ncclDevComm_v22902, windowTable) == 24);
static_assert(offsetof(struct ncclDevComm_v22902, resourceWindow) == 32);
static_assert(offsetof(struct ncclDevComm_v22902, resourceWindow_inlined) == 40);
static_assert(offsetof(struct ncclDevComm_v22902, lsaMultimem) == 112);
static_assert(offsetof(struct ncclDevComm_v22902, lsaBarrier) == 120);
static_assert(offsetof(struct ncclDevComm_v22902, railGinBarrier) == 128);
static_assert(offsetof(struct ncclDevComm_v22902, ginContextCount) == 136);
static_assert(offsetof(struct ncclDevComm_v22902, ginNetDeviceTypes) == 137);
static_assert(offsetof(struct ncclDevComm_v22902, ginHandles) == 144);
static_assert(offsetof(struct ncclDevComm_v22902, ginSignalBase) == 176);
static_assert(offsetof(struct ncclDevComm_v22902, ginSignalCount) == 180);
static_assert(offsetof(struct ncclDevComm_v22902, ginCounterBase) == 184);
static_assert(offsetof(struct ncclDevComm_v22902, ginCounterCount) == 188);
static_assert(offsetof(struct ncclDevComm_v22902, ginSignalShadows) == 192);
static_assert(sizeof(struct ncclDevComm_v22902) == 200);


static ncclResult_t ncclCommPropertiesFilter_v22902(ncclComm_t comm, struct ncclCommProperties* props) {
  // We don't provide backwards compatibility for GIN with 2.29.2.  If a communicator needs it, we indicate that
  // the Device API is not available.
  props->deviceApiSupport = (props->deviceApiSupport && ncclTeamLsa(comm).nRanks == comm->nRanks);

  // v22902 ncclCommProperties is _almost_ compatible with newer ones, with the exception of ginType, which in that
  // version was based on uint_8, not an int.
  ((struct ncclCommProperties_v22902*)props)->ginType = NCCL_GIN_TYPE_NONE_v22902;

  return ncclSuccess;
}

static ncclResult_t ncclDevCommRequirementsFilter_v22902(ncclComm_t comm, ncclDevCommRequirements_t* reqs) {
  bool userRequestedGin = reqs->ginForceEnable || reqs->ginSignalCount > 0 || reqs->ginCounterCount > 0;
  {
    struct ncclDevResourceRequirements* rr = reqs->resourceRequirementsList;
    while (!userRequestedGin && rr != nullptr) {
      userRequestedGin = rr->ginSignalCount > 0 || rr->ginCounterCount > 0;
      rr = rr->next;
    }
  }
  if (userRequestedGin) {
    char compiledBuf[16], runtimeBuf[16];
    WARN("The application was compiled with too old version of NCCL. It was compiled with NCCL version %s, but is running with NCCL library version %s. Because of its use of GIN device kernels, it needs to be recompiled, preferably with the same NCCL version that it will be running with.",
         ncclVersionToString(reqs->version, compiledBuf, sizeof(compiledBuf)),
         ncclVersionToString(NCCL_VERSION_CODE, runtimeBuf, sizeof(runtimeBuf)));
    return ncclInvalidUsage;
  }

  // Prior to 2.29.4, a non-zero barrierCount did not imply GIN, but it does since.
  if (reqs->barrierCount) {
    reqs->lsaBarrierCount = std::max(reqs->lsaBarrierCount, reqs->barrierCount);
    reqs->barrierCount = 0;
  }
  // Strangely, neither did railGinBarrierCount.
  reqs->railGinBarrierCount = 0;

  return ncclSuccess;
}

static ncclResult_t ncclDevCommCopyNewToOld_v22902(ncclComm_t comm, void* oldDevComm,
                                                   struct ncclDevComm const* newDevComm) {
  struct ncclDevComm_v22902* old = (struct ncclDevComm_v22902*)oldDevComm;

  memset(old, '\0', sizeof(*old));
  ncclDevCommCopyLsaData(&old->rank, &newDevComm->rank);
  // No need to copy GIN-specific fields since we don't provide backwards compatibility for GIN with 2.29.2.

  return ncclSuccess;
}

static ncclResult_t ncclDevCommCopyOldToNew_v22902(ncclComm_t comm, struct ncclDevComm* newDevComm,
                                                   void const* oldDevComm) {
  struct ncclDevComm_v22902* old = (struct ncclDevComm_v22902*)oldDevComm;

  // Note: this callback will be used with v22907 as well because, prior to 2.30.0, ncclDevComm was unversioned,
  // so v22902 and v22907 variants are indistinguishable.  Primary differences between them are related to GIN
  // but, since we don't support GIN here with either, the differences are irrelevant to us.
  ncclDevCommCopyLsaData(&newDevComm->rank, &old->rank);

  return ncclSuccess;
}

struct ncclDevCommCompat ncclDevCommCompat_v22902 = {
  NCCL_VERSION(2, 29, 2), NCCL_VERSION(2, 29, 3), // minVersion, maxVersion
  ncclCommPropertiesFilter_v22902,                // commPropertiesFilter
  ncclDevCommRequirementsFilter_v22902,           // devCommRequirementsFilter
  ncclDevCommCopyNewToOld_v22902,                 // devCommCopyNewToOld
  ncclDevCommCopyOldToNew_v22902,                 // devCommCopyOldToNew
};
