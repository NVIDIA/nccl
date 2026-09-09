/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_RUNTIME_INTERNAL_H_
#define NCCL_DEVICE_RUNTIME_INTERNAL_H_

#include "dev_runtime.h"
#include "nccl_device/core.h"
#include <cuda.h>
#include <cuda_runtime.h>

struct ncclComm;
struct ncclSegmentWindow;
struct ncclWindow_vidmem;

// Complete type for dev_runtime.h's forward declaration.
struct ncclDevrTeam {
  struct ncclDevrTeam* next;
  struct ncclTeam team;
  CUmemGenericAllocationHandle mcHandle;
  void* mcBasePtr;
  ncclCftLeId ucLeId;
  ncclCftLeId mcLeId;
  int worldRankList[];
};

// Non-static functions in dev_runtime.cc also called from cft_dev_runtime.cc:
int computeLsaSize(struct ncclComm* comm);
ncclResult_t symTeamObtain(struct ncclComm* comm, struct ncclTeam team, bool multimem, bool wantsLeUc, bool wantsLeMc,
                           struct ncclDevrTeam** outTeam, bool* needBarrier);
ncclResult_t findCommAndHostWindowFromDeviceWindow(ncclWindow_t devWindow, ncclComm_t* foundComm,
                                                   struct ncclDevrWindow** hostWindow);

// Functions in cft_dev_runtime.cc called from dev_runtime.cc:
int computeCftSize(struct ncclComm* comm);
int computeCftMcSize(struct ncclComm* comm);
ncclResult_t symBindTeamLe(struct ncclComm* comm, struct ncclDevrMemory* mem, ncclCftLeId le);
ncclResult_t symUnbindTeamLe(struct ncclComm* comm, struct ncclDevrMemory* mem, ncclCftLeId le);
ncclResult_t symTeamObtainUcLe(struct ncclComm* comm, struct ncclDevrTeam* t, struct ncclDevrState* devr,
                               bool* needBarrier);
ncclResult_t symTeamObtainMcLe(struct ncclComm* comm, struct ncclDevrTeam* t, struct ncclDevrState* devr,
                               bool* needBarrier);

struct ncclDevrGinSegmentInfo {
  void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS];
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS];
  CUmemLocationType memType;
  size_t segmentSize;
};

// Complete type for src/include/dev_runtime.h's forward declaration.
struct ncclDevrMemory {
  int refCount;
  struct ncclDevrMemory* next;
  CUmemGenericAllocationHandle* memHandles;
  void* primaryAddr; // What we hope is the VA of this memory's first mapping.
  size_t size;
  size_t bigOffset; // offset in big VA space
  void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS];
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS];
  void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS];
  int winFlags;
  // Per-rank info derived from this rank's own allocation.
  int numSegments;         // number of physical segments backing this rank's buffer
  bool hasSysmemSegment;   // true if any segment is CPU-backed (HOST_NUMA memory type)
  size_t* segmentSizes;    // size of each segment, length numSegments
  // Communicator-wide aggregates over all nRanks, populated via bootstrapAllGather.
  int maxGlobalNumSegments;    // max(numSegments) across all communicator ranks
  bool globalHasSysmemSegment; // true if any communicator rank has a sysmem segment
  // LSA-team aggregates, derived from a global allgather.
  int* lsaNumSegments;   // numSegments for each LSA rank, length lsaSize
  size_t lsaMinSize; // min size across ranks in the LSA team
  size_t lsaMaxSize; // max size across LSA ranks, used for ncclSpaceAlloc/Free
  // GIN registration state.
  int numGinSegments;                              // 1 unless GIN needs multiple per-segment windows
  struct ncclDevrGinSegmentInfo* ginSegmentInfos;  // per-GIN-segment info
};

ncclResult_t ncclDevrPopulateSegmentSizes(struct ncclDevrMemory* mem, int numSegments);

ncclResult_t ncclDevrCheckRegistrationSupport(void* userPtr, size_t userSize, struct ncclComm* comm,
                                              bool hasSysmemSegment);

ncclResult_t ncclDevrValidateHandleLocationType(CUmemGenericAllocationHandle memHandle, int segment);

ncclResult_t ncclDevrVerifySegmentLayouts(struct ncclDevrMemory* mem, struct ncclComm* comm);

ncclResult_t ncclDevrBuildGinSegmentInfos(struct ncclDevrMemory* mem);

ncclResult_t ncclDevrAllocAndPopulateSegmentWindows(struct ncclDevrState* devr, struct ncclDevrMemory* mem,
                                                    cudaStream_t stream,
                                                    struct ncclSegmentWindow** outSegmentWindowsDev);

#endif
