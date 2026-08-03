/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_HOST_H_
#define _NCCL_DEVICE_HOST_H_

#include "impl/comm__types.h"

typedef struct ncclDevResourceRequirements {
  struct ncclDevResourceRequirements* next;
  size_t bufferSize, bufferAlign;
  ncclDevResourceHandle_t* outBufferHandle; // If non-null, target assigned during ncclDevCommCreate.
  int ginSignalCount;
  int ginCounterCount;
  ncclGinSignal_t* outGinSignalStart;
  ncclGinCounter_t* outGinCounterStart;
} ncclDevResourceRequirements_t;

typedef struct ncclTeamRequirements {
  struct ncclTeamRequirements* next;
  ncclTeam_t team;
  bool multimem;
  ncclMultimemHandle_t* outMultimemHandle; // If non-null, target assigned during ncclDevCommCreate.
} ncclTeamRequirements_t;

typedef struct ncclDevCommRequirements {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;

  /* attributes that users are able to customize. */
  ncclDevResourceRequirements_t* resourceRequirementsList;
  ncclTeamRequirements_t* teamRequirementsList;

  bool lsaMultimem; // Enable multimem on lsa team

  int barrierCount;
  int lsaBarrierCount;
  int railGinBarrierCount;

  int lsaLLA2ABlockCount, lsaLLA2ASlotCount;

  bool ginForceEnable;

  int ginContextCount; // This is a hint, the actual context count in the devcomm may not match.
  int ginSignalCount; // Guaranteed to start at id=0
  int ginCounterCount; // Guaranteed to start at id=0
  ncclGinConnectionType_t ginConnectionType;
  bool ginExclusiveContexts;
  int ginQueueDepth;
  int ginTrafficClass;

  int worldGinBarrierCount;

  // Set to false if GIN strong signals will not be needed by the kernels using this devComm (defaults to true).
  // When false, the use of GIN strong signals results in undefined behavior.
  bool ginStrongSignalsRequired;

  // Set to false if GIN VA signals will not be needed by the kernels using this devComm (defaults to true).
  // When false, the use of GIN VA signals results in undefined behavior.
  bool ginVaSignalsRequired;

  // Stride of ranks to connect for GIN if ginConnectionType is NCCL_GIN_CONNECTION_CUSTOM_STRIDE.
  int ginCustomStride;

  ncclGinType_t ginType;
  // If true, initialize the devComm assuming the version of the device code is the same
  // as the runtime version of the NCCL library (i.e., the device code is JIT-compiled).
  // When true, the DevComm must be allocated according to devCommRuntimeVersionSize.
  bool useRuntimeVersion;

  int cftCaps; // Bitmask of ncclCftCap_t values
  int cftBarrierCount;
} ncclDevCommRequirements_t;

// clang-format off: maintain hand-formatted code
#define NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER {                               \
    sizeof(ncclDevCommRequirements_t),           /* size */                    \
    NCCL_API_MAGIC,                              /* magic */                   \
    NCCL_VERSION_CODE,                           /* version */                 \
    NULL,                                        /* resourceRequirementsList*/ \
    NULL,                                        /* teamRequirementsList */    \
    false,                                       /* lsaMultimem */             \
    0,                                           /* barrierCount */            \
    0,                                           /* lsaBarrierCount */         \
    0,                                           /* railGinBarrierCount */     \
    0,                                           /* lsaLLA2ABlockCount */      \
    0,                                           /* lsaLLA2ASlotCount */       \
    false,                                       /* ginForceEnable */          \
    4,                                           /* ginContextCount */         \
    0,                                           /* ginSignalCount */          \
    0,                                           /* ginCounterCount */         \
    NCCL_GIN_CONNECTION_NONE,                    /* ginConnectionType */       \
    false,                                       /* ginExclusiveContexts */    \
    0,                                           /* ginQueueDepth */           \
    NCCL_CONFIG_UNDEF_INT,                       /* ginTrafficClass */         \
    0,                                           /* worldGinBarrierCount */    \
    true,                                        /* ginStrongSignalsRequired */ \
    true,                                        /* ginVaSignalsRequired */     \
    1,                                           /* ginCustomStride      */     \
    NCCL_GIN_TYPE_NONE,                          /* ginType */                  \
    false,                                       /* useRuntimeVersion */        \
    NCCL_CFT_NONE,                               /* cftCaps */                 \
    0,                                           /* cftBarrierCount */         \
}
// clang-format on

typedef struct ncclCommProperties {
  /* internal use only */
  size_t size;
  unsigned int magic;
  unsigned int version;

  /* attributes for users. */
  int rank;
  int nRanks;
  int cudaDev;
  int nvmlDev;
  bool deviceApiSupport;
  bool multimemSupport;
  ncclGinType_t ginType;
  int nLsaTeams;
  bool hostRmaSupport;
  ncclGinType_t railedGinType;
  uint64_t commHash;
  int ginMinStride;
  ncclGinConnectionType_t ginConnectionType;
  bool ginSupport[64]; // ginSupport[i] is true if gin type i is supported
  size_t devCommRuntimeVersionSize;
} ncclCommProperties_t;

#define NCCL_COMM_PROPERTIES_INITIALIZER \
  { \
    sizeof(ncclCommProperties_t),                  /* size */ \
    NCCL_API_MAGIC,                                /* magic */ \
    NCCL_VERSION_CODE,                             /* version */ \
  }

////////////////////////////////////////////////////////////////////////////////
// Device communicator API:

NCCL_EXTERN_C ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props);
NCCL_EXTERN_C ncclResult_t ncclDevCommCreate(ncclComm_t comm, ncclDevCommRequirements_t const* reqs,
                                             ncclDevComm_t* outDevComm);
NCCL_EXTERN_C ncclResult_t ncclDevCommDestroy(ncclComm_t comm, ncclDevComm_t const* devComm);

////////////////////////////////////////////////////////////////////////////////
// Window API:

// VA pointer based query functions for host code
NCCL_EXTERN_C ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr);
NCCL_EXTERN_C ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset,
                                                        ncclMultimemHandle_t multimem, void** outPtr);
NCCL_EXTERN_C ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr);
NCCL_EXTERN_C ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr);

// CFT handle based query functions for host code
NCCL_EXTERN_C ncclResult_t ncclGetMultimemDeviceLeInfo(ncclWindow_t window, size_t offset, ncclCftLeId* leId,
                                                       size_t* leOffset);
NCCL_EXTERN_C ncclResult_t ncclGetCftDeviceLeInfo(ncclWindow_t window, size_t offset, int peerCft, ncclTeam_t cftTeam,
                                                  ncclCftLeId* leId, size_t* leOffset);
NCCL_EXTERN_C ncclResult_t ncclGetPeerDeviceLeInfo(ncclWindow_t window, size_t offset, int peerWorld, ncclCftLeId* leId,
                                                   size_t* leOffset);

////////////////////////////////////////////////////////////////////////////////
// Team API:

NCCL_EXTERN_C ncclTeam_t ncclTeamWorld(ncclComm_t comm);
NCCL_EXTERN_C ncclTeam_t ncclTeamLsa(ncclComm_t comm);
#if __cplusplus
NCCL_EXTERN_C ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode = NCCL_CFT_TEAM_FLAT);
#else
NCCL_EXTERN_C ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode);
#endif
NCCL_EXTERN_C ncclTeam_t ncclTeamCftMultimem(ncclComm_t comm);
NCCL_EXTERN_C int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank);
NCCL_EXTERN_C int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank);
NCCL_EXTERN_C ncclTeam_t ncclTeamRail(ncclComm_t comm);

////////////////////////////////////////////////////////////////////////////////
// Device resource requirement API:

NCCL_EXTERN_C ncclResult_t ncclLsaBarrierCreateRequirement(
  ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);
NCCL_EXTERN_C ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers,
                                                           ncclGinBarrierHandle_t* outHandle,
                                                           ncclDevResourceRequirements_t* outReq);
NCCL_EXTERN_C ncclResult_t ncclCftBarrierCreateRequirement(
  ncclTeam_t team, int nBarriers, ncclCftBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);
NCCL_EXTERN_C int ncclLLA2ACalcSlots(int maxElts, int maxEltSize);
NCCL_EXTERN_C ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t* outHandle,
                                                      ncclDevResourceRequirements_t* outReq);

#if defined(NCCL_OS_WINDOWS)
NCCL_EXTERN_C ncclResult_t ncclGinOutboxCreateRequirement(int nBlocks, int size_log2, ncclGinOutboxHandle* outHandle,
                                                          ncclDevResourceRequirements* outReq);
NCCL_EXTERN_C ncclResult_t ncclGinInboxA2ACreateRequirement(
  ncclTeam peers, int nBlocks, int size_log2, ncclGinInboxA2AHandle* outHandle, ncclDevResourceRequirements* outReq);
#endif

#endif // _NCCL_DEVICE_HOST_H_
