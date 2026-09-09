/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CORE_H_
#define _NCCL_DEVICE_CORE_H_
#include <nccl.h>
#include "coop.h"
#include "utility.h"

struct ncclDevComm;
typedef struct ncclDevComm ncclDevComm_t;

struct ncclTeam;
typedef struct ncclTeam ncclTeam_t;

// typedef struct ncclWindow_vidmem* ncclWindow_t; // in nccl.h
typedef struct ncclWindow_vidmem ncclWindow_vidmem_t;

struct ncclMultimemHandle;
typedef struct ncclMultimemHandle ncclMultimemHandle_t;

typedef uint32_t ncclDevResourceHandle;
typedef ncclDevResourceHandle ncclDevResourceHandle_t;

typedef uint32_t ncclGinSignal_t;
typedef uint32_t ncclGinCounter_t;

typedef struct {
  uint64_t opaque[2];
} ncclGinRequest_t;

struct ncclLsaBarrierHandle;
typedef struct ncclLsaBarrierHandle ncclLsaBarrierHandle_t;

struct ncclGinBarrierHandle;
typedef struct ncclGinBarrierHandle ncclGinBarrierHandle_t;

struct ncclCftBarrierHandle;
typedef struct ncclCftBarrierHandle ncclCftBarrierHandle_t;

struct ncclLLA2AHandle;
typedef struct ncclLLA2AHandle ncclLLA2AHandle_t;

typedef uint32_t ncclCftLeId;
typedef ncclCftLeId ncclCftLeId_t;

struct ncclTeam {
  int nRanks, rank, stride;
};

#if __cplusplus
template <typename T>
struct ncclSymPtr;
#endif

#if __cplusplus
struct ncclTeamTagWorld {};
struct ncclTeamTagLsa {};
struct ncclTeamTagRail {};
#endif

struct ncclDevCommRequirements;
typedef struct ncclDevCommRequirements ncclDevCommRequirements_t;

struct ncclDevResourceRequirements;
typedef struct ncclDevResourceRequirements ncclDevResourceRequirements_t;

struct ncclTeamRequirements;
typedef struct ncclTeamRequirements ncclTeamRequirements_t;

struct ncclCommProperties;
typedef struct ncclCommProperties ncclCommProperties_t;

typedef enum {
  NCCL_GIN_CONNECTION_NONE,
  NCCL_GIN_CONNECTION_FULL,
  NCCL_GIN_CONNECTION_RAIL,
  NCCL_GIN_CONNECTION_CUSTOM_STRIDE,
} ncclGinConnectionType_t;

typedef enum {
  NCCL_GIN_TYPE_NONE = 0, // Sentinel: accept any available backend (used in ncclDevCommRequirements)
  NCCL_GIN_TYPE_PROXY = 2, // intentionally not 1. Must match NCCL_NET_DEVICE_GIN_PROXY for backward compatibility
  NCCL_GIN_TYPE_GDAKI = 3, // intentionally not 2. Must match NCCL_NET_DEVICE_GIN_GDAKI for backward compatibility
  NCCL_GIN_TYPE_GPI = 4, // Must match NCCL_NET_DEVICE_GIN_GPI
  NCCL_GIN_TYPE_EFA_GDA = 5, // Must match NCCL_NET_DEVICE_GIN_EFA_GDA for backward compatibility
  NCCL_GIN_MAX_TYPES = 6,
} ncclGinType_t;

typedef enum {
  NCCL_CFT_TEAM_FLAT,
  NCCL_CFT_TEAM_HIER_MULTIMEM,
  NCCL_CFT_TEAM_HIER_LSA,
} ncclCftTeamMode_t;

typedef enum {
  NCCL_CFT_NONE = 0x0,
  NCCL_CFT = 0x1,
  NCCL_CFT_MULTIMEM = 0x2,
} ncclCftCap_t;

struct ncclDevCommRequirements {
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
};

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

struct ncclDevResourceRequirements {
  ncclDevResourceRequirements_t* next;
  size_t bufferSize, bufferAlign;
  ncclDevResourceHandle_t* outBufferHandle; // If non-null, target assigned during ncclDevCommCreate.
  int ginSignalCount;
  int ginCounterCount;
  ncclGinSignal_t* outGinSignalStart;
  ncclGinCounter_t* outGinCounterStart;
};

struct ncclTeamRequirements {
  ncclTeamRequirements_t* next;
  ncclTeam_t team;
  bool multimem;
  ncclMultimemHandle_t* outMultimemHandle; // If non-null, target assigned during ncclDevCommCreate.
};

#define NCCL_COMM_PROPERTIES_INITIALIZER \
  { \
    sizeof(ncclCommProperties_t),                  /* size */ \
    NCCL_API_MAGIC,                                /* magic */ \
    NCCL_VERSION_CODE,                             /* version */ \
  }

#define NCCL_GIN_MAX_ACTIVE_BACKENDS 4

struct ncclCommProperties {
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
};

NCCL_EXTERN_C __host__ ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props);
NCCL_EXTERN_C __host__ ncclResult_t ncclDevCommCreate(ncclComm_t comm, ncclDevCommRequirements_t const* reqs,
                                                      ncclDevComm_t* outDevComm);
NCCL_EXTERN_C __host__ ncclResult_t ncclDevCommDestroy(ncclComm_t comm, ncclDevComm_t const* devComm);

// VA pointer based query functions for host code
NCCL_EXTERN_C __host__ ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset,
                                                                 ncclMultimemHandle_t multimem, void** outPtr);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank,
                                                            void** outPtr);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer,
                                                             void** outPtr);

// CFT handle based query functions for host code
NCCL_EXTERN_C __host__ ncclResult_t ncclGetMultimemDeviceLeInfo(ncclWindow_t window, size_t offset, ncclCftLeId* leId,
                                                                size_t* leOffset);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetCftDeviceLeInfo(ncclWindow_t window, size_t offset, int peerCft,
                                                           ncclTeam_t cftTeam, ncclCftLeId* leId, size_t* leOffset);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetPeerDeviceLeInfo(ncclWindow_t window, size_t offset, int peerWorld,
                                                            ncclCftLeId* leId, size_t* leOffset);

////////////////////////////////////////////////////////////////////////////////
// Team API:
#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE ncclTeam ncclTeamWorld(ncclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamWorld(ncclComm_t comm);
#endif

#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamLsa(ncclComm_t comm);
#endif

#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE ncclTeam ncclTeamCft(ncclDevComm const&,
                                                         ncclCftTeamMode_t mode = NCCL_CFT_TEAM_FLAT);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE ncclTeam ncclTeamCftMultimem(ncclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
#if __cplusplus
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode = NCCL_CFT_TEAM_FLAT);
#else
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode);
#endif
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamCftMultimem(ncclComm_t comm);
#endif

NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE bool ncclTeamRankIsMember(ncclTeam_t a, ncclTeam_t b, int bPeer);
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankToTeam(ncclTeam_t a, ncclTeam_t b, int bPeer);

#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE int ncclTeamRankToWorld(ncclDevComm const&, ncclTeam, int rank);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank);
#endif

#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE int ncclTeamRankToLsa(ncclDevComm const&, ncclTeam, int rank);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank);
#endif

NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamInnerFactor(ncclTeam_t parent, int innerSize);
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamOuterFactor(ncclTeam_t parent, int innerSize);

// Interpret each team as a set of ranks. This function assumes that `subset`
// is a subset of `parent`. Thus the number of ranks in the set difference of
// `parent` minus `subset` is `super.nRanks - subset.nRanks`. Given `index` this
// function returns the index'th element of `parent` minus `subset`.
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankInDifference(ncclTeam_t parent, ncclTeam_t subset, int index);

// Equivalent to ncclTeamOuterFactor of lsa team.
#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE ncclTeam ncclTeamRail(ncclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamRail(ncclComm_t comm);
#endif

// Get offset of resource buffer within `comm.resourceWindow`.
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE size_t ncclGetResourceBufferOffset(ncclDevResourceHandle_t h);

#ifdef __CUDACC__
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclGetResourceBuffer(ncclDevComm const&, ncclDevResourceHandle);
#endif

////////////////////////////////////////////////////////////////////////////////
// Window API:

#ifdef __CUDACC__
// VA pointer based query functions
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetLocalPointer(ncclWindow_t w, size_t offset);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam tm, int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetMultimemPointer(ncclWindow_t w, size_t offset,
                                                                 ncclMultimemHandle mmHandle);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const&);

// CFT handle based query functions
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGetCftLeInfo(ncclWindow_t w, size_t offset, int peerCft, ncclTeam cftTeam,
                                                          ncclDevComm const& comm, ncclCftLeId* leId, size_t* leOffset);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGetPeerLeInfo(
  ncclWindow_t w, size_t offset, int peerWorld, ncclDevComm const& comm, ncclCftLeId* leId, size_t* leOffset);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGetMultimemLeInfo(ncclWindow_t w, size_t offset, ncclDevComm const&,
                                                               ncclCftLeId* leId, size_t* leOffset);
#endif

#ifdef __CUDACC__
// Convenience for combining ncclGet***Pointer() with resource handle.
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferLocalPointer(ncclDevComm const&, ncclDevResourceHandle);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaPointer(ncclDevComm const&, ncclDevResourceHandle,
                                                                          int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferPeerPointer(ncclDevComm const&, ncclDevResourceHandle,
                                                                           ncclTeam, int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferMultimemPointer(
  ncclDevComm const&, ncclDevResourceHandle, ncclMultimemHandle);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaMultimemPointer(ncclDevComm const&,
                                                                                  ncclDevResourceHandle);

// Convenience for combining ncclGet***LeInfo() with resource handle.
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGetResourceBufferCftLeInfo(
  ncclDevComm const&, ncclDevResourceHandle, int peerCft, ncclCftLeId* leId, size_t* leOffset);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGetResourceBufferPeerLeInfo(
  ncclDevComm const&, ncclDevResourceHandle, int peerWorld, ncclCftLeId* leId, size_t* leOffset);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGetResourceBufferMultimemLeInfo(ncclDevComm const&, ncclDevResourceHandle,
                                                                             ncclCftLeId* leId, size_t* leOffset);
#endif

#endif // _NCCL_DEVICE_CORE_H_
