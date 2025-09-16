/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

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

struct ncclMultimemHandle;
typedef struct ncclMultimemHandle ncclMultimemHandle_t;

typedef uint32_t ncclDevResourceHandle;
typedef ncclDevResourceHandle ncclDevResourceHandle_t;

typedef uint32_t ncclGinSignal_t;
typedef uint32_t ncclGinCounter_t;

struct ncclLsaBarrierHandle;
typedef struct ncclLsaBarrierHandle ncclLsaBarrierHandle_t;

struct ncclGinBarrierHandle;
typedef struct ncclGinBarrierHandle ncclGinBarrierHandle_t;

struct ncclLLA2AHandle;
typedef struct ncclLLA2AHandle ncclLLA2AHandle_t;

struct ncclTeam {
  int nRanks, rank, stride;
};

#if __cplusplus
template<typename T> struct ncclSymPtr;
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

struct ncclDevCommRequirements {
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
};

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

NCCL_EXTERN_C __host__ ncclResult_t ncclDevCommCreate(ncclComm_t, ncclDevCommRequirements_t const*, ncclDevComm_t* outDevComm);
NCCL_EXTERN_C __host__ ncclResult_t ncclDevCommDestroy(ncclComm_t, ncclDevComm_t const* devComm);

////////////////////////////////////////////////////////////////////////////////
// Team API:

#if __cplusplus
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamWorld(ncclDevComm const&);
#endif
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamWorld(ncclComm_t);

#if __cplusplus
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclDevComm const&);
#endif
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamLsa(ncclComm_t);

NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE bool ncclTeamRankIsMember(ncclTeam_t a, ncclTeam_t b, int bPeer);
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankToTeam(ncclTeam_t a, ncclTeam_t b, int bPeer);

#if __cplusplus
NCCL_HOST_DEVICE_INLINE int ncclTeamRankToWorld(ncclDevComm const&, ncclTeam, int rank);
#endif
NCCL_EXTERN_C __host__ int ncclTeamRankToWorld(ncclComm_t, ncclTeam_t, int rank);

#if __cplusplus
NCCL_HOST_DEVICE_INLINE int ncclTeamRankToLsa(ncclDevComm const&, ncclTeam, int rank);
#endif
NCCL_EXTERN_C __host__ int ncclTeamRankToLsa(ncclComm_t, ncclTeam_t, int rank);

NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamInnerFactor(ncclTeam_t parent, int innerSize);
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamOuterFactor(ncclTeam_t parent, int innerSize);

// Interpret each team as a set of ranks. This function assumes that `subset`
// is a subset of `parent`. Thus the number of ranks in the set difference of
// `parent` minus `subset` is `super.nRanks - subset.nRanks`. Given `index` this
// function returns the index'th element of `parent` minus `subset`.
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankInDifference(ncclTeam_t parent, ncclTeam_t subset, int index);

// Equivalent to ncclTeamOuterFactor of lsa team.
#if __cplusplus
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamRail(ncclDevComm const&);
#endif
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamRail(ncclComm_t);

// Get offset of resource buffer within `comm.resourceWindow`.
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE size_t ncclGetResourceBufferOffset(ncclDevResourceHandle_t);

#if __CUDACC__
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclGetResourceBuffer(ncclDevComm const&, ncclDevResourceHandle);
#endif

////////////////////////////////////////////////////////////////////////////////
// Window API:

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclGetLocalPointer(ncclWindow_t w, size_t offset);
NCCL_DEVICE_INLINE void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam tm, int peer);
NCCL_DEVICE_INLINE void* ncclGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mmHandle);
NCCL_DEVICE_INLINE void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const&);
#endif

#if __CUDACC__
// Convenience for combining ncclGet***Pointer() with resource handle.
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLocalPointer(ncclDevComm const&, ncclDevResourceHandle);
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaPointer(ncclDevComm const&, ncclDevResourceHandle, int peer);
NCCL_DEVICE_INLINE void* ncclGetResourceBufferPeerPointer(ncclDevComm const&, ncclDevResourceHandle, ncclTeam, int peer);
NCCL_DEVICE_INLINE void* ncclGetResourceBufferMultimemPointer(ncclDevComm const&, ncclDevResourceHandle, ncclMultimemHandle);
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaMultimemPointer(ncclDevComm const&, ncclDevResourceHandle);
#endif

#endif // _NCCL_DEVICE_CORE_H_
