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

#define NCCL_GIN_MAX_ACTIVE_BACKENDS 4

////////////////////////////////////////////////////////////////////////////////
// Team API:
#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE ncclTeam ncclTeamWorld(ncclDevComm const&);
#endif
#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclDevComm const&);
#endif

#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE ncclTeam ncclTeamCft(ncclDevComm const&,
                                                         ncclCftTeamMode_t mode = NCCL_CFT_TEAM_FLAT);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE ncclTeam ncclTeamCftMultimem(ncclDevComm const&);
#endif
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE bool ncclTeamRankIsMember(ncclTeam_t a, ncclTeam_t b, int bPeer);
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankToTeam(ncclTeam_t a, ncclTeam_t b, int bPeer);

#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE int ncclTeamRankToWorld(ncclDevComm const&, ncclTeam, int rank);
#endif
#ifdef __CUDACC__
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE int ncclTeamRankToLsa(ncclDevComm const&, ncclTeam, int rank);
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
