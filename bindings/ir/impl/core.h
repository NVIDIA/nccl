/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER_IMPL_CORE_H_
#define _NCCL_DEVICE_WRAPPER_IMPL_CORE_H_

// Internal implementation fragment; include only from nccl_device_wrapper.cu
// after its prerequisite headers.

/* Team APIs */
NCCL_DEVICE_INLINE ncclTeam ncclIrTeamWorld(ncclDevComm const* comm) {
  return ncclTeamWorld(*comm);
}
NCCL_DEVICE_INLINE ncclTeam ncclIrTeamLsa(ncclDevComm const* comm) {
  return ncclTeamLsa(*comm);
}
NCCL_DEVICE_INLINE ncclTeam ncclIrTeamCft(ncclDevComm const* comm, ncclCftTeamMode_t mode) {
  return ncclTeamCft(*comm, mode);
}
NCCL_DEVICE_INLINE ncclTeam ncclIrTeamCftMultimem(ncclDevComm const* comm) {
  return ncclTeamCftMultimem(*comm);
}
NCCL_DEVICE_INLINE int ncclIrTeamRankToWorld(ncclDevComm const* comm, ncclTeam team, int rank) {
  return ncclTeamRankToWorld(*comm, team, rank);
}
NCCL_DEVICE_INLINE int ncclIrTeamRankToLsa(ncclDevComm const* comm, ncclTeam team, int rank) {
  return ncclTeamRankToLsa(*comm, team, rank);
}
NCCL_DEVICE_INLINE ncclTeam ncclIrTeamRail(ncclDevComm const* comm) {
  return ncclTeamRail(*comm);
}

/* Window APIs */
NCCL_DEVICE_INLINE void* ncclIrGetLocalPointer(ncclWindow_t w, size_t offset) {
  return ncclGetLocalPointer(w, offset);
}
NCCL_DEVICE_INLINE void* ncclIrGetLsaPointer(ncclWindow_t w, size_t offset, int peer) {
  return ncclGetLsaPointer(w, offset, peer);
}
NCCL_DEVICE_INLINE void* ncclIrGetPeerPointer(ncclWindow_t w, size_t offset, int peer) {
  return ncclGetPeerPointer(w, offset, peer);
}
NCCL_DEVICE_INLINE void* ncclIrGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mmHandle) {
  return ncclGetMultimemPointer(w, offset, mmHandle);
}
NCCL_DEVICE_INLINE void* ncclIrGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const* comm) {
  return ncclGetLsaMultimemPointer(w, offset, *comm);
}
NCCL_DEVICE_INLINE void ncclIrGetCftLeInfo(ncclWindow_t w, size_t offset, int peerCft, ncclTeam cftTeam,
                                           ncclDevComm const* comm, ncclCftLeId* leId, size_t* leOffset) {
  ncclGetCftLeInfo(w, offset, peerCft, cftTeam, *comm, leId, leOffset);
}
NCCL_DEVICE_INLINE void ncclIrGetPeerLeInfo(ncclWindow_t w, size_t offset, int peerWorld, ncclDevComm const* comm,
                                            ncclCftLeId* leId, size_t* leOffset) {
  ncclGetPeerLeInfo(w, offset, peerWorld, *comm, leId, leOffset);
}
NCCL_DEVICE_INLINE void ncclIrGetMultimemLeInfo(ncclWindow_t w, size_t offset, ncclDevComm const* comm,
                                                ncclCftLeId* leId, size_t* leOffset) {
  ncclGetMultimemLeInfo(w, offset, *comm, leId, leOffset);
}

/* Resource buffer APIs */
NCCL_DEVICE_INLINE void* ncclIrGetResourceBufferLocalPointer(ncclDevComm const* comm, ncclDevResourceHandle h) {
  return ncclGetResourceBufferLocalPointer(*comm, h);
}
NCCL_DEVICE_INLINE void* ncclIrGetResourceBufferLsaPointer(ncclDevComm const* comm, ncclDevResourceHandle h, int peer) {
  return ncclGetResourceBufferLsaPointer(*comm, h, peer);
}
NCCL_DEVICE_INLINE void* ncclIrGetResourceBufferPeerPointer(ncclDevComm const* comm, ncclDevResourceHandle h,
                                                            ncclTeam team, int peer) {
  return ncclGetResourceBufferPeerPointer(*comm, h, team, peer);
}
NCCL_DEVICE_INLINE void* ncclIrGetResourceBufferMultimemPointer(ncclDevComm const* comm, ncclDevResourceHandle h,
                                                                ncclMultimemHandle mmHandle) {
  return ncclGetResourceBufferMultimemPointer(*comm, h, mmHandle);
}
NCCL_DEVICE_INLINE void* ncclIrGetResourceBufferLsaMultimemPointer(ncclDevComm const* comm, ncclDevResourceHandle h) {
  return ncclGetResourceBufferLsaMultimemPointer(*comm, h);
}
NCCL_DEVICE_INLINE void ncclIrGetResourceBufferCftLeInfo(ncclDevComm const* comm, ncclDevResourceHandle h, int peerCft,
                                                         ncclCftLeId* leId, size_t* leOffset) {
  ncclGetResourceBufferCftLeInfo(*comm, h, peerCft, leId, leOffset);
}
NCCL_DEVICE_INLINE void ncclIrGetResourceBufferPeerLeInfo(ncclDevComm const* comm, ncclDevResourceHandle h,
                                                          int peerWorld, ncclCftLeId* leId, size_t* leOffset) {
  ncclGetResourceBufferPeerLeInfo(*comm, h, peerWorld, leId, leOffset);
}
NCCL_DEVICE_INLINE void ncclIrGetResourceBufferMultimemLeInfo(ncclDevComm const* comm, ncclDevResourceHandle h,
                                                              ncclCftLeId* leId, size_t* leOffset) {
  ncclGetResourceBufferMultimemLeInfo(*comm, h, leId, leOffset);
}

/* ncclDevComm field accessors */
NCCL_DEVICE_INLINE int ncclIrDevCommRank(ncclDevComm const* comm) {
  return comm->rank;
}
NCCL_DEVICE_INLINE int ncclIrDevCommNRanks(ncclDevComm const* comm) {
  return comm->nRanks;
}
NCCL_DEVICE_INLINE int ncclIrDevCommLsaRank(ncclDevComm const* comm) {
  return comm->lsaRank;
}
NCCL_DEVICE_INLINE int ncclIrDevCommLsaSize(ncclDevComm const* comm) {
  return comm->lsaSize;
}
NCCL_DEVICE_INLINE ncclLsaBarrierHandle ncclIrDevCommLsaBarrier(ncclDevComm const* comm) {
  return comm->lsaBarrier;
}
NCCL_DEVICE_INLINE ncclGinBarrierHandle ncclIrDevCommRailGinBarrier(ncclDevComm const* comm) {
  return comm->railGinBarrier;
}
NCCL_DEVICE_INLINE ncclLsaBarrierHandle ncclIrDevCommHybridLsaBarrier(ncclDevComm const* comm) {
  return comm->hybridLsaBarrier;
}
NCCL_DEVICE_INLINE ncclGinBarrierHandle ncclIrDevCommHybridRailGinBarrier(ncclDevComm const* comm) {
  return comm->hybridRailGinBarrier;
}
NCCL_DEVICE_INLINE ncclGinBarrierHandle ncclIrDevCommWorldGinBarrier(ncclDevComm const* comm) {
  return comm->worldGinBarrier;
}
NCCL_DEVICE_INLINE ncclMultimemHandle ncclIrDevCommLsaMultimem(ncclDevComm const* comm) {
  return comm->lsaMultimem;
}

NCCL_DEVICE_INLINE void* ncclIrGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam tm, int peer) {
  return ncclGetPeerPointer(w, offset, tm, peer);
}

#endif // _NCCL_DEVICE_WRAPPER_IMPL_CORE_H_
