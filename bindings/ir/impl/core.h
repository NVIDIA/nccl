/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
// Internal implementation fragment; include only from nccl_device_ir.cu
// after its prerequisite headers.

#include <type_traits>

static_assert(sizeof(ncclIrDevComm) == sizeof(ncclDevComm), "DevComm storage size mismatch");
static_assert(alignof(ncclIrDevComm) == alignof(ncclDevComm), "DevComm storage alignment mismatch");
static_assert(std::is_trivially_copyable<ncclDevComm>::value, "DevComm byte ABI requires a trivially copyable type");

/* Team APIs */
NCCL_IR_DEVICE_INLINE ncclTeam ncclIrTeamWorld(ncclIrDevComm const* comm) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclTeamWorld(commImpl);
}
NCCL_IR_DEVICE_INLINE ncclTeam ncclIrTeamLsa(ncclIrDevComm const* comm) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclTeamLsa(commImpl);
}
NCCL_IR_DEVICE_INLINE ncclTeam ncclIrTeamCft(ncclIrDevComm const* comm, ncclCftTeamMode_t mode) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclTeamCft(commImpl, mode);
}
NCCL_IR_DEVICE_INLINE ncclTeam ncclIrTeamCftMultimem(ncclIrDevComm const* comm) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclTeamCftMultimem(commImpl);
}
NCCL_IR_DEVICE_INLINE bool ncclIrTeamRankIsMember(ncclTeam a, ncclTeam b, int bPeer) {
  return ncclTeamRankIsMember(a, b, bPeer);
}
NCCL_IR_DEVICE_INLINE int ncclIrTeamRankToTeam(ncclTeam a, ncclTeam b, int bPeer) {
  return ncclTeamRankToTeam(a, b, bPeer);
}
NCCL_IR_DEVICE_INLINE int ncclIrTeamRankToWorld(ncclIrDevComm const* comm, ncclTeam team, int rank) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclTeamRankToWorld(commImpl, team, rank);
}
NCCL_IR_DEVICE_INLINE int ncclIrTeamRankToLsa(ncclIrDevComm const* comm, ncclTeam team, int rank) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclTeamRankToLsa(commImpl, team, rank);
}
NCCL_IR_DEVICE_INLINE ncclTeam ncclIrTeamInnerFactor(ncclTeam parent, int innerSize) {
  return ncclTeamInnerFactor(parent, innerSize);
}
NCCL_IR_DEVICE_INLINE ncclTeam ncclIrTeamOuterFactor(ncclTeam parent, int innerSize) {
  return ncclTeamOuterFactor(parent, innerSize);
}
NCCL_IR_DEVICE_INLINE int ncclIrTeamRankInDifference(ncclTeam parent, ncclTeam subset, int index) {
  return ncclTeamRankInDifference(parent, subset, index);
}
NCCL_IR_DEVICE_INLINE ncclTeam ncclIrTeamRail(ncclIrDevComm const* comm) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclTeamRail(commImpl);
}

/* Window APIs */
NCCL_IR_DEVICE_INLINE void* ncclIrGetLocalPointer(ncclWindow_t w, size_t offset) {
  return ncclGetLocalPointer(w, offset);
}
NCCL_IR_DEVICE_INLINE void* ncclIrGetLsaPointer(ncclWindow_t w, size_t offset, int peer) {
  return ncclGetLsaPointer(w, offset, peer);
}
NCCL_IR_DEVICE_INLINE void* ncclIrGetPeerPointer(ncclWindow_t w, size_t offset, int peer) {
  return ncclGetPeerPointer(w, offset, peer);
}
NCCL_IR_DEVICE_INLINE void* ncclIrGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mmHandle) {
  return ncclGetMultimemPointer(w, offset, mmHandle);
}
NCCL_IR_DEVICE_INLINE void* ncclIrGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclIrDevComm const* comm) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclGetLsaMultimemPointer(w, offset, commImpl);
}
NCCL_IR_DEVICE_INLINE void ncclIrGetCftLeInfo(ncclWindow_t w, size_t offset, int peerCft, ncclTeam cftTeam,
                                              ncclIrDevComm const* comm, ncclCftLeId* leId, size_t* leOffset) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ncclGetCftLeInfo(w, offset, peerCft, cftTeam, commImpl, leId, leOffset);
}
NCCL_IR_DEVICE_INLINE void ncclIrGetPeerLeInfo(ncclWindow_t w, size_t offset, int peerWorld, ncclIrDevComm const* comm,
                                               ncclCftLeId* leId, size_t* leOffset) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ncclGetPeerLeInfo(w, offset, peerWorld, commImpl, leId, leOffset);
}
NCCL_IR_DEVICE_INLINE void ncclIrGetMultimemLeInfo(ncclWindow_t w, size_t offset, ncclIrDevComm const* comm,
                                                   ncclCftLeId* leId, size_t* leOffset) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ncclGetMultimemLeInfo(w, offset, commImpl, leId, leOffset);
}

/* Resource buffer APIs */
NCCL_IR_DEVICE_INLINE void ncclIrGetResourceBuffer(ncclIrDevComm const* comm, ncclDevResourceHandle h,
                                                   ncclWindow_t* window, size_t* offset) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  *window = commImpl.resourceWindow;
  *offset = ncclGetResourceBufferOffset(h);
}
NCCL_IR_DEVICE_INLINE size_t ncclIrGetResourceBufferOffset(ncclDevResourceHandle h) {
  return ncclGetResourceBufferOffset(h);
}
NCCL_IR_DEVICE_INLINE void* ncclIrGetResourceBufferLocalPointer(ncclIrDevComm const* comm, ncclDevResourceHandle h) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclGetResourceBufferLocalPointer(commImpl, h);
}
NCCL_IR_DEVICE_INLINE void* ncclIrGetResourceBufferLsaPointer(ncclIrDevComm const* comm, ncclDevResourceHandle h,
                                                              int peer) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclGetResourceBufferLsaPointer(commImpl, h, peer);
}
NCCL_IR_DEVICE_INLINE void* ncclIrGetResourceBufferPeerPointer(ncclIrDevComm const* comm, ncclDevResourceHandle h,
                                                               ncclTeam team, int peer) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclGetResourceBufferPeerPointer(commImpl, h, team, peer);
}
NCCL_IR_DEVICE_INLINE void* ncclIrGetResourceBufferMultimemPointer(ncclIrDevComm const* comm, ncclDevResourceHandle h,
                                                                   ncclMultimemHandle mmHandle) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclGetResourceBufferMultimemPointer(commImpl, h, mmHandle);
}
NCCL_IR_DEVICE_INLINE void* ncclIrGetResourceBufferLsaMultimemPointer(ncclIrDevComm const* comm,
                                                                      ncclDevResourceHandle h) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return ncclGetResourceBufferLsaMultimemPointer(commImpl, h);
}
NCCL_IR_DEVICE_INLINE void ncclIrGetResourceBufferCftLeInfo(ncclIrDevComm const* comm, ncclDevResourceHandle h,
                                                            int peerCft, ncclCftLeId* leId, size_t* leOffset) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ncclGetResourceBufferCftLeInfo(commImpl, h, peerCft, leId, leOffset);
}
NCCL_IR_DEVICE_INLINE void ncclIrGetResourceBufferPeerLeInfo(ncclIrDevComm const* comm, ncclDevResourceHandle h,
                                                             int peerWorld, ncclCftLeId* leId, size_t* leOffset) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ncclGetResourceBufferPeerLeInfo(commImpl, h, peerWorld, leId, leOffset);
}
NCCL_IR_DEVICE_INLINE void ncclIrGetResourceBufferMultimemLeInfo(ncclIrDevComm const* comm, ncclDevResourceHandle h,
                                                                 ncclCftLeId* leId, size_t* leOffset) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ncclGetResourceBufferMultimemLeInfo(commImpl, h, leId, leOffset);
}

/* ncclDevComm field accessors */
NCCL_IR_DEVICE_INLINE int ncclIrDevCommRank(ncclIrDevComm const* comm) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return commImpl.rank;
}
NCCL_IR_DEVICE_INLINE int ncclIrDevCommNRanks(ncclIrDevComm const* comm) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return commImpl.nRanks;
}
NCCL_IR_DEVICE_INLINE int ncclIrDevCommLsaRank(ncclIrDevComm const* comm) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return commImpl.lsaRank;
}
NCCL_IR_DEVICE_INLINE int ncclIrDevCommLsaSize(ncclIrDevComm const* comm) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return commImpl.lsaSize;
}

NCCL_IR_DEVICE_INLINE uint32_t ncclIrDevCommGinContextCount(ncclIrDevComm const* comm) {
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  return commImpl.ginContextCount;
}

NCCL_IR_DEVICE_INLINE void* ncclIrGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam tm, int peer) {
  return ncclGetPeerPointer(w, offset, tm, peer);
}
