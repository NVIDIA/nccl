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

/* ncclDevComm field accessors */
NCCL_DEVICE_INLINE int ncclDevComm_Rank(ncclDevComm const* comm) {
  return comm->rank;
}
NCCL_DEVICE_INLINE int ncclDevComm_NRanks(ncclDevComm const* comm) {
  return comm->nRanks;
}
NCCL_DEVICE_INLINE int ncclDevComm_LsaRank(ncclDevComm const* comm) {
  return comm->lsaRank;
}
NCCL_DEVICE_INLINE int ncclDevComm_LsaSize(ncclDevComm const* comm) {
  return comm->lsaSize;
}
NCCL_DEVICE_INLINE ncclLsaBarrierHandle ncclDevComm_LsaBarrier(ncclDevComm const* comm) {
  return comm->lsaBarrier;
}
NCCL_DEVICE_INLINE ncclGinBarrierHandle ncclDevComm_RailGinBarrier(ncclDevComm const* comm) {
  return comm->railGinBarrier;
}
NCCL_DEVICE_INLINE ncclLsaBarrierHandle ncclDevComm_HybridLsaBarrier(ncclDevComm const* comm) {
  return comm->hybridLsaBarrier;
}
NCCL_DEVICE_INLINE ncclGinBarrierHandle ncclDevComm_HybridRailGinBarrier(ncclDevComm const* comm) {
  return comm->hybridRailGinBarrier;
}
NCCL_DEVICE_INLINE ncclGinBarrierHandle ncclDevComm_WorldGinBarrier(ncclDevComm const* comm) {
  return comm->worldGinBarrier;
}
NCCL_DEVICE_INLINE ncclMultimemHandle ncclDevComm_LsaMultimem(ncclDevComm const* comm) {
  return comm->lsaMultimem;
}

NCCL_DEVICE_INLINE void* ncclGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam tm, int peer) {
  return ncclGetPeerPointer(w, offset, tm, peer);
}

#endif // _NCCL_DEVICE_WRAPPER_IMPL_CORE_H_
