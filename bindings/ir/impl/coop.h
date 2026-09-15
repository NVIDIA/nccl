/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER_IMPL_COOP_H_
#define _NCCL_DEVICE_WRAPPER_IMPL_COOP_H_

// Internal implementation fragment; include only from nccl_device_wrapper.cu
// after its prerequisite headers.

NCCL_DEVICE_INLINE void ncclCoopAnyInitThread(ncclCoopAny* coop) {
  ::new (coop) ncclCoopAny(ncclCoopThread());
}
NCCL_DEVICE_INLINE void ncclCoopAnyInitWarp(ncclCoopAny* coop) {
  ::new (coop) ncclCoopAny(ncclCoopWarp());
}
NCCL_DEVICE_INLINE void ncclCoopAnyInitLanes(ncclCoopAny* coop, uint32_t lane_mask) {
  ::new (coop) ncclCoopAny(ncclCoopLanes(lane_mask));
}
NCCL_DEVICE_INLINE void ncclCoopAnyInitWarpSpan(ncclCoopAny* coop, int warp0, int nWarps, int id) {
  ::new (coop) ncclCoopAny(ncclCoopWarpSpan(warp0, nWarps, id));
}
NCCL_DEVICE_INLINE void ncclCoopAnyInitCta(ncclCoopAny* coop) {
  ::new (coop) ncclCoopAny(ncclCoopCta());
}

NCCL_DEVICE_INLINE int ncclCoopThreadRank(const ncclCoopAny* coop) {
  return coop->thread_rank();
}
NCCL_DEVICE_INLINE int ncclCoopSize(const ncclCoopAny* coop) {
  return coop->size();
}
NCCL_DEVICE_INLINE int ncclCoopNumThreads(const ncclCoopAny* coop) {
  return coop->num_threads();
}
NCCL_DEVICE_INLINE void ncclCoopSync(const ncclCoopAny* coop) {
  const_cast<ncclCoopAny*>(coop)->sync();
}

#endif // _NCCL_DEVICE_WRAPPER_IMPL_COOP_H_
