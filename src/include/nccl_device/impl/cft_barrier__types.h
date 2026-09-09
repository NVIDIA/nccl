/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CFT_BARRIER__TYPES_H_
#define _NCCL_DEVICE_CFT_BARRIER__TYPES_H_

#include "cft__types.h"
#include "core__types.h"
#include "../cft_barrier.h"

#define NCCL_CFT_BARRIER_ALIGN 16
#define NCCL_CFT_BARRIER_GRAN NCCL_CFT_BARRIER_ALIGN

struct ncclCftBarrierHandle {
  ncclDevResourceHandle_t bufHandle;
  int nBarriers;
};

#ifdef __CUDACC__
template <typename Coop>
struct ncclCftBarrierSession_internal {
  Coop coop;
  ncclDevComm const& comm;
  ncclTeam team;
  ncclCftBarrierHandle handle;
  uint32_t index;
  bool multimem;
  uint32_t epoch;

  NCCL_DEVICE_INLINE bool useMultimem() const {
    return multimem;
  }

  NCCL_DEVICE_INLINE void mcInbox(ncclCftLeId* leId, size_t* leOffset) {
    ncclGetResourceBufferMultimemLeInfo(comm, handle.bufHandle, leId, leOffset);
    *leOffset += (2 * handle.nBarriers + index) * NCCL_CFT_BARRIER_GRAN;
  }

  NCCL_DEVICE_INLINE void ucInbox(int owner, int peer, ncclCftLeId* leId, size_t* leOffset) {
    ncclGetResourceBufferCftLeInfo(comm, handle.bufHandle, owner, leId, leOffset);
    *leOffset += (3 * handle.nBarriers + index * team.nRanks + peer) * NCCL_CFT_BARRIER_GRAN;
  }

  NCCL_DEVICE_INLINE uint32_t* mcInbox() {
#if CUDART_VERSION >= 13000
    void* state = ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
    return (uint32_t*)(reinterpret_cast<char (*)[NCCL_CFT_BARRIER_ALIGN]>(state) + (2 * handle.nBarriers + index));
#else
    return nullptr;
#endif
  }

  NCCL_DEVICE_INLINE uint32_t* ucInbox(int peer) {
#if CUDART_VERSION >= 13000
    void* state = ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
    return (uint32_t*)(reinterpret_cast<char (*)[NCCL_CFT_BARRIER_ALIGN]>(state) +
                       (3 * handle.nBarriers + index * team.nRanks + peer));
#else
    return nullptr;
#endif
  }

  template <bool EnableTimeout>
  NCCL_DEVICE_INLINE ncclResult_t waitInternal(Coop, cuda::memory_order, ncclMemProxyType producer,
                                               ncclMemProxyType consumer, uint64_t timeoutCycles);
};
#endif // __CUDACC__

#endif // _NCCL_DEVICE_CFT_BARRIER__TYPES_H_
