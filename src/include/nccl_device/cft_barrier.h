/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CFT_BARRIER_H_
#define _NCCL_DEVICE_CFT_BARRIER_H_
#include "cft.h"

NCCL_EXTERN_C __host__ ncclResult_t ncclCftBarrierCreateRequirement(
  ncclTeam_t team, int nBarriers, ncclCftBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);

#ifdef __CUDACC__
template <typename Coop>
struct ncclCftBarrierSession_internal;

template <typename Coop>
struct ncclCftBarrierSession : ncclCftBarrierSession_internal<Coop> {
  NCCL_DEVICE_INLINE ncclCftBarrierSession(Coop coop, ncclDevComm const& comm, uint32_t index, bool multimem = false);

  NCCL_DEVICE_INLINE ~ncclCftBarrierSession();

  ncclCftBarrierSession(ncclCftBarrierSession const&) = delete;

  NCCL_DEVICE_INLINE void arrive(Coop, cuda::memory_order, ncclMemProxyType producer, ncclMemProxyType consumer);
  NCCL_DEVICE_INLINE void wait(Coop, cuda::memory_order, ncclMemProxyType producer, ncclMemProxyType consumer);
  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order, ncclMemProxyType producer, ncclMemProxyType consumer);
  NCCL_DEVICE_INLINE ncclResult_t wait(Coop, cuda::memory_order, ncclMemProxyType producer, ncclMemProxyType consumer,
                                       uint64_t timeoutCycles);
  NCCL_DEVICE_INLINE ncclResult_t sync(Coop, cuda::memory_order, ncclMemProxyType producer, ncclMemProxyType consumer,
                                       uint64_t timeoutCycles);
};

#endif // __CUDACC__

#endif // _NCCL_DEVICE_CFT_BARRIER_H_
