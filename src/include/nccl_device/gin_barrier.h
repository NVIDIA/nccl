/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER_H_
#define _NCCL_DEVICE_GIN_BARRIER_H_
#include "core.h"
#include "gin.h"

struct ncclGinBarrierHandle;

NCCL_EXTERN_C __host__ ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t, ncclTeam_t, int nBarriers, ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);

#if NCCL_CHECK_CUDACC
enum class ncclGinFenceLevel {
  Relaxed
};

template<typename Coop>
struct ncclGinBarrierSession_internal;

template<typename Coop>
struct ncclGinBarrierSession: ncclGinBarrierSession_internal<Coop> {
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeam, ncclGinBarrierHandle, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagRail, uint32_t index);

  NCCL_DEVICE_INLINE ~ncclGinBarrierSession();

  ncclGinBarrierSession(ncclGinBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order, ncclGinFenceLevel);
};
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER_H_
