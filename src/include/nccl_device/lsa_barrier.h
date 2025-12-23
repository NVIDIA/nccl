/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_MEM_BARRIER_H_
#define _NCCL_DEVICE_MEM_BARRIER_H_
#include "impl/core__types.h"

struct ncclLsaBarrierHandle;

NCCL_EXTERN_C __host__ ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);

#if NCCL_CHECK_CUDACC
template<typename Coop>
struct ncclLsaBarrierSession_internal;

template<typename Coop>
struct ncclLsaBarrierSession: ncclLsaBarrierSession_internal<Coop> {
  NCCL_DEVICE_INLINE ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeam, ncclLsaBarrierHandle, uint32_t index, bool multimem=false, ncclMultimemHandle mmHandle={});

  NCCL_DEVICE_INLINE ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeamTagLsa, uint32_t index, bool multimem=false);

  NCCL_DEVICE_INLINE ~ncclLsaBarrierSession();

  ncclLsaBarrierSession(ncclLsaBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE void arrive(Coop, cuda::memory_order);
  NCCL_DEVICE_INLINE void wait(Coop, cuda::memory_order);
  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order);
};
#endif

#endif // _NCCL_DEVICE_MEM_BARRIER_H_
