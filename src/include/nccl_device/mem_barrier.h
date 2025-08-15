#ifndef _NCCL_DEVICE_MEM_BARRIER_H_
#define _NCCL_DEVICE_MEM_BARRIER_H_
#include "impl/core__types.h"
#include <cuda/atomic>

struct ncclLsaBarrierHandle;

__host__ ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam, int nBarriers, ncclLsaBarrierHandle* outHandle, ncclDevResourceRequirements* outReq);

#if __CUDACC__
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
