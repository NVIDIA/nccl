/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_BARRIER__FUNCS_H_
#include "barrier__types.h"
#include "lsa_barrier__funcs.h"
#include "gin_barrier__funcs.h"
#include "../utility.h"

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeam innerTeam, ncclTeam outerTeam, ncclGin gin,
    ncclLsaBarrierHandle innerHandle, ncclGinBarrierHandle outerHandle,
    uint32_t index, bool multimem, ncclMultimemHandle innerMmHandle
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::present(gin),
    nccl::utility::present(coop, gin.comm, innerTeam, innerHandle, index, multimem, innerMmHandle),
    nccl::utility::present(coop, gin, outerTeam, outerHandle, index)
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagWorld, ncclGin gin, uint32_t index, bool multimem
  ):
  ncclBarrierSession<Coop>(
    coop, ncclTeamLsa(gin.comm), ncclTeamRail(gin.comm), gin,
    gin.comm.lsaBarrier, gin.comm.railGinBarrier,
    index, multimem, gin.comm.lsaMultimem
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagLsa, ncclDevComm const& comm, uint32_t index, bool multimem
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::Absent(),
    nccl::utility::present(coop, comm, ncclTeamLsa(comm), comm.lsaBarrier, index, multimem, comm.lsaMultimem),
    nccl::utility::Absent()
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagRail, ncclGin gin, uint32_t index
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::present(gin),
    nccl::utility::Absent(),
    nccl::utility::present(coop, gin, ncclTeamRail(gin.comm), gin.comm.railGinBarrier, index)
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>& ncclBarrierSession<Coop>::lsaBarrier() {
  return this->innerLsaBar.thing;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>& ncclBarrierSession<Coop>::ginBarrier() {
  return this->outerGinBar.thing;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrierSession<Coop>::sync(Coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  if (this->innerLsaBar.present) {
    this->innerLsaBar.thing.sync(this->coop, this->outerGinBar.present ? nccl::utility::releaseOrderOf(ord) : ord);
  }
  if (this->outerGinBar.present) {
    this->outerGinBar.thing.sync(this->coop, this->innerLsaBar.present ? nccl::utility::acquireOrderOf(ord) : ord, fence);
  }
}
#endif

#endif // _NCCL_DEVICE_BARRIER__FUNCS_H_
