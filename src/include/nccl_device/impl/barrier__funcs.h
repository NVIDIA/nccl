/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_BARRIER__FUNCS_H_
#include "barrier__types.h"
#include "lsa_barrier__funcs.h"
#if defined(NCCL_OS_LINUX)
#include "gin_barrier__funcs.h"
#endif
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
    nccl::utility::present(coop, gin, outerTeam, outerHandle, index),
    nccl::utility::Absent()
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagWorld, ncclGin gin, uint32_t index, bool multimem
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::present(gin),
    nccl::utility::present(coop, gin.comm, ncclTeamLsa(gin.comm), gin.comm.hybridLsaBarrier, index, multimem, gin.comm.lsaMultimem),
    nccl::utility::present(coop, gin, ncclTeamRail(gin.comm), gin.comm.hybridRailGinBarrier, index),
    nccl::utility::present(coop, gin, ncclTeamWorld(gin.comm), gin.comm.hybridWorldGinBarrier, index)
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
    nccl::utility::present(coop, comm, ncclTeamLsa(comm), comm.hybridLsaBarrier, index, multimem, comm.lsaMultimem),
    nccl::utility::Absent(),
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
    nccl::utility::present(coop, gin, ncclTeamRail(gin.comm), gin.comm.hybridRailGinBarrier, index),
    nccl::utility::Absent()
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
  return this->outerRailGinBar.thing;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE bool ncclBarrierSession<Coop>::useWorldForFence(ncclGinFenceLevel fence) const {
  bool wantPut = fence & ncclGinFenceLevel::Put;
  return wantPut
      && this->gin.present
      && !this->gin.thing.comm.ginContextsRailed
      && this->outerWorldGinBar.present;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrierSession<Coop>::sync(Coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  if (this->useWorldForFence(fence)) {
    this->outerWorldGinBar.thing.sync(this->coop, ord, fence);
    return;
  }

  if (this->innerLsaBar.present) {
    this->innerLsaBar.thing.sync(this->coop, this->outerRailGinBar.present ? nccl::utility::releaseOrderOf(ord) : ord);
  }
  if (this->outerRailGinBar.present) {
    this->outerRailGinBar.thing.sync(this->coop, this->innerLsaBar.present ? nccl::utility::acquireOrderOf(ord) : ord, fence);
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclBarrierSession<Coop>::sync(
    Coop, cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  if (this->useWorldForFence(fence)) {
    return this->outerWorldGinBar.thing.sync(this->coop, ord, fence, timeoutCycles);
  }

  ncclResult_t lsaResult = ncclSuccess, railResult = ncclSuccess;

  if (this->innerLsaBar.present) {
    uint64_t startCycle = clock64();
    lsaResult = this->innerLsaBar.thing.sync(
      this->coop,
      this->outerRailGinBar.present ? nccl::utility::releaseOrderOf(ord) : ord,
      timeoutCycles
    );
    uint64_t elapsed = clock64() - startCycle;
    timeoutCycles -= min(elapsed, timeoutCycles);
    // Because threads within a coop don't synchronize about the timeout condition,
    // we need to invoke the second barrier even if the first one times out,
    // to ensure that all the threads arrive at the coop sync.
  }

  if (this->outerRailGinBar.present) {
    railResult = this->outerRailGinBar.thing.sync(
      this->coop,
      this->innerLsaBar.present ? nccl::utility::acquireOrderOf(ord) : ord,
      fence,
      timeoutCycles
    );
  }

  if (lsaResult != ncclSuccess) return lsaResult;
  return railResult;
}
#endif

// Free-function hybrid barrier: thin wrappers around session construct + sync + destruct.
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(
    Coop coop, ncclTeamTagWorld tag, ncclGin gin, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence, bool multimem) {
  ncclBarrierSession<Coop> session(coop, tag, gin, index, multimem);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(
    Coop coop, ncclTeamTagRail tag, ncclGin gin, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclBarrierSession<Coop> session(coop, tag, gin, index);
  session.sync(coop, ord, fence);
}
#endif

#endif // _NCCL_DEVICE_BARRIER__FUNCS_H_
