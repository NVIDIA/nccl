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

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(Coop coop, ncclTeam innerTeam, ncclTeam outerTeam,
                                                                ncclGin gin, ncclLsaBarrierHandle innerHandle,
                                                                ncclGinBarrierHandle outerHandle, uint32_t index,
                                                                bool multimem, ncclMultimemHandle innerMmHandle)
  : ncclBarrierSession_internal<Coop>(
      coop, nccl::utility::present(gin),
      nccl::utility::present(coop, gin.comm, innerTeam, innerHandle, index, multimem, innerMmHandle),
      nccl::utility::present(coop, gin, outerTeam, outerHandle, index), nccl::utility::Absent()) {}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(Coop coop, ncclTeamTagWorld, ncclGin gin,
                                                                uint32_t index, bool multimem)
  : ncclBarrierSession_internal<Coop>(
      coop, nccl::utility::present(gin),
      nccl::utility::present(coop, gin.comm, ncclTeamLsa(gin.comm), gin.comm.hybridLsaBarrier, index, multimem,
                             gin.comm.lsaMultimem),
      nccl::utility::present(coop, gin, ncclTeamRail(gin.comm), gin.comm.hybridRailGinBarrier, index),
      nccl::utility::present(coop, gin,
                             ncclTeam{gin.comm.nRanks / gin.comm.ginContextStride,
                                      gin.comm.rank / gin.comm.ginContextStride, gin.comm.ginContextStride},
                             gin.comm.hybridDenseGinBarrier, index)) {}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(Coop coop, ncclTeamTagLsa, ncclDevComm const& comm,
                                                                uint32_t index, bool multimem)
  : ncclBarrierSession_internal<Coop>(coop, nccl::utility::Absent(),
                                      nccl::utility::present(coop, comm, ncclTeamLsa(comm), comm.hybridLsaBarrier,
                                                             index, multimem, comm.lsaMultimem),
                                      nccl::utility::Absent(), nccl::utility::Absent()) {}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(Coop coop, ncclTeamTagRail, ncclGin gin, uint32_t index)
  : ncclBarrierSession_internal<Coop>(coop, nccl::utility::present(gin), nccl::utility::Absent(),
                                      nccl::utility::present(coop, gin, ncclTeamRail(gin.comm),
                                                             gin.comm.hybridRailGinBarrier, index),
                                      nccl::utility::Absent()) {}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>& ncclBarrierSession<Coop>::lsaBarrier() {
  return this->innerLsaBar.thing;
}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>& ncclBarrierSession<Coop>::ginBarrier() {
  return this->outerRailGinBar.thing;
}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE void ncclBarrierSession<Coop>::selectBarrierAlgo(
  ncclGinFenceLevel fence, bool* needsLsaBarrier, bool* needsRailGinBarrier, bool* needsDenseGinBarrier) const {
  // Barrier on TeamLsa
  if (!this->gin.present) {
    *needsLsaBarrier = this->innerLsaBar.present;
    *needsRailGinBarrier = false;
    *needsDenseGinBarrier = false;
    return;
  }

  // Barrier on TeamRail
  if (!this->innerLsaBar.present) {
    *needsLsaBarrier = false;
    *needsRailGinBarrier = true;
    *needsDenseGinBarrier = false;
    return;
  }

  bool wantPut = fence & ncclGinFenceLevel::Put;
  const ncclDevComm& comm = this->gin.thing.comm;

  // Use the hierarchical barrier if:
  // (1) Remote data visibility is not needed or
  // (2) The only GIN connections that exist are the rail team
  if (!wantPut || comm.ginContextStride == ncclTeamRail(comm).stride) {
    *needsLsaBarrier = this->innerLsaBar.present;
    *needsRailGinBarrier = this->outerRailGinBar.present;
    *needsDenseGinBarrier = false;
    return;
  }

  // If all ranks are connected via GIN, use a dense GIN barrier (=full GIN barrier) to sync all ranks.
  if (comm.ginContextStride == 1) {
    *needsLsaBarrier = false;
    *needsRailGinBarrier = false;
    *needsDenseGinBarrier = this->outerDenseGinBar.present;
    return;
  }

  // Else, GIN is denser than the rail team but not fully connected.
  // LSA barrier first, then a GIN barrier among all connected ranks.
  // Note: GIN is never less dense than ncclTeamRail (checked in DevCommCreate).
  *needsLsaBarrier = this->innerLsaBar.present;
  *needsRailGinBarrier = false;
  *needsDenseGinBarrier = this->outerDenseGinBar.present;
}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE void ncclBarrierSession<Coop>::sync(Coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  bool needsLsaBarrier, needsRailGinBarrier, needsDenseGinBarrier;
  selectBarrierAlgo(fence, &needsLsaBarrier, &needsRailGinBarrier, &needsDenseGinBarrier);
  if (needsLsaBarrier) {
    this->innerLsaBar.thing.sync(
      this->coop, (needsRailGinBarrier || needsDenseGinBarrier) ? nccl::utility::releaseOrderOf(ord) : ord);
  }
  if (needsRailGinBarrier) {
    this->outerRailGinBar.thing.sync(this->coop, needsLsaBarrier ? nccl::utility::acquireOrderOf(ord) : ord, fence);
  }
  if (needsDenseGinBarrier) {
    this->outerDenseGinBar.thing.sync(this->coop, needsLsaBarrier ? nccl::utility::acquireOrderOf(ord) : ord, fence);
  }
}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclBarrierSession<Coop>::sync(Coop, cuda::memory_order ord, ncclGinFenceLevel fence,
                                                               uint64_t timeoutCycles) {
  bool needsLsaBarrier, needsRailGinBarrier, needsDenseGinBarrier;
  selectBarrierAlgo(fence, &needsLsaBarrier, &needsRailGinBarrier, &needsDenseGinBarrier);

  ncclResult_t lsaResult = ncclSuccess, railResult = ncclSuccess, denseResult = ncclSuccess;

  if (needsLsaBarrier) {
    uint64_t startCycle = clock64();
    lsaResult = this->innerLsaBar.thing.sync(
      this->coop, (needsRailGinBarrier || needsDenseGinBarrier) ? nccl::utility::releaseOrderOf(ord) : ord,
      timeoutCycles);
    uint64_t elapsed = clock64() - startCycle;
    timeoutCycles -= min(elapsed, timeoutCycles);
    // Because threads within a coop don't synchronize about the timeout condition,
    // we need to invoke the second barrier even if the first one times out,
    // to ensure that all the threads arrive at the coop sync.
  }

  if (needsRailGinBarrier) {
    railResult = this->outerRailGinBar.thing.sync(
      this->coop, needsLsaBarrier ? nccl::utility::acquireOrderOf(ord) : ord, fence, timeoutCycles);
  }

  if (needsDenseGinBarrier) {
    denseResult = this->outerDenseGinBar.thing.sync(
      this->coop, needsLsaBarrier ? nccl::utility::acquireOrderOf(ord) : ord, fence, timeoutCycles);
  }

  if (lsaResult != ncclSuccess) return lsaResult;
  if (railResult != ncclSuccess) return railResult;
  return denseResult;
}
#endif

// Free-function hybrid barrier: thin wrappers around session construct + sync + destruct.
#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(Coop coop, ncclTeamTagWorld tag, ncclGin gin, uint32_t index,
                                    cuda::memory_order ord, ncclGinFenceLevel fence, bool multimem) {
  ncclBarrierSession<Coop> session(coop, tag, gin, index, multimem);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(Coop coop, ncclTeamTagRail tag, ncclGin gin, uint32_t index, cuda::memory_order ord,
                                    ncclGinFenceLevel fence) {
  ncclBarrierSession<Coop> session(coop, tag, gin, index);
  session.sync(coop, ord, fence);
}
#endif

#endif // _NCCL_DEVICE_BARRIER__FUNCS_H_
