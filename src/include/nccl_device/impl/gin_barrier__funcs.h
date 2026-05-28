/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#include "gin_barrier__types.h"

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex
  ):
  ncclGinBarrierSession_internal<Coop>{coop, net, team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
  this->fenceAllContexts = false;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeamTagRail, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, net, ncclTeamRail(net.comm), net.comm.railGinBarrier, barrierIndex) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeamTagWorld, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, net, ncclTeamWorld(net.comm), net.comm.worldGinBarrier, barrierIndex) {
}
#endif

// All-contexts constructors: build a single-context gin (context 0) for the signal/wait
// path, then flip the `fenceAllContexts` flag so the fence iterates every GIN context on
// the comm.
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGinAllContexts allCtx, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex
  ):
  ncclGinBarrierSession_internal<Coop>{coop, ncclGin(allCtx.comm, 0), team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
  this->fenceAllContexts = true;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGinAllContexts allCtx, ncclTeamTagRail, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, allCtx, ncclTeamRail(allCtx.comm), allCtx.comm.railGinBarrier, barrierIndex) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGinAllContexts allCtx, ncclTeamTagWorld, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, allCtx, ncclTeamWorld(allCtx.comm), allCtx.comm.worldGinBarrier, barrierIndex) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::~ncclGinBarrierSession() {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<bool EnableTimeout>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession_internal<Coop>::syncInternal(Coop, cuda::memory_order ord,
                                                                      ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  uint64_t startCycle;
  ncclResult_t ret = ncclSuccess;
  this->coop.sync();

  // Drain outgoing puts/gets across either the bound context or every GIN context on the
  // comm. The multi-context branch flattens (ctx, peer) to 1D and assigns one (ctx, peer)
  // pair per thread so the flush is parallelised on both axes.
  auto fenceFlush = [&](cuda::memory_order order) {
    if (this->fenceAllContexts) {
      ncclTeam fenceTeam = this->net.comm.ginContextsRailed ? ncclTeamRail(this->net.comm)
                                                            : ncclTeamWorld(this->net.comm);
      int nCtx   = (int)this->net.comm.ginContextCount;
      int nPeers = fenceTeam.nRanks;
      int total  = nCtx * nPeers;
      NVCC_PRAGMA_UNROLL_DISABLED
      for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
        int ctx  = i / nPeers;
        int peer = i - ctx * nPeers;
        ncclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
        ncclGinRequest_t req;
        scratch.flushAsync(fenceTeam, (uint32_t)peer, &req);
        scratch.wait(req, ncclCoopThread{}, ncclGin_None{}, order);
      }
    } else {
      this->net.flush(this->coop, order);
    }
  };

  // Signal `peer` on `net` and wait for `peer`'s reciprocal signal on the calling rank's
  // matching slot. Returns ncclTimeout (timeout path only) if the wait exceeds budget.
  auto signalAndWait = [&](ncclGin& net, int peer) -> ncclResult_t {
    net.signal(
      this->team, peer, ncclGin_SignalInc{this->signal + this->team.rank}, ncclCoopThread(), ncclGin_None(),
      nccl::utility::releaseOrderOf(ord) != cuda::memory_order_relaxed
        ? cuda::thread_scope_thread
        : cuda::thread_scope_system);
    uint32_t* shadowPtr = (uint32_t*)net.getSignalShadowPtr(this->signal + peer);
    int waitVal = ++*shadowPtr;
    if NCCL_IF_CONSTEXPR (EnableTimeout) {
      while (true) {
        uint64_t got = net.readSignal(this->signal + peer, 32, nccl::utility::acquireOrderOf(ord));
        if (nccl::utility::rollingLessEq(static_cast<uint64_t>(waitVal), got, 32)) break;
        if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
      }
    } else {
      net.waitSignal(ncclCoopThread(), this->signal + peer, waitVal, 32, nccl::utility::acquireOrderOf(ord));
    }
    return ncclSuccess;
  };

  if NCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  }

  // Signal/wait with the calling rank's own slot included on Put so self-puts get the same
  // visibility guarantee as puts to other peers. Peer rotation `peer = (rank+1+i) % nRanks`
  // spreads the load and visits self last (only when Put is requested).
  int nPeerSigs = (fence & ncclGinFenceLevel::Put)
    ? this->team.nRanks
    : this->team.nRanks - 1;
  if (this->fenceAllContexts) {
    // Signal on each context, not just context 0: signals and puts on different QPs are
    // not ordered at the receiving NIC, so a ctx-0 signal could overtake an in-flight
    // ctx-X put. Each context has its own signal memory and shadow slot for the same
    // signal id, so no extra slot allocation is needed.
    int nCtx = (int)this->net.comm.ginContextCount;
    int total = nCtx * nPeerSigs;
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
      // Unflatten i back into (ctx, peerStep): ctx picks the GIN context to signal on,
      // peerStep is the index into the peer rotation (peer is computed just below).
      int ctx = i / nPeerSigs;
      int peerStep = i - ctx * nPeerSigs;
      int peer = 1 + this->team.rank + peerStep;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      ncclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
      if ((ret = signalAndWait(scratch, peer)) != ncclSuccess) goto exit;
    }
  } else {
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = this->coop.thread_rank(); i < nPeerSigs; i += this->coop.size()) {
      int peer = 1 + this->team.rank + i;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      if ((ret = signalAndWait(this->net, peer)) != ncclSuccess) goto exit;
    }
  }

  // Post-signal flush ensures our prior gets have completed before the barrier returns.
  // Placed after signal/wait so peers don't have to wait for our gets to complete.
  if (fence & ncclGinFenceLevel::Get) {
    fenceFlush(nccl::utility::acquireOrderOf(ord));
  }
  goto exit; // Silence a compiler warning.
exit:
  this->coop.sync();
  return ret;
}
#endif


#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrierSession<Coop>::sync(Coop coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  (void)(this->template syncInternal</*EnableTimeout=*/false>(coop, ord, fence, 0ULL));
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession<Coop>::sync(
    Coop coop, cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  return this->template syncInternal</*EnableTimeout=*/true>(coop, ord, fence, timeoutCycles);
}
#endif

// Free-function GIN barrier: thin wrappers around session construct + sync + destruct.
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGin gin, ncclTeam team, ncclGinBarrierHandle handle, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, team, handle, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGin gin, ncclTeamTagRail tag, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, tag, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGin gin, ncclTeamTagWorld tag, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, tag, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGinAllContexts allCtx, ncclTeam team, ncclGinBarrierHandle handle,
    uint32_t index, cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, team, handle, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGinAllContexts allCtx, ncclTeamTagRail tag, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, tag, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGinAllContexts allCtx, ncclTeamTagWorld tag, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, tag, index);
  session.sync(coop, ord, fence);
}
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
