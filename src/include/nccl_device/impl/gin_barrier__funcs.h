/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#include "gin_barrier__types.h"

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
  Coop coop, ncclGin net, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex)
  : ncclGinBarrierSession_internal<Coop>{coop, net, team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * ncclGinBarrierSlots(this->net._barrierOptions(), team.nRanks);
  this->fenceAllContexts = false;
}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(Coop coop, ncclGin net, ncclTeamTagRail,
                                                                      uint32_t barrierIndex)
  : ncclGinBarrierSession(coop, net, ncclTeamRail(net.comm), net.comm.railGinBarrier, barrierIndex) {}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(Coop coop, ncclGin net, ncclTeamTagWorld,
                                                                      uint32_t barrierIndex)
  : ncclGinBarrierSession(coop, net, ncclTeamWorld(net.comm), net.comm.worldGinBarrier, barrierIndex) {}
#endif

// All-contexts constructors: build a single-context gin (context 0) for the signal/wait
// path, then flip the `fenceAllContexts` flag so the fence iterates every GIN context on
// the comm.
#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
  Coop coop, ncclGinAllContexts allCtx, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex)
  : ncclGinBarrierSession_internal<Coop>{coop, ncclGin(allCtx.comm, 0), team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * ncclGinBarrierSlots(this->net._barrierOptions(), team.nRanks);
  this->fenceAllContexts = true;
}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(Coop coop, ncclGinAllContexts allCtx,
                                                                      ncclTeamTagRail, uint32_t barrierIndex)
  : ncclGinBarrierSession(coop, allCtx, ncclTeamRail(allCtx.comm), allCtx.comm.railGinBarrier, barrierIndex) {}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(Coop coop, ncclGinAllContexts allCtx,
                                                                      ncclTeamTagWorld, uint32_t barrierIndex)
  : ncclGinBarrierSession(coop, allCtx, ncclTeamWorld(allCtx.comm), allCtx.comm.worldGinBarrier, barrierIndex) {}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::~ncclGinBarrierSession() {}
#endif

#ifdef __CUDACC__
template <typename Coop>
template <bool EnableTimeout>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession_internal<Coop>::syncInternal(
  Coop, cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  // A backend that asked for the signal-efficient barrier runs the two-signal phase-bit barrier in syncPhase.
  // Every other backend runs the per-peer barrier below.
  if (this->net._barrierOptions() == NCCL_GIN_BARRIER_SIGNAL_EFFICIENT)
    return this->template syncPhase<EnableTimeout>(ord, fence, timeoutCycles);

  uint64_t startCycle;
  ncclResult_t ret = ncclSuccess;
  this->coop.sync();

  // Drain outgoing puts/gets across either the bound context or every GIN context on the
  // comm. The multi-context branch flattens (ctx, peer) to 1D and assigns one (ctx, peer)
  // pair per thread so the flush is parallelised on both axes.
  auto fenceFlush = [&](cuda::memory_order order) {
    if (this->fenceAllContexts) {
      int nCtx = (int)this->net.comm.ginContextCount;
      int nPeers = this->team.nRanks;
      int total = nCtx * nPeers;
      NVCC_PRAGMA_UNROLL_DISABLED
      for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
        int ctx = i / nPeers;
        int peer = i - ctx * nPeers;
        ncclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
        ncclGinRequest_t req;
        scratch.flushAsync(this->team, (uint32_t)peer, &req);
        scratch.wait(req, ncclCoopThread{}, ncclGin_None{}, order);
      }
    } else {
      this->net.flush(this->coop, order);
    }
  };
  auto signalPeer = [&](ncclGin& net, int peer) {
    net.signal(this->team, peer, ncclGin_SignalInc{this->signal + this->team.rank}, ncclCoopThread(), ncclGin_None(),
               nccl::utility::releaseOrderOf(ord) != cuda::memory_order_relaxed ? cuda::thread_scope_thread :
                                                                                  cuda::thread_scope_system);
  };

  auto waitForPeer = [&](ncclGin& net, int peer) -> ncclResult_t {
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

  if ((fence & ncclGinFenceLevel::Put) && !this->net._supportsStrongSignal()) {
    fenceFlush(nccl::utility::acquireOrderOf(ord));
  }

  // Signal/wait with the calling rank's own slot included on Put so self-puts get the same
  // visibility guarantee as puts to other peers. Peer rotation `peer = (rank+1+i) % nRanks`
  // spreads the load and visits self last (only when Put is requested).
  int nPeerSigs = (fence & ncclGinFenceLevel::Put) ? this->team.nRanks : this->team.nRanks - 1;
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
      signalPeer(scratch, peer);
    }
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
      int ctx = i / nPeerSigs;
      int peerStep = i - ctx * nPeerSigs;
      int peer = 1 + this->team.rank + peerStep;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      ncclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
      if ((ret = waitForPeer(scratch, peer)) != ncclSuccess) goto exit;
    }
  } else {
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = this->coop.thread_rank(); i < nPeerSigs; i += this->coop.size()) {
      int peer = 1 + this->team.rank + i;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      signalPeer(this->net, peer);
    }
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = this->coop.thread_rank(); i < nPeerSigs; i += this->coop.size()) {
      int peer = 1 + this->team.rank + i;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      if ((ret = waitForPeer(this->net, peer)) != ncclSuccess) goto exit;
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

#ifdef __CUDACC__
// Two-signal phase-bit barrier: every arrival increments one counting signal and the waiter compares it
// against the cumulative arrivals it expects. Alternating between two slots prevents a rank that races into
// the next barrier from inflating the count of the one its peers are still in. Kept separate from the
// per-peer barrier above, at the cost of some duplication, so that body stays byte-for-byte unchanged.

template <typename Coop>
template <bool EnableTimeout>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession_internal<Coop>::syncPhase(
  cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  uint64_t startCycle;
  ncclResult_t ret = ncclSuccess;
  int nCtx = this->fenceAllContexts ? (int)this->net.comm.ginContextCount : 1;
  ncclGinBarrierState st = {0, 0};
  this->coop.sync();

  // Same flush as syncInternal's fenceFlush: the bound context, or every context when the fence spans all.
  auto fenceFlush = [&](cuda::memory_order order) {
    if (this->fenceAllContexts) {
      int nPeers = this->team.nRanks;
      int total = nCtx * nPeers;
      NVCC_PRAGMA_UNROLL_DISABLED
      for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
        int ctx = i / nPeers;
        int peer = i - ctx * nPeers;
        ncclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
        ncclGinRequest_t req;
        scratch.flushAsync(this->team, (uint32_t)peer, &req);
        scratch.wait(req, ncclCoopThread{}, ncclGin_None{}, order);
      }
    } else {
      this->net.flush(this->coop, order);
    }
  };
  // Load shadow state for one context and the arrival count that ends this barrier.
  auto loadState = [&](ncclGin& net, int nArrivals) -> ncclGinBarrierState {
    uint64_t s0 = *net.getSignalShadowPtr(this->signal + 0);
    int phase = (int)(s0 >> ncclGinBarrierPhaseBit);
    uint32_t seen = (uint32_t)(phase == 0 ? s0 : *net.getSignalShadowPtr(this->signal + 1));
    return ncclGinBarrierState{phase, seen + (uint32_t)nArrivals};
  };
  // Record the arrivals just consumed and hand the next barrier the other slot.
  auto storeState = [&](ncclGin& net, ncclGinBarrierState state) {
    uint64_t* s0 = net.getSignalShadowPtr(this->signal + 0);
    if (state.phase == 0) {
      *s0 = (uint64_t(1) << ncclGinBarrierPhaseBit) | state.waitVal;
    } else {
      *net.getSignalShadowPtr(this->signal + 1) = state.waitVal;
      *s0 &= ~(uint64_t(1) << ncclGinBarrierPhaseBit);
    }
  };
  auto signalPeer = [&](ncclGin& net, int peer, int phase) {
    net.signal(this->team, peer, ncclGin_SignalInc{this->signal + phase}, ncclCoopThread(), ncclGin_None(),
               nccl::utility::releaseOrderOf(ord) != cuda::memory_order_relaxed ? cuda::thread_scope_thread :
                                                                                  cuda::thread_scope_system);
  };
  auto waitArrivals = [&](ncclGin& net, ncclGinBarrierState state) -> ncclResult_t {
    if NCCL_IF_CONSTEXPR (EnableTimeout) {
      while (true) {
        uint64_t got = net.readSignal(this->signal + state.phase, 32, nccl::utility::acquireOrderOf(ord));
        if (nccl::utility::rollingLessEq(static_cast<uint64_t>(state.waitVal), got, 32)) break;
        if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
      }
    } else {
      net.waitSignal(ncclCoopThread(), this->signal + state.phase, state.waitVal, 32,
                     nccl::utility::acquireOrderOf(ord));
    }
    return ncclSuccess;
  };

  if NCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  }

  // The coop.sync keeps a thread from signalling a peer that another thread is still flushing.
  if ((fence & ncclGinFenceLevel::Put) && !this->net._supportsStrongSignal()) {
    fenceFlush(nccl::utility::acquireOrderOf(ord));
    this->coop.sync();
  }

  // Same signal/wait rotation and self-inclusion as syncInternal. nPeerSigs is also the count this rank
  // receives: every other rank signals it regardless of the fence they asked for.
  int nPeerSigs = (fence & ncclGinFenceLevel::Put) ? this->team.nRanks : this->team.nRanks - 1;
  if (this->fenceAllContexts) {
    // Signals on every context, as in syncInternal. Each context keeps its own phase in its own shadows.
    int total = nCtx * nPeerSigs;
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
      int ctx = i / nPeerSigs;
      int peerStep = i - ctx * nPeerSigs;
      int peer = 1 + this->team.rank + peerStep;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      ncclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
      signalPeer(scratch, peer, loadState(scratch, nPeerSigs).phase);
    }
    // One counter per context, so the wait loop is over contexts only.
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int ctx = this->coop.thread_rank(); ctx < nCtx; ctx += this->coop.size()) {
      ncclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
      ncclResult_t r = waitArrivals(scratch, loadState(scratch, nPeerSigs));
      if (r != ncclSuccess) ret = r;
    }
  } else {
    st = loadState(this->net, nPeerSigs);
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = this->coop.thread_rank(); i < nPeerSigs; i += this->coop.size()) {
      int peer = 1 + this->team.rank + i;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      signalPeer(this->net, peer, st.phase);
    }
    // One counter, so thread 0 polls it and the trailing coop.sync() publishes its acquire. Only thread 0 can
    // therefore return ncclTimeout; it records it rather than jumping past the coop-collective Get fence.
    if (this->coop.thread_rank() == 0) {
      ncclResult_t r = waitArrivals(this->net, st);
      if (r != ncclSuccess) ret = r;
    }
  }

  if (fence & ncclGinFenceLevel::Get) {
    fenceFlush(nccl::utility::acquireOrderOf(ord));
  }
  this->coop.sync();
  // Store after the sync, once every thread is past its load. Skip it on timeout.
  if (ret == ncclSuccess) {
    if (this->fenceAllContexts) {
      NVCC_PRAGMA_UNROLL_DISABLED
      for (int ctx = this->coop.thread_rank(); ctx < nCtx; ctx += this->coop.size()) {
        ncclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
        storeState(scratch, loadState(scratch, nPeerSigs));
      }
    } else if (this->coop.thread_rank() == 0) {
      storeState(this->net, st);
    }
  }
  return ret;
}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrierSession<Coop>::sync(Coop coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  (void)(this->template syncInternal</*EnableTimeout=*/false>(coop, ord, fence, 0ULL));
}
#endif

#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession<Coop>::sync(Coop coop, cuda::memory_order ord,
                                                                  ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  return this->template syncInternal</*EnableTimeout=*/true>(coop, ord, fence, timeoutCycles);
}
#endif

// Free-function GIN barrier: thin wrappers around session construct + sync + destruct.
#ifdef __CUDACC__
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGin gin, ncclTeam team, ncclGinBarrierHandle handle,
                                       uint32_t index, cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, team, handle, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGin gin, ncclTeamTagRail tag, uint32_t index,
                                       cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, tag, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGin gin, ncclTeamTagWorld tag, uint32_t index,
                                       cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, tag, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGinAllContexts allCtx, ncclTeam team, ncclGinBarrierHandle handle,
                                       uint32_t index, cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, team, handle, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGinAllContexts allCtx, ncclTeamTagRail tag, uint32_t index,
                                       cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, tag, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGinAllContexts allCtx, ncclTeamTagWorld tag, uint32_t index,
                                       cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, tag, index);
  session.sync(coop, ord, fence);
}
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
