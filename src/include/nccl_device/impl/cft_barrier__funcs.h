/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CFT_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_CFT_BARRIER__FUNCS_H_

#include "cft__funcs.h"
#include "cft_barrier__types.h"
#include "comm__types.h"

#ifdef __CUDACC__
constexpr uint32_t ncclCftOpElemGran = 4;
constexpr uint32_t ncclCftOpByteGran = ncclCftOpElemGran * sizeof(uint32_t);

template <typename Coop>
NCCL_DEVICE_INLINE ncclCftBarrierSession<Coop>::ncclCftBarrierSession(Coop coop, struct ncclDevComm const& comm,
                                                                      uint32_t index, bool multimem)
  : ncclCftBarrierSession_internal<Coop>{coop,
                                         comm,
                                         multimem ? ncclTeamCftMultimem(comm) : ncclTeamCft(comm),
                                         multimem ? comm.cftMultimemBarrier : comm.cftBarrier,
                                         index,
                                         multimem,
                                         /*epoch*/ 0} {
  void* state = ncclGetResourceBufferLocalPointer(comm, this->handle.bufHandle);
  this->epoch = *(uint32_t*)(reinterpret_cast<char (*)[NCCL_CFT_BARRIER_GRAN]>(state) +
                             (this->useMultimem() ? 0 : 1) * this->handle.nBarriers + this->index);
}

template <typename Coop>
NCCL_DEVICE_INLINE ncclCftBarrierSession<Coop>::~ncclCftBarrierSession() {
  void* state = ncclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
  if (this->coop.thread_rank() == 0) {
    *(uint32_t*)(reinterpret_cast<char (*)[NCCL_CFT_BARRIER_GRAN]>(state) +
                 (this->useMultimem() ? 0 : 1) * this->handle.nBarriers + this->index) = this->epoch;
  }
  this->coop.sync();
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclCftBarrierSession<Coop>::arrive(Coop, cuda::memory_order order, ncclMemProxyType producer,
                                                            ncclMemProxyType consumer) {
  this->coop.sync();
  if (this->coop.thread_rank() == 0) {
    alignas(NCCL_CFT_BARRIER_ALIGN) __shared__ uint32_t payload[ncclCftOpElemGran];
    payload[0] = this->useMultimem() ? 1 : this->epoch + 1;
    for (int i = 1; i < ncclCftOpElemGran; i++) payload[i] = 0;

    ncclCoopThread coop;
    __shared__ ncclCftSmem cftSmem;
    ncclCft<ncclCoopThread> cft{coop, cftSmem};

    ncclMemFence(coop, nccl::utility::releaseOrderOf(order), producer, consumer, ncclMemFenceScope::Sys);

    // Push generic proxy shared memory updates to point of consistency for fabric proxy access
    if (!(order == cuda::memory_order_release && producer == ncclMemProxyType::Generic &&
          consumer == ncclMemProxyType::Fabric)) {
      ncclMemFence(coop, cuda::memory_order_release, ncclMemProxyType::Generic, ncclMemProxyType::Fabric,
                   ncclMemFenceScope::Cta);
    }

    ncclCftLeId leId;
    size_t leOffset;
    if (this->useMultimem()) {
      this->mcInbox(&leId, &leOffset);
      cft.redMultimem(coop, leId, leOffset, ncclCftOpSum<uint32_t>{}, payload, ncclCftOpByteGran);
      cft.submit(coop);
      cft.flush(coop);
    } else {
      int nPeers = this->team.nRanks - 1;
      for (int i = 0; i < nPeers; i++) {
        int peer = i + (i < this->team.rank ? 0 : 1);
        this->ucInbox(peer, this->team.rank, &leId, &leOffset);
        cft.put(coop, leId, leOffset, payload, ncclCftOpByteGran);
      }
      if (nPeers > 0) {
        cft.submit(coop);
        cft.flush(coop);
      }
    }
  }
}

template <typename Coop>
template <bool EnableTimeout>
NCCL_DEVICE_INLINE ncclResult_t ncclCftBarrierSession_internal<Coop>::waitInternal(
  Coop, cuda::memory_order order, ncclMemProxyType producer, ncclMemProxyType consumer, uint64_t timeoutCycles) {
  using nccl::utility::testAbort;
  uint32_t steps = 0;
  uint64_t startCycle = 0;
  ncclResult_t ret = ncclSuccess;
  if NCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  }

  if (this->useMultimem()) {
    if (this->coop.thread_rank() == 0) {
      cuda::atomic_ref<uint32_t> inbox(*this->mcInbox());
      NVCC_PRAGMA_UNROLL_DISABLED
      while (true) {
        uint32_t got = inbox.load(cuda::memory_order_relaxed);
        if (got - (this->epoch + this->team.nRanks) <= uint32_t(-1) >> 1) break;
        if NCCL_IF_CONSTEXPR (EnableTimeout) {
          if (clock64() - startCycle >= timeoutCycles) {
            ret = ncclTimeout;
            goto exit;
          }
        } else {
          if (testAbort(this->comm.abortFlag, steps)) goto exit;
        }
      }
      this->epoch += this->team.nRanks;
    }
  } else {
    uint32_t expected = this->epoch + 1;
    for (int i = this->coop.thread_rank(); i < this->team.nRanks - 1; i += this->coop.size()) {
      int peer = i + (i < this->team.rank ? 0 : 1);
      cuda::atomic_ref<uint32_t> inbox(*this->ucInbox(peer));
      while (true) {
        uint32_t got = inbox.load(cuda::memory_order_relaxed);
        if (got - expected <= uint32_t(-1) >> 1) break;
        if NCCL_IF_CONSTEXPR (EnableTimeout) {
          if (clock64() - startCycle >= timeoutCycles) {
            ret = ncclTimeout;
            goto exit;
          }
        } else {
          if (testAbort(this->comm.abortFlag, steps)) goto exit;
        }
      }
    }
    this->epoch = expected;
  }
  goto exit;
exit:
  this->coop.sync();
  ncclMemFence(this->coop, nccl::utility::acquireOrderOf(order), producer, consumer, ncclMemFenceScope::Sys);
  return ret;
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclCftBarrierSession<Coop>::wait(Coop coop, cuda::memory_order order,
                                                          ncclMemProxyType producer, ncclMemProxyType consumer) {
  (void)this->template waitInternal</*EnableTimeout=*/false>(coop, order, producer, consumer, /*timeoutCycles=*/0ULL);
}

template <typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclCftBarrierSession<Coop>::wait(
  Coop coop, cuda::memory_order order, ncclMemProxyType producer, ncclMemProxyType consumer, uint64_t timeoutCycles) {
  return this->template waitInternal</*EnableTimeout=*/true>(coop, order, producer, consumer, timeoutCycles);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclCftBarrierSession<Coop>::sync(Coop coop, cuda::memory_order order,
                                                          ncclMemProxyType producer, ncclMemProxyType consumer) {
  this->arrive(coop, order, producer, consumer);
  this->wait(coop, order, producer, consumer);
}

template <typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclCftBarrierSession<Coop>::sync(
  Coop coop, cuda::memory_order order, ncclMemProxyType producer, ncclMemProxyType consumer, uint64_t timeoutCycles) {
  this->arrive(coop, order, producer, consumer);
  return this->wait(coop, order, producer, consumer, timeoutCycles);
}
#endif // __CUDACC__

#endif // _NCCL_DEVICE_CFT_BARRIER__FUNCS_H_
