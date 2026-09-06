/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_OPTIMIZATIONS_RING_EXCHANGE_KERNELS_CUH_
#define GIN_OPTIMIZATIONS_RING_EXCHANGE_KERNELS_CUH_

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"
#include <stddef.h>

#define CTA_COUNT 4
#define PUTS_PER_CTA 32
#define ELEMS_PER_PUT 256

/*
 * NCCL Device API GIN Ring Exchange Kernels
 *
 * Every rank sends to (rank + 1) % nRanks and receives from
 * (rank - 1 + nRanks) % nRanks.
 *
 * Compared dimensions:
 * 1. Producer model:
 *    - CTA-cooperative: every logical put is posted with ncclCoopCta.
 *    - Thread-per-put: every producer thread posts one ncclCoopThread put.
 * 2. Completion pattern:
 *    - Weak signals: every put increments a weak signal.
 *    - Weak AggregateRequests: every put increments a weak signal, while
 *      non-final puts hint to the backend that more requests are coming.
 */

__global__ void ctaCooperativePutKernel(ncclWindow_t sendWin, ncclWindow_t recvWin,
                                        int iterations, ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam world = ncclTeamWorld(devComm);

  const unsigned int ctaIndex = blockIdx.x;
  const int contextIndex = ctaIndex % devComm.ginContextCount;
  ncclGin context{devComm, contextIndex};

  const ncclGinSignal_t signalIndex = ctaIndex;
  const int dstRank = (devComm.rank + 1) % devComm.nRanks;
  const auto signalBase = context.readSignal(signalIndex);
  const size_t bytesPerPut = ELEMS_PER_PUT * sizeof(int);

  ncclGinBarrierSession<ncclCoopCta> bar{coop, context, ncclTeamTagWorld(), ctaIndex};
  bar.sync(coop, cuda::memory_order_acquire, ncclGinFenceLevel::None);

  for (int batch = 0; batch < iterations; ++batch) {
    for (int putIndex = 0; putIndex < PUTS_PER_CTA; ++putIndex) {
      const size_t offset = (ctaIndex * PUTS_PER_CTA + putIndex) * bytesPerPut;
      context.put(world, dstRank, recvWin, offset, sendWin, offset, bytesPerPut,
                  ncclGin_WeakSignalInc{signalIndex}, ncclGin_None{}, coop, ncclGin_None{},
                  cuda::thread_scope_thread, cuda::thread_scope_device,
                  ncclGinOptFlagsDefault);
    }

    context.waitSignal(coop, signalIndex, signalBase + (batch + 1) * PUTS_PER_CTA);
    if (coop.thread_rank() == 0) {
      context.flush(ncclCoopThread{});
    }
    coop.sync();
  }

  bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);
}

__global__ void ctaCooperativePutAggregateRequestsKernel(ncclWindow_t sendWin, ncclWindow_t recvWin,
                                                         int iterations, ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam world = ncclTeamWorld(devComm);

  const unsigned int ctaIndex = blockIdx.x;
  const int contextIndex = ctaIndex % devComm.ginContextCount;
  ncclGin context{devComm, contextIndex};

  const ncclGinSignal_t signalIndex = ctaIndex;
  const int dstRank = (devComm.rank + 1) % devComm.nRanks;
  const auto signalBase = context.readSignal(signalIndex);
  const size_t bytesPerPut = ELEMS_PER_PUT * sizeof(int);

  ncclGinBarrierSession<ncclCoopCta> bar{coop, context, ncclTeamTagWorld(), ctaIndex};
  bar.sync(coop, cuda::memory_order_acquire, ncclGinFenceLevel::None);

  for (int batch = 0; batch < iterations; ++batch) {
    for (int putIndex = 0; putIndex < PUTS_PER_CTA; ++putIndex) {
      const size_t offset = (ctaIndex * PUTS_PER_CTA + putIndex) * bytesPerPut;
      const uint32_t optFlags = putIndex == PUTS_PER_CTA - 1 ?
                                ncclGinOptFlagsDefault :
                                ncclGinOptFlagsAggregateRequests;
      context.put(world, dstRank, recvWin, offset, sendWin, offset, bytesPerPut,
                  ncclGin_WeakSignalInc{signalIndex}, ncclGin_None{}, coop, ncclGin_None{},
                  cuda::thread_scope_thread, cuda::thread_scope_device, optFlags);
    }

    context.waitSignal(coop, signalIndex, signalBase + (batch + 1) * PUTS_PER_CTA);
    if (coop.thread_rank() == 0) {
      context.flush(ncclCoopThread{});
    }
    coop.sync();
  }

  bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);
}

__global__ void threadPerPutKernel(ncclWindow_t sendWin, ncclWindow_t recvWin,
                                   int iterations, ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam world = ncclTeamWorld(devComm);

  const unsigned int ctaIndex = blockIdx.x;
  const int contextIndex = ctaIndex % devComm.ginContextCount;
  ncclGin context{devComm, contextIndex};

  const ncclGinSignal_t signalIndex = ctaIndex;
  const int dstRank = (devComm.rank + 1) % devComm.nRanks;
  const auto signalBase = context.readSignal(signalIndex);
  const int putIndex = threadIdx.x;
  const size_t bytesPerPut = ELEMS_PER_PUT * sizeof(int);
  const size_t offset = (ctaIndex * PUTS_PER_CTA + putIndex) * bytesPerPut;

  ncclGinBarrierSession<ncclCoopCta> bar{coop, context, ncclTeamTagWorld(), ctaIndex};
  bar.sync(coop, cuda::memory_order_acquire, ncclGinFenceLevel::None);

  for (int batch = 0; batch < iterations; ++batch) {
    context.put(world, dstRank, recvWin, offset, sendWin, offset, bytesPerPut,
                ncclGin_WeakSignalInc{signalIndex}, ncclGin_None{}, ncclCoopThread{});

    context.waitSignal(coop, signalIndex, signalBase + (batch + 1) * PUTS_PER_CTA);
    context.flush(coop);
  }

  bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);
}

__global__ void threadPerPutAggregateRequestsKernel(ncclWindow_t sendWin, ncclWindow_t recvWin,
                                                    int iterations, ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclTeam world = ncclTeamWorld(devComm);

  const unsigned int ctaIndex = blockIdx.x;
  const int contextIndex = ctaIndex % devComm.ginContextCount;
  ncclGin context{devComm, contextIndex};

  const ncclGinSignal_t signalIndex = ctaIndex;
  const int dstRank = (devComm.rank + 1) % devComm.nRanks;
  const auto signalBase = context.readSignal(signalIndex);
  const int putIndex = threadIdx.x;
  const size_t bytesPerPut = ELEMS_PER_PUT * sizeof(int);
  const size_t offset = (ctaIndex * PUTS_PER_CTA + putIndex) * bytesPerPut;

  ncclGinBarrierSession<ncclCoopCta> bar{coop, context, ncclTeamTagWorld(), ctaIndex};
  bar.sync(coop, cuda::memory_order_acquire, ncclGinFenceLevel::None);

  for (int batch = 0; batch < iterations; ++batch) {
    if (putIndex != 0) {
      context.put(world, dstRank, recvWin, offset, sendWin, offset, bytesPerPut,
                  ncclGin_WeakSignalInc{signalIndex}, ncclGin_None{}, ncclCoopThread{},
                  ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_device,
                  ncclGinOptFlagsAggregateRequests);
    }

    coop.sync();

    if (putIndex == 0) {
      context.put(world, dstRank, recvWin, offset, sendWin, offset, bytesPerPut,
                  ncclGin_WeakSignalInc{signalIndex}, ncclGin_None{}, ncclCoopThread{});
    }

    context.waitSignal(coop, signalIndex, signalBase + (batch + 1) * PUTS_PER_CTA);
    context.flush(coop);
  }

  bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);
}

#endif // GIN_OPTIMIZATIONS_RING_EXCHANGE_KERNELS_CUH_
