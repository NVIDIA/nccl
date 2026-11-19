// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "collective/rs/devctl.h"

#include "util/check.h"

namespace tide {
namespace {

static_assert(sizeof(DcWaveArgs) <= 4000, "wave tables must fit the kernel parameter space");

__device__ __forceinline__ uint64_t loadAcquireSys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

// Wall-clock ns: GIN's own timeouts count SM cycles, which stretch when the clock drops while a kernel only polls.
__device__ __forceinline__ uint64_t nowNs() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

// Never returns (nothing clears the code; the monitor exits the process): the kernel's completion event must not
// fire after a protocol error. The spin must read memory: a side-effect-free infinite loop is UB, and nvcc then
// deletes every call site.
__device__ __noinline__ void dcFail(DcError* e, uint32_t code, int epoch, int wave, int round, int peer, int ctx,
                                    uint64_t expected, uint64_t seen) {
  e->epoch = epoch;
  e->wave = wave;
  e->round = round;
  e->peer = peer;
  e->ctx = ctx;
  e->expected = expected;
  e->seen = seen;
  __threadfence_system();
  volatile uint32_t* posted = &e->code;
  *posted = code;
  __threadfence_system();
  while (*posted != 0) __nanosleep(1000000);
}

// where: 1 team waveDone spin, 2 first-round credits out, 3 credit wait, 4 puts, 5 data wait, 6 credits out, 7 flush, 8 done
__device__ __forceinline__ void crumb(const DcWaveArgs& a, uint32_t where, int round, int peer, int ctx, uint64_t least,
                                      uint64_t seen, uint64_t waitedNs = 0) {
  if (!a.debug) return;
  volatile DcError* e = a.err;
  e->bWaitedNs = waitedNs;
  e->bTimeoutNs = a.timeoutNs;
  e->bEpoch = a.epoch;
  e->bWave = a.wave;
  e->bRound = round;
  e->bPeer = peer;
  e->bCtx = ctx;
  e->bLeast = least;
  e->bSeen = seen;
  e->where = where;
  __threadfence_system();
}

__device__ void waitSig(const DcWaveArgs& a, int ctx, ncclGinSignal_t sig, uint64_t least, uint32_t code, int round, int peer) {
  ncclGin gin{a.devComm, ctx};
  const uint64_t t0 = nowNs();
  uint64_t v;
  uint32_t n = 0;
  while ((v = gin.readSignal(sig, 64, cuda::memory_order_acquire)) < least) {
    if (nowNs() - t0 > a.timeoutNs) dcFail(a.err, code, a.epoch, a.wave, round, peer, ctx, least, v);
    if ((++n & 4095) == 1) crumb(a, code == 1 ? 3 : 5, round, peer, ctx, least, v, nowNs() - t0);
    __nanosleep(100);
  }
}

__global__ void dcWaveKernel(DcWaveArgs a) {
  if (threadIdx.x != 0) return;
  ncclTeam rail = ncclTeamRail(a.devComm);
  const int active = a.nChunks < a.ctx ? a.nChunks : a.ctx;
  DcWaveTrace* const tr = a.trace;
  if (tr && a.round0 == 0) tr->start = nowNs();
  if (a.doneBase) {
    for (int i = 0; i < a.nDone; ++i) {
      const uint64_t* p = a.doneBase + a.donePeer[i];
      const uint64_t t0 = nowNs();
      uint64_t v;
      crumb(a, 1, -1, a.donePeer[i], -1, a.doneSeq, 0);
      while ((v = loadAcquireSys(p)) < a.doneSeq) {
        if (nowNs() - t0 > a.timeoutNs) dcFail(a.err, 4, a.epoch, a.wave, -1, a.donePeer[i], -1, a.doneSeq, v);
        __nanosleep(200);
      }
    }
  }
  if (tr && a.round0 == 0) tr->doneSeen = nowNs();
  {
    crumb(a, 2, -1, -1, 0, 0, 0);
    ncclGin gin{a.devComm, 0};
    for (int i = 0; i < a.nPro; ++i)
      gin.signal(rail, a.proTo[i], ncclGin_SignalInc{static_cast<ncclGinSignal_t>(a.sigCredit0 + a.me)});
  }
  if (tr && a.round0 == 0) tr->creditsOut = nowNs();
  for (int ri = 0; ri < a.nRounds; ++ri) {
    const int rIdx = a.round0 + ri;
    const DcRound& R = a.r[ri];
    for (int i = 0; i < R.k; ++i)
      if (R.creditThr[i])
        waitSig(a, 0, static_cast<ncclGinSignal_t>(a.sigCredit0 + R.to[i]), R.creditThr[i], 1, rIdx, R.to[i]);
    if (tr) tr->round[rIdx][0] = nowNs();
    crumb(a, 4, rIdx, -1, -1, 0, 0);
    for (int i = 0; i < R.k; ++i)
      for (int c = 0; c < a.nChunks; ++c) {
        const int x = c % a.ctx;
        ncclGin gin{a.devComm, x};
        const size_t dst = R.dstOff[i] + a.chunkOff[c], src = R.srcOff[i] + a.chunkOff[c];
        if (c + a.ctx >= a.nChunks)  // the last put of context x to this peer this round
          gin.put(rail, R.to[i], a.win, dst, a.win, src, a.chunkBytes[c],
                  ncclGin_StrongSignalInc{static_cast<ncclGinSignal_t>(a.sigData0 + a.me * a.ctx + x)});
        else
          gin.put(rail, R.to[i], a.win, dst, a.win, src, a.chunkBytes[c]);
      }
    if (tr) tr->round[rIdx][1] = nowNs();
    for (int i = 0; i < R.k; ++i)
      for (int x = 0; x < active; ++x)
        waitSig(a, x, static_cast<ncclGinSignal_t>(a.sigData0 + R.from[i] * a.ctx + x), R.dataThr[i][x], 2, rIdx, R.from[i]);
    if (tr) tr->round[rIdx][2] = nowNs();
    if (R.nRel) {
      crumb(a, 6, rIdx, -1, 0, 0, 0);
      ncclGin gin{a.devComm, 0};
      for (int i = 0; i < R.nRel; ++i)
        gin.signal(rail, R.relTo[i], ncclGin_SignalInc{static_cast<ncclGinSignal_t>(a.sigCredit0 + a.me)});
    }
    if (tr) tr->round[rIdx][3] = nowNs();
  }
  for (int x = 0; x < active; ++x) {  // the wave's outbox blocks may be reused once its event fires
    crumb(a, 7, -1, -1, x, 0, 0);
    ncclGin gin{a.devComm, x};
    // with debug, in slices of about kDcFlushSliceNs so the breadcrumb shows how long a stuck flush has waited
    const uint64_t slice =  // GIN's flush timeout counts ~2 GHz cycles
      a.debug && kDcFlushSliceNs * 2 < a.flushCycles ? kDcFlushSliceNs * 2 : a.flushCycles;
    uint64_t spent = 0;
    const uint64_t t0 = nowNs();
    while (gin.flush(ncclCoopThread{}, cuda::memory_order_acquire, ncclGin_None{}, slice) != ncclSuccess) {
      spent += slice;
      crumb(a, 7, -1, -1, x, 0, 0, nowNs() - t0);
      if (spent >= a.flushCycles) dcFail(a.err, 3, a.epoch, a.wave, -1, -1, x, 0, 0);
    }
  }
  crumb(a, 8, -1, -1, -1, 0, 0);
  const bool lastLaunch = a.round0 + a.nRounds >= a.waveRounds;
  if (tr && lastLaunch) {
    tr->end = nowNs();
    tr->nRounds = a.waveRounds;
  }
  __threadfence_system();
  if (tr && lastLaunch) *reinterpret_cast<volatile int32_t*>(&tr->epoch) = a.epoch;
}

__global__ void dcFinKernel(DcFinArgs a) {
  if (threadIdx.x != 0) return;
  ncclTeam rail = ncclTeamRail(a.devComm);
  ncclGin gin{a.devComm, 0};
  for (int i = 0; i < a.nRel; ++i)
    gin.signal(rail, a.relTo[i], ncclGin_SignalInc{static_cast<ncclGinSignal_t>(a.sigCredit0 + a.me)});
}

}  // namespace

void launchDcWave(const DcWaveArgs& a, cudaStream_t s) {
  dcWaveKernel<<<1, 32, 0, s>>>(a);
  CUDACHECK(cudaGetLastError());
}

void launchDcFin(const DcFinArgs& a, cudaStream_t s) {
  dcFinKernel<<<1, 32, 0, s>>>(a);
  CUDACHECK(cudaGetLastError());
}

}  // namespace tide
