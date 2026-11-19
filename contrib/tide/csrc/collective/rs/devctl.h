// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// TIDE_RS_NET=device: the network phase of a pipelined reduce-scatter driven by GIN device kernels (one 1-CTA kernel
// per wave, W, and a final one, F) instead of host RMA calls and stream memops. Every table travels in the kernel
// parameters, so nothing outlives the launch.
#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstdint>

namespace tide {

constexpr int kDcMaxRounds = 8, kDcMaxK = 4, kDcMaxCtx = 4, kDcMaxChunks = 8, kDcMaxPeers = 8, kDcMaxAreas = 4;
constexpr uint64_t kDcFlushSliceNs = 60000000000ull;  // TIDE_RS_NET_DEBUG: a waiting flush refreshes its breadcrumb this often

// Mapped pinned host page. The first W / F that times out writes it and never returns (its completion event must not
// fire: landing, outbox reuse and the RS consumers wait on it); the monitor thread then exits the process.
struct DcError {
  uint32_t code;  // 0 ok; 1 credit wait, 2 data wait, 3 flush, 4 team waveDone wait
  int32_t epoch, wave, round, peer, ctx;
  uint64_t expected, seen;
  // TIDE_RS_NET_DEBUG=1: where the running wave kernel is (see dcWaveKernel), printed by the monitor every 5 s
  uint32_t where;
  int32_t bEpoch, bWave, bRound, bPeer, bCtx;
  uint64_t bLeast, bSeen, bWaitedNs, bTimeoutNs;
};

struct DcRound {
  int k;                                     // destinations this round (as many sources)
  int to[kDcMaxK], from[kDcMaxK];            // rail ranks = node indices
  uint64_t creditThr[kDcMaxK];               // credit count from to[i] to wait for (0: none, the area's first call)
  uint64_t srcOff[kDcMaxK], dstOff[kDcMaxK];  // window-relative block bases; chunks are relative to both
  uint64_t dataThr[kDcMaxK][kDcMaxCtx];      // strong-tail count from from[i] per context
  int nRel, relTo[kDcMaxK];                  // after the arrivals: credits to the next round's sources (same wave)
};

// TIDE_RS_NET_TRACE: %globaltimer stamps of one wave kernel; round[r] = credit granted, puts posted, data in, release
// sent. epoch is written last (a record is complete when it matches the call's).
constexpr int kDcTraceCalls = 64, kDcTraceWaves = 16;
struct DcWaveTrace {
  uint64_t start, doneSeen, creditsOut, end;
  uint64_t round[8][4];
  int32_t nRounds, epoch;
};

struct DcWaveArgs {
  ncclDevComm devComm;
  ncclWindow_t win;
  int me, ctx, nChunks;                      // chunk c goes on context c % ctx; its last chunk carries the strong signal
  uint64_t chunkOff[kDcMaxChunks], chunkBytes[kDcMaxChunks];
  int sigData0;                              // data signal of (source s, context x) = sigData0 + s * ctx + x
  int sigCredit0;                            // credit signal of target t = sigCredit0 + t (this area's segment)
  // Prologue (first round of a later wave): the team's waveDone of the parity this wave reuses (peers' slots in my
  // sync window; my own landing is a CUDA event the launching stream waited on), then the first round's credits.
  const uint64_t* doneBase;                  // null: no wait
  int nDone, donePeer[kDcMaxPeers];
  uint64_t doneSeq;
  int nPro, proTo[kDcMaxK];
  int nRounds;
  int round0, waveRounds;                    // partial bank (1 round/launch): round offset and total rounds in the wave
  DcRound r[kDcMaxRounds];
  DcError* err;
  int epoch, wave;
  uint64_t timeoutNs;                        // per signal / flag wait (wall clock)
  uint64_t flushCycles;                      // GIN's flush timeout counts SM cycles
  int debug;                                 // breadcrumbs in *err (TIDE_RS_NET_DEBUG)
  DcWaveTrace* trace;                        // null: no stamps
};
static_assert(kDcMaxRounds <= 8, "DcWaveTrace::round holds 8 rounds");

struct DcFinArgs {
  ncclDevComm devComm;
  int me, sigCredit0, nRel, relTo[kDcMaxK];  // the next call's first-round sources
};

void launchDcWave(const DcWaveArgs& a, cudaStream_t s);
void launchDcFin(const DcFinArgs& a, cudaStream_t s);

}  // namespace tide
