// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Exhaustive host-only check of the multicast landing's wave schedule (collective/rs/waves.h),
// no GPU needed:
//   g++ -std=c++17 -O2 -Icsrc test/test_waves.cpp csrc/collective/rs/waves.cpp -o /tmp/test_waves && /tmp/test_waves
// For every node count, local rank, wave size and round width: the waves partition the
// contributors in order with my node last; the rounds partition the remote ones, at most k each,
// never straddling a wave; no two contributors of a wave share an inbox block; the destination a
// sender puts in a round lands in the wave the receiver expects it in; and, replaying three calls,
// every block a landing reads holds only this wave's partials (the zeroing removed the rest); the same
// for the schedule of 2-4 passes (repeatWaves).
#include <cstdio>
#include <stdexcept>
#include <string>

#include "collective/rs/waves.h"

using tide::planWaves;
using tide::rsBlock;
using tide::RsWavePlan;

static int failures = 0;
#define EXPECT(cond, ...)                         \
  do {                                            \
    if (!(cond)) {                                \
      if (++failures <= 20) {                     \
        std::fprintf(stderr, "FAIL %s: ", #cond); \
        std::fprintf(stderr, __VA_ARGS__);        \
        std::fprintf(stderr, "\n");               \
      }                                           \
    }                                             \
  } while (0)

static void check(int M, int L, int me, int W, int k, const RsWavePlan& wp) {
  const int nW = static_cast<int>(wp.waves.size());
  // waves: contributors 1..M in order, at most W each, my node (M) last
  int next = 1;
  for (int w = 0; w < nW; ++w) {
    const auto& wv = wp.waves[w];
    EXPECT(wv.first == next && wv.count >= 1 && wv.count <= W, "M=%d W=%d wave %d", M, W, w);
    next += wv.count;
    EXPECT(wv.parity == w % 2, "M=%d wave %d parity", M, w);
    EXPECT(__builtin_popcount(wv.live) == wv.count, "M=%d W=%d me=%d wave %d: shared block", M, W, me, w);
    EXPECT((wv.zero & wv.live) == 0, "M=%d wave %d zeroes a block it feeds", M, w);
  }
  EXPECT(next == M + 1, "M=%d W=%d: waves cover %d contributors", M, W, next - 1);
  // rounds: the remote contributors in order, at most k each, aligned to the waves
  int sum = 0, r0 = 0;
  for (int x : wp.kRounds) {
    EXPECT(x >= 1 && x <= k, "M=%d k=%d round of %d", M, k, x);
    sum += x;
  }
  EXPECT(sum == M - 1, "M=%d: rounds carry %d destinations", M, sum);
  for (int w = 0; w < nW; ++w) {
    const auto& wv = wp.waves[w];
    const int remote = wv.count - (wv.first + wv.count - 1 == M ? 1 : 0);
    EXPECT(wv.firstRound == r0, "M=%d wave %d firstRound", M, w);
    int carried = 0;
    for (int r = wv.firstRound; r < wv.firstRound + wv.rounds; ++r) carried += wp.kRounds[r];
    EXPECT(carried == remote, "M=%d W=%d k=%d wave %d: rounds carry %d of %d", M, W, k, w, carried, remote);
    r0 += wv.rounds;
  }
  // sender view: destination j (1..M-1) goes out in round R(j), whose wave must be the receiver's
  // wave of contributor j (same M, W, k on every node, so both sides read the same plan)
  int j = 1;
  for (int w = 0; w < nW; ++w)
    for (int r = wp.waves[w].firstRound; r < wp.waves[w].firstRound + wp.waves[w].rounds; ++r)
      for (int d = 0; d < wp.kRounds[r]; ++d, ++j) {
        const auto& wv = wp.waves[w];
        EXPECT(j >= wv.first && j < wv.first + wv.count, "M=%d: destination %d sent in wave %d", M, j, w);
        EXPECT(wv.live >> rsBlock(me, j, M, L) & 1, "M=%d: block of contributor %d not live", M, j);
      }
  // replay three calls on one rank: S[parity] = blocks holding data
  unsigned S[2] = {0, 0};
  for (int call = 0; call < 3; ++call)
    for (int w = 0; w < nW; ++w) {
      const auto& wv = wp.waves[w];
      unsigned& s = S[wv.parity];
      if (w >= 2 || call > 0) s &= ~static_cast<unsigned>(wv.zero);  // w < 2: zeroed at the previous call's end
      EXPECT((s & ~static_cast<unsigned>(wv.live)) == 0, "M=%d W=%d me=%d call %d wave %d: stale blocks 0x%x", M, W, me,
             call, w, s & ~static_cast<unsigned>(wv.live));
      s |= wv.live;
    }
}

// repeatWaves: each pass is the single-pass schedule (same contributors, blocks and rounds, rounds
// offset by the pass), parities alternate along the whole sequence, and the replay over three calls
// finds no stale block in any landing; refused exactly when a pass has a wave of my node alone.
static void checkParts(int M, int me, int W, int parts, const RsWavePlan& base) {
  RsWavePlan wp = base;
  bool ownOnly = false;
  for (const auto& wv : base.waves) ownOnly |= wv.rounds == 0;
  const bool ok = tide::repeatWaves(wp, parts);
  EXPECT(ok == !ownOnly, "M=%d W=%d parts=%d: repeatWaves %s", M, W, parts, ok ? "accepted" : "refused");
  if (!ok) return;
  const int nW = static_cast<int>(base.waves.size()), nR = static_cast<int>(base.kRounds.size());
  EXPECT(static_cast<int>(wp.waves.size()) == parts * nW && static_cast<int>(wp.kRounds.size()) == parts * nR,
         "M=%d parts=%d: sizes", M, parts);
  EXPECT(wp.inboxParities == 2, "M=%d parts=%d: parities", M, parts);
  for (int g = 0; g < static_cast<int>(wp.waves.size()); ++g) {
    const auto &wv = wp.waves[g], &b = base.waves[g % nW];
    EXPECT(wv.part == g / nW && wv.parity == g % 2 && wv.first == b.first && wv.count == b.count && wv.live == b.live &&
               wv.rounds == b.rounds && wv.firstRound == b.firstRound + (g / nW) * nR,
           "M=%d W=%d parts=%d wave %d", M, W, parts, g);
    EXPECT((wv.zero & wv.live) == 0, "M=%d parts=%d wave %d zeroes a block it feeds", M, parts, g);
  }
  for (int r = 0; r < static_cast<int>(wp.kRounds.size()); ++r)
    EXPECT(wp.kRounds[r] == base.kRounds[r % nR], "M=%d parts=%d round %d", M, parts, r);
  unsigned S[2] = {0, 0};
  for (int call = 0; call < 3; ++call)
    for (int g = 0; g < static_cast<int>(wp.waves.size()); ++g) {
      const auto& wv = wp.waves[g];
      unsigned& s = S[wv.parity];
      if (g >= 2 || call > 0) s &= ~static_cast<unsigned>(wv.zero);
      EXPECT((s & ~static_cast<unsigned>(wv.live)) == 0, "M=%d W=%d me=%d parts=%d call %d wave %d: stale blocks 0x%x", M, W,
             me, parts, call, g, s & ~static_cast<unsigned>(wv.live));
      s |= wv.live;
    }
}

int main() {
  const int L = 8;
  long plans = 0, refused = 0;
  std::string refusedAt;
  for (int M = 2; M <= 1024; ++M)
    for (int W = 1; W <= L; ++W)
      for (int k = 1; k <= L; ++k)
        for (int me = 0; me < L; ++me) {
          try {
            const RsWavePlan wp = planWaves(M, L, me, W, k);
            check(M, L, me, W, k, wp);
            for (int parts : {2, 3, 4}) checkParts(M, me, W, parts, wp);
            ++plans;
          } catch (const std::logic_error&) {
            // a wave would put my node next to a multiple of L: allowed only for test wave sizes that
            // do not divide L, never for the default W = L
            EXPECT(W != L && L % W != 0, "M=%d W=%d refused", M, W);
            if (refused++ == 0) refusedAt = "M=" + std::to_string(M) + " W=" + std::to_string(W);
          }
        }
  std::printf("%ld plans checked, %ld refused (first: %s), %d failures\n", plans, refused, refusedAt.c_str(), failures);
  std::printf(failures ? "FAIL\n" : "PASS\n");
  return failures ? 1 : 0;
}
