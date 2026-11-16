// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Wave schedule of the multicast landing: host-only, no CUDA (test/test_waves.cpp checks it
// exhaustively on any machine).

#include <cstdint>
#include <vector>

namespace tide {

// Inbox block that contributor c's partial occupies on local rank `me` of the receiving node --
// and, the same number, the sub-slice local rank `me` of the sending node computes for it.
// Contributors of a receiver m are the nodes in send order, c = (m - n) mod M for sender n (1..M-1),
// and c = M (or 0) for m itself. The one formula both sides use.
inline int rsBlock(int me, int c, int M, int L) { return (me + c % M) % L; }

struct RsWave {
  int first = 0, count = 0;        // contributor indices [first, first + count); index M = my own node
  int parity = 0;
  int firstRound = 0, rounds = 0;  // NIC rounds carrying its remote contributors (0 rounds: own only)
  uint8_t live = 0;                // my blocks p fed in this wave (bit p)
  uint8_t zero = 0;                // my blocks to zero before this wave lands: fed by the parity's previous wave
                                   // (w >= 2: in this call; w < 2: the previous call's last wave of the parity), not by this one
  int part = 0;                    // pass over the sub-slices it carries (repeatWaves)
};

struct RsWavePlan {
  std::vector<RsWave> waves;
  std::vector<int> kRounds;        // remote destinations per NIC round, in send order
  int waveNodes = 0, inboxParities = 1;
};

// Waves of waveNodes contributors (<= L) filled in send order, my own node last; NIC rounds of at
// most k aligned to them (the first wave ramps up 1, 2, .., the last ramps down, the ones between
// are rounds of k). Throws when a wave's contributors would share a block: rsBlock is (me + c) mod L
// and my own node is c = 0, so a wave may not hold my node and a multiple of L -- filling in order
// never does (the last wave's remote contributors sit strictly between two multiples of L), at the
// price of a wave of my node alone when M = 1 mod waveNodes.
RsWavePlan planWaves(int M, int L, int me, int waveNodes, int k);

// The schedule of `parts` passes, each carrying one part of every sub-slice, as one wave sequence:
// parities alternate along it (two from parts = 2 on) and each wave's zero mask is against the
// previous wave of its parity. Returns false, leaving wp alone, when a pass has a wave of my node
// alone: its landing would come after the next pass's first rounds, which wait for it.
bool repeatWaves(RsWavePlan& wp, int parts);

}  // namespace tide
