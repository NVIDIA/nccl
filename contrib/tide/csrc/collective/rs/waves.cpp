// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "collective/rs/waves.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace tide {

RsWavePlan planWaves(int M, int L, int me, int waveNodes, int k) {
  if (M < 2 || L < 1 || L > 8 || me < 0 || me >= L || waveNodes < 1 || k < 1)
    throw std::invalid_argument("[tide] planWaves: bad arguments");
  RsWavePlan wp;
  wp.waveNodes = std::min(waveNodes, L);
  const int nW = (M + wp.waveNodes - 1) / wp.waveNodes;
  wp.inboxParities = nW > 1 ? 2 : 1;
  auto rampUp = [&](int r) {
    std::vector<int> v;
    for (int d = 1, left = r; left > 0; d *= 2) {
      int take = std::min({d, k, left});
      if (left - take < take) take = std::min(left, k);  // finish with the remainder rather than a smaller step
      v.push_back(take);
      left -= take;
    }
    return v;
  };
  for (int w = 0; w < nW; ++w) {
    RsWave wv;
    wv.first = w * wp.waveNodes + 1;
    wv.count = std::min(wp.waveNodes, M - w * wp.waveNodes);
    wv.parity = w % 2;
    const bool own = wv.first + wv.count - 1 == M;
    const int remote = wv.count - (own ? 1 : 0);
    std::vector<int> rounds;
    if (remote > 0) {
      if (w == 0) rounds = rampUp(remote);
      else if (w == nW - 1) {
        rounds = rampUp(remote);
        std::reverse(rounds.begin(), rounds.end());
      } else
        for (int left = remote; left > 0; left -= std::min(k, left)) rounds.push_back(std::min(k, left));
    }
    wv.firstRound = static_cast<int>(wp.kRounds.size());
    wv.rounds = static_cast<int>(rounds.size());
    wp.kRounds.insert(wp.kRounds.end(), rounds.begin(), rounds.end());
    for (int c = wv.first; c < wv.first + wv.count; ++c) {
      const uint8_t bit = static_cast<uint8_t>(1u << rsBlock(me, c, M, L));
      if (wv.live & bit)
        throw std::logic_error("[tide] planWaves: contributors " + std::to_string(c) + " and another of wave " +
                               std::to_string(w) + " share inbox block " + std::to_string(rsBlock(me, c, M, L)) +
                               " (M=" + std::to_string(M) + ", waveNodes=" + std::to_string(wp.waveNodes) + ")");
      wv.live |= bit;
    }
    wv.zero = w >= 2 ? static_cast<uint8_t>(wp.waves[w - 2].live & ~wv.live) : 0;
    wp.waves.push_back(wv);
  }
  // A parity's first wave inherits the buffer from its last wave of the previous call: blocks that
  // one fed and the other does not are zeroed at the call's end (registration zeroes the rest once).
  for (int p = 0; p < wp.inboxParities; ++p) {
    int last = -1;
    for (int w = nW - 1; w >= 0; --w)
      if (wp.waves[w].parity == p) {
        last = w;
        break;
      }
    if (last > p) wp.waves[p].zero = static_cast<uint8_t>(wp.waves[last].live & ~wp.waves[p].live);
  }
  return wp;
}

bool repeatWaves(RsWavePlan& wp, int parts) {
  if (parts <= 1) return true;
  for (const RsWave& wv : wp.waves)
    if (wv.rounds == 0) return false;
  const int nW = static_cast<int>(wp.waves.size()), nR = static_cast<int>(wp.kRounds.size());
  std::vector<RsWave> waves;
  std::vector<int> rounds;
  for (int q = 0; q < parts; ++q) {
    for (int w = 0; w < nW; ++w) {
      RsWave wv = wp.waves[w];
      wv.part = q;
      wv.parity = (q * nW + w) % 2;
      wv.firstRound += q * nR;
      waves.push_back(wv);
    }
    rounds.insert(rounds.end(), wp.kRounds.begin(), wp.kRounds.end());
  }
  const int G = static_cast<int>(waves.size());
  for (int g = 0; g < G; ++g) waves[g].zero = g >= 2 ? static_cast<uint8_t>(waves[g - 2].live & ~waves[g].live) : 0;
  for (int p = 0; p < 2; ++p) {
    int last = -1;
    for (int g = G - 1; g >= 0; --g)
      if (waves[g].parity == p) {
        last = g;
        break;
      }
    if (last > p) waves[p].zero = static_cast<uint8_t>(waves[last].live & ~waves[p].live);
  }
  wp.waves = std::move(waves);
  wp.kRounds = std::move(rounds);
  wp.inboxParities = 2;
  return true;
}

}  // namespace tide
