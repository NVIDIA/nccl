// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "collective/ag/plan.h"

#include <algorithm>
#include <utility>

#include "runtime/comm.h"
#include "util/check.h"

namespace tide {

namespace {

// The table comes across the Python boundary: every sum and product is checked.
size_t addChecked(size_t a, size_t b) {
  size_t r;
  TIDE_CHECK(!__builtin_add_overflow(a, b, &r), "byte table overflows size_t");
  return r;
}
size_t mulChecked(size_t a, size_t b) {
  size_t r;
  TIDE_CHECK(!__builtin_mul_overflow(a, b, &r), "byte table overflows size_t");
  return r;
}

// Cut [begin, begin+bytes) into n 4 KiB-aligned pieces (fewer when the range is small).
std::vector<std::pair<size_t, size_t>> pieces(size_t begin, size_t bytes, size_t n) {
  std::vector<std::pair<size_t, size_t>> out;
  const size_t piece = ((bytes + n - 1) / n + 4095) / 4096 * 4096;
  for (size_t off = 0; off < bytes; off += piece) out.emplace_back(begin + off, std::min(piece, bytes - off));
  return out;
}

}  // namespace

std::shared_ptr<AgPlan> buildAgPlan(const Comm& comm, std::shared_ptr<Window> arena, size_t regionOff,
                                    const std::vector<size_t>& sizes, const std::vector<size_t>& offsets,
                                    long stageOff, long ownOff) {
  TIDE_CHECK(arena && arena->base, "arena window is null");
  TIDE_CHECK(!sizes.empty() && sizes.size() == offsets.size(), "sizes/offsets length mismatch");
  const size_t N = static_cast<size_t>(comm.nranks());
  const size_t r = static_cast<size_t>(comm.rank());

  auto plan = std::make_shared<AgPlan>();
  plan->owner = &comm;
  plan->arena = arena;
  plan->regionOff = regionOff;
  plan->sizes = sizes;
  plan->offsets = offsets;

  size_t expect = 0;
  for (size_t p = 0; p < sizes.size(); ++p) {
    TIDE_CHECK(offsets[p] == expect, "offsets must be the prefix sum of sizes (shard is segment-ordered)");
    TIDE_CHECK(sizes[p] > 0, "zero-length segment");
    expect = addChecked(expect, sizes[p]);
  }
  const size_t S = expect;
  plan->shardBytes = S;
  plan->regionBytes = mulChecked(S, N);
  TIDE_CHECK(addChecked(regionOff, plan->regionBytes) <= arena->bytes, "region exceeds arena window");

  char* mine = static_cast<char*>(arena->base);
  const size_t M = static_cast<size_t>(comm.nNodes()), L = static_cast<size_t>(comm.localRanks());
  if (M > 1 && L == 1) {
    plan->direct = true;
    plan->paced = comm.pacesShard(S);
    plan->staged = ownOff >= 0;
    if (plan->staged) {
      plan->ownOff = static_cast<size_t>(ownOff);
      TIDE_CHECK(addChecked(plan->ownOff, S) <= arena->bytes, "own shard exceeds arena window");
    }
    auto landing = [&](size_t p, size_t rank) { return regionOff + offsets[p] * N + rank * sizes[p]; };
    for (size_t p = 0; p < sizes.size(); ++p) {
      plan->ownPieces.push_back(plan->staged ? mine + plan->ownOff + offsets[p] : mine + landing(p, r));
      if (plan->staged) plan->copies.add(mine + landing(p, r), plan->ownPieces[p], sizes[p]);
    }
    auto sends = [&](size_t q, size_t cnt) {  // the puts of rank q: ranks q .. q + cnt - 1, one run per wrap
      std::vector<AgPlan::Put> puts;
      for (size_t p = 0; p < sizes.size(); ++p)
        for (size_t i = 0; i < cnt;) {
          const size_t first = (q + i) % M, run = std::min(cnt - i, M - first);
          AgPlan::Put put;
          put.srcOff = put.dstOff = landing(p, first);
          put.chunks = pieces(0, run * sizes[p], comm.rmaCtx());
          puts.push_back(std::move(put));
          i += run;
        }
      return puts;
    };
    for (size_t step = 1; step < M; step *= 2) {
      AgPlan::Round rd;
      rd.recvSlot = step;
      rd.cnt = std::min(step, M - step);
      rd.to = static_cast<int>((r + M - step) % M);
      rd.from = static_cast<int>((r + step) % M);
      rd.groupRanges.emplace_back(0, rd.cnt);
      rd.groupPuts.push_back(sends(r, rd.cnt));
      rd.recvPuts = sends(static_cast<size_t>(rd.from), rd.cnt);
      rd.scatter.emplace_back();
      plan->rounds.push_back(std::move(rd));
    }
    return plan;
  }

  // Every landing write is a copy to the NVLS multimem address of its landing slot: the switch
  // replicates it into every LSA rank's window (self included). The source is the packed shard at
  // ownOff when given, else the in-place own slot (the self-delivered copy then rewrites the same
  // bytes it reads).
  TIDE_CHECK(arena->multimem != nullptr, "the all-gather needs an NVLS-bound arena window (multimem)");
  char* mc = static_cast<char*>(arena->multimem);
  if (M > 1) {
    TIDE_CHECK(stageOff >= 0, "multi-node all-gather needs a receive stage area (stage_off)");
    if (ownOff < 0) {  // own shard = the first of M slots at stageOff
      ownOff = stageOff;
      stageOff += static_cast<long>(S);
      plan->ownInStage = true;
    }
    plan->recvBase = static_cast<size_t>(stageOff);
    TIDE_CHECK(addChecked(plan->recvBase, mulChecked(M - 1, S)) <= arena->bytes, "receive stage area exceeds arena window");
  }
  plan->staged = ownOff >= 0;
  if (plan->staged) {
    plan->ownOff = static_cast<size_t>(ownOff);
    TIDE_CHECK(addChecked(plan->ownOff, S) <= arena->bytes, "own shard exceeds arena window");
  }
  for (size_t p = 0; p < sizes.size(); ++p)
    plan->ownPieces.push_back(plan->staged ? mine + plan->ownOff + offsets[p] : mine + regionOff + offsets[p] * N + r * sizes[p]);

  if (M == 1) {
    for (size_t p = 0; p < sizes.size(); ++p)
      plan->copies.add(mc + regionOff + offsets[p] * N + r * sizes[p], plan->ownPieces[p], sizes[p]);
    return plan;
  }

  // Rail-aligned hierarchical form: the NIC delivers packed shards into the receive slots, the
  // copy engines multicast every slot (own shard included) into the param-major landing.
  plan->paced = comm.pacesShard(S);
  const size_t local = r % L, node = static_cast<size_t>(comm.node());
  auto slotSrc = [&](size_t i) { return i == 0 ? plan->ownOff : plan->recvBase + (i - 1) * S; };  // window-relative
  // Scatter of slots [slot0, slot0 + slots) -- consecutive in memory: the own shard alone, or receive
  // slots -- whose shards belong to nodes node0, node0 + 1, .. (mod M): rows over consecutive
  // rail-peer ranks, one run per wrap of the node index.
  auto scatterSlots = [&](PushBatch& batch, size_t slot0, size_t slots, size_t node0) {
    for (size_t i = 0; i < slots;) {
      const size_t run = std::min(slots - i, M - (node0 + i) % M);
      const size_t srcRank = ((node0 + i) % M) * L + local;
      for (size_t p = 0; p < sizes.size(); ++p) {
        const size_t landing = regionOff + offsets[p] * N + srcRank * sizes[p];
        batch.addRows(mc + landing, L * sizes[p], mine + slotSrc(slot0 + i) + offsets[p], S, sizes[p], run);
      }
      i += run;
    }
  };
  scatterSlots(plan->ownScatter, 0, 1, node);
  for (size_t step = 1; step < M; step *= 2) {
    AgPlan::Round rd;
    rd.recvSlot = step;
    rd.cnt = std::min(step, M - step);
    rd.to = comm.railPeer(static_cast<int>((node + M - step) % M));
    rd.from = comm.railPeer(static_cast<int>((node + step) % M));
    const bool last = step * 2 >= M;
    const size_t G = last ? std::min(rd.cnt, static_cast<size_t>(comm.nicGroups())) : 1;
    std::vector<std::vector<std::pair<size_t, size_t>>> chunkLists;  // every put of the round, for the pace tail
    for (size_t g = 0; g < G; ++g) {  // whole slots per group; the remainder spread over the first groups
      const size_t slot0 = g * (rd.cnt / G) + std::min(g, rd.cnt % G), slots = rd.cnt / G + (g < rd.cnt % G ? 1 : 0);
      rd.groupRanges.emplace_back(slot0, slots);
      std::vector<AgPlan::Put> puts;
      // the block [slot0, slot0 + slots) lands in `to`'s slots [recvSlot + slot0, ..): the own shard and
      // the receive slots are separate runs in memory, one put each
      for (size_t i = slot0; i < slot0 + slots;) {
        const size_t run = i == 0 ? 1 : slot0 + slots - i;
        AgPlan::Put put;
        put.srcOff = slotSrc(i);
        put.dstOff = plan->recvBase + (rd.recvSlot + i - 1) * S;
        put.chunks = pieces(0, run * S, comm.rmaCtx());
        chunkLists.push_back(put.chunks);
        puts.push_back(std::move(put));
        i += run;
      }
      rd.groupPuts.push_back(std::move(puts));
      rd.scatter.emplace_back();  // my slots [recvSlot + slot0, ..) are receive slots: one run
      scatterSlots(rd.scatter.back(), rd.recvSlot + slot0, slots, node + rd.recvSlot + slot0);
    }
    if (plan->paced) rd.paceTail = comm.paceTail(chunkLists.data(), static_cast<int>(chunkLists.size()));
    plan->rounds.push_back(std::move(rd));
  }
  return plan;
}

}  // namespace tide
