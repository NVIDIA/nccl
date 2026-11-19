// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "collective/rs/plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "runtime/comm.h"
#include "util/check.h"

namespace tide {

namespace {

size_t addChecked(size_t a, size_t b) {
  size_t r;
  TIDE_CHECK(!__builtin_add_overflow(a, b, &r), "reduce-scatter table overflows size_t");
  return r;
}
size_t alignUp(size_t x, size_t a) { return addChecked(x, a - 1) / a * a; }
size_t mulChecked(size_t a, size_t b) {
  size_t r;
  TIDE_CHECK(!__builtin_mul_overflow(a, b, &r), "reduce-scatter table overflows size_t");
  return r;
}

struct Layout {
  std::vector<RsSegment> segs;
  size_t outElems = 0, stgElems = 0, outBytes = 0;
  size_t chunkElems = 0, chunkBytes = 0, slotBytes = 0, stride = 0;  // stride: inbox/outbox block (bytes)
  int chunks = 1;
  std::vector<int> kRounds;
  int outboxSlots = 2;
  size_t subElems = 0;  // multi-node: sub-slice elems
  int parts = 1;
  size_t partElems = 0;  // multi-node: sub-slice elems per pass
  std::vector<RsWave> waves;
  int waveNodes = kRsMaxLocal, inboxParities = 1;
  int ownSpares = 0;
  int pipeline = 1, outboxBank = 0, ownWaves = 0;
  int railBlocks = 0;
};

// Outbox ring: destination j sums into slot j % O and may not wait for a put of its own round, and
// the previous round's puts should still be able to be in flight -> two consecutive rounds.
int outboxSlots(const std::vector<int>& rounds) {
  int o = 2, prev = 0;
  for (int d : rounds) {
    o = std::max(o, d + prev);
    prev = d;
  }
  return o;
}

// TIDE_RS_OUTBOX deepens the ring up to one block per remote destination of a call.
int outboxDepth(const Comm& comm, const std::vector<int>& rounds) {
  int dests = 0;
  for (int d : rounds) dests += d;
  return std::max(outboxSlots(rounds), std::min(comm.config().rsOutbox, dests));
}

// Rail rounds over D remote destinations: widths 1, 2, 4, .. up to k, then k; a remainder smaller
// than the round before it joins that round (the ramp the solver also counts).
std::vector<int> railRounds(int D, int k) {
  std::vector<int> v;
  for (int d = 1, left = D; left > 0; d = std::min(2 * d, k)) {
    int take = std::min({d, k, left});
    if (left - take < take) take = std::min(left, k);
    v.push_back(take);
    left -= take;
  }
  return v;
}

bool isRail(const Comm& comm) { return comm.nNodes() > 1 && comm.localRanks() == 1; }
// the sum kernel stages up to kRsMaxLocal blocks of a rail round
int railBlocksOf(const Comm& comm) { return std::min({comm.rsKway(), kRsMaxLocal, comm.nNodes() - 1}); }

// Waves of the multicast landing and the NIC rounds aligned to them (collective/rs/waves.h);
// TIDE_RS_WAVE_NODES per wave (default L; a test knob below L makes several waves out of a small fleet).
void buildWaves(Layout& lo, const Comm& comm) {
  RsWavePlan wp;
  try {
    wp = planWaves(comm.nNodes(), comm.localRanks(), comm.rank() % comm.localRanks(), comm.config().rsWaveNodes,
                   comm.rsKway());
  } catch (const std::exception& e) {
    TIDE_CHECK(false, e.what());
  }
  lo.parts = comm.rsParts() > 1 && repeatWaves(wp, comm.rsParts()) ? comm.rsParts() : 1;
  lo.waves = std::move(wp.waves);
  lo.kRounds = std::move(wp.kRounds);
  lo.waveNodes = wp.waveNodes;
  lo.inboxParities = wp.inboxParities;
  lo.outboxSlots = outboxDepth(comm, lo.kRounds);
  for (size_t w = 0; w < lo.waves.size(); ++w) {
    const bool own = lo.waves[w].first + lo.waves[w].count - 1 == comm.nNodes();
    lo.ownWaves += own;
    lo.ownSpares += own && w >= 2;
  }
  if (comm.rsPipeline() == 2) {  // two banks: one block per remote destination, or TIDE_RS_OUTBOX / tune_rs blocks each
    int dests = 0;
    for (int d : lo.kRounds) dests += d;
    lo.pipeline = 2;
    lo.outboxBank = comm.config().rsOutbox > 0 ? std::min(dests, lo.outboxSlots) : dests;
    lo.outboxSlots = 2 * lo.outboxBank;
    lo.ownSpares = 2 * lo.ownWaves;
  }
}

// Chunk count for a staging span of `stg` elements against the per-source chunk hint: the count
// nearest to stg/hint, so a chunk may grow up to 1.5x the hint. Every extra chunk is another
// chunk-round (another land/free barrier across the local team per destination), which the six-node
// numbers priced at ~1 ms of median for a 0.34% overflow of the hint (32 MiB, 16 -> 30 segments).
int chunkCount(size_t stg, size_t hintElems) {
  if (stg <= hintElems) return 1;
  return static_cast<int>(std::max<size_t>(1, (stg + hintElems / 2) / hintElems));
}

Layout layout(const Comm& comm, const std::vector<std::pair<size_t, size_t>>& shapes, int itemsize,
              size_t chunkBytesHint) {
  TIDE_CHECK(itemsize == 2 || itemsize == 4, "reduce-scatter input must be bf16 or fp32");
  TIDE_CHECK(!shapes.empty() && shapes.size() <= static_cast<size_t>(kRsMaxSegments), "1..256 segments");
  const size_t N = static_cast<size_t>(comm.nranks()), L = static_cast<size_t>(comm.localRanks());
  // The sum kernel holds one shared-memory stage per local source; a wider NVLink domain
  // (NVL16/NVL72) would overrun it.
  TIDE_CHECK(L >= 1 && L <= static_cast<size_t>(kRsMaxLocal),
             "reduce-scatter supports up to " + std::to_string(kRsMaxLocal) + " ranks per NVLink domain; this team has " +
                 std::to_string(L));
  const size_t M = static_cast<size_t>(comm.nNodes());
  Layout lo;
  size_t out = 0, stg = 0;
  for (const auto& [numel, dim0] : shapes) {
    TIDE_CHECK(dim0 > 0 && numel > 0 && numel % dim0 == 0, "segment shape must be (numel, dim0) with dim0 | numel");
    RsSegment s;
    s.numel = numel;
    s.dim0 = dim0;
    s.pieceElems = mulChecked((dim0 + N - 1) / N, numel / dim0);
    s.outOff = out;
    s.stgOff = stg;
    out = addChecked(out, s.pieceElems);
    stg = alignUp(addChecked(stg, s.pieceElems), kRsVec);
    lo.segs.push_back(s);
  }
  lo.outElems = out;
  lo.stgElems = stg;
  lo.outBytes = mulChecked(out, 4);
  const size_t hintElems = alignUp(std::max<size_t>(chunkBytesHint / itemsize, kRsVec), kRsVec);
  if (isRail(comm)) {  // no staging: blocks of my pieces for one destination, in the input dtype
    const size_t parts = static_cast<size_t>(comm.config().rsRailParts);
    lo.chunkElems = std::min(stg, alignUp((stg + parts - 1) / parts, kRsVec));
    lo.chunks = static_cast<int>((stg + lo.chunkElems - 1) / lo.chunkElems);
    lo.chunkBytes = mulChecked(lo.chunkElems, itemsize);
    lo.subElems = stg;
    lo.stride = alignUp(lo.chunkBytes, 256);
    lo.railBlocks = railBlocksOf(comm);
    lo.kRounds = railRounds(static_cast<int>(M) - 1, lo.railBlocks);
    lo.outboxSlots = outboxDepth(comm, lo.kRounds);
    lo.inboxParities = 2;
    return lo;
  }
  TIDE_CHECK(M == 1 || comm.props().multimem, "multi-node reduce-scatter needs NVLS multimem (the cross-node sum is a multicast landing)");
  if (M == 1) {  // the owner sums its whole destination: staging [source][chunk] over stg
    const int want = chunkCount(stg, hintElems);
    lo.chunkElems = std::min(stg, alignUp((stg + want - 1) / want, kRsVec));
    lo.chunks = static_cast<int>((stg + lo.chunkElems - 1) / lo.chunkElems);
    lo.chunkBytes = mulChecked(lo.chunkElems, itemsize);
    lo.slotBytes = alignUp(mulChecked(L, lo.chunkBytes), 256);
    return lo;
  }
  // Sub-slice p of every destination is one node round's work on one local rank; a round moves
  // L destinations x partElems per source through the slot in `chunks` chunk-rounds, once per part.
  // Every table runs the communicator's part count (a small one may have empty parts), so plans
  // sharing a stage area keep one parity count and one wave numbering.
  lo.subElems = alignUp((stg + L - 1) / L, kRsVec);
  buildWaves(lo, comm);
  lo.partElems = alignUp((lo.subElems + lo.parts - 1) / lo.parts, kRsVec);
  const int want = chunkCount(lo.partElems, hintElems);
  lo.chunkElems = std::min(lo.partElems, alignUp((lo.partElems + want - 1) / want, kRsVec));
  lo.chunks = static_cast<int>((lo.partElems + lo.chunkElems - 1) / lo.chunkElems);
  lo.chunkBytes = mulChecked(lo.chunkElems, itemsize);
  lo.slotBytes = alignUp(mulChecked(mulChecked(L, L), lo.chunkBytes), 256);  // [source][dest][chunk]
  lo.stride = alignUp(mulChecked(mulChecked(L, lo.partElems), 4), 256);       // [dest][partElems] fp32
  return lo;
}

std::vector<std::pair<size_t, size_t>> pieces(size_t begin, size_t bytes, size_t n) {
  std::vector<std::pair<size_t, size_t>> out;
  const size_t piece = alignUp(addChecked(bytes, n - 1) / n, 4096);
  for (size_t off = 0; off < bytes; off += piece) out.emplace_back(begin + off, std::min(piece, bytes - off));
  return out;
}

}  // namespace

RsAreaOffsets rsAreaOffsets(const Comm& comm, int outboxSlots, int inboxParities, int ownSpares, size_t slotBytes,
                            size_t stride) {
  const size_t M = static_cast<size_t>(comm.nNodes()), L = static_cast<size_t>(comm.localRanks());
  RsAreaOffsets o;
  size_t bytes = mulChecked(static_cast<size_t>(comm.config().rsStageSlots), slotBytes);
  if (M > 1) {
    const size_t perParity = isRail(comm) ? static_cast<size_t>(railBlocksOf(comm)) : L;
    o.inboxOff = bytes;
    bytes = addChecked(bytes, mulChecked(mulChecked(static_cast<size_t>(inboxParities), perParity), stride));
    o.outboxOff = bytes;
    bytes = addChecked(bytes, mulChecked(static_cast<size_t>(outboxSlots), stride));
    o.ownOff = bytes;
    bytes = addChecked(bytes, mulChecked(static_cast<size_t>(ownSpares), stride));
  }
  o.bytes = alignUp(bytes, 4096);
  return o;
}

size_t rsAreaBytes(const Comm& comm, const std::vector<std::pair<std::vector<std::pair<size_t, size_t>>, int>>& tables,
                   size_t chunkBytesHint) {
  TIDE_CHECK(!tables.empty(), "no tables");
  size_t slotBytes = 0, stride = 0;
  int outboxSlots = 2, parities = 1, ownSpares = 0;
  for (const auto& [shapes, itemsize] : tables) {
    const Layout lo = layout(comm, shapes, itemsize, chunkBytesHint);
    slotBytes = std::max(slotBytes, lo.slotBytes);
    stride = std::max(stride, lo.stride);
    outboxSlots = lo.outboxSlots;
    parities = lo.inboxParities;
    ownSpares = lo.ownSpares;
  }
  return rsAreaOffsets(comm, outboxSlots, parities, ownSpares, slotBytes, stride).bytes;
}

RsPlan::~RsPlan() {
  if (segsDev) cudaFree(segsDev);
}

size_t rsStageBytes(const Comm& comm, const std::vector<std::pair<size_t, size_t>>& shapes, int itemsize,
                    size_t chunkBytesHint) {
  return rsAreaBytes(comm, {{shapes, itemsize}}, chunkBytesHint);
}

std::shared_ptr<RsPlan> buildRsPlan(const Comm& comm, std::shared_ptr<Window> arena, size_t stageOff,
                                    const std::vector<std::pair<size_t, size_t>>& shapes, int itemsize,
                                    size_t chunkBytesHint) {
  TIDE_CHECK(arena && arena->base, "arena window is null");
  Layout lo = layout(comm, shapes, itemsize, chunkBytesHint);
  auto plan = std::make_shared<RsPlan>();
  plan->owner = &comm;
  plan->arena = arena;
  plan->itemsize = itemsize;
  plan->segs = std::move(lo.segs);
  plan->outElems = lo.outElems;
  plan->stgElems = lo.stgElems;
  plan->outBytes = lo.outBytes;
  const RsAreaOffsets off = rsAreaOffsets(comm, lo.outboxSlots, lo.inboxParities, lo.ownSpares, lo.slotBytes, lo.stride);
  plan->stageBase = stageOff;
  plan->stageBytes = off.bytes;
  TIDE_CHECK(stageOff % 256 == 0, "stage_off must be 256-byte aligned");
  TIDE_CHECK(addChecked(stageOff, off.bytes) <= arena->bytes, "reduce-scatter stage area exceeds the arena window");
  plan->chunkElems = lo.chunkElems;
  plan->chunkBytes = lo.chunkBytes;
  plan->slotBytes = lo.slotBytes;
  plan->outStride = lo.stride;
  plan->chunks = lo.chunks;
  plan->kRounds = lo.kRounds;
  plan->outboxSlots = lo.outboxSlots;
  plan->subElems = lo.subElems;
  plan->parts = lo.parts;
  plan->partElems = lo.partElems;
  plan->waves = lo.waves;
  plan->waveNodes = lo.waveNodes;
  plan->inboxParities = lo.inboxParities;
  plan->ownSpares = lo.ownSpares;
  plan->pipeline = lo.pipeline;
  plan->outboxBank = lo.outboxBank;
  plan->ownWaves = lo.ownWaves;
  plan->railBlocks = lo.railBlocks;
  if (comm.nNodes() > 1) {
    plan->inboxBase = stageOff + off.inboxOff;
    plan->outboxBase = stageOff + off.outboxOff;
    plan->ownBase = stageOff + off.ownOff;
    const size_t nicBytes = lo.railBlocks ? lo.chunkBytes  // my pieces for one destination
                                          : mulChecked(mulChecked(static_cast<size_t>(comm.localRanks()), lo.partElems), 4);  // one [dest][partElems] block
    plan->paced = comm.pacesShard(nicBytes);
    plan->nicChunks = pieces(0, nicBytes, static_cast<size_t>(comm.rmaCtx()));
    if (plan->paced) plan->paceTail = comm.paceTail(&plan->nicChunks, 1);
  }
  std::vector<RsSegDev> dev(plan->segs.size());
  for (size_t i = 0; i < dev.size(); ++i)
    dev[i] = {plan->segs[i].numel, plan->segs[i].pieceElems, plan->segs[i].outOff, plan->segs[i].stgOff};
  CUDACHECK(cudaMalloc(&plan->segsDev, dev.size() * sizeof(RsSegDev)));
  CUDACHECK(cudaMemcpy(plan->segsDev, dev.data(), dev.size() * sizeof(RsSegDev), cudaMemcpyHostToDevice));
  return plan;
}

}  // namespace tide
