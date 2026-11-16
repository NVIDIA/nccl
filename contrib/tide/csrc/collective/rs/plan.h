// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "collective/rs/kernels.h"
#include "collective/rs/waves.h"

namespace tide {

class Comm;
struct Window;

// Reduce-scatter over a list of gradients read in place (FSDP2's unsharded grads): gradient p has
// `numel` elements in `dim0` rows; rank q's piece is rows [q*rows, (q+1)*rows) with rows =
// ceil(dim0/nranks), i.e. elements [q*pieceElems, (q+1)*pieceElems) of the flat tensor, the part
// beyond `numel` being padding (zero). The fp32 output is the concatenation of the pieces owned by
// this rank, segment-ordered: exactly torch._chunk_cat + reduce_scatter_tensor.
struct RsSegment {
  size_t numel = 0, dim0 = 0, pieceElems = 0;
  size_t outOff = 0;  // element offset in the packed output (prefix sum of pieceElems)
  size_t stgOff = 0;  // element offset in the packed staging (prefix sum aligned to 8 elements)
};

struct RsPlan {
  const Comm* owner = nullptr;
  std::shared_ptr<Window> arena;
  int itemsize = 2;  // input dtype: 2 = bf16, 4 = fp32
  std::vector<RsSegment> segs;
  size_t outElems = 0, stgElems = 0, outBytes = 0;

  // Stage area (window-relative bytes), same offsets on every rank:
  //   [local staging: TIDE_RS_STAGE_SLOTS (2) slots x slotBytes] [inbox: inboxParities x L blocks] [outbox: outboxSlots blocks]
  //   [own spares: ownSpares blocks]
  // (inbox/outbox/spares on multi-node only; blocks of outStride bytes). A staging slot holds one chunk of
  // every local source's pieces for the destinations of one node round.
  size_t stageBase = 0, stageBytes = 0;
  size_t chunkElems = 0, chunkBytes = 0, slotBytes = 0;  // staging chunk granularity
  int chunks = 1;                                        // chunk-rounds per node round
  // NIC rounds over the M-1 remote destinations in send order: destinations per round (TIDE_RS_KWAY
  // = k, ramped 1, 2, .. within the first wave and down within the last; see buildWaves).
  std::vector<int> kRounds;
  int outboxSlots = 2;
  size_t inboxBase = 0, outboxBase = 0, outStride = 0;  // adopted from the stage area at registration
  // Own-node partials of the waves that reuse a parity within the call (w >= 2): summed into a spare block each and
  // copied into the inbox once the parity's previous wave has landed, so no sum waits for a landing.
  int ownSpares = 0;
  size_t ownBase = 0;  // adopted with inboxBase
  // TIDE_RS_PIPELINE=2 (multicast): two banks of outboxBank blocks (read_early: one per remote destination of a call;
  // throughput mode: two consecutive NIC rounds) and of ownWaves spares (one per wave holding my own node, from wave 0
  // on), so the read phase of the next call can run while this call's network phase still reads its bank
  // (Comm::issueRsMcPipelined).
  int pipeline = 1;
  int outboxBank = 0, ownWaves = 0;
  // Multi-node (NVLS multimem): the output is cut into L sub-slices of subElems staging elements.
  // The M contributors to my output -- the M-1 remote nodes in send order (contributor c = node
  // (n - c) mod M), then my own node (c = M) -- go in waves of waveNodes; within a wave, contributor
  // c's partial for sub-slice p of destination (m, l) is computed on its node's local rank (p - c)
  // mod L and lands in inbox block p, sub-block l, of rank (m, that rank): consecutive c have
  // distinct residues, so block p at the same offset on the L local ranks holds each of the wave's
  // partials once and the owner reads their sum with one multimem.ld_reduce per 16 B, accumulating
  // wave after wave into the output. Waves alternate two inbox parities; a block no contributor of
  // a wave feeds must read zero (zeroed at registration, and per call where the parity's previous
  // wave fed it). Staging slot: [source][dest l][chunk]; single node: [source][chunk], the owner
  // sums its whole destination. With parts > 1 (TIDE_RS_PARTS) the rounds and waves run once per
  // part, part q carrying staging elements [q*partElems, (q+1)*partElems) of every sub-slice, so an
  // inbox / outbox block is [l][partElems]; waves are numbered along the whole call (RsWave::part).
  size_t subElems = 0;
  int parts = 1;
  size_t partElems = 0;
  using Wave = RsWave;  // collective/rs/waves.h
  std::vector<Wave> waves;           // multi-node
  int waveNodes = kRsMaxLocal, inboxParities = 1;
  // One rank per node (rail, L = 1: no local team, no multicast): the copy engines pack my pieces for
  // destination t (input dtype, staging-indexed) into an outbox block, the NIC rounds put them into the
  // owners' inboxes -- [2 parities][railBlocks] blocks, round r in parity r % 2, the j-th destination of
  // a round in block j - first -- and the owner sums each round's blocks (plus, in the first round, its
  // own piece read in place) into `out` with rsCtas() CTAs. Same k-way rounds, release credits and
  // outbox reuse as the multicast form. A block holds one part (chunkElems staging elements, `chunks`
  // parts = TIDE_RS_RAIL_PARTS) and the rounds run once per part. railBlocks = 0: not a rail plan.
  int railBlocks = 0;
  int area = -1;   // credit / flag channel shared by plans whose stage areas overlap
  bool paced = false;
  size_t paceTail = 0;  // see AgPlan::paceTail
  std::vector<std::pair<size_t, size_t>> nicChunks;  // (offset, bytes) of the outbox per RMA context

  RsSegDev* segsDev = nullptr;  // device-side static table (owned)

  uint64_t calls = 0;
  ~RsPlan();
};

// stageOff: window-relative start of the plan's stage area (rs_stage_bytes(...) bytes).
std::shared_ptr<RsPlan> buildRsPlan(const Comm& comm, std::shared_ptr<Window> arena, size_t stageOff,
                                    const std::vector<std::pair<size_t, size_t>>& shapes, int itemsize,
                                    size_t chunkBytesHint);

// Stage-area footprint for a geometry (staging slot bytes, inbox/outbox block stride) as the plans
// sharing the area lay it out: two staging slots; multi-node adds inboxParities x L inbox blocks,
// outboxSlots outbox blocks and ownSpares spare blocks.
struct RsAreaOffsets {
  size_t inboxOff = 0, outboxOff = 0, ownOff = 0, bytes = 0;
};
RsAreaOffsets rsAreaOffsets(const Comm& comm, int outboxSlots, int inboxParities, int ownSpares, size_t slotBytes,
                            size_t stride);

// Bytes of stage area a plan over `shapes` needs on this communicator.
size_t rsStageBytes(const Comm& comm, const std::vector<std::pair<size_t, size_t>>& shapes, int itemsize,
                    size_t chunkBytesHint);
// Bytes of one stage area shared by plans over `tables` ((shapes, itemsize) each): the geometry of the
// largest table in every dimension (a table with fewer, larger chunks needs the bigger staging slot,
// one with more output the bigger blocks).
size_t rsAreaBytes(const Comm& comm, const std::vector<std::pair<std::vector<std::pair<size_t, size_t>>, int>>& tables,
                   size_t chunkBytesHint);

}  // namespace tide
