// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "runtime/copy.h"

namespace tide {

class Comm;
struct Window;

// Layout-aware all-gather over one arena region.
// Segment p of rank r lands at  region + offsets[p]*nranks + r*sizes[p]  (bytes);
// this rank's own segments are expected to be written there by the caller
// (in-place), so the plan only moves data to LSA peers. Every landing write is a copy-engine
// copy to the NVLS multimem address (the switch delivers it to every local rank).
struct AgPlan {
  const Comm* owner = nullptr;
  std::shared_ptr<Window> arena;
  size_t regionOff = 0;
  std::vector<size_t> sizes, offsets;  // per-segment bytes, shard-relative byte offsets
  size_t shardBytes = 0;               // sum(sizes)
  size_t regionBytes = 0;              // shardBytes * nranks

  // Single node: one descriptor per segment.
  CopyBatch copies;

  // Where this rank's own segment p is written before a call: its packed shard at ownOff (staged
  // plans; the sharded parameters themselves may live there), else in place in the landing.
  std::vector<char*> ownPieces;
  size_t ownOff = 0;
  bool staged = false;
  bool ownInStage = false;  // own shard = the first slot of the stage area (no own_off given): written per call

  // Multi-node: recvBase holds M-1 receive slots, slot i (1..M-1) the shard of my rail peer on node
  // (n + i) mod M; "slot 0" is the own shard at ownOff. stageSlot identifies the credit channel
  // shared by every plan whose receive area overlaps this one.
  size_t recvBase = 0;
  int stageSlot = -1;
  bool paced = false;

  // Recursive doubling over the rail: round r puts slots [0, cnt) to rail peer n - 2^r -- whose
  // slots [2^r, 2^r + cnt) they are -- and receives the same block from n + 2^r into my slots
  // [2^r, 2^r + cnt); cnt = min(2^r, M - 2^r), ceil(log2 M) rounds. Received blocks are scattered
  // param-major as they land (consecutive slots are consecutive rail-peer ranks: strided
  // descriptors, one or two runs per segment at the wrap); the last round goes in TIDE_NIC_GROUPS
  // slot ranges, each scattered as it lands. A group's puts (`Put`: the own shard, then the run of
  // receive slots) are each striped over the RMA contexts. A paced round releases the next round's
  // source once all but paceTail bytes of every chunk have landed (0: after the whole round).
  struct Put {
    size_t srcOff = 0, dstOff = 0;                      // window-relative
    std::vector<std::pair<size_t, size_t>> chunks;      // (offset, bytes) per context, relative to both
  };
  struct Round {
    int to = 0, from = 0;
    size_t recvSlot = 0, cnt = 0;
    size_t paceTail = 0;
    std::vector<std::pair<size_t, size_t>> groupRanges;  // (first slot of the block, slots) per group
    std::vector<std::vector<Put>> groupPuts;             // per group
    std::vector<PushBatch> scatter;                      // per group
    std::vector<Put> recvPuts;  // direct: what `from` sends me (its runs split at its own wrap)
  };
  std::vector<Round> rounds;
  PushBatch ownScatter;  // own shard -> landing

  // One rank per node (L = 1, e.g. an expert group of node-local EP): no NVLink peer, no multicast,
  // and a receive stage would be as large as the landing. The rounds put straight from my landing
  // into the target's landing at the same offsets -- a block of consecutive ranks is one run per
  // segment (two at the wrap) -- so there is no stage and no scatter; `copies` moves a separate own
  // shard into my landing first. A target is written only after its `ready` signal for the call.
  // The arrivals to wait for follow the sender's runs (Round::recvPuts). Paced like the staged rounds:
  // each round is a permutation, but a rank that got its round r early would start round r + 1 into a
  // target still receiving its round r (two flows into one port: ECN storms on the expert rails).
  bool direct = false;

  uint64_t calls = 0;
};

// ownOff >= 0: this rank's packed shard lives at that window-relative offset (else in place, single
// node only). stageOff: multi-node receive area of (nNodes - 1) packed slots (required there); when
// ownOff < 0 and stageOff >= 0 the own shard is the first of nNodes slots at stageOff.
std::shared_ptr<AgPlan> buildAgPlan(const Comm& comm, std::shared_ptr<Window> arena, size_t regionOff,
                                    const std::vector<size_t>& sizes, const std::vector<size_t>& offsets,
                                    long stageOff = -1, long ownOff = -1);

}  // namespace tide
