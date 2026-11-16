// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "runtime/comm.h"
#include "util/nvtx.h"

#include <nccl_device.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <memory>
#include <cstring>
#include <string>
#include <utility>

#include "collective/ag/plan.h"
#include "util/check.h"
#include "runtime/comm_internal.h"

namespace tide {

using namespace detail;

void Comm::traceBegin(uint64_t k, size_t phases) {
  if (traceEv_.empty()) {
    traceEv_.resize(kTraceCalls);
    traceSeq_.assign(kTraceCalls, 0);
  }
  auto& v = traceEv_[traceIdx_ % kTraceCalls];
  while (v.size() < phases) {
    v.push_back(nullptr);
    CUDACHECK(cudaEventCreate(&v.back()));
  }
  traceSeq_[traceIdx_ % kTraceCalls] = k;
  tracePhase_ = 0;
}

void Comm::traceMark(cudaStream_t s) {
  auto& v = traceEv_[traceIdx_ % kTraceCalls];
  if (tracePhase_ < v.size()) CUDACHECK(cudaEventRecord(v[tracePhase_], s));
  ++tracePhase_;
}

std::vector<std::pair<uint64_t, std::vector<float>>> Comm::traceAgDump() {
  std::vector<std::pair<uint64_t, std::vector<float>>> out;
  if (traceEv_.empty()) return out;
  for (size_t i = 0; i < kTraceCalls; ++i) {  // oldest first
    const size_t slot = (traceIdx_ + 1 + i) % kTraceCalls;
    const auto& v = traceEv_[slot];
    if (traceSeq_[slot] == 0 || v.empty() || cudaEventQuery(v.back()) != cudaSuccess) continue;
    std::vector<float> ms;
    for (size_t p = 1; p < v.size(); ++p) {
      float t = 0.f;
      if (cudaEventElapsedTime(&t, v[p - 1], v[p]) != cudaSuccess) { ms.clear(); break; }
      ms.push_back(t);
    }
    if (!ms.empty()) out.emplace_back(traceSeq_[slot], std::move(ms));
    traceSeq_[slot] = 0;  // report each call once
  }
  return out;
}

std::shared_ptr<AgPlan> Comm::registerAg(std::shared_ptr<Window> arena, size_t regionOff, const std::vector<size_t>& sizes,
                                         const std::vector<size_t>& offsets, long stageOff, long ownOff) {
  live();
  TIDE_CHECK(arena && arena->comm == comm_ && !arena->freed, "arena window belongs to another communicator or was freed");
  auto plan = buildAgPlan(*this, arena, regionOff, sizes, offsets, stageOff, ownOff);
  const Range landing{arena->id, regionOff, regionOff + plan->regionBytes};
  for (const auto& a : rsAreas_) TIDE_CHECK(!a.live || !landing.overlaps(a), "landing overlaps a reduce-scatter stage area");
  for (const auto& a : stageAreas_) TIDE_CHECK(!a.live || !landing.overlaps(a), "landing overlaps a registered stage area");
  for (const auto& o : ownShards_) TIDE_CHECK(!landing.overlaps(o), "landing overlaps a registered own shard");
  const size_t S = plan->shardBytes;
  if (plan->staged && !plan->ownInStage) {
    // An own shard outside the stage is written by the caller at any time (the sharded parameters
    // live there): it may never be under a landing or a receive area some other group's peers write into.
    const Range own{arena->id, plan->ownOff, plan->ownOff + S};
    TIDE_CHECK(!own.overlaps(landing), "own shard overlaps the plan's own landing");
    for (const auto& l : landings_) TIDE_CHECK(!own.overlaps(l), "own shard overlaps a registered landing");
    for (const auto& a : stageAreas_) TIDE_CHECK(!a.live || !own.overlaps(a), "own shard overlaps a registered receive area");
    for (const auto& a : rsAreas_) TIDE_CHECK(!a.live || !own.overlaps(a), "own shard overlaps a reduce-scatter stage area");
    ownShards_.push_back(own);
  }
  if (plan->direct) {
    // one `ready` channel for every direct plan: they all have the same rounds, so the counts pair up per call
    if (readySig_ < 0) {
      checkSignalBudget(stageAreas_.size() + 1, rsAreas_.size());
      stageAreas_.push_back({{-1, 0, 0}});  // window -1: never overlaps, never freed
      readySig_ = kSigCredit0 + static_cast<int>(stageAreas_.size()) - 1;
    }
  } else if (nNodes_ > 1) {
    // Writes into a stage area are gated only by its channel's credits (plans whose areas overlap
    // share the channel and, called in the same order everywhere, protect each other's bytes), so
    // it may never touch a landing some other group may be computing on, nor a persistent own shard.
    // An own shard inside the stage (the first slot) belongs to the channel area.
    const Range stage{arena->id, plan->ownInStage ? plan->ownOff : plan->recvBase, plan->recvBase + static_cast<size_t>(nNodes_ - 1) * S};
    TIDE_CHECK(!stage.overlaps(landing), "stage area overlaps the plan's own landing");
    for (const auto& l : landings_) TIDE_CHECK(!stage.overlaps(l), "stage area overlaps a registered landing");
    for (const auto& o : ownShards_) TIDE_CHECK(!stage.overlaps(o), "stage area overlaps a registered own shard");
    for (const auto& a : rsAreas_) TIDE_CHECK(!a.live || !stage.overlaps(a), "stage area overlaps a reduce-scatter stage area");
    plan->stageSlot = stageSlotOf(*arena, stage.begin, stage.end);
  } else if (plan->staged && plan->ownInStage) {
    const Range own{arena->id, plan->ownOff, plan->ownOff + S};
    TIDE_CHECK(!own.overlaps(landing), "stage overlaps the plan's own landing");
    for (const auto& l : landings_) TIDE_CHECK(!own.overlaps(l), "stage overlaps a registered landing");
  }
  landings_.push_back(landing);
  return plan;
}


void Comm::allGather(AgPlan& plan, cudaStream_t s, cudaEvent_t freeEv, cudaEvent_t doneEv) {
  live();
  TIDE_CHECK(!failed_, "an earlier all-gather failed mid-protocol; the communicator must be recreated");
  TIDE_CHECK(plan.owner == this && plan.arena && plan.arena->comm == comm_ && !plan.arena->freed,
             "plan belongs to another communicator or its window was freed");
  // The fences are plain writes, so calls must execute in issue order even across streams: each
  // call ends by marking its stream, and the next one starts by waiting for that mark.
  if (ordered_) CUDACHECK(cudaStreamWaitEvent(s, orderEv_, 0));
  const uint64_t k = ++seq_;
  nvtx::Range range("tide.ag", nvtx::kAg);
  try {
    flushCredits(s);
    if (plan.direct) issueDirect(plan, k, s);
    else if (nNodes_ > 1) issueMultiNode(plan, k, s, freeEv, doneEv);
    else issueSingleNode(plan, k, s, freeEv, doneEv);
    if (doneEv && plan.direct) CUDACHECK(cudaEventRecord(doneEv, s));
    CUDACHECK(cudaEventRecord(orderEv_, s));
  } catch (...) {
    failed_ = true;
    throw;
  }
  ordered_ = true;
  ++plan.calls;
}

void Comm::awaitStageReads(cudaStream_t s) {
  live();
  if (!pendingCreditTo_.empty()) CUDACHECK(cudaStreamWaitEvent(s, pendingCreditEv_, 0));
}

// The previous decoupled call's stage credits, from `s` once its scatters finished (normally long before).
void Comm::flushCredits(cudaStream_t s) {
  if (pendingCreditTo_.empty()) return;
  nvtx::Range range("tide.ag.credits", nvtx::kSync);
  CUDACHECK(cudaStreamWaitEvent(s, pendingCreditEv_, 0));
  Group grp;
  for (int to : pendingCreditTo_) signal(to, pendingCreditSig_, /*ctx=*/0, s);
  grp.end();
  pendingCreditTo_.clear();
}

// Decoupled (doneEv): a slow rank's previous call no longer holds back my `ready`, so the node does not
// re-synchronize on every call; the landing's previous tenant is protected by freeEv instead.
void Comm::announceReady(uint64_t k, cudaStream_t s, cudaEvent_t freeEv, cudaEvent_t doneEv) {
  if (!doneEv) {
    writeFlags(*sync_, readyOff_, k, s);
    return;
  }
  if (freeEv && !faultOn(Fault::nofree, rank_)) CUDACHECK(cudaStreamWaitEvent(readyStream_, freeEv, 0));
  writeFlags(*sync_, readyOff_, k, readyStream_);
}

// The exit wait on `s`, or (decoupled) on doneStream_ so the next call on `s` goes ahead; returns the waiting stream.
cudaStream_t Comm::awaitDone(uint64_t k, cudaStream_t s, cudaEvent_t doneEv) {
  cudaStream_t w = s;
  if (doneEv) {
    CUDACHECK(cudaEventRecord(doneForkEv_, s));
    CUDACHECK(cudaStreamWaitEvent(doneStream_, doneForkEv_, 0));
    w = doneStream_;
  }
  if (!faultOn(Fault::nodone, rank_)) waitFlags(*sync_, doneOff_, k, w);
  return w;
}

void Comm::issueSingleNode(AgPlan& plan, uint64_t k, cudaStream_t s, cudaEvent_t freeEv, cudaEvent_t doneEv) {
  // 1. tell peers my region is free to be written into, 2. wait until every peer's region is free
  announceReady(k, s, freeEv, doneEv);
  if (!faultOn(Fault::nofence, rank_)) waitFlags(*sync_, readyOff_, k, s);
  // 3. scatter my segments into every peer's region (copy engines only)
  if (!faultOn(Fault::nowrite, rank_)) {
    if (faultOn(Fault::slow, rank_)) faultDelay(s);
    batchCopy(plan.copies, s);
  }
  // 4. announce completion, 5. wait until everyone finished writing into my region
  writeFlags(*sync_, doneOff_, k, s);
  const cudaStream_t w = awaitDone(k, s, doneEv);
  if (doneEv) CUDACHECK(cudaEventRecord(doneEv, w));
}

// Multi-node all-gather, rail-aligned recursive doubling (AgPlan::Round): round r forwards
// everything held so far -- slots [0, cnt) -- to rail peer n - 2^r and receives the same block from
// n + 2^r into slots [2^r, 2^r + cnt): ceil(log2 M) rounds, the shard's bytes once per node on the
// wire. The copy engines multicast each received block into the param-major landing as it lands
// (one write lands on every local rank); the last round goes in TIDE_NIC_GROUPS slot ranges so the
// exposed tail scatter is a fraction of the block. Pacing: the put of round r + 1 to `to` waits for
// `to`'s signal that its round r landed (all but a tail), and my round r + 1 source is signalled
// once my round r heads landed -- a receiving port never sees two full flows. A round's put waits
// for the previous round's arrival by stream order (it forwards that data). Once every slot is
// scattered (my slots are also read as put sources until then), a credit (signal index 2 + stage
// slot) tells each source that its next put into this area may proceed. Every NCCL call stays on
// `s`: NCCL orders all operations of a communicator through one internal stream, so an RMA call on
// ceStream_ would serialize the scatters with the next put.
void Comm::issueMultiNode(AgPlan& plan, uint64_t k, cudaStream_t s, cudaEvent_t freeEv, cudaEvent_t doneEv) {
  const int R = static_cast<int>(plan.rounds.size());
  StageArea& area = stageAreas_[plan.stageSlot];
  const int creditSig = kSigCredit0 + plan.stageSlot;
  const bool skipData = faultOn(Fault::nowrite, rank_);
  size_t nEv = 0;
  for (const auto& rd : plan.rounds) nEv += rd.groupPuts.size();
  while (roundEv_.size() < nEv) {
    roundEv_.push_back(nullptr);
    CUDACHECK(cudaEventCreateWithFlags(&roundEv_.back(), cudaEventDisableTiming));
  }

  if (cfg_.traceAg) traceBegin(k, static_cast<size_t>(R) + 5);
  if (cfg_.traceAg) traceMark(s);  // start
  auto sync = std::make_unique<nvtx::Range>("tide.ag.sync", nvtx::kSync);
  announceReady(k, s, freeEv, doneEv);
  if (area.uses > 0 && !skipData && !faultOn(Fault::nocredit, rank_)) {
    std::vector<std::array<int, 4>> credits;  // my targets have scattered what I put into them last call
    for (const auto& rd : plan.rounds) credits.push_back({1, rd.to, creditSig, 0});
    waitSignal(credits, s);
  }
  if (cfg_.traceAg) traceMark(s);  // credits
  if (!faultOn(Fault::nofence, rank_)) waitFlags(*sync_, readyOff_, k, s);
  if (cfg_.traceAg) traceMark(s);  // fence
  sync.reset();

  CUDACHECK(cudaEventRecord(forkEv_, s));
  CUDACHECK(cudaStreamWaitEvent(ceStream_, forkEv_, 0));
  auto scatter = [&](PushBatch& b) {
    if (!faultOn(Fault::noscatter, rank_)) batchCopy(b, ceStream_);
  };
  if (!skipData) scatter(plan.ownScatter);
  size_t ev = 0;
  for (int r = 0; r < R && !skipData; ++r) {
    nvtx::Range round("tide.ag.round", nvtx::kAg);
    AgPlan::Round& rd = plan.rounds[r];
    const bool last = r + 1 == R;
    if (plan.paced && r > 0) waitSignal({{1, rd.to, kSigPace, 0}}, s);  // `to` has received round r-1
    const int paceTo = plan.paced && !last ? plan.rounds[r + 1].from : -1;
    const int G = static_cast<int>(rd.groupPuts.size());
    // rounds before the last: one group over the block, pacing the next round's source; the last:
    // group by group, each scattered as it lands
    for (int g = 0; g < G; ++g) {
      std::vector<PutSpec> specs;
      for (const AgPlan::Put& put : rd.groupPuts[g]) specs.push_back({put.srcOff, put.dstOff, rd.to, rd.from, &put.chunks});
      putRounds(*plan.arena, specs, paceTo >= 0 && g == 0 ? rd.paceTail : 0, kSigData,
                paceTo >= 0 && g == 0 ? std::vector<int>{paceTo} : std::vector<int>{}, kSigPace, s);
      cudaEvent_t e = roundEv_[ev++];
      CUDACHECK(cudaEventRecord(e, s));
      CUDACHECK(cudaStreamWaitEvent(ceStream_, e, 0));
      if (faultOn(Fault::slow, rank_)) faultDelay(ceStream_);
      scatter(rd.scatter[g]);
    }
    if (cfg_.traceAg) traceMark(s);  // round r landed
  }
  if (cfg_.traceAg && skipData) for (int r = 0; r < R; ++r) traceMark(s);
  nvtx::Range tail("tide.ag.tail", nvtx::kAgTail);
  CUDACHECK(cudaEventRecord(ceJoinEv_, ceStream_));
  if (doneEv) {  // decoupled: `s` is free for the next operation once the last round is issued
    if (!skipData) {
      CUDACHECK(cudaEventRecord(pendingCreditEv_, ceStream_));
      for (const auto& rd : plan.rounds) pendingCreditTo_.push_back(rd.from);
      pendingCreditSig_ = creditSig;
      ++area.uses;
    }
    CUDACHECK(cudaStreamWaitEvent(doneStream_, ceJoinEv_, 0));
    if (cfg_.traceAg) traceMark(doneStream_);  // scatters joined
    writeFlags(*sync_, doneOff_, k, doneStream_);
    if (!faultOn(Fault::nodone, rank_)) waitFlags(*sync_, doneOff_, k, doneStream_);
    if (cfg_.traceAg) { traceMark(doneStream_); ++traceIdx_; }  // done fence
    CUDACHECK(cudaEventRecord(doneEv, doneStream_));
    return;
  }
  CUDACHECK(cudaStreamWaitEvent(s, ceJoinEv_, 0));
  if (cfg_.traceAg) traceMark(s);  // scatters joined
  if (!skipData) {
    Group grp;  // every slot is scattered (and no longer a put source): sources may overwrite my stage
    for (const auto& rd : plan.rounds) signal(rd.from, creditSig, /*ctx=*/0, s);
    grp.end();
    ++area.uses;
  }
  writeFlags(*sync_, doneOff_, k, s);
  const cudaStream_t w = awaitDone(k, s, doneEv);
  if (cfg_.traceAg) { traceMark(w); ++traceIdx_; }  // done fence
  if (doneEv) CUDACHECK(cudaEventRecord(doneEv, w));
}

// One rank per node (AgPlan::direct): the same recursive doubling, landing to landing. My landing is
// free once the caller's stream reached this call (FSDP orders it behind the previous occupant's
// compute), so `ready` goes to every rank that writes into it and a round's put waits for its
// target's; the arrivals of each round are awaited before the next forwards them. No stage, no scatter.
void Comm::issueDirect(AgPlan& plan, uint64_t k, cudaStream_t s) {
  const int R = static_cast<int>(plan.rounds.size());
  const bool skipData = faultOn(Fault::nowrite, rank_);
  if (cfg_.traceAg) traceBegin(k, static_cast<size_t>(R) + 5);
  if (cfg_.traceAg) traceMark(s);  // start
  if (!skipData) batchCopy(plan.copies, s);
  {
    Group grp;
    for (const auto& rd : plan.rounds) signal(rd.from, readySig_, /*ctx=*/0, s);
    grp.end();
  }
  if (cfg_.traceAg) traceMark(s);  // credits
  if (!faultOn(Fault::nofence, rank_) && !faultOn(Fault::nocredit, rank_)) {
    std::vector<std::array<int, 4>> ready;
    for (const auto& rd : plan.rounds) ready.push_back({1, rd.to, readySig_, 0});
    waitSignal(ready, s);
  }
  if (cfg_.traceAg) traceMark(s);  // fence
  for (int r = 0; r < R && !skipData; ++r) {
    AgPlan::Round& rd = plan.rounds[r];
    if (faultOn(Fault::slow, rank_)) faultDelay(s);
    if (plan.paced && r > 0) waitSignal({{1, rd.to, kSigPace, 0}}, s);  // `to` has received its round r-1
    // no head / tail split: my sends and my arrivals split at different wraps, so the tails would not pair up
    const int paceTo = plan.paced && r + 1 < R ? plan.rounds[r + 1].from : -1;
    std::vector<PutSpec> specs, arrivals;
    for (const AgPlan::Put& put : rd.groupPuts[0]) specs.push_back({put.srcOff, put.dstOff, rd.to, rd.from, &put.chunks});
    for (const AgPlan::Put& put : rd.recvPuts) arrivals.push_back({put.srcOff, put.dstOff, rd.to, rd.from, &put.chunks});
    putRounds(*plan.arena, specs, 0, kSigData, paceTo >= 0 ? std::vector<int>{paceTo} : std::vector<int>{}, kSigPace, s, {},
              &arrivals);
    if (cfg_.traceAg) traceMark(s);  // round r landed
  }
  if (cfg_.traceAg && skipData) for (int r = 0; r < R; ++r) traceMark(s);
  if (cfg_.traceAg) { traceMark(s); traceMark(s); ++traceIdx_; }  // no scatter, no done fence
}

}  // namespace tide
