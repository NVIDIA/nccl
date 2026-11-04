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

#include "collective/rs/plan.h"
#include "util/check.h"
#include "runtime/comm_internal.h"

namespace tide {

using namespace detail;

// ------------------------------------------------------------------------------------------------
// Reduce-scatter. Gradients are read in place; the node-local 8 -> 1 sum is "copy engines push the
// other ranks' pieces into the owner's staging slot, the owner sums them in fp32 on a few CTAs";
// the cross-node step sends fp32 partial sums over the rail with the all-gather's NIC leg (k-way
// rounds, context striping, pacing, per-area credits) and adds them in the NVSwitch (multicast
// landing). Every wait is a stream memop or a signal wait, so no SM spins on a peer.

namespace {
constexpr size_t kRsMaxAreas = 16;
}  // namespace

void Comm::setRsTuning(int sumCtas, int mcCtas, int kway, long chunkBytes, int parts, int outbox, int pipeline) {
  live();
  TIDE_CHECK(!rsSync_, "setRsTuning after the first reduce-scatter registration");
  if (pipeline > 0 && !(cfg_.envSet & Config::kEnvRsPipeline)) cfg_.rsPipeline = std::min(pipeline, 2);
  if (sumCtas > 0 && !(cfg_.envSet & Config::kEnvRsCtas)) cfg_.rsCtas = sumCtas;
  if (mcCtas > 0 && !(cfg_.envSet & Config::kEnvRsMcCtas)) cfg_.rsMcCtas = mcCtas;
  if (kway > 0 && !(cfg_.envSet & Config::kEnvRsKway)) cfg_.rsKway = kway;
  if (chunkBytes > 0 && !(cfg_.envSet & Config::kEnvRsChunk)) cfg_.rsChunk = static_cast<size_t>(chunkBytes);
  if (parts > 0 && !(cfg_.envSet & Config::kEnvRsParts)) cfg_.rsParts = std::min(parts, Config::kRsMaxParts);
  if (outbox > 0 && !(cfg_.envSet & Config::kEnvRsOutbox)) cfg_.rsOutbox = outbox;
}

size_t Comm::rsStageBytes(const std::vector<std::pair<size_t, size_t>>& shapes, int itemsize) const {
  return tide::rsStageBytes(*this, shapes, itemsize, rsChunkBytes());
}

size_t Comm::rsAreaBytes(const std::vector<std::pair<std::vector<std::pair<size_t, size_t>>, int>>& tables) const {
  return tide::rsAreaBytes(*this, tables, rsChunkBytes());
}

// Flags live in their own window: [area][kind][slot][rank] uint64. Collective (the first
// registration on every rank allocates it).
void Comm::initRsSync() {
  if (rsSync_) return;
  rsFlagArrayBytes_ = alignUp(static_cast<size_t>(nranks_) * sizeof(uint64_t), 16);
  rsSync_ = allocWindow(kRsMaxAreas * kRsFlagKinds * rsFlagArrayBytes_, /*rail=*/false);  // node-local flags
  // per destination index (level-1 sums done, puts issued) and per inbox parity
  auto grow = [](std::vector<cudaEvent_t>& v, size_t n) {
    for (v.reserve(n); v.size() < n;) {
      v.push_back(nullptr);
      CUDACHECK(cudaEventCreateWithFlags(&v.back(), cudaEventDisableTiming));
    }
  };
  for (auto* v : {&rsSumEv_, &rsPutEv_}) grow(*v, static_cast<size_t>(std::max(nNodes_, 2)));
  for (auto* v : {&rsReadyEv_, &rsDoneEv_, &rsZeroEv_}) grow(*v, 2);
}

size_t Comm::rsFlagOff(int area, RsFlag kind, int slot) const {
  // [land0..land{S-1} free0..free{S-1} waveReady0 waveReady1 waveDone0 waveDone1], S = kRsMaxStageSlots
  constexpr int S = Config::kRsMaxStageSlots;
  const int idx = kind <= kFree ? kind * S + slot : 2 * S + (kind - kWaveReady) * 2 + slot;
  return (static_cast<size_t>(area) * kRsFlagKinds + static_cast<size_t>(idx)) * rsFlagArrayBytes_;
}

// Same channel rules as stageSlotOf; a revived area keeps its flag and credit counters but takes
// the geometry of the first plan registered on it anew.
int Comm::rsAreaOf(const Window& w, size_t begin, size_t end) {
  int hit = -1, dead = -1;
  for (size_t i = 0; i < rsAreas_.size(); ++i) {
    RsArea& a = rsAreas_[i];
    if (!a.live) {
      if (dead < 0) dead = static_cast<int>(i);
      continue;
    }
    if (a.winId != w.id || begin >= a.end || a.begin >= end) continue;
    TIDE_CHECK(hit < 0, "reduce-scatter stage area overlaps two distinct areas");
    hit = static_cast<int>(i);
    a.begin = std::min(a.begin, begin);
    a.end = std::max(a.end, end);
  }
  if (hit >= 0) return hit;
  if (dead >= 0) {
    RsArea& a = rsAreas_[dead];
    a.winId = w.id;
    a.begin = begin;
    a.end = end;
    a.live = true;
    a.slotBytes = a.inboxBase = a.outboxBase = a.ownBase = a.outStride = 0;
    a.geometrySet = a.fixed = false;
    return dead;
  }
  TIDE_CHECK(rsAreas_.size() < kRsMaxAreas, "out of reduce-scatter stage areas");
  checkSignalBudget(stageAreas_.size(), rsAreas_.size() + 1);
  rsAreas_.push_back({{w.id, begin, end}});
  return static_cast<int>(rsAreas_.size()) - 1;
}

// Plans sharing a stage area adopt one geometry (staging slot bytes, inbox/outbox block stride), so
// slot i and block p mean the same bytes for all of them: the area grows to the largest table in
// every dimension while it is unused (a table with fewer, larger chunks needs the bigger slot; one
// with more output the bigger blocks), and is fixed by its first call. Plans copy the geometry when
// a call is issued (Comm::adoptArea).
void Comm::growArea(RsArea& a, const RsPlan& plan, const Window& arena) {
  if (a.geometrySet && plan.slotBytes <= a.slotBytes && plan.outStride <= a.outStride) return;
  TIDE_CHECK(!a.fixed, "a reduce-scatter stage area's geometry is fixed by its first call: register every table sharing it before that call");
  a.slotBytes = std::max(a.slotBytes, plan.slotBytes);
  a.outStride = std::max(a.outStride, plan.outStride);
  a.geometrySet = true;
  const RsAreaOffsets off = rsAreaOffsets(*this, plan.outboxSlots, plan.inboxParities, plan.ownSpares, a.slotBytes, a.outStride);
  a.inboxBase = a.begin + off.inboxOff;
  a.outboxBase = a.begin + off.outboxOff;
  a.ownBase = a.begin + off.ownOff;
  a.end = a.begin + off.bytes;
  TIDE_CHECK(a.end <= arena.bytes,
             "reduce-scatter stage area exceeds the arena window (size it with rs_area_bytes over every table sharing it)");
  for (const auto& l : landings_) TIDE_CHECK(!a.overlaps(l), "reduce-scatter stage area overlaps an all-gather landing");
  for (const auto& sa : stageAreas_) TIDE_CHECK(!sa.live || !a.overlaps(sa), "reduce-scatter stage area overlaps an all-gather stage area");
  for (size_t i = 0; i < rsAreas_.size(); ++i)
    TIDE_CHECK(&rsAreas_[i] == &a || !rsAreas_[i].live || !a.overlaps(rsAreas_[i]), "reduce-scatter stage area overlaps another one");
  if (nNodes_ > 1 && !plan.railBlocks) {
    // Blocks the first wave of each parity does not feed must read as zero under the owners'
    // multimem sum (the per-call zeroing handles later waves, RsPlan::Wave::zero).
    const int L = localRanks_;
    for (size_t w = 0; w < plan.waves.size() && w < static_cast<size_t>(plan.inboxParities); ++w)
      for (int p = 0; p < L; ++p)
        if (!(plan.waves[w].live >> p & 1))
          CUDACHECK(cudaMemset(static_cast<char*>(arena.base) + a.inboxBase +
                                   (static_cast<size_t>(plan.waves[w].parity) * L + static_cast<size_t>(p)) * a.outStride,
                               0, a.outStride));
    CUDACHECK(cudaDeviceSynchronize());
  }
}

void Comm::adoptArea(RsPlan& plan) const {
  const RsArea& a = rsAreas_[plan.area];
  plan.slotBytes = a.slotBytes;
  plan.outStride = a.outStride;
  plan.inboxBase = a.inboxBase;
  plan.outboxBase = a.outboxBase;
  plan.ownBase = a.ownBase;
}

std::shared_ptr<RsPlan> Comm::registerRs(std::shared_ptr<Window> arena, size_t stageOff,
                                         const std::vector<std::pair<size_t, size_t>>& shapes, int itemsize) {
  live();
  TIDE_CHECK(arena && arena->comm == comm_ && !arena->freed, "arena window belongs to another communicator or was freed");
  rsRequireTma();
  initRsSync();
  auto plan = buildRsPlan(*this, arena, stageOff, shapes, itemsize, rsChunkBytes());
  plan->area = rsAreaOf(*arena, stageOff, stageOff + plan->stageBytes);
  RsArea& a = rsAreas_[plan->area];
  TIDE_CHECK(plan->stageBase == a.begin, "tables sharing a reduce-scatter stage area must start at the same offset");
  TIDE_CHECK(nNodes_ == 1 || plan->railBlocks || arena->multimem != nullptr, "multi-node reduce-scatter needs an NVLS-bound arena window");
  growArea(a, *plan, *arena);
  adoptArea(*plan);
  return plan;
}

namespace {
RsSumArgs sumArgs(const RsPlan& plan, const std::vector<const void*>& src, int L, int me, float scale, bool accumulate) {
  RsSumArgs a{};
  a.segs = plan.segsDev;
  a.nseg = static_cast<int>(plan.segs.size());
  for (size_t p = 0; p < src.size(); ++p) a.src[p] = src[p];
  a.itemsize = plan.itemsize;
  a.chunkBytes = plan.chunkBytes;
  a.L = L;
  a.me = me;
  a.scale = scale;
  a.accumulate = accumulate ? 1 : 0;
  return a;
}

// TIDE_RS_DEBUG=3: host time per call by category, averaged over each 20 calls and printed on rank 0.
struct ProfCall {
  bool on;
  int rank;
  double t0;
  ProfCall(bool on_, int rank_) : on(on_), rank(rank_), t0(on_ ? nowUs() : 0) { g_prof.on = on_; }
  ~ProfCall() {
    if (!on) return;
    g_prof.on = false;
    g_prof.total += nowUs() - t0;
    if (++g_prof.calls % 20 != 0) return;
    if (rank == 0) {
      const double* u = g_prof.us;
      const double* r = g_prof.readUs;
      const double t = g_prof.total, rd = g_prof.read;
      fprintf(stderr,
              "[TIDE-RS host] calls %llu, us/call over 20: total %.0f = copy %.0f + memop %.0f + nccl %.0f + other %.0f; "
              "read phase %.0f = copy %.0f + memop %.0f + nccl %.0f + other %.0f\n",
              static_cast<unsigned long long>(g_prof.calls), t / 20, u[0] / 20, u[1] / 20, u[2] / 20,
              (t - u[0] - u[1] - u[2]) / 20, rd / 20, r[0] / 20, r[1] / 20, r[2] / 20, (rd - r[0] - r[1] - r[2]) / 20);
    }
    g_prof.total = g_prof.read = 0;
    for (int c = 0; c < RsHostProf::kCats; ++c) g_prof.us[c] = g_prof.readUs[c] = 0;
  }
};
}  // namespace

void Comm::rsMarkRead(cudaStream_t st, int i) {
  if (!rsReadWanted_) return;
  CUDACHECK(cudaEventRecord(rsReadMark_[i], st));
  CUDACHECK(cudaStreamWaitEvent(rsReadStream_, rsReadMark_[i], 0));
}

void Comm::reduceScatter(RsPlan& plan, const std::vector<const void*>& src, float* out, float scale, bool accumulate,
                         cudaStream_t s, cudaEvent_t readEv, cudaEvent_t readyEv, cudaEvent_t gateEv) {
  live();
  TIDE_CHECK(!failed_, "an earlier call failed mid-protocol; the communicator must be recreated");
  TIDE_CHECK(plan.owner == this && plan.arena && plan.arena->comm == comm_ && !plan.arena->freed,
             "plan belongs to another communicator or its window was freed");
  TIDE_CHECK(src.size() == plan.segs.size(), "gradient count differs from the registered table");
  TIDE_CHECK(out != nullptr, "null output");
  for (const void* p : src) TIDE_CHECK(p != nullptr, "null gradient");
  const bool pipelined = plan.pipeline == 2 && nNodes_ > 1 && !plan.railBlocks;
  TIDE_CHECK(!pipelined || readyEv, "a pipelined reduce-scatter (TIDE_RS_PIPELINE=2) needs ready_event (gradients complete)");
  TIDE_CHECK(pipelined || !gateEv, "gate_event needs a pipelined multi-node reduce-scatter (TIDE_RS_PIPELINE=2)");
  // Flags are plain writes: reduce-scatter calls must execute in issue order across streams (each
  // call marks its stream at the end, the next one waits for the mark; same scheme as allGather).
  // A pipelined call orders only its network phase that way; its reads follow the previous call's on the CE / sum streams.
  if (!pipelined) {
    if (rsOrdered_) CUDACHECK(cudaStreamWaitEvent(s, rsOrderEv_, 0));
    if (readyEv) CUDACHECK(cudaStreamWaitEvent(s, readyEv, 0));
  }
  adoptArea(plan);
  rsAreas_[plan.area].fixed = true;
  const uint64_t u = rsAreas_[plan.area].uses + 1;
  nvtx::Range range("tide.rs", nvtx::kRsNet);
  try {
    ProfCall prof(cfg_.rsDebug == 3, rank_);
    if (faultOn(Fault::abort, rank_)) {
      waitFlags(*rsSync_, rsFlagOff(plan.area, kLand, 0), ~0ull, s);
      TIDE_CHECK(false, "TIDE_FAULT=abort: failed mid-protocol");
    }
    if (pipelined) {
      issueRsMcPipelined(plan, src, out, scale, accumulate, s, readEv, readyEv, gateEv);
    } else {
      const bool earlyRead = readEv && faultOn(Fault::earlyread, rank_);
      if (earlyRead) CUDACHECK(cudaEventRecord(readEv, s));
      rsReadWanted_ = readEv != nullptr && !earlyRead;
      if (plan.railBlocks) issueRsRail(plan, src, out, scale, accumulate, s);
      else if (nNodes_ > 1) issueRsMc(plan, src, out, scale, accumulate, s);
      else issueRsLocal(plan, src, out, scale, accumulate, s);
      if (rsReadWanted_) CUDACHECK(cudaEventRecord(readEv, rsReadStream_));
      rsReadWanted_ = false;
    }
    CUDACHECK(cudaEventRecord(rsOrderEv_, s));
  } catch (...) {
    rsReadWanted_ = false;
    failed_ = true;
    throw;
  }
  rsOrdered_ = true;
  ++plan.calls;
  if (cfg_.rsDebug == 2) rsDumpIfStuck(plan, u, s);
}

std::array<cudaStream_t, Comm::kRsCeStreams + 3> Comm::rsStreams() const {
  std::array<cudaStream_t, kRsCeStreams + 3> v;
  std::copy(rsCeStreams_.begin(), rsCeStreams_.end(), v.begin());
  v[kRsCeStreams] = rsSumStream_, v[kRsCeStreams + 1] = rsAddStream_, v[kRsCeStreams + 2] = rsZeroStream_;
  return v;
}

void Comm::rsFork(cudaStream_t s) {
  CUDACHECK(cudaEventRecord(rsForkEv_, s));
  for (cudaStream_t st : rsStreams()) CUDACHECK(cudaStreamWaitEvent(st, rsForkEv_, 0));
}

void Comm::rsJoin(cudaStream_t s) {
  for (cudaStream_t st : rsStreams()) {
    CUDACHECK(cudaEventRecord(rsJoinEv_, st));
    CUDACHECK(cudaStreamWaitEvent(s, rsJoinEv_, 0));
  }
}

int Comm::rsSubOf(int r, int t) const { return rsBlock(r, (t - node_ + nNodes_) % nNodes_, nNodes_, localRanks_); }

// TIDE_RS_DEBUG=1: serialize and report every phase.
void Comm::rsDebugSync(uint64_t u, const char* what, cudaStream_t st) {
  if (cfg_.rsDebug != 1) return;
  fprintf(stderr, "[TIDE-RS mc] rank %d call %llu: %s ...\n", rank_, static_cast<unsigned long long>(u), what);
  CUDACHECK(cudaStreamSynchronize(st));
  fprintf(stderr, "[TIDE-RS mc] rank %d call %llu: %s done\n", rank_, static_cast<unsigned long long>(u), what);
}

// TIDE_RS_DEBUG=2: poll for the call's completion; after 30 s dump the stream states and the area's flags.
void Comm::rsDumpIfStuck(const RsPlan& plan, uint64_t u, cudaStream_t s) {
  fprintf(stderr, "[TIDE-RS mc] rank %d call %llu issued\n", rank_, static_cast<unsigned long long>(u));
  std::vector<cudaStream_t> streams{s};
  for (cudaStream_t st : rsStreams()) streams.push_back(st);
  for (int i = 0; i < 300; ++i) {
    if (std::all_of(streams.begin(), streams.end(), [](cudaStream_t q) { return cudaStreamQuery(q) == cudaSuccess; })) {
      fprintf(stderr, "[TIDE-RS mc] rank %d call %llu complete\n", rank_, static_cast<unsigned long long>(u));
      return;
    }
    usleep(100000);
  }
  std::string msg = "[TIDE-RS mc] rank " + std::to_string(rank_) + " STUCK call " + std::to_string(u) +
                    " streams(caller,ce0-3,sum,land,zero)=";
  for (cudaStream_t q : streams) msg += " " + std::to_string(cudaStreamQuery(q));
  fprintf(stderr, "%s\n", msg.c_str());
  // read on a private stream: the caller's (possibly the legacy) stream is the one that is stuck
  const size_t stride = rsFlagArrayBytes_ / 8;
  std::vector<uint64_t> flags(kRsFlagKinds * stride);
  cudaStream_t peek;
  cudaStreamCreateWithFlags(&peek, cudaStreamNonBlocking);
  cudaMemcpyAsync(flags.data(), static_cast<char*>(rsSync_->base) + rsFlagOff(plan.area, kLand, 0), flags.size() * 8,
                  cudaMemcpyDeviceToHost, peek);
  cudaStreamSynchronize(peek);
  cudaStreamDestroy(peek);
  msg.clear();
  constexpr int S = Config::kRsMaxStageSlots;
  for (int k = 0; k < kRsFlagKinds; ++k) {
    if (k < 2 * S && k % S >= cfg_.rsStageSlots) continue;
    const std::string name = k < 2 * S ? (k < S ? "land" : "free") + std::to_string(k % S)
                                       : (k < 2 * S + 2 ? "waveReady" : "waveDone") + std::to_string(k % 2);
    msg += "  " + name + ":";
    for (int q = 0; q < nranks_; ++q) msg += " " + std::to_string(flags[static_cast<size_t>(k) * stride + q]);
    msg += "\n";
  }
  fprintf(stderr, "%s", msg.c_str());
}

// Level 1 for node t: the copy engines push every local rank's pieces of part `part` of my sub-slice
// of the L destinations on node t into my staging slot, K chunk-rounds over the staging slots (land/free
// flags per slot), and rsCtas() CTAs sum them in fp32 into outBuf ([l][partElems]). whole (single
// node): my whole destination into out with the caller's scale/accumulate.
void Comm::rsLevel1(RsPlan& plan, RsSumArgs& sum, int t, float* outBuf, bool whole, int part) {
  if (faultOn(Fault::nowrite, rank_)) return;
  const int L = localRanks_, me = rank_ % L;
  RsArea& area = rsAreas_[plan.area];
  const size_t sub = plan.subElems, stg = plan.stgElems;
  auto range = [&](int o) -> std::pair<size_t, size_t> {  // owner o's staging elements of this round
    if (whole) return {0, stg};
    const size_t s0 = static_cast<size_t>(rsSubOf(o, t)) * sub, s1 = std::min(stg, s0 + sub);
    const size_t b = std::min(s1, s0 + static_cast<size_t>(part) * plan.partElems);
    return {b, std::min(s1, b + plan.partElems)};
  };
  const auto [sub0, sub1] = range(me);
  char* mine = static_cast<char*>(plan.arena->base);
  sum.srcStride = whole ? plan.chunkBytes : static_cast<size_t>(L) * plan.chunkBytes;  // staging [source][dest][chunk]
  sum.destStride = whole ? 0 : plan.chunkBytes;
  sum.nDest = whole ? 1 : L;
  sum.outDestStride = whole ? 0 : plan.partElems;
  sum.destRank = whole ? rank_ : t * L;
  sum.finalMode = whole ? 1 : 0;
  const int nSlots = cfg_.rsStageSlots;
  for (int c = 0; c < plan.chunks; ++c) {
    uint64_t pushes = 0;
    for (int i = 0; i < nSlots; ++i) pushes += area.slotUses[i];
    const int slot = static_cast<int>(pushes % static_cast<uint64_t>(nSlots));
    const uint64_t su = ++area.slotUses[slot];
    // nofree: one stream, or a slot's land flags (absolute values) from two chunks could land out of
    // order and hang the owners instead of corrupting their sums
    const bool nofree = faultOn(Fault::nofree, rank_);
    const cudaStream_t ce = rsCeStreams_[nofree ? 0 : (pushes + 1) % kRsCeStreams];
    const size_t c0 = std::min(sub1, sub0 + static_cast<size_t>(c) * plan.chunkElems), c1 = std::min(sub1, c0 + plan.chunkElems);
    if (su > 1 && !nofree) waitFlags(*rsSync_, rsFlagOff(plan.area, kFree, slot), su - 1, ce);
    if (faultOn(Fault::slowpush, rank_)) faultDelay(ce);
    PushBatch push;
    for (int i = 1; i < L; ++i) {
      const int o = (me + i) % L;
      const auto [oSub0, oSub1] = range(o);
      const size_t o0 = std::min(oSub1, oSub0 + static_cast<size_t>(c) * plan.chunkElems), o1 = std::min(oSub1, o0 + plan.chunkElems);
      if (o0 >= o1) continue;
      char* theirs = static_cast<char*>(plan.arena->peer[node_ * L + o]);
      TIDE_CHECK(theirs != nullptr, "local peer without a mapped window");
      char* dstSlot = theirs + plan.stageBase + static_cast<size_t>(slot) * plan.slotBytes + static_cast<size_t>(me) * sum.srcStride;
      // one strided descriptor per segment: the L destinations t*L .. t*L+L-1 are consecutive ranks
      // (source pitch = piece), landing in the slot's [dest] rows (pitch = chunk); whole: one row, rank o
      for (size_t p = 0; p < plan.segs.size(); ++p) {
        const RsSegment& sg = plan.segs[p];
        const size_t lo = std::max(o0, sg.stgOff), hi = std::min(o1, sg.stgOff + sg.pieceElems);
        if (lo >= hi) continue;
        addPieceRows(push, dstSlot + (lo - o0) * plan.itemsize, plan.chunkBytes, static_cast<const char*>(sum.src[p]),
                     static_cast<size_t>(t * L + (whole ? o : 0)) * sg.pieceElems + (lo - sg.stgOff), sg.pieceElems, hi - lo,
                     whole ? 1 : static_cast<size_t>(L), sg.numel, plan.itemsize);
      }
    }
    batchCopy(push, ce);
    writeFlags(*rsSync_, rsFlagOff(plan.area, kLand, slot), su, ce);
    if (!faultOn(Fault::nofence, rank_)) waitFlags(*rsSync_, rsFlagOff(plan.area, kLand, slot), su, rsSumStream_);
    if (faultOn(Fault::slowsum, rank_)) faultDelay(rsSumStream_);
    sum.stage = mine + plan.stageBase + static_cast<size_t>(slot) * plan.slotBytes;
    sum.chunkBegin = c0;
    sum.chunkEnd = c1;
    sum.subBegin = sub0;
    sum.out = outBuf;
    launchRsSum(sum, cfg_.rsCtas, rsSumStream_);
    writeFlags(*rsSync_, rsFlagOff(plan.area, kFree, slot), su, rsSumStream_);
  }
}

// Single node: one level-1 round over my whole destination, straight into `out`.
void Comm::issueRsLocal(RsPlan& plan, const std::vector<const void*>& src, float* out, float scale, bool accumulate,
                        cudaStream_t s) {
  RsSumArgs sum = sumArgs(plan, src, localRanks_, rank_ % localRanks_, scale, accumulate);
  rsFork(s);
  rsLevel1(plan, sum, node_, out, true);
  for (int i = 0; i < kRsCeStreams; ++i) rsMarkRead(rsCeStreams_[i], i);
  rsMarkRead(rsSumStream_, kRsCeStreams);
  rsJoin(s);
  if (!faultOn(Fault::nowrite, rank_)) ++rsAreas_[plan.area].uses;
}

// Multi-node: the multicast landing for any node count (RsPlan::Wave). Per NIC round
// (plan.kRounds): the round's destinations are summed on rsSumStream_ (level 1) into outbox ring
// blocks, put to their rail peers in one group and the same round's partials for me arrive in the
// inbox blocks of the current wave's parity. When a wave's rounds (and, for the last wave, my own
// node's partial: summed into my block, or from wave 2 on into a spare block copied in on `s`) are in, every local rank raises
// waveReady[parity]; the landing stream waits the local ready event, lands the wave with one
// 4-CTA multimem kernel accumulating into `out`, raises waveDone[parity] and records the done
// event. Buffer reuse and pacing share one `release` signal per (source, destination) per call:
// after a round's heads landed I release the next round's sources -- first waiting, when that
// round opens a wave whose parity is being reused, for my landing of the parity's previous wave
// (the CUDA event, so the memop wait below has a CUDA-visible predecessor) and for the eight
// local owners' waveDone flags; at the call's end I release the next call's first-round sources.
// Blocks a wave does not feed but the parity's previous wave did are zeroed on rsZeroStream_ after
// that wave's landing, before this wave's landing waits for them. All NCCL calls stay on `s`; the
// landing stream carries no memop waits (only events), and every kernel a memop wait depends on is
// ordered before that wait by an event -- CUDA's own caveat for cuStreamBatchMemOp.
// With plan.parts passes the rounds restart at destination 1 in every part (RsPlan::Wave::part)
// and the waves, parities and credits run on along the call.
void Comm::issueRsMc(RsPlan& plan, const std::vector<const void*>& src, float* out, float scale, bool accumulate,
                     cudaStream_t s) {
  const int M = nNodes_, n = node_, L = localRanks_, me = rank_ % L, O = plan.outboxSlots, D = M - 1;
  RsArea& area = rsAreas_[plan.area];
  const bool skipData = faultOn(Fault::nowrite, rank_);
  char* mine = static_cast<char*>(plan.arena->base);
  const size_t blockStride = plan.outStride, sub = plan.subElems;
  const uint64_t u = area.uses + 1;    // this call's use counter
  const uint64_t seq0 = area.waveSeq;  // wave w of this call has sequence seq0 + w + 1
  const int nW = static_cast<int>(plan.waves.size());
  const int nR = static_cast<int>(plan.kRounds.size());
  // round ri's first destination index j and part; J = part * D + j indexes the outbox ring and the events
  std::vector<int> firstJ(nR + 1, 1), partOf(nR + 1, 0);
  for (int w = 0, j = 1, q = 0; w < nW; ++w) {
    if (plan.waves[w].part != q) {
      q = plan.waves[w].part;
      j = 1;
    }
    for (int ri = plan.waves[w].firstRound; ri < plan.waves[w].firstRound + plan.waves[w].rounds; ++ri) {
      firstJ[ri] = j;
      partOf[ri] = q;
      j += plan.kRounds[ri];
    }
  }
  for (auto* v : {&rsSumEv_, &rsPutEv_})
    for (v->reserve(static_cast<size_t>(D * plan.parts + 1)); v->size() < static_cast<size_t>(D * plan.parts + 1);) {
      v->push_back(nullptr);
      CUDACHECK(cudaEventCreateWithFlags(&v->back(), cudaEventDisableTiming));
    }
  rsFork(s);
  RsSumArgs sum = sumArgs(plan, src, L, me, scale, accumulate);
  auto block = [&](size_t base, int parity, int p) {
    return reinterpret_cast<float*>(mine + base + (static_cast<size_t>(parity) * L + static_cast<size_t>(p)) * blockStride);
  };

  // Owner side of wave w: multimem sum of block p, sub-block me, over the local team for every block,
  // accumulated into out (a part's first wave applies the caller's scale/accumulate).
  auto land = [&](int w) {
    nvtx::Range landRange("tide.rs.land", nvtx::kRsLand);
    const RsPlan::Wave& wv = plan.waves[w];
    if (faultOn(Fault::slow, rank_)) faultDelay(rsAddStream_);
    RsMcArgs a{};
    a.segs = plan.segsDev;
    a.nseg = static_cast<int>(plan.segs.size());
    const size_t parityOff =
        plan.inboxBase + static_cast<size_t>(wv.parity) * L * blockStride + static_cast<size_t>(me) * plan.partElems * 4;
    a.mc = reinterpret_cast<const float*>(static_cast<char*>(plan.arena->multimem) + parityOff);
    a.blockStride = blockStride / 4;
    a.subElems = sub;
    a.stgElems = plan.stgElems;
    a.partBegin = static_cast<size_t>(wv.part) * plan.partElems;
    a.partElems = plan.partElems;
    a.blocks = L;
    a.only = -1;
    a.out = out;
    a.finalMode = 1;
    a.scale = scale;
    a.accumulate = (w % (nW / plan.parts) > 0 || accumulate) ? 1 : 0;  // a part's first wave writes its elements
    if (cfg_.rsMcCtas > 0) launchRsMc(a, cfg_.rsMcCtas, rsAddStream_);
  };
  // The parity of wave w may be overwritten (by wave w + 2's sources) once every local owner landed
  // wave w: my own landing by its event (CUDA-visible), the team's by their waveDone flags.
  auto waitConsumed = [&](int w) {
    const RsPlan::Wave& wv = plan.waves[w];
    CUDACHECK(cudaStreamWaitEvent(s, rsDoneEv_[wv.parity], 0));
    if (!faultOn(Fault::nodone, rank_))
      waitFlags(*rsSync_, rsFlagOff(plan.area, kWaveDone, wv.parity), seq0 + static_cast<uint64_t>(w) + 1, s);
  };
  // Wave w complete on my side (its rounds landed on s; own partial summed for the last wave): raise
  // waveReady, wait the team's, hand the landing to rsAddStream_ by event, raise waveDone there.
  int spare = 0;
  auto finishWave = [&](int w) {
    const RsPlan::Wave& wv = plan.waves[w];
    const uint64_t seq = seq0 + static_cast<uint64_t>(w) + 1;
    if (wv.first + wv.count - 1 == M) {  // my own node's partial into block `me` of this parity
      // from w = 2 on the block is read by every local owner's landing of wave w-2 (multimem): sum into a spare
      // and copy it in on s, behind the wait for those landings, so the sums (the read event) never wait for one
      float* own = block(plan.inboxBase, wv.parity, me);
      if (w >= 2) {
        TIDE_CHECK(spare < plan.ownSpares, "own-node spare blocks exhausted");
        own = reinterpret_cast<float*>(mine + plan.ownBase + static_cast<size_t>(spare++) * blockStride);
      }
      rsLevel1(plan, sum, n, own, false, wv.part);
      CUDACHECK(cudaEventRecord(rsSumEv_[0], rsSumStream_));
      CUDACHECK(cudaStreamWaitEvent(s, rsSumEv_[0], 0));
      if (w >= 2) {
        if (wv.rounds == 0) waitConsumed(w - 2);  // else the round before the wave's first one waited (beforeRelease)
        CUDACHECK(cudaMemcpyAsync(block(plan.inboxBase, wv.parity, me), own, static_cast<size_t>(L) * plan.partElems * 4,
                                  cudaMemcpyDeviceToDevice, s));
      }
    }
    if (wv.zero) CUDACHECK(cudaStreamWaitEvent(s, rsZeroEv_[wv.parity], 0));  // the zeroing issued for this wave (w < 2: at the previous call's end)
    writeFlags(*rsSync_, rsFlagOff(plan.area, kWaveReady, wv.parity), seq, s);
    if (!faultOn(Fault::nofence, rank_)) waitFlags(*rsSync_, rsFlagOff(plan.area, kWaveReady, wv.parity), seq, s);
    CUDACHECK(cudaEventRecord(rsReadyEv_[wv.parity], s));
    CUDACHECK(cudaStreamWaitEvent(rsAddStream_, rsReadyEv_[wv.parity], 0));
    land(w);
    writeFlags(*rsSync_, rsFlagOff(plan.area, kWaveDone, wv.parity), seq, rsAddStream_);
    CUDACHECK(cudaEventRecord(rsDoneEv_[wv.parity], rsAddStream_));
    rsDebugSync(u, "wave landed", rsAddStream_);
    if (w + 2 < nW && plan.waves[w + 2].zero) {  // blocks this wave fed but wave w+2 will not: zero once every owner landed them
      CUDACHECK(cudaStreamWaitEvent(rsZeroStream_, rsDoneEv_[wv.parity], 0));
      if (!faultOn(Fault::nodone, rank_)) waitFlags(*rsSync_, rsFlagOff(plan.area, kWaveDone, wv.parity), seq, rsZeroStream_);
      if (!faultOn(Fault::nozero, rank_))
        for (int p = 0; p < L; ++p)
          if (plan.waves[w + 2].zero >> p & 1)
            CUDACHECK(cudaMemsetAsync(block(plan.inboxBase, wv.parity, p), 0, blockStride, rsZeroStream_));
      CUDACHECK(cudaEventRecord(rsZeroEv_[wv.parity], rsZeroStream_));
    }
  };
  auto waveOfRound = [&](int ri) {
    for (int w = 0; w < nW; ++w)
      if (ri >= plan.waves[w].firstRound && ri < plan.waves[w].firstRound + plan.waves[w].rounds) return w;
    return -1;
  };

  for (int ri = 0; ri < nR && !skipData; ++ri) {
    const int Kr = plan.kRounds[ri], r0 = firstJ[ri], r1 = r0 + Kr, q = partOf[ri], w = waveOfRound(ri);
    for (int j = r0; j < r1; ++j) {
      const int J = q * D + j;
      if (J > O && !faultOn(Fault::noputwait, rank_))
        CUDACHECK(cudaStreamWaitEvent(rsSumStream_, rsPutEv_[J - O], 0));  // outbox block reuse
      rsLevel1(plan, sum, (n + j) % M, block(plan.outboxBase, 0, J % O), false, q);
      CUDACHECK(cudaEventRecord(rsSumEv_[J], rsSumStream_));
    }
    rsDebugSync(u, "sum round", rsSumStream_);
    std::vector<PutSpec> specs;
    std::vector<std::array<int, 4>> releases;
    for (int j = r0; j < r1; ++j) {
      const int J = q * D + j, t = (n + j) % M, to = railPeer(t), from = railPeer((n - j + M) % M), pMe = rsSubOf(me, t);
      CUDACHECK(cudaStreamWaitEvent(s, rsSumEv_[J], 0));
      specs.push_back({plan.outboxBase + static_cast<size_t>(J % O) * blockStride,
                       plan.inboxBase + (static_cast<size_t>(plan.waves[w].parity) * L + static_cast<size_t>(pMe)) * blockStride, to, from,
                       &plan.nicChunks});
      if ((ri > 0 || area.uses > 0) && !faultOn(Fault::nocredit, rank_)) releases.push_back({1, to, rsSigCredit(plan.area), 0});
    }
    waitSignal(releases, s);  // my targets are ready for this round (paced, and its parity consumed)
    std::vector<int> releaseTo;
    std::function<void()> beforeRelease;
    if (ri + 1 < nR) {
      for (int j = firstJ[ri + 1]; j < firstJ[ri + 1] + plan.kRounds[ri + 1]; ++j) releaseTo.push_back(railPeer((n - j + M) % M));
      const int wNext = waveOfRound(ri + 1);
      if (wNext != w && wNext >= 2) beforeRelease = [&, wNext] { waitConsumed(wNext - 2); };  // next round opens a reused parity
    }
    putRounds(*plan.arena, specs, releaseTo.empty() || !plan.paced ? 0 : plan.paceTail, rsSigData(), releaseTo, rsSigCredit(plan.area), s,
              beforeRelease);
    for (int j = r0; j < r1; ++j) CUDACHECK(cudaEventRecord(rsPutEv_[q * D + j], s));
    rsDebugSync(u, "nic round", s);
    if (w >= 0 && ri == plan.waves[w].firstRound + plan.waves[w].rounds - 1) finishWave(w);
  }
  if (!skipData) {
    for (int w = 0; w < nW; ++w)
      if (plan.waves[w].rounds == 0) finishWave(w);  // a wave of my own node only
    // every wave landed here and on the team: the next call's first round may overwrite my inbox
    for (int w = std::max(0, nW - 2); w < nW; ++w) waitConsumed(w);
    // blocks the parities' last waves fed but their first waves (next call) will not: zero them now
    CUDACHECK(cudaEventRecord(rsForkEv_, s));
    for (int w = 0; w < std::min(nW, plan.inboxParities); ++w) {
      if (!plan.waves[w].zero) continue;
      CUDACHECK(cudaStreamWaitEvent(rsZeroStream_, rsForkEv_, 0));
      if (!faultOn(Fault::nozero, rank_))
        for (int p = 0; p < L; ++p)
          if (plan.waves[w].zero >> p & 1)
            CUDACHECK(cudaMemsetAsync(block(plan.inboxBase, plan.waves[w].parity, p), 0, blockStride, rsZeroStream_));
      CUDACHECK(cudaEventRecord(rsZeroEv_[plan.waves[w].parity], rsZeroStream_));
    }
  }
  // every rsLevel1 (the only reader of src) is issued
  for (int i = 0; i < kRsCeStreams; ++i) rsMarkRead(rsCeStreams_[i], i);
  rsMarkRead(rsSumStream_, kRsCeStreams);
  rsJoin(s);
  if (!skipData) {
    Group grp;
    for (int j = 1; j <= (plan.kRounds.empty() ? 0 : plan.kRounds[0]); ++j) signal(railPeer((n - j + M) % M), rsSigCredit(plan.area), /*ctx=*/0, s);
    grp.end();
    ++area.uses;
    area.waveSeq += static_cast<uint64_t>(nW);
  }
}

// TIDE_RS_PIPELINE=2: issueRsMc's rounds, waves, sums and landings issued as two phases. Read phase: the first
// outboxBank level-1 sums of the call (a full bank: every remote destination's) into their outbox blocks and each
// own-node partial into its spare on the CE / sum streams, which wait for readyEv (not for s). Network phase, on s
// behind the previous call's network phase and gateEv: rounds (when outboxBank < dests, a round's put releases its
// outbox blocks to the sum of destination J + outboxBank; readEv is recorded once the last sum is queued), own spares
// copied into the inbox, landings, zeroing, the next call's credits. Outbox blocks, spares and their events come in
// two banks alternating by the communicator's call count; a bank's first outboxBank blocks are summed into again once
// the call two back has put them out (a spare: copied it in), so the initial reads of call e+1 run during the network
// phase of call e. Every wait's local producer is issued earlier on this thread.
void Comm::issueRsMcPipelined(RsPlan& plan, const std::vector<const void*>& src, float* out, float scale,
                              bool accumulate, cudaStream_t s, cudaEvent_t readEv, cudaEvent_t readyEv,
                              cudaEvent_t gateEv) {
  const int M = nNodes_, n = node_, L = localRanks_, me = rank_ % L, Ob = plan.outboxBank, D = M - 1;
  const int dests = D * plan.parts;
  RsArea& area = rsAreas_[plan.area];
  const bool skipData = faultOn(Fault::nowrite, rank_);
  char* mine = static_cast<char*>(plan.arena->base);
  const size_t blockStride = plan.outStride;
  // this call's snapshot: the network phase reads these, never the cursors advanced at its end
  const uint64_t u = area.uses + 1;
  const bool hadPrevious = area.uses > 0;  // the first call's first round has no credit to wait for
  const uint64_t seq0 = area.waveSeq;      // wave w of this call has sequence seq0 + w + 1
  const int bank = static_cast<int>(rsEpoch_ % 2);
  const bool reuse = rsBankUsed_[bank];
  const int nW = static_cast<int>(plan.waves.size());
  const int nR = static_cast<int>(plan.kRounds.size());
  const int maxRound = plan.kRounds.empty() ? 0 : *std::max_element(plan.kRounds.begin(), plan.kRounds.end());
  TIDE_CHECK(Ob >= maxRound && Ob <= dests, "pipelined reduce-scatter outbox bank out of range");
  std::vector<int> firstJ(nR + 1, 1), partOf(nR + 1, 0);
  for (int w = 0, j = 1, q = 0; w < nW; ++w) {
    if (plan.waves[w].part != q) {
      q = plan.waves[w].part;
      j = 1;
    }
    for (int ri = plan.waves[w].firstRound; ri < plan.waves[w].firstRound + plan.waves[w].rounds; ++ri) {
      firstJ[ri] = j;
      partOf[ri] = q;
      j += plan.kRounds[ri];
    }
  }
  auto grow = [](std::vector<cudaEvent_t>& v, size_t k) {
    for (v.reserve(k); v.size() < k;) {
      v.push_back(nullptr);
      CUDACHECK(cudaEventCreateWithFlags(&v.back(), cudaEventDisableTiming));
    }
  };
  grow(rsSumEvB_[bank], static_cast<size_t>(dests + 1));
  grow(rsPutEvB_[bank], static_cast<size_t>(Ob));
  grow(rsOwnEvB_[bank], static_cast<size_t>(std::max(plan.ownWaves, 1)));
  grow(rsSpareEvB_[bank], static_cast<size_t>(std::max(plan.ownWaves, 1)));
  auto block = [&](size_t base, int parity, int p) {
    return reinterpret_cast<float*>(mine + base + (static_cast<size_t>(parity) * L + static_cast<size_t>(p)) * blockStride);
  };
  auto outOff = [&](int J) {
    return plan.outboxBase + (static_cast<size_t>(bank) * Ob + static_cast<size_t>(J % Ob)) * blockStride;
  };
  auto spare = [&](int k) {
    return reinterpret_cast<float*>(mine + plan.ownBase +
                                    (static_cast<size_t>(bank) * plan.ownWaves + static_cast<size_t>(k)) * blockStride);
  };
  std::vector<int> ownOf(nW, -1);  // spare index of each wave holding my own node
  for (int w = 0, k = 0; w < nW; ++w)
    if (plan.waves[w].first + plan.waves[w].count - 1 == M) ownOf[w] = k++;

  // ---- read phase
  auto phase = std::make_unique<nvtx::Range>("tide.rs.read", nvtx::kRsRead);
  const double prof0 = g_prof.on ? nowUs() : 0;
  double profUs0[RsHostProf::kCats];
  std::copy(g_prof.us, g_prof.us + RsHostProf::kCats, profUs0);
  RsSumArgs sum = sumArgs(plan, src, L, me, scale, accumulate);
  for (cudaStream_t st : rsCeStreams_) CUDACHECK(cudaStreamWaitEvent(st, readyEv, 0));
  CUDACHECK(cudaStreamWaitEvent(rsSumStream_, readyEv, 0));
  const bool earlyRead = readEv && faultOn(Fault::earlyread, rank_);
  if (earlyRead) {  // negative test: the read event before any read
    CUDACHECK(cudaStreamWaitEvent(rsReadStream_, readyEv, 0));
    CUDACHECK(cudaEventRecord(readEv, rsReadStream_));
  }
  rsReadWanted_ = readEv != nullptr && !earlyRead;
  auto issueSum = [&](int J) {
    const int q = (J - 1) / D, j = (J - 1) % D + 1;
    // J <= Ob: put out by the call two back; J > Ob: put out by destination J - Ob of this call
    if ((reuse || J > Ob) && !faultOn(Fault::noputwait, rank_))
      CUDACHECK(cudaStreamWaitEvent(rsSumStream_, rsPutEvB_[bank][J % Ob], 0));
    rsLevel1(plan, sum, (n + j) % M, reinterpret_cast<float*>(mine + outOff(J)), false, q);
    CUDACHECK(cudaEventRecord(rsSumEvB_[bank][J], rsSumStream_));
  };
  auto finishReads = [&] {
    for (int i = 0; i < kRsCeStreams; ++i) rsMarkRead(rsCeStreams_[i], i);
    rsMarkRead(rsSumStream_, kRsCeStreams);
    if (rsReadWanted_) CUDACHECK(cudaEventRecord(readEv, rsReadStream_));
    rsReadWanted_ = false;
    rsDebugSync(u, "read phase", rsSumStream_);
  };
  if (!skipData) {
    for (int J = 1; J <= std::min(Ob, dests); ++J) issueSum(J);
    for (int w = 0; w < nW; ++w) {
      if (ownOf[w] < 0) continue;
      if (reuse) CUDACHECK(cudaStreamWaitEvent(rsSumStream_, rsSpareEvB_[bank][ownOf[w]], 0));  // copied in two calls back
      rsLevel1(plan, sum, n, spare(ownOf[w]), false, plan.waves[w].part);
      CUDACHECK(cudaEventRecord(rsOwnEvB_[bank][ownOf[w]], rsSumStream_));
    }
  }
  if (skipData || Ob >= dests) finishReads();
  if (g_prof.on) {
    g_prof.read += nowUs() - prof0;
    for (int c = 0; c < RsHostProf::kCats; ++c) g_prof.readUs[c] += g_prof.us[c] - profUs0[c];
  }

  // ---- network phase
  phase.reset();
  phase = std::make_unique<nvtx::Range>("tide.rs.net", nvtx::kRsNet);
  auto afterRoundPuts = [&](int ri, cudaStream_t putStream) {
    const int r0 = firstJ[ri], r1 = r0 + plan.kRounds[ri], q = partOf[ri];
    for (int j = r0; j < r1; ++j) CUDACHECK(cudaEventRecord(rsPutEvB_[bank][(q * D + j) % Ob], putStream));
    for (int j = r0; j < r1; ++j) {
      const int J = q * D + j + Ob;
      if (J <= dests) {
        issueSum(J);
        if (J == dests) finishReads();
      }
    }
  };
  if (rsOrdered_) CUDACHECK(cudaStreamWaitEvent(s, rsOrderEv_, 0));
  if (gateEv) CUDACHECK(cudaStreamWaitEvent(s, gateEv, 0));
  auto land = [&](int w) {
    nvtx::Range landRange("tide.rs.land", nvtx::kRsLand);
    const RsPlan::Wave& wv = plan.waves[w];
    if (faultOn(Fault::slow, rank_)) faultDelay(rsAddStream_);
    RsMcArgs a{};
    a.segs = plan.segsDev;
    a.nseg = static_cast<int>(plan.segs.size());
    const size_t parityOff =
        plan.inboxBase + static_cast<size_t>(wv.parity) * L * blockStride + static_cast<size_t>(me) * plan.partElems * 4;
    a.mc = reinterpret_cast<const float*>(static_cast<char*>(plan.arena->multimem) + parityOff);
    a.blockStride = blockStride / 4;
    a.subElems = plan.subElems;
    a.stgElems = plan.stgElems;
    a.partBegin = static_cast<size_t>(wv.part) * plan.partElems;
    a.partElems = plan.partElems;
    a.blocks = L;
    a.only = -1;
    a.out = out;
    a.finalMode = 1;
    a.scale = scale;
    a.accumulate = (w % (nW / plan.parts) > 0 || accumulate) ? 1 : 0;  // a part's first wave writes its elements
    if (cfg_.rsMcCtas > 0) launchRsMc(a, cfg_.rsMcCtas, rsAddStream_);
  };
  auto waitConsumed = [&](int w) {
    const RsPlan::Wave& wv = plan.waves[w];
    CUDACHECK(cudaStreamWaitEvent(s, rsDoneEv_[wv.parity], 0));
    if (!faultOn(Fault::nodone, rank_))
      waitFlags(*rsSync_, rsFlagOff(plan.area, kWaveDone, wv.parity), seq0 + static_cast<uint64_t>(w) + 1, s);
  };
  auto finishWave = [&](int w) {
    const RsPlan::Wave& wv = plan.waves[w];
    const uint64_t seq = seq0 + static_cast<uint64_t>(w) + 1;
    if (ownOf[w] >= 0) {  // my own node's partial, summed into its spare in the read phase
      CUDACHECK(cudaStreamWaitEvent(s, rsOwnEvB_[bank][ownOf[w]], 0));
      // block `me` of this parity is free once every local owner landed the parity's previous wave: w >= 2 waited
      // before the wave's first round (beforeRelease) unless it has none; w < 2: the previous call's network phase
      // ended waiting for its last two waves, and this one runs behind it on s
      if (w >= 2 && wv.rounds == 0) waitConsumed(w - 2);
      CUDACHECK(cudaMemcpyAsync(block(plan.inboxBase, wv.parity, me), spare(ownOf[w]),
                                static_cast<size_t>(L) * plan.partElems * 4, cudaMemcpyDeviceToDevice, s));
      CUDACHECK(cudaEventRecord(rsSpareEvB_[bank][ownOf[w]], s));
    }
    if (wv.zero) CUDACHECK(cudaStreamWaitEvent(s, rsZeroEv_[wv.parity], 0));
    writeFlags(*rsSync_, rsFlagOff(plan.area, kWaveReady, wv.parity), seq, s);
    if (!faultOn(Fault::nofence, rank_)) waitFlags(*rsSync_, rsFlagOff(plan.area, kWaveReady, wv.parity), seq, s);
    CUDACHECK(cudaEventRecord(rsReadyEv_[wv.parity], s));
    CUDACHECK(cudaStreamWaitEvent(rsAddStream_, rsReadyEv_[wv.parity], 0));
    land(w);
    writeFlags(*rsSync_, rsFlagOff(plan.area, kWaveDone, wv.parity), seq, rsAddStream_);
    CUDACHECK(cudaEventRecord(rsDoneEv_[wv.parity], rsAddStream_));
    rsDebugSync(u, "wave landed", rsAddStream_);
    if (w + 2 < nW && plan.waves[w + 2].zero) {
      CUDACHECK(cudaStreamWaitEvent(rsZeroStream_, rsDoneEv_[wv.parity], 0));
      if (!faultOn(Fault::nodone, rank_)) waitFlags(*rsSync_, rsFlagOff(plan.area, kWaveDone, wv.parity), seq, rsZeroStream_);
      if (!faultOn(Fault::nozero, rank_))
        for (int p = 0; p < L; ++p)
          if (plan.waves[w + 2].zero >> p & 1)
            CUDACHECK(cudaMemsetAsync(block(plan.inboxBase, wv.parity, p), 0, blockStride, rsZeroStream_));
      CUDACHECK(cudaEventRecord(rsZeroEv_[wv.parity], rsZeroStream_));
    }
  };
  auto waveOfRound = [&](int ri) {
    for (int w = 0; w < nW; ++w)
      if (ri >= plan.waves[w].firstRound && ri < plan.waves[w].firstRound + plan.waves[w].rounds) return w;
    return -1;
  };
  const bool devctl = cfg_.rsDevctl != 0 && !skipData;
  if (devctl) {
    // Every wave's rounds in one GIN device kernel W on rsDevStream_; s waits for each W's event, then lands the wave as
    // before. The round tables (destinations, block offsets, absolute signal counts) are built here in call order, the
    // same on every rank, and travel in the kernel parameters.
    ensureDevComm();
    TIDE_CHECK(plan.area < kDcMaxAreas, "TIDE_RS_NET=device: more reduce-scatter areas than the credit signals cover");
    TIDE_CHECK(static_cast<int>(plan.nicChunks.size()) <= kDcMaxChunks, "TIDE_RS_NET=device: more NIC chunks than the tables hold");
    TIDE_CHECK(*std::max_element(plan.kRounds.begin(), plan.kRounds.end()) <= kDcMaxK,
               "TIDE_RS_NET=device: NIC rounds wider than " + std::to_string(kDcMaxK) + " destinations (TIDE_RS_KWAY)");
    for (int w = 0; w + 1 < nW; ++w)  // own-node-only waves land after every network wave
      TIDE_CHECK(plan.waves[w].rounds > 0 || plan.waves[w + 1].rounds == 0, "TIDE_RS_NET=device: an own-node-only wave before a network wave");
    const int C = dcCtx_, nCh = static_cast<int>(plan.nicChunks.size()), active = std::min(nCh, C);
    DcWaveArgs base{};
    base.devComm = devComm_;
    base.win = plan.arena->railWin;  // the GIN device communicator is the rail one
    base.me = n;
    base.ctx = C;
    base.nChunks = nCh;
    for (int c = 0; c < nCh; ++c) {
      base.chunkOff[c] = plan.nicChunks[c].first;
      base.chunkBytes[c] = plan.nicChunks[c].second;
    }
    base.sigData0 = 0;
    base.sigCredit0 = M * C + plan.area * M;
    base.err = dcErrDev_;
    base.epoch = static_cast<int>(rsEpoch_);
    base.timeoutNs = static_cast<uint64_t>(cfg_.rsDevctlTimeoutS * 1.0e9);
    base.flushCycles = static_cast<uint64_t>(cfg_.rsDevctlTimeoutS * 2.0e9);
    base.debug = cfg_.rsDevctlDebug;
    auto sources = [&](int ri, int* to) {
      int k = 0;
      for (int j = firstJ[ri]; j < firstJ[ri] + plan.kRounds[ri]; ++j) to[k++] = (n - j + M) % M;
      return k;
    };
    // behind everything already on s: this call's early all-gather, the previous call's tail
    CUDACHECK(cudaEventRecord(dcGateEv_, s));
    CUDACHECK(cudaStreamWaitEvent(rsDevStream_, dcGateEv_, 0));
    auto& waveEv = dcWaveEvB_[bank];
    grow(waveEv, static_cast<size_t>(std::max(nW, 1)));
    const size_t traceSlot = static_cast<size_t>(rsEpoch_ % kDcTraceCalls);
    uint32_t traceMask = 0;
    for (int w = 0; w < nW; ++w) {
      const RsPlan::Wave& wv = plan.waves[w];
      if (wv.rounds == 0) continue;
      nvtx::Range waveRange("tide.rs.wave", nvtx::kRsNet);
      TIDE_CHECK(wv.rounds <= kDcMaxRounds, "TIDE_RS_NET=device: more rounds in a wave than the tables hold");
      const int ri0 = wv.firstRound;
      const int step = Ob < dests ? 1 : wv.rounds;  // partial bank: flush each round so J + Ob can reuse its blocks
      for (int r0 = ri0; r0 < ri0 + wv.rounds; r0 += step) {
        const int nRnds = std::min(step, ri0 + wv.rounds - r0);
        DcWaveArgs a = base;
        a.wave = w;
        a.round0 = r0 - ri0;
        a.waveRounds = wv.rounds;
        if (dcTraceDev_ && w < kDcTraceWaves) {
          a.trace = &dcTraceDev_[traceSlot * kDcTraceWaves + static_cast<size_t>(w)];
          traceMask |= 1u << w;
        }
        if (r0 == ri0 && ri0 > 0) {  // the release the host path sends after round ri0 - 1: to this wave's first sources
          if (w >= 2) {  // they write the parity of wave w - 2: my landing of it (event), the team's (device-side)
            CUDACHECK(cudaStreamWaitEvent(rsDevStream_, rsDoneEv_[plan.waves[w - 2].parity], 0));
            if (!faultOn(Fault::nodone, rank_)) {
              a.doneBase = reinterpret_cast<const uint64_t*>(static_cast<char*>(rsSync_->base) +
                                                             rsFlagOff(plan.area, kWaveDone, plan.waves[w - 2].parity));
              a.nDone = static_cast<int>(lsaPeers_.size());
              for (int i = 0; i < a.nDone; ++i) a.donePeer[i] = lsaPeers_[i];
              a.doneSeq = seq0 + static_cast<uint64_t>(w - 2) + 1;
            }
          }
          a.nPro = sources(ri0, a.proTo);
        }
        a.nRounds = nRnds;
        for (int ri = r0; ri < r0 + nRnds; ++ri) {
          DcRound& R = a.r[ri - r0];
          R.k = plan.kRounds[ri];
          TIDE_CHECK(R.k <= kDcMaxK, "TIDE_RS_NET=device: rounds wider than the tables hold");
          for (int i = 0; i < R.k; ++i) {
            const int j = firstJ[ri] + i, J = partOf[ri] * D + j, t = (n + j) % M, f = (n - j + M) % M, pMe = rsSubOf(me, t);
            CUDACHECK(cudaStreamWaitEvent(rsDevStream_, rsSumEvB_[bank][J], 0));
            R.to[i] = t;
            R.from[i] = f;
            const uint64_t credit = ri > 0 || hadPrevious ? ++dcCreditExp_[static_cast<size_t>(plan.area) * M + t] : 0;
            R.creditThr[i] = faultOn(Fault::nocredit, rank_) ? 0 : credit;
            R.srcOff[i] = outOff(J);
            R.dstOff[i] = plan.inboxBase + (static_cast<size_t>(wv.parity) * L + static_cast<size_t>(pMe)) * blockStride;
            for (int x = 0; x < active; ++x) R.dataThr[i][x] = ++dcDataExp_[static_cast<size_t>(f) * C + x];
          }
          R.nRel = ri + 1 < ri0 + wv.rounds ? sources(ri + 1, R.relTo) : 0;
        }
        if (a.trace) {
          DcTracePeers& tp = dcTracePeers_[traceSlot * kDcTraceWaves + static_cast<size_t>(w)];
          for (int r = 0; r < a.nRounds; ++r) {
            tp.k[a.round0 + r] = static_cast<int8_t>(a.r[r].k);
            for (int i = 0; i < a.r[r].k; ++i) {
              tp.to[a.round0 + r][i] = static_cast<int8_t>(a.r[r].to[i]);
              tp.from[a.round0 + r][i] = static_cast<int8_t>(a.r[r].from[i]);
            }
          }
        }
        if (!(faultOn(Fault::dcstall, rank_) && rsEpoch_ > 0)) launchDcWave(a, rsDevStream_);
        for (int ri = r0; ri < r0 + nRnds; ++ri) afterRoundPuts(ri, rsDevStream_);
      }
      CUDACHECK(cudaEventRecord(waveEv[w], rsDevStream_));
      CUDACHECK(cudaStreamWaitEvent(s, waveEv[w], 0));
      finishWave(w);
    }
    if (dcTraceHost_) {
      dcTraceMask_[traceSlot] = traceMask;
      if (rsEpoch_ >= kDcTraceLag) dcTraceConsume(rsEpoch_ - kDcTraceLag);
    }
  }
  for (int ri = 0; ri < nR && !skipData && !devctl; ++ri) {
    const int Kr = plan.kRounds[ri], r0 = firstJ[ri], r1 = r0 + Kr, q = partOf[ri], w = waveOfRound(ri);
    std::vector<PutSpec> specs;
    std::vector<std::array<int, 4>> releases;
    for (int j = r0; j < r1; ++j) {
      const int J = q * D + j, t = (n + j) % M, to = railPeer(t), from = railPeer((n - j + M) % M), pMe = rsSubOf(me, t);
      CUDACHECK(cudaStreamWaitEvent(s, rsSumEvB_[bank][J], 0));
      specs.push_back({outOff(J),
                       plan.inboxBase + (static_cast<size_t>(plan.waves[w].parity) * L + static_cast<size_t>(pMe)) * blockStride, to, from,
                       &plan.nicChunks});
      if ((ri > 0 || hadPrevious) && !faultOn(Fault::nocredit, rank_)) releases.push_back({1, to, rsSigCredit(plan.area), 0});
    }
    waitSignal(releases, s);  // my targets are ready for this round (paced, and its parity consumed)
    std::vector<int> releaseTo;
    std::function<void()> beforeRelease;
    if (ri + 1 < nR) {
      for (int j = firstJ[ri + 1]; j < firstJ[ri + 1] + plan.kRounds[ri + 1]; ++j) releaseTo.push_back(railPeer((n - j + M) % M));
      const int wNext = waveOfRound(ri + 1);
      if (wNext != w && wNext >= 2) beforeRelease = [&, wNext] { waitConsumed(wNext - 2); };  // next round opens a reused parity
    }
    putRounds(*plan.arena, specs, releaseTo.empty() || !plan.paced ? 0 : plan.paceTail, rsSigData(), releaseTo, rsSigCredit(plan.area), s,
              beforeRelease);
    afterRoundPuts(ri, s);
    rsDebugSync(u, "nic round", s);
    if (w >= 0 && ri == plan.waves[w].firstRound + plan.waves[w].rounds - 1) finishWave(w);
  }
  TIDE_CHECK(!rsReadWanted_, "pipelined reduce-scatter finished its rounds without recording read_event");
  if (!skipData) {
    for (int w = 0; w < nW; ++w)
      if (plan.waves[w].rounds == 0) finishWave(w);  // a wave of my own node only
  }
  phase.reset();
  nvtx::Range tailRange("tide.rs.tail", nvtx::kRsTail);
  if (!skipData) {
    // every wave landed here and on the team: the next call's first round may overwrite my inbox
    for (int w = std::max(0, nW - 2); w < nW; ++w) waitConsumed(w);
    // blocks the parities' last waves fed but their first waves (next call) will not: zero them now
    CUDACHECK(cudaEventRecord(rsForkEv_, s));
    for (int w = 0; w < std::min(nW, plan.inboxParities); ++w) {
      if (!plan.waves[w].zero) continue;
      CUDACHECK(cudaStreamWaitEvent(rsZeroStream_, rsForkEv_, 0));
      if (!faultOn(Fault::nozero, rank_))
        for (int p = 0; p < L; ++p)
          if (plan.waves[w].zero >> p & 1)
            CUDACHECK(cudaMemsetAsync(block(plan.inboxBase, plan.waves[w].parity, p), 0, blockStride, rsZeroStream_));
      CUDACHECK(cudaEventRecord(rsZeroEv_[plan.waves[w].parity], rsZeroStream_));
    }
  }
  rsJoin(s);  // landing / zeroing (the CE / sum streams hold nothing later than this call's reads yet)
  if (!skipData) {
    const int k0 = plan.kRounds.empty() ? 0 : plan.kRounds[0];
    if (devctl) {  // the next call's first-round credits from F; s waits for it (the merged stream's completion)
      CUDACHECK(cudaEventRecord(dcPreFEv_, s));
      CUDACHECK(cudaStreamWaitEvent(rsDevStream_, dcPreFEv_, 0));
      DcFinArgs fa{};
      fa.devComm = devComm_;
      fa.me = n;
      fa.sigCredit0 = M * dcCtx_ + plan.area * M;
      for (int j = 1; j <= k0; ++j) fa.relTo[fa.nRel++] = (n - j + M) % M;
      if (!(faultOn(Fault::dcstall, rank_) && rsEpoch_ > 0)) launchDcFin(fa, rsDevStream_);
      CUDACHECK(cudaEventRecord(dcNetEv_, rsDevStream_));
      CUDACHECK(cudaStreamWaitEvent(s, dcNetEv_, 0));
    } else {
      Group grp;
      for (int j = 1; j <= k0; ++j) signal(railPeer((n - j + M) % M), rsSigCredit(plan.area), /*ctx=*/0, s);
      grp.end();
    }
    ++area.uses;
    area.waveSeq += static_cast<uint64_t>(nW);
    rsBankUsed_[bank] = true;
  }
  ++rsEpoch_;
}
// One rank per node (RsPlan::railBlocks): per k-way round, the copy engines pack my pieces for each
// destination into an outbox block, the NIC puts them into the owners' inbox blocks of the round's
// parity, and once they landed (stream order on s) the sum stream folds the round into out. The
// release a source waits before round r + 2 reuses a parity is sent after the owner's sum of round r.
// With plan.chunks parts (TIDE_RS_RAIL_PARTS) the rounds run once per staging part, parities and
// credits continuing across parts: every element still sums its blocks in the same order.
void Comm::issueRsRail(RsPlan& plan, const std::vector<const void*>& src, float* out, float scale, bool accumulate,
                       cudaStream_t s) {
  const int M = nNodes_, n = node_, O = plan.outboxSlots, B = plan.railBlocks, D = M - 1;
  const int nR = static_cast<int>(plan.kRounds.size()), R = nR * plan.chunks;
  RsArea& area = rsAreas_[plan.area];
  const bool skipData = faultOn(Fault::nowrite, rank_);
  char* mine = static_cast<char*>(plan.arena->base);
  const size_t stride = plan.outStride, isz = static_cast<size_t>(plan.itemsize);
  const uint64_t u = area.uses + 1;
  auto inbox = [&](int parity, int b) { return plan.inboxBase + (static_cast<size_t>(parity) * B + static_cast<size_t>(b)) * stride; };
  auto outbox = [&](int J) { return plan.outboxBase + static_cast<size_t>(J % O) * stride; };
  std::vector<int> first(nR + 1, 1);  // destination index j of the first target of round ri, per part
  for (int i = 0; i < nR; ++i) first[i + 1] = first[i] + plan.kRounds[i];
  for (auto* v : {&rsSumEv_, &rsPutEv_})  // indexed by J = part * D + j
    for (v->reserve(static_cast<size_t>(D * plan.chunks + 1)); v->size() < static_cast<size_t>(D * plan.chunks + 1);) {
      v->push_back(nullptr);
      CUDACHECK(cudaEventCreateWithFlags(&v->back(), cudaEventDisableTiming));
    }
  rsFork(s);
  const int ctas = cfg_.rsCtas > 0 ? cfg_.rsCtas : 4;

  for (int r = 0; r < R && !skipData; ++r) {
    const int c = r / nR, ri = r % nR, Kr = plan.kRounds[ri], parity = r % 2;
    const size_t x0 = static_cast<size_t>(c) * plan.chunkElems, x1 = std::min(plan.stgElems, x0 + plan.chunkElems);
    for (int j = first[ri]; j < first[ri + 1]; ++j) {
      const int J = c * D + j;
      cudaStream_t ce = rsCeStreams_[J % kRsCeStreams];
      if (J > O) CUDACHECK(cudaStreamWaitEvent(ce, rsPutEv_[J - O], 0));  // outbox block reuse
      const size_t t = static_cast<size_t>((n + j) % M);
      CopyBatch pack;
      for (size_t p = 0; p < plan.segs.size(); ++p) {
        const RsSegment& sg = plan.segs[p];
        const size_t lo = std::max(x0, sg.stgOff), hi = std::min(x1, sg.stgOff + sg.pieceElems);
        if (lo >= hi) continue;
        const size_t start = t * sg.pieceElems + (lo - sg.stgOff);
        if (start >= sg.numel) continue;
        pack.add(mine + outbox(J) + (lo - x0) * isz, static_cast<const char*>(src[p]) + start * isz,
                 std::min(hi - lo, sg.numel - start) * isz);
      }
      batchCopy(pack, ce);
      CUDACHECK(cudaEventRecord(rsSumEv_[J], ce));
    }
    std::vector<PutSpec> specs;
    std::vector<std::array<int, 4>> releases;
    for (int j = first[ri]; j < first[ri + 1]; ++j) {
      const int J = c * D + j, to = (n + j) % M, from = (n - j + M) % M;
      CUDACHECK(cudaStreamWaitEvent(s, rsSumEv_[J], 0));
      specs.push_back({outbox(J), inbox(parity, j - first[ri]), to, from, &plan.nicChunks});
      if ((r > 0 || area.uses > 0) && !faultOn(Fault::nocredit, rank_)) releases.push_back({1, to, rsSigCredit(plan.area), 0});
    }
    waitSignal(releases, s);  // my targets summed what they last held in this parity
    std::vector<int> releaseTo;
    std::function<void()> beforeRelease;
    if (r + 1 < R) {
      const int rn = (r + 1) % nR;
      for (int j = first[rn]; j < first[rn + 1]; ++j) releaseTo.push_back((n - j + M) % M);
      if (r >= 1) beforeRelease = [&, r] { CUDACHECK(cudaStreamWaitEvent(s, rsDoneEv_[(r - 1) % 2], 0)); };
    }
    putRounds(*plan.arena, specs, releaseTo.empty() || !plan.paced ? 0 : plan.paceTail, rsSigData(), releaseTo, rsSigCredit(plan.area), s,
              beforeRelease);
    for (int j = first[ri]; j < first[ri + 1]; ++j) CUDACHECK(cudaEventRecord(rsPutEv_[c * D + j], s));
    rsDebugSync(u, "nic round", s);
    CUDACHECK(cudaEventRecord(rsReadyEv_[parity], s));
    CUDACHECK(cudaStreamWaitEvent(rsSumStream_, rsReadyEv_[parity], 0));
    if (faultOn(Fault::slow, rank_)) faultDelay(rsSumStream_);
    // the round's blocks at a constant stride, in order, then the own piece (first round only)
    RsSumArgs t = sumArgs(plan, src, Kr + 1, Kr, scale, ri > 0 || accumulate);
    t.stage = mine + inbox(parity, 0);
    t.srcStride = stride;
    t.chunkBegin = x0;
    t.chunkEnd = x1;
    t.destRank = rank_;
    t.nDest = 1;
    t.out = out;
    t.finalMode = 1;
    t.skipOwn = ri > 0;
    launchRsSum(t, ctas, rsSumStream_);
    if (ri == 0) rsMarkRead(rsSumStream_, kRsCeStreams);  // a part's first round reads my own piece in place
    CUDACHECK(cudaEventRecord(rsDoneEv_[parity], rsSumStream_));
    rsDebugSync(u, "rail landed", rsSumStream_);
  }
  for (int i = 0; i < kRsCeStreams; ++i) rsMarkRead(rsCeStreams_[i], i);  // the packs
  rsJoin(s);
  if (!skipData) {
    Group grp;
    for (int j = 1; j <= plan.kRounds[0]; ++j) signal((n - j + M) % M, rsSigCredit(plan.area), /*ctx=*/0, s);
    grp.end();
    ++area.uses;
  }
}

void Comm::dcTraceConsume(uint64_t epoch) {
  const size_t slot = static_cast<size_t>(epoch % kDcTraceCalls);
  const uint32_t mask = dcTraceMask_[slot];
  if (!mask) return;
  auto ms = [](uint64_t a, uint64_t b) { return b > a ? static_cast<double>(b - a) * 1e-6 : 0.0; };
  // network span, team waveDone wait, credit wait, puts, data wait, release, flush, between kernels,
  // credit wait in the waves' first rounds, in their later rounds
  std::array<double, 10> m{};
  std::vector<double> node(dcTraceNode_.size(), 0.0);
  uint64_t first = 0, prevEnd = 0;
  for (int w = 0; w < kDcTraceWaves; ++w) {
    if (!(mask >> w & 1)) continue;
    const volatile DcWaveTrace& t = dcTraceHost_[slot * kDcTraceWaves + static_cast<size_t>(w)];
    if (t.epoch != static_cast<int32_t>(epoch)) return;  // not complete: skip the call
    const DcTracePeers& tp = dcTracePeers_[slot * kDcTraceWaves + static_cast<size_t>(w)];
    if (prevEnd) m[7] += ms(prevEnd, t.start);
    else first = t.start;
    m[1] += ms(t.start, t.doneSeen);
    uint64_t last = t.creditsOut;
    const int nRounds = std::min(static_cast<int>(t.nRounds), kDcMaxRounds);
    for (int r = 0; r < nRounds; ++r) {
      const double credit = ms(last, t.round[r][0]), data = ms(t.round[r][1], t.round[r][2]);
      m[2] += credit;
      m[r == 0 ? 8 : 9] += credit;
      m[3] += ms(t.round[r][0], t.round[r][1]);
      m[4] += data;
      m[5] += ms(t.round[r][2], t.round[r][3]);
      for (int i = 0; i < tp.k[r]; ++i) {  // a round waits for all its peers: each gets the round's wait
        node[4 * static_cast<size_t>(tp.to[r][i])] += credit;
        node[4 * static_cast<size_t>(tp.to[r][i]) + 1] += 1;
        node[4 * static_cast<size_t>(tp.from[r][i]) + 2] += data;
        node[4 * static_cast<size_t>(tp.from[r][i]) + 3] += 1;
      }
      last = t.round[r][3];
    }
    m[6] += ms(last, t.end);
    prevEnd = t.end;
  }
  m[0] = ms(first, prevEnd);
  dcTraceAcc_.push_back(m);
  for (size_t i = 0; i < node.size(); ++i) dcTraceNode_[i] += node[i];
  if (dcTraceAcc_.size() < static_cast<size_t>(cfg_.rsDevctlTrace)) return;
  auto pick = [&](int i, double q) {
    std::vector<double> v;
    for (const auto& a : dcTraceAcc_) v.push_back(a[i]);
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))];
  };
  fprintf(stderr,
          "[TIDE-RS net trace] rank %d, %zu calls, ms per call (median): network %.1f (p90 %.1f); in the wave kernels: "
          "waveDone wait %.1f, credit wait %.1f (waves' first rounds %.1f p90 %.1f, later rounds %.1f p90 %.1f), puts %.1f, "
          "data wait %.1f, release %.1f, flush %.1f; between kernels %.1f\n",
          rank_, dcTraceAcc_.size(), pick(0, 0.5), pick(0, 0.9), pick(1, 0.5), pick(2, 0.5), pick(8, 0.5), pick(8, 0.9),
          pick(9, 0.5), pick(9, 0.9), pick(3, 0.5), pick(4, 0.5), pick(5, 0.5), pick(6, 0.5), pick(7, 0.5));
  std::string credit, data;
  char buf[48];
  for (int p = 0; p < nNodes_; ++p) {
    const double* v = &dcTraceNode_[4 * static_cast<size_t>(p)];
    snprintf(buf, sizeof(buf), " n%d %.1f", p, v[1] > 0 ? v[0] / v[1] : 0.0);
    credit += buf;
    snprintf(buf, sizeof(buf), " n%d %.1f", p, v[3] > 0 ? v[2] / v[3] : 0.0);
    data += buf;
  }
  fprintf(stderr, "[TIDE-RS net trace] rank %d, ms per round by peer node (mean): credit wait for the target:%s; data wait for the source:%s\n",
          rank_, credit.c_str(), data.c_str());
  dcTraceAcc_.clear();
  std::fill(dcTraceNode_.begin(), dcTraceNode_.end(), 0.0);
}
}  // namespace tide
