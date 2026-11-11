// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "collective/rs/devctl.h"
#include "runtime/config.h"

namespace tide {

// The NCCL communicator. Comm::close() destroys it at a program point of the caller's choosing
// (ncclCommDestroy barriers within the node); otherwise it goes with the last Window or Comm.
struct CommHandle {
  ncclComm_t comm = nullptr;
  bool aborted = false;  // a call failed mid-protocol: nothing may wait for this device or its peers any more
  void destroy();
  void abort();
  ~CommHandle();
};

// A symmetric window: same size registered on every rank; peer[q] is q's base mapped into this
// process (nullptr when q is not NVLink/LSA-reachable). Owns its registration and its memory -- or,
// for a registered external buffer, a reference to whatever owns it -- so a tensor over the window
// stays valid for as long as it holds the Window.
struct Window {
  std::shared_ptr<CommHandle> comm;
  std::shared_ptr<CommHandle> railComm;  // multi-node: the rail communicator the same memory is registered on
  int cudaDev = 0;
  int id = -1;          // allocation ordinal on its communicator; registrations key on it, not on the address
  bool freed = false;   // Comm::freeWindow ran: plans over it must not be used again
  void* base = nullptr;
  size_t bytes = 0;
  ncclWindow_t win = nullptr;
  ncclWindow_t railWin = nullptr;  // registration on railComm (NIC puts); nullptr: single node or a flags window
  std::vector<void*> peer;
  void* multimem = nullptr;
  std::shared_ptr<void> external;  // owner of an external buffer: released only after the deregistration
  ~Window();
};

struct Props {
  int rank = 0, nranks = 0, cudaDev = 0;
  bool deviceApi = false, multimem = false, hostRma = false;
  int nLsaTeams = 0, ginType = 0;
  int ncclVersion = 0;
};

struct AgPlan;
struct RsPlan;
struct RsSumArgs;

class Comm {
 public:
  // numRmaCtx: QPs the inter-node leg stripes over (0 = 4; one QP ramps slowly on RoCE). numRmaSig:
  // signal indices; 0 carries data arrival, 1 round pacing, 2.. one credit channel per stage area (0 = 64).
  Comm(const std::string& uid, int nranks, int rank, int numRmaCtx, int numRmaSig);
  ~Comm();
  Comm(const Comm&) = delete;
  Comm& operator=(const Comm&) = delete;

  int rank() const { return rank_; }
  int nranks() const { return nranks_; }
  const Props& props() const { return props_; }
  const std::vector<int>& lsaPeers() const { return lsaPeers_; }
  int node() const { return node_; }
  double initCommS() const { return initCommS_; }  // ncclCommInitRankConfig (no RMA contexts: no connect to every rank)
  double initSyncS() const { return initSyncS_; }  // first window (sync flags): LSA / multicast mapping only
  double initRailS() const { return initRailS_; }  // ncclCommSplit of the rail communicator: RMA connect to M - 1 peers
  int railRank(int peer) const { return peer / localRanks_; }  // a rail peer's rank in the rail communicator (= its node)
  int nNodes() const { return nNodes_; }
  int localRanks() const { return localRanks_; }
  int rmaCtx() const { return rmaCtx_; }
  int rmaSig() const { return rmaSig_; }
  const Config& config() const { return cfg_; }
  int nicGroups() const { return cfg_.nicGroups; }
  int paceMode() const { return cfg_.paceMode; }
  size_t paceLead() const { return cfg_.paceLead; }
  bool pacesShard(size_t shardBytes) const;
  // Tail bytes per chunk whose arrival a paced round waits only after signalling (plan builders).
  size_t paceTail(const std::vector<std::pair<size_t, size_t>>* groups, int ngroups) const;
  uint64_t seq() const { return seq_; }
  bool closed() const { return closed_; }

  // Collective: every rank registers the same size.
  std::shared_ptr<Window> allocWindow(size_t bytes, bool rail = true);  // rail: also registered on the rail communicator
  // Same, over memory the caller allocated with ncclMemAlloc (e.g. a torch MemPool on NCCL's allocator,
  // so torch accounts for it): base and size 2 MiB-aligned; `owner` keeps it alive until deregistered.
  std::shared_ptr<Window> registerWindow(void* base, size_t bytes, std::shared_ptr<void> owner);
  // Forgets `w` and every registration on it (collective in effect: same program point on every
  // rank, no call in flight, no plan over it used again). Memory and mapping go with the last
  // tensor over the window; a later window may take over the freed credit channels.
  void freeWindow(const std::shared_ptr<Window>& w);
  // Collective: drains the device, releases the windows and destroys the NCCL communicator (which
  // barriers within the node -- hence at a program point of the caller's choosing, not the garbage
  // collector's). Every later call throws. The destructor closes when the caller did not. After a
  // call failed mid-protocol it aborts instead: no drain (peers may never answer), the windows'
  // memory is left to the process exit.
  void close();

  // Collective (same table on every rank, in the same order). Multi-node plans whose receive areas
  // overlap share one credit channel; a receive area may not touch any registered landing, and an
  // own shard (ownOff, written by the caller at any time) may not touch any landing or receive area.
  std::shared_ptr<AgPlan> registerAg(std::shared_ptr<Window> arena, size_t regionOff, const std::vector<size_t>& sizes,
                                     const std::vector<size_t>& offsets, long stageOff, long ownOff);

  // One layout-aware all-gather over `plan` on `stream`; zero kernels. The caller has already
  // written this rank's pieces in place (or its packed shard at the plan's ownOff -- where the
  // sharded parameters themselves may live, so nothing is copied in). Calls on one comm must be
  // issued in the same order on every rank; each call is ordered behind the previous one whatever
  // stream it is issued on. With doneEv (decoupled): doneEv is recorded once every piece has landed
  // here and `stream` does not wait for the other ranks' writes, so consumers must wait on doneEv;
  // peers may write into the landing once freeEv (the release of its previous tenant; null: none)
  // has passed, not when `stream` reaches this call. One-rank-per-node plans record doneEv on `stream`.
  void allGather(AgPlan& plan, cudaStream_t stream, cudaEvent_t freeEv = nullptr, cudaEvent_t doneEv = nullptr);
  // A caller that writes its packed shard into a receive stage (the plan's own slot) waits here on `stream`
  // first: the previous decoupled call's scatters may still read that stage, and plans of different shard
  // sizes lay their slots out differently, so the own slot can cover another plan's receive slot.
  void awaitStageReads(cudaStream_t stream);

  // Reduce-scatter over gradients read in place (see collective/rs/plan.h). Collective registration:
  // same shapes, dtype and stage offset on every rank; stage areas that overlap share a channel.
  // Sizing from tide/solver.py; values <= 0 keep the current one, and each TIDE_RS_CTAS /
  // TIDE_RS_MC_CTAS / TIDE_RS_KWAY / TIDE_RS_CHUNK_MIB / TIDE_RS_PARTS / TIDE_RS_OUTBOX set in the environment wins.
  // Only before the first reduce-scatter registration: areas and plans are sized with it.
  void setRsTuning(int sumCtas, int mcCtas, int kway, long chunkBytes, int parts = 0, int outbox = 0, int pipeline = 0);
  int rsPipeline() const { return cfg_.rsPipeline; }
  int rsCtas() const { return cfg_.rsCtas; }
  int rsMcCtas() const { return cfg_.rsMcCtas; }
  int rsKway() const { return cfg_.rsKway; }
  size_t rsChunkBytes() const { return cfg_.rsChunk; }
  int rsParts() const { return cfg_.rsParts; }
  int rsOutbox() const { return cfg_.rsOutbox; }
  size_t rsStageBytes(const std::vector<std::pair<size_t, size_t>>& shapes, int itemsize) const;
  size_t rsAreaBytes(const std::vector<std::pair<std::vector<std::pair<size_t, size_t>>, int>>& tables) const;
  int railPeer(int node) const { return node * localRanks_ + rank_ % localRanks_; }  // my rail peer on `node`
  std::shared_ptr<RsPlan> registerRs(std::shared_ptr<Window> arena, size_t stageOff,
                                     const std::vector<std::pair<size_t, size_t>>& shapes, int itemsize);
  // src[p]: gradient p (contiguous, itemsize of the plan); out: packed fp32 of plan.outElems.
  // out = (accumulate ? out : 0) + scale * sum over ranks. Same call order on every rank.
  // readEv (optional) is recorded once the call has read all of src (its copy-engine pushes and the sum
  // kernels' in-place reads), before the call completes: a stream that waited on it may reuse src.
  // A pipelined plan (TIDE_RS_PIPELINE=2, multicast) needs readyEv, recorded where src is complete: the
  // reads wait for it instead of for `stream`, and the network phase on `stream` waits for gateEv if given.
  void reduceScatter(RsPlan& plan, const std::vector<const void*>& src, float* out, float scale, bool accumulate,
                     cudaStream_t stream, cudaEvent_t readEv = nullptr, cudaEvent_t readyEv = nullptr,
                     cudaEvent_t gateEv = nullptr);

  // Host RMA primitives (NCCL 2.31 public API), used by the inter-node leg and its benchmark.
  void putSignal(const Window& w, size_t srcOff, size_t bytes, int peer, size_t peerOff, int sigIdx, int ctx,
                 cudaStream_t s);
  void signal(int peer, int sigIdx, int ctx, cudaStream_t s);
  // descs: (opCnt, peer, sigIdx, ctx)
  void waitSignal(const std::vector<std::array<int, 4>>& descs, cudaStream_t s);

 private:
  struct Range {
    int winId;
    size_t begin, end;  // window-relative bytes
    bool overlaps(const Range& o) const { return winId == o.winId && begin < o.end && o.begin < end; }
  };
  // A credit channel. Its counters outlive the window (a freed area is revived by the next new
  // one), so a credit is never consumed one call late.
  struct StageArea : Range {
    uint64_t uses = 0;  // puts issued into this area so far
    bool live = true;
  };

  void live() const;
  void initSync();
  void teardown();
  std::shared_ptr<Window> finishWindow(std::shared_ptr<Window> w, bool rail);  // zero, register (+ rail), map peers
  void initRail();  // multi-node: split the rail communicator (one rank per node) that carries the RMA
  int stageSlotOf(const Window& w, size_t begin, size_t end);
  struct PutSpec {
    size_t srcBase, dstBase;  // window-relative; `chunks` (offset, bytes; one per RMA context) are relative to both
    int to, from;             // put to `to`; the same ranges arrive from `from`
    const std::vector<std::pair<size_t, size_t>>* chunks;
  };
  void putRounds(const Window& w, const std::vector<PutSpec>& specs, size_t tail, int sigData, const std::vector<int>& paceTo,
                 int sigPace, cudaStream_t s, const std::function<void()>& beforeSignal = {},
                 const std::vector<PutSpec>* arrivals = nullptr);  // what arrives, when it differs from what is sent
  void fillFlagParams(const Window& w, size_t offBytes, uint64_t value, bool write);
  void writeFlags(const Window& w, size_t offBytes, uint64_t value, cudaStream_t s);
  void waitFlags(const Window& w, size_t offBytes, uint64_t value, cudaStream_t s);
  void announceReady(uint64_t k, cudaStream_t s, cudaEvent_t freeEv, cudaEvent_t doneEv);
  cudaStream_t awaitDone(uint64_t k, cudaStream_t s, cudaEvent_t doneEv);
  void issueSingleNode(AgPlan& plan, uint64_t k, cudaStream_t s, cudaEvent_t freeEv, cudaEvent_t doneEv);
  void issueMultiNode(AgPlan& plan, uint64_t k, cudaStream_t s, cudaEvent_t freeEv, cudaEvent_t doneEv);
  void issueDirect(AgPlan& plan, uint64_t k, cudaStream_t s);
  int readySig_ = -1;  // direct all-gathers (one rank per node): "my landing may be written" (AgPlan::direct)

  // Reduce-scatter internals (collective/rs).
  struct RsArea : Range {
    uint64_t uses = 0;         // calls issued (inbox credit bookkeeping)
    uint64_t slotUses[Config::kRsMaxStageSlots] = {};  // pushes into each local-staging slot so far (land/free flag values)
    uint64_t waveSeq = 0;      // waves landed so far (waveReady/waveDone flag values); kept across revivals
    bool live = true;
    // Geometry every plan on this area adopts, so slot i means the same bytes for all of them (the
    // free flags are per slot index): the largest table registered in each dimension, fixed by the
    // area's first call (growArea / adoptArea).
    size_t slotBytes = 0, inboxBase = 0, outboxBase = 0, ownBase = 0, outStride = 0;
    bool geometrySet = false, fixed = false;  // fixed: a call was issued on this incarnation of the area
  };
  void initRsSync();
  void growArea(RsArea& a, const RsPlan& plan, const Window& arena);
  void adoptArea(RsPlan& plan) const;
  int rsAreaOf(const Window& w, size_t begin, size_t end);
  // Flag arrays per area, values are use / wave counters. Land/Free: per staging slot; WaveReady: per
  // inbox parity, my blocks of the wave are written or zeroed; WaveDone: per parity, I have landed the
  // wave (its parity may be overwritten).
  enum RsFlag { kLand, kFree, kWaveReady, kWaveDone };
  static constexpr int kRsFlagKinds = 2 * Config::kRsMaxStageSlots + 2 * 2;
  size_t rsFlagOff(int area, RsFlag kind, int slot = 0) const;
  void issueRsLocal(RsPlan& plan, const std::vector<const void*>& src, float* out, float scale, bool accumulate,
                    cudaStream_t s);
  void issueRsRail(RsPlan& plan, const std::vector<const void*>& src, float* out, float scale, bool accumulate, cudaStream_t s);
  void issueRsMc(RsPlan& plan, const std::vector<const void*>& src, float* out, float scale, bool accumulate,
                 cudaStream_t s);
  void issueRsMcPipelined(RsPlan& plan, const std::vector<const void*>& src, float* out, float scale, bool accumulate,
                          cudaStream_t s, cudaEvent_t readEv, cudaEvent_t readyEv, cudaEvent_t gateEv);
  void rsLevel1(RsPlan& plan, RsSumArgs& sum, int t, float* outBuf, bool whole, int part = 0);
  int rsSubOf(int r, int t) const;  // sub-slice of node t's destinations local rank r of my node sums (collective/rs/waves.h)
  void rsFork(cudaStream_t s);
  void rsJoin(cudaStream_t s);
  void rsDebugSync(uint64_t u, const char* what, cudaStream_t st);
  void rsDumpIfStuck(const RsPlan& plan, uint64_t u, cudaStream_t s);
  // Signal indices: data 0, pace 1, all-gather credits 2.. upward, reduce-scatter data at the top and its
  // credits downward from below it; a new channel on either side must leave the two ranges disjoint.
  void checkSignalBudget(size_t agAreas, size_t rsAreas) const;
  int rsSigData() const { return rmaSig_ - 1; }
  int rsSigCredit(int area) const { return rmaSig_ - 2 - area; }  // `release`: pacing and inbox credit in one channel

  std::vector<RsArea> rsAreas_;
  std::shared_ptr<Window> rsSync_;  // flags: [area][kind][slot][nranks] uint64 (see rsFlagOff)
  size_t rsFlagArrayBytes_ = 0;
  // rsZeroStream_: the inbox zeroing, whose wait for the team's landing would otherwise hold up the level-1 pushes
  // queued behind it on the CE streams (and with them the sums, the NIC and the next landing: one wave at a time).
  // Level-1 pushes rotate over kRsCeStreams streams: a stream holds only ~1000 queued copy descriptors, and a
  // call queued behind a long kernel would otherwise block its issuing thread until that kernel drains.
  static constexpr int kRsCeStreams = 4;
  std::array<cudaStream_t, kRsCeStreams> rsCeStreams_{};
  cudaStream_t rsSumStream_ = nullptr, rsAddStream_ = nullptr, rsZeroStream_ = nullptr;
  std::array<cudaStream_t, kRsCeStreams + 3> rsStreams() const;  // every internal reduce-scatter stream
  cudaEvent_t rsForkEv_ = nullptr, rsJoinEv_ = nullptr;
  std::vector<cudaEvent_t> rsSumEv_, rsPutEv_;
  std::vector<cudaEvent_t> rsReadyEv_, rsDoneEv_, rsZeroEv_;  // per inbox parity: wave ready (s), landed (landing stream), zeroed
  // Pipelined calls alternate two banks (outbox blocks, own spares and these events) by a communicator-wide call
  // count, so a call's reads never touch what the previous call's network phase still reads: per bank, block J
  // summed (sum), put out (put: the outbox block may be overwritten), own wave k summed (own), copied in (spare).
  std::array<std::vector<cudaEvent_t>, 2> rsSumEvB_, rsPutEvB_, rsOwnEvB_, rsSpareEvB_;
  std::array<bool, 2> rsBankUsed_{};
  uint64_t rsEpoch_ = 0;
  bool rsOrdered_ = false;  // rsOrderEv_ marks the end of the previous reduce-scatter call (its network phase, pipelined)
  cudaEvent_t rsOrderEv_ = nullptr;
  // TIDE_RS_NET=device (collective/rs/devctl.h): the device communicator (created by the first pipelined call, on every
  // rank in the same call order), the stream its W / F kernels run on, the cumulative signal counts the waits expect
  // (identical bookkeeping on every rank), and the fault page the monitor thread watches.
  void ensureDevComm();
  void stopDcMonitor();
  bool devCommOk_ = false;
  ncclDevComm devComm_{};
  int dcCtx_ = 1;
  cudaStream_t rsDevStream_ = nullptr;
  cudaEvent_t dcGateEv_ = nullptr, dcPreFEv_ = nullptr, dcNetEv_ = nullptr;
  std::array<std::vector<cudaEvent_t>, 2> dcWaveEvB_;
  std::vector<uint64_t> dcCreditExp_, dcDataExp_;  // [area][target], [source][context]
  DcError* dcErrHost_ = nullptr;
  DcError* dcErrDev_ = nullptr;
  // TIDE_RS_NET_TRACE: [call % kDcTraceCalls][wave] records (mapped), read back kDcTraceLag calls later
  static constexpr uint64_t kDcTraceLag = 32;
  DcWaveTrace* dcTraceHost_ = nullptr;
  DcWaveTrace* dcTraceDev_ = nullptr;
  std::array<uint32_t, kDcTraceCalls> dcTraceMask_{};  // waves of the call that launched a kernel
  std::vector<std::array<double, 10>> dcTraceAcc_;
  struct DcTracePeers {  // per traced wave, written at launch: each round's targets and sources (node indices)
    int8_t k[kDcMaxRounds];
    int8_t to[kDcMaxRounds][kDcMaxK], from[kDcMaxRounds][kDcMaxK];
  };
  std::vector<DcTracePeers> dcTracePeers_;  // kDcTraceCalls x kDcTraceWaves
  std::vector<double> dcTraceNode_;         // per node: credit wait as target (sum, rounds), data wait as source (sum, rounds)
  void dcTraceConsume(uint64_t epoch);
  std::thread dcMonitor_;
  std::atomic<bool> dcMonitorStop_{false};
  // readEv of reduceScatter: rsReadStream_ waits on a mark after the last read of src on each reading stream
  // (rsCeStreams_, then rsSumStream_) and records it.
  void rsMarkRead(cudaStream_t st, int i);
  bool rsReadWanted_ = false;
  cudaStream_t rsReadStream_ = nullptr;
  std::array<cudaEvent_t, kRsCeStreams + 1> rsReadMark_{};

  std::shared_ptr<CommHandle> comm_;
  // Multi-node: the NIC leg (ncclPutSignal / ncclSignal / ncclWaitSignal, the GIN device communicator) runs on a
  // communicator of this rank's rail peers (one rank per node, rank = node), split from comm_. NCCL 2.31's host RMA
  // connects every context to every rank of its communicator at the first window registration, with a ring barrier
  // per peer (~ctx x ranks^2: 81-184 s at 512 ranks); TIDE only ever puts to the M - 1 rail peers, so comm_ runs
  // with numRmaCtx = 0 (LSA pointers and multicast unaffected) and the connect happens on M ranks instead.
  std::shared_ptr<CommHandle> railComm_;
  int rank_, nranks_;
  int node_ = 0, nNodes_ = 1, localRanks_ = 1;  // derived from the LSA team (ranks assumed node-major)
  Config cfg_;  // knobs (runtime/config.h); setRsTuning may resize the reduce-scatter
  double initCommS_ = 0, initSyncS_ = 0, initRailS_ = 0;  // constructor phases (seconds)
  int rmaCtx_ = 4, rmaSig_ = 64;
  Props props_;
  std::vector<int> lsaPeers_;         // world ranks in my LSA team, excluding self
  std::shared_ptr<Window> sync_;      // ready[nranks] | done[nranks], uint64 each
  size_t readyOff_ = 0, doneOff_ = 0;
  uint64_t seq_ = 0;
  bool failed_ = false;                // an earlier call threw mid-protocol; peers may be waiting forever
  bool closed_ = false;
  bool ordered_ = false;               // orderEv_ marks the end of the previous call
  cudaEvent_t orderEv_ = nullptr;
  std::vector<CUstreamBatchMemOpParams> memops_;
  std::vector<std::shared_ptr<Window>> windows_;
  std::vector<std::weak_ptr<Window>> everyWindow_;  // incl. freed ones still held by tensors: close() deregisters them
  int nextWinId_ = 0;
  std::vector<StageArea> stageAreas_;
  std::vector<Range> landings_, ownShards_;
  // Multi-node: NIC rounds run on the caller's stream, scatters on ceStream_ gated by roundEv_.
  cudaStream_t ceStream_ = nullptr;
  cudaEvent_t forkEv_ = nullptr, ceJoinEv_ = nullptr;
  // Decoupled calls: `ready` on readyStream_ behind the caller's release, the exit wait on doneStream_.
  cudaStream_t readyStream_ = nullptr, doneStream_ = nullptr;
  cudaEvent_t doneForkEv_ = nullptr;
  // Decoupled multi-node calls leave `s` after their last round: the scatters' join, the done fence and doneEv go
  // to doneStream_, and the stage credits (NCCL calls, so on `s`) are sent at the start of the next call, behind
  // pendingCreditEv_ (that call's scatters), so `s` never waits for a scatter.
  cudaEvent_t pendingCreditEv_ = nullptr;
  int pendingCreditSig_ = -1;
  std::vector<int> pendingCreditTo_;
  void flushCredits(cudaStream_t s);
  std::vector<cudaEvent_t> roundEv_;
  // TIDE_TRACE_AG=1: GPU-side phase stamps of multi-node all-gathers (timing events on the caller's
  // stream), ring of the last kTraceCalls calls; traceAgDump() reads the completed ones.
  static constexpr size_t kTraceCalls = 256;
  std::vector<std::vector<cudaEvent_t>> traceEv_;
  std::vector<uint64_t> traceSeq_;
  size_t traceIdx_ = 0, tracePhase_ = 0;
  void traceBegin(uint64_t k, size_t phases);
  void traceMark(cudaStream_t s);

 public:
  // [(call seq, [ms between consecutive phases])]: start>credits>fence>round 1..M-1>scatter join>done
  std::vector<std::pair<uint64_t, std::vector<float>>> traceAgDump();
};

std::string getUniqueId();

}  // namespace tide
