// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "runtime/comm.h"
#include "util/nvtx.h"

#include <fcntl.h>
#include <nccl_device.h>
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <cstring>
#include <string>
#include <utility>

#include "collective/ag/plan.h"
#include "collective/rs/plan.h"
#include "util/check.h"
#include "runtime/comm_internal.h"

namespace tide {

using namespace detail;

namespace {

// SIGKILL every child of this process (data-loader workers): they hold the launcher wrapper's output pipe open, so it
// would never see our exit. No allocation -- another thread may be stuck holding the allocator's lock.
void killChildren() {
  struct Dirent64 {
    uint64_t ino;
    int64_t off;
    unsigned short reclen;
    unsigned char type;
    char name[1];
  };
  const int dir = open("/proc/self/task", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dir < 0) return;
  char ents[4096], path[64], kids[4096];
  for (long n; (n = syscall(SYS_getdents64, dir, ents, sizeof(ents))) > 0;) {
    for (long off = 0; off < n; off += reinterpret_cast<Dirent64*>(ents + off)->reclen) {
      const char* tid = reinterpret_cast<Dirent64*>(ents + off)->name;
      if (tid[0] < '0' || tid[0] > '9') continue;
      snprintf(path, sizeof(path), "/proc/self/task/%s/children", tid);
      const int f = open(path, O_RDONLY | O_CLOEXEC);
      if (f < 0) continue;
      const ssize_t len = read(f, kids, sizeof(kids) - 1);
      close(f);
      long pid = 0;
      for (ssize_t i = 0; i < len; ++i) {
        if (kids[i] >= '0' && kids[i] <= '9') {
          pid = pid * 10 + (kids[i] - '0');
        } else if (pid > 0) {
          kill(static_cast<pid_t>(pid), SIGKILL);
          pid = 0;
        }
      }
      if (pid > 0) kill(static_cast<pid_t>(pid), SIGKILL);
    }
  }
  close(dir);
}

}  // namespace

void CommHandle::destroy() {
  if (comm) ncclCommDestroy(comm);
  comm = nullptr;
}

void CommHandle::abort() {
  if (comm) ncclCommAbort(comm);
  comm = nullptr;
  aborted = true;
}

CommHandle::~CommHandle() { destroy(); }

Window::~Window() {
  // Peers' copy engines may still be landing data here if the caller dropped the last reference
  // early; the mapping goes away only once this device has drained. After Comm::close() the
  // registration is already gone and only the memory remains to be released.
  if (comm && comm->aborted) {  // peers may still write here and this device may never drain
    if (external) new std::shared_ptr<void>(std::move(external));  // keep the memory to the process exit
    return;
  }
  onDevice(cudaDev, [&] {
    cudaDeviceSynchronize();
    if (railWin && railComm && railComm->comm) ncclCommWindowDeregister(railComm->comm, railWin);
    if (win && comm && comm->comm) ncclCommWindowDeregister(comm->comm, win);
    if (base && !external) ncclMemFree(base);
  });
}

std::string getUniqueId() {
  ncclUniqueId id;
  NCCLCHECK(ncclGetUniqueId(&id));
  return std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
}

Comm::Comm(const std::string& uid, int nranks, int rank, int numRmaCtx, int numRmaSig)
    : rank_(rank), nranks_(nranks) {
  TIDE_CHECK(uid.size() == NCCL_UNIQUE_ID_BYTES, "bad unique id length");
  TIDE_CHECK(kFaultInjection || !std::getenv("TIDE_FAULT"), "TIDE_FAULT set but this build has no fault injection");
  int ver = 0;
  NCCLCHECK(ncclGetVersion(&ver));
  TIDE_CHECK(ver >= 23102, "TIDE needs the NCCL 2.31.2+ runtime (multi-context host RMA); found " + std::to_string(ver));
  props_.ncclVersion = ver;

  ncclUniqueId id;
  std::memcpy(id.internal, uid.data(), NCCL_UNIQUE_ID_BYTES);
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  cfg_ = Config::fromEnv();
  rmaCtx_ = numRmaCtx > 0 ? numRmaCtx : cfg_.rmaCtx;
  rmaSig_ = numRmaSig > 0 ? numRmaSig : 64;
  TIDE_CHECK(rmaSig_ > kSigCredit0, "num_rma_sig must be >= 3 (data arrival, pacing, one credit channel)");
  cfg.numRmaCtx = 0;  // the RMA lives on the rail communicator (initRail); here only LSA pointers and multicast
  cfg.numRmaSig = rmaSig_;
  // TIDE never issues an NCCL collective, so one channel is enough; under NCCL_RUNTIME_CONNECT=0 every
  // channel would otherwise preconnect ring/tree buffers (4.6 GiB at 48 ranks x 32 channels). NCCL's own
  // NCCL_MIN_NCHANNELS still duplicates channels past this cap. TIDE_COMM_CTAS=0 leaves NCCL's default.
  if (cfg_.commCtas > 0) cfg.minCTAs = cfg.maxCTAs = cfg_.commCtas;
  comm_ = std::make_shared<CommHandle>();
  const auto tInit0 = std::chrono::steady_clock::now();
  NCCLCHECK(ncclCommInitRankConfig(&comm_->comm, nranks, id, rank, &cfg));
  initCommS_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - tInit0).count();

  ncclCommProperties_t p = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(comm_->comm, &p));
  props_.rank = p.rank;
  props_.nranks = p.nRanks;
  props_.cudaDev = p.cudaDev;
  props_.deviceApi = p.deviceApiSupport;
  props_.multimem = p.multimemSupport;
  props_.hostRma = p.hostRmaSupport;
  props_.nLsaTeams = p.nLsaTeams;
  props_.ginType = static_cast<int>(p.ginType);
  TIDE_CHECK(props_.deviceApi, "communicator lacks device-API/symmetric-window support (cuMem/NVLink P2P off?)");

  for (cudaStream_t* st : {&ceStream_, &readyStream_, &doneStream_}) CUDACHECK(cudaStreamCreateWithFlags(st, cudaStreamNonBlocking));
  for (cudaEvent_t* e : {&forkEv_, &ceJoinEv_, &orderEv_, &doneForkEv_, &pendingCreditEv_})
    CUDACHECK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
  for (cudaStream_t& st : rsCeStreams_) CUDACHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  int least = 0, greatest = 0;
  CUDACHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  const int sumPrio = std::max(greatest, least - cfg_.rsKernelPriority);
  CUDACHECK(cudaStreamCreateWithPriority(&rsSumStream_, cudaStreamNonBlocking, sumPrio));
  CUDACHECK(cudaStreamCreateWithPriority(&rsAddStream_, cudaStreamNonBlocking, sumPrio));
  CUDACHECK(cudaStreamCreateWithFlags(&rsZeroStream_, cudaStreamNonBlocking));
  for (cudaEvent_t* e : {&rsForkEv_, &rsJoinEv_, &rsOrderEv_})
    CUDACHECK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
  CUDACHECK(cudaStreamCreateWithFlags(&rsReadStream_, cudaStreamNonBlocking));
  for (cudaEvent_t& e : rsReadMark_) CUDACHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  nvtx::nameStream(ceStream_, "TIDE ag scatter");
  nvtx::nameStream(readyStream_, "TIDE ag ready");
  nvtx::nameStream(doneStream_, "TIDE ag done");
  for (int i = 0; i < kRsCeStreams; ++i) {
    const std::string n = "TIDE rs push " + std::to_string(i);
    nvtx::nameStream(rsCeStreams_[i], n.c_str());
  }
  nvtx::nameStream(rsSumStream_, "TIDE rs sum");
  nvtx::nameStream(rsAddStream_, "TIDE rs land");
  nvtx::nameStream(rsZeroStream_, "TIDE rs zero");
  nvtx::nameStream(rsReadStream_, "TIDE rs read event");
  // The first symmetric window registration runs NCCL's collective RMA init (ncclRmaProxyConnectOnce: every RMA
  // context connects to every rank of the communicator, a ring barrier after each): the start-up cost that grows
  // with the square of the rank count. Timed apart so the start-up log can show it.
  const auto tSync0 = std::chrono::steady_clock::now();
  initSync();
  initSyncS_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - tSync0).count();
  const auto tRail0 = std::chrono::steady_clock::now();
  if (nNodes_ > 1) initRail();
  initRailS_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - tRail0).count();
  // single node: the sum is the whole collective; one rank per node: each round sums whole blocks
  // (tide/solver.py _plan_rs_rail)
  if (cfg_.rsCtas <= 0) cfg_.rsCtas = nNodes_ == 1 ? 8 : localRanks_ == 1 ? 12 : 4;
  TIDE_CHECK(nNodes_ == 1 || props_.hostRma, "multi-node all-gather needs host RMA (IB/RoCE with GDR, numRmaCtx > 0)");
}

void Comm::initRail() {
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  cfg.numRmaCtx = rmaCtx_;
  cfg.numRmaSig = rmaSig_;
  cfg.rmaEagerInit = 1;  // connect the rail RMA here (sync_ is not registered on railComm_), so initRailS_ and device_bytes include it
  if (cfg_.commCtas > 0) cfg.minCTAs = cfg.maxCTAs = cfg_.commCtas;
  railComm_ = std::make_shared<CommHandle>();
  // color = my local rank (one communicator per rail), key = my node: the rail rank is the node index (railRank)
  NCCLCHECK(ncclCommSplit(comm_->comm, rank_ % localRanks_, node_, &railComm_->comm, &cfg));
  int n = 0, r = 0;
  NCCLCHECK(ncclCommCount(railComm_->comm, &n));
  NCCLCHECK(ncclCommUserRank(railComm_->comm, &r));
  TIDE_CHECK(n == nNodes_ && r == node_, "rail communicator does not match the node topology (ranks must be node-major)");
  ncclCommProperties_t p = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(railComm_->comm, &p));
  props_.hostRma = p.hostRmaSupport;  // the RMA is this communicator's
  props_.ginType = static_cast<int>(p.ginType);
}

Comm::~Comm() { close(); }

void Comm::live() const { TIDE_CHECK(!closed_, "communicator is closed"); }

void Comm::ensureDevComm() {
  if (devCommOk_) return;
  TIDE_CHECK(railComm_, "TIDE_RS_NET=device needs the multi-node rail communicator");
  ncclCommProperties_t p = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(railComm_->comm, &p));
  TIDE_CHECK(p.deviceApiSupport && p.ginSupport[NCCL_GIN_TYPE_GDAKI],
             "TIDE_RS_NET=device needs GDAKI device GIN on this communicator (the PROXY backend is slower than host RMA)");
  dcCtx_ = std::min(rmaCtx_, kDcMaxCtx);
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  reqs.ginType = NCCL_GIN_TYPE_GDAKI;
  reqs.ginContextCount = dcCtx_;
  reqs.ginSignalCount = nNodes_ * dcCtx_ + kDcMaxAreas * nNodes_;
  reqs.ginExclusiveContexts = true;
  reqs.ginStrongSignalsRequired = true;  // a context's last put of a round settles its earlier ones
  reqs.ginVaSignalsRequired = false;
  const ncclResult_t r = ncclDevCommCreate(railComm_->comm, &reqs, &devComm_);
  TIDE_CHECK(r == ncclSuccess, std::string("TIDE_RS_NET=device: ncclDevCommCreate failed: ") + ncclGetErrorString(r));
  int lo = 0, hi = 0;
  CUDACHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  CUDACHECK(cudaStreamCreateWithPriority(&rsDevStream_, cudaStreamNonBlocking, hi));
  for (cudaEvent_t* e : {&dcGateEv_, &dcPreFEv_, &dcNetEv_}) CUDACHECK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
  nvtx::nameStream(rsDevStream_, "TIDE rs net");
  dcCreditExp_.assign(static_cast<size_t>(kDcMaxAreas) * nNodes_, 0);
  dcDataExp_.assign(static_cast<size_t>(nNodes_) * dcCtx_, 0);
  CUDACHECK(cudaHostAlloc(reinterpret_cast<void**>(&dcErrHost_), sizeof(DcError), cudaHostAllocMapped));
  memset(dcErrHost_, 0, sizeof(DcError));
  CUDACHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dcErrDev_), dcErrHost_, 0));
  if (cfg_.rsDevctlTrace > 0) {
    const size_t bytes = sizeof(DcWaveTrace) * kDcTraceCalls * kDcTraceWaves;
    CUDACHECK(cudaHostAlloc(reinterpret_cast<void**>(&dcTraceHost_), bytes, cudaHostAllocMapped));
    memset(dcTraceHost_, 0xff, bytes);  // epoch -1: no record yet
    CUDACHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dcTraceDev_), dcTraceHost_, 0));
    dcTracePeers_.assign(static_cast<size_t>(kDcTraceCalls) * kDcTraceWaves, DcTracePeers{});
    dcTraceNode_.assign(static_cast<size_t>(4) * nNodes_, 0.0);
  }
  dcMonitorStop_ = false;
  dcMonitor_ = std::thread([this, page = dcErrHost_, rank = rank_, debug = cfg_.rsDevctlDebug, nc = railComm_->comm] {  // the GIN communicator's async state
    // A W / F that hit a timeout never returns, so nothing on the device moves past it: exit without waiting on CUDA / NCCL
    // (their state is queried on a detached thread, given 3 s).
    int tick = 0;
    bool asked = false;  // debug: NCCL's async state once a wait passes 60 s, without exiting
    while (!dcMonitorStop_.load()) {
      const volatile DcError* e = page;
      if (debug && ++tick % 50 == 0)
        fprintf(stderr, "[TIDE-RS net dbg] rank %d where %u epoch %d wave %d round %d peer %d ctx %d least %llu seen %llu "
                "waited %.3f s of %.3f code %u\n",
                rank, e->where, e->bEpoch, e->bWave, e->bRound, e->bPeer, e->bCtx, static_cast<unsigned long long>(e->bLeast),
                static_cast<unsigned long long>(e->bSeen), e->bWaitedNs / 1e9, e->bTimeoutNs / 1e9, e->code);
      if (debug && !asked && e->code == 0 && e->bWaitedNs > 60000000000ull) {
        asked = true;
        std::thread([nc, rank] {
          ncclResult_t async = ncclSuccess;
          const ncclResult_t q = ncclCommGetAsyncError(nc, &async);
          fprintf(stderr, "[TIDE-RS net dbg] rank %d waits over 60 s; NCCL async state: %s\n", rank,
                  q == ncclSuccess ? ncclGetErrorString(async) : "query failed");
        }).detach();
      }
      if (e->code != 0) {
        static const char* what[] = {"?", "credit", "data", "flush", "team waveDone"};
        char buf[512];
        const int n = snprintf(buf, sizeof(buf),
                               "[TIDE-RS net] rank %d FATAL: device %s wait timed out (epoch %d, wave %d, round %d, peer %d, "
                               "context %d, expected %llu, seen %llu); exiting\n",
                               rank, what[e->code < 5 ? e->code : 0], e->epoch, e->wave, e->round, e->peer, e->ctx,
                               static_cast<unsigned long long>(e->expected), static_cast<unsigned long long>(e->seen));
        if (n > 0) {
          const ssize_t w = write(2, buf, std::min(static_cast<size_t>(n), sizeof(buf)));
          (void)w;
        }
        // NCCL's view: a GIN progress thread that stopped on an error (CPU-proxy doorbells), GIN / NIC queue errors
        struct State {
          std::atomic<bool> done{false};
          ncclResult_t query = ncclSuccess, async = ncclSuccess;
        };
        auto st = std::make_shared<State>();
        std::thread([st, nc] {
          st->query = ncclCommGetAsyncError(nc, &st->async);
          st->done = true;
        }).detach();
        for (int i = 0; i < 30 && !st->done; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const int m = st->done ? snprintf(buf, sizeof(buf), "[TIDE-RS net] rank %d NCCL async state: %s\n", rank,
                                          st->query == ncclSuccess ? ncclGetErrorString(st->async) : "query failed")
                               : snprintf(buf, sizeof(buf), "[TIDE-RS net] rank %d NCCL async state: no answer in 3 s\n", rank);
        if (m > 0) {
          const ssize_t w = write(2, buf, std::min(static_cast<size_t>(m), sizeof(buf)));
          (void)w;
        }
        killChildren();
        _exit(70);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });
  devCommOk_ = true;
  if (rank_ == 0) {
    const char* tc = getenv("NCCL_GIN_IB_TC");
    fprintf(stderr, "[TIDE-RS net] network phase in GIN device kernels: %d contexts, %d signals, timeout %.0f s, traffic class %s\n",
            dcCtx_, reqs.ginSignalCount, cfg_.rsDevctlTimeoutS, tc ? tc : "NCCL's default");
  }
}

void Comm::stopDcMonitor() {
  if (!dcMonitor_.joinable()) return;
  dcMonitorStop_ = true;
  dcMonitor_.join();
}

void Comm::teardown() {
  stopDcMonitor();
  if (devCommOk_) ncclDevCommDestroy(railComm_->comm, &devComm_);
  devCommOk_ = false;
  if (rsDevStream_) cudaStreamDestroy(rsDevStream_);
  for (cudaEvent_t e : {dcGateEv_, dcPreFEv_, dcNetEv_})
    if (e) cudaEventDestroy(e);
  for (auto& v : dcWaveEvB_)
    for (auto e : v) cudaEventDestroy(e);
  if (dcErrHost_) cudaFreeHost(dcErrHost_);
  dcErrHost_ = nullptr;
  if (dcTraceHost_) cudaFreeHost(dcTraceHost_);
  dcTraceHost_ = nullptr;
  for (cudaStream_t st : {ceStream_, readyStream_, doneStream_})
    if (st) cudaStreamDestroy(st);
  for (cudaStream_t st : rsStreams())
    if (st) cudaStreamDestroy(st);
  if (rsReadStream_) cudaStreamDestroy(rsReadStream_);
  for (cudaEvent_t e : {forkEv_, ceJoinEv_, orderEv_, doneForkEv_, pendingCreditEv_, rsForkEv_, rsJoinEv_, rsOrderEv_})
    if (e) cudaEventDestroy(e);
  for (cudaEvent_t e : rsReadMark_)
    if (e) cudaEventDestroy(e);
  for (auto* v : {&roundEv_, &rsSumEv_, &rsPutEv_, &rsReadyEv_, &rsDoneEv_, &rsZeroEv_})
    for (auto e : *v) cudaEventDestroy(e);
  for (auto* banks : {&rsSumEvB_, &rsPutEvB_, &rsOwnEvB_, &rsSpareEvB_})
    for (auto& v : *banks)
      for (auto e : v) cudaEventDestroy(e);
  for (auto& v : traceEv_)
    for (auto e : v) cudaEventDestroy(e);
}

void Comm::close() {
  if (closed_) return;
  closed_ = true;
  if (failed_) {
    stopDcMonitor();
    onDevice(props_.cudaDev, [&] {
      if (railComm_) railComm_->abort();
      comm_->abort();
    });
    sync_.reset();
    rsSync_.reset();
    windows_.clear();
    everyWindow_.clear();
    railComm_.reset();
    comm_.reset();
    return;
  }
  // Peers may still be writing into our windows until every rank's last call has completed:
  // tear down only after this device drained, in the same program order on every rank. Every
  // window ever allocated (freed ones included: a tensor may still hold one) is deregistered here
  // and the communicator destroyed now, so ncclCommDestroy's intra-node barrier runs at this
  // program point on every rank and never at garbage-collection time; the windows' memory stays
  // mapped for as long as tensors hold it.
  onDevice(props_.cudaDev, [&] {
    cudaDeviceSynchronize();
    teardown();
    for (auto& weak : everyWindow_) {
      if (auto w = weak.lock()) {
        w->freed = true;
        if (w->railWin && railComm_) ncclCommWindowDeregister(railComm_->comm, w->railWin);
        w->railWin = nullptr;
        if (w->win) ncclCommWindowDeregister(comm_->comm, w->win);
        w->win = nullptr;
      }
    }
    if (railComm_) railComm_->destroy();
    comm_->destroy();
  });
  sync_.reset();
  rsSync_.reset();
  windows_.clear();
  everyWindow_.clear();
  railComm_.reset();
  comm_.reset();
}

bool Comm::pacesShard(size_t shardBytes) const {
  return cfg_.paceMode == 1 || (cfg_.paceMode < 0 && cfg_.paceLead > 0 && shardBytes >= 2 * cfg_.paceLead);
}

// Bytes at the end of every context's chunk that stay in flight when the pace signal leaves, so
// the signal's round trip overlaps the round's tail instead of idling the port: the lead spread
// over the chunks, 4 KiB-aligned; none when a chunk could not hold a head at least as long.
size_t Comm::paceTail(const std::vector<std::pair<size_t, size_t>>* groups, int ngroups) const {
  size_t chunks = 0, smallest = SIZE_MAX;
  for (int g = 0; g < ngroups; ++g)
    for (const auto& [off, bytes] : groups[g]) {
      ++chunks;
      smallest = std::min(smallest, bytes);
    }
  if (cfg_.paceLead == 0 || chunks == 0) return 0;
  const size_t tail = alignUp(cfg_.paceLead / chunks, 4096);
  return smallest > 2 * tail ? tail : 0;
}

std::shared_ptr<Window> Comm::allocWindow(size_t bytes, bool rail) {
  live();
  auto w = std::make_shared<Window>();
  w->comm = comm_;
  w->cudaDev = props_.cudaDev;
  w->id = nextWinId_++;
  w->bytes = alignUp(bytes, kWindowAlign);
  NCCLCHECK(ncclMemAlloc(&w->base, w->bytes));
  return finishWindow(std::move(w), rail);
}

std::shared_ptr<Window> Comm::registerWindow(void* base, size_t bytes, std::shared_ptr<void> owner) {
  live();
  TIDE_CHECK(base && reinterpret_cast<uintptr_t>(base) % kWindowAlign == 0 && bytes > 0 && bytes % kWindowAlign == 0,
             "an external window needs a 2 MiB-aligned base and size (ncclMemAlloc memory)");
  auto w = std::make_shared<Window>();
  w->comm = comm_;
  w->cudaDev = props_.cudaDev;
  w->id = nextWinId_++;
  w->bytes = bytes;
  w->base = base;
  w->external = std::move(owner);
  return finishWindow(std::move(w), true);
}

std::shared_ptr<Window> Comm::finishWindow(std::shared_ptr<Window> w, bool rail) {
  CUDACHECK(cudaMemset(w->base, 0, w->bytes));
  NCCLCHECK(ncclCommWindowRegister(comm_->comm, w->base, w->bytes, &w->win, NCCL_WIN_COLL_SYMMETRIC));
  if (rail && railComm_) {  // the same memory, registered for the NIC leg: its symmetric offsets are the same
    NCCLCHECK(ncclCommWindowRegister(railComm_->comm, w->base, w->bytes, &w->railWin, NCCL_WIN_COLL_SYMMETRIC));
    w->railComm = railComm_;
  }
  w->peer.assign(nranks_, nullptr);
  for (int q = 0; q < nranks_; ++q) {
    void* p = nullptr;
    NCCLCHECK(ncclGetPeerDevicePointer(w->win, 0, q, &p));  // nullptr when q is not LSA-reachable
    w->peer[q] = p;
  }
  w->peer[rank_] = w->base;
  if (props_.multimem) {
    void* mc = nullptr;
    if (ncclGetLsaMultimemDevicePointer(w->win, 0, &mc) == ncclSuccess) w->multimem = mc;
  }
  windows_.push_back(w);
  everyWindow_.push_back(w);
  return w;
}

void Comm::freeWindow(const std::shared_ptr<Window>& w) {
  live();
  TIDE_CHECK(w && w->comm == comm_ && w != sync_ && w != rsSync_, "not a window of this communicator");
  auto it = std::find(windows_.begin(), windows_.end(), w);
  TIDE_CHECK(it != windows_.end(), "window already freed");
  w->freed = true;
  for (auto* v : {&landings_, &ownShards_})
    v->erase(std::remove_if(v->begin(), v->end(), [&](const Range& r) { return r.winId == w->id; }), v->end());
  for (auto& a : stageAreas_) a.live = a.live && a.winId != w->id;
  for (auto& a : rsAreas_) a.live = a.live && a.winId != w->id;
  windows_.erase(it);
}

// Stage areas that overlap share one credit channel. An area that would bridge two existing
// channels is refused: their credits could then no longer vouch for each other's bytes. A new
// area takes over a freed channel (counters included) before a fresh one is spent.
int Comm::stageSlotOf(const Window& w, size_t begin, size_t end) {
  int hit = -1, dead = -1;
  for (size_t i = 0; i < stageAreas_.size(); ++i) {
    const StageArea& a = stageAreas_[i];
    if (!a.live) {
      if (dead < 0) dead = static_cast<int>(i);
      continue;
    }
    if (a.winId != w.id || begin >= a.end || a.begin >= end) continue;
    TIDE_CHECK(hit < 0, "stage area overlaps two distinct stage areas");
    hit = static_cast<int>(i);
  }
  if (hit >= 0) {
    stageAreas_[hit].begin = std::min(stageAreas_[hit].begin, begin);
    stageAreas_[hit].end = std::max(stageAreas_[hit].end, end);
    return hit;
  }
  if (dead >= 0) {
    StageArea& a = stageAreas_[dead];
    a.winId = w.id;
    a.begin = begin;
    a.end = end;
    a.live = true;
    return dead;
  }
  checkSignalBudget(stageAreas_.size() + 1, rsAreas_.size());
  stageAreas_.push_back({{w.id, begin, end}});
  return static_cast<int>(stageAreas_.size()) - 1;
}

void Comm::checkSignalBudget(size_t agAreas, size_t rsAreas) const {
  TIDE_CHECK(kSigCredit0 + agAreas + rsAreas + 1 <= static_cast<size_t>(rmaSig_),
             "out of signal indices: raise num_rma_sig (3 + one per distinct stage area of either collective)");
}

void Comm::initSync() {
  const size_t flagBytes = alignUp(static_cast<size_t>(nranks_) * sizeof(uint64_t), 16);
  readyOff_ = 0;
  doneOff_ = flagBytes;
  sync_ = allocWindow(2 * flagBytes, /*rail=*/false);  // node-local flags (memops on LSA pointers)
  lsaPeers_.clear();
  for (int q = 0; q < nranks_; ++q)
    if (q != rank_ && sync_->peer[q] != nullptr) lsaPeers_.push_back(q);
  // Node topology from LSA reachability; ranks are node-major (torchrun: node_rank*nproc + local).
  localRanks_ = static_cast<int>(lsaPeers_.size()) + 1;
  TIDE_CHECK(nranks_ % localRanks_ == 0, "world size not a multiple of the LSA team size");
  nNodes_ = nranks_ / localRanks_;
  node_ = rank_ / localRanks_;
  for (int q : lsaPeers_) TIDE_CHECK(q / localRanks_ == node_, "LSA peers must share my node index (node-major ranks)");
  memops_.assign(std::max<size_t>(1, lsaPeers_.size()), CUstreamBatchMemOpParams{});
}

// Fences are stream memops: flag[rank] := value written straight into every LSA peer's sync
// window (a copy-engine write would queue behind a running scatter), and GEQ waits on our own.
void Comm::fillFlagParams(const Window& w, size_t offBytes, uint64_t value, bool write) {
  for (size_t i = 0; i < lsaPeers_.size(); ++i) {
    auto& op = memops_[i];
    op = CUstreamBatchMemOpParams{};
    if (write) {
      op.writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
      op.writeValue.address = reinterpret_cast<CUdeviceptr>(static_cast<char*>(w.peer[lsaPeers_[i]]) + offBytes +
                                                            static_cast<size_t>(rank_) * sizeof(uint64_t));
      op.writeValue.value64 = value;
      op.writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    } else {
      op.waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
      op.waitValue.address = reinterpret_cast<CUdeviceptr>(static_cast<char*>(w.base) + offBytes +
                                                           static_cast<size_t>(lsaPeers_[i]) * sizeof(uint64_t));
      op.waitValue.value64 = value;
      op.waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
    }
  }
}

void Comm::writeFlags(const Window& w, size_t offBytes, uint64_t value, cudaStream_t s) {
  ProfScope prof(RsHostProf::memop);
  if (lsaPeers_.empty()) return;
  fillFlagParams(w, offBytes, value, /*write=*/true);
  CUCHECK(cuStreamBatchMemOp(s, static_cast<unsigned>(lsaPeers_.size()), memops_.data(), 0));
}

void Comm::waitFlags(const Window& w, size_t offBytes, uint64_t value, cudaStream_t s) {
  ProfScope prof(RsHostProf::memop);
  if (lsaPeers_.empty()) return;
  fillFlagParams(w, offBytes, value, /*write=*/false);
  CUCHECK(cuStreamBatchMemOp(s, static_cast<unsigned>(lsaPeers_.size()), memops_.data(), 0));
}

// One NIC round: every spec's ranges go out in one NCCL group (k targets at once for the k-way
// reduce-scatter rounds), then the heads of every source are awaited, the next round's sources
// are released (pace), then the tails.
void Comm::putRounds(const Window& w, const std::vector<PutSpec>& specs, size_t tail, int sigData, const std::vector<int>& paceTo,
                     int sigPace, cudaStream_t s, const std::function<void()>& beforeSignal,
                     const std::vector<PutSpec>* arrivals) {
  ProfScope prof(RsHostProf::nccl);
  // arrivals per (source, context): a put's signal lands on (from, sigData, ctx); a context's puts arrive in order
  std::vector<std::array<int, 4>> heads, tails;
  auto expect = [&](std::vector<std::array<int, 4>>& v, int from, int c) {
    for (auto& d : v)
      if (d[1] == from && d[3] == c) {
        ++d[0];
        return;
      }
    v.push_back({1, from, sigData, c});
  };
  {
    Group grp;  // one launch for the whole round
    for (const PutSpec& sp : specs)
      for (size_t c = 0; c < sp.chunks->size(); ++c) {
        const auto [off, bytes] = (*sp.chunks)[c];
        const size_t head = tail > 0 && bytes > tail ? bytes - tail : bytes;
        putSignal(w, sp.srcBase + off, head, sp.to, sp.dstBase + off, sigData, static_cast<int>(c), s);
        if (head < bytes)
          putSignal(w, sp.srcBase + off + head, bytes - head, sp.to, sp.dstBase + off + head, sigData, static_cast<int>(c), s);
      }
    grp.end();
  }
  for (const PutSpec& sp : arrivals ? *arrivals : specs)
    for (size_t c = 0; c < sp.chunks->size(); ++c) {
      const size_t bytes = (*sp.chunks)[c].second;
      expect(heads, sp.from, static_cast<int>(c));
      if (tail > 0 && bytes > tail) expect(tails, sp.from, static_cast<int>(c));
    }
  waitSignal(heads, s);
  if (!paceTo.empty()) {
    if (beforeSignal) beforeSignal();
    Group grp;
    for (int p : paceTo) signal(p, sigPace, /*ctx=*/0, s);
    grp.end();
  }
  waitSignal(tails, s);
}

void Comm::putSignal(const Window& w, size_t srcOff, size_t bytes, int peer, size_t peerOff, int sigIdx, int ctx,
                     cudaStream_t s) {
  live();
  TIDE_CHECK(!failed_ && w.comm == comm_ && !w.freed, "communicator failed earlier, or not a live window of it");
  TIDE_CHECK(railComm_ && w.railWin, "put on a window without a rail registration (single node, or a flags window)");
  TIDE_CHECK(srcOff <= w.bytes && bytes <= w.bytes - srcOff && peerOff <= w.bytes && bytes <= w.bytes - peerOff,
             "put out of window");
  // Callers reuse the source once s passes this call: NCCL documents localbuff as reusable at the put's completion on
  // the stream. `peer` is a global rank (a rail peer); the rail communicator knows it as its node index.
  NCCLCHECK(ncclPutSignal(static_cast<char*>(w.base) + srcOff, bytes, ncclUint8, railRank(peer), w.railWin, peerOff,
                          sigIdx, ctx, 0, railComm_->comm, s));
}

void Comm::signal(int peer, int sigIdx, int ctx, cudaStream_t s) {
  ProfScope prof(RsHostProf::nccl);
  live();
  TIDE_CHECK(!failed_ && railComm_, "communicator failed earlier, or single node");
  NCCLCHECK(ncclSignal(railRank(peer), sigIdx, ctx, 0, railComm_->comm, s));
}

void Comm::waitSignal(const std::vector<std::array<int, 4>>& descs, cudaStream_t s) {
  ProfScope prof(RsHostProf::nccl);
  live();
  TIDE_CHECK(!failed_, "communicator failed earlier");
  std::vector<ncclWaitSignalDesc_t> d;
  for (const auto& x : descs) {
    if (x[0] == 0) continue;
    ncclWaitSignalDesc_t w{};
    w.opCnt = x[0];
    w.peer = railRank(x[1]);
    w.sigIdx = x[2];
    w.ctx = x[3];
    d.push_back(w);
  }
  if (d.empty()) return;
  TIDE_CHECK(railComm_, "wait on a signal on a single node");
  NCCLCHECK(ncclWaitSignal(static_cast<int>(d.size()), d.data(), railComm_->comm, s));
}

}  // namespace tide
