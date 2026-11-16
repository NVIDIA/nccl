// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Internal to the Comm translation units (runtime/comm.cpp, collective/*/issue.cpp).
#pragma once

#include <nccl.h>
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
#include "util/check.h"
#include "runtime/copy.h"

namespace tide::detail {


constexpr size_t kWindowAlign = 2u << 20;  // ncclMemAlloc granularity is >= 2 MiB

// Host-RMA signal indices: data arrival, round pacing, then one credit channel per stage area.
constexpr int kSigData = 0, kSigPace = 1, kSigCredit0 = 2;

inline size_t alignUp(size_t x, size_t a) {
  TIDE_CHECK(x <= SIZE_MAX - a, "size overflows size_t");
  return (x + a - 1) / a * a;
}

// Runs f with `dev` current (teardown paths may be reached from any device context).
template <typename F>
void onDevice(int dev, F f) {
  int saved = -1;
  cudaGetDevice(&saved);
  cudaSetDevice(dev);
  f();
  if (saved >= 0) cudaSetDevice(saved);
}

// Test-only fault injection, compiled in with -DTIDE_FAULT_INJECTION (scripts/build.sh does; a
// plain install does not and then refuses to start with TIDE_FAULT set). TIDE_FAULT is a
// comma-separated set of: nowrite (skip the data movement), nofence (skip the entry wait;
// reduce-scatter: the staging-landed wait), nodone (skip the exit wait), nocredit (skip the
// stage-credit wait), noscatter (multi-node: NIC leg only, no copy-engine scatter), nofree
// (reduce-scatter: skip the staging-slot-free wait; decoupled all-gather: `ready` does not wait for
// the release event), abort (reduce-scatter: queue a wait nothing
// satisfies, then throw -- close() must not drain), earlyread (reduce-scatter: record the read event
// before any read is issued) -- these apply on every rank -- and the
// delays, which apply on TIDE_FAULT_RANK only (default: every rank) and stall for TIDE_FAULT_US:
// slow (all-gather: the copies / each inter-node scatter; reduce-scatter: each inbox accumulate),
// slowsum (reduce-scatter: each local sum), slowpush (reduce-scatter: each push); and dcstall
// (TIDE_RS_NET=device: from the second call on, launch no device network kernels -- the peers' waits time out).
struct Fault {
  enum Mode {
    nowrite = 1, nofence = 2, nodone = 4, nocredit = 8, slow = 16, noscatter = 32, nofree = 64, slowsum = 128,
    slowpush = 256, nozero = 512, abort = 1024, earlyread = 2048, dcstall = 4096, noputwait = 8192
  };
  unsigned mask = 0;
  int rank = -1;
  int us = 1000;
};

#ifdef TIDE_FAULT_INJECTION
inline const Fault& fault() {
  static const Fault f = [] {
    Fault f;
    static const std::pair<const char*, Fault::Mode> names[] = {
        {"nowrite", Fault::nowrite}, {"nofence", Fault::nofence}, {"nodone", Fault::nodone},
        {"nocredit", Fault::nocredit}, {"slow", Fault::slow}, {"noscatter", Fault::noscatter},
        {"nofree", Fault::nofree}, {"slowsum", Fault::slowsum}, {"slowpush", Fault::slowpush}, {"nozero", Fault::nozero},
        {"abort", Fault::abort}, {"earlyread", Fault::earlyread}, {"dcstall", Fault::dcstall},
        {"noputwait", Fault::noputwait}};
    if (const char* env = std::getenv("TIDE_FAULT")) {
      std::string list(env), tok;
      for (size_t i = 0; i <= list.size(); ++i) {
        if (i < list.size() && list[i] != ',') { tok += list[i]; continue; }
        for (const auto& [name, mode] : names) if (tok == name) f.mask |= mode;
        tok.clear();
      }
    }
    if (const char* r = std::getenv("TIDE_FAULT_RANK")) f.rank = std::atoi(r);
    if (const char* u = std::getenv("TIDE_FAULT_US")) f.us = std::atoi(u);
    return f;
  }();
  return f;
}

inline bool faultOn(Fault::Mode mode, int rank) {
  const Fault& f = fault();
  const bool delay = mode == Fault::slow || mode == Fault::slowsum || mode == Fault::slowpush || mode == Fault::dcstall;
  return (f.mask & mode) && (!delay || f.rank < 0 || f.rank == rank);
}

inline void CUDART_CB hostSleep(void* us) { usleep(*static_cast<const int*>(us)); }

// Stalls `s` for TIDE_FAULT_US without touching an SM (a host callback blocks the stream).
inline void faultDelay(cudaStream_t s) { CUDACHECK(cudaLaunchHostFunc(s, hostSleep, const_cast<int*>(&fault().us))); }
constexpr bool kFaultInjection = true;
#else
constexpr bool faultOn(Fault::Mode, int) { return false; }
inline void faultDelay(cudaStream_t) {}
constexpr bool kFaultInjection = false;
#endif

// TIDE_RS_DEBUG=3: host time of a reduce-scatter call by category, printed by rank 0 every 20 calls.
// Scopes nest (putRounds calls putSignal): only the outermost one counts.
struct RsHostProf {
  enum Cat { copy, memop, nccl, kCats };
  bool on = false;
  int depth = 0;
  double us[kCats] = {}, total = 0, read = 0, readUs[kCats] = {};  // read*: TIDE_RS_PIPELINE=2's read phase
  uint64_t calls = 0;
};
inline RsHostProf g_prof;
inline double nowUs() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct ProfScope {
  int cat;
  double t0 = 0;
  bool counts;
  explicit ProfScope(int c) : cat(c), counts(g_prof.on && g_prof.depth++ == 0) { if (counts) t0 = nowUs(); }
  ~ProfScope() {
    if (!g_prof.on) return;
    if (counts) g_prof.us[cat] += nowUs() - t0;
    --g_prof.depth;
  }
};

inline void batchCopy(CopyBatch& b, cudaStream_t s) {
  ProfScope prof(RsHostProf::copy);
  if (b.len.empty()) return;
  if (s == nullptr) {  // cudaMemcpyBatchAsync rejects the legacy NULL stream
    for (size_t i = 0; i < b.len.size(); ++i)
      CUDACHECK(cudaMemcpyAsync(b.dst[i], b.src[i], b.len[i], cudaMemcpyDeviceToDevice, s));
    return;
  }
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  // Every descriptor is peer-to-peer by construction; the flag keeps a same-device copy, should one
  // sneak in, on a copy engine instead of SMs.
  attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
  size_t attrIdx = 0;
  CUDACHECK(cudaMemcpyBatchAsync(b.dst.data(), const_cast<const void* const*>(b.src.data()), b.len.data(),
                                 b.len.size(), &attr, &attrIdx, 1, s));
}

inline void batchCopy(StridedBatch& b, cudaStream_t s) {
  ProfScope prof(RsHostProf::copy);
  if (b.ops.empty()) return;
  if (s == nullptr) {
    for (const auto& op : b.ops)
      CUDACHECK(cudaMemcpy2DAsync(op.dst.op.ptr.ptr, op.dst.op.ptr.rowLength, op.src.op.ptr.ptr, op.src.op.ptr.rowLength,
                                  op.extent.width, op.extent.height, cudaMemcpyDeviceToDevice, s));
    return;
  }
  CUDACHECK(cudaMemcpy3DBatchAsync(b.ops.size(), b.ops.data(), 0, s));
}

inline void batchCopy(PushBatch& b, cudaStream_t s) {
  batchCopy(b.flat, s);
  batchCopy(b.strided, s);
}

// Rows r = 0..rows-1 of one segment's pieces for consecutive destinations: elements
// [start0 + r*stride, +width) of a gradient of `numel` elements into dst + r*dstPitch. Rows past
// `numel` are dim-0 padding: the last live row is shortened, the ones after it are not copied (the
// sum kernel reads padding as zero).
inline void addPieceRows(PushBatch& b, char* dst, size_t dstPitch, const char* src, size_t start0, size_t stride, size_t width,
                  size_t rows, size_t numel, int itemsize) {
  if (start0 >= numel) return;
  const size_t full = start0 + width <= numel ? std::min(rows, (numel - start0 - width) / stride + 1) : 0;
  b.addRows(dst, dstPitch, src + start0 * itemsize, stride * itemsize, width * itemsize, full);
  const size_t startL = start0 + full * stride;
  if (full < rows && startL < numel) b.flat.add(dst + full * dstPitch, src + startL * itemsize, (numel - startL) * itemsize);
}

// ncclGroupStart/End pair that is closed even when a call inside throws.
struct Group {
  bool open = true;
  Group() { NCCLCHECK(ncclGroupStart()); }
  void end() {
    open = false;
    NCCLCHECK(ncclGroupEnd());
  }
  ~Group() {
    if (open) ncclGroupEnd();
  }
};

}  // namespace tide::detail
