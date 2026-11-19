// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "collective/rs/kernels.h"
#include "util/check.h"

namespace tide {

namespace {

template <typename T> __device__ __forceinline__ T zeroOf();
template <> __device__ __forceinline__ float zeroOf<float>() { return 0.f; }
template <> __device__ __forceinline__ __nv_bfloat16 zeroOf<__nv_bfloat16>() { return __float2bfloat16(0.f); }

struct F8 {
  float v[kRsVec];
};

// Raw 16 B-vector loads of kRsVec elements (one uint4 for bf16, two for fp32); decoding happens
// after every load of a step has been issued, so the loads overlap instead of each waiting on
// its conversion.
template <typename T>
struct Raw {
  static constexpr int N = static_cast<int>(sizeof(T) * kRsVec / 16);
  uint4 q[N];
};

template <typename T>
__device__ __forceinline__ Raw<T> ld16(const T* p) {
  Raw<T> r;
#pragma unroll
  for (int i = 0; i < Raw<T>::N; ++i) r.q[i] = __ldg(reinterpret_cast<const uint4*>(p) + i);
  return r;
}

__device__ __forceinline__ float4 ldg16f(const float* p) {
  float4 v;
  asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void stg16f(float* p, const float4& v) {
  asm volatile("st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

__device__ __forceinline__ F8 decode(const Raw<float>& r) {
  F8 f;
  f.v[0] = __uint_as_float(r.q[0].x); f.v[1] = __uint_as_float(r.q[0].y);
  f.v[2] = __uint_as_float(r.q[0].z); f.v[3] = __uint_as_float(r.q[0].w);
  f.v[4] = __uint_as_float(r.q[1].x); f.v[5] = __uint_as_float(r.q[1].y);
  f.v[6] = __uint_as_float(r.q[1].z); f.v[7] = __uint_as_float(r.q[1].w);
  return f;
}

// sum[k] += element k, rounding once to fp32 exactly like convert + fp32 add (bf16 -> fp32 is
// exact), so sm_90 and sm_100+ give the same bits; the mixed-precision add is one SASS instruction
// per element (sm_100+) where convert + add is two.
__device__ __forceinline__ void accum(float* sum, const Raw<__nv_bfloat16>& r) {
#if __CUDA_ARCH__ >= 1000
  const uint32_t w[4] = {r.q[0].x, r.q[0].y, r.q[0].z, r.q[0].w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    unsigned short lo, hi;
    asm("mov.b32 {%0,%1}, %2;" : "=h"(lo), "=h"(hi) : "r"(w[i]));
    asm("add.rn.f32.bf16 %0, %1, %0;" : "+f"(sum[2 * i]) : "h"(lo));
    asm("add.rn.f32.bf16 %0, %1, %0;" : "+f"(sum[2 * i + 1]) : "h"(hi));
  }
#else
  const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&r.q[0]);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 t = __bfloat1622float2(b[k]);
    sum[2 * k] += t.x;
    sum[2 * k + 1] += t.y;
  }
#endif
}
__device__ __forceinline__ void accum(float* sum, const Raw<float>& r) {
  const F8 f = decode(r);
#pragma unroll
  for (int k = 0; k < kRsVec; ++k) sum[k] += f.v[k];
}

constexpr int kMaxL = kRsMaxLocal;  // local ranks a staging slot can hold sources for (host-checked)

// The sum order -- staged slots in order (source rank r = s + (s >= me)), then the own piece -- is
// fixed per rank, hence bitwise reproducible. A vector of kRsVec staging elements never straddles two
// segments (segment staging offsets are kRsVec-aligned); elements past a piece's real extent (dim-0
// padding) are zero and nothing is written past pieceElems (the next segment's output).
// vecMeta: per-vector addressing for a tile that straddles a segment boundary -- segment lookup (p is
// a cursor that only moves forward within one destination), real/written element counts, output
// pointer (packed, or staging-indexed with padding written for nDest > 1) and the own piece's address.
template <typename T>
__device__ __forceinline__ void vecMeta(const RsSumArgs& a, const RsSegDev* segs, int& p, int l, size_t x, bool live,
                                        int& nreal, int& nwrite, float*& outp, const T*& ownp, int& seg) {
  while (p + 1 < a.nseg && segs[p + 1].stgOff <= x) ++p;
  const RsSegDev sg = segs[p];
  const size_t i0 = x - sg.stgOff;
  live = live && i0 < sg.pieceElems;  // alignment padding between segments
  const long long pieceStart = static_cast<long long>(a.destRank + l) * sg.pieceElems;
  const long long rc = static_cast<long long>(sg.numel) - pieceStart - static_cast<long long>(i0);  // real elements from i0 on
  const long long wc = static_cast<long long>(sg.pieceElems) - static_cast<long long>(i0);
  nreal = live ? static_cast<int>(rc < 0 ? 0 : (rc > kRsVec ? kRsVec : rc)) : 0;
  nwrite = live ? static_cast<int>(wc > kRsVec ? kRsVec : wc) : 0;
  if (a.nDest > 1) nwrite = live ? kRsVec : 0;  // staging-indexed block: padding positions get zeros
  nreal = nreal < nwrite ? nreal : nwrite;
  outp = a.nDest > 1 ? a.out + static_cast<size_t>(l) * a.outDestStride + (x - a.subBegin) : a.out + sg.outOff + i0;
  ownp = reinterpret_cast<const T*>(a.src[p]) + pieceStart + i0;
  seg = p;
}

// ---------------------------------------------------------------------------------------------
// The sum kernel: the staged sources are streamed into shared memory with cp.async.bulk, so their
// bytes in flight hold no registers; the own piece is a register-prefetched 16 B load, and the
// threads only read shared memory and add.

__device__ __forceinline__ uint32_t smemU32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbarInit(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemU32(bar)), "r"(count));
}
__device__ __forceinline__ void mbarInitFence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbarExpectTx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemU32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulkG2S(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smemU32(dst)),
               "l"(src), "r"(bytes), "r"(smemU32(bar))
               : "memory");
}
__device__ __forceinline__ void mbarArrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smemU32(bar)) : "memory");
}
__device__ __forceinline__ bool mbarTryWait(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok)
      : "r"(smemU32(bar)), "r"(phase)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t phase) {
  while (!mbarTryWait(bar, phase)) {
  }
}

// Tile = kThr threads x kRsVec elements per source (8 KB either dtype); a CTA owns a contiguous
// tile range. The consumer is latency-bound with few warps (ncu, 8 warps: 28% issue utilization =
// 2 warps per scheduler, long-scoreboard stalls on the own-piece loads), so bf16 runs 16 warps per
// CTA and the own piece is register-prefetched kOwnAhead tiles ahead instead of one.
// kS staged sources: kMaxL - 1 for a local team (the own piece is the L-th), kMaxL for a rail round of
// kMaxL blocks (192 KB of stages).
template <typename T, int kS = kMaxL - 1>
struct TmaCfg {
  static constexpr int kThr = sizeof(T) == 2 ? 512 : 256;  // one 16 B vector per thread per source per tile
  static constexpr int kTileElems = kThr * kRsVec;
  static constexpr int kTileBytes = kTileElems * static_cast<int>(sizeof(T));  // 8 KB either way
  static constexpr int kStages = 3;
  static constexpr int kSrc = kS;
  static constexpr int kOwnAhead = 4;
  static constexpr int kRing = kOwnAhead + 2;  // tile descriptors in flight
  static constexpr size_t kSmem =
      static_cast<size_t>(kStages) * kSrc * kTileBytes + (2 * kStages + kRing) * sizeof(uint64_t) + 128;
};

struct VecMeta {
  float* outp;
  int nreal, nwrite;
};

// Own piece kept as raw bits until the tile is summed: decoding right after the load would stall
// the warp on the load's latency at prefetch time.
template <typename T>
union OwnRaw {
  Raw<T> r;
  T e[kRsVec];
};

// Per-tile addressing, computed once by the producer warp and read by every consumer thread: for a
// tile that lies within one segment (the common case) a thread's vector is `off = tid*8` elements
// past the tile start and every quantity is an add or a clamp from these; a tile that straddles a
// segment boundary falls back to the per-vector lookup (`vecMeta`, cursor from `seg`).
template <typename T>
struct TileDesc {
  const T* own0;    // own piece at the tile start
  float* out0;      // output at the tile start
  long long rc0;    // real elements from the tile start (numel - pieceStart - i0)
  long long wc0;    // piece elements from the tile start (pieceElems - i0)
  size_t x0;        // staging offset at the tile start
  int len;          // elements of this tile inside the chunk
  int seg, l;       // segment at the tile start, destination
  int uniform;      // no segment boundary inside the tile
  int ownAligned;   // own0 is 16 B aligned (so every vector of the tile is)
};

template <typename T, bool kFull, int kS = kMaxL - 1>  // kFull: L == kS + 1, every staged slot live (compile-time source loop)
__global__ void __launch_bounds__(TmaCfg<T, kS>::kThr + 32) rsSumTmaKernel(RsSumArgs a) {
  using C = TmaCfg<T, kS>;
  constexpr int D = C::kOwnAhead, kRing = C::kRing;
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ RsSegDev segs[kRsMaxSegments];
  __shared__ TileDesc<T> descs[kRing];
  // Pipeline barriers: full[s] (TMA bytes landed), empty[s] (every consumer warp done reading the
  // stage), descBar[r] (descriptor written). Consumers and the producer never block-synchronize.
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + static_cast<size_t>(C::kStages) * C::kSrc * C::kTileBytes);
  uint64_t* empty = full + C::kStages;
  uint64_t* descBar = empty + C::kStages;
  for (int i = threadIdx.x; i < a.nseg; i += blockDim.x) segs[i] = a.segs[i];
  __syncthreads();  // the producer reads the segment table before the first barrier below
  const bool producer = threadIdx.x == C::kThr;  // lane 0 of the extra warp: TMA issue + tile descriptors
  const bool consumer = threadIdx.x < C::kThr;
  constexpr uint32_t kConsumerWarps = C::kThr / 32;

  const size_t total = a.chunkEnd - a.chunkBegin;
  // Tiles are flattened over (destination l, tile t): the CTA owns a contiguous range of them.
  const size_t tilesPerDest = (total + C::kTileElems - 1) / C::kTileElems;
  const size_t tilesAll = tilesPerDest * static_cast<size_t>(a.nDest);
  const size_t perCta = (tilesAll + gridDim.x - 1) / gridDim.x;
  const size_t t0 = static_cast<size_t>(blockIdx.x) * perCta;
  const size_t t1 = t0 + perCta < tilesAll ? t0 + perCta : tilesAll;
  if (t0 >= t1) return;
  const int ntiles = static_cast<int>(t1 - t0);
  const int nst = kFull ? C::kSrc : a.L - 1;
  const size_t srcStride = a.srcStride / sizeof(T), destStride = a.destStride / sizeof(T);
  const T* stageBase = reinterpret_cast<const T*>(a.stage);

  // Producer-side tile cursors: tiles are issued and described in increasing order, so (l, tt) is
  // tracked incrementally (one 64-bit division per kernel instead of per tile).
  struct Cursor {
    size_t l, tt;
  };
  auto advance = [&](Cursor& c) {
    if (++c.tt == tilesPerDest) {
      c.tt = 0;
      ++c.l;
    }
  };
  Cursor issCur{t0 / tilesPerDest, 0};
  issCur.tt = t0 - issCur.l * tilesPerDest;
  Cursor descCur = issCur;
  auto issue = [&](int stage) {  // producer: one bulk copy per staged source for the next tile
    const size_t e0 = issCur.tt * static_cast<size_t>(C::kTileElems);
    const size_t elems = total - e0 < static_cast<size_t>(C::kTileElems) ? total - e0 : C::kTileElems;
    const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(T));
    mbarExpectTx(&full[stage], bytes * static_cast<uint32_t>(nst));
    unsigned char* dst = smem + (static_cast<size_t>(stage) * C::kSrc) * C::kTileBytes;
    const T* src = stageBase + issCur.l * destStride + e0;
#pragma unroll
    for (int s = 0; s < C::kSrc; ++s) {
      if (!kFull && s >= nst) break;
      const int r = s + (s >= a.me ? 1 : 0);
      bulkG2S(dst + static_cast<size_t>(s) * C::kTileBytes, src + static_cast<size_t>(r) * srcStride, bytes, &full[stage]);
    }
    advance(issCur);
  };
  int dp = 0;
  size_t dl = ~size_t(0);  // producer's segment cursor (restarts per destination)
  auto describe = [&](int t) {  // producer: descriptor of the next tile into ring slot t % kRing
    if (descCur.l != dl) {
      dl = descCur.l;
      dp = 0;
    }
    const size_t x0 = a.chunkBegin + descCur.tt * static_cast<size_t>(C::kTileElems);
    while (dp + 1 < a.nseg && segs[dp + 1].stgOff <= x0) ++dp;
    const RsSegDev sg = segs[dp];
    TileDesc<T>& d = descs[t % kRing];
    const size_t i0 = x0 - sg.stgOff;
    const long long pieceStart = static_cast<long long>(a.destRank + dl) * sg.pieceElems;
    d.x0 = x0;
    d.len = static_cast<int>(a.chunkEnd - x0 < static_cast<size_t>(C::kTileElems) ? a.chunkEnd - x0 : C::kTileElems);
    d.uniform = (dp + 1 >= a.nseg) || (segs[dp + 1].stgOff >= x0 + static_cast<size_t>(d.len));
    d.seg = dp;
    d.l = static_cast<int>(dl);
    d.rc0 = static_cast<long long>(sg.numel) - pieceStart - static_cast<long long>(i0);
    d.wc0 = static_cast<long long>(sg.pieceElems) - static_cast<long long>(i0);
    d.own0 = reinterpret_cast<const T*>(a.src[dp]) + pieceStart + i0;
    d.out0 = a.nDest > 1 ? a.out + dl * a.outDestStride + (x0 - a.subBegin) : a.out + sg.outOff + i0;
    d.ownAligned = reinterpret_cast<uintptr_t>(d.own0) % 16 == 0;
    mbarArrive(&descBar[t % kRing]);  // release: the descriptor is visible to consumers that acquire it
    advance(descCur);
  };
  if (producer) {
    for (int s = 0; s < C::kStages; ++s) {
      mbarInit(&full[s], 1);
      mbarInit(&empty[s], kConsumerWarps);
    }
    for (int r = 0; r < kRing; ++r) mbarInit(&descBar[r], 1);
    mbarInitFence();
  }
  __syncthreads();
  if (producer) {
    for (int t = 0; t <= D && t < ntiles; ++t) describe(t);
    for (int t = 0; t < C::kStages && t < ntiles; ++t) issue(t);
    // Refill stage t % kStages with tile t + kStages once every consumer warp released tile t;
    // describe tile t + D + 1 first so consumers' prefetch never waits on this thread.
    for (int t = 0; t < ntiles; ++t) {
      if (t + D + 1 < ntiles) describe(t + D + 1);
      if (t + C::kStages < ntiles) {
        mbarWait(&empty[t % C::kStages], static_cast<uint32_t>(t / C::kStages) & 1u);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic reads before the async-proxy write
        issue(t % C::kStages);
      }
    }
    return;
  }
  if (!consumer) return;

  // finalMode with scale 1 and no accumulation is a plain store (skips the scale FMAs).
  const bool plainStore = !a.accumulate && a.scale == 1.f;

  // Consumer: vector metadata for tile t from its descriptor (cheap path) or the per-vector lookup.
  auto meta = [&](int t, VecMeta& m, OwnRaw<T>& own) {
    mbarWait(&descBar[t % kRing], static_cast<uint32_t>(t / kRing) & 1u);
    const TileDesc<T>& d = descs[t % kRing];
    const int off = threadIdx.x * kRsVec;
    const T* ownp;
    if (d.uniform) {
      const bool live = off < d.len && static_cast<long long>(off) < d.wc0;
      const long long rc = d.rc0 - off, wc = d.wc0 - off;
      m.nreal = live ? static_cast<int>(rc < 0 ? 0 : (rc > kRsVec ? kRsVec : rc)) : 0;
      m.nwrite = live ? (a.nDest > 1 ? kRsVec : static_cast<int>(wc > kRsVec ? kRsVec : wc)) : 0;
      m.nreal = m.nreal < m.nwrite ? m.nreal : m.nwrite;
      m.outp = d.out0 + off;
      ownp = d.own0 + off;
      if (a.skipOwn) {
        own.r = Raw<T>{};
        return;
      }
      if (d.ownAligned && m.nreal == kRsVec) {
        own.r = ld16(ownp);
        return;
      }
    } else {
      int p = d.seg, seg;
      vecMeta<T>(a, segs, p, d.l, d.x0 + off, off < d.len, m.nreal, m.nwrite, m.outp, ownp, seg);
      if (a.skipOwn) {
        own.r = Raw<T>{};
        return;
      }
      if ((reinterpret_cast<uintptr_t>(ownp) % 16 == 0) && m.nreal == kRsVec) {
        own.r = ld16(ownp);
        return;
      }
    }
#pragma unroll
    for (int k = 0; k < kRsVec; ++k) own.e[k] = k < m.nreal ? ownp[k] : zeroOf<T>();
  };

  auto consume = [&](int t, const VecMeta& m, const OwnRaw<T>& own) {
    const int stage = t % C::kStages;
    mbarWait(&full[stage], static_cast<uint32_t>(t / C::kStages) & 1u);
    const unsigned char* tile = smem + (static_cast<size_t>(stage) * C::kSrc) * C::kTileBytes;
    float sum[kRsVec];
#pragma unroll
    for (int k = 0; k < kRsVec; ++k) sum[k] = 0.f;
    if (m.nwrite != 0) {  // (warp-uniform in practice; the release below is unconditional)
      Raw<T> r[C::kSrc];
#pragma unroll
      for (int s = 0; s < C::kSrc; ++s) {
        const int ss = kFull ? s : (s < nst ? s : nst - 1);
#pragma unroll
        for (int q = 0; q < Raw<T>::N; ++q)
          r[s].q[q] = reinterpret_cast<const uint4*>(tile + static_cast<size_t>(ss) * C::kTileBytes)[threadIdx.x * Raw<T>::N + q];
      }
#pragma unroll
      for (int s = 0; s < C::kSrc; ++s)
        if (kFull || s < nst) accum(sum, r[s]);
      accum(sum, own.r);  // own piece last
    }
    __syncwarp();  // every lane's stage reads have completed (their values were consumed above)
    if ((threadIdx.x & 31) == 0) mbarArrive(&empty[stage]);
    if (m.nwrite != 0) {
      float* o = m.outp;
      if (m.nwrite == kRsVec && m.nreal == kRsVec && (reinterpret_cast<uintptr_t>(o) % 16 == 0)) {
        float4 r0, r1;
        if (a.finalMode && !plainStore) {
          float4 o0{0.f, 0.f, 0.f, 0.f}, o1{0.f, 0.f, 0.f, 0.f};
          if (a.accumulate) {
            o0 = ldg16f(o);
            o1 = ldg16f(o + 4);
          }
          r0 = float4{o0.x + a.scale * sum[0], o0.y + a.scale * sum[1], o0.z + a.scale * sum[2], o0.w + a.scale * sum[3]};
          r1 = float4{o1.x + a.scale * sum[4], o1.y + a.scale * sum[5], o1.z + a.scale * sum[6], o1.w + a.scale * sum[7]};
        } else {
          r0 = float4{sum[0], sum[1], sum[2], sum[3]};
          r1 = float4{sum[4], sum[5], sum[6], sum[7]};
        }
        stg16f(o, r0);
        stg16f(o + 4, r1);
      } else {
#pragma unroll
        for (int k = 0; k < kRsVec; ++k) {
          if (k >= m.nwrite) break;
          const float v = k < m.nreal ? sum[k] : 0.f;  // padding rows and staged garbage past the piece
          o[k] = a.finalMode ? (a.accumulate ? o[k] : 0.f) + a.scale * v : v;
        }
      }
    }
  };

  // Own-piece prefetch ring of depth D with compile-time slot indices: slot d holds tile tb + d.
  VecMeta m[D];
  OwnRaw<T> own[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < ntiles) meta(d, m[d], own[d]);
  for (int tb = 0; tb < ntiles; tb += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int t = tb + d;
      if (t < ntiles) {
        consume(t, m[d], own[d]);
        if (t + D < ntiles) meta(t + D, m[d], own[d]);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Multicast landing. One multimem.ld_reduce per 16 B returns the fp32 sum of the L local ranks'
// inbox blocks at that offset (the switch adds); the per-CTA return rate is bound by bytes in
// flight against the ~2 us NVLink round trip, so 32 warps issue 8 loads each before any store
// (48 GB/s per CTA; two CTAs reach the 8-issuer fabric ceiling of ~90 GB/s per GPU).
constexpr int kMcThreads = 1024, kMcUnroll = 8;

__device__ __forceinline__ uint4 mcLoad(const float* p) {
  uint4 v;
  asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ void landBlock(const RsMcArgs& a, const RsSegDev* segs, int p);

// One-dimensional grid: every CTA takes its share of every block p in turn (a wave's landing is
// one launch of `ctas` CTAs whatever the block count; the SM budget is the CTA count, not L x it).
__global__ void __launch_bounds__(kMcThreads) rsMcKernel(RsMcArgs a) {
  __shared__ RsSegDev segs[kRsMaxSegments];
  for (int i = threadIdx.x; i < a.nseg; i += blockDim.x) segs[i] = a.segs[i];
  __syncthreads();
  const int p0 = a.only >= 0 ? a.only : 0, p1 = a.only >= 0 ? a.only + 1 : a.blocks;
  for (int p = p0; p < p1; ++p) landBlock(a, segs, p);
}

__device__ __forceinline__ void landBlock(const RsMcArgs& a, const RsSegDev* segs, int p) {
  const size_t subStart = static_cast<size_t>(p) * a.subElems;
  if (subStart >= a.stgElems) return;
  const size_t subEnd = a.stgElems - subStart < a.subElems ? a.stgElems : subStart + a.subElems;
  const size_t subBegin = subStart + a.partBegin;  // staging start of the block's range
  if (subBegin >= subEnd) return;
  const size_t n = subEnd - subBegin < a.partElems ? subEnd - subBegin : a.partElems;
  const float* mc = a.mc + static_cast<size_t>(p) * a.blockStride;
  const size_t nq = n / 4;  // 16 B groups; a group never straddles a segment (staging offsets are 8-aligned)
  const size_t perCta = (nq + gridDim.x - 1) / gridDim.x;
  const size_t q0 = static_cast<size_t>(blockIdx.x) * perCta, q1 = q0 + perCta < nq ? q0 + perCta : nq;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
  constexpr int kWarpGroups = kMcUnroll * 32;  // a warp's iteration covers 4 KB of contiguous groups
  int seg = 0;  // warp-uniform segment cursor: a warp's ranges are increasing
  auto store = [&](float* o, int nwrite, const uint4& v) {
    const float f[4] = {__uint_as_float(v.x), __uint_as_float(v.y), __uint_as_float(v.z), __uint_as_float(v.w)};
    if (nwrite == 4 && (reinterpret_cast<uintptr_t>(o) & 15) == 0) {
      float4 r;
      if (a.finalMode) {
        float4 c{0.f, 0.f, 0.f, 0.f};
        if (a.accumulate) c = *reinterpret_cast<const float4*>(o);
        r = float4{c.x + a.scale * f[0], c.y + a.scale * f[1], c.z + a.scale * f[2], c.w + a.scale * f[3]};
      } else {
        r = float4{f[0], f[1], f[2], f[3]};
      }
      *reinterpret_cast<float4*>(o) = r;
    } else {
      for (int k = 0; k < nwrite; ++k) o[k] = a.finalMode ? (a.accumulate ? o[k] : 0.f) + a.scale * f[k] : f[k];
    }
  };
  for (size_t wbase = q0 + static_cast<size_t>(warp) * kWarpGroups; wbase < q1;
       wbase += static_cast<size_t>(nwarps) * kWarpGroups) {
    const size_t base = wbase + lane;
    uint4 v[kMcUnroll];
#pragma unroll
    for (int u = 0; u < kMcUnroll; ++u) {
      const size_t q = base + static_cast<size_t>(u) * 32;
      v[u] = q < q1 ? mcLoad(mc + q * 4) : make_uint4(0, 0, 0, 0);
    }
    // Segment lookup once per warp range; the per-group work is an add and a compare unless a
    // segment boundary falls inside the range (then the per-group cursor path below).
    const size_t xw0 = subBegin + wbase * 4;
    const size_t wend = wbase + kWarpGroups < q1 ? wbase + kWarpGroups : q1;
    while (seg + 1 < a.nseg && segs[seg + 1].stgOff <= xw0) ++seg;
    const bool uniform = seg + 1 >= a.nseg || segs[seg + 1].stgOff >= subBegin + wend * 4;
    if (uniform) {
      const size_t stgOff = segs[seg].stgOff, pe = segs[seg].pieceElems;
      float* outSeg = a.out + segs[seg].outOff;
#pragma unroll
      for (int u = 0; u < kMcUnroll; ++u) {
        const size_t q = base + static_cast<size_t>(u) * 32;
        if (q >= q1) continue;
        const size_t i0 = subBegin + q * 4 - stgOff;
        if (i0 >= pe) continue;  // alignment gap between segments
        store(outSeg + i0, pe - i0 < 4 ? static_cast<int>(pe - i0) : 4, v[u]);
      }
    } else {
      int sg = seg;
#pragma unroll
      for (int u = 0; u < kMcUnroll; ++u) {
        const size_t q = base + static_cast<size_t>(u) * 32;
        if (q >= q1) continue;
        const size_t x = subBegin + q * 4;
        while (sg + 1 < a.nseg && segs[sg + 1].stgOff <= x) ++sg;
        const size_t i0 = x - segs[sg].stgOff, pe = segs[sg].pieceElems;
        if (i0 >= pe) continue;
        store(a.out + segs[sg].outOff + i0, pe - i0 < 4 ? static_cast<int>(pe - i0) : 4, v[u]);
      }
    }
  }
}

template <typename T, bool kFull, int kS>
void allowSmem() {
  CUDACHECK(cudaFuncSetAttribute(rsSumTmaKernel<T, kFull, kS>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(TmaCfg<T, kS>::kSmem)));
}

}  // namespace

// cp.async.bulk needs sm_90+; a rail round of kMaxL staged blocks needs ~207 KB of shared memory per
// CTA (the 8 KB static segment table on top of kSmem), which sm_80's 164 KB opt-in does not hold.
void rsRequireTma() {
  static const bool ok = [] {
    int dev = 0, major = 0, optin = 0;
    CUDACHECK(cudaGetDevice(&dev));
    CUDACHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDACHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    const size_t need =
        std::max(TmaCfg<__nv_bfloat16, kMaxL>::kSmem, TmaCfg<float, kMaxL>::kSmem) + kRsMaxSegments * sizeof(RsSegDev);
    TIDE_CHECK(major >= 9 && static_cast<size_t>(optin) >= need,
               "the reduce-scatter needs sm_90+ (cp.async.bulk) and " + std::to_string(need) +
                   " B of shared memory per block; this device is sm_" + std::to_string(major) + "x with " +
                   std::to_string(optin) + " B");
    allowSmem<__nv_bfloat16, true, kMaxL - 1>();
    allowSmem<__nv_bfloat16, false, kMaxL - 1>();
    allowSmem<float, true, kMaxL - 1>();
    allowSmem<float, false, kMaxL - 1>();
    allowSmem<__nv_bfloat16, true, kMaxL>();
    allowSmem<float, true, kMaxL>();
    return true;
  }();
  (void)ok;
}

void launchRsSum(const RsSumArgs& a, int ctas, cudaStream_t s) {
  if (a.chunkEnd <= a.chunkBegin) return;
  rsRequireTma();
  TIDE_CHECK(a.L >= 1 && a.L <= kMaxL + 1, "sum kernel: " + std::to_string(a.L) + " sources do not fit");
  if (a.L == kMaxL + 1) {  // a rail round of kMaxL blocks
    if (a.itemsize == 2)
      rsSumTmaKernel<__nv_bfloat16, true, kMaxL>
          <<<ctas, TmaCfg<__nv_bfloat16, kMaxL>::kThr + 32, TmaCfg<__nv_bfloat16, kMaxL>::kSmem, s>>>(a);
    else
      rsSumTmaKernel<float, true, kMaxL><<<ctas, TmaCfg<float, kMaxL>::kThr + 32, TmaCfg<float, kMaxL>::kSmem, s>>>(a);
  } else {
    const bool full = a.L == kMaxL;
    if (a.itemsize == 2) {
      if (full) rsSumTmaKernel<__nv_bfloat16, true><<<ctas, TmaCfg<__nv_bfloat16>::kThr + 32, TmaCfg<__nv_bfloat16>::kSmem, s>>>(a);
      else rsSumTmaKernel<__nv_bfloat16, false><<<ctas, TmaCfg<__nv_bfloat16>::kThr + 32, TmaCfg<__nv_bfloat16>::kSmem, s>>>(a);
    } else {
      if (full) rsSumTmaKernel<float, true><<<ctas, TmaCfg<float>::kThr + 32, TmaCfg<float>::kSmem, s>>>(a);
      else rsSumTmaKernel<float, false><<<ctas, TmaCfg<float>::kThr + 32, TmaCfg<float>::kSmem, s>>>(a);
    }
  }
  CUDACHECK(cudaGetLastError());
}

void launchRsMc(const RsMcArgs& a, int ctas, cudaStream_t s) {
  TIDE_CHECK(a.subElems % 4 == 0 && a.partBegin % 4 == 0 && a.partElems % 4 == 0 && a.blockStride % 4 == 0 &&
                 reinterpret_cast<uintptr_t>(a.mc) % 16 == 0,
             "multicast landing needs 16 B-aligned blocks");
  rsMcKernel<<<ctas, kMcThreads, 0, s>>>(a);
  CUDACHECK(cudaGetLastError());
}

}  // namespace tide
