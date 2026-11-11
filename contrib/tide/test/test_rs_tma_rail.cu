// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Rail rounds (one rank per node): the sum kernel against a host reference in its order (the round's
// blocks, then the own piece in the first round; fp32 adds, the output step one fma), bitwise, for 1..8
// blocks on one GPU -- 8 blocks is the form with kRsMaxLocal staged sources, which no fleet of 8 nodes
// reaches (7 peers):
//   nvcc -std=c++17 -O2 -arch=sm_100 -Icsrc -I$NCCL_HOME/include test/test_rs_tma_rail.cu \
//       csrc/collective/rs/kernels.cu -o /tmp/test_rs_tma_rail && /tmp/test_rs_tma_rail
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "collective/rs/kernels.h"
#include "util/check.h"

using namespace tide;

namespace {

constexpr int N = 33;  // ranks of the rail group
struct Shape {
  size_t numel, dim0;
};
// The last ranks' pieces end in padding rows (5120 and 1000 rows over 33); 1023-element pieces leave an
// alignment gap and unaligned own / output pointers; 70 rows leave ranks 24.. without real rows.
const Shape kShapes[] = {{5120ull * 2560, 5120}, {1000ull * 33, 1000}, {70ull * 128, 70}};

size_t alignUp(size_t x, size_t a) { return (x + a - 1) / a * a; }

float toHost(float v) { return v; }
float toHost(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T>
T fromHost(float v);
template <>
float fromHost<float>(float v) { return v; }
template <>
__nv_bfloat16 fromHost<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

int failures = 0, cases = 0;

template <typename T>
void runDtype(std::mt19937& rng) {
  const size_t isz = sizeof(T);
  std::vector<RsSegDev> segs;
  size_t outElems = 0, stg = 0;
  for (const Shape& s : kShapes) {
    RsSegDev d{};
    d.numel = s.numel;
    d.pieceElems = (s.dim0 + N - 1) / N * (s.numel / s.dim0);
    d.outOff = outElems;
    d.stgOff = stg;
    outElems += d.pieceElems;
    stg = alignUp(stg + d.pieceElems, kRsVec);
    segs.push_back(d);
  }
  const int nseg = static_cast<int>(segs.size());
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> ex(-6, 6);
  auto draw = [&] { return std::ldexp(nd(rng), ex(rng)); };

  RsSegDev* segsDev;
  CUDACHECK(cudaMalloc(&segsDev, segs.size() * sizeof(RsSegDev)));
  CUDACHECK(cudaMemcpy(segsDev, segs.data(), segs.size() * sizeof(RsSegDev), cudaMemcpyHostToDevice));
  std::vector<std::vector<T>> own(nseg);
  std::vector<void*> ownDev(nseg);
  for (int p = 0; p < nseg; ++p) {
    own[p].resize(segs[p].numel);
    for (auto& v : own[p]) v = fromHost<T>(draw());
    CUDACHECK(cudaMalloc(&ownDev[p], own[p].size() * isz));
    CUDACHECK(cudaMemcpy(ownDev[p], own[p].data(), own[p].size() * isz, cudaMemcpyHostToDevice));
  }
  const size_t maxStride = alignUp(stg * isz, 256);
  std::vector<T> blocks(kRsMaxLocal * maxStride / isz);  // every word random, padding / gaps included
  for (auto& v : blocks) v = fromHost<T>(draw());
  char* blocksDev;
  CUDACHECK(cudaMalloc(&blocksDev, blocks.size() * isz));
  CUDACHECK(cudaMemcpy(blocksDev, blocks.data(), blocks.size() * isz, cudaMemcpyHostToDevice));
  std::vector<float> base(outElems), got(outElems), ref(outElems);
  for (auto& v : base) v = draw();
  float* outDev;
  CUDACHECK(cudaMalloc(&outDev, outElems * 4));

  const size_t part = alignUp((stg + 3) / 4, kRsVec);  // TIDE_RS_RAIL_PARTS=4
  const size_t ranges[][3] = {{0, stg, maxStride}, {part, 2 * part, alignUp(part * isz, 256)},
                              {3 * part, stg, alignUp(part * isz, 256)}};
  for (int nblk : {1, 3, 7, 8})
    for (int round : {0, 1, 2})  // first round, first round accumulating into the grads, a later round
      for (float scale : {1.f, 1.f / N})
        for (int dest : {0, 23, N - 1})
          for (const auto& rg : ranges)
            for (int ctas : {1, 16}) {
              const bool first = round < 2, accumulate = round > 0;
              const size_t x0 = rg[0], x1 = rg[1], stride = rg[2];
              RsSumArgs t{};
              t.segs = segsDev;
              t.nseg = nseg;
              for (int p = 0; p < nseg; ++p) t.src[p] = ownDev[p];
              t.itemsize = static_cast<int>(isz);
              t.L = nblk + 1;
              t.me = nblk;
              t.scale = scale;
              t.accumulate = accumulate;
              t.stage = blocksDev;
              t.srcStride = stride;
              t.chunkBegin = x0;
              t.chunkEnd = x1;
              t.destRank = dest;
              t.nDest = 1;
              t.out = outDev;
              t.finalMode = 1;
              t.skipOwn = !first;
              CUDACHECK(cudaMemcpy(outDev, base.data(), outElems * 4, cudaMemcpyHostToDevice));
              launchRsSum(t, ctas, nullptr);
              CUDACHECK(cudaDeviceSynchronize());
              CUDACHECK(cudaMemcpy(got.data(), outDev, outElems * 4, cudaMemcpyDeviceToHost));
              ++cases;
              ref = base;
              for (int p = 0; p < nseg; ++p) {
                const RsSegDev& sg = segs[p];
                const long long start = static_cast<long long>(dest) * sg.pieceElems;
                for (size_t i0 = 0; i0 < sg.pieceElems; ++i0) {
                  const size_t x = sg.stgOff + i0;
                  if (x < x0 || x >= x1) continue;
                  const bool real = start + static_cast<long long>(i0) < static_cast<long long>(sg.numel);
                  float sum = 0.f;
                  for (int b = 0; b < nblk; ++b) sum += toHost(blocks[b * stride / isz + (x - x0)]);
                  if (first && real) sum += toHost(own[p][start + i0]);
                  ref[sg.outOff + i0] = std::fma(scale, real ? sum : 0.f, accumulate ? base[sg.outOff + i0] : 0.f);
                }
              }
              if (std::memcmp(ref.data(), got.data(), outElems * 4) != 0 && ++failures <= 20) {
                size_t i = 0;
                while (i < outElems && std::memcmp(&ref[i], &got[i], 4) == 0) ++i;
                std::fprintf(stderr, "FAIL itemsize=%zu nblk=%d round=%d scale=%g dest=%d range=[%zu,%zu) ctas=%d: "
                                     "first difference at out[%zu] host %.9g kernel %.9g\n",
                             isz, nblk, round, scale, dest, x0, x1, ctas, i, ref[i], got[i]);
              }
            }
  for (void* p : ownDev) CUDACHECK(cudaFree(p));
  CUDACHECK(cudaFree(blocksDev));
  CUDACHECK(cudaFree(outDev));
  CUDACHECK(cudaFree(segsDev));
}

}  // namespace

int main() {
  rsRequireTma();
  std::mt19937 rng(1234);
  runDtype<__nv_bfloat16>(rng);
  runDtype<float>(rng);
  std::printf("%s: %d cases, %d failures\n", failures ? "FAIL" : "PASS", cases, failures);
  return failures ? 1 : 0;
}
