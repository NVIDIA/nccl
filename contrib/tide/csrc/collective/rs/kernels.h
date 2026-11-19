// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace tide {

constexpr int kRsMaxSegments = 256;
constexpr int kRsMaxLocal = 8;  // local ranks (sources) a staging slot and the kernels are sized for
constexpr int kRsVec = 8;       // elements per thread step (16 B of bf16, 32 B of fp32)

struct RsSegDev {
  unsigned long long numel, pieceElems, outOff, stgOff;
};

// Sources of one reduce step: my own gradient pieces read in place plus the pieces the other local
// ranks pushed into my staging slot. Sum in fp32 in a fixed source order (local rank 0..L-1).
// nDest > 1 (multicast mode): the same staging element range for nDest consecutive destination
// ranks destRank..destRank+nDest-1; source s's pieces for destination l sit at
// stage + s*srcStride + l*destStride, and the output is staging-indexed per destination
// (out + l*outDestStride + (x - subBegin), padding written as zero) instead of packed.
struct RsSumArgs {
  const RsSegDev* segs;
  int nseg;
  const void* src[kRsMaxSegments];  // gradient p base pointer (kernel parameter space; changes per call)
  int itemsize;                      // 2 (bf16) or 4 (fp32)
  const char* stage;       // my staging slot: source s at s*srcStride (+ l*destStride per destination)
  size_t chunkBytes;
  size_t srcStride, destStride;  // bytes
  int L, me;
  size_t chunkBegin, chunkEnd;  // staging element range of this chunk
  int destRank;                 // whose pieces are being reduced (own-piece addressing, padding clamp)
  int nDest;                    // 1: packed output; L: multicast layout (see above)
  size_t subBegin, outDestStride;  // multicast layout: sub-slice start (staging elems), block stride (elems)
  float* out;                   // packed fp32, indexed by segment outOff (nDest == 1)
  int finalMode;                // 0: out[i] = sum; 1: out[i] = (accumulate ? out[i] : 0) + scale * sum
  float scale;
  int accumulate;
  int skipOwn;                  // the own piece is not part of this sum (rail rounds after the first)
};

// Multicast landing: out (packed fp32) = (accumulate ? out : 0) + scale * sum over the L local ranks'
// inbox blocks read with multimem.ld_reduce (fp32 add in the switch). Block p holds part
// [partBegin, partBegin + partElems) of sub-slice p: staging elements from p*subElems + partBegin,
// up to the sub-slice's end and stgElems, of this rank's output, at `mc + p*blockStride` (multimem
// address of this rank's sub-block, same offset on every local rank); staging indices map to packed
// offsets through the segment table (gaps are skipped).
struct RsMcArgs {
  const RsSegDev* segs;
  int nseg;
  const float* mc;
  size_t blockStride;  // elements between blocks
  size_t subElems, stgElems;
  size_t partBegin, partElems;  // whole sub-slices: 0, subElems
  int blocks;          // blocks 0..blocks-1 are landed, or block `only` alone when it is >= 0
  int only;
  float* out;
  int finalMode;
  float scale;
  int accumulate;
};

// L: staged sources + own piece, up to kRsMaxLocal (a local team) or kRsMaxLocal + 1 (a rail round of
// kRsMaxLocal blocks: the senders' raw pieces at srcStride in the input dtype, the own piece last).
void launchRsSum(const RsSumArgs& a, int ctas, cudaStream_t s);
void launchRsMc(const RsMcArgs& a, int ctas, cudaStream_t s);
void rsRequireTma();  // throws unless this device runs the sum kernel (sm_90+, shared memory); once per process

}  // namespace tide
