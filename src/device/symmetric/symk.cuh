/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_SYMMETRIC_SYMK_H_
#define NCCL_DEVICE_SYMMETRIC_SYMK_H_

#include "sym_kernels.h"
#include "bitops.h"
#include "collectives.h"
#include "../work.cuh"
#include "../common_kernel.h"
#if !defined(NCCL_OS_WINDOWS)
#include "gin_scratch.h"
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13010
#include <cuda/std/ranges>
#endif

#if __CUDA_ARCH__ >= 1000
#include <cuda/barrier>
#include <cuda/ptx>

namespace ptx = cuda::ptx;

template <typename Pack, int UnrollPacks, int UnrollPeers = 1>
struct tmaSmemStruct {
  alignas(16) Pack buff[UnrollPeers][UnrollPacks * WARP_SIZE];
  cuda::barrier<cuda::thread_scope_block> bar;
};
#endif

template <bool val>
struct BoolTag {
  static constexpr bool value = val;
};

// A cheap approximation of std::decay
template <typename T>
struct ncclDecayType {
  using Type = T;
};
template <typename T>
struct ncclDecayType<T&> {
  using Type = T;
};
template <typename T>
struct ncclDecayType<T&&> {
  using Type = T;
};
template <typename T>
struct ncclDecayType<T const> {
  using Type = T;
};
template <typename T>
struct ncclDecayType<T volatile> {
  using Type = T;
};

template <typename T>
using ncclDecayType_t = typename ncclDecayType<T>::Type;

// flattenIx(pos0, dim0, pos1, dim1, pos2, dim2, ...)
// Given a position vector `pos` in a rectangular index space with lengths in the `dim`
// vector, flatten that down to a linear index. The fastest moving dimension is given first.
__device__ __forceinline__ int flattenIx() {
  return 0;
}

template <typename Int0, typename Int1, typename... Ints>
static __device__ Int0 flattenIx(Int0 pos, Int1 size, Ints... more) {
  return pos + size * flattenIx(more...);
}

template <typename T>
static __device__ void partitionElts(unsigned nParts, unsigned part, size_t* nElts, ncclSymPtr<T>* inPtr,
                                     ncclSymPtr<T>* outPtr) {
  constexpr int eltPerB16 = 16 / sizeof(T);
  size_t nB16 = (*nElts + eltPerB16 - 1) / eltPerB16;
  size_t beginB16 = part * (nB16 / nParts) + min(part, uint32_t(nB16 % nParts));
  *inPtr += beginB16 * eltPerB16;
  *outPtr += beginB16 * eltPerB16;
  if (part < nParts - 1) {
    nB16 = nB16 / nParts + (part < nB16 % nParts ? 1 : 0);
    *nElts = nB16 * eltPerB16;
  } else {
    *nElts = *nElts - beginB16 * eltPerB16;
  }
}

namespace {
struct ncclSymkArgsHandler : ncclSymkWorkArgsHandler {
  ncclLLA2AHandle const& lsaLLA2A;
  ncclGinOutboxHandle const& ginOutbox;
  ncclGinInboxA2AHandle const& ginInboxRail;
  ncclGinSyncHandle const& ginSyncHandle;
  ncclDevResourceHandle rsGinAccumBuf;
  uint32_t rsGinAccumBytesPerBlock;

  __device__ ncclSymkArgsHandler(ncclSymkDevWorkArgs const* args)
    : ncclSymkWorkArgsHandler(args->kcomm.devComm, ncclSymkGetWorkRange(args),
                              ncclSymkGetWorks(args, args->nMaxChannels)),
      lsaLLA2A(args->kcomm.lsaLLA2A), ginOutbox(args->kcomm.ginOutbox), ginInboxRail(args->kcomm.ginInboxRail),
      ginSyncHandle(args->kcomm.ginSyncHandle), rsGinAccumBuf(args->kcomm.rsGinAccumBuf),
      rsGinAccumBytesPerBlock(args->kcomm.rsGinAccumBytesPerBlock) {}
};
} // namespace

template <template <typename> typename Red, typename T, bool nvls>
struct ncclSymkAccumType {
  using Type = T;
};

// Only Red's whose opArg is invariant w.r.t. the datatype can have a different
// accumulator type. At the moment this excludes integer min/max, sumpostdiv,
// and premulsum.
template <>
struct ncclSymkAccumType<FuncSum, __half, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, __half, false> {
  using Type = float;
};
#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct ncclSymkAccumType<FuncSum, __nv_bfloat16, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, __nv_bfloat16, false> {
  using Type = float;
};
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template <>
struct ncclSymkAccumType<FuncSum, __nv_fp8_e4m3, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSum, __nv_fp8_e5m2, false> {
  using Type = float;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, __nv_fp8_e4m3, false> {
  using Type = __half;
};
template <>
struct ncclSymkAccumType<FuncSumPostDiv, __nv_fp8_e5m2, false> {
  using Type = __half;
};
#endif

// Accumulator type held in smem for GIN algos.
template <template <typename> typename Red, typename T>
struct ncclSymkGinAccumType {
  using Type = T;
};

template <>
struct ncclSymkGinAccumType<FuncSum, __half> {
  using Type = float;
};
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, __half> {
  using Type = float;
};
#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct ncclSymkGinAccumType<FuncSum, __nv_bfloat16> {
  using Type = float;
};
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, __nv_bfloat16> {
  using Type = float;
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
// fp8 types accumulate in fp16. Multimem algo sends fp8 on wire because it's
// impossible to get fp16 accumulator from switch. Non-multimem sends fp16 to
// give users a higher precision alternative.
template <>
struct ncclSymkGinAccumType<FuncSum, __nv_fp8_e4m3> {
  using Type = __half;
};
template <>
struct ncclSymkGinAccumType<FuncSum, __nv_fp8_e5m2> {
  using Type = __half;
};
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, __nv_fp8_e4m3> {
  using Type = __half;
};
template <>
struct ncclSymkGinAccumType<FuncSumPostDiv, __nv_fp8_e5m2> {
  using Type = __half;
};
#endif

#if __CUDA_ARCH__ >= 1000
static __device__ __forceinline__ void tmaLoadStoreMc(char* dest, char* smem, char* source, size_t size,
                                                      cuda::barrier<cuda::thread_scope_block>& bar) {
  cuda::device::memcpy_async_tx((char*)smem, (const char*)source, cuda::aligned_size_t<16>(size), bar);
  cuda::barrier<cuda::thread_scope_block>::arrival_token token = cuda::device::barrier_arrive_tx(bar, 1, size);
  bar.wait(std::move(token));
  ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, dest, smem, size);
  ptx::cp_async_bulk_commit_group();
  ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
}
#endif

// TODO: move this into data_ops.cuh
template <typename T, bool EnableTma = false>
static __device__ void bcastMultimem(ncclSymkArgsHandler& handler, int tn, int t, ncclSymPtr<T> input,
                                     ncclSymPtr<T> output, size_t nElts, uint32_t rank = 0) {
  size_t nBytes = nElts * sizeof(T);
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input.localPtr());
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output.multimemPtr(handler.comm.lsaMultimem));
  uint32_t alignment = uint32_t(inputUptr - outputUptr);
  uint32_t nPreBytes =
#if __CUDA_ARCH__ >= 1000
    (EnableTma && alignment % 256 == 0) ? (256 - input.offset) % 256 :
#endif
                                          (16 - input.offset) % 16;

  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

#if __CUDA_ARCH__ >= 1000
  int lane = t % WARP_SIZE;
  int lw = threadIdx.x / WARP_SIZE;
  extern __shared__ char smemScratch[];
#endif

  if (alignment % 16 == 0) {
    constexpr int BytePerPack = ncclSymkBytePerPack, UnrollPacks = ncclSymkDeepUnrollPacks;
    constexpr int BytePerChunk = ncclSymkMultimemDeepBytePerChunk;
    uint32_t nChunks = (nBytes - nPreBytes) / BytePerChunk;
    uintptr_t cursorAfter = nPreBytes + uintptr_t(nChunks) * BytePerChunk;

#if __CUDA_ARCH__ >= 1000
    // Initialize share memory pointer and barrier
    constexpr size_t tileSize = UnrollPacks * WARP_SIZE * BytePerPack;
    using tmaSmemStruct_t = tmaSmemStruct<BytePack<BytePerPack>, UnrollPacks>;
    constexpr int smemSizePerWarp = ncclTmaShmemScratchWarpSize();
    tmaSmemStruct_t* tmaSmem = reinterpret_cast<tmaSmemStruct_t*>(smemScratch + lw * smemSizePerWarp);
    if NCCL_IF_CONSTEXPR (EnableTma) {
      if (lane == 0) init(&tmaSmem->bar, 1);
    }
#endif

    nSufBytes = nBytes - cursorAfter;
    size_t nMainBytes = nBytes - nPreBytes - nSufBytes;

    // Ranks use offset multipliers 0 1 3 2 4 5 7 6 etc. This ensures proper address distribution even if
    // each rank has only 64MiB to process.
    uint32_t startOffsetMultiplier = rank ^ ((rank >> 1) & 1);
    size_t startBytes = nMainBytes > 0 ? (size_t)startOffsetMultiplier * ncclSymkMcPerRankOffsetBytes % nMainBytes : 0;
    uintptr_t wrapAt = cursorAfter + (t % WARP_SIZE) * BytePerPack;
    uintptr_t cursor = nPreBytes + uintptr_t(startBytes / BytePerChunk) * BytePerChunk;
    cursor += (t / WARP_SIZE) * UnrollPacks * WARP_SIZE * BytePerPack;
    cursor += (t % WARP_SIZE) * BytePerPack;
    int nIters = (int)nChunks - (int)(t / WARP_SIZE);
    NVCC_PRAGMA_UNROLL_DISABLED
    while (0 < nIters) {
      if (cursor >= wrapAt) cursor -= nMainBytes;
#if __CUDA_ARCH__ >= 1000
      if NCCL_IF_CONSTEXPR (EnableTma) {
        if (lane == 0)
          tmaLoadStoreMc((char*)(outputUptr + cursor), (char*)tmaSmem->buff[0], (char*)(inputUptr + cursor), tileSize,
                         tmaSmem->bar);
      } else
#endif
      {
        BytePack<BytePerPack> tmp[UnrollPacks];
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPacks; u++) {
          tmp[u] = *reinterpret_cast<BytePack<BytePerPack>*>(inputUptr + cursor + u * WARP_SIZE * BytePerPack);
        }
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPacks; u++) {
          multimem_st_global(outputUptr + cursor + u * WARP_SIZE * BytePerPack, tmp[u]);
        }
      }
      cursor += tn * UnrollPacks * BytePerPack;
      nIters -= tn / WARP_SIZE;
    }
  } else {
    nPreBytes = 0;
    nSufBytes = nBytes;
  }

  // Get the prefix+suffix element one at a time.
  NVCC_PRAGMA_UNROLL(4)
  for (uintptr_t i = t * sizeof(T); i < nPreBytes + nSufBytes; i += tn * sizeof(T)) {
    uintptr_t cursor = i < nPreBytes ? i : nBytes - nSufBytes + (i - nPreBytes);
    BytePack<sizeof(T)> val = *reinterpret_cast<BytePack<sizeof(T)>*>(inputUptr + cursor);
    multimem_st_global(outputUptr + cursor, val);
  }
}

extern __shared__ ulong2 ncclSymkSmem[];

static __device__ void ncclSymkSmemPartition_help(int bumper) {}
template <typename T, typename... More>
static __device__ void ncclSymkSmemPartition_help(int bumper, T** ptr, int size, More... more) {
  T* ans = reinterpret_cast<T*>(ncclSymkSmem + bumper);
  __builtin_assume(__isShared(ans)); // Let compiler know this is shared memory (reinterpret_cast obscured as much).
  __builtin_assume_aligned(ans, sizeof(ulong2));
  *ptr = ans;
  bumper += (size * sizeof(T) + sizeof(ulong2) - 1) / sizeof(ulong2);
  ncclSymkSmemPartition_help(bumper, more...);
}

template <typename... Arg>
static __device__ void ncclSymkSmemPartition(Arg... args) {
  ncclSymkSmemPartition_help(/*bumper=*/0, args...);
}

////////////////////////////////////////////////////////////////////////////////
// Extensions to nccl_device.h needed to help compiler make good SASS:
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct ncclLsaPointerGetter {
  void* base;
  uint32_t stride4G;
  __device__ ncclLsaPointerGetter(ncclSymPtr<T> ptr) {
    base = (char*)nccl::utility::loadConst(&ptr.window->lsaFlatBase);
    base = (char*)base + ptr.offset;
    stride4G = nccl::utility::loadConst(&ptr.window->stride4G);
  }
  __device__ T* operator()(int lsaPeer) const {
    return (T*)nccl::utility::add4G(base, lsaPeer * stride4G);
  }
};
#endif // NCCL_DEVICE_SYMMETRIC_SYMK_H_
