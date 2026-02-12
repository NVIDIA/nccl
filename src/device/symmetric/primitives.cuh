#ifndef NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_
#define NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_

#include "sym_kernels.h"
#include "bitops.h"
#include "collectives.h"
#include "../op128.h"
#include "../reduce_kernel.h"
#include "gin_scratch.h"

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

template<bool val> struct BoolTag { static constexpr bool value = val; };

// A cheap approximation of std::decay
template<typename T> struct ncclDecayType { using Type = T; };
template<typename T> struct ncclDecayType<T&> { using Type = T; };
template<typename T> struct ncclDecayType<T&&> { using Type = T; };
template<typename T> struct ncclDecayType<T const> { using Type = T; };
template<typename T> struct ncclDecayType<T volatile> { using Type = T; };

template<typename T>
using ncclDecayType_t = typename ncclDecayType<T>::Type;


// flattenIx(pos0, dim0, pos1, dim1, pos2, dim2, ...)
// Given a position vector `pos` in a rectangular index space with lengths in the `dim`
// vector, flatten that down to a linear index. The fastest moving dimension is given first.
__device__ __forceinline__ int flattenIx() { return 0; }

template<typename Int0, typename Int1, typename ...Ints>
static __device__ Int0 flattenIx(Int0 pos, Int1 size, Ints ...more) {
  return pos + size*flattenIx(more...);
}

template<typename T>
static __device__ void partitionElts(
    unsigned nParts, unsigned part,
    size_t* nElts, ncclSymPtr<T>* inPtr, ncclSymPtr<T>* outPtr
  ) {
  constexpr int eltPerB16 = 16/sizeof(T);
  size_t nB16 = (*nElts + eltPerB16-1)/eltPerB16;
  size_t beginB16 = part*(nB16/nParts) + min(part, uint32_t(nB16%nParts));
  *inPtr += beginB16*eltPerB16;
  *outPtr += beginB16*eltPerB16;
  if (part < nParts-1) {
    nB16 = nB16/nParts + (part < nB16%nParts ? 1 : 0);
    *nElts = nB16*eltPerB16;
  } else {
    *nElts = *nElts - beginB16*eltPerB16;
  }
}

namespace {
struct ncclSymkArgsHandler {
  ncclDevComm const& comm;
  ncclLLA2AHandle const& lsaLLA2A;
  ncclGinOutboxHandle const& ginOutbox;
  ncclGinInboxA2AHandle const& ginInboxRail;
  ncclGinCounter_t ginCounterPerBlock;
  ncclGinSyncHandle const& ginSyncHandle;
  struct ncclSymkChannelWorkRange* channelWorkRange;
  struct ncclSymkDevWork* devWork;
  uint32_t nRanks_rcp32;

  __device__ ncclSymkArgsHandler(ncclSymkDevWorkArgs const* args):
    comm(args->kcomm.devComm),
    lsaLLA2A(args->kcomm.lsaLLA2A),
    ginOutbox(args->kcomm.ginOutbox),
    ginInboxRail(args->kcomm.ginInboxRail),
    ginCounterPerBlock(args->kcomm.ginCounterPerBlock),
    ginSyncHandle(args->kcomm.ginSyncHandle) {
    channelWorkRange = args->getWorkRange();

    devWork = args->getWorks(args->nMaxChannels);
    nRanks_rcp32 = comm.nRanks_rcp32;
  }

  template<typename T>
  __device__ void getWorkRange(int block, uint16_t& workLo, size_t& indexLo, uint16_t& workHi, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    uint32_t fracLo, fracHi;

    // Where the work begins
    workLo = (block==0) ? 0 : channelWorkRange[block-1].workHi; // start where predecessor ends
    fracLo = (block==0) ? 0 : channelWorkRange[block-1].fracHi + 1;
    // If the predecessor ended on the work boundary, then we step to the beginning of the next work.
    // This ensures we never have empty parts.
    if (fracLo == 0x10000) {
      workLo++;
      fracLo = 0;
    }
    struct ncclSymkDevWork const& dwLo = devWork[workLo];
    indexLo = ((fracLo * divUp(dwLo.nElts, EltPerCell)) >> 16) * EltPerCell;

    // Where the work ends
    workHi = channelWorkRange[block].workHi;
    fracHi = channelWorkRange[block].fracHi + 1;
    struct ncclSymkDevWork const& dwHi = devWork[workHi];
    indexHi = min(((fracHi * divUp(dwHi.nElts, EltPerCell)) >> 16) * EltPerCell, dwHi.nElts);
  }

  template<typename T>
  __device__ void getWorkRangeFused(int blockIdx, int w, int& block, int& nBlocks, size_t& indexLo, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    struct ncclSymkDevWork const& dw = devWork[w];
    uint32_t fracLo, fracHi;
    int lastBlock;

    block = blockIdx - dw.sChannelId;
    nBlocks = dw.nChannels;
    lastBlock = dw.sChannelId+dw.nChannels-1;

    // Where the work begins
    fracLo = (dw.sChannelId>0 && channelWorkRange[dw.sChannelId-1].workHi == w) ? ((channelWorkRange[dw.sChannelId-1].fracHi + 1) & 0xFFFF) : 0;
    indexLo = ((fracLo * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell;
    fracHi = (channelWorkRange[lastBlock].workHi == w) ? channelWorkRange[lastBlock].fracHi + 1 : 0x10000;
    indexHi = min(((fracHi * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell, dw.nElts);
  }

  template<typename T, typename Fn>
  __device__ void forEachWork(Fn const& fn) {
    uint16_t workLo, workHi;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

    #pragma unroll 1
    for (int w = workLo; w <= workHi; w++) {
      struct ncclSymkDevWork const& dw = devWork[w];
      size_t const& nAllElts = dw.nElts;
      size_t currentIndexLo, currentIndexHi;
      int block, nBlocks;
      if (blockIdx.x >= dw.sChannelId && blockIdx.x < dw.sChannelId + dw.nChannels) {
        getWorkRangeFused<T>(blockIdx.x, w, block, nBlocks, currentIndexLo, currentIndexHi);
      } else {
        currentIndexLo = (w > workLo) ? 0 : indexLo;
        currentIndexHi = (w < workHi) ? nAllElts : indexHi;
        block = 0;
        nBlocks = 1;
      }

      fn(block, nBlocks, currentIndexHi - currentIndexLo, nAllElts,
         ncclSymPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
         ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo);

      currentIndexLo = 0;
    }
  }

  template<typename T, typename Fn>
  __device__ void singleWork(Fn const& fn) {
    uint16_t w;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, w, indexLo, w, indexHi);

    struct ncclSymkDevWork const& dw = devWork[w];

    fn(indexHi - indexLo, dw.nElts,
       ncclSymPtr<T>(dw.inputWin, dw.inputOff) + indexLo,
       ncclSymPtr<T>(dw.outputWin, dw.outputOff) + indexLo);
  }

  template<typename T, typename Fn>
    __device__ void forEachWorkNoFusion(Fn const& fn) {
      uint16_t workLo, workHi;
      size_t indexLo, indexHi;

      getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

      #pragma unroll 1
      for (int w = workLo; w <= workHi; w++) {
        struct ncclSymkDevWork const& dw = devWork[w];
        size_t const& nAllElts = dw.nElts;
        size_t currentIndexLo, currentIndexHi;
        currentIndexLo = (w > workLo) ? 0 : indexLo;
        currentIndexHi = (w < workHi) ? nAllElts : indexHi;

        fn(currentIndexHi - currentIndexLo, nAllElts,
           ncclSymPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
           ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo);
      }
  }
};
}

template<template<typename> typename Red, typename T, bool nvls>
struct ncclSymkAccumType { using Type = T; };

// Only Red's whose opArg is invariant w.r.t. the datatype can have a different
// accumulator type. At the moment this excludes integer min/max, sumpostdiv,
// and premulsum.
template<> struct ncclSymkAccumType<FuncSum, __half, false> { using Type = float; };
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<> struct ncclSymkAccumType<FuncSum, __nv_bfloat16, false> { using Type = float; };
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<> struct ncclSymkAccumType<FuncSum, __nv_fp8_e4m3, false> { using Type = float; };
template<> struct ncclSymkAccumType<FuncSum, __nv_fp8_e5m2, false> { using Type = float; };
#endif

// Accumulator type held in smem for GIN algos.
template<template<typename> typename Red, typename T>
struct ncclSymkGinAccumType { using Type = T; };

template<> struct ncclSymkGinAccumType<FuncSum, __half> { using Type = float; };
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<> struct ncclSymkGinAccumType<FuncSum, __nv_bfloat16> { using Type = float; };
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
// fp8 types accumulate in fp16. Multimem algo sends fp8 on wire because it's
// impossible to get fp16 accumulator from switch. Non-multimem sends fp16 to
// give users a higher precision alternative.
template<> struct ncclSymkGinAccumType<FuncSum, __nv_fp8_e4m3> { using Type = __half; };
template<> struct ncclSymkGinAccumType<FuncSum, __nv_fp8_e5m2> { using Type = __half; };
#endif

// TODO: move this into data_ops.cuh
template<typename T>
static __device__ void bcastMultimem(
    ncclSymkArgsHandler& handler, int tn, int t, ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {
  size_t nBytes = nElts*sizeof(T);
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input.localPtr());
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output.multimemPtr(handler.comm.lsaMultimem));
  uint32_t nPreBytes = (16 - input.offset)%16;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

  if ((inputUptr-outputUptr)%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 8;
    constexpr int BytePerChunk = UnrollPacks*WARP_SIZE*BytePerPack;
    uintptr_t cursor = nPreBytes;
    uint32_t nChunks = (nBytes-cursor)/BytePerChunk;
    uintptr_t cursorAfter = cursor + uintptr_t(nChunks)*BytePerChunk;
    nSufBytes = nBytes - cursorAfter;
    cursor += (t/WARP_SIZE)*UnrollPacks*WARP_SIZE*BytePerPack;
    cursor += (t%WARP_SIZE)*BytePerPack;
    int nIters = nChunks - t/WARP_SIZE;
    #pragma unroll 1
    while (0 < nIters) {
      BytePack<BytePerPack> tmp[UnrollPacks];
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        tmp[u] = *reinterpret_cast<BytePack<BytePerPack>*>(inputUptr + cursor + u*WARP_SIZE*BytePerPack);
      }
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        multimem_st_global(outputUptr + cursor + u*WARP_SIZE*BytePerPack, tmp[u]);
      }
      cursor += tn*UnrollPacks*BytePerPack;
      nIters -= tn/WARP_SIZE;
    }
  } else {
    nPreBytes = 0;
    nSufBytes = nBytes;
  }

  // Get the prefix+suffix element one at a time.
  #pragma unroll 4
  for (uintptr_t i = t*sizeof(T); i < nPreBytes + nSufBytes; i += tn*sizeof(T)) {
    uintptr_t cursor = i < nPreBytes ? i : nBytes-nSufBytes+(i-nPreBytes);
    BytePack<sizeof(T)> val = *reinterpret_cast<BytePack<sizeof(T)>*>(inputUptr + cursor);
    multimem_st_global(outputUptr + cursor, val);

  }
}

extern __shared__ ulong2 ncclSymkSmem[];

static __device__ void ncclSymkSmemPartition_help(int bumper) {}
template<typename T, typename ...More>
static __device__ void ncclSymkSmemPartition_help(int bumper, T** ptr, int size, More ...more) {
  T *ans = reinterpret_cast<T*>(ncclSymkSmem + bumper);
  __builtin_assume(__isShared(ans)); // Let compiler know this is shared memory (reinterpret_cast obscured as much).
  __builtin_assume_aligned(ans, sizeof(ulong2));
  *ptr = ans;
  bumper += (size*sizeof(T) + sizeof(ulong2)-1)/sizeof(ulong2);
  ncclSymkSmemPartition_help(bumper, more...);
}

template<typename ...Arg>
static __device__ void ncclSymkSmemPartition(Arg ...args) {
  ncclSymkSmemPartition_help(/*bumper=*/0, args...);
}

////////////////////////////////////////////////////////////////////////////////
// Extensions to nccl_device.h needed to help compiler make good SASS:
////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct ncclLsaPointerGetter {
  void* base;
  uint32_t stride4G;
  __device__ ncclLsaPointerGetter(ncclSymPtr<T> ptr) {
    base = (char*)nccl::utility::loadConst(&ptr.window->lsaFlatBase);
    base = (char*)base + ptr.offset;
    stride4G = nccl::utility::loadConst(&ptr.window->stride4G);
  }
  __device__ T* operator()(int lsaPeer) const {
    return (T*)nccl::utility::add4G(base, lsaPeer*stride4G);
  }
};
#endif // NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_
