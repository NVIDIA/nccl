#ifndef NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_
#define NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_

#include "sym_kernels.h"
#include "bitops.h"
#include "collectives.h"
#include "../op128.h"
#include "../reduce_kernel.h"

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

// flattenIx(pos0, dim0, pos1, dim1, pos2, dim2, ...)
// Given a position vector `pos` in a rectangular index space with lengths in the `dim`
// vector, flatten that down to a linear index. The fastest moving dimension is given first.
__device__ __forceinline__ int flattenIx() { return 0; }

template<typename Int0, typename Int1, typename ...Ints>
static __device__ Int0 flattenIx(Int0 pos, Int1 size, Ints ...more) {
  return pos + size*flattenIx(more...);
}

namespace {
struct ncclSymkKernelStuff {
  ncclDevComm const& comm;
  struct ncclSymkChannelWorkRange* channelWorkRange;
  struct ncclSymkDevWork* devWork;
  uint32_t nRanks_rcp32;

  __device__ ncclSymkKernelStuff(ncclSymkDevWorkArgs const* args):
    comm(args->comm) {
    channelWorkRange = args->getWorkRange();

    devWork = args->getWorks(args->nMaxChannels);
    nRanks_rcp32 = comm.nRanks_rcp32;
  }

  template<typename T>
    __device__ void getWorkRange(int block,
                                 uint16_t& workLo, size_t& indexLo, uint16_t& workHi, size_t& indexHi) {
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
    struct ncclSymkDevWork const& dw = devWork[workLo];
    indexLo = ((fracLo * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell;

    // Where the work ends
    workHi = channelWorkRange[block].workHi;
    fracHi = channelWorkRange[block].fracHi + 1;
    indexHi = min(((fracHi * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell, dw.nElts);
  }

  template<typename T>
    __device__ void getWorkRangeFused(int blockIdx, int w,
                                      int& block, int& nBlocks, size_t& indexLo, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    struct ncclSymkDevWork const& dw = devWork[w];
    uint32_t fracLo, fracHi;
    int lastBlock;

    block = blockIdx - dw.sChannelId;
    nBlocks = dw.nChannels;
    lastBlock = dw.sChannelId+dw.nChannels-1;

    // Where the work begins
    fracLo = (dw.sChannelId==0) ? 0 : ((channelWorkRange[dw.sChannelId-1].fracHi + 1) & 0xFFFF);
    indexLo = ((fracLo * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell;
    fracHi = (channelWorkRange[lastBlock].workHi == w) ? channelWorkRange[lastBlock].fracHi + 1 : 0x10000;
    indexHi = min(((fracHi * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell, dw.nElts);
  }
};
}

#ifndef NCCL_SYMK_GROUP_SINGLE_WORK

#define NCCL_SYMK_GROUP_START(stuff, type)           \
  uint16_t ncclSymkGroupWorkLo, ncclSymkGroupWorkHi; \
  size_t ncclSymkGroupIndexLo, ncclSymkGroupIndexHi; \
  stuff.getWorkRange<type>(blockIdx.x, ncclSymkGroupWorkLo, ncclSymkGroupIndexLo, \
                           ncclSymkGroupWorkHi, ncclSymkGroupIndexHi); \
  size_t ncclSymkGroupCurrentIndexLo = ncclSymkGroupIndexLo; \
  _Pragma("unroll 1") \
  for (int ncclSymkGroupW = ncclSymkGroupWorkLo; ncclSymkGroupW <= ncclSymkGroupWorkHi; ncclSymkGroupW++) { \
    struct ncclSymkDevWork const& ncclSymkGroupDevWork = stuff.devWork[ncclSymkGroupW]; \
    size_t const& ncclSymkGroupNAllElts = ncclSymkGroupDevWork.nElts; \
    size_t ncclSymkGroupCurrentIndexHi; \
    int ncclSymkGroupBlock, ncclSymkGroupNBlocks; \
    if (blockIdx.x >= ncclSymkGroupDevWork.sChannelId && \
        blockIdx.x < ncclSymkGroupDevWork.sChannelId + ncclSymkGroupDevWork.nChannels) \
      stuff.getWorkRangeFused<type>(blockIdx.x, ncclSymkGroupW, ncclSymkGroupBlock, ncclSymkGroupNBlocks, \
                                    ncclSymkGroupCurrentIndexLo, ncclSymkGroupCurrentIndexHi); \
    else { \
      ncclSymkGroupCurrentIndexHi = (ncclSymkGroupW < ncclSymkGroupWorkHi) ? \
        ncclSymkGroupNAllElts : ncclSymkGroupIndexHi; \
      ncclSymkGroupBlock = 0; \
      ncclSymkGroupNBlocks = 1; \
    } \
    ncclSymPtr<type> ncclSymkGroupInput(ncclSymkGroupDevWork.inputWin, \
                                        ncclSymkGroupDevWork.inputOff + ncclSymkGroupCurrentIndexLo*sizeof(type)); \
    ncclSymPtr<type> ncclSymkGroupOutput(ncclSymkGroupDevWork.outputWin, \
                                         ncclSymkGroupDevWork.outputOff + ncclSymkGroupCurrentIndexLo*sizeof(type)); \
    const size_t ncclSymkGroupNElts = ncclSymkGroupCurrentIndexHi - ncclSymkGroupCurrentIndexLo

#define NCCL_SYMK_GROUP_NOFUSE_START(stuff, type) \
  uint16_t ncclSymkGroupWorkLo, ncclSymkGroupWorkHi; \
  size_t ncclSymkGroupIndexLo, ncclSymkGroupIndexHi; \
  stuff.getWorkRange<type>(blockIdx.x, ncclSymkGroupWorkLo, ncclSymkGroupIndexLo, \
                           ncclSymkGroupWorkHi, ncclSymkGroupIndexHi); \
  size_t ncclSymkGroupCurrentIndexLo = ncclSymkGroupIndexLo; \
  _Pragma("unroll 1") \
  for (int ncclSymkGroupW = ncclSymkGroupWorkLo; ncclSymkGroupW <= ncclSymkGroupWorkHi; ncclSymkGroupW++) { \
    struct ncclSymkDevWork& ncclSymkGroupDevWork = stuff.devWork[ncclSymkGroupW]; \
    size_t const& ncclSymkGroupNAllElts = ncclSymkGroupDevWork.nElts; \
    size_t ncclSymkGroupCurrentIndexHi = (ncclSymkGroupW < ncclSymkGroupWorkHi) ? \
      ncclSymkGroupNAllElts : ncclSymkGroupIndexHi; \
    ncclSymPtr<type> ncclSymkGroupInput(ncclSymkGroupDevWork.inputWin, \
                                        ncclSymkGroupDevWork.inputOff + ncclSymkGroupCurrentIndexLo*sizeof(type)); \
    ncclSymPtr<type> ncclSymkGroupOutput(ncclSymkGroupDevWork.outputWin, \
                                         ncclSymkGroupDevWork.outputOff + ncclSymkGroupCurrentIndexLo*sizeof(type)); \
    const size_t ncclSymkGroupNElts = ncclSymkGroupCurrentIndexHi - ncclSymkGroupCurrentIndexLo

#define NCCL_SYMK_GROUP_END \
    ncclSymkGroupCurrentIndexLo = 0; \
  } \
  do {} while(0)

#else // NCCL_SYMK_GROUP_SINGLE_WORK

#define NCCL_SYMK_GROUP_START(stuff, type) \
  struct ncclSymkDevWork const& ncclSymkGroupDevWork = stuff.devWork[0]; \
  size_t const& ncclSymkGroupNAllElts = ncclSymkGroupDevWork.nElts; \
  int const& ncclSymkGroupBlock = blockIdx.x; \
  int const& ncclSymkGroupNBlocks = gridDim.x; \
  ncclSymPtr<type> ncclSymkGroupInput(ncclSymkGroupDevWork.inputWin, ncclSymkGroupDevWork.inputOff); \
  ncclSymPtr<type> ncclSymkGroupOutput(ncclSymkGroupDevWork.outputWin, ncclSymkGroupDevWork.outputOff); \
  size_t const& ncclSymkGroupNElts = ncclSymkGroupDevWork.nElts

#define NCCL_SYMK_GROUP_NOFUSE_START(stuff, type) \
  uint16_t ncclSymkGroupWorkLo, ncclSymkGroupWorkHi; \
  size_t ncclSymkGroupIndexLo, ncclSymkGroupIndexHi; \
  stuff.getWorkRange<type>(blockIdx.x, ncclSymkGroupWorkLo, ncclSymkGroupIndexLo, \
                           ncclSymkGroupWorkHi, ncclSymkGroupIndexHi); \
  struct ncclSymkDevWork const& ncclSymkGroupDevWork = stuff.devWork[ncclSymkGroupWorkLo]; \
  size_t const& ncclSymkGroupNAllElts = ncclSymkGroupDevWork.nElts; \
  ncclSymPtr<type> ncclSymkGroupInput(ncclSymkGroupDevWork.inputWin, \
                                      ncclSymkGroupDevWork.inputOff + ncclSymkGroupIndexLo*sizeof(type)); \
  ncclSymPtr<type> ncclSymkGroupOutput(ncclSymkGroupDevWork.outputWin, \
                                       ncclSymkGroupDevWork.outputOff + ncclSymkGroupIndexLo*sizeof(type)); \
  const size_t ncclSymkGroupNElts = ncclSymkGroupIndexHi - ncclSymkGroupIndexLo

#define NCCL_SYMK_GROUP_END                     \
  do {} while(0)

#endif // NCCL_SYMK_GROUP_SINGLE_WORK

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
#endif
