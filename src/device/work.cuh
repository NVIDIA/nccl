/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_WORK_H_
#define NCCL_DEVICE_WORK_H_

#include "sym_kernels.h"

namespace {
struct ncclSymkWorkArgsHandler {
  ncclDevComm const& comm;
  struct ncclSymkChannelWorkRange* channelWorkRange;
  struct ncclSymkDevWork* devWork;
  uint32_t nRanks_rcp32;

  __device__ ncclSymkWorkArgsHandler(ncclDevComm const& comm_, struct ncclSymkChannelWorkRange* workRange,
                                     struct ncclSymkDevWork* works)
    : comm(comm_), channelWorkRange(workRange), devWork(works) {
    nRanks_rcp32 = comm.nRanks_rcp32;
  }

  template <typename T>
  __device__ void getWorkRange(int block, uint16_t& workLo, size_t& indexLo, uint16_t& workHi, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    uint32_t fracLo, fracHi;

    // Where the work begins
    workLo = (block == 0) ? 0 : channelWorkRange[block - 1].workHi; // start where predecessor ends
    fracLo = (block == 0) ? 0 : channelWorkRange[block - 1].fracHi + 1;
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

  template <typename T>
  __device__ void getWorkRangeFused(int blockIdx, int w, int& block, int& nBlocks, size_t& indexLo, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    struct ncclSymkDevWork const& dw = devWork[w];
    uint32_t fracLo, fracHi;
    int lastBlock;

    block = blockIdx - dw.sChannelId;
    nBlocks = dw.nChannels;
    lastBlock = dw.sChannelId + dw.nChannels - 1;

    // Where the work begins
    fracLo = (dw.sChannelId > 0 && channelWorkRange[dw.sChannelId - 1].workHi == w) ?
               ((channelWorkRange[dw.sChannelId - 1].fracHi + 1) & 0xFFFF) :
               0;
    indexLo = ((fracLo * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell;
    fracHi = (channelWorkRange[lastBlock].workHi == w) ? channelWorkRange[lastBlock].fracHi + 1 : 0x10000;
    indexHi = min(((fracHi * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell, dw.nElts);
  }

  template <typename T, typename Fn>
  __device__ void forEachWork(Fn const& fn) {
    uint16_t workLo, workHi;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

    NVCC_PRAGMA_UNROLL_DISABLED
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
         ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo, dw.redOpArg);

      currentIndexLo = 0;
    }
  }

  template <typename T, typename Fn>
  __device__ void forEachWorkWithRoot(Fn const& fn) {
    uint16_t workLo, workHi;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

    NVCC_PRAGMA_UNROLL_DISABLED
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

      fn(dw.rootRank, block, nBlocks, currentIndexHi - currentIndexLo, nAllElts,
         ncclSymPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
         ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo, dw.redOpArg);

      currentIndexLo = 0;
    }
  }

  template <typename T, typename Fn>
  __device__ void singleWork(Fn const& fn) {
    uint16_t w;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, w, indexLo, w, indexHi);

    struct ncclSymkDevWork const& dw = devWork[w];

    fn(indexHi - indexLo, dw.nElts, ncclSymPtr<T>(dw.inputWin, dw.inputOff) + indexLo,
       ncclSymPtr<T>(dw.outputWin, dw.outputOff) + indexLo);
  }

  template <typename T, typename Fn>
  __device__ void forEachWorkNoFusion(Fn const& fn) {
    uint16_t workLo, workHi;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

    NVCC_PRAGMA_UNROLL_DISABLED
    for (int w = workLo; w <= workHi; w++) {
      struct ncclSymkDevWork const& dw = devWork[w];
      size_t const& nAllElts = dw.nElts;
      size_t currentIndexLo, currentIndexHi;
      currentIndexLo = (w > workLo) ? 0 : indexLo;
      currentIndexHi = (w < workHi) ? nAllElts : indexHi;

      fn(currentIndexHi - currentIndexLo, nAllElts, ncclSymPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
         ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo);
    }
  }
};
} // namespace

#endif // NCCL_DEVICE_WORK_H_
