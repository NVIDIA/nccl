/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "nccl_device.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include <stdio.h>

template<int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, bool EnableTma, typename Red>
static __device__ __forceinline__ void allreduceDeep(
    ncclSymkArgsHandler const& handler, int tn, int t,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    Red red, ncclSymPtr<char> input, ncclSymPtr<char> output, int32_t nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  using Acc = typename Red::EltType;
  using AccPack = BytePack<BytePerPack*sizeof(Acc)/sizeof(T)>;

  ncclTeam world = ncclTeamWorld(handler.comm);
  int wn = tn/WARP_SIZE;
  int w = t/WARP_SIZE;
  int lane = t%WARP_SIZE;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  ncclSymPtr<Pack> inpPacks = (ncclSymPtr<Pack>)input + intptr_t(w)*UnrollPacks*WARP_SIZE + (
#if __CUDA_ARCH__ >= 1000
      EnableTma ? 0 :
#endif
      lane);

  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w)*UnrollPacks*WARP_SIZE + (
#if __CUDA_ARCH__ >= 1000
      EnableTma ? 0 :
#endif
      lane);

  Pack acc0[UnrollPacks];

#if __CUDA_ARCH__ >= 1000
  int lw = threadIdx.x / WARP_SIZE;
  extern __shared__ char smemScratch[];
  using tmaSmemStruct_t = tmaSmemStruct<Pack, UnrollPacks, UnrollPeers>;
  constexpr int smemSizePerWarp = ncclTmaShmemScratchWarpSize();
  tmaSmemStruct_t* tmaSmem = reinterpret_cast<tmaSmemStruct_t*>(smemScratch+lw*smemSizePerWarp);
  constexpr size_t tilePack = UnrollPacks*WARP_SIZE;
  constexpr size_t tileSize = tilePack*BytePerPack;
  size_t tmaSize = 0;

  if NCCL_IF_CONSTEXPR (EnableTma) {
    if (lane == 0) {
      init(&tmaSmem->bar, WARP_SIZE);
    }
  }
#endif

  nIters -= w;
  if (0 < nIters) {
#if __CUDA_ARCH__ >= 1000
    if NCCL_IF_CONSTEXPR (EnableTma) {
      if (lane == 0) {
        cuda::device::memcpy_async_tx(tmaSmem->buff[0], inpPacks.peerPtr(world, rank), cuda::aligned_size_t<16>(tileSize), tmaSmem->bar);
        tmaSize += tileSize;
      }
    } else
#endif
    {
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u=0; u < UnrollPacks; u++) {
        acc0[u] = inpPacks.peerPtr(world, rank)[u*WARP_SIZE];
      }
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);

  if (0 < nIters) {
    while (true) {
      AccPack acc1[UnrollPacks];
      int r = rank;
      if (++r == nRanks) r = 0;
      { Pack tmp1[UnrollPacks];
#if __CUDA_ARCH__ >= 1000
        if NCCL_IF_CONSTEXPR (EnableTma) {
          if (lane == 0) {
            cuda::device::memcpy_async_tx(tmaSmem->buff[1], inpPacks.peerPtr(world, r), cuda::aligned_size_t<16>(tileSize), tmaSmem->bar);
            tmaSize += tileSize;
          }
          cuda::barrier<cuda::thread_scope_block>::arrival_token token = cuda::device::barrier_arrive_tx(tmaSmem->bar, 1, tmaSize);
          tmaSmem->bar.wait(std::move(token));
          tmaSize = 0;
        } else
#endif
        {
          NVCC_PRAGMA_UNROLL_AUTO
          for (int u=0; u < UnrollPacks; u++) {
            tmp1[u] = inpPacks.peerPtr(world, r)[u*WARP_SIZE];
          }
        }
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u=0; u < UnrollPacks; u++) {
#if __CUDA_ARCH__ >= 1000
          if NCCL_IF_CONSTEXPR (EnableTma) {
            acc0[u] = tmaSmem->buff[0][lane+WARP_SIZE*u];
            tmp1[u] = tmaSmem->buff[1][lane+WARP_SIZE*u];
          }
#endif
          acc1[u] = applyReduce(red, applyCast<T, Acc>(acc0[u]), applyCast<T, Acc>(tmp1[u]));
        }
      }

      if (++r == nRanks) r = 0;

      int dr = 2;
      NVCC_PRAGMA_UNROLL(2)
      for (int partial=0; partial <= 1; partial++) {
        NVCC_PRAGMA_UNROLL_DISABLED
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          if (partial && dr == nRanks) break;

          Pack tmp1[UnrollPeers][UnrollPacks];
#if __CUDA_ARCH__ >= 1000
          if NCCL_IF_CONSTEXPR (EnableTma) {
            // lane 0 waits for all threads to reduce tmp1 before next batch of TMA loads
            __syncwarp();
          }
#endif

          NVCC_PRAGMA_UNROLL_AUTO
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && ur!=0 && dr+ur == nRanks) break;
#if __CUDA_ARCH__ >= 1000
            if NCCL_IF_CONSTEXPR (EnableTma) {
              if (lane == 0) {
                cuda::device::memcpy_async_tx(tmaSmem->buff[ur], inpPacks.peerPtr(world, r), cuda::aligned_size_t<16>(tileSize), tmaSmem->bar);
                tmaSize += tileSize;
              }
            } else
#endif
            {
              NVCC_PRAGMA_UNROLL(UnrollPacks)
              for (int u=0; u < UnrollPacks; u++) {
                tmp1[ur][u] = inpPacks.peerPtr(world, r)[u*WARP_SIZE];
              }
            }
            if (++r == nRanks) r = 0;
          }
#if __CUDA_ARCH__ >= 1000
          if NCCL_IF_CONSTEXPR (EnableTma) {
            cuda::barrier<cuda::thread_scope_block>::arrival_token token = cuda::device::barrier_arrive_tx(tmaSmem->bar, 1, tmaSize);
            tmaSmem->bar.wait(std::move(token));
            tmaSize = 0;
          }
#endif
          NVCC_PRAGMA_UNROLL_AUTO
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && ur!=0 && dr+ur == nRanks) break;
            NVCC_PRAGMA_UNROLL(UnrollPacks)
            for (int u=0; u < UnrollPacks; u++) {
#if __CUDA_ARCH__ >= 1000
              if NCCL_IF_CONSTEXPR (EnableTma) {
                tmp1[ur][u] = tmaSmem->buff[ur][lane+WARP_SIZE*u];
              }
#endif
              acc1[u] = applyReduce(red, acc1[u], applyCast<T, Acc>(tmp1[ur][u]));
            }
          }
        }
      }

      NVCC_PRAGMA_UNROLL_AUTO
      for (int u=0; u < UnrollPacks; u++) {
#if __CUDA_ARCH__ >= 1000
        if NCCL_IF_CONSTEXPR (EnableTma) {
          tmaSmem->buff[0][lane+WARP_SIZE*u] = applyCast<Acc, T>(acc1[u]);
        } else
#endif
        {
          acc0[u] = applyCast<Acc, T>(acc1[u]);
        }
      }

#if __CUDA_ARCH__ >= 1000
      if NCCL_IF_CONSTEXPR (EnableTma) {
        // threads flush data to point of consistency for async proxy
        ptx::fence_proxy_async(ptx::space_shared);
        __syncwarp();
      }
#endif

      dr = 0;
      r = rank;
      NVCC_PRAGMA_UNROLL(2)
      for (int partial=0; partial <= 1; partial++) {
        NVCC_PRAGMA_UNROLL_DISABLED
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          NVCC_PRAGMA_UNROLL_AUTO
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && dr == nRanks) break;
#if __CUDA_ARCH__ >= 1000
            if NCCL_IF_CONSTEXPR (EnableTma) {
              if (lane == 0) {
                ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, outPacks.peerPtr(world, r), tmaSmem->buff[0], tileSize);
              }
            } else
#endif
            {
              NVCC_PRAGMA_UNROLL(UnrollPacks)
              for (int u=0; u < UnrollPacks; u++) {
                outPacks.peerPtr(world, r)[u*WARP_SIZE] = acc0[u];
              }
            }
            if (++r == nRanks) r = 0;
          }
        }
      }
#if __CUDA_ARCH__ >= 1000
      if NCCL_IF_CONSTEXPR (EnableTma) {
        if (lane == 0) {
          ptx::cp_async_bulk_commit_group();
          ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
        }
        __syncwarp();
      }
#endif

      inpPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      outPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

#if __CUDA_ARCH__ >= 1000
      // Load data for next iteration.
      if NCCL_IF_CONSTEXPR (EnableTma) {
        if (lane == 0) {
          cuda::device::memcpy_async_tx(tmaSmem->buff[0], inpPacks.peerPtr(world, rank), cuda::aligned_size_t<16>(tileSize), tmaSmem->bar);
          tmaSize += tileSize;
        }
      } else
#endif
      {
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u=0; u < UnrollPacks; u++) {
          acc0[u] = inpPacks.peerPtr(world, rank)[u*WARP_SIZE];
        }
      }
    }
  }
}

template<int UnrollPeers, typename Red, typename T>
static __device__ __forceinline__ void allreduceEnds(
    ncclSymkArgsHandler const& handler, int tn, int t, Red red,
    ncclSymPtr<T> input, ncclSymPtr<T> output,
    size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  using Acc = typename Red::EltType;

  ncclTeam world = ncclTeamWorld(handler.comm);
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  ncclSymPtr<BytePack<sizeof(T)>> inpPacks = (ncclSymPtr<BytePack<sizeof(T)>>)input;
  ncclSymPtr<BytePack<sizeof(T)>> outPacks = (ncclSymPtr<BytePack<sizeof(T)>>)output;

  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nSufElts-nPreElts+i;
    BytePack<sizeof(T)> acc0 = inpPacks.peerPtr(world, rank)[elt];
    BytePack<sizeof(Acc)> acc1;
    BytePack<sizeof(T)> tmp[UnrollPeers];
    int dr = 1;
    int r = rank+1;
    if (nRanks == r) r = 0;
    bool first = true;

    NVCC_PRAGMA_UNROLL(2)
    for (int partial=0; partial <= 1; partial++) {
      NVCC_PRAGMA_UNROLL_DISABLED
      for (int j = 0;
           partial ? j < 1 : (dr + UnrollPeers <= nRanks);
           partial ? j++ : (dr += UnrollPeers)) {
        if (partial && dr == nRanks) break;

        NVCC_PRAGMA_UNROLL_AUTO
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && u!=0 && dr+u == nRanks) break;
          tmp[u] = inpPacks.peerPtr(world, r)[elt];
          r += 1;
          if (r == nRanks) r = 0;
        }
        if (first) {
          first = false;
          acc1 = applyCast<T, Acc>(acc0);
        }
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && u!=0 && dr+u == nRanks) break;
          acc1 = applyReduce(red, acc1, applyCast<T, Acc>(tmp[u]));
        }
      }
    }

    acc0 = applyCast<Acc, T>(acc1);
    dr = 0;
    r = rank;
    NVCC_PRAGMA_UNROLL(2)
    for (int partial=0; partial <= 1; partial++) {
      NVCC_PRAGMA_UNROLL_DISABLED
      for (int j=0;
           partial ? j < 1 : (dr + UnrollPeers <= nRanks);
           partial ? j++ : (dr += UnrollPeers)) {
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && dr+u == nRanks) break;
          outPacks.peerPtr(world, r)[elt] = acc0;
          r += 1;
          if (r == nRanks) r = 0;
        }
      }
    }
  }
}

template<typename Red, typename T, bool EnableTma>
static __device__ void allreduce(
    ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {
  int const& nRanks = handler.comm.nRanks;
  int const& nRanks_rcp32 = handler.nRanks_rcp32;
  size_t nBytes = nElts*sizeof(T);
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);
  uint32_t nRanks_nBlocks_rcp32 = nccl::utility::imulRcp32(nRanks, nRanks_rcp32, nBlocks, nBlocks_rcp32);

  uint32_t nPreBytes = (16u - input.offset)%16u;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if ((input.offset - output.offset)%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks =
#if __CUDA_ARCH__ >= 1000
      EnableTma ? 8 :
#endif
      4, UnrollPeers = 2;

    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      allreduceDeep<BytePerPack, UnrollPacks, UnrollPeers, T, EnableTma>(
        handler, tn, t, waitNeeded, bar, red,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && (input.offset - output.offset)%4 == 0)) {
    constexpr int BytePerPack = 4, UnrollPacks = 4, UnrollPeers = 4;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      allreduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T, /*EnableTma*/false>(
        handler, tn, t, waitNeeded, bar, red,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  allreduceEnds<UnrollPeers>(handler, tn, t, red, input, output, nElts, nPreBytes/sizeof(T), nSufElts);
}

template<template<typename> typename Red, typename T, bool EnableTma>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x
  };

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);

  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);

  bool waitNeeded = true;
  handler.forEachWork<T>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<T> input, ncclSymPtr<T> output) {
        // Threads numbered globally such that we round robin warps by rank then block.
        int gt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                           rank, nRanks,
                           block, nBlocks,
                           threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int gtn = nRanks*nBlocks*blockDim.x;

        allreduce<decltype(red), T, EnableTma>(handler, gtn, gt, nBlocks, waitNeeded, bar, red, input, output, nElts);

        waitNeeded = false;
      }
    );

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_RSxLD_AGxST_impl<Red, T, /*EnableTma=*/false>(args);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxTmaLD_AGxTmaST(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_RSxLD_AGxST_impl<Red, T, /*EnableTma=*/true>(args);
}

template<typename Red, typename T>
static __device__ void allreduceMultimem(
    int tn, int t, Red red, T* input, T* output, size_t nElts
  ) {
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts*sizeof(T);

  constexpr int BytePerPack = LoadMultimem_BigPackSize<Red>::BigPackSize;
  uint32_t nPreBytes = (BytePerPack - inputUptr)%BytePerPack;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

  if (alignof(T) == BytePerPack || (inputUptr-outputUptr)%BytePerPack == 0) {
    constexpr int UnrollPacks = 16*8/BytePerPack;
    constexpr int BytePerChunk = UnrollPacks*WARP_SIZE*BytePerPack;
    uintptr_t cursor = nPreBytes;
    int nChunks = (nBytes-cursor)/BytePerChunk;
    uintptr_t cursorAfter = cursor + uintptr_t(nChunks)*BytePerChunk;
    nSufBytes = nBytes - cursorAfter;
    cursor += (t/WARP_SIZE)*UnrollPacks*WARP_SIZE*BytePerPack;
    cursor += (t%WARP_SIZE)*BytePerPack;
    int nIters = nChunks - t/WARP_SIZE;
    NVCC_PRAGMA_UNROLL_DISABLED
    while (0 < nIters) {
      BytePack<BytePerPack> tmp[UnrollPacks];
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u=0; u < UnrollPacks; u++) {
        tmp[u] = applyLoadMultimem<Red, BytePerPack>(red, inputUptr + cursor + u*WARP_SIZE*BytePerPack);
      }
      NVCC_PRAGMA_UNROLL_AUTO
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
  NVCC_PRAGMA_UNROLL(4)
  for (uintptr_t i = t*sizeof(T); i < nPreBytes + nSufBytes; i += tn*sizeof(T)) {
    uintptr_t cursor = i < nPreBytes ? i : nBytes-nSufBytes+(i-nPreBytes);
    BytePack<sizeof(T)> val = applyLoadMultimem<Red, sizeof(T)>(red, inputUptr + cursor);
    multimem_st_global(outputUptr + cursor, val);
  }
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true
  };

  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/true>::Type> red(handler.devWork->redOpArg);

  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  auto const& multimem = handler.comm.lsaMultimem;

  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  handler.forEachWork<T>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<T> input, ncclSymPtr<T> output) {
        // Threads numbered globally such that we round robin warps by rank then block.
        int gt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                           rank, nRanks,
                           block, nBlocks,
                           threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int gtn = nRanks*nBlocks*blockDim.x;

        allreduceMultimem(gtn, gt, red, input.multimemPtr(multimem), output.multimemPtr(multimem), nElts);
      }
    );

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R_impl(ncclSymkDevWorkArgs const* args, bool multimem) {
  ncclSymkArgsHandler handler{args};
  ncclLLA2ASession<ncclCoopCta> lla2a(
    ncclCoopCta(), handler.comm, ncclTeamLsa(handler.comm), handler.lsaLLA2A,
    blockIdx.x, ncclSymkMaxThreads, multimem, handler.comm.lsaMultimem
  );

  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  using Acc = typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type;
  Red<Acc> red(handler.devWork->redOpArg);

  using Pack = BytePack<8>;
  using AccPack = BytePack<8*sizeof(Acc)/sizeof(T)>;
  constexpr int EltPerPack = 8/sizeof(T);

  handler.singleWork<T>(
      [&]__device__(int nElts, int nAllElts,
                    ncclSymPtr<T> inputPtr, ncclSymPtr<T> outputPtr) {
        int nPacks = divUp(nElts, EltPerPack);

        T* input = (T*)inputPtr.localPtr();
        T* output = (T*)outputPtr.localPtr();

        bool packAligned = 8 <= alignof(T) || (nElts*sizeof(T) | (uintptr_t)input | (uintptr_t)output)%8 == 0;

        ncclCoopCta cta;
        int t = threadIdx.x;
        int tn = ncclSymkMaxThreads;

        if (__builtin_expect(packAligned, true)) {
          NVCC_PRAGMA_UNROLL_DISABLED
          while (0 < nPacks) {
            if (t < nPacks) {
              int nIterPacks = min(nPacks, tn);
              Pack inp = loadPack<Pack>((Pack*)input, t, nPacks);
              lla2a.bcast(/*slot=*/nIterPacks*rank + t, inp);
              AccPack out = lla2a.template recvReduce</*Unroll=*/8, Pack>(
                /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
                /*eltToAcc=*/[&] __device__ (Pack x)->AccPack {
                  return applyCast<T, Acc>(x);
                },
                /*reduce=*/[&] __device__ (AccPack a, AccPack b)->AccPack {
                  return applyReduce(red, a, b);
                }
              );
              storePack((Pack*)output, t, nPacks, applyCast<Acc, T>(out));
            }
            lla2a.endEpoch(cta);

            input += tn*EltPerPack;
            output += tn*EltPerPack;
            nPacks -= tn;
          }
        } else {
          NVCC_PRAGMA_UNROLL_DISABLED
          while (0 < nElts) {
            if (t*EltPerPack < nElts) {
              int nIterPacks = min(nPacks, tn);
              Pack inp = loadPack<Pack>(input, t*EltPerPack, nElts);
              lla2a.bcast(/*slot=*/nIterPacks*rank + t, inp);
              AccPack out = lla2a.template recvReduce</*Unroll=*/8, Pack>(
                /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
                /*eltToAcc=*/[&] __device__ (Pack x)->AccPack {
                  return applyCast<T, Acc>(x);
                },
                /*reduce=*/[&] __device__ (AccPack a, AccPack b)->AccPack {
                  return applyReduce(red, a, b);
                }
              );
              storePack(output, t*EltPerPack, nElts, applyCast<Acc, T>(out));
            }
            lla2a.endEpoch(cta);

            input += tn*EltPerPack;
            output += tn*EltPerPack;
            nElts -= tn*EltPerPack;
            nPacks -= tn;
          }
        }
      }
    );
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_AGxLL_R_impl<Red, T>(args, /*multimem=*/false);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLLMC_R(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllReduce_AGxLL_R_impl<Red, T>(args, /*multimem=*/true);
}
