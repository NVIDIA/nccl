#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"

template<int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, typename Red>
static __device__ void reduceDeep(
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
  ncclSymPtr<Pack> inpPacks = (ncclSymPtr<Pack>)input + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  Pack acc0[UnrollPacks];

  nIters -= w;
  if (0 < nIters) {
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      acc0[u] = inpPacks.peerPtr(world, rank)[u*WARP_SIZE];
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  if (0 < nIters) {
    while (true) {
      AccPack acc1[UnrollPacks];
      int r = rank+1;
      if (r == nRanks) r = 0;
      { Pack tmp1[UnrollPacks];
        #pragma unroll
        for (int u=0; u < UnrollPacks; u++) {
          tmp1[u] = inpPacks.peerPtr(world, r)[u*WARP_SIZE];
        }
        #pragma unroll
        for (int u=0; u < UnrollPacks; u++) {
          acc1[u] = applyReduce(red, applyCast<T, Acc>(acc0[u]), applyCast<T, Acc>(tmp1[u]));
        }
      }

      r += 1;
      if (r == nRanks) r = 0;

      int dr = 2;
      #pragma unroll 2
      for (int partial=0; partial <= 1; partial++) {
        #pragma unroll 1
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          if (partial && dr == nRanks) break;

          Pack tmp1[UnrollPeers][UnrollPacks];
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && ur!=0 && dr+ur == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              tmp1[ur][u] = inpPacks.peerPtr(world, r)[u*WARP_SIZE];
            }
            r += 1;
            if (r == nRanks) r = 0;
          }
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && ur!=0 && dr+ur == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              acc1[u] = applyReduce(red, acc1[u], applyCast<T, Acc>(tmp1[ur][u]));
            }
          }
        }
      }

      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) acc0[u] = applyCast<Acc, T>(acc1[u]);

      #pragma unroll UnrollPacks
      for (int u=0; u < UnrollPacks; u++) outPacks.localPtr()[u*WARP_SIZE] = acc0[u];

      inpPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      outPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

      // Load data for next iteration.
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        acc0[u] = inpPacks.peerPtr(world, rank)[u*WARP_SIZE];
      }
    }
  }
}

template<int UnrollPeers, typename Red, typename T>
static __device__ void reduceEnds(
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
  #pragma unroll 1
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nSufElts-nPreElts+i;
    BytePack<sizeof(T)> acc0 = inpPacks.peerPtr(world, rank)[elt];
    BytePack<sizeof(Acc)> acc1;
    BytePack<sizeof(T)> tmp[UnrollPeers];
    int dr = 1;
    int r = rank+1;
    if (nRanks == r) r = 0;
    bool first = true;

    #pragma unroll 2
    for (int partial=0; partial <= 1; partial++) {
      #pragma unroll 1
      for (int j = 0;
           partial ? j < 1 : (dr + UnrollPeers <= nRanks);
           partial ? j++ : (dr += UnrollPeers)) {
        if (partial && dr == nRanks) break;

        #pragma unroll
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
        #pragma unroll
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && u!=0 && dr+u == nRanks) break;
          acc1 = applyReduce(red, acc1, applyCast<T, Acc>(tmp[u]));
        }
      }
    }

    acc0 = applyCast<Acc, T>(acc1);
    outPacks.localPtr()[elt] = acc0;
  }
}

template<typename Red, typename T>
static __device__ void reduce(
    ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {
  int const& nRanks = handler.comm.nRanks;
  int const& nRanks_rcp32 = handler.nRanks_rcp32;
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);
  uint32_t nRanks_nBlocks_rcp32 = nccl::utility::imulRcp32(nRanks, nRanks_rcp32, nBlocks, nBlocks_rcp32);

  uint32_t alignment = uint32_t(input.offset - output.offset);
  size_t nBytes = nElts*sizeof(T);

  uint32_t nPreBytes = (16u - input.offset)%16u;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if (alignment%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 2;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      reduceDeep<BytePerPack, UnrollPacks, UnrollPeers, T>(
        handler, tn, t, waitNeeded, bar, red,
        (ncclSymPtr<char>)input + cursor, (ncclSymPtr<char>)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && alignment%4 == 0)) {
    constexpr int BytePerPack = 4, UnrollPacks = 4, UnrollPeers = 4;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      reduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T>(
        handler, tn, t, waitNeeded, bar, red,
        (ncclSymPtr<char>)input + cursor, (ncclSymPtr<char>)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  reduceEnds<UnrollPeers>(handler, tn, t, red, input, output, nElts, nPreBytes/sizeof(T), nSufElts);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LD(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x
  };
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);
  int const& rank = handler.comm.rank;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);

  bool waitNeeded = true;
  handler.forEachWork<T>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<T> input, ncclSymPtr<T> output) {
        // Round robin warps over blocks.
        int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                          block, nBlocks,
                          threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int tn = nBlocks*blockDim.x;

        reduce(handler, tn, t, nBlocks, waitNeeded, bar, red, input + rank*nAllElts, output, nElts);

        waitNeeded = false;
      }
    );

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}

template<typename Red, typename T>
static __device__ void reduceMultimem(
    int tn, int t, Red red, T* input, T* output, size_t nElts
  ) {
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts*sizeof(T);

  constexpr int BytePerPack = LoadMultimem_BigPackSize<Red>::BigPackSize;
  uint32_t nPreBytes = (BytePerPack - inputUptr)%BytePerPack;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

  if (sizeof(T) == BytePerPack || (inputUptr-outputUptr)%BytePerPack == 0) {
    constexpr int UnrollPacks = 8*(16/BytePerPack);
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
        tmp[u] = applyLoadMultimem<Red, BytePerPack>(red, inputUptr + cursor + u*WARP_SIZE*BytePerPack);
      }
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        *reinterpret_cast<BytePack<BytePerPack>*>(outputUptr + cursor + u*WARP_SIZE*BytePerPack) = tmp[u];
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
    BytePack<sizeof(T)> val = applyLoadMultimem<Red, sizeof(T)>(red, inputUptr + cursor);
    *reinterpret_cast<BytePack<sizeof(T)>*>(outputUptr + cursor) = val;
  }
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LDMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true
  };
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/true>::Type> red(handler.devWork->redOpArg);

  int const& rank = handler.comm.rank;
  auto const& multimem = handler.comm.lsaMultimem;

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  handler.forEachWork<T>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<T> input, ncclSymPtr<T> output) {
        // Round robin warps over blocks.
        int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                          block, nBlocks,
                          threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int tn = nBlocks*blockDim.x;

        reduceMultimem(tn, t, red, input.multimemPtr(multimem) + rank*nAllElts, output.localPtr(), nElts);
      }
    );

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}

// T is user type, EltType is the most aligned type
template<typename T, typename Red, typename EltType>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL_body(
    ncclSymkArgsHandler& handler, ncclLLA2ASession<ncclCoopCta>& lla2a,
    Red red, EltType* input, EltType* output, int nElts, int nPacks, int nStrideElts) {
  using Pack = BytePack<8>;
  using Acc = typename Red::EltType;
  using AccPack = BytePack<8*sizeof(Acc)/sizeof(T)>;
  constexpr int EltPerPack = 8/sizeof(EltType);

  int const& nRanks = handler.comm.nRanks;
  int const& rank = handler.comm.rank;
  int t = threadIdx.x;
  constexpr int tn = ncclSymkMaxThreads;
  ncclCoopCta cta;

  #pragma unroll 1
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    int tn_div_nPacks = tn/nIterPacks;
    int tn_mod_nPacks = tn%nIterPacks;
    int peer = t/nIterPacks;
    int pack = t%nIterPacks;

    #pragma unroll 1
    for (int i = t; i < nRanks*nIterPacks; i += tn) {
      Pack got = loadPack<Pack>(input + peer*nStrideElts, pack*EltPerPack, nElts);
      lla2a.send(peer, rank*nIterPacks + pack, got);
      peer += tn_div_nPacks;
      pack += tn_mod_nPacks;
      if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
    }

    if (t < nIterPacks) {
      AccPack got = lla2a.template recvReduce</*Unroll=*/8, Pack>(
        /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
        /*eltToAcc=*/[&] __device__ (Pack x)->AccPack {
          return applyCast<T, Acc>(x);
        },
        /*reduce=*/[&] __device__ (AccPack a, AccPack b)->AccPack {
          return applyReduce(red, a, b);
        }
      );
      storePack(output, t*EltPerPack, nElts, applyCast<Acc, T>(got));
    }
    lla2a.endEpoch(cta);

    input += tn*EltPerPack;
    output += tn*EltPerPack;
    nElts -= tn*EltPerPack;
    nPacks -= tn;
  }
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLLA2ASession<ncclCoopCta> lla2a(
    ncclCoopCta(), handler.comm, ncclTeamLsa(handler.comm), handler.lsaLLA2A, blockIdx.x, ncclSymkMaxThreads
  );
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(handler.devWork->redOpArg);
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(T);

  handler.singleWork<T>(
      [&]__device__(int nElts, int nAllElts,
                    ncclSymPtr<T> inputPtr, ncclSymPtr<T> outputPtr) {
        int nPacks = divUp(nElts, EltPerPack);

        T* input = (T*)inputPtr.localPtr();
        T* output = (T*)outputPtr.localPtr();

        uint32_t lowBits = nAllElts*sizeof(T);
        lowBits |= (uintptr_t)input;
        lowBits |= (uintptr_t)output;
        if (__builtin_expect(lowBits%8 == 0, true)) {
          ncclSymkRun_ReduceScatter_LL_body<T>(handler, lla2a, red, (Pack*)input, (Pack*)output,
                                               nPacks, nPacks, divUp(nAllElts, EltPerPack));
        } else {
          ncclSymkRun_ReduceScatter_LL_body<T>(handler, lla2a, red, input, output, nElts, nPacks, nAllElts);
        }
      }
    );
}
