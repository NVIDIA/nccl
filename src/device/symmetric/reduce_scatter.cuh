#include "symmetric.h"
#include "symmetric/kernel.cuh"
#include "symmetric/primitives.cuh"

template<int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, typename Red>
static __device__ void reduceDeep(
    ncclSymPrims& prim, int tn, int t, bool waitNeeded,
    Red red, char* inputRank0, char* outputHere, int32_t nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  using Acc = typename Red::EltType;
  using AccPack = BytePack<BytePerPack*sizeof(Acc)/sizeof(T)>;

  int wn = tn/WARP_SIZE;
  int w = t/WARP_SIZE;
  int lane = t%WARP_SIZE;
  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;
  uint32_t const& stride4G = prim.stride4G;
  Pack* inpRank0 = (Pack*)inputRank0 + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  Pack* outHere = (Pack*)outputHere + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  Pack acc0[UnrollPacks];

  nIters -= w;
  if (0 < nIters) {
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      acc0[u] = add4G(inpRank0, rank*stride4G)[u*WARP_SIZE];
    }
  }

  if (waitNeeded) prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  if (0 < nIters) {
    while (true) {
      AccPack acc1[UnrollPacks];
      int r = rank+1;
      if (r == nRanks) r = 0;
      { Pack tmp1[UnrollPacks];
        #pragma unroll
        for (int u=0; u < UnrollPacks; u++) {
          tmp1[u] = add4G(inpRank0, r*stride4G)[u*WARP_SIZE];
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
              tmp1[ur][u] = add4G(inpRank0, r*stride4G)[u*WARP_SIZE];
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
      for (int u=0; u < UnrollPacks; u++) outHere[u*WARP_SIZE] = acc0[u];

      inpRank0 += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      outHere += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

      // Load data for next iteration.
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        acc0[u] = add4G(inpRank0, rank*stride4G)[u*WARP_SIZE];
      }
    }
  }
}

template<int UnrollPeers, typename Red, typename T>
static __device__ void reduceEnds(
    ncclSymPrims& prim, int tn, int t, Red red,
    T* inputRank0, T* outputHere, size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  using Acc = typename Red::EltType;

  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;
  uint32_t const& stride4G = prim.stride4G;
  BytePack<sizeof(T)>* inpRank0 = (BytePack<sizeof(T)>*)inputRank0;
  BytePack<sizeof(T)>* outHere = (BytePack<sizeof(T)>*)outputHere;
  #pragma unroll 1
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nSufElts-nPreElts+i;
    BytePack<sizeof(T)> acc0 = *add4G(inpRank0+elt, rank*stride4G);
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
          tmp[u] = *add4G(inpRank0+elt, r*stride4G);
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
    outHere[elt] = acc0;
  }
}

template<typename Red, typename T>
static __device__ void reduce(
    ncclSymPrims& prim, int tn, int t, bool waitNeeded,
    Red red, T* input, T* output, size_t nElts
  ) {
  int nRanks = prim.nRanks;
  int nBlocks = prim.nBlocks;
  // Mpve input to rank=0
  input = prim.peerPtr(0, input);

  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  uint32_t alignment = uint32_t(inputUptr - outputUptr);
  size_t nBytes = nElts*sizeof(T);

  uint32_t nPreBytes = (16u - inputUptr)%16u;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if (alignment%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 2;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, prim.nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      reduceDeep<BytePerPack, UnrollPacks, UnrollPeers, T>(
        prim, tn, t, waitNeeded, red,
        (char*)input + cursor, (char*)output + cursor,
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
    chunks -= imodFast32(chunks, nRanks*nBlocks, prim.nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      reduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T>(
        prim, tn, t, waitNeeded, red,
        (char*)input + cursor, (char*)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  reduceEnds<UnrollPeers>(prim, tn, t, red, input, output, nElts, nPreBytes/sizeof(T), nSufElts);
}


template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymRun_ReduceScatter_LD(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, ncclSymPrims_UseBarrier);
  Red<typename ncclSymAccumType<Red, T, /*nvls=*/false>::Type> red(args->redOpArg);

  // Round robin warps over blocks.
  int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                    prim.block, prim.nBlocks,
                    threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int tn = prim.nBlocks*blockDim.x;

  prim.barrierArrive(ncclCoopCta(), /*release=*/false);
  //prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  reduce(prim, tn, t, /*waitNeeded=*/true, red, (T*)args->input + prim.rank*args->nElts, (T*)args->output, args->nElts);

  prim.barrierArrive(ncclCoopCta(), /*release=*/false);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
}


template<typename Red, typename T>
static __device__ void reduceMultimem(
    ncclSymPrims& prim, int tn, int t, Red red, T* input, T* output, size_t nElts
  ) {
  // Mpve input to multimem
  input = prim.multimemPtr(input);

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
    cursor += tn*sizeof(T);
  }
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymRun_ReduceScatter_LDMC(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, ncclSymPrims_UseBarrier|ncclSymPrims_UseMultimem);
  Red<typename ncclSymAccumType<Red, T, /*nvls=*/true>::Type> red(args->redOpArg);

  // Round robin warps over blocks.
  int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                    prim.block, prim.nBlocks,
                    threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int tn = prim.nBlocks*blockDim.x;

  prim.barrierArrive(ncclCoopCta(), /*release=*/false);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  reduceMultimem(prim, tn, t, red, (T*)args->input + prim.rank*args->nElts, (T*)args->output, args->nElts);

  prim.barrierArrive(ncclCoopCta(), /*release=*/false);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
}

// T is user type, EltType is the most aligned type
template<typename T, typename Red, typename EltType>
__device__ __forceinline__ void ncclSymRun_ReduceScatter_LL_body(
    ncclSymPrims &prim, Red red, EltType* input, EltType* output, int nElts, int nPacks, int nStrideElts) {
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(EltType);

  int nRanks = prim.nRanks;
  int rank = prim.rank;
  int t = threadIdx.x;
  int tn = ncclSymMaxThreads;
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
      prim.sendLL(peer, rank*nIterPacks + pack, got);
      peer += tn_div_nPacks;
      pack += tn_mod_nPacks;
      if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
    }

    if (t < nIterPacks) {
      Pack got = prim.template recvReduceLL<Pack, T>(t, nIterPacks, red);
      storePack(output, t*EltPerPack, nElts, got);
    }
    prim.endLL(cta);

    input += tn*EltPerPack;
    output += tn*EltPerPack;
    nElts -= tn*EltPerPack;
    nPacks -= tn;
  }
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymRun_ReduceScatter_LL(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, ncclSymPrims_UseLL);
  Red<typename ncclSymAccumType<Red, T, /*nvls=*/false>::Type> red(args->redOpArg);

  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(T);
  int nAllElts = args->nElts;
  int nAllPacks = divUp(nAllElts, EltPerPack);
  uint32_t nPackPerBlock, nPackModBlock;
  idivmodFast32(&nPackPerBlock, &nPackModBlock, nAllPacks, prim.nBlocks, prim.nBlocks_rcp32);
  int blockPackBegin = prim.block*nPackPerBlock + minval<int>(prim.block, nPackModBlock);
  int blockPackEnd = blockPackBegin + nPackPerBlock + (prim.block < nPackModBlock ? 1 : 0);
  int nPacks = blockPackEnd - blockPackBegin;
  int nElts = nAllElts - blockPackBegin*EltPerPack;
  nElts = min(nElts, nPacks*EltPerPack);
  T* input = (T*)args->input + blockPackBegin*EltPerPack;
  T* output = (T*)args->output + blockPackBegin*EltPerPack;

  uint32_t lowBits = args->nElts*sizeof(T);
  lowBits |= (uint32_t)reinterpret_cast<uintptr_t>(args->input);
  lowBits |= (uint32_t)reinterpret_cast<uintptr_t>(args->output);
  if (__builtin_expect(lowBits%8 == 0, true)) {
    ncclSymRun_ReduceScatter_LL_body<T>(prim, red, (Pack*)input, (Pack*)output, nPacks, nPacks, nAllElts/EltPerPack);
  } else {
    ncclSymRun_ReduceScatter_LL_body<T>(prim, red, input, output, nElts, nPacks, nAllElts);
  }
}
