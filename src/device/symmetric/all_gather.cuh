#include "symmetric.h"
#include "symmetric/kernel.cuh"
#include "symmetric/primitives.cuh"

template<int BytePerPack, int UnrollPacks, int UnrollPeers>
static __device__ void bcastDeep(
    ncclSymPrims& prim, int tn, int t, bool waitNeeded,
    char* inputHere, char* outputRank0, bool inPlace, int nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  int wn = tn/WARP_SIZE;
  int w = t/WARP_SIZE;
  int lane = t%WARP_SIZE;
  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;
  uint32_t const& stride4G = prim.stride4G;
  Pack* inpHere = (Pack*)inputHere + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  Pack* outRank0 = (Pack*)outputRank0 + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  Pack tmp[UnrollPacks];

  nIters -= w;
  if (0 < nIters) {
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      tmp[u] = inpHere[u*WARP_SIZE];
    }
  }

  if (waitNeeded) prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  if (0 < nIters) {
    while (true) {
      int dr = inPlace ? 1 : 0;
      int r = rank + dr;
      if (r == nRanks) r = 0;
      #pragma unroll 2
      for (int partial=0; partial <= 1; partial++) {
        #pragma unroll 1
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && dr == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              add4G(outRank0, r*stride4G)[u*WARP_SIZE] = tmp[u];
            }
            if (++r == nRanks) r = 0;
          }
        }
      }
      inpHere += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      outRank0 += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

      // Load data for next iteration.
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        tmp[u] = inpHere[u*WARP_SIZE];
      }
    }
  }
}

template<int UnrollPeers, typename T>
static __device__ void bcastEnds(
    ncclSymPrims& prim, int tn, int t,
    T* inputHere, T* outputRank0, bool inPlace, size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;
  uint32_t const& stride4G = prim.stride4G;
  BytePack<sizeof(T)>* inpHere = (BytePack<sizeof(T)>*)inputHere;
  BytePack<sizeof(T)>* outRank0 = (BytePack<sizeof(T)>*)outputRank0;
  #pragma unroll 1
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nPreElts-nSufElts+i;
    BytePack<sizeof(T)> tmp = inpHere[elt];
    int dr = inPlace ? 1 : 0;
    int r = rank + dr;
    if (r == nRanks) r = 0;
    #pragma unroll 1
    for (; dr + UnrollPeers <= nRanks; dr += UnrollPeers) {
      #pragma unroll UnrollPeers
      for (int u=0; u < UnrollPeers; u++) {
        *add4G(outRank0+elt, r*stride4G) = tmp;
        if (++r == nRanks) r = 0;
      }
    }
    #pragma unroll UnrollPeers
    for (int u=0; u < UnrollPeers; u++) {
      if (dr+u == nRanks) break;
      *add4G(outRank0+elt, r*stride4G) = tmp;
      if (++r == nRanks) r = 0;
    }
  }
}

template<typename T>
static __device__ void bcast(
    ncclSymPrims& prim, int tn, int t, bool waitNeeded, T* input, T* output, size_t nElts
  ) {
  bool inPlace = (input == output);
  // Mpve to rank=0
  output = prim.peerPtr(0, output);

  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts*sizeof(T);

  uint32_t nPreBytes = (128u - inputUptr)%128u;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if ((inputUptr-outputUptr)%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 2;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, prim.nBlocks, prim.nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      bcastDeep<BytePerPack, UnrollPacks, UnrollPeers>(
        prim, tn, t, waitNeeded,
        (char*)input + cursor, (char*)output + cursor, inPlace,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && (inputUptr-outputUptr)%4 == 0)) {
    constexpr int BytePerPack = 4, UnrollPacks = 4, UnrollPeers = 4;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, prim.nBlocks, prim.nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      bcastDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers>(
        prim, tn, t, waitNeeded,
        (char*)input + cursor, (char*)output + cursor, inPlace,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  bcastEnds<UnrollPeers>(prim, tn, t, input, output, inPlace, nElts, nPreBytes/sizeof(T), nSufElts);
}

__device__ __forceinline__ void ncclSymRun_AllGather_ST(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, ncclSymPrims_UseBarrier);
  int const& rank = prim.rank;

  // Threads numbered over rank.
  int bt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                     prim.block, prim.nBlocks,
                     threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int btn = prim.nBlocks*blockDim.x;

  prim.barrierArrive(ncclCoopCta(), /*release=*/false);
  //prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  bcast(prim, btn, bt, /*waitNeeded=*/true, (char*)args->input, (char*)args->output + rank*args->nElts, args->nElts);

  prim.barrierArrive(ncclCoopCta(), /*release=*/true);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
}


template<typename T>
static __device__ void bcastMultimem(
    ncclSymPrims& prim, int tn, int t, T* input, T* output, size_t nElts
  ) {
  // Move output to multimem
  output = prim.multimemPtr(output);

  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts*sizeof(T);

  uint32_t nPreBytes = (16-inputUptr)%16;
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
    cursor += tn*sizeof(T);
  }
}

__device__ __forceinline__ void ncclSymRun_AllGather_STMC(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, ncclSymPrims_UseBarrier|ncclSymPrims_UseMultimem);
  int const& rank = prim.rank;

  char* input = args->input;
  char* output = args->output;
  size_t bytes = args->nElts;
  // Round robin memory to blocks.
  int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                    prim.block, prim.nBlocks,
                    threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int tn = prim.nBlocks*blockDim.x;

  prim.barrierArrive(ncclCoopCta(), /*release=*/false);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  bcastMultimem(prim, tn, t, input, output + rank*bytes, bytes);

  prim.barrierArrive(ncclCoopCta(), /*release=*/true);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
}

template<typename EltType>
static __device__ void allgather_LL_body(
    ncclSymPrims &prim, EltType* input, EltType* output, int nElts, int nPacks, int nStrideElts
  ) {
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(EltType);

  ncclCoopCta cta;
  int rank = prim.rank;
  int nRanks = prim.nRanks;
  constexpr int tn = ncclSymMaxThreads;
  int t = threadIdx.x;

  #pragma unroll 1
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    if (t < nIterPacks) {
      Pack x = loadPack<Pack>(input, t*EltPerPack, nElts);
      prim.bcastLL(/*slot=*/nIterPacks*rank + t, x);
    }

    int tn_div_nPacks = tn/nIterPacks;
    int tn_mod_nPacks = tn%nIterPacks;
    int peer = t/nIterPacks;
    int pack = t%nIterPacks;
    #if 1
      // NOTE: Unrolling speedup on eos nranks=8 size=64K: 5.7us vs 6.7us
      constexpr int Unroll = 4;
      #pragma unroll 1
      for (int i = t; i < (nRanks*nIterPacks & -(Unroll*tn)); i += Unroll*tn) {
        Pack got[Unroll];
        prim.template recvLL<Unroll, Unroll>(i, Unroll, tn, /*&*/got);
        #pragma unroll
        for (int u=0; u < Unroll; u++) {
          storePack<Pack>(output + peer*nStrideElts, pack*EltPerPack, nElts, got[u]);
          peer += tn_div_nPacks;
          pack += tn_mod_nPacks;
          if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
        }
      }

      int i = (nRanks*nIterPacks & -(Unroll*tn)) + t;
      int n = (nRanks*nIterPacks)/tn % Unroll;
      if (i + n*tn < nRanks*nIterPacks) n += 1;
      if (n != 0) {
        Pack got[Unroll];
        prim.template recvLL<1, Unroll>(i, n, tn, /*&*/got);
        #pragma unroll
        for (int u=0; u < Unroll; u++) {
          if (u != 0 && u == n) break;
          storePack(output + peer*nStrideElts, pack*EltPerPack, nElts, got[u]);
          peer += tn_div_nPacks;
          pack += tn_mod_nPacks;
          if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
        }
      }
    #else
      // The non-unrolled but "obviously correct" implementation for reference.
      #pragma unroll 1
      for (int i = t; i < nRanks*nIterPacks; i += tn) {
        Pack got = prim.template recvLL<Pack>(i);
        storePack(output + peer*nStrideElts, pack*EltPerPack, nElts, got);
        peer += tn_div_nPacks;
        pack += tn_mod_nPacks;
        if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
      }
    #endif

    prim.endLL(cta);

    input += tn*EltPerPack;
    output += tn*EltPerPack;
    nElts -= tn*EltPerPack;
    nPacks -= tn;
  }
}

static __device__ void ncclSymRun_AllGather_LL_impl(ncclSymDevArgs const* args, bool multimem) {
  ncclSymPrims prim(args->comm, ncclSymPrims_UseLL | multimem*ncclSymPrims_UseMultimem);
  using Pack = BytePack<8>;
  constexpr int BytePerPack = 8;
  int nElts = args->nElts;
  int nPacks = divUp(nElts, BytePerPack);

  uint32_t nPackPerBlock, nPackModBlock;
  idivmodFast32(&nPackPerBlock, &nPackModBlock, nPacks, prim.nBlocks, prim.nBlocks_rcp32);
  int blockPackBegin = prim.block*nPackPerBlock + minval<int>(prim.block, nPackModBlock);
  int blockPackEnd = blockPackBegin + nPackPerBlock + (prim.block < nPackModBlock ? 1 : 0);
  int nBlockPacks = blockPackEnd - blockPackBegin;
  int nBlockElts = nElts - blockPackBegin*BytePerPack;
  nBlockElts = min(nBlockElts, nBlockPacks*BytePerPack);
  char* blockInput = args->input + blockPackBegin*BytePerPack;
  char* blockOutput = args->output + blockPackBegin*BytePerPack;

  uint32_t lowBits = args->nElts;
  lowBits |= (uint32_t)reinterpret_cast<uintptr_t>(args->input);
  lowBits |= (uint32_t)reinterpret_cast<uintptr_t>(args->output);
  if (__builtin_expect(lowBits%8 == 0, true)) {
    // NOTE: Specializing for 8-byte alignment in one case help at size=65K: 8.9us vs 5.6us
    allgather_LL_body(prim, (BytePack<8>*)blockInput, (BytePack<8>*)blockOutput, nBlockElts/8, nBlockPacks, nElts/8);
  } else {
    allgather_LL_body(prim, blockInput, blockOutput, nBlockElts, nBlockPacks, nElts);
  }
}

__device__ __forceinline__ void ncclSymRun_AllGather_LL(ncclSymDevArgs const* args) {
  ncclSymRun_AllGather_LL_impl(args, /*multimem=*/false);
}

__device__ __forceinline__ void ncclSymRun_AllGather_LLMC(ncclSymDevArgs const* args) {
  ncclSymRun_AllGather_LL_impl(args, /*multimem=*/true);
}
