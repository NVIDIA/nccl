#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"

template<int BytePerPack, int UnrollPacks, int UnrollPeers>
static __device__ void bcastDeep(
    ncclSymkArgsHandler const& handler, int tn, int t,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    ncclSymPtr<char> input, ncclSymPtr<char> output, bool inPlace, int nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  int wn = tn/WARP_SIZE;
  int w = t/WARP_SIZE;
  int lane = t%WARP_SIZE;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  Pack* inpPacks = (Pack*)input.localPtr() + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  Pack tmp[UnrollPacks];

  nIters -= w;
  if (0 < nIters) {
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      tmp[u] = inpPacks[u*WARP_SIZE];
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

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
              outPacks.lsaPtr(r)[u*WARP_SIZE] = tmp[u];
            }
            if (++r == nRanks) r = 0;
          }
        }
      }
      inpPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      outPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

      // Load data for next iteration.
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        tmp[u] = inpPacks[u*WARP_SIZE];
      }
    }
  }
}

template<int UnrollPeers, typename T>
static __device__ void bcastEnds(
    ncclSymkArgsHandler const& handler, int tn, int t,
    ncclSymPtr<T> input, ncclSymPtr<T> output, bool inPlace, size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  BytePack<sizeof(T)>* inpPacks = (BytePack<sizeof(T)>*)input.localPtr();
  ncclSymPtr<BytePack<sizeof(T)>> outPacks = (ncclSymPtr<BytePack<sizeof(T)>>)output;
  #pragma unroll 1
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nPreElts-nSufElts+i;
    BytePack<sizeof(T)> tmp = inpPacks[elt];
    int dr = inPlace ? 1 : 0;
    int r = rank + dr;
    if (r == nRanks) r = 0;
    #pragma unroll 1
    for (; dr + UnrollPeers <= nRanks; dr += UnrollPeers) {
      #pragma unroll UnrollPeers
      for (int u=0; u < UnrollPeers; u++) {
        outPacks.lsaPtr(r)[elt] = tmp;
        if (++r == nRanks) r = 0;
      }
    }
    #pragma unroll UnrollPeers
    for (int u=0; u < UnrollPeers; u++) {
      if (dr+u == nRanks) break;
      outPacks.lsaPtr(r)[elt] = tmp;
      if (++r == nRanks) r = 0;
    }
  }
}

template<typename T>
static __device__ void bcast(
    ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {
  bool inPlace = (input == output);
  size_t nBytes = nElts*sizeof(T);
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);

  uint32_t nPreBytes = (16 - input.offset)%16;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if ((input.offset - output.offset)%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 2;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      bcastDeep<BytePerPack, UnrollPacks, UnrollPeers>(
        handler, tn, t, waitNeeded, bar,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        inPlace, chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && (input.offset - output.offset)%4 == 0)) {
    constexpr int BytePerPack = 4, UnrollPacks = 4, UnrollPeers = 4;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      bcastDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers>(
        handler, tn, t, waitNeeded, bar,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        inPlace, chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  bcastEnds<UnrollPeers>(handler, tn, t, input, output, inPlace, nElts, nPreBytes/sizeof(T), nSufElts);
}

__device__ __forceinline__ void ncclSymkRun_AllGather_ST(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x
  };
  int const& rank = handler.comm.rank;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);

  bool waitNeeded = true;
  handler.forEachWork<char>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<char> input, ncclSymPtr<char> output) {
        // Threads numbered over rank.
        int bt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                           block, nBlocks,
                           threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int btn = nBlocks*blockDim.x;

        bcast(handler, btn, bt, nBlocks, waitNeeded, bar, input, output + rank*nAllElts, nElts);

        waitNeeded = false;
      }
    );

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

__device__ __forceinline__ void ncclSymkRun_AllGather_STMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar(
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true
  );
  int const& rank = handler.comm.rank;

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  handler.forEachWork<char>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<char> input, ncclSymPtr<char> output) {
        // Round robin memory to blocks.
        int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                          block, nBlocks,
                          threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int tn = nBlocks*blockDim.x;

        bcastMultimem(handler, tn, t, input, output + rank*nAllElts, nElts);
      }
    );

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template<typename EltType>
static __device__ void allgather_LL_body(
    ncclSymkArgsHandler& handler, ncclLLA2ASession<ncclCoopCta>& lla2a,
    EltType* input, EltType* output, int nElts, int nPacks, int nStrideElts
  ) {
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(EltType);
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  int t = threadIdx.x;
  constexpr int tn = ncclSymkMaxThreads;

  #pragma unroll 1
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    if (t < nIterPacks) {
      Pack x = loadPack<Pack>(input, t*EltPerPack, nElts);
      lla2a.bcast(/*slot=*/nIterPacks*rank + t, x);
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
        lla2a.template recvUnrolled<Unroll, Unroll>(i, Unroll, tn, /*&*/got);
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
        lla2a.template recvUnrolled<1, Unroll>(i, n, tn, /*&*/got);
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
        Pack got = lla2a.template recv<Pack>(i);
        storePack(output + peer*nStrideElts, pack*EltPerPack, nElts, got);
        peer += tn_div_nPacks;
        pack += tn_mod_nPacks;
        if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
      }
    #endif

    lla2a.endEpoch(ncclCoopCta());

    input += tn*EltPerPack;
    output += tn*EltPerPack;
    nElts -= tn*EltPerPack;
    nPacks -= tn;
  }
}

static __device__ void ncclSymkRun_AllGather_LL_impl(ncclSymkDevWorkArgs const* args, bool multimem) {
  ncclSymkArgsHandler handler{args};
  ncclLLA2ASession<ncclCoopCta> lla2a(
    ncclCoopCta(), handler.comm, ncclTeamLsa(handler.comm), handler.lsaLLA2A, blockIdx.x, /*maxElts=*/ncclSymkMaxThreads, multimem, handler.comm.lsaMultimem
  );

  using Pack = BytePack<8>;
  constexpr int BytePerPack = 8;

  handler.singleWork<char>(
      [&]__device__(int nElts, int nAllElts,
                    ncclSymPtr<char> input, ncclSymPtr<char> output) {
        int nPacks = divUp(nElts, BytePerPack);

        char* blockInput = input.localPtr();
        char* blockOutput = output.localPtr();

        uint32_t lowBits = nAllElts;
        lowBits |= (uintptr_t)blockInput;
        lowBits |= (uintptr_t)blockOutput;
        if (__builtin_expect(lowBits%8 == 0, true)) {
          // NOTE: Specializing for 8-byte alignment in one case help at size=65K: 8.9us vs 5.6us
          allgather_LL_body(handler, lla2a, (BytePack<8>*)blockInput, (BytePack<8>*)blockOutput,
                            nElts/8, nPacks, nAllElts/8);
        } else {
          allgather_LL_body(handler, lla2a, blockInput, blockOutput, nElts, nPacks, nAllElts);
        }
      }
    );
}

__device__ __forceinline__ void ncclSymkRun_AllGather_LL(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_LL_impl(args, /*multimem=*/false);
}

__device__ __forceinline__ void ncclSymkRun_AllGather_LLMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_LL_impl(args, /*multimem=*/true);
}
