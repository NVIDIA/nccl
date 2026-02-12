#include "primitives.cuh"

struct SMemTag {}; // Shared memory
struct GMemTag {}; // Global memory
struct GenMemTag {}; // Generic memory (either global or shared)

// Like CUDA's __ldcs() except works for all types T and handles smem so long as
// it is tagged accurately.
template<typename T>
static __device__ __forceinline__ T ldcs(GenMemTag, T *p) {
  return *p;
}
template<typename T>
static __device__ __forceinline__ T ldcs(SMemTag, T *p) {
  __builtin_assume(__isShared(p));
  return *p;
}
template<typename T>
static __device__ __forceinline__ T ldcs(GMemTag, T *p) {
  union {
    T x;
    uint8_t u8[sizeof(T)];
    uint16_t u16[(sizeof(T)+2-1)/2];
    uint32_t u32[(sizeof(T)+4-1)/4];
    uint64_t u64[(sizeof(T)+8-1)/8];
    uint4 u32v4[(sizeof(T)+16-1)/16];
  };
  switch (alignof(T)) {
  case 1: for (int i=0; i < sizeof(T)/1; i++) u8[i] = __ldcs((uint8_t*)p + i); break;
  case 2: for (int i=0; i < sizeof(T)/2; i++) u16[i] = __ldcs((uint16_t*)p + i); break;
  case 4: for (int i=0; i < sizeof(T)/4; i++) u32[i] = __ldcs((uint32_t*)p + i); break;
  case 8: for (int i=0; i < sizeof(T)/8; i++) u64[i] = __ldcs((uint64_t*)p + i); break;
  case 16: for (int i=0; i < sizeof(T)/16; i++) u32v4[i] = __ldcs((uint4*)p + i); break;
  default: __builtin_unreachable();
  }
  return x;
}

// Like CUDA's __stcs() except works for all types T and handles smem so long as
// it is tagged accurately.
template<typename T>
static __device__ __forceinline__ void stcs(GenMemTag, T *p, T val) {
  *p = val;
}
template<typename T>
static __device__ __forceinline__ void stcs(SMemTag, T *p, T val) {
  __builtin_assume(__isShared(p));
  *p = val;
}
template<typename T>
static __device__ __forceinline__ void stcs(GMemTag, T *p, T val) {
  union {
    T x;
    uint8_t u8[sizeof(T)];
    uint16_t u16[(sizeof(T)+2-1)/2];
    uint32_t u32[(sizeof(T)+4-1)/4];
    uint64_t u64[(sizeof(T)+8-1)/8];
    uint4 u32v4[(sizeof(T)+16-1)/16];
  };
  x = val;
  switch (alignof(T)) {
  case 1: for (int i=0; i < sizeof(T)/1; i++) __stcs((uint8_t*)p + i, u8[i]); break;
  case 2: for (int i=0; i < sizeof(T)/2; i++) __stcs((uint16_t*)p + i, u16[i]); break;
  case 4: for (int i=0; i < sizeof(T)/4; i++) __stcs((uint32_t*)p + i, u32[i]); break;
  case 8: for (int i=0; i < sizeof(T)/8; i++) __stcs((uint64_t*)p + i, u64[i]); break;
  case 16: for (int i=0; i < sizeof(T)/16; i++) __stcs((uint4*)p + i, u32v4[i]); break;
  default: __builtin_unreachable();
  }
}

// Load packs from element buffer. Pack index=0 is loaded from the buffer rounded
// down to pack alignment so it begins with unused padding elements equal in
// number to how misaligned the buffer is. The last pack may alias out of bounds
// elements. If it is completely out of bounds it will not be loaded.
template<typename Pack, int nPacks, typename MemTag, typename Elt>
static __device__ __forceinline__ void loadPacks(
    Pack(&packs)[nPacks],
    MemTag mem,
    unsigned eltsAlignMin, // (Compile time) Minimum byte alignment of elts. Zero = infinitely aligned.
    Elt* elts, // The element buffer. Need not be aligned.
    int nElts, // Total number of elements so we can tell what is legal to load.
    int padElts, // Number of misaligned elements in first pack.
    int packIx, // Index of our first pack (typically coop.thread_rank()).
    int stride, // Index stride of packs (typically coop.size()).
    bool lastEmpty // Is our last pack completely out of bounds.
  ) {
  constexpr int nEltPerPack = sizeof(Pack)/sizeof(Elt);
  bool eltsAligned = sizeof(Pack)-1 <= eltsAlignMin-1 || reinterpret_cast<uintptr_t>(elts)%sizeof(Pack) == 0;
  if (__builtin_expect(padElts == 0 && eltsAligned, true)) {
    #pragma unroll nPacks
    for (int p=0; p < nPacks; p++) {
      if (p < nPacks-1 || !lastEmpty) {
        packs[p] = ldcs(mem, (Pack*)elts + packIx + p*stride);
      }
    }
  } else {
    Pack stage[nPacks];
    #pragma unroll 16/nEltPerPack
    for (int p=0; p < nPacks; p++) {
      union { Pack tmp; Elt elt[nEltPerPack]; };
      tmp = {};
      #pragma unroll 8
      for (int pe=0; pe < nEltPerPack; pe++) {
        int e = -padElts + (packIx + p*stride)*nEltPerPack + pe;
        // if (0 <= e && e < nElts)
        if (unsigned(e) < unsigned(nElts)) elt[pe] = ldcs(mem, elts + e);
      }
      stage[p] = tmp;
    }
    #pragma unroll
    for (int p=0; p < nPacks; p++) packs[p] = stage[p];
  }
}

// Reverse of loadPacks except the destination buffer alignemnt has no assumed
// relationship with the number of padding elements.
template<typename Pack, int nPacks, typename MemTag, typename Elt>
static __device__ __forceinline__ void storePacks(
    Pack(&packs)[nPacks], MemTag mem, unsigned eltsAlignMin, Elt* elts,
    int nElts, int padElts, int packIx, int stride, bool lastEmpty
  ) {
  constexpr int nEltPerPack = sizeof(Pack)/sizeof(Elt);
  bool eltsAligned = sizeof(Pack)-1 <= eltsAlignMin-1 || reinterpret_cast<uintptr_t>(elts)%sizeof(Pack) == 0;
  if (__builtin_expect(padElts == 0 && eltsAligned, true)) {
    #pragma unroll nPacks
    for (int p=0; p < nPacks; p++) {
      if (p < nPacks-1 || !lastEmpty) {
        stcs(mem, (Pack*)elts + packIx + p*stride, packs[p]);
      }
    }
  } else {
    Pack stage[nPacks];
    #pragma unroll
    for (int p=0; p < nPacks; p++) stage[p] = packs[p];
    #pragma unroll 16/nEltPerPack
    for (int p=0; p < nPacks; p++) {
      union { Pack tmp; Elt elt[nEltPerPack]; };
      tmp = stage[p];
      #pragma unroll 8
      for (int pe=0; pe < nEltPerPack; pe++) {
        int e = -padElts + (packIx + p*stride)*nEltPerPack + pe;
        // if (0 <= e && e < nElts)
        if (unsigned(e) < unsigned(nElts)) stcs(mem, elts + e, elt[pe]);
      }
    }
  }
}

template<typename Elt, template<typename> typename Red,
         typename Acc, typename AccPack, int UnrollData, typename GetSrcFn>
static __device__ __forceinline__ void accumulateLoads(
    Red<Acc> red, bool inPlace, AccPack(&accs)[UnrollData], int stride, bool lastPackEmpty,
    int nSrcs, GetSrcFn getSrc
  ) {
  using EltPack = ncclDecayType_t<decltype(*getSrc(0))>;
  static_assert(sizeof(Acc)/sizeof(Elt) == sizeof(AccPack)/sizeof(EltPack), "Required");
  constexpr int UnrollSrcs = 4*16 < sizeof(AccPack)*UnrollData ? 2 :
                             1*16 <= sizeof(AccPack)*UnrollData ? 4 :
                             8;
  int srcIx = 0;
  bool accValid = inPlace;
  #pragma unroll 1
  while (srcIx + UnrollSrcs <= nSrcs) {
    EltPack tmp[UnrollSrcs][UnrollData] = {};
    #pragma unroll UnrollSrcs
    for (int su=0; su < UnrollSrcs; su++) {
      EltPack* srcPtr = getSrc(srcIx + su);
      #pragma unroll UnrollData
      for (int du=0; du < UnrollData; du++) {
        if (du != UnrollData-1 || !lastPackEmpty) {
          tmp[su][du] = ldcs(GMemTag(), srcPtr + du*stride);
        }
      }
    }
    #pragma unroll UnrollSrcs
    for (int su=0; su < UnrollSrcs; su++) {
      #pragma unroll UnrollData
      for (int du=0; du < UnrollData; du++) {
        if (du != UnrollData-1 || !lastPackEmpty) {
          AccPack a = fromPack<AccPack>(applyCast<Elt, Acc>(tmp[su][du]));
          accs[du] = su == 0 && !accValid ? a : applyReduce(red, accs[du], a);
        }
      }
    }
    accValid = true;
    srcIx += UnrollSrcs;
  }

  if (srcIx < nSrcs) {
    EltPack tmp[UnrollSrcs-1 ? UnrollSrcs-1 : 1][UnrollData] = {};
    #pragma unroll (UnrollSrcs-1 ? UnrollSrcs-1 : 1)
    for (int su=0; su < UnrollSrcs-1; su++) {
      EltPack* srcPtr = getSrc(srcIx + su);
      if (!(su != 0 && nSrcs <= srcIx + su)) {
        #pragma unroll UnrollData
        for (int du=0; du < UnrollData; du++) {
          if (du != UnrollData-1 || !lastPackEmpty) {
            tmp[su][du] = ldcs(GMemTag(), srcPtr + du*stride);
          }
        }
      }
    }
    #pragma unroll (UnrollSrcs-1 ? UnrollSrcs-1 : 1)
    for (int su=0; su < UnrollSrcs-1; su++) {
      if (su != 0 && nSrcs <= srcIx + su) break;
      #pragma unroll UnrollData
      for (int du=0; du < UnrollData; du++) {
        if (du != UnrollData-1 || !lastPackEmpty) {
          AccPack a = fromPack<AccPack>(applyCast<Elt, Acc>(tmp[su][du]));
          accs[du] = su == 0 && !accValid ? a : applyReduce(red, accs[du], a);
        }
      }
      accValid = true;
    }
  }
}

static __device__ inline unsigned getWorstPackCount(unsigned nElts, unsigned nEltPerPack) {
  // Count packs rounding up for worst possible alignment.
  //   nElts=1 --> nPacks=1
  //   nElts=2 --> nPacks=2
  //   nElts=nEltPerPack+1 --> nPacks=2
  //   nElts=nEltPerPack+2 --> nPacks=3
  return (nElts + 2*(nEltPerPack-1))/nEltPerPack;
}

template<typename Coop, typename DstSpace, typename GetDst,
         template<typename> typename Red, typename Acc,
         typename GetSrc, typename GetSrcPtrMasked>
 __device__ void reduceBatch(
    Coop coop, bool inPlace, int nBatch, int nElts,
    DstSpace dstMem, unsigned dstAlignMin, /*(int i)->DstT* */GetDst getDst,
    Red<Acc> red, int nSrcs, /*(int i, int srcIx)->SrcT* */GetSrc getSrc,
    // srcPtrCommonMask: All srcs must have matching values of srcPtr & srcPtrCommonMask
    unsigned srcPtrCommonMask,
    // getSrcPtrMasked: The common srcPtr & srcPtrCommonMask value shared by all srcs in group
    /*(int i)->unsigned*/GetSrcPtrMasked getSrcPtrMasked
  ) {
  using DstT = ncclDecayType_t<decltype(*getDst(0))>;
  using SrcT = ncclDecayType_t<decltype(*getSrc(0, 0))>;
  static_assert(sizeof(SrcT) <= sizeof(DstT), "Required");

  int tn = coop.size();
  int t = coop.thread_rank();

  using SrcPack = BytePack<16>;
  constexpr unsigned nEltPerPack = 16/sizeof(SrcT);
  using DstPack = BytePack<sizeof(SrcPack)*sizeof(DstT)/sizeof(SrcT)>;
  using AccPack = BytePack<sizeof(SrcPack)*sizeof(Acc)/sizeof(SrcT)>;

  // Handle source elements as packs if:
  // 1. All sources share a sufficient common alignment to support pack access.
  // 2. There is enough for every thread to have one.
  if (sizeof(SrcT) == sizeof(SrcPack) ||
      (/*1*/sizeof(SrcPack)-1 <= srcPtrCommonMask &&
       /*2*/tn*(int)sizeof(SrcPack) <= nBatch*nElts*(int)sizeof(SrcT))
  ) {
    int nPacks = getWorstPackCount(nElts, nEltPerPack);
    constexpr int UnrollData = 4;
    constexpr int nPackPerBlob = UnrollData*32; // A blob is a whole warp's worth of unrolled packs
    int nBlobs = unsigned(nPacks)/nPackPerBlob;
    #pragma unroll 1
    for (int row = unsigned(t)/32; row < nBatch*nBlobs; row += unsigned(tn)/32) {
      int batchIx = unsigned(row)/unsigned(nBlobs);
      int blobIx = unsigned(row)%unsigned(nBlobs);
      int padElts = (getSrcPtrMasked(batchIx) % sizeof(SrcPack))/sizeof(SrcT);
      DstT* dstPtr = getDst(batchIx);

      int lane = unsigned(t)%32;
      int packIx = blobIx*nPackPerBlob + lane;
      bool lastPackEmpty = padElts + nElts <= (packIx + (UnrollData-1)*32)*nEltPerPack;
      AccPack acc[UnrollData] = {};
      if (inPlace) {
        DstPack dstVal[UnrollData];
        loadPacks(dstVal, dstMem, dstAlignMin, dstPtr, nElts, padElts, packIx, 32, lastPackEmpty);
        #pragma unroll
        for (int u=0; u < UnrollData; u++) acc[u] = applyCast<DstT, Acc>(dstVal[u]);
      }
      accumulateLoads<SrcT>(
        red, inPlace, acc, /*stride=*/32, lastPackEmpty, nSrcs,
        [&]__device__(int srcIx)->SrcPack* {
          return (SrcPack*)(getSrc(batchIx, srcIx) - padElts) + packIx;
        }
      );
      DstPack dstVal[UnrollData];
      #pragma unroll
      for (int u=0; u < UnrollData; u++) dstVal[u] = applyCast<Acc, DstT>(acc[u]);
      storePacks(dstVal, dstMem, dstAlignMin, dstPtr, nElts, padElts, packIx, 32, lastPackEmpty);
    }
    int nRemPacks = nPacks - nBlobs*nPackPerBlob;
    #pragma unroll 1
    for (int row = t; row < nBatch*nRemPacks; row += tn) {
      int batchIx = unsigned(row)/unsigned(nRemPacks);
      int packIx = nBlobs*nPackPerBlob + unsigned(row)%unsigned(nRemPacks);
      int padElts = (getSrcPtrMasked(batchIx) % sizeof(SrcPack))/sizeof(SrcT);
      DstT* dstPtr = getDst(batchIx);

      bool packEmpty = padElts + nElts <= packIx*nEltPerPack;
      packEmpty = __builtin_expect(packEmpty, false);
      if (!packEmpty) {
        AccPack acc[1];
        if (inPlace) {
          DstPack dstVal[1];
          loadPacks(dstVal, dstMem, dstAlignMin, dstPtr, nElts, padElts, packIx, tn, false);
          acc[0] = applyCast<DstT, Acc>(dstVal[0]);
        }
        accumulateLoads<SrcT>(
          red, inPlace, acc, /*stride(no data unroll)=*/0, /*lastPackEmpty=*/false, nSrcs,
          [&]__device__(int srcIx)->SrcPack* {
            return (SrcPack*)(getSrc(batchIx, srcIx) - padElts) + packIx;
          }
        );
        DstPack dstVal[1] = {applyCast<Acc, DstT>(acc[0])};
        storePacks(dstVal, dstMem, dstAlignMin, dstPtr, nElts, padElts, packIx, tn, false);
      }
    }
  } else {
    #pragma unroll 1
    for (int row = t; row < nBatch*nElts; row += tn) {
      int batchIx = unsigned(row)/unsigned(nElts);
      int eltIx = unsigned(row)%unsigned(nElts);
      DstT* dstPtr = getDst(batchIx);
      Acc acc[1];
      if (inPlace) acc[0] = (Acc)ldcs(dstMem, dstPtr + eltIx);
      accumulateLoads<SrcT>(
        red, inPlace, acc, /*stride(no data unroll)=*/0, /*lastPackEmpty=*/false, nSrcs,
        [&]__device__(int srcIx)->SrcT* {
          return getSrc(batchIx, srcIx) + eltIx;
        }
      );
      stcs(dstMem, dstPtr + eltIx, (DstT)acc[0]);
    }
  }
}

template<typename Coop, template<typename> typename Red, typename Acc,
         typename DstSpace, typename DstT, typename GetSrc>
static __device__ void reduce(
    Coop coop, Red<Acc> red, bool inPlace, int nElts,
    DstSpace dstMem, unsigned dstAlignMin, DstT* dst,
    int nSrcs, unsigned srcPtrCommonMask, unsigned srcPtrMasked,
    /*(int srcIx)->SrcT* */ GetSrc getSrc
  ) {
  reduceBatch(coop, inPlace, /*nBatch=*/1, nElts,
    dstMem, dstAlignMin, [=]__device__(int d)->DstT* { return dst; },
    red, nSrcs, [&]__device__(int d, int s) { return getSrc(s); },
    srcPtrCommonMask,
    /*getSrcPtrMasked=*/[=]__device__(int d)->unsigned { return srcPtrMasked; }
  );
}

template<typename Coop, typename DstSpace, typename GetDst,
         template<typename> typename Red, typename SrcT, typename GetSrc>
 __device__ void reduceMultimemBatch(
    Coop coop, int nBatch, int nElts,
    DstSpace dstMem, unsigned dstAlignMin, /*(int i)->DstT* */ GetDst getDst,
    Red<SrcT> srcRed, /*(int i)->SrcT* */ GetSrc getSrc
  ) {
  using DstT = ncclDecayType_t<decltype(*getDst(0))>;
  using SrcT_1 = ncclDecayType_t<decltype(*getSrc(0))>;
  static_assert(sizeof(SrcT_1) == sizeof(SrcT), "Required");
  static_assert(sizeof(SrcT) <= sizeof(DstT), "Required");

  int tn = coop.size();
  int t = coop.thread_rank();

  using SrcPack = BytePack<LoadMultimem_BigPackSize<Red<SrcT>>::BigPackSize>;
  constexpr int nEltPerPack = sizeof(SrcPack)/sizeof(SrcT);
  using DstPack = BytePack<sizeof(SrcPack)*sizeof(DstT)/sizeof(SrcT)>;

  // Handle source elements as packs if there is enough for every thread to have one.
  if (sizeof(SrcT) == sizeof(SrcPack) || tn*(int)sizeof(SrcPack) <= nBatch*nElts*(int)sizeof(SrcT)) {
    int nPacks = getWorstPackCount(nElts, nEltPerPack);
    constexpr int UnrollData = 4;
    constexpr int nPackPerBlob = UnrollData*32; // A blob is a whole warp's worth of unrolled packs
    int nBlobs = unsigned(nPacks)/nPackPerBlob;
    #pragma unroll 1
    for (int row = unsigned(t)/32; row < nBatch*nBlobs; row += unsigned(tn)/32) {
      int batchIx = unsigned(row)/unsigned(nBlobs);
      int blobIx = unsigned(row)%unsigned(nBlobs);
      DstT* dstPtr = getDst(batchIx);
      SrcT* srcPtr = getSrc(batchIx);
      int padElts = (reinterpret_cast<uintptr_t>(srcPtr) % sizeof(SrcPack))/sizeof(SrcT);

      int lane = unsigned(t)%32;
      int packIx = blobIx*nPackPerBlob + lane;
      bool lastPackEmpty = padElts + nElts <= (packIx + (UnrollData-1)*32)*nEltPerPack;
      SrcPack* srcPackPtr = (SrcPack*)(srcPtr - padElts) + packIx;
      SrcPack srcVal[UnrollData] = {};
      #pragma unroll
      for (int u=0; u < UnrollData; u++) {
        if (u < UnrollData-1 || !lastPackEmpty) {
          srcVal[u] = applyLoadMultimem<Red<SrcT>, sizeof(SrcPack)>(
            srcRed, reinterpret_cast<uintptr_t>(srcPackPtr + u*32)
          );
        }
      }
      DstPack dstVal[UnrollData];
      #pragma unroll
      for (int u=0; u < UnrollData; u++) dstVal[u] = applyCast<SrcT, DstT>(srcVal[u]);
      storePacks(dstVal, dstMem, dstAlignMin, dstPtr, nElts, padElts, packIx, 32, lastPackEmpty);
    }
    int nRemPacks = nPacks - nBlobs*nPackPerBlob;
    #pragma unroll 1
    for (int row = t; row < nBatch*nRemPacks; row += tn) {
      int batchIx = unsigned(row)/unsigned(nRemPacks);
      int packIx = nBlobs*nPackPerBlob + unsigned(row)%unsigned(nRemPacks);
      DstT* dstPtr = getDst(batchIx);
      SrcT* srcPtr = getSrc(batchIx);
      int padElts = (reinterpret_cast<uintptr_t>(srcPtr) % sizeof(SrcPack))/sizeof(SrcT);

      bool packEmpty = padElts + nElts <= packIx*nEltPerPack;
      packEmpty = __builtin_expect(packEmpty, false);
      if (!packEmpty) {
        SrcPack* srcPackPtr = (SrcPack*)(srcPtr - padElts) + packIx;
        SrcPack srcVal = applyLoadMultimem<Red<SrcT>, sizeof(SrcPack)>(
          srcRed, reinterpret_cast<uintptr_t>(srcPackPtr)
        );
        DstPack dstVal[1] = {applyCast<SrcT, DstT>(srcVal)};
        storePacks(dstVal, dstMem, dstAlignMin, dstPtr, nElts, padElts, packIx, tn, false);
      }
    }
  } else {
    #pragma unroll 1
    for (int row = t; row < nBatch*nElts; ) {
      int batchIx = unsigned(row)/unsigned(nElts);
      int eltIx = unsigned(row)%unsigned(nElts);
      DstT* dstPtr = getDst(batchIx) + eltIx;
      SrcT* srcPtr = getSrc(batchIx) + eltIx;
      constexpr int UnrollData = 4;
      // Test if the whole warp can be unrolled within the same run of elts
      // by testing if both lane 0 and 31 are in our run.
      int lane = unsigned(t)%32;
      if ((0 <= eltIx-lane) && (eltIx-lane + 31 + 1 + (UnrollData-1)*tn <= nElts)) {
        SrcT srcVal[UnrollData] = {};
        #pragma unroll
        for (int u=0; u < UnrollData; u++) {
          srcVal[u] = fromPack<SrcT>(applyLoadMultimem<Red<SrcT>, sizeof(SrcT)>(
            srcRed, reinterpret_cast<uintptr_t>(srcPtr + u*tn)
          ));
        }
        #pragma unroll
        for (int u=0; u < UnrollData; u++) {
          stcs(dstMem, dstPtr + u*tn, (DstT)srcVal[u]);
        }
        row += UnrollData*tn;
      } else {
        SrcT srcVal = fromPack<SrcT>(applyLoadMultimem<Red<SrcT>, sizeof(SrcT)>(
          srcRed, reinterpret_cast<uintptr_t>(srcPtr)
        ));
        stcs(dstMem, dstPtr, (DstT)srcVal);
        row += tn;
      }
    }
  }
}

// reduceLsa helpers sum symmetric elements from whole LSA team.
template<typename Coop, typename DstSpace, typename GetDst,
         template<typename> typename Red, typename Acc, typename SrcT, typename GetSrcOffset>
static __device__ void reduceLsaBatch(
    Coop coop, int nBatch, int nElts,
    DstSpace dstMem, unsigned dstAlignMin, /*(int i)->DstT* */GetDst getDst,
    Red<Acc> srcRedUc, Red<SrcT> srcRedMc,
    ncclSymPtr<SrcT> srcBase, /*(int i)->size_t*/GetSrcOffset getSrcOffset,
    ncclDevComm const& comm, BoolTag</*multimem=*/true> multimemTrue
  ) {
  SrcT* srcPtr = srcBase.lsaMultimemPtr(comm);
  reduceMultimemBatch(coop, nBatch, nElts, dstMem, dstAlignMin, getDst, srcRedMc,
    /*getSrc=*/[&]__device__(int i) { return srcPtr + getSrcOffset(i); }
  );
}
template<typename Coop, typename DstSpace, typename GetDst,
         template<typename> typename Red, typename Acc, typename SrcT, typename GetSrcOffset>
static __device__ void reduceLsaBatch(
    Coop coop, int nBatch, int nElts,
    DstSpace dstMem, unsigned dstAlignMin, /*(int i)->DstT* */GetDst getDst,
    Red<Acc> srcRedUc, Red<SrcT> srcRedMc,
    ncclSymPtr<SrcT> srcBase, /*(int i)->size_t*/GetSrcOffset getSrcOffset,
    ncclDevComm const& comm, BoolTag</*multimem=*/false> multimemFalse
  ) {
  ncclTeam lsa = ncclTeamLsa(comm);
  ncclLsaPointerGetter<SrcT> getLsaPtr{srcBase};
  reduceBatch(coop, /*inPlace=*/false, nBatch, nElts,
    dstMem, dstAlignMin, getDst, srcRedUc, lsa.nRanks,
    /*getSrcOffset=*/[&]__device__(int i, int s)->SrcT* {
      int r = lsa.rank + s;
      if (lsa.nRanks <= r) r -= lsa.nRanks;
      // This could be `srcBase.lsaPtr(r) + getSrcOffset(i)` but by using the
      // `getLsaPtr` functor we have hoisted the window meta data loads outside
      // of the reduction loop this lambda is embedded in. The compiler isn't
      // being smart about the `__ldg()'s` in `.lsaPtr()` despite it having all
      // the static knowledge necessary to do so.
      return getLsaPtr(r) + getSrcOffset(i);
    },
    /*srcPtrCommonMask=*/-1u, // LSA is 4GB common
    /*getSrcPtrMasked=*/[&]__device__(int i)->unsigned {
      __builtin_assume(srcBase.offset % alignof(SrcT) == 0);
      return unsigned(srcBase.offset + getSrcOffset(i)*sizeof(SrcT));
    }
  );
}

template<typename Coop,
         typename DstSpace, typename DstT,
         template<typename> typename Red, typename Acc, typename SrcT,
         bool multimem>
static __device__ void reduceLsa(
    Coop coop, int nElts,
    DstSpace dstMem, unsigned dstAlignMin, DstT* dstPtr,
    Red<Acc> srcRedUc, Red<SrcT> srcRedMc, ncclSymPtr<SrcT> srcPtr,
    ncclDevComm const& comm, BoolTag<multimem> multimemTag
  ) {
  reduceLsaBatch(coop, /*nBatch=*/1, nElts,
    dstMem, dstAlignMin, [=]__device__(int)->DstT* { return dstPtr; },
    srcRedUc, srcRedMc, srcPtr,
    /*getSrcOffset=*/[=]__device__(int)->int { return 0; },
    comm, multimemTag);
}

template<typename Coop, typename DstT, typename SrcT>
static __device__ void copy(
    Coop coop, int nElts, GMemTag dstMem, DstT* dst, SMemTag srcMem, SrcT* src
  ) {
  static_assert(sizeof(DstT) <= sizeof(SrcT), "Required");
  __builtin_assume(__isShared(src));
  int tn = coop.size();
  int t = coop.thread_rank();

  constexpr int nEltPerPack = 16/sizeof(DstT);
  using DstPack = BytePack<16>;
  using SrcPack = BytePack<16*sizeof(SrcT)/sizeof(DstT)>;
  int nPreElts = (16 - reinterpret_cast<uintptr_t>(dst))%16/sizeof(DstT);
  int nSufElts = reinterpret_cast<uintptr_t>(dst + nElts)%16/sizeof(DstT);
  int nPacks = (nElts-nPreElts-nSufElts)/nEltPerPack;
  unsigned srcAlign16 = reinterpret_cast<uintptr_t>(src + nPreElts)%16;

  if (srcAlign16 == 0) {
    // TODO: When SrcT==DstT we can replace loop with single `cp.async.bulk`
    #pragma unroll 1
    for (int p = t; p < nPacks; p += tn) {
      SrcPack srcPack = ((SrcPack*)(src + nPreElts))[p];
      stcs(GMemTag(), (DstPack*)(dst + nPreElts) + p,  applyCast<SrcT, DstT>(srcPack));
    }
  } else {
    unsigned srcAlign4 = srcAlign16%4;
    uint32_t* srcWords = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(src + nPreElts) & -uintptr_t(4));
    #pragma unroll 1
    for (int p = t; p < nPacks; p += tn) {
      constexpr int nWordPerPack = sizeof(SrcPack)/4;
      union { SrcPack srcPack; uint32_t w[nWordPerPack + 1]; };
      int i;
      #pragma unroll
      for (i = 0; i < nWordPerPack; i++) w[i] = (srcWords + p*nWordPerPack)[i];
      if (srcAlign4 != 0) {
        w[i] = srcWords[i];
        #pragma unroll
        for (i = 0; i < nWordPerPack; i++) w[i] = __funnelshift_r(w[i], w[i+1], 8*srcAlign4);
      }
      stcs(GMemTag(), (DstPack*)(dst + nPreElts) + p,  applyCast<SrcT, DstT>(srcPack));
    }
  }

  #pragma unroll 1
  for (int i = t; i < nPreElts + nSufElts; i += tn) {
    int e = i < nPreElts ? i : nElts - nSufElts + (i - nPreElts);
    stcs(GMemTag(), dst + e, (DstT)src[e]);
  }
}
