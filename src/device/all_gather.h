/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = (int)args->nWarps * WARP_SIZE;
    ncclRing *ring = &ncclShmem.channel.ring;
    const int *ringRanks = ring->userRanks;
    const int nranks = ncclShmem.comm.nRanks;
    const size_t chunkCount = args->chunkCount;
    const size_t channelCount = args->workCount;
    const size_t gridOffset = args->workOffset;
    const size_t count = args->count;
    size_t offset;
    size_t dataOffset;
    int nelem;
    int rankDest;

    T *inputBuf = (T*)args->sendbuff;
    T *outputBuf = (T*)args->recvbuff;
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, inputBuf, outputBuf, args->redOpArg);

    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      /////////////// begin AllGather steps ///////////////
      nelem = min(chunkCount, channelCount - elemOffset);
      dataOffset = gridOffset + elemOffset;

      // step 0: push data to next GPU
      rankDest = ringRanks[0];
      offset = dataOffset + rankDest * count;

      if (inputBuf + dataOffset == outputBuf + offset) { // In place
        prims.directSend(dataOffset, offset, nelem);
      } else {
        prims.directCopySend(dataOffset, offset, nelem);
      }

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ringRanks[nranks-j];
        offset = dataOffset + rankDest * count;

        prims.directRecvCopySend(offset, nelem);
      }

      // Make final copy from buffer to dest.
      rankDest = ringRanks[1];
      offset = dataOffset + rankDest * count;

      // Final wait/copy.
      prims.directRecv(offset, nelem);
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL128>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const ssize_t count = args->count;
    const ssize_t rank = ncclShmem.comm.rank;
    const size_t chunkCount = args->chunkCount;
    size_t gridOffset = args->workOffset;
    size_t channelCount = args->workCount;
    size_t offset;
    int nelem;

    const int nThreadsBcast = args->regUsed ? (NCCL_MAX_NTHREADS - WARP_SIZE) : 4 * WARP_SIZE;
    const int nThreadsGather = args->regUsed ? WARP_SIZE : NCCL_MAX_NTHREADS - nThreadsBcast;
    const int tidEndGather = nThreadsGather;
    const int tidEndBcast = tidEndGather + nThreadsBcast;

    if (!args->regUsed) {
      if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsGather, nvls->up, NULL, NULL, args->recvbuff,
            args->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.gather(offset, nvls->nHeads * count, nelem, count, -1, 0);
        }
      } else if (tid < tidEndBcast) {
        // Bcast through NVLS
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndGather, nThreadsBcast, NULL, &nvls->down, args->sendbuff, NULL,
            args->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0);
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.send(offset, nelem);
        }
      }
    } else {
      /* direct allgather */
      if (tid < tidEndGather) {
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanSymmetric<NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsGather, nvls->up, nvls->up, NULL, NULL,
            args->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);

        /* used as sync */
        prims.scatter(0, 0, 0, 0, -1, 0);

        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          prims.gather(0, 0, 0, 0, -1, 0);
        }
      } else if (tid < tidEndBcast) {
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsBcast, &nvls->down, &nvls->down, args->sendbuff, NULL,
            args->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0, args);
        /* used as sync */
        prims.recv(0, 0);

        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          ssize_t inpOffset = gridOffset + elemOffset;
          ssize_t outOffset = inpOffset + rank * count;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSend(inpOffset, outOffset, nelem);
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  template<bool BcastSendNotRecv>
  struct Scatterer {
    struct ncclWorkElem* args;
    ssize_t chunkSize;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes
      ) {
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = direct->nHeads;
      int bid = args->bid;
      char* inbuf = (char*)args->sendbuff;
      char* outbuf = (char*)args->recvbuff;
      ssize_t sizePerRank = args->count*sizeof(T);
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank*sizePerRank);

      ssize_t railAllBeg = min(railGridOffset + bid*chunkSize, nNodes*sizePerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*sizePerRank);
      int railAllSize = railAllEnd - railAllBeg;
      if (tid < nDsts) dstSizes[tid] = railAllSize;

      int src = 0;
      int rail;
      if (BcastSendNotRecv) {
        rail = direct->headRank;
      } else {
        rail = direct->headRank+1;
        if (rail == nRails) rail = 0;
      }
      do {
        int node = railAllBeg/sizePerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node*sizePerRank;
          ssize_t railOneEnd = railOneBeg + sizePerRank;
          ssize_t railOneOffset = (railAllBeg+railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg+railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node*nRails + rail];
          ssize_t userOneBeg = rank*sizePerRank + railOneOffset;
          int outIsDst = (inPlace && rank == ncclShmem.comm.rank) ? 0 : 1;
          reduceCopy<ncclCollUnroll(), RedOp, T,
                     /*MultimemSrcs,MinSrcs,MaxSrcs=*/0,1,1,
                     /*MultimemDsts=*/0, 0+MinDsts, 1+MaxDsts,
                     /*PreOpSrcs=*/0>
            (tid, tn, 0, nullptr, false,
             /*nSrcs=*/1, [=]__device__(int s/*==0*/) -> void* {
               return (char*)srcPtrs[src] + railAllOffset;
             },
             /*nDsts=*/outIsDst+nDsts, [=]__device__(int d) -> void* {
               return d < outIsDst ? outbuf + userOneBeg
                                   : (char*)dstPtrs[d-outIsDst] + railAllOffset;
             },
             delta);
          railAllOffset += delta;
          node += 1;
        }
        src += 1;
        rail += 1;
        if (rail == nRails) rail = 0;
      } while (!BcastSendNotRecv && src < nRails-1);
    }
  };

  __device__ __forceinline__ void run(ncclWorkElem *args) {
    int tid = threadIdx.x;
    const int nChannels = args->nChannels;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    int const &nNodes = ncclShmem.comm.nNodes;
    ssize_t chunkSize = int(args->chunkCount);
    ssize_t const &sizePerRank = args->count;

    bool isMultiRail = (direct->nHeads > 1);
    int nWarps1 = 1;
    int nWarps2 = (isMultiRail ? 2 : 1);
    int nWarps3 = (isMultiRail ? 2 : 0);
    float denom = float(args->nWarps)/float(nWarps1+nWarps2+nWarps3);
    nWarps3 = int(denom*nWarps3);
    nWarps2 = int(denom*nWarps2);
    nWarps1 = args->nWarps - (nWarps2+nWarps3);

    using Proto = ProtoSimple<1, 1>;

    int tn = nWarps1*WARP_SIZE;
    if (tid < tn) {
      // Phase 1: send to network
      Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
        prims(tid, tn, nullptr, &direct->out, args->sendbuff, nullptr,
              /*redOpArg=*/0, 0*Proto::MaxGroupWidth, 1, 1);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*sizePerRank; railGridOffset += nChannels*chunkSize) {
        ssize_t railAllBeg = railGridOffset + args->bid*chunkSize;
        ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*sizePerRank);
        ssize_t railOneBeg = ncclShmem.comm.node*sizePerRank;
        ssize_t railOneEnd = railOneBeg + sizePerRank;
        ssize_t beg = max(railAllBeg, railOneBeg);
        ssize_t end = min(railAllEnd, railOneEnd);
        prims.send(beg-railOneBeg, max(ssize_t(0), end-beg));
      }
      return;
    }
    tid -= tn;

    tn = nWarps2*WARP_SIZE;
    if (tid < tn) {
      // Phase 2: Recv network -> deposit output + send to bcast
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, tn, &direct->out, direct->heads+1, nullptr, nullptr,
              /*redOpArg=*/0, 1*Proto::MaxGroupWidth, 0, 0);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*sizePerRank; railGridOffset += nChannels*chunkSize) {
        Scatterer</*BcastSendNotRecv=*/true> scat;
        scat.args = args;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        prims.process</*Recv=*/1, /*Send=*/1>(scat);
      }
      return;
    }
    tid -= tn;

    tn = nWarps3*WARP_SIZE;
    if (tid < tn) {
      // Phase 3: Recv bcast -> deposit output
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid, tn, direct->heads+1, nullptr, nullptr, nullptr,
              /*redOpArg=*/0, 2*Proto::MaxGroupWidth, 0, 0);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*sizePerRank; railGridOffset += nChannels*chunkSize) {
        Scatterer</*BcastSendNotRecv=*/false> scat;
        scat.args = args;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        prims.process</*Recv=*/1, /*Send=*/0>(scat);
      }
      return;
    }
  }
};
