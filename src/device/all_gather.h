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
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    const int *ringRanks = ring->userRanks;
    const int nranks = ncclShmem.comm.nRanks;
    size_t count, partOffset, partCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount, &chunkCount);
    size_t offset;
    size_t dataOffset;
    int nelem;
    int rankDest;

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, 0, 0, work);

    for (size_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkCount) {
      /////////////// begin AllGather steps ///////////////
      nelem = min(chunkCount, partCount - elemOffset);
      dataOffset = partOffset + elemOffset;

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

        prims.directRecvCopyDirectSend(offset, nelem);
      }

      // Make final copy from buffer to dest.
      rankDest = ringRanks[1];
      offset = dataOffset + rankDest * count;

      // Final wait/copy.
      prims.directRecv(offset, offset, nelem);
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<1, 1>;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    size_t count, channelOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &channelOffset, &channelCount, &chunkCount);

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
      (tid, nthreads, NULL, NULL, inputBuf, outputBuf, work->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, nullptr, false, false, 0, primsModePatAg);

    PatAGAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, channelOffset, channelOffset + channelCount, count, chunkCount, rank, nranks);
    int last = 0;
    while (!last) {
      int recvDim, sendDim, recvOffset, sendOffset, recvStepOffset, postRecv, postSend, nelem;
      size_t inpIx, outIx;
      patAlgo.getNextOp(recvDim, sendDim, inpIx, outIx, recvOffset, sendOffset, recvStepOffset, nelem, postRecv, postSend, last);
      prims.patCopy(recvDim, sendDim, inpIx, outIx, recvOffset, sendOffset, recvStepOffset, nelem, postRecv, postSend);
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const ssize_t rank = ncclShmem.comm.rank;
    size_t count, gridOffset, channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), &count, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;

    const int nThreadsBcast = work->regUsed ? (NCCL_MAX_NTHREADS - WARP_SIZE) : 4 * WARP_SIZE;
    const int nThreadsGather = work->regUsed ? WARP_SIZE : NCCL_MAX_NTHREADS - nThreadsBcast;
    const int tidEndGather = nThreadsGather;
    const int tidEndBcast = tidEndGather + nThreadsBcast;

    if (!work->regUsed) {
      if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
            work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.gather(offset, nvls->nHeads * count, nelem, count, -1, 0);
        }
      } else if (tid < tidEndBcast) {
        // Bcast through NVLS
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndGather, nThreadsBcast, NULL, &nvls->down, work->sendbuff, NULL,
            work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0);
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
            work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);

        /* used as sync */
        prims.scatter(0, 0, 0, 0, -1, 0);

        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          prims.gather(0, 0, 0, 0, -1, 0);
        }
      } else if (tid < tidEndBcast) {
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsBcast, &nvls->down, &nvls->down, work->sendbuff, NULL,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0, work);
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
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  template<bool BcastSendNotRecv>
  struct Scatterer {
    struct ncclDevWorkColl* work;
    ssize_t chunkSize;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = direct->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      char* inbuf = (char*)work->sendbuff;
      char* outbuf = (char*)work->recvbuff;
      ssize_t sizePerRank = work->collnet.count*sizeof(T);
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank*sizePerRank);

      ssize_t railAllBeg = min(railGridOffset + part*chunkSize, nNodes*sizePerRank);
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
          if (nSrcs != 0 && outIsDst+nDsts != 0) {
            reduceCopy<ncclCollUnroll(), RedOp, T,
                     /*MultimemSrcs,MinSrcs,MaxSrcs=*/0,1,1,
                     /*MultimemDsts=*/0, 0+MinDsts, 1+MaxDsts,
                     /*PreOpSrcs=*/0>
            (tid, tn, 0, nullptr, false,
             /*nSrcs=*/1, [=]__device__(int s/*==0*/) -> void* {
               return work->regUsed && (recvDirectFlag & NCCL_DIRECT_READ) ? (char*)srcPtrs[src] + userOneBeg : (char*)srcPtrs[src] + railAllOffset;
             },
             /*nDsts=*/outIsDst+nDsts, [=]__device__(int d) -> void* {
               return d < outIsDst ? outbuf + userOneBeg
                                   : work->regUsed && (sendDirectFlag & NCCL_DIRECT_WRITE) ? (char*)dstPtrs[d-outIsDst] + userOneBeg
                                   : (char*)dstPtrs[d-outIsDst] + railAllOffset;
             },
             delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        src += 1;
        rail += 1;
        if (rail == nRails) rail = 0;
      } while (!BcastSendNotRecv && src < nRails-1);
    }
  };

  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    const int part = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    int const &nNodes = ncclShmem.comm.nNodes;
    ssize_t sizePerRank = work->collnet.count*sizeof(T);
    size_t chunkSize = work->collnet.chunkCount;
    bool isMultiRail = (direct->nHeads > 1);
    int nWarps1 = 1;
    int nWarps2 = (isMultiRail ? 2 : 1);
    int nWarps3 = (isMultiRail ? 2 : 0);
    float denom = float(work->nWarps)/float(nWarps1+nWarps2+nWarps3);
    nWarps3 = int(denom*nWarps3);
    nWarps2 = int(denom*nWarps2);
    nWarps1 = work->nWarps - (nWarps2+nWarps3);

    using Proto = ProtoSimple<1, 1>;

    int tn = nWarps1*WARP_SIZE;
    if (tid < tn) {
      if (work->regUsed == NCCL_COLLNET_REG_BUFFER) {
        if (tid == 0) {
          int steps = (int)divUp(nNodes * sizePerRank * sizeof(T), NCCL_MAX_COLLNET_SIZE);
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>::sendPeerNotify(direct->out, 1, steps);
        }
        __syncwarp();
      } else {
        // Phase 1: send to network
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid, tn, nullptr, &direct->out, work->sendbuff, nullptr,
            /*redOpArg=*/0, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * sizePerRank; railGridOffset += nChannels * chunkSize) {
          ssize_t railAllBeg = railGridOffset + part * chunkSize;
          ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * sizePerRank);
          ssize_t railOneBeg = ncclShmem.comm.node * sizePerRank;
          ssize_t railOneEnd = railOneBeg + sizePerRank;
          ssize_t beg = max(railAllBeg, railOneBeg);
          ssize_t end = min(railAllEnd, railOneEnd);
          prims.send(beg - railOneBeg, max(ssize_t(0), end - beg));
        }
      }
      return;
    }
    tid -= tn;

    tn = nWarps2*WARP_SIZE;
    if (tid < tn) {
      if (work->regUsed == NCCL_COLLNET_REG_BUFFER) {
        if (tid == 0) {
          int steps = (int)divUp(nNodes * sizePerRank * sizeof(T), NCCL_MAX_COLLNET_SIZE);
          Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, steps);
        }
        __syncwarp();
      } else {
        // Phase 2: Recv network -> deposit output + send to bcast
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/1, Proto, 0>
          prims(tid, tn, &direct->out, direct->heads + 1, nullptr, work->recvbuff,
            /*redOpArg=*/0, 1 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * sizePerRank; railGridOffset += nChannels * chunkSize) {
          Scatterer</*BcastSendNotRecv=*/true> scat;
          scat.work = work;
          scat.chunkSize = chunkSize;
          scat.railGridOffset = railGridOffset;
          prims.template process</*Recv=*/1, /*Send=*/1>(scat, work->direct, 0);
        }
      }
      return;
    }
    tid -= tn;

    tn = nWarps3*WARP_SIZE;
    if (tid < tn) {
      // Phase 3: Recv bcast -> deposit output
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/1, Proto, 0>
        prims(tid, tn, direct->heads+1, nullptr, nullptr, work->recvbuff,
              /*redOpArg=*/0, 2*Proto::MaxGroupWidth, 0, 0, work);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*sizePerRank; railGridOffset += nChannels*chunkSize) {
        Scatterer</*BcastSendNotRecv=*/false> scat;
        scat.work = work;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        prims.template process</*Recv=*/1, /*Send=*/0>(scat, 0, work->direct);
      }
      return;
    }
  }
};
