/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

template<typename T, typename RedOp>
struct RunWork<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWork *work) {
    int tid = threadIdx.x;
    int group = 0;
    const int rank = ncclShmem.comm.rank;
    const int nRanks = ncclShmem.comm.nRanks;
    using Proto = ProtoSimple<1, 1>;

    for (int s=0; s<NCCL_MAX_WORK_ELEMENTS; s++) {
      ncclWorkElem *args = &work->elems[s];
      int nThreadsSegment = args->p2p.nThreads;
      if (args->active == 0 || nThreadsSegment == 0) break;

      int nThreadsSplit = (nThreadsSegment - (nThreadsSegment > 128 ? WARP_SIZE : 0))/2;
      int groupRecv = group;
      group += Proto::calcGroupWidth(/*send=*/false, nThreadsSplit);
      int groupSend = group;
      group += Proto::calcGroupWidth(/*send=*/true, nThreadsSegment - nThreadsSplit);

      if (tid < nThreadsSegment) {
        // Compute pointers
        T const* sendbuff = (const T*)args->sendbuff;
        T* recvbuff = (T*)args->recvbuff;
        ssize_t const sendCount = args->p2p.sendCount;
        ssize_t const recvCount = args->p2p.recvCount;
        int const delta = args->p2p.delta;

        if (delta == 0) {
          if (sendbuff != recvbuff) {
            ReduceOrCopyMulti<COLL_UNROLL, RedOp, T, 1, 1, 1, 1, 0>(tid, nThreadsSegment, nullptr, false, 1, &sendbuff, 1, &recvbuff, sendCount);
          }
        }
        else {
          if ((tid < nThreadsSplit) && recvCount >= 0) {
            int const peer = (rank - delta + nRanks)%nRanks;
            int const t0 = 0;
            int const nt = nThreadsSplit;
            int const chunkSize = args->p2p.recvChunkSize/sizeof(T);
            Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto> prims
              (tid-t0, nt, &peer, nullptr, nullptr, recvbuff, /*redOpArg(ignored)=*/0, groupRecv);
            ssize_t offset = 0;
            do {
              int nelem = roundUp(chunkSize, nt*(sizeof(uint64_t)/sizeof(T)));
              nelem = min(chunkSize, recvCount-offset);
              prims.directRecv(offset, nelem);
              offset += nelem;
            } while(offset < recvCount);
          }

          if ((tid >= nThreadsSplit) && sendCount >= 0) {
            int const peer = (rank + delta)%nRanks;
            int const t0 = nThreadsSplit;
            int const nt = nThreadsSegment - nThreadsSplit;
            int const chunkSize = args->p2p.sendChunkSize/sizeof(T);
            Primitives<T, RedOp, FanAsymmetric<0, 1>, 1, Proto> prims
              (tid-t0, nt, nullptr, &peer, sendbuff, nullptr, /*redOpArg(ignored)=*/0, groupSend);
            ssize_t offset = 0;
            do {
              int nelem = roundUp(chunkSize, nt*(sizeof(uint64_t)/sizeof(T)));
              nelem = min(chunkSize, sendCount-offset);
              prims.directSend(offset, offset, nelem);
              offset += nelem;
            } while(offset < sendCount);
          }
        }
        break;
      }
      tid -= nThreadsSegment;
    }
  }
};
