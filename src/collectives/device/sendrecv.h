/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  static_assert(sizeof(T)==1, "Required sizeof(T)==1");

  template<typename Proto>
  __device__ void runSend(const int tid, const int nthreads, struct ncclDevWorkP2p* work) {
    int peer = work->fifo.peer;
    void* buf = work->fifo.localBuf;
    int group = work->group;
    Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/1, Proto, /*P2p=*/1> prims
      (tid, nthreads, nullptr, &peer, buf, nullptr, /*redOpArg(ignored)=*/0, group, 1, 1);
    int32_t chunkBytes = work->fifo.chunkBytes;
    ssize_t ahead = work->bytes;
    ssize_t behind = 0;
    do {
      int32_t nelem = min(chunkBytes, ahead);
      prims.directSend(behind, behind, nelem);
      behind += nelem;
      ahead -= nelem;
    } while (ahead != 0);
  }

  template<typename Proto>
  __device__ void runRecv(const int tid, const int nthreads, struct ncclDevWorkP2p* work) {
    int peer = work->fifo.peer;
    void* buf = work->fifo.localBuf;
    int group = work->group;
    Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/1, Proto, /*P2p=*/1> prims
      (tid, nthreads, &peer, nullptr, nullptr, buf, /*redOpArg(ignored)=*/0, group, 1, 1);
    int32_t chunkBytes = work->fifo.chunkBytes;
    ssize_t ahead = work->bytes;
    ssize_t behind = 0;
    do {
      int32_t nelem = min(chunkBytes, ahead);
      prims.directRecv(behind, nelem);
      behind += nelem;
      ahead -= nelem;
    } while (ahead != 0);
  }

  __device__ __forceinline__ void run() {
    int tid = threadIdx.x;
    int wid = tid/WARP_SIZE;
    int lane = tid%WARP_SIZE;
    struct ncclDevWorkBatchHeader* batch = &ncclShmem.workBatch;
    int nWorks = batch->nWorks;
    uint32_t lasts = batch->p2p.lastWarpMask;
    uint32_t lastsBelow = lasts & ((1u<<wid)-1); // Last warps beneath self
    uint32_t lastsAbove = lasts & ~((1u<<wid)-1); // Last warps not beneath self.
    int workIx = __popc(lastsBelow);
    if (workIx < nWorks) {
      int nWarpsBelow = 32 - __clz(lastsBelow);
      int subwid = wid - nWarpsBelow;
      int lastWarp = 31 - __clz(lastsAbove & -lastsAbove); // Index of last warp of our group
      int subwn = lastWarp+1 - nWarpsBelow;
      int subtid = subwid*WARP_SIZE + lane;
      int subtn = subwn*WARP_SIZE;
      struct ncclDevWorkP2p* work = (struct ncclDevWorkP2p*)(batch+1) + workIx;
      switch (work->p2pType) {
      case ncclDevWorkP2pTypeRecv:
        { if (work->protocol == NCCL_PROTO_LL) {
            runRecv<ProtoLL>(subtid, subtn, work);
          } else {
            runRecv<ProtoSimple<1,1>>(subtid, subtn, work);
          }
        } break;
      case ncclDevWorkP2pTypeSend:
        { if (work->protocol == NCCL_PROTO_LL) {
            runSend<ProtoLL>(subtid, subtn, work);
          } else {
            runSend<ProtoSimple<1,1>>(subtid, subtn, work);
          }
        } break;
      default: // ncclDevWorkP2pTypeCopy:
        { copyGlobalGlobal</*DstAligned=*/false, /*SrcAligned=*/false>
            (subwn, subwid, lane,
             cvta_to_global(work->copy.dstBuf), cvta_to_global(work->copy.srcBuf),
             (int64_t)work->bytes, cvta_to_shared(ncclScratchForWarp(wid)));
        } break;
      }
    }
  }
};
