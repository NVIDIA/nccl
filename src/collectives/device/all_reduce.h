/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclRing *ring = &ncclShmem.channel.ring;
    int ringIx = ring->index;
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t loopSize = nChannels*nranks*chunkSize;
    const ssize_t size = args->count;

    int minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL)
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    if (Proto::Id == NCCL_PROTO_LL128) {
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
    }

    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->redOpArg);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*nranks));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize);
      realChunkSize = int(realChunkSize);

      auto calcOffset = [&]__device__(int chunk)->ssize_t {
        if (Proto::Id == NCCL_PROTO_SIMPLE)
          return gridOffset + bid*nranks*realChunkSize + chunk*realChunkSize;
        else
          return gridOffset + (chunk*nChannels + bid)*realChunkSize;
      };
      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = modRanks(ringIx + nranks-1);
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);
      prims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = modRanks(ringIx + nranks-j);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);
        prims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringIx + 0;
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);
      prims.directRecvReduceCopySend(offset, offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = modRanks(ringIx + nranks-j);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);
        prims.directRecvCopySend(offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = modRanks(ringIx + 1);
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);
      prims.directRecv(offset, nelem);
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeUpDown(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclTree *tree = &ncclShmem.channel.tree;
    ssize_t chunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? args->lastChunkSize
                   /* LL & LL128 */  : Proto::calcBytePerStep()/sizeof(T));
    const ssize_t minChunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? (nthreads-2*WARP_SIZE)*8*(sizeof(uint64_t)/sizeof(T))
                   /* LL & LL128 */  : nthreads*(Proto::calcBytePerGrain()/sizeof(T)));
    const ssize_t loopSize = int(nChannels*chunkSize);
    const ssize_t size = args->count;

    if (loopSize > size)
      chunkSize = divUp((int)size, int(nChannels*minChunkSize))*int(minChunkSize);

    { // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, /*Direct=*/0, Proto, 0> prims
        (tid, nthreads, tree->down, &tree->up, args->sendbuff, args->recvbuff, args->redOpArg);
      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceSend(offset, nelem);
        }
      }
    }

    { // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, /*Direct=*/1, Proto, 0> prims
        (tid, nthreads, &tree->up, tree->down, args->sendbuff, args->recvbuff, args->redOpArg);
      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directSendFromOutput(offset, offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecvCopySend(offset, offset, nelem);
        }
      }
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeSplit(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclTree *tree = &ncclShmem.channel.tree;
    ssize_t chunkSize = int(
      Proto::Id != NCCL_PROTO_LL ? args->lastChunkSize
                                 : Proto::calcBytePerStep()/sizeof(T));
    const ssize_t minChunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? (nthreads - 2*WARP_SIZE)*8*(sizeof(uint64_t)/sizeof(T)) :
      Proto::Id == NCCL_PROTO_LL     ? nthreads*(Proto::calcBytePerGrain()/sizeof(T))
                   /* LL128 */       : nthreads*(Proto::calcBytePerGrain()/sizeof(T))/8);
    const ssize_t loopSize = int(nChannels*chunkSize);
    const ssize_t size = args->count;

    int nthreadsSplit;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      nthreadsSplit = nthreads/2;
      if (nthreadsSplit >= 256) nthreadsSplit += 64;
    } else { // LL & LL128
      // Receiving from up to 3 sources is more compute intensive than sending
      // to 3 dests. Use 70% for reduce and 30% for bcast.
      nthreadsSplit = (nthreads*7/(10*WARP_SIZE))*WARP_SIZE;
    }

    if (loopSize > size)
      chunkSize = divUp((int)size, nChannels*int(minChunkSize))*int(minChunkSize);

    if (tree->up == -1) {
      // Reduce and broadcast. Max number of recv is 3, max number of send is 3
      Primitives<T, RedOp, FanSymmetric<NCCL_MAX_DEV_ARITY>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreads, tree->down, tree->down, args->sendbuff, args->recvbuff, args->redOpArg);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*int(chunkSize);
        int nelem = min(chunkSize, size-offset);
        prims.directRecvReduceCopySend(offset, offset, offset, nelem, /*doPost=*/true);
      }
    }
    else if (tid < nthreadsSplit) {
      /* Reduce up. Max number of recv is 3, max number of send is 1 (binary tree + local).
       * Why Direct=1????
       * Answer: Because despite not performing any direct operations, the ctor
       * must assume Direct so that it can exchange direct pointers with remote ctors
       * that are Direct, otherwise it hangs. A cleaner solution would be to seperate
       * into DirectRecv and DirectSend capabilities, this ctor would have both=0,
       * but the ctor above for tree roots would be DirectRecv=0 DirectSend=1.
       */
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreadsSplit, tree->down, &tree->up, args->sendbuff, args->recvbuff, args->redOpArg, 0*Proto::MaxGroupWidth);
      if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceSend(offset, nelem);
        }
      }
    }
    else {
      // Broadcast down. Max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, /*Direct=*/1, Proto, 0>
        prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, args->sendbuff, args->recvbuff, args->redOpArg, 1*Proto::MaxGroupWidth);
      if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecvCopySend(offset, offset, nelem);
        }
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    #if CUDART_VERSION >= 11020 && CUDART_VERSION < 11040 && __CUDA_ARCH__ >= 800
      runTreeUpDown<T, RedOp, ProtoSimple<1, 1>>(args);
    #else
      runTreeSplit<T, RedOp, ProtoSimple<1, 1>>(args);
    #endif
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    static constexpr int COLLNET_COPY_THREADS = 96;
    const int tid = threadIdx.x;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    const ssize_t chunkSize = int(args->lastChunkSize);
    const ssize_t size = args->count;
    const ssize_t loopSize = nChannels*direct->nHeads*chunkSize;

    const int hasUp = (direct->up[0] >= 0) ? 1 : 0;
    const int hasDn = (direct->down[0] >= 0) ? 1 : 0;
    const int nThreadsScatter = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 3*COLLNET_COPY_THREADS : 0);
    const int nThreadsGather  =             ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 2*COLLNET_COPY_THREADS : 0);
    const int nThreadsBcast   = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 0 : 2*COLLNET_COPY_THREADS);
    const int nThreadsReduce = args->nWarps*WARP_SIZE - nThreadsScatter - nThreadsGather - nThreadsBcast;
    const int tidStartBcast = nThreadsGather;
    const int tidStartScatter = tidStartBcast + nThreadsBcast;
    const int tidStartReduce = tidStartScatter + nThreadsScatter;

    using Proto = ProtoSimple<1, 1>;

    if (tid >= tidStartScatter && tid < tidStartReduce && hasUp) {
      // Scatter
      int group = (2*Proto::MaxGroupWidth) | (1<<16);
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/1, Proto, 0>
        prims(tid-tidStartScatter, nThreadsScatter, NULL, direct->up, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*direct->nHeads*chunkSize;
        int nelem = min(direct->nHeads*chunkSize, size-offset);
        if (args->regUsed) {
          prims.directScatter(offset, nelem, chunkSize, chunkSize, direct->headRank, direct->shift);
        } else {
          prims.scatter(offset, nelem, chunkSize, chunkSize, direct->headRank, direct->shift);
        }
      }
    } else if (tid >= tidStartReduce && direct->out != -1) {
      int group = (3*Proto::MaxGroupWidth) | (1<<16);
      if (hasDn) {
        // Reduce, send to network
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/1, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, direct->down, &direct->out, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*direct->nHeads+direct->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (args->regUsed) {
            prims.directRecvReduceSend(offset, offset, nelem);
          } else {
            prims.recvReduceSend(offset, nelem);
          }
        }
      } else {
        // Directly send to network
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, nullptr, &direct->out, args->sendbuff, args->recvbuff, args->redOpArg, group);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*direct->nHeads+direct->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
    } else if (tid < tidStartBcast && hasUp) {
      // Gather
      int group = (0*Proto::MaxGroupWidth) | (0<<16);
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/1, Proto, 0>
        prims(tid, nThreadsGather, direct->up, NULL, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*direct->nHeads*chunkSize;
        int nelem = min(direct->nHeads*chunkSize, size-offset);
        prims.directGather(offset, nelem, chunkSize, chunkSize, direct->headRank, direct->shift);
      }
    } else if (tid >= tidStartBcast && tid < tidStartScatter && direct->out != -1) {
      int group = (1*Proto::MaxGroupWidth) | (0<<16);
      if (hasDn) {
        // Recv from network, broadcast
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/1, Proto, 0>
          prims(tid-tidStartBcast, nThreadsBcast, &direct->out, direct->down, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*direct->nHeads+direct->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recvCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
        }
      } else {
        // Recv from network (no post thread needed)
        Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartBcast, nThreadsBcast, &direct->out, nullptr, args->sendbuff, args->recvbuff, args->redOpArg, group);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*direct->nHeads+direct->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recv(offset, nelem, /*postOp=*/true);
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
  #if NCCL_NVLS_ENABLED
    const int tid = threadIdx.x;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const ssize_t chunkSize = int(args->lastChunkSize);
    const ssize_t size = args->count;
    const ssize_t loopSize = nChannels*nvls->nHeads*chunkSize;
    const int nranks = ncclShmem.comm.nRanks;
    const int reduceWarps = nranks <= 6 ? 6 : 4;
    const int copyWarps = ((NCCL_MAX_NTHREADS/WARP_SIZE) - reduceWarps)/2;

    const int nThreadsScatter = copyWarps*WARP_SIZE;
    const int nThreadsGather  = (copyWarps-1)*WARP_SIZE;
    const int nThreadsReduce = (reduceWarps+1)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;

    using Proto = ProtoSimple<1, 1, COLL_UNROLL, /*NVLS=*/true>;

    if (tid < tidEndScatter) {
      // Scatter
      int group = (0*Proto::MaxGroupWidth) | (0<<16);
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, nThreadsScatter, NULL, nvls->up, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*nvls->nHeads*chunkSize;
        int nelem = min(nvls->nHeads*chunkSize, size-offset);
        prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
      }
    } else if (tid < tidEndGather) {
      // Gather
      int group = (2*Proto::MaxGroupWidth) | (0<<16);
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid-tidEndScatter, nThreadsGather, nvls->up, NULL, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*nvls->nHeads*chunkSize;
        int nelem = min(nvls->nHeads*chunkSize, size-offset);
        prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
      }
    } else if (tid < tidEndReduce) {
      int group = (3*Proto::MaxGroupWidth) | (1<<16);
      // Reduce, broadcast through NVLS
      Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/0, Proto, 0>
        prims(tid-tidEndGather, nThreadsReduce, &nvls->down, &nvls->down, args->sendbuff, args->recvbuff, args->redOpArg, group, args);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + (bid*nvls->nHeads+nvls->headRank)*chunkSize;
        int nelem = min(chunkSize, size-offset);
        prims.recvSend(nelem);
      }
    }
  #endif // NCCL_NVLS_ENABLED
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclTree *tree = &ncclShmem.channel.collnetChain;
    ssize_t chunkSize = int(args->lastChunkSize);
    const ssize_t loopSize = int(nChannels*chunkSize);
    const ssize_t size = args->count;

    int nthreadsSplit = nthreads/2;
    if (nthreadsSplit >= 256) nthreadsSplit += 64;

    int group, send, recv, groupTid, groupNthreads;
    using Proto = ProtoSimple<1, 1>;
    if (tid < nthreadsSplit) {
      group = (0*Proto::MaxGroupWidth) | (1<<16);
      recv = tree->down[0];
      send = tree->up;
      groupTid = tid;
      groupNthreads = nthreadsSplit;
    } else {
      group = (1*Proto::MaxGroupWidth);
      recv = tree->up;
      send = tree->down[0];
      groupTid = tid - nthreadsSplit;
      groupNthreads = nthreads-nthreadsSplit;
    }

    Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
      prims(groupTid, groupNthreads, &recv, &send, args->sendbuff, args->recvbuff, args->redOpArg, group);

    if (tid < nthreadsSplit) {
      if (recv == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      } else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceSend(offset, nelem);
        }
      }
    }
    else {
      if (send == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecv(offset, nelem);
        }
      } else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecvCopySend(offset, offset, nelem);
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runTreeSplit<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL128>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runTreeSplit<T, RedOp, ProtoLL128>(args);
  }
};
