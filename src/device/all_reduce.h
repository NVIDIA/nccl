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
    int ringIx = ring->index;
    ssize_t chunkCount = args->chunkCount;
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t loopCount = nranks * chunkCount;
    ssize_t offset;
    ssize_t gridOffset = args->workOffset;
    ssize_t channelCount = args->workCount;
    int nelem;
    int chunk;

    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->redOpArg);

    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
      ssize_t remCount = channelCount - elemOffset;
      ssize_t chunkOffset;

      if (remCount < loopCount) chunkCount = args->lastChunkCount;

      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      // step 0: push data to next GPU
      chunk = modRanks(ringIx + nranks - 1);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j = 2; j < nranks; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = (int)min(chunkCount, remCount - chunkOffset);
        prims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringIx + 0;
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directRecvReduceCopySend(offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j = 1; j < nranks - 1; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = (int)min(chunkCount, remCount - chunkOffset);
        prims.directRecvCopySend(offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = modRanks(ringIx + 1);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directRecv(offset, nelem);
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeUpDown(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = (int)args->nWarps * WARP_SIZE;
    ncclTree *tree = &ncclShmem.channel.tree;
    const size_t channelCount = args->workCount;
    const size_t gridOffset = args->workOffset;
    const size_t chunkCount = args->chunkCount;
    size_t offset;
    int nelem;

    { // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/0, Proto, 0> prims
        (tid, nthreads, tree->down, &tree->up, args->sendbuff, args->recvbuff, args->redOpArg);
      if (tree->up == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.send(offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.recvReduceSend(offset, nelem);
        }
      }
    }

    { // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0> prims
        (tid, nthreads, &tree->up, tree->down, args->sendbuff, args->recvbuff, args->redOpArg);
      if (tree->up == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSendFromOutput(offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvCopySend(offset, nelem);
        }
      }
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeSplit(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = (int)args->nWarps * WARP_SIZE;
    ncclTree *tree = &ncclShmem.channel.tree;
    const size_t chunkCount = args->chunkCount;
    const size_t gridOffset = args->workOffset;
    const size_t channelCount = args->workCount;
    size_t offset;
    int nelem;

    int nthreadsSplit;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      nthreadsSplit = nthreads/2;
      if (nthreadsSplit >= 256) nthreadsSplit += 64;
    } else { // LL & LL128
      // Receiving from up to 3 sources is more compute intensive than sending
      // to 3 dests. Use 70% for reduce and 30% for bcast.
      nthreadsSplit = (nthreads*7/(10*WARP_SIZE))*WARP_SIZE;
    }

    if (tree->up == -1) {
      // Reduce and broadcast. Max number of recv is 2, max number of send is 2
      Primitives<T, RedOp, FanSymmetric<NCCL_MAX_TREE_ARITY_TOP>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreads, tree->down, tree->down, args->sendbuff, args->recvbuff, args->redOpArg);
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecvReduceCopySend(offset, offset, nelem, /*doPost=*/true);
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
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreadsSplit, tree->down, &tree->up, args->sendbuff, args->recvbuff, args->redOpArg, 0*Proto::MaxGroupWidth);
      if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.send(offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.recvReduceSend(offset, nelem);
        }
      }
    }
    else {
      // Broadcast down. Max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0>
        prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, args->sendbuff, args->recvbuff,
            args->redOpArg, 1*Proto::MaxGroupWidth);
      if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvCopySend(offset, nelem);
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
    const ssize_t chunkSize = args->chunkCount;
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
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/1, Proto, 0>
        prims(tid-tidStartScatter, nThreadsScatter, NULL, direct->up, args->sendbuff, args->recvbuff,
           args->redOpArg, 2*Proto::MaxGroupWidth, 1, 1, args);
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
      if (hasDn) {
        // Reduce, send to network
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/1, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, direct->down, &direct->out, args->sendbuff, args->recvbuff,
             args->redOpArg, 3*Proto::MaxGroupWidth, 1, 1, args);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*direct->nHeads+direct->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (args->regUsed) {
            prims.directRecvReduceSend(offset, nelem);
          } else {
            prims.recvReduceSend(offset, nelem);
          }
        }
      } else {
        // Directly send to network
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, nullptr, &direct->out, args->sendbuff, args->recvbuff,
             args->redOpArg, 3*Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*direct->nHeads+direct->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
    } else if (tid < tidStartBcast && hasUp) {
      // Gather
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/1, Proto, 0>
        prims(tid, nThreadsGather, direct->up, NULL, args->sendbuff, args->recvbuff,
           args->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, args);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*direct->nHeads*chunkSize;
        int nelem = min(direct->nHeads*chunkSize, size-offset);
        prims.directGather(offset, nelem, chunkSize, chunkSize, direct->headRank, direct->shift);
      }
    } else if (tid >= tidStartBcast && tid < tidStartScatter && direct->out != -1) {
      if (hasDn) {
        // Recv from network, broadcast
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/1, Proto, 0>
          prims(tid-tidStartBcast, nThreadsBcast, &direct->out, direct->down, args->sendbuff, args->recvbuff,
             args->redOpArg, 1*Proto::MaxGroupWidth, 0, 0, args);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid*direct->nHeads+direct->headRank)*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recvCopyDirectSend(offset, nelem, /*postOp=*/true);
        }
      } else {
        // Recv from network (no post thread needed)
        Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartBcast, nThreadsBcast, &direct->out, nullptr, args->sendbuff, args->recvbuff,
             args->redOpArg, 1*Proto::MaxGroupWidth, 0, 0);
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
    const int tid = threadIdx.x;
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    ssize_t chunkSize = args->chunkCount;
    const bool hasOut = nvls->out != -1;
    const int nranks = ncclShmem.comm.nRanks;
    const int totalWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
    const int bcastWarps = hasOut ? (args->regUsed ? ((totalWarps - 2) >> 1) - 1 : 2) : 0;
    const int reduceWarps = args->regUsed ? (totalWarps - bcastWarps - 2) : (hasOut ? 3 : nranks <= 6 ? 7 : 5);
    const int scatterWarps = args->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
    const int gatherWarps = args->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;

    const int nThreadsScatter = scatterWarps*WARP_SIZE;
    const int nThreadsGather  = gatherWarps*WARP_SIZE;
    const int nThreadsReduce = reduceWarps*WARP_SIZE;
    const int nThreadsBcast  = (bcastWarps)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;
    const int tidEndBcast = tidEndReduce + nThreadsBcast;

    if (args->oneNode) {
      const ssize_t loopCount = nvls->nHeads * chunkSize;
      const ssize_t channelCount = args->workCount;
      const ssize_t gridOffset = args->workOffset;
      ssize_t offset;
      int nelem;

      if (tid < tidEndScatter) {
        // Scatter
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsScatter, NULL, nvls->up, args->sendbuff, NULL,
            args->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          if (channelCount - elemOffset < loopCount) chunkSize = args->lastChunkCount;
          offset = gridOffset + elemOffset;
          nelem = args->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
          prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, args->recvbuff,
            args->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          if (channelCount - elemOffset < loopCount) chunkSize = args->lastChunkCount;
          offset = gridOffset + elemOffset;
          nelem = args->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
          prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndReduce) {
        // Reduce, broadcast through NVLS
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, &nvls->down, &nvls->down, NULL, NULL,
            args->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, args);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkSize = args->lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkSize;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkSize, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    } else {
      const int bid = args->bid;
      const ssize_t loopSize = args->nChannels * nvls->nHeads * chunkSize;
      const ssize_t size = args->count;

      if (tid < tidEndScatter) {
        // Scatter
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsScatter, NULL, nvls->up, args->sendbuff, NULL,
            args->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * nvls->nHeads * chunkSize;
          int nelem = args->regUsed ? 0 : min(nvls->nHeads * chunkSize, size - offset);
          prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, args->recvbuff,
            args->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * nvls->nHeads * chunkSize;
          int nelem = args->regUsed ? 0 :min(nvls->nHeads * chunkSize, size - offset);
          prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndReduce && nvls->headRank != -1) {
        if (!hasOut) {
          // Reduce, broadcast through NVLS
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndGather, nThreadsReduce, &nvls->down, &nvls->down, NULL, NULL,
              args->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, args);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.directRecvDirectSend(offset, offset, nelem);
          }
        } else {
          // Reduce, send to network
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndGather, nThreadsReduce, &nvls->down, &nvls->out, NULL, NULL,
              args->redOpArg, 2 * Proto::MaxGroupWidth, 0, 1, args);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.directRecvDirectSend(offset, offset, nelem);
          }
        }
      } else if (tid < tidEndBcast && nvls->headRank != -1) {
        // Recv from network, broadcast
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndReduce, nThreadsBcast, &nvls->out, &nvls->down, NULL, NULL,
            args->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, args);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const int treeUp = nvls->treeUp;
    const int* treeDown = nvls->treeDown;
    ssize_t chunkCount = args->chunkCount;
    const ssize_t loopCount = nvls->nHeads * chunkCount;
    const ssize_t channelCount = args->workCount;
    const ssize_t gridOffset = args->workOffset;
    const int nranks = ncclShmem.comm.nRanks;
    const bool hasUp = treeUp != -1;
    const int totalWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
    const int bcastWarps = hasUp ? (args->regUsed ? ((totalWarps - 2) >> 1) - 1 : 4) : 0;
    const int reduceWarps = args->regUsed ? (totalWarps - bcastWarps - 2) : (hasUp ? 5 : nranks <= 6 ? 7 : 5);
    const int scatterWarps = args->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
    const int gatherWarps = args->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;
    ssize_t offset;
    int nelem;

    const int nThreadsScatter = scatterWarps*WARP_SIZE;
    const int nThreadsGather  = gatherWarps*WARP_SIZE;
    const int nThreadsReduce = reduceWarps*WARP_SIZE;
    const int nThreadsBcast  = (bcastWarps)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;
    const int tidEndBcast = tidEndReduce + nThreadsBcast;

    if (tid < tidEndScatter) {
      // Scatter
      using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, nThreadsScatter, NULL, nvls->up, args->sendbuff, NULL,
          args->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        if (channelCount - elemOffset < loopCount) chunkCount = args->lastChunkCount;
        offset = gridOffset + elemOffset;
        nelem = args->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
        prims.scatter(offset, nelem, chunkCount, chunkCount, -1, 0);
      }
    } else if (tid < tidEndGather) {
      // Gather
      using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, args->recvbuff,
          args->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        if (channelCount - elemOffset < loopCount) chunkCount = args->lastChunkCount;
        offset = gridOffset + elemOffset;
        nelem = args->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
        prims.gather(offset, nelem, chunkCount, chunkCount, -1, 0);
      }
    } else if (tid < tidEndReduce && nvls->headRank != -1) {
      if (!hasUp) {
        // Reduce and Broadcast
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
        Primitives<T, RedOp, FanSymmetric<3>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, treeDown, treeDown, NULL, NULL,
            args->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, args);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkCount = args->lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkCount;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkCount, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      } else {
        // Reduce, send to network
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
        Primitives<T, RedOp, FanAsymmetric<3, 1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, treeDown, &treeUp, NULL, NULL,
            args->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, args);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkCount = args->lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkCount;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkCount, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    } else if (tid < tidEndBcast && nvls->headRank != -1) {
      // Recv from network, broadcast
      using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
      Primitives<T, RedOp, FanAsymmetric<1, 3>, /*Direct=*/1, Proto, 0>
        prims(tid - tidEndReduce, nThreadsBcast, &treeUp, treeDown, NULL, NULL,
          args->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, args);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        ssize_t chunkOffset;
        if (channelCount - elemOffset < loopCount) chunkCount = args->lastChunkCount;
        chunkOffset = elemOffset + nvls->headRank * chunkCount;
        offset = gridOffset + chunkOffset;
        nelem = min(chunkCount, channelCount - chunkOffset);
        prims.directRecvDirectSend(offset, offset, nelem);
      }
    }
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
    ssize_t chunkSize = args->chunkCount;
    const ssize_t loopSize = int(nChannels*chunkSize);
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t size = args->count;

    int nthreadsSplit = nthreads/2;
    if (nthreadsSplit >= 256) nthreadsSplit += 64;

    int group, connIndex, send, recv, groupTid, groupNthreads;
    using Proto = ProtoSimple<1, 1>;
    if (tid < nthreadsSplit) {
      // Reduce up the chain
      group = 0;
      connIndex = 1;
      recv = tree->down[0];
      send = tree->up;
      groupTid = tid;
      groupNthreads = nthreadsSplit;
    } else {
      // Broadcast down the chain
      group = 1;
      connIndex = 0;
      recv = tree->up;
      send = tree->down[0];
      groupTid = tid - nthreadsSplit;
      groupNthreads = nthreads-nthreadsSplit;
    }

    Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
      prims(groupTid, groupNthreads, &recv, &send, args->sendbuff, args->recvbuff,
          args->redOpArg, group*Proto::MaxGroupWidth, connIndex, connIndex);

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
      if (recv == nranks) {
        // I'm the first in the broadcast chain, I need to perform the division (postOp)
        if (send == -1) {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*int(chunkSize);
            int nelem = min(chunkSize, size-offset);
            prims.recv(offset, nelem, /*postOp*/true);
          }
        } else {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*int(chunkSize);
            int nelem = min(chunkSize, size-offset);
            prims.recvCopyDirectSend(offset, nelem, /*postOp*/true);
          }
        }
      } else {
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
            prims.directRecvCopySend(offset, nelem);
          }
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
