/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

#include "msccl/msccl_struct.h"
#include "msccl/msccl_kernel.h"

__shared__ struct mscclShmemData mscclShmem;

#define MSCCL_MAX_ITER 65536
#define DEBUG_PRINT 0

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
  MSCCL_MAX_ITER*MSCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * MSCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)

// a copy of the volatile load/store from prims_ll
template<typename U>
__device__ static U load(U *src) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };

  if(sizeof(U) == 1)
    asm("ld.volatile.global.b8 %0,[%1];" : "=r"(u4) : "l"(src));
  else if(sizeof(U) == 2)
    asm("ld.volatile.global.b16 %0,[%1];" : "=h"(u2) : "l"(src));
  else if(sizeof(U) == 4)
    asm("ld.volatile.global.b32 %0,[%1];" : "=r"(u4) : "l"(src));
  else
    asm("ld.volatile.global.b64 %0,[%1];" : "=l"(u8) : "l"(src));
  return elt;
}

template<typename U>
__device__ static void store(U *dst, U val) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };

  elt = val;
  if(sizeof(U) == 1)
    asm("st.volatile.global.b8 [%0],%1;" :: "l"(dst), "r"(u4));
  else if(sizeof(U) == 2)
    asm("st.volatile.global.b16 [%0],%1;" :: "l"(dst), "h"(u2));
  else if(sizeof(U) == 4)
    asm("st.volatile.global.b32 [%0],%1;" :: "l"(dst), "r"(u4));
  else
    asm("st.volatile.global.b64 [%0],%1;" :: "l"(dst), "l"(u8));
}

inline __device__ static void barrier(int nthreads) {
    asm volatile ("bar.sync %1, %0;" :: "r"(nthreads), "r"(15));
}

__device__ __forceinline__ static void threadBlockCopy(
  uint64_t *dst, uint64_t const *src, uint64_t size, int tid, int nthreads) {
  for (int i = tid; i < size; i += nthreads) {
    dst[i] = src[i];
  }
}

template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void mscclRunInterpreter(
  struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork work) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  // initialize mscclShmem.mscclTB
  threadBlockCopy(
    (uint64_t *)&mscclShmem.mscclTB, (uint64_t *)(algo->mscclTBs + bid),
    sizeof(struct mscclThreadBlock)/sizeof(uint64_t), tid, nthreads);
  __syncthreads(); // publish mscclShmem.mscclTB.channelId

  // initialize ncclShmem and mscclShmem.work
  int channelId = mscclShmem.mscclTB.channelId;
  {
    void *dst, *src;
    int bytes = 0;
    // Use first 3 warps to load comm, channel, and work into shmem
    switch (tid/WARP_SIZE) {
    case 0:
      dst = &ncclShmem.comm;
      src = comm;
      bytes = sizeof(ncclDevComm);
      static_assert(sizeof(ncclDevComm) <= 16 * WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
      break;
    case 1:
      // Get address of channel without incurring indirect load from ncclDevComm::channels
      dst = &ncclShmem.channel;
      src = &((ncclDevCommAndChannels*)comm)->channels[channelId];
      bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16 * WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      break;
    case 2:
      dst = &mscclShmem.work;
      src = &work;
      bytes = sizeof(mscclWork);
      static_assert(sizeof(mscclWork) <= 16 * WARP_SIZE, "mscclWork cannot be loaded by a single warp in one insn.");
      break;
    case 3:
      /* set abort flag to 0 */
      if (tid == 3 * WARP_SIZE) ncclShmem.aborted = 0;
      break;
    default:
      break;
    }
    copyToShmem16(tid%WARP_SIZE, dst, src, bytes);
  }
  __syncthreads(); // publish shmem
  
  // Deference reduce args if required
  if (tid == 0 && mscclShmem.work.hasReduce && mscclShmem.work.redOpArgIsPtr) {
    switch (sizeof(T)) {
      case 1:
        mscclShmem.work.redOpArg = *reinterpret_cast<uint8_t*>(mscclShmem.work.redOpArg);
        break;
      case 2:
        mscclShmem.work.redOpArg = *reinterpret_cast<uint16_t*>(mscclShmem.work.redOpArg);
        break;
      case 4:
        mscclShmem.work.redOpArg = *reinterpret_cast<uint32_t*>(mscclShmem.work.redOpArg);
        break;
      case 8:
        mscclShmem.work.redOpArg = *reinterpret_cast<uint64_t*>(mscclShmem.work.redOpArg);
        break;
      default:
        break;
    }
  }
  __syncthreads(); // publish shmem

  // User pointers for primitives
  T* thisInput = (T*)mscclShmem.work.sendBuff;
  T* thisOutput = (T*)mscclShmem.work.recvBuff;
  T* thisScratch = (T*)mscclShmem.work.scratchBuffer;
  int recvPeer = mscclShmem.mscclTB.recvPeer;
  int sendPeer = mscclShmem.mscclTB.sendPeer;

  const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? MSCCL_CHUNKSTEPS : 1));
  int minChunkSize;
  if (Proto::Id == NCCL_PROTO_LL)
    minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
  if (Proto::Id == NCCL_PROTO_LL128) {
    // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
  }

  RedOp redFn(mscclShmem.work.redOpArg);
  Primitives<T, RedOp, FanAsymmetric<1,1>, 1, Proto, 0> prims
    (tid, nthreads, &recvPeer, &sendPeer, thisInput, thisOutput, mscclShmem.work.redOpArg);

  const ssize_t sizePerMscclChunk = mscclShmem.work.count / mscclShmem.work.nChunksPerLoop;
  uint32_t maxAllowedCount = mscclShmem.work.maxAllowedCount;

  // msccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
  // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
  const int64_t workIndex = mscclShmem.work.workIndex;
  volatile struct mscclFlag* mscclFlags = mscclShmem.work.syncFlags;
  for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerMscclChunk; gridOffset += chunkSize, iter++) {
    ssize_t realChunkSize;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      realChunkSize = min(chunkSize, sizePerMscclChunk-gridOffset);
      realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
    }
    else
      realChunkSize = min(chunkSize, divUp(sizePerMscclChunk-gridOffset, minChunkSize)*minChunkSize);
    
    realChunkSize = int(realChunkSize);
    int nelem = min(realChunkSize, sizePerMscclChunk-gridOffset);

    ssize_t srcOffset, dstOffset;
    T *srcPointer, *dstPointer;
    int step = 0;
    for (int i = 0; i < mscclShmem.mscclTB.nSteps; i++){
      struct mscclTransmission* t = &mscclShmem.mscclTB.transmissions[i];
      // first wait if there is a dependence
      int16_t numDependencies = t->numDependencies;
      if (numDependencies > 0){
        if (tid < numDependencies) {
          int16_t dependentPointer = t->dependencePointer;
          int8_t dependentBid = mscclShmem.mscclTB.dependentBid[dependentPointer+tid];
          int16_t dependentStep = mscclShmem.mscclTB.dependentStep[dependentPointer+tid];
          uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
          while ((mscclFlags + dependentBid)->flag < goalFlag);
        }
        step += numDependencies-1;
        barrier(nthreads);
      }

      srcPointer = (t->srcBuffer == MSCCL_INPUT_BUFFER) ? thisInput : ((t->srcBuffer == MSCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
      dstPointer = (t->dstBuffer == MSCCL_INPUT_BUFFER) ? thisInput : ((t->dstBuffer == MSCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
      prims.setDataPtrs(srcPointer, dstPointer);
      int count = t->count;
      for (int c = 0; c < count; c += maxAllowedCount) {
        srcOffset = gridOffset + (ssize_t) (t->srcOffset+c) * sizePerMscclChunk;
        dstOffset = gridOffset + (ssize_t) (t->dstOffset+c) * sizePerMscclChunk;
        int thisCount = min(maxAllowedCount, count - c);
        int thisNelem = nelem * thisCount;
        if (t->type == MSCCL_SEND)
          prims.sendWithBarrier(srcOffset, thisNelem); // LL.send is the only situation where there is no barrier at the end.
        else if (t->type == MSCCL_RECV){
          prims.recv(dstOffset, thisNelem);
        }
        else if (t->type == MSCCL_REDUCE) {
          int numReductions = t->numReductions;
          if (thisNelem < nthreads){
            if (tid < thisNelem){
              dstOffset = gridOffset + (ssize_t) (t->dstOffset+c) * sizePerMscclChunk;
              T* dstIndex = dstPointer + dstOffset + tid;
              T reduceInput;
              T o = load(dstIndex);
              ssize_t srcBaseOffset = gridOffset + (ssize_t)c * sizePerMscclChunk + tid;
              for (int r = 0; r < numReductions; r++){
                  srcOffset = srcBaseOffset + (ssize_t)mscclShmem.mscclTB.reductionSrcOffsets[t->reductionPointer+r] * sizePerMscclChunk;
                  reduceInput = load(srcPointer + srcOffset);
                  o = applyReduce(redFn, reduceInput, o);
              }
              store(dstIndex, o);
            }
            barrier(nthreads);
          } else {
            T* srcs[MSCCL_MAX_REDUCE_FUSION+1]; // +1 is for SIMPLE protocol as dst is added in the list of srcs
            dstOffset = gridOffset + (ssize_t) (t->dstOffset+c) * sizePerMscclChunk;
            T* dst = dstPointer + dstOffset;
            ssize_t srcBaseOffset = gridOffset + (ssize_t)c * sizePerMscclChunk;
            for (int r = 0; r < numReductions; r++){
                srcOffset = srcBaseOffset + (ssize_t)mscclShmem.mscclTB.reductionSrcOffsets[t->reductionPointer+r] * sizePerMscclChunk;
                srcs[r] = srcPointer + srcOffset;
            }
            prims.reduce(srcs, numReductions, &dst, 1, thisNelem);
          }
          if (c == 0) step += (numReductions-1); // only advance step once!
        } else if (t->type == MSCCL_RECV_COPY_SEND)
          prims.recvCopySend(dstOffset, thisNelem);
        else if (t->type == MSCCL_RECV_REDUCE_SEND)
          prims.recvReduceSend(srcOffset, thisNelem);
        else if (t->type == MSCCL_RECV_REDUCE_COPY_SEND)
          prims.recvReduceCopySend(srcOffset, dstOffset, thisNelem);
        else if (t->type == MSCCL_RECV_REDUCE_COPY)
          prims.recvReduceCopy(srcOffset, dstOffset, thisNelem);
        else if (t->type == MSCCL_LOCAL_COPY)
          prims.localCopy(srcPointer+srcOffset, dstPointer+dstOffset, thisNelem);
        else
          return;
      }
      if (t->hasDependence && tid == nthreads-1){
        mscclFlags[bid].flag = (uint64_t) COMPUTE_FLAG(workIndex, iter, step);
      }
      step++;
    }
  }
}

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, type) \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork work) { \
  mscclRunInterpreter<type, Func##devredop<type>, ProtoLL>(comm, algo, work); \
} \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL128)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork work) { \
  mscclRunInterpreter<type, Func##devredop<type>, ProtoLL128>(comm, algo, work); \
} \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, Simple)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork work) { \
  mscclRunInterpreter<type, Func##devredop<type>, ProtoSimple<MSCCL_CHUNKSTEPS/MSCCL_SLICESTEPS, MSCCL_SLICESTEPS>>(comm, algo, work); \
}

#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(__CUDA_FP8_TYPES_EXIST__)
#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(devredop) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, half) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, float) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, double) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, __nv_bfloat16) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, __nv_fp8_e4m3) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, __nv_fp8_e5m2)
#elif defined(__CUDA_BF16_TYPES_EXIST__)
#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(devredop) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, half) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, float) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, double) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, __nv_bfloat16)
#else
#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(devredop) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, half) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, float) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, double)
#endif

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_NOFLOAT(devredop) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t)

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC() \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(Sum) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(Prod) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(Max) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(Min) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(PreMulSum) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_NOFLOAT(SumPostDiv)

MSCCL_IMPL_KERNEL_ENTRY_FUNC()