/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "device.h"
#include "op128.h"
#include "reduce_kernel.h"
#include "network/unpack/unpack_defs.h"

#define COLL_UNROLL (ncclCollUnroll())

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

typedef void(*ncclDevFuncPtr_t)();
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable[];

struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_ARITY];
  void* userInput;
  void* userOutput;
  void* srcs[NCCL_MAX_ARITY+1];
  void* dsts[NCCL_MAX_ARITY+1];
  union {
    unpackGroupShmem unpack;
  } devicePlugin;
  int32_t dstSizes[NCCL_MAX_ARITY+1];
  uint64_t redOpArgs;
};

struct ncclShmemData {
  struct ncclDevKernelArgs args;
  int channelId;
  int aborted;
  alignas(16) struct ncclKernelComm comm;
  alignas(16) struct ncclDevChannel channel;

  int batchIx, nextBatchIx;
  enum ncclDevWorkType workType;
  uint8_t directMode;
  uint16_t funcId;
  int nWorks;
  int workSize;
  uint64_t workCounter;
  bool profilerEnabled;
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];

  alignas(16) char workStorage[ncclMaxDevWorkBatchBytes()];

  alignas(16) union {
    unpackShmem unpack;
  } devicePlugin;
};

extern __shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ >= 700
  extern __shared__ ulong2 ncclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];
#else
  extern __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

__device__ inline void* ncclScratchForWarp(int warp) {
  return (char*)ncclShmemPerWarp + warp*ncclShmemScratchWarpSize();
}

__device__ inline void barrier_sync(int name) {
  #if 0
  asm volatile("barrier.sync %0;" :: "r"(name) : "memory");
  #else
  asm volatile("barrier.sync.aligned %0;" :: "r"(name) : "memory");
  #endif
}
__device__ inline void barrier_sync(int name, int nThreads) {
  #if 0
  asm volatile("barrier.sync %0, %1;" :: "r"(name), "r"(nThreads) : "memory");
  #else
  asm volatile("barrier.sync.aligned %0, %1;" :: "r"(name), "r"(nThreads) : "memory");
  #endif
}
__device__ inline void barrier_sync_aligned(int name) {
  asm volatile("barrier.sync.aligned %0;" :: "r"(name) : "memory");
}
__device__ inline void barrier_sync_aligned(int name, int nThreads) {
  asm volatile("barrier.sync.aligned %0, %1;" :: "r"(name), "r"(nThreads) : "memory");
}

__device__ inline bool barrier_red_or(bool vote, int name) {
  int ans;
  asm volatile("{ .reg .pred p;"
      "  setp.ne.s32 p, %1, 0;"
      "  barrier.red.or.pred p, %2, p; "
      "  selp.s32 %0, 1, 0, p; }"
      : "=r"(ans) : "r"((int)vote), "r"(name) : "memory");
  return bool(ans);
}
__device__ inline bool barrier_red_or(bool vote, int name, int nThreads) {
  int ans;
  asm volatile("{ .reg .pred p;"
      "  setp.ne.s32 p, %1, 0;"
      "  barrier.red.or.pred p, %2, %3, p; "
      "  selp.s32 %0, 1, 0, p; }"
      : "=r"(ans) : "r"((int)vote), "r"(name), "r"(nThreads) : "memory");
  return bool(ans);
}

// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    uint64_t a=0, b=0;
    asm volatile("ld.v2.u64 {%0,%1},[%2];" : "=l"(a),"=l"(b) : "l"((char const*)src + offset) : "memory");
    uint32_t udst = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("st.shared.v2.u64 [%0],{%1,%2};" :: "r"(udst + offset), "l"(a), "l"(b) : "memory");
  }
}

// Must run with at least 64 threads
__device__ __forceinline__ void loadWorkBatchToShmem(
    int tid, int tn, struct ncclDevKernelArgs const* args, int batchIx
  ) {
  int lane = tid%WARP_SIZE;
  int workCursor = 0; // num works written in previous loop iterations.
  while (true) {
    struct ncclDevWorkBatch batch = ((struct ncclDevWorkBatch*)(args+1))[batchIx];

    // fnsOfBitset[n] = index of n'th set bit in batch.offsetBitset.
    // PTX has instruction "fns" (find n-th set) but it expands to a lot of SASS,
    // since we know all lanes will be querying the same bitmask we can compute
    // much faster using shared memory.
    uint8_t* fnsOfBitset = (uint8_t*)ncclScratchForWarp(threadIdx.x/WARP_SIZE);
    __syncwarp();
    if (uint32_t(batch.offsetBitset) & (1u<<lane)) {
      int nWorksBelow = __popc(uint32_t(batch.offsetBitset) & ((1u<<lane)-1));
      fnsOfBitset[nWorksBelow] = lane;
    }
    int nWorksLow32 = __popc(uint32_t(batch.offsetBitset)); // just of low 32 bits
    if (uint32_t(batch.offsetBitset>>32) & (1u<<lane)) {
      int nWorksBelow = nWorksLow32;
      nWorksBelow += __popc(uint32_t(batch.offsetBitset>>32) & ((1u<<lane)-1));
      fnsOfBitset[nWorksBelow] = 32 + lane;
    }
    int nWorks = nWorksLow32 + __popc(uint32_t(batch.offsetBitset>>32)); // add high 32 bits
    __syncwarp();

    int workSize;
    int nPacks; // total number of packs loaded, each pack is 16 bytes
    int packInWork; // my pack index within work struct
    int dstWork; // my work index in contiguous destination shmem
    switch (batch.workType) {
    case (int)ncclDevWorkTypeP2p:
      workSize = sizeof(struct ncclDevWorkP2p);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    case (int)ncclDevWorkTypeColl:
      workSize = sizeof(struct ncclDevWorkColl);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    case (int)ncclDevWorkTypeBcast:
      workSize = sizeof(struct ncclDevWorkBcast);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    case (int)ncclDevWorkTypeCollReg:
    default:
      workSize = sizeof(struct ncclDevWorkCollReg);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    }
    if (tid == 0) {
      ncclShmem.workSize = workSize;
    }
    // We deliberately replicate these div and mod calculations into the case
    // blocks above so that they get constant divisor optimizations by the compiler.
    //   packInWork = tid%(workSize/16);
    //   dstWork = tid/(workSize/16);

    // We can only assume we have 64 threads, which means we can read at most 1024 bytes
    // here which is the per batch maximum.
    if (tid < nPacks) {
      int srcWork = fnsOfBitset[dstWork]; // find n'th set bit in batch.offsetBitset
      ulong2 tmp;
      // The loads done in these two cases must be kept separate since we are
      // relying on the compiler to use "ld.param" in the first one. The parameter
      // space is not generically addressable, so any attempt to load through
      // a pointer that *might* be parameter space backed will cause the
      // compiler to spill the parameter struct (4K!) to each thread's local space
      // before creating a pointer (to the spill) and decimate perf.
      //
      // An example of what not to do would be the following:
      //
      // if (condition) {
      //   // The compiler could spill parameter_variable to local space and take
      //   // the address of that, since when src is loaded below it could also
      //   // be global space.
      //   src = &parameter_variable;
      // } else {
      //   src = &global_variable;
      // }
      // memcpy(dst, src, n);
      if (ncclShmem.args.workStorageType == ncclDevWorkStorageTypeArgs) {
        char* src = (char*)args + (batch.offsetBase + srcWork*workSize + packInWork*16);
        tmp = *(ulong2*)src; // becomes ld.param.v2.u64
      } else {
        char* src = (char*)ncclShmem.args.workBuf + ((batch.offsetBase + srcWork*workSize + packInWork*16) & ncclShmem.args.workMask);
        tmp = *(ulong2*)src; // becomes ld.v2.u64
      }
      char* dst = ncclShmem.workStorage;
      dst += (workCursor + dstWork)*workSize + packInWork*16;
      *(ulong2*)dst = tmp;
    }
    workCursor += nWorks;

    if (batch.nextExtends) {
      batchIx += batch.nextJump;
      tid -= 64; // Rotate threads so we use the next two warps for next batch struct.
      if (tid < 0) tid += tn;
    } else {
      if (tid == 0) {
        ncclShmem.batchIx = batchIx;
        ncclShmem.nextBatchIx = (batch.nextJump == 0) ? -1 : batchIx + batch.nextJump;
        ncclShmem.workType = (enum ncclDevWorkType)batch.workType;
        ncclShmem.nWorks = workCursor;
        ncclShmem.funcId = batch.funcId;
      }
      break;
    }
  }
}

__device__ __forceinline__ unsigned long long int globaltimer() {
  unsigned long long int timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkColl {
  __device__ void run(int tid, int tn, struct ncclDevWorkColl* work) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch;

// Specialized for P2p in sendrecv.h
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE>;

template<typename T, typename RedOp, int Proto>
struct RunWorkBatch<ncclFuncAllGatherV, T, RedOp, NCCL_ALGO_RING, Proto>;

// Specialized here for non-P2p (Coll and CollReg)
template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run() {
    int tid = threadIdx.x;
    int tn = blockDim.x;

    if (RedOpArg<RedOp>::ArgUsed) {
      int nWorks = ncclShmem.nWorks;
      for (int w=tid; w < nWorks; w += tn) {
        struct ncclDevWorkColl* work = (ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
        if (work->redOpArgIsPtr) {
          work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
        }
      }
      __syncthreads();
    }

    #pragma unroll 1
    for (int w=0; w < ncclShmem.nWorks; w++) {
      struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
      if (w != 0) {
        struct ncclDevWorkColl* workPrev = (struct ncclDevWorkColl*)(ncclShmem.workStorage + (w-1)*ncclShmem.workSize);
        if (work->nWarps != workPrev->nWarps) __syncthreads();
      }
      int subtn = work->nWarps*WARP_SIZE;
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      if (tid < subtn) RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(tid, subtn, work);
    }
  }
};

#define START 0
#define STOP  1
#define FINI  2

__device__ __forceinline__ bool profilerEnabled(int workItemIdx) {
  return (ncclShmem.workType == ncclDevWorkTypeP2p) ?
    ((struct ncclDevWorkP2p*)ncclShmem.workStorage)[workItemIdx].profilerEnabled :
    ((struct ncclDevWorkColl*)ncclShmem.workStorage)[workItemIdx].profilerEnabled;
}

__device__ __forceinline__ void profiler(int action) {
  if (threadIdx.x == 0) {
    int idx = 0;
    uint64_t wc = ncclShmem.channel.workCounter + 1;
    if (action == START) {
      for (; wc <= ncclShmem.channel.workCounter + ncclShmem.nWorks; wc++) {
        if (!profilerEnabled(idx++)) continue;
        ncclShmem.comm.workStarted[ncclShmem.channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp = globaltimer();
        ncclShmem.comm.workStarted[ncclShmem.channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].counter = wc;
      }
    } else {
      for (; wc <= ncclShmem.channel.workCounter + ncclShmem.nWorks; wc++) {
        if (!profilerEnabled(idx++)) continue;
        ncclShmem.comm.workCompleted[ncclShmem.channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp = globaltimer();
        ncclShmem.comm.workCompleted[ncclShmem.channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].counter = wc;
      }
      ncclShmem.channel.workCounter += ncclShmem.nWorks;
      if (action == FINI) ((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId].workCounter = ncclShmem.channel.workCounter;
    }
  }
}

template<int SpecializedFnId, typename SpecializedRunWorkBatch>
__device__ __forceinline__ void ncclKernelMain(struct ncclDevKernelArgs const* args) {
  int tid = threadIdx.x;
  int tn = blockDim.x;

  // Copy kernel args to shmem and then only read those. Otherwise the compiler
  // will end up putting the args into thread local stack which is very wasteful.
  if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
    ((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid];
  }

  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  // PTX has the fns instruction which does this but is extremely slow. We can
  // do better when we know all threads are querying the same bitmask.
  if (tid < MAXCHANNELS && (args->channelMask & (1ull<<tid))) {
    int n = __popcll(args->channelMask & ((1ull<<tid)-1));
    if (blockIdx.x == n) ncclShmem.channelId = tid;
  }
  __syncthreads(); // publish ncclShmem.{args, channelId}
  /* set abort flag to 0 */
  if (tid == 0) {
    ncclShmem.aborted = 0;
    ncclShmem.channel.workCounter = ((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId].workCounter;
  }

  // Use first 2 warps to load comm and channel, and remaining load work batch.
  switch (tid/WARP_SIZE) {
  case 0:
    { void* dst = &ncclShmem.comm;
      void* src = ncclShmem.args.comm;
      int bytes = sizeof(ncclKernelComm);
      static_assert(sizeof(ncclKernelComm) <= 16*WARP_SIZE, "ncclKernelComm cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid, dst, src, bytes);
    } break;
  case 1:
    { // Get address of channel without incurring indirect load from ncclKernelComm::channels
      void* dst = &ncclShmem.channel;
      void* src = &((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId];
      int bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid-WARP_SIZE, dst, src, bytes);
    } break;
  default:
    { int subtid = tid - 2*WARP_SIZE;
      int subtn = tn - 2*WARP_SIZE;
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      loadWorkBatchToShmem(subtid, subtn, args, /*batchIx=*/blockIdx.x);
    } break;
  }
  __syncthreads(); // publish ncclShmem

  while (ncclShmem.aborted == 0) {
    profiler(START);
    if (0 <= SpecializedFnId && ncclShmem.funcId == (unsigned)SpecializedFnId) {
      SpecializedRunWorkBatch().run();
    } else {
      ncclDevFuncTable[ncclShmem.funcId]();
    }

    if (ncclShmem.nextBatchIx == -1) break;
    int batchIx = ncclShmem.nextBatchIx;
    __syncthreads();
    profiler(STOP);
    loadWorkBatchToShmem(tid, tn, args, batchIx);
    __syncthreads();
  }
  profiler(FINI);
}

__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K);
__device__ void ncclDevFunc_Nop();

#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain<specializedFnId, RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args); \
  }

#define DEFINE_ncclDevKernel_nop(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {}

#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto) \
  __device__ void ncclDevFunc_##suffix() { \
    RunWorkBatch<coll, ty, redop<ty>, algo, proto>().run(); \
  }

#endif
