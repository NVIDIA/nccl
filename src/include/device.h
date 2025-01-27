/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "nccl_common.h"
#include "bitops.h"
#include <algorithm>
#include <stdint.h>
#include <sys/types.h>

extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS];

extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];

extern const char* ncclProtoStr[NCCL_NUM_PROTOCOLS];

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

#ifdef __CUDA_ARCH__
  #define NCCL_CUDA_ARCH __CUDA_ARCH__
#else
  #define NCCL_CUDA_ARCH 0
#endif

#include "net_device.h"

enum ncclDevRedOp_t {
  ncclDevSum, ncclDevProd, ncclDevMinMax,
  ncclDevPreMulSum, ncclDevSumPostDiv,
  ncclNumDevRedOps
};
struct ncclDevRedOpFull {
  ncclDevRedOp_t op;
  ncclRedOp_t proxyOp;
  bool scalarArgIsPtr;
  uint64_t scalarArg;
};

union ncclLLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};

#define WARP_SIZE 32
#define MAXCHANNELS 64
#define NCCL_MAX_LOCAL_RANKS 72
#define NCCL_MAX_NTHREADS 640
#define NCCL_MIN_NTHREADS (4*WARP_SIZE)
#define NCCL_SIMPLE_MAX_NTHREADS 512
#define NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE (3*WARP_SIZE)
#define NCCL_LL_MAX_NTHREADS 512
#define NCCL_LL_LINES_PER_THREAD 8
#ifdef TEST_LL_CLEANUP
#define NCCL_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup
#define NCCL_LL_FLAG_MAX   0x100
#define NCCL_LL_FLAG(a) ((uint32_t)((a) % NCCL_LL_FLAG_MAX))
#else
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#endif
// Make sure the clean mask will last for at least NCCL_NSTEPS
static_assert(NCCL_LL_CLEAN_MASK % NCCL_STEPS == 0, "Invalid NCCL_LL_CLEAN_MASK value");

#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)

#define NCCL_LL128_MAX_NTHREADS 640
#define NCCL_LL128_ELEMS_PER_THREAD 120

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)

#define NCCL_P2P_WRITE 0x01
#define NCCL_P2P_READ  0x02
#define NCCL_DIRECT_NIC   0x04
#define NCCL_NVLS_MIN_POLL 0x80

// Number of named barriers supported by CUDA
#define NCCL_MAX_GROUPS 16

#define NCCL_REGULAR_BUFFER 0x00
#define NCCL_IPC_REG_BUFFER 0x01
#define NCCL_NVLS_REG_BUFFER 0x02
#define NCCL_NET_REG_BUFFER 0x04

struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int flags;          // Direct communication / other flags
  int shared;         // Buffers are shared
  int stepSize;       // Step size for the SIMPLE buffer
  void **ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  struct ncclConnFifo* connFifo; // Used for GPU - Proxy communication

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
  ncclNetDeviceHandle_t netDeviceHandle;
};

struct ncclProxyConnector {
  bool initialized;
  int rank;
  int tpRank;
  int tpLocalRank;
  int sameProcess;
  struct ncclProxyConnection* connection;
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*); // Copied from transport if necessary
};

struct ncclConnector {
  int connected;
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;
  void* transportResources;
  struct ncclConnInfo conn;
};

struct ncclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;

  int index; // This rank's index in the ring
};


// The root of each tree only has one node down (+1 intra-node).
#define NCCL_MAX_TREE_ARITY_TOP 2
// Nodes inside the binary tree can have to two nodes down (+1 intra-node).
#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

#define NCCL_MAX_DIRECT_ARITY 7
struct ncclDirect {
  int depth;
  int out;
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int shift;    // Shuffling of send/recv for scatter/gather operations, basically localRank%nHeads
  // The heads[...] are guaranteed to be in rotated order start with self:
  //   headRank, (headRank+1)%nHeads, (headRank+2)%nHeads, ...
  int heads[NCCL_MAX_DIRECT_ARITY+1];
  int up[NCCL_MAX_DIRECT_ARITY];
  int down[NCCL_MAX_DIRECT_ARITY];
};

#define NCCL_MAX_NVLS_ARITY 32
#define NCCL_MAX_NVLS_TREE_ARITY 3
struct ncclNvls {
  int out;
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int up[NCCL_MAX_NVLS_ARITY];
  int down;
  int treeUp;
  int treeDown[NCCL_MAX_NVLS_TREE_ARITY];
};

#if __CUDA_ARCH__ >= 900
#define NCCL_MAX_ARITY NCCL_MAX_NVLS_ARITY
#else
#define NCCL_MAX_ARITY NCCL_MAX_DIRECT_ARITY
#endif

#define NCCL_MAX_CONNS 2
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];
  struct ncclConnector recv[NCCL_MAX_CONNS];
  int refCount;
};

struct ncclDevComm;

struct alignas(16) ncclDevWorkP2p {
  void *sendAddr, *recvAddr;
  size_t sendBytes, recvBytes;
  int sendRank, recvRank;
  // From the part index, nP2pChannels, and channelBase the device code can
  // calculate which part of the transfer a channel is responsible for.
  uint8_t nP2pChannels; // Always equal to comm->p2pnChannels
  uint8_t channelBase; // Channel owning first part.
  // Zero channels indicates no work in that direction.
  uint8_t nSendChannels, nRecvChannels;
  // Chunk size stored in 8 bits via u32fp8Encode/Decode.
  uint8_t sendChunkSize_u32fp8, recvChunkSize_u32fp8;

  uint8_t sendProtoLL:1, recvProtoLL:1;
  uint8_t sendNetReg:1, recvNetReg:1;
  uint8_t sendIpcReg:1, recvIpcReg:1;
};

// Compute the subset of the data transfer corresponding to the given part index.
inline __host__ __device__ void ncclP2pPartBounds(int nParts, int part, size_t bytes, size_t* partBeg, size_t* partEnd) {
  size_t partBytes = alignUp(divUp(bytes, nParts), 4<<10);
  #if __CUDA_ARCH__
    *partBeg = min((part+0)*partBytes, bytes);
    *partEnd = min((part+1)*partBytes, bytes);
  #else
    *partBeg = std::min<size_t>((part+0)*partBytes, bytes);
    *partEnd = std::min<size_t>((part+1)*partBytes, bytes);
  #endif
}

// implemented in channel.h
inline __host__ uint8_t ncclP2pChannelBaseForRound(struct ncclComm* comm, int p2pRound);

// ncclP2pChannelToPart and ncclP2pChannelForPart are inverses. The device code
// uses ncclP2pChannelToPart to determine which part "this" channel is responsible for.
inline __host__ int ncclP2pChannelForPart(int nP2pChannels, int base, int part) {
  // Only works because nP2pChannels is pow2
  int nChannelsLog2 = countOneBits(nP2pChannels-1);
  int delta = reverseBits(part, nChannelsLog2);
  return (base + delta) & (nP2pChannels-1);
}
inline __device__ int ncclP2pChannelToPart(int nP2pChannels, int base, int channel) {
  // Only works because nP2pChannels is pow2
  int nChannelsLog2 = countOneBits(nP2pChannels-1);
  int delta = (channel-base) & (nP2pChannels-1);
  return reverseBits(delta, nChannelsLog2);
}

struct alignas(16) ncclDevWorkColl {
  // Running on channels [channelLo..channelHi], hi is inclusive.
  //   nChannels == (channelHi - channelLo) + 1
  uint32_t channelLo:8, channelHi:8;
  uint32_t nWarps:8;
  uint32_t redOpArgIsPtr:1, regUsed:1, netRegUsed:1, oneNode:1, direct:2, isOneRPN:1;
  uint32_t root;
  void* recvbuff;
  void* sendbuff;
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;
  union {
    // Continuous-byte-distribution scheduling. The lo and hi channels are of
    // different size than the channels in the middle.
    struct {
      size_t countLo, countMid, countHi;
      // Chunk counts where units are ncclProtoGrainSize(protocol) bytes
      uint64_t chunkGrainsLo:21, chunkGrainsMid:21, chunkGrainsHi:21;
    } cbd;
    // Collnet scheduling. All channels divide work evenly.
    struct {
      size_t count; // Total size, not divided per channel.
      uint32_t chunkCount;
    } collnet;
  };
  uint64_t redOpArg;
};


__host__ __device__ constexpr int ncclProtoGrainSize(int proto) {
  return proto == NCCL_PROTO_LL ? 16 :
         proto == NCCL_PROTO_LL128 ? WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD/NCCL_LL128_LINEELEMS*NCCL_LL128_DATAELEMS*sizeof(uint64_t) :
         proto == NCCL_PROTO_SIMPLE ? 512 :
         -1;
}

template<typename Int>
__host__ __device__ inline void ncclCollCbdPart(
    struct ncclDevWorkColl* work, uint32_t channelId, int proto, int eltSize,
    Int* count, Int* partOffset, Int* partCount, Int* chunkCount
  ) {
  int eltPerGrain = ncclProtoGrainSize(proto)/eltSize;
  int nMidChannels = work->channelHi - work->channelLo - 1;
  // We can assum that nMidChannels<0 implies countMid==0, which let's us assume
  // that countMid*nMidChannels == 0.
  if (count != nullptr) {
    *count = work->cbd.countLo + work->cbd.countMid*nMidChannels + work->cbd.countHi;
  }
  if (channelId == work->channelLo) {
    *partOffset = 0;
    *partCount = work->cbd.countLo;
    *chunkCount = work->cbd.chunkGrainsLo*eltPerGrain;
  } else if (channelId == work->channelHi) {
    *partOffset = work->cbd.countLo + nMidChannels*work->cbd.countMid;
    *partCount = work->cbd.countHi;
    *chunkCount = work->cbd.chunkGrainsHi*eltPerGrain;
  } else {
    int mid = channelId - work->channelLo - 1;
    *partOffset = work->cbd.countLo + mid*work->cbd.countMid;
    *partCount = work->cbd.countMid;
    *chunkCount = work->cbd.chunkGrainsMid*eltPerGrain;
  }
}

struct alignas(16) ncclDevWorkCollReg {
  struct ncclDevWorkColl coll;
  void* dnInputs[NCCL_MAX_DIRECT_ARITY+1];
  void* dnOutputs[NCCL_MAX_DIRECT_ARITY+1];
  void* upOutputs[NCCL_MAX_DIRECT_ARITY+1];
};

enum ncclDevWorkType: uint8_t {
  ncclDevWorkTypeP2p,
  ncclDevWorkTypeColl,
  ncclDevWorkTypeCollReg
};

constexpr size_t ncclDevWorkSize(enum ncclDevWorkType type) {
  return type == ncclDevWorkTypeP2p ? sizeof(ncclDevWorkP2p) :
         type == ncclDevWorkTypeColl ? sizeof(ncclDevWorkColl) : sizeof(ncclDevWorkCollReg);
}

#define NCCL_MAX_DEV_WORK_BATCH_BYTES 1024
#define NCCL_MAX_DEV_WORK_BATCH_COLLS (NCCL_MAX_DEV_WORK_BATCH_BYTES/sizeof(ncclDevWorkColl))
#define NCCL_MAX_DEV_WORK_P2P_PER_BATCH 8
struct alignas(16) ncclDevWorkBatch {
  union {
    struct {
      // nextExtends: should next one be merged into this one.
      // nextJump=0: end of this channel's batch list
      // nextJump>0: batches[thisIndex+nextJump] is next batch in this list
      uint32_t nextJump:14, nextExtends:1;
      uint32_t workType:2, funcId:15;
    };
    // Unioning bitfields with underlying type hints compiler to emit the best
    // SASS LD/ST accesses.
    uint32_t flags;
  };
  // Rolling offset in fifo where this batch's work structs begin
  uint32_t offsetBase;
  // Set of relative offsets from offsetBase for this channel's subset of the batch:
  // For each bit index i in offsetMask, find work at fifo offset: offsetBase + i*sizeof(WorkStructType)
  uint64_t offsetBitset;
};

struct ncclDevChannelPeer {
  // Stripped version of ncclChannelPeer where we only keep the ncclConnInfo
  // instead of the full ncclConnector.
  struct ncclConnInfo send[NCCL_MAX_CONNS];
  struct ncclConnInfo recv[NCCL_MAX_CONNS];
};

struct alignas(16) ncclDevChannel {
  struct ncclDevChannelPeer** peers;
  struct ncclRing ring;
  struct ncclTree tree;
  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;
  struct ncclNvls nvls;
  uint32_t* workFifoDone; // Location of done counter, device writes index+1 of last work processed
};

struct ncclDevComm {
  int rank;
  int nRanks;
  int node;
  int nNodes;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  int p2pChunkSize;
  int isAllNvlink;

  // Work fifo return credits
  uint32_t* workConsumed/*[MAXCHANNELS]*/;

  int* collNetDenseToUserRank;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t* abortFlag;

  // Channels, device side
  struct ncclDevChannel* channels/*[MAXCHANNELS]*/;
  int* rankToLocalRank;
};

struct alignas(16) ncclDevCommAndChannels {
  struct ncclDevComm comm;
  struct ncclDevChannel channels[MAXCHANNELS];
};

enum ncclDevWorkStorageType: uint8_t {
  ncclDevWorkStorageTypeArgs=0,
  ncclDevWorkStorageTypeFifo=1,
  ncclDevWorkStorageTypePersistent=2
};

struct alignas(16) ncclDevKernelArgs {
  struct ncclDevComm* comm;
  uint64_t channelMask;
  enum ncclDevWorkStorageType workStorageType;
  uint32_t workMask;
  void* workBuf;
  // A channel's first batch is at `blockIdx.x`. Use `nextJump` to follow rest of list.
  // struct ncclDevWorkBatch batches[];
};

__host__ __device__ constexpr int ncclMaxKernelArgsSize(/*int cudaDriver, */int cudaArch=NCCL_CUDA_ARCH) {
  //return (cudaArch < 700 || cudaDriver < 12010) ? 4<<10 : (32<<10)-4;
  return 4<<10;
}

template<size_t capacity>
struct alignas(16) ncclDevKernelArgsStorage {
  union {
    struct ncclDevKernelArgs args;
    ulong2 storage[capacity/sizeof(ulong2)];
  };
};

typedef ncclDevKernelArgsStorage<(4<<10)> ncclDevKernelArgs4K;
//typedef ncclDevKernelArgsStorage<(32<<10)-4> ncclDevKernelArgs31K;

template<typename T>
__host__ __device__ constexpr T min_constexpr(T a) { return a; }
template<typename T, typename ...Ts>
__host__ __device__ constexpr T min_constexpr(T a, T b, Ts ...c) {
  return min_constexpr<T>((a < b ? a : b), c...);
}

template<typename T>
__host__ __device__ constexpr T max_constexpr(T a) { return a; }
template<typename T, typename ...Ts>
__host__ __device__ constexpr T max_constexpr(T a, T b, Ts ...c) {
  return max_constexpr<T>((a > b ? a : b), c...);
}

constexpr int ncclDevMaxChannelsForArgsBytes(size_t argsBytes) {
  return min_constexpr<size_t>(MAXCHANNELS, (argsBytes - sizeof(struct ncclDevKernelArgs))/sizeof(struct ncclDevWorkBatch));
}

// Calculate the unroll factor given:
// * bytePerPack: number of bytes accessed per instruction
// * insns: max permissible unroll value
// * bytes: desired number of in-flight bytes per iteration ( = unroll*bytePerPack)
__host__ __device__ constexpr int ncclCalcUnroll(int bytePerPack, int insns, int bytes) {
  return min_constexpr(insns, (bytes + bytePerPack-1)/bytePerPack);
}

// Note that all unroll value logic should depend on a given cudaArch argument
// and not __CUDA_ARCH__ since these need to be host-side executable where the
// arch value is strictly runtime only. By defaulting to NCCL_CUDA_ARCH, device
// side code can elide passing the arch for brevity.

__host__ __device__ constexpr int ncclCollUnroll(int cudaArch = NCCL_CUDA_ARCH) {
  // Our collective unroll should move to the same bytes&insns model as NVLS.
  return cudaArch >= 800 ? 8 : 4;
}

__host__ __device__ constexpr int ncclNvlsUnrollBytes(int cudaArch = NCCL_CUDA_ARCH) { return 4*16; }
__host__ __device__ constexpr int ncclNvlsUnrollInsns(int cudaArch = NCCL_CUDA_ARCH) { return 16; }

__host__ __device__ constexpr int ncclNvlsUnroll(int bytePerPack, int cudaArch = NCCL_CUDA_ARCH) {
  return ncclCalcUnroll(bytePerPack, ncclNvlsUnrollInsns(cudaArch), ncclNvlsUnrollBytes(cudaArch));
}

// The amount of dynamic shmem per warp
__host__ __device__ constexpr int ncclShmemScratchWarpSize(int cudaArch = NCCL_CUDA_ARCH) {
  return (max_constexpr<int>(
      /*LL    */0,
      /*LL128 */(NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE)*sizeof(uint64_t),
      /*SIMPLE*/(ncclCollUnroll(cudaArch)*WARP_SIZE + 1)*16,
      // NVLS needs an extra 16B to read unaligned data.
      /*NVLS  */WARP_SIZE*(cudaArch >= 900 ? ncclNvlsUnrollBytes(cudaArch) : 0) + 16
    ) + 15) & -16; // pad to 16 bytes
}

// The amount of dynamic shmem per block
__host__ __device__ constexpr int ncclShmemDynamicSize(int cudaArch = NCCL_CUDA_ARCH) {
  return cudaArch < 700 ? 0 : ncclShmemScratchWarpSize(cudaArch)*(NCCL_MAX_NTHREADS/WARP_SIZE);
}

// Host-side table of kernel function pointers.
extern int const ncclDevKernelCount;
extern void* const ncclDevKernelList[/*ncclDevKernelCount*/];

// Table of most specialized kernel function to run given func index.
extern int const ncclDevFuncIdCount;
extern int const ncclDevFuncRowToId[];
extern void* const ncclDevKernelForFunc[/*funcIndex*/];
extern bool const ncclDevKernelForFuncIsSpecialized[/*funcIndex*/];

// Launch a one-rank reduction on stream.
ncclResult_t ncclLaunchOneRank(void* dst, void const* src, size_t nElts, struct ncclDevRedOpFull redOp, ncclDataType_t type, cudaStream_t stream);

// `ncclNvlsSupported()` needs to be in sync with "func_valid" in "src/device/generate.py"
inline bool ncclNvlsSupported(int devRedOp, int type) {
  switch (type) {
  case ncclInt32:
  case ncclUint32:
  case ncclInt64:
  case ncclUint64:
  case ncclFloat16:
  case ncclBfloat16:
    return devRedOp == ncclDevSum || devRedOp == ncclDevMinMax;
  case ncclFloat:
  case ncclDouble:
    return devRedOp == ncclDevSum;
  default:
    return false;
  }
}

// `ncclDevFuncIndex()` needs to be in sync with "all_functions()" in "src/device/generate.py"
inline int ncclDevFuncId(int coll, int devRedOp, int type, int algo, int proto) {
  #if defined(__CUDA_BF16_TYPES_EXIST__)
  constexpr int NumTypes = ncclNumTypes;
  #else
  constexpr int NumTypes = ncclNumTypes + 1;
  #endif
  int row;
  do {
    row = 0; // ncclDevFuncIndex_P2p
    if (coll == ncclFuncSendRecv) break;
    row += 1;

    int nAlgos = 4;
    if (coll == ncclFuncAllGather) {
      int algo1 = algo == NCCL_ALGO_RING ? 0 :
                  algo == NCCL_ALGO_COLLNET_DIRECT ? 1 :
                  algo == NCCL_ALGO_NVLS ? 2 :
                /*algo == NCCL_ALGO_PAT*/ 3;
      row += algo1*NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += nAlgos*NCCL_NUM_PROTOCOLS;

    nAlgos = 1;
    if (coll == ncclFuncBroadcast) {
      row += proto;
      break;
    }
    row += nAlgos*NCCL_NUM_PROTOCOLS;

    nAlgos = 6;
    if (coll == ncclFuncAllReduce) {
      row += ((devRedOp*NumTypes + type)*nAlgos + algo)*NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += ncclNumDevRedOps*NumTypes*nAlgos*NCCL_NUM_PROTOCOLS;

    nAlgos = 1;
    if (coll == ncclFuncReduce) {
      row += (devRedOp*NumTypes + type)*NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += ncclNumDevRedOps*NumTypes*nAlgos*NCCL_NUM_PROTOCOLS;

    nAlgos = 4;
    if (coll == ncclFuncReduceScatter) {
      int algo1 = algo == NCCL_ALGO_RING ? 0 :
                  algo == NCCL_ALGO_COLLNET_DIRECT ? 1 :
                  algo == NCCL_ALGO_NVLS ? 2 :
                /*algo == NCCL_ALGO_PAT*/ 3;
      row += ((devRedOp*NumTypes + type)*nAlgos + algo1)*NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += ncclNumDevRedOps*NumTypes*nAlgos*NCCL_NUM_PROTOCOLS;
  } while (false);

  return ncclDevFuncRowToId[row];
}

inline int ncclDevFuncId_P2p() { return ncclDevFuncRowToId[0]; }

#endif
