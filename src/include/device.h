/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "nccl_common.h"
#include "align.h"
#include <stdint.h>

extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS];

extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];

extern const char* ncclProtoStr[NCCL_NUM_PROTOCOLS];

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

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
#define MAXCHANNELS 32
#define NCCL_MAX_NTHREADS 640
#define NCCL_SIMPLE_MAX_NTHREADS 512
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

#define NCCL_DIRECT_WRITE 0x01
#define NCCL_DIRECT_READ  0x02
#define NCCL_DIRECT_NIC   0x04
#define NCCL_IPC_WRITE    0x08
#define NCCL_IPC_READ     0x10
#define NCCL_NVLS_MIN_POLL 0x20

struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int flags;          // Direct communication / other flags
  int shared;         // Buffers are shared
  void **ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  struct ncclConnFifo* connFifo; // Used for GPU - Proxy communication

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
  ncclNetDeviceHandle_t netDeviceHandle;
};

struct ncclProxyConnector {
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

#define NCCL_MAX_NVLS_ARITY 8
#define NCCL_MAX_NVLS_TREE_ARITY 3
struct ncclNvls {
  int out;
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int up[NCCL_MAX_NVLS_ARITY];
  int down;
  int treeUp;
  int treeDown[NCCL_MAX_NVLS_TREE_ARITY];
  int node;
  int nNodes;
};

#define NCCL_MAX_CONNS 2
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];
  struct ncclConnector recv[NCCL_MAX_CONNS];
  int refCount;
};

struct ncclDevComm;

/* ncclWork is to be a power of two, currently 8x64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclWorkElem. */
#define NCCL_WORK_SIZE 512

enum ncclWorkType : uint8_t {
   ncclWorkTypeUnused=0,
   ncclWorkTypeColl=1,
   ncclWorkTypeP2p=2,
   ncclWorkTypeRegColl=3
};
enum ncclWorkP2PType : uint8_t {
  ncclWorkP2pTypeUnused=0,
  ncclWorkP2pTypeSend,
  ncclWorkP2pTypeRecv
};

struct ncclWorkHeader {
  union {
    int32_t workNext;  // when isLast=0: Offset from kernel argument workHead
    uint32_t doneAcks; // when isLast=1: Monotonic (mod 1<<32) ack value to send back.
  };
  uint16_t funcIndex;
  uint8_t isLast:1; // last work for this kernel
  uint8_t inFifo:1; // is this work in the fifo
  enum ncclWorkType type;
};

struct ncclWorkElem {
  union {
    uint8_t flagBits;
    struct {
      uint8_t isUsed:1, redOpArgIsPtr:1, regUsed:1, oneNode:1;
    };
  };
  uint8_t nWarps;
  uint8_t direct;
  uint32_t root;
  const void *sendbuff;
  void *recvbuff;

  size_t count;
  uint64_t redOpArg;
  uint64_t chunkCount:25, workCount:39;
  union {
    struct {
      uint64_t lastChunkCount:25;
      uint64_t workOffset:39;
    };
    struct {
      uint64_t bid:32;
      uint64_t nChannels:32;
    };
  };
};

#define NCCL_MAX_WORK_ELEMENTS ((NCCL_WORK_SIZE - alignUp(sizeof(ncclWorkHeader), alignof(ncclWorkElem)))/sizeof(ncclWorkElem))
static_assert(NCCL_MAX_WORK_ELEMENTS == 9, "Sanity check: NCCL_MAX_WORK_ELEMENTS == 9");

struct ncclWorkElemP2p {
  int peer : 30;
  int proto : 2;

  enum ncclWorkP2PType p2pType;
  uint8_t reg:1;
  uint8_t nWarps:5;
  uint8_t warpStart;
  uint8_t ngroups;
  // Important not to use any fields with greater than 4-byte alignment since
  // we need sizeof(ncclWorkElemP2p)==28, but that would be padded up to 32 if
  // there were 8-byte fields.
  //void* buff;
  uint32_t buffHi32, buffLo32; // buff = buffHi32<<32 | buffLo32;
  //size_t count;
  uint32_t countHi32, countLo32; // count = countHi32<<32 | countLo32;
  int chunkSize;
};

static_assert(((NCCL_WORK_SIZE - alignUp(sizeof(ncclWorkHeader), alignof(ncclWorkElemP2p)))/sizeof(ncclWorkElemP2p)) >= 16, "Sanity check: NCCL_MAX_WORK_ELEMENTS_P2P == 16");
#define NCCL_MAX_WORK_ELEMENTS_P2P 16

struct ncclWorkElemReg {
  struct ncclWorkElem elem;
  void* dnInputs[NCCL_MAX_DIRECT_ARITY+1];
  void* dnOutputs[NCCL_MAX_DIRECT_ARITY+1];
  void* upOutputs[NCCL_MAX_DIRECT_ARITY+1];
};

#define NCCL_MAX_WORK_ELEMENTS_REG ((NCCL_WORK_SIZE - alignUp(sizeof(ncclWorkHeader), alignof(ncclWorkElemReg)))/sizeof(ncclWorkElemReg))
static_assert(NCCL_MAX_WORK_ELEMENTS_REG == 2, "Sanity check: NCCL_MAX_WORK_ELEMENTS_REG == 2");

// Number of named barriers supported by CUDA
#define NCCL_MAX_GROUPS 16

struct ncclWork {
  struct ncclWorkHeader header;
  union {
    char pad[NCCL_WORK_SIZE - sizeof(struct ncclWorkHeader)];
    struct ncclWorkElem elems[NCCL_MAX_WORK_ELEMENTS];
    struct ncclWorkElemP2p p2pElems[NCCL_MAX_WORK_ELEMENTS_P2P];
    struct ncclWorkElemReg regElems[NCCL_MAX_WORK_ELEMENTS_REG];
  };
};
static_assert(sizeof(struct ncclWork) == NCCL_WORK_SIZE, "Sanity check: sizeof(struct ncclWork) == NCCL_WORK_SIZE");
static_assert(sizeof(struct ncclWork)%16 == 0, "Sanity check: sizeof(struct ncclWork)%16 == 0");

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

  // Operation list for aggregation
  int workFifoDepth;
  struct ncclWork* workFifoHeap; // may be cudaHost or GDR memory

  int* collNetDenseToUserRank;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t* abortFlag;

  // Channels, device side
  struct ncclDevChannel* channels/*[MAXCHANNELS]*/;
};

struct alignas(16) ncclDevCommAndChannels {
  struct ncclDevComm comm;
  struct ncclDevChannel channels[MAXCHANNELS];
};

#ifdef __CUDA_ARCH__
  #define NCCL_CUDA_ARCH __CUDA_ARCH__
#else
  #define NCCL_CUDA_ARCH 0
#endif

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
  #if defined(__CUDA_BF16_TYPES_EXIST__)
  case ncclBfloat16:
  #endif
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

    int nAlgos = 3;
    if (coll == ncclFuncAllGather) {
      int algo1 = algo == NCCL_ALGO_RING ? 0 :
                  algo == NCCL_ALGO_COLLNET_DIRECT ? 1 :
                /*algo == NCCL_ALGO_NVLS*/ 2;
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

    nAlgos = NCCL_NUM_ALGORITHMS;
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

    nAlgos = 3;
    if (coll == ncclFuncReduceScatter) {
      int algo1 = algo == NCCL_ALGO_RING ? 0 :
                  algo == NCCL_ALGO_COLLNET_DIRECT ? 1 :
                /*algo == NCCL_ALGO_NVLS*/ 2;
      row += ((devRedOp*NumTypes + type)*nAlgos + algo1)*NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += ncclNumDevRedOps*NumTypes*nAlgos*NCCL_NUM_PROTOCOLS;
  } while (false);

  return ncclDevFuncRowToId[row];
}

inline int ncclDevFuncId_P2p() { return ncclDevFuncRowToId[0]; }

#endif
