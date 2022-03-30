/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "align.h"
#include <stdint.h>

#define NCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum { ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce, ncclFuncSendRecv, ncclFuncSend, ncclFuncRecv, ncclNumFuncs} ncclFunc_t;
extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS];

#define NCCL_NUM_ALGORITHMS 3 // Tree/Ring/CollNet
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET 2
extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2
extern const char* ncclProtoStr[NCCL_NUM_PROTOCOLS];

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

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

struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int direct;         // Direct communication
  int shared;         // Buffers are shared
  void **ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  int *sizesFifo;     // Sizes fifo from GPU to proxy
  int *offsFifo;      // Buffer fifo from proxy to GPU

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
};

struct ncclProxyConnector {
  int rank;
  int localRank;
  struct ncclProxyConnection* connection;
  struct ncclComm* comm;
};

struct ncclConnector {
  int connected;
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;
  void* transportResources;
  struct ncclConnInfo conn;
  struct ncclComm *comm;
};

struct ncclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;
  int* devUserRanks;

  int index; // This rank's index in the ring
};


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
  int nHeads;
  int headRank;
  int shift;
  int up[NCCL_MAX_DIRECT_ARITY];
  int down[NCCL_MAX_DIRECT_ARITY];
};

#define NCCL_MAX_CONNS 2
struct ncclPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];
  struct ncclConnector recv[NCCL_MAX_CONNS];
};

struct ncclDevComm;

/* ncclWork is to be a power of two, currently 8x64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclWorkElem. */
#define NCCL_WORK_SIZE 512

enum ncclWorkElemType : uint8_t {
   ncclWorkTypeUnused=0,
   ncclWorkTypeColl=1,
   ncclWorkTypeP2p=2,
   ncclWorkTypeRegColl=3
};
enum ncclWorkElemSubType : uint8_t {
  ncclWorkSubTypeUnused =0,
  ncclWorkSubTypeSend,
  ncclWorkSubTypeRecv
};

struct ncclWorkElemHeader {
  uint16_t funcIndex;
  enum ncclWorkElemType type;
  unsigned nWarps:5;
  unsigned isLast:1;
};

struct ncclWorkElem {
  struct ncclWorkElemHeader header;
  uint8_t regUsed;
  uint8_t direct;
  uint8_t redOpArgIsPtr;

  const void * sendbuff;
  void * recvbuff;

  size_t count;
  size_t lastChunkSize;
  uint32_t root;
  uint8_t bid;
  uint8_t nChannels;
  uint64_t redOpArg;
  uint64_t pad;
};
static_assert(NCCL_WORK_SIZE % sizeof(struct ncclWorkElem) == 0, "ncclWorkElem size must be a multiple of ncclWork size");

struct ncclWorkElemP2p {
  struct ncclWorkElemHeader header;
  int32_t peer;
  void* buff;
  size_t count;
  int chunkSize;
  uint8_t ngroups;
  uint8_t warpStart;
  uint8_t nWarps;
  enum ncclWorkElemSubType subType;
};
static_assert(NCCL_WORK_SIZE % sizeof(struct ncclWorkElemP2p) == 0, "ncclWorkElemP2p size must be a multiple of ncclWork size");

struct ncclWorkElemReg {
  struct ncclWorkElem elem;
  void* dnInputs[NCCL_MAX_DIRECT_ARITY+1];
  void* dnOutputs[NCCL_MAX_DIRECT_ARITY+1];
  void* upOutputs[NCCL_MAX_DIRECT_ARITY+1];
};
static_assert(NCCL_WORK_SIZE % sizeof(struct ncclWorkElemReg) == 0, "ncclWork size must be a multiple of ncclWorkElemReg size");
static_assert(sizeof(struct ncclWorkElemReg) % sizeof(struct ncclWorkElem) == 0, "ncclWorkElemReg size must be a multiple of ncclWorkElem size");

#define NCCL_MAX_WORK_ELEMENTS (NCCL_WORK_SIZE/sizeof(struct ncclWorkElem))
#define NCCL_MAX_WORK_ELEMENTS_P2P (NCCL_WORK_SIZE/sizeof(struct ncclWorkElemP2p))
#define NCCL_MAX_WORK_ELEMENTS_REG (NCCL_WORK_SIZE/sizeof(struct ncclWorkElemReg))
// Number of named barriers supported by CUDA
#define NCCL_MAX_GROUPS 16

struct ncclWork {
  union {
    char pad[NCCL_WORK_SIZE];
    struct ncclWorkElemHeader header;
    struct ncclWorkElem elems[NCCL_MAX_WORK_ELEMENTS];
    struct ncclWorkElemP2p p2pElems[NCCL_MAX_WORK_ELEMENTS_P2P];
    struct ncclWorkElemReg regElems[NCCL_MAX_WORK_ELEMENTS_REG];
  };
};

static_assert(sizeof(struct ncclWork) == NCCL_WORK_SIZE, "ncclWork size needs to be well aligned");

struct ncclChannel {
  union {
    struct {
      struct ncclRing ring;
      struct ncclTree tree;
      struct ncclDirect collTree;

      int id;

      // Communication structures
      struct ncclPeer* peers;
      struct ncclPeer* devPeers;

      // Operation list for aggregation
      struct ncclWork* workFifo;
      int workCount;
      size_t totalSize;
      uint64_t workFifoTail; // Only used by CPU
      uint16_t index;        // Only used by GPU

      // GDRCOPY support
      struct ncclWork* workFifoGdr;
      struct ncclWork* workFifoDev;
      void* gdrMemDesc;
    };
    int data[0x80];
  };
};
static_assert(sizeof(struct ncclChannel) == 0x80*sizeof(int), "ncclChannel must have a pow2 size");

struct ncclDevComm {
  int rank;
  int nRanks;
  int buffSizes[NCCL_NUM_PROTOCOLS];

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;

  // Channels, device side
  struct ncclChannel* channels;
};

struct ncclDevCommAndChannels {
  ncclDevComm comm;
  ncclChannel channels[MAXCHANNELS];
};

#endif
