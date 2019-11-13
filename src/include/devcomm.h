/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include <stdint.h>

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

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
#define NCCL_MAX_NTHREADS 512
#define NCCL_LL_MAX_NTHREADS NCCL_MAX_NTHREADS
#define NCCL_LL_LINES_PER_THREAD 8
#define NCCL_LL_SLICE_LINES (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS)
#define NCCL_LL_BUFF_LINES (NCCL_LL_SLICE_LINES*NCCL_STEPS)
#define NCCL_LL_BUFF_SIZE (NCCL_LL_BUFF_LINES*sizeof(union ncclLLFifoLine))
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

// Receiving from up to 3 sources is more compute intensive than sending
// to 3 dests. Use 70% for reduce and 30% for bcast.
#define NCCL_LL128_SPLIT(nt) ((nt*7/(10*32))*32)

#define NCCL_LL128_SLICE_ELEMS (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)
#define NCCL_LL128_BUFF_ELEMS (NCCL_LL128_SLICE_ELEMS*NCCL_STEPS)
#define NCCL_LL128_BUFF_SIZE (NCCL_LL128_BUFF_ELEMS*sizeof(uint64_t))

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)

struct ncclConnInfo {
  // Regular comm mechanism
  char *buff;         // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv
  uint64_t *opCountLoc; // opCount of local rank
  uint64_t *opCountRem; // opCount of remote rank

  int direct;         // Direct communication
  void **ptrExchange; // Pointer exchange for direct communication

  int *fifo;          // Size fifo for proxy

  uint64_t step;      // Keep where we are

  // Low latency mechanism
  union ncclLLFifoLine *llBuff; // Local for recv, remote for send
  uint64_t llLastCleaning;

  // High bandwidth, low latency protocol
  uint64_t* ll128Buff; // Local for recv, remote for send
};

struct ncclConnector {
  int connected;
  struct ncclProxyArgs *proxyAppend;
  struct ncclTransportComm* transportComm;
  void* transportResources; // Host-side resources
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
};


#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

struct ncclPeer {
  struct ncclConnector send;
  struct ncclConnector recv;
};

struct ncclDevComm;

/* CollectiveArgs + ncclColl are to be a power of two, currently 64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclColl. */
struct CollectiveArgs {
  struct ncclDevComm* comm;
  uint64_t opCount;

  // local and remote input, output, and buffer
  const void * ThisInput;
  void * ThisOutput;

  // general parameters
  size_t N;
  uint32_t root;
  uint8_t bid;
  uint8_t nChannels;
  uint16_t nThreads;

  int lastChunkSize;
};
struct ncclColl {
  union {
    struct {
      struct CollectiveArgs args;
      uint16_t funcIndex;
      uint16_t nextIndex;
      uint8_t  active;
    };
    int data[0x10];
  };
};
static_assert(sizeof(struct ncclColl) == (0x10*sizeof(int)), "ncclColl must have a pow2 size");

struct ncclChannel {
  union {
    struct {
      struct ncclRing ring;
      struct ncclTree treeUp;
      struct ncclTree treeDn;

      int id;
      int nthreads;
      int buffSize;

      // Communication structures
      struct ncclPeer* peers;
      struct ncclPeer* devPeers;

      // Operation list for aggregation
      struct ncclColl* collectives;
      struct ncclColl* devCollectives;
      int collStart;
      int collCount;
      int collFifoHead; // Only used by GPU
      int collFifoTail; // Only used by CPU
    };
    int data[0x80];
  };
};
static_assert(sizeof(struct ncclChannel) == 0x80*sizeof(int), "ncclChannel must have a pow2 size");

typedef enum {
  ncclDevSuccess,
  ncclDevAssertedMismatch,
  ncclDevSuspectedMismatch
} ncclDevError_t;

struct ncclDevComm {
  int rank;
  int nRanks;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;
  volatile ncclDevError_t *fatalDevError;

  // Channels, device side
  struct ncclChannel* channels;
};

#endif
