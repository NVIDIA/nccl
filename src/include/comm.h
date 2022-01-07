/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

#include "transport.h"
#include "p2p.h"
#include "collectives.h"

#if CUDART_VERSION < 9000
struct cudaLaunchParams {
  void *func;
  dim3 gridDim;
  dim3 blockDim;
  void **args;
  size_t sharedMem;
  cudaStream_t stream;
};
#endif

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096
#define CUDA_IPC_MIN 2097152UL

// Channels / LL tuning
#define NCCL_LL_THREAD_THRESHOLD 8
#define NCCL_LL128_THREAD_THRESHOLD 8
#define NCCL_SIMPLE_THREAD_THRESHOLD 64

struct ncclSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      void* ptrExchange;
      uint64_t redOpArgExchange[2];
      char pad2[CACHE_LINE_SIZE-sizeof(void*)-2*sizeof(uint64_t)];
      int offsFifo[NCCL_STEPS];
    };
    char pad3[MEM_ALIGN];
  };
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      int sizesFifo[NCCL_STEPS];
      int offsFifo[NCCL_STEPS];
      int flush; // For GDRCopy-based flush
    };
    char pad4[MEM_ALIGN];
  };
};

typedef cudaError_t(*pfn_cuMemGetAddressRange_t)(void**, size_t*, void*);

enum helperThreadState {ThreadStart, ThreadStop};

#define NCCL_IPC_POOL_SIZE (2*NCCL_MAX_LOCAL_RANKS*NCCL_MAX_OPS)

struct ncclGraphHelperResources {
  ncclComm* comm;
  pthread_mutex_t threadLock;
  pthread_cond_t  threadCond;
  enum helperThreadState threadState;
  void* ipcBases[NCCL_IPC_POOL_SIZE];
  int ipcTail;
  int ipcHead;
};

struct ncclUserRedOp {
  int freeNext; // -1=allocated, otherwise index of next free entry in array
  ncclDataType_t datatype;
  ncclDevRedOpFull opFull;
};

struct ncclNodeRanks {
  int localRanks;
  int* localRankToRank;
};

struct ncclComm {
  struct ncclChannel channels[MAXCHANNELS];

  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;

  void* bootstrap;
  // Bitmasks for ncclTransportP2pSetup
  int connect;
  uint32_t* connectSend;
  uint32_t* connectRecv;

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
  int64_t busId;   // my PCI bus ID in int format
  cpu_set_t cpuAffinity; // CPU affinity of the GPU

  int node;
  int nNodes;
  int localRank;
  int localRanks;
  int maxLocalRanks;
  int* rankToNode;
  int* rankToLocalRank;
  int* localRankToRank;
  // localRanks and localRanktoRank for all nodes
  struct ncclNodeRanks* nodeRanks;

  enum { GROUP, PARALLEL, GROUP_GRAPH } launchMode;
  cudaStream_t userStream;
  bool userStreamSet;
  cudaEvent_t doneEvent;
  cudaEvent_t intDoneEvent;
  bool checkPointers;

  // Counter for tracking CUDA launches (P2P and collectives included)
  uint64_t opCount;
  // Collective operation counter
  uint64_t collOpCount;

  // Channels for collectives
  int nChannels;
  // Channels (per peer) for p2p
  int p2pnChannels;
  int p2pnChannelsPerPeer;
  int p2pChannels[MAXCHANNELS];

  // Buffer sizes
  int buffSizes[NCCL_NUM_PROTOCOLS];

  // Algorithm/Protocols thresholds
  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  // An internal CUDA stream for NCCL kernel CGMD launches
  int groupCudaStream;
  cudaStream_t groupStream;

  // Whether there has been a fatal error in this communicator.
  ncclResult_t fatalError;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;

  // Device side of the communicator
  struct ncclDevComm *devComm;
  // Host copy of the devComm (to free CUDA allocs)
  struct ncclDevComm hostDevComm;

  // Intra-process sync
  int intraRank;
  int intraRanks;
  int* intraBarrier;
  int intraPhase;

  // Storage for deferred intra-process launch
  struct cudaLaunchParams * intraParams;
  struct cudaLaunchParams *myParams;
  pthread_t* intraThreads;
  int* intraCudaDevs;
  int* intraCGMode; // Whether we can use CUDA9 CGMD or not
  int* intraCC; // Only to check all have the same ComputeCap and disable CGMode if not
  struct ncclWorkElem args;
  void* argsptrs[2];

  struct ncclProxyState proxyState;

  // Whether this communicator uses collNet
  int collNetSupport;
  int intraHighestTransportType;

  // Store info of async operations
  struct ncclInfo* asyncOps;
  int asyncOpCount;
  size_t asyncTotalSize;
  ssize_t channelSize;
  int lastChannel;
  enum { ROUND_ROBIN, SHORTEST_QUEUE } asyncAllocMode;

  //list of async p2p operation queued in a group semantics
  ncclP2Plist** p2pSends;
  ncclP2Plist** p2pRecvs;
  int p2pSendCount;
  int p2pRecvCount;

  // Store info for cudaGraph
  int usingCudaGraph; // Only use it during capture time, not launch time
  struct ncclQueueInfo* enqueueInfo;
  int nQueueInfoCreated;
  int nQueueInfoDestroyed;
  cudaGraphNode_t lastSetupNode;
  unsigned long long lastCudaGraphId;
  int driverVersion;
  pfn_cuMemGetAddressRange_t pfnCuMemGetAddressRange;
  pthread_t graphHelperThread;
  struct ncclGraphHelperResources* graphHelperResources;
  int disableGraphHelper;
  int graphRegister;

  // user-created reduction ops
  int userRedOpCapacity, userRedOpFreeHead;
  ncclUserRedOp *userRedOps;
};

// Scrambles the bits of non-builtin values of ncclRedOp_t according to the
// communicator memory address. Used to catch bugs so that integer handles
// associated with this communicator won't collide with handles of other
// communicatrs. This function is its own inverse.
static inline ncclRedOp_t ncclUserRedOpMangle(ncclComm *comm, ncclRedOp_t op) {
  // Preserve the built-in values.
  if(int(op) < int(ncclNumOps))
    return op;
  uint64_t h = reinterpret_cast<uint64_t>(comm);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13u; // Knuth's 64-bit magical hash constant
  h >>= 32; // h is now an excellent 32-bit hash of the comm pointer
  h &= int(ncclMaxRedOp); // ncclMaxRedOp is a power of 2 minus 1
  int op1 = int(h) ^ int(op);
  // Since builtin values are preserved, we also have to preserve their preimage.
  return op1 < int(ncclNumOps) ? op : ncclRedOp_t(op1);
}

#endif
