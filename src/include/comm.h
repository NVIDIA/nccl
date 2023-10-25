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
#include "nccl_tuner.h"
#include "proxy.h"
#include "strongstream.h"
#include "nccl_net.h"

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

struct ncclDestructor {
  struct ncclDestructor* next;
  void* obj;
  ncclResult_t(*fn)(struct ncclDestructor* me);
};

struct ncclCommCallback {
  struct ncclCommCallback* next;
  ncclResult_t(*fn)(struct ncclComm* comm, struct ncclCommCallback* cb);
};

struct ncclSharedResources {
  int refCount;
  struct ncclComm* owner; /* comm which creates this shared res. */
  struct ncclChannelPeer* peers[MAXCHANNELS];
  struct ncclDevChannelPeer* devPeers[MAXCHANNELS];
  /* P2P operation counter, one per channel */
  uint64_t p2pOpCount[MAXCHANNELS];
  /* Collective operation counter */
  uint64_t collOpCount;
  int tpNRanks;
  int tpNLocalRanks;
  int tpNChannels;
  int tpP2pNChannels;
  int tpP2pChunkSize;
  uint64_t magic;

  // top parent rank to localRank translation table
  int* tpRankToLocalRank;
  // Internal streams
  struct ncclStrongStream deviceStream, hostStream;

  /* proxy related shared res */
  struct ncclProxyState* proxyState;
};

struct ncclChannel {
  struct ncclChannelPeer** peers;
  struct ncclDevChannelPeer** devPeers;
  /* devPeer pointer array used for host side access */
  struct ncclDevChannelPeer** devPeersHostPtr;
  struct ncclRing ring;
  int* devRingUserRanks;
  struct ncclTree tree;

  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;

  struct ncclNvls nvls;

  int id; // index of this channel
  uint32_t workFifoSent; // last used work index+1

  /* comm split sharable resources */
  struct ncclChannelPeer* collnetPeers;
  struct ncclDevChannelPeer* collnetDevPeers;
  struct ncclChannelPeer* nvlsPeers;
  struct ncclDevChannelPeer* nvlsDevPeers;
};

struct ncclWorkList {
  struct ncclWorkList* next;
  struct ncclWork work;
};

struct ncclPointerList {
  struct ncclPointerList* next;
  void *ptr;
};

struct ncclNvlsMcHandleList {
  struct ncclNvlsMcHandleList *next;
  CUmemGenericAllocationHandle mcHandle;
  CUdeviceptr ptr;
  int dev;
  size_t size;
};

struct ncclKernelPlan {
  // A kernel plan is also a callback that reclaims itself. Hence this must
  // be the first member.
  struct ncclCommCallback reclaimer;
  struct ncclMemoryPool memPool_ncclProxyOp; // memory to return to comm in cleanup

  struct ncclComm* comm;
  struct ncclKernelPlan* next;

  bool persistent; // aka captured in a graph
  bool kernelSpecialized;
  void *kernelFn;
  int channelUbound; // only channels c < channelUbound are present
  int channelCount; // number of channels present
  uint64_t channelMask; // which channels are present, channelCount == popcount(channelMask)
  bool hasProxyOps; // does any channel have a non-empty proxyOpQueue
  int threadPerBlock;
  // workHeap fields are null until uploadWorkFifo() or preparePersistentKernel()
  struct ncclWork* workHead;

  int collOpCount; // zero based for this plan

  struct ncclIntruQueue<struct ncclPointerList, &ncclPointerList::next> ipcMemQueue;
  struct ncclIntruQueue<struct ncclNvlsMcHandleList, &ncclNvlsMcHandleList::next> nvlsMcHandleQueue;

  struct Channel {
    int nWork;
    union {
      int nWorkElem; // used for coll and reg coll
      int p2pTailElem[2]; // used for p2p, indexed by ncclWorkElemP2pType-1
    };
    size_t collBytes;
    struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workQueue;
    struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;
  } channels[MAXCHANNELS];
};

struct ncclRegRequest {
  uintptr_t buff;
  size_t size;
  struct ncclRegRequest *next;
};

struct ncclRegRecord {
  uintptr_t buff;
  size_t size;
  CUdeviceptr regAddr;
  size_t regSize;
  int dev;
  CUmemGenericAllocationHandle mcHandle;
  uintptr_t *addrs; /* use to check if NVLS buffers match among intra-node ranks */
  struct ncclRegRecord *next;
};

struct ncclComm {
  struct ncclMemoryStack memPermanent, memScoped;
  // List of destructors to run when comm is destructed
  struct ncclDestructor* destructorHead;

  struct ncclSharedResources* sharedRes;
  /* map to top parent ranks. */
  int* topParentRanks;
  int* topParentLocalRanks;
  struct ncclChannel channels[MAXCHANNELS];
  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;

  ncclNet_t* ncclNet;
  ncclCollNet_t* ncclCollNet;
  void* bootstrap;
  // Bitmasks for ncclTransportP2pSetup
  uint64_t* connectSend;
  uint64_t* connectRecv;

  uint64_t magic; // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.

  uint64_t commHash;
  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
  int nvmlDev; // my nvml device index
  int compCap; // compute capability of the GPU
  int minCompCap, maxCompCap; // min/max compute capability in the communicator
  int64_t busId;   // my PCI bus ID in int format
  cpu_set_t cpuAffinity; // CPU affinity of the GPU
  int cudaArch; // matches __CUDA_ARCH__ of device

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

  bool checkPointers;
  bool dmaBufSupport;

  // Counter for tracking CUDA launches (P2P and collectives included)
  uint64_t opCount;

  // Channels for collectives
  int nChannels;
  int nvlsChannels;
  int collNetChannels;
  // Channels (per peer) for p2p
  int p2pnChannels;
  int p2pnChannelsPerPeer;
  int p2pChannels[MAXCHANNELS];

  // Should this comm allocate LL buffers for network P2P connections?
  bool allocP2pNetLLBuffers;

  // Buffer sizes
  int buffSizes[NCCL_NUM_PROTOCOLS];
  int p2pChunkSize;

  // Algorithm/Protocols thresholds
  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float ringbdw[NCCL_NUM_FUNCTIONS][NCCL_NUM_PROTOCOLS];
  int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  /* This attribute can indicate the states of communicators and return code of
   * asynchronous NCCL operations. */
  ncclResult_t asyncResult;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;
  volatile uint32_t *childAbortFlag;
  volatile uint32_t *abortFlagRefCount;

  // Device side of the communicator (for cudaFree's)
  struct ncclDevComm* devComm; // actually = &ncclDevCommAndChannels::comm

  // Operation pool.
  int workFifoDepth; // size of workFifoHeap[], power of 2
  struct ncclWork* workFifoHeap;
  struct ncclWork* devWorkFifoHeap;
  void* workFifoHeapGdrHandle;

  // Work completion notificaion
  uint32_t* workFifoDone/*[MAXCHANNELS]*/; // in cudaHost memory
  uint32_t workFifoSent; // Monotonic (mod 1<<32) index of next unused fifo slot.
  uint32_t workFifoAckdMin; // Monotonic index of least unprocessed fifo slot over all channels.

  // Intra-process sync
  struct ncclComm* intraComm0; // leader of intra-process comms (self possible)
  struct ncclComm* intraNext; // next of intra-process comms, intraComm0 is head
  int intraRank;
  int intraRanks;
  uint32_t intraBarrierPhase;
  char intraPad1[64 - sizeof(uint64_t)];
  uint64_t intraBarrierCounter; // only used if this is intraComm0
  char intraPad2[64 - sizeof(uint64_t)];
  uint64_t intraBarrierGate; // only used if this is intraComm0

  struct ncclProxyState* proxyState;
  int proxyRefCountOld; /* store proxy post-atomic-sub refcount */
  // Whether this communicator uses collNet
  int collNetSupport;
  uint8_t collNetSupportMatrix[4/*sum,prod,min,max*/][ncclNumTypes];
  int intraHighestTransportType;
  int* collNetHeads;
  int collNetHeadsNum;
  /* sharable collNet proxy progress resource. */
  struct ncclCollNetSharedRes* collNetSharedRes;

  // NVLink SHARP (NVLS) support
  int nvlsSupport;
  int nvlsRegSupport;
  /* sharable NVLS resource. */
  struct ncclNvlsSharedRes* nvlsResources;
  struct ncclShmemCollBuff nvlsShmem;
  void *nvlsShmemHandle;

  ssize_t channelSize; // User requested work size (bytes) for channel partitions

  // pools backed by comm->memPermanent
  struct ncclMemoryPool memPool_ncclProxyOp;
  struct ncclMemoryPool memPool_ncclKernelPlan;
  struct ncclMemoryPool memPool_ncclPointerList;
  struct ncclMemoryPool memPool_ncclNvlsHandleList;
  // Next comm in this thread's active ncclGroup[Start|End](). Holds "0x1" when
  // this comm is not yet in a group.
  struct ncclComm* groupNext;
  // Subset of those in groupNext list. Holds 0x1 if not needing preconnect.
  struct ncclComm* preconnectNext;
  int persistentRefs; // number of persistent plan-lists capturing this comm
  struct ncclTasks tasks;

  // user-created reduction ops
  int userRedOpCapacity, userRedOpFreeHead;
  ncclUserRedOp *userRedOps;

  // Queue of things for the main thread to do
  struct ncclIntruQueueMpsc<struct ncclCommCallback, &ncclCommCallback::next> callbackQueue;

  // List of kernel plans built form tasks.
  struct ncclIntruQueue<struct ncclKernelPlan, &ncclKernelPlan::next> planQueue;
  // First of the unlaunched kernels in `planQueue`
  struct ncclKernelPlan* unlaunchedPlansHead;

  ncclConfig_t config;
  // initState is to more conveniently reclaim resources when errors happen.
  ncclResult_t initState;
  // flag to indicate if ncclCommFinalize() is called
  bool finalizeCalled;
  // shared structures for finalization
  int finalizeRankCnt;
  // group job to support multi-thread FT
  struct ncclGroupJob *groupJob;

  /* store to buffer register request */
  struct ncclIntruQueue<struct ncclRegRequest, &ncclRegRequest::next> regRequestQueue;
  /* store registered buffer */
  struct ncclIntruQueue<struct ncclRegRecord, &ncclRegRecord::next> regRecordQueue;

  // Tuning plugin
  ncclTuner_t* tuner;
};

enum ncclLaunchMode {
  ncclLaunchModeInvalid=0,
  ncclLaunchModeParallel,
  ncclLaunchModeGroup
};
extern enum ncclLaunchMode ncclParamLaunchMode;

void ncclCommPushFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle);

inline ncclResult_t ncclCommPollCallbacks(struct ncclComm* comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  struct ncclCommCallback* cb = ncclIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
  while (cb != nullptr) {
    struct ncclCommCallback* next = cb->next;
    ncclResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
    cb = next;
  }
  NCCLCHECK(result);
  return ncclSuccess;
}

inline void ncclCommIntraBarrierIn(struct ncclComm* comm, uint32_t x) {
  int phase = comm->intraBarrierPhase;
  if (comm->intraRanks == 1) {
    // Release everyone (just me).
    comm->intraBarrierGate = (uint64_t(x)<<32) | (phase^1);
  } else {
    struct ncclComm* comm0 = comm->intraComm0;
    uint64_t count = __atomic_add_fetch(&comm0->intraBarrierCounter, (uint64_t(x)<<32) + 1, __ATOMIC_RELEASE);
    if (uint32_t(count) == uint32_t(comm->intraRanks)) {
      // Reset.
      __atomic_store_n(&comm0->intraBarrierCounter, 0, __ATOMIC_RELAXED);
      // Release everyone.
      __atomic_store_n(&comm0->intraBarrierGate, (count>>32<<32) | (phase^1), __ATOMIC_RELEASE);
    }
  }
}

// returns sum of x values contributed to ncclCommIntraBarrierIn(comm, x)
inline uint32_t ncclCommIntraBarrierOut(struct ncclComm* comm) {
  struct ncclComm* comm0 = comm->intraComm0;
  comm->intraBarrierPhase ^= 1;
  uint32_t phase = comm->intraBarrierPhase;
  uint64_t gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
  if ((gate & 1) != phase) {
    uint64_t t0 = clockNano();
    do {
      // Spin vigorously for first 5us.
      if (clockNano()-t0 >= 5*1000) sched_yield();
      gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
    } while ((gate & 1) != phase);
  }
  if (comm->intraRanks != 1) __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return gate>>32;
}

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

ncclResult_t ncclCommEnsureReady(ncclComm_t comm);
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);

#endif
