/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

#include "transport.h"

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

#define DEFAULT_BUFFER_SIZE_BYTES (1LL << 22) /* 4MiB */

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
      char pad2[CACHE_LINE_SIZE-sizeof(void*)];
      uint64_t opCount;
    };
    char pad3[MEM_ALIGN];
  };
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      uint64_t opCount;
      char pad2[CACHE_LINE_SIZE-sizeof(uint64_t)];
      int sizesFifo[NCCL_STEPS];
    };
    char pad4[MEM_ALIGN];
  };
  ncclLLFifoLine llBuff[NCCL_LL_BUFF_LINES];
  uint64_t ll128Buff[NCCL_LL128_BUFF_ELEMS];
  char buff[1]; // Actually larger than that
};

struct ncclComm {
  struct ncclChannel channels[MAXCHANNELS];

  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;

  void* bootstrap;

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
  int64_t busId;   // my PCI bus ID in int format

  int node;
  int nNodes;
  int localRanks;

  enum { GROUP, PARALLEL } launchMode;
  cudaStream_t userStream;
  bool userStreamSet;
  cudaEvent_t doneEvent;
  bool checkPointers;

  // Counter to make sure collectives match (needed for bcast/reduce
  // where syncs are not symmetric).
  uint64_t opCount;
  uint64_t lastOpCount;

  // Channels for collectives
  int nChannels;

  // Only nvlink is used for inter-GPU communication
  int nvlink;

  // Algorithm/Protocols thresholds
  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  int maxThreads[NCCL_NUM_PROTOCOLS];

  // An internal CUDA stream for NCCL kernel CGMD launches
  int groupCudaStream;
  cudaStream_t groupStream;

  // Whether there has been a fatal error in this communicator.
  ncclResult_t fatalError;

  // Error reported by GPU
  volatile ncclDevError_t* fatalDevError;

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
  int* intraCudaDevs;
  int* intraCGMode; // Whether we can use CUDA9 CGMD or not
  int* intraCC; // Only to check all have the same ComputeCap and disable CGMode if not
  struct ncclColl args;
  void* argsptr;

  // Global proxy thread
  pthread_t proxyThread;
  struct ncclProxyState proxyState;
};

#endif
