/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "device.h"
#include "collectives.h"
#include "core.h"
#include "utils.h"
#include "strongstream.h"
#define NCCL_MAX_LOCAL_RANKS 64

typedef enum : uint8_t {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollnetChain,
  ncclPatternCollnetDirect,
  ncclPatternNvls,
  ncclPatternNvlsTree,
  ncclPatternSend,
  ncclPatternRecv
} ncclPattern_t;

enum ncclRegBufferType {
  NCCL_REGULAR_BUFFER = 0,
  NCCL_IPC_REG_BUFFER = 1,
  NCCL_NVLS_REG_BUFFER = 2,
  NCCL_REG_BUFFER_NUM = 3
};

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root; // peer for p2p operations
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  // Computed later
  ncclDevRedOpFull opFull;
  ncclPattern_t pattern;
  size_t nBytes;
  size_t aggnBytes;
  size_t workBytes;
  size_t sendbuffSize;
  size_t recvbuffSize;
  int stepSize;
  int chunkCount;
  int chunkSize;
  int channelId;
  int workFuncIndex;
  ncclRegBufferType regBufType;
  void* regBufSend[NCCL_MAX_LOCAL_RANKS];
  void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
  // Need to initialize
  int nThreads;
  int nChannels;
  int algorithm;
  int protocol;
  bool userTuned;
  struct ncclInfo *next;
};

inline ncclResult_t ncclInfoSetDerived(struct ncclInfo* info, int nRanks) {
  info->nBytes = info->workBytes = info->count * ncclTypeSize(info->datatype);
  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncBroadcast) {
    info->count = info->workBytes;
    info->datatype = ncclInt8;
  }
  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncReduceScatter) info->nBytes *= nRanks; // count is per rank

  /* compute buffer size for NVLS buffer registration */
  if (info->coll == ncclFuncAllGather) {
    info->sendbuffSize = info->workBytes;
    info->recvbuffSize = info->sendbuffSize * nRanks;
  } else if (info->coll == ncclFuncReduceScatter) {
    info->recvbuffSize = info->workBytes;
    info->sendbuffSize = info->recvbuffSize * nRanks;
  } else {
    info->sendbuffSize = info->recvbuffSize = info->workBytes;
  }
  return ncclSuccess;
}

struct ncclTaskColl {
  struct ncclTaskColl* next;
  ncclFunc_t func;
  void const* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  ncclDataType_t datatype;
  ncclDevRedOpFull op;
  int chunkSteps, sliceSteps;
  struct ncclInfo info;
};
struct ncclTaskP2p {
  ncclTaskP2p *next;
  void *buff;
  size_t bytes;
  // Stateful chunk index. If a p2p gets "cut" over two plans this keeps track
  // of where it left off.
  int chunk;
};

struct ncclCudaStreamList {
  struct ncclCudaStreamList *next;
  cudaStream_t stream;
};
struct ncclTasks {
  struct Peer {
    bool sendSeen, recvSeen;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> sendQueue;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> recvQueue;
  };
  struct ncclIntruQueue<struct ncclInfo, &ncclInfo::next> collQueue;
  // Queue for user-tuned executed collectives
  struct ncclIntruQueue<struct ncclInfo, &ncclInfo::next> collTunedQueue;
  // Queue for continuous bytes distribution (CBD) collectives
  struct ncclIntruQueue<struct ncclInfo, &ncclInfo::next> collCBDQueue;
  // Queue for collnet
  struct ncclIntruQueue<struct ncclInfo, &ncclInfo::next> collnetQueue;
  size_t workBytesTotal;
  int usableChannels;
  bool sorted;
  struct Peer* peers/*[nRanks]*/;
  int *p2pSendOrder, *p2pRecvOrder;
  int p2pOrderSteps;
  int nTasksColl, nTasksP2p;

  // The list of user streams aggregated over all tasks present.
  struct ncclCudaStreamList* streams;
  // The most recent user stream. Ignored if streams==nullptr
  cudaStream_t streamRecent;
  // The graph capturing all user streams or invalid if none. Thus we restrict the
  // user that all streams must be captured in the same graph or not captured
  // at all. Technically we could probably relax this, but that would mean
  // collecting a different `ncclTasks` per graph and one for non-graph.
  struct ncclCudaGraph capturingGraph;
};

#endif
