/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "devcomm.h"
#include "collectives.h"
#include "core.h"
#include "utils.h"
#include "strongstream.h"

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
  ncclPatternSend,
  ncclPatternRecv
} ncclPattern_t;

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
  int algorithm;
  int protocol;
  ncclPattern_t pattern;
  int nChannels;
  int nThreads;
  size_t nBytes;
  int nstepsPerLoop;
  int nchunksPerLoop;
  int chunkSize;
  int channelId;
};

inline ncclResult_t ncclInfoSetDerived(struct ncclInfo* info, int nRanks) {
  info->nBytes = info->count * ncclTypeSize(info->datatype);
  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncBroadcast) {
    info->count = info->nBytes;
    info->datatype = ncclInt8;
  }
  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncReduceScatter) info->nBytes *= nRanks; // count is per rank
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
  struct ncclIntruQueue<ncclTaskColl, &ncclTaskColl::next> collQueue;
  size_t collBytesTotal;
  struct Peer* peers/*[nRanks]*/;
  int *p2pSendOrder/*[nRanks]*/, *p2pRecvOrder/*[nRanks]*/;
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
