/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_STRUCT_H_
#define MSCCL_STRUCT_H_

#include <cstdint>
#include <map>
#include <set>
#include <vector>
#include "devcomm.h"
//#include "proxy.h"
#include "msccl/msccl_scheduler.h"

#define MSCCL_MAX_NUM_STEPS 64
#define MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL 32
#define MSCCL_MAX_NUM_THREAD_BLOCKS (MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL * MAXCHANNELS)
#define MSCCL_MAX_COUNT 72 // max concurrent number of msccl chunk transmission
#define MSCCL_MAX_REDUCE_FUSION 16
#define MSCCL_MAX_NUM_ALGOS 1024

#define MSCCL_SLICESTEPS (NCCL_STEPS/4)
#define MSCCL_CHUNKSTEPS (NCCL_STEPS/2)

#define MSCCL_INPUT_BUFFER 0
#define MSCCL_OUTPUT_BUFFER 1
#define MSCCL_SCRATCH_BUFFER 2

#define MSCCL_SEND 0
#define MSCCL_RECV 1
#define MSCCL_RECV_COPY_SEND 2
#define MSCCL_RECV_REDUCE_SEND 3
#define MSCCL_RECV_REDUCE_COPY 4
#define MSCCL_RECV_REDUCE_COPY_SEND 5
#define MSCCL_LOCAL_COPY 6
#define MSCCL_REDUCE 7

struct alignas(16) mscclTransmission {
  int16_t dependencePointer; // index to the first dependence
  int16_t numDependencies; // dependencePointer+numDependencies indicate the last dependence
  int16_t reductionPointer; // where the reduction starts
  int16_t numReductions; // number of reductions with the same dst
  int16_t srcOffset;
  int16_t dstOffset;
  uint8_t srcBuffer : 4; // input/output/scratch
  uint8_t dstBuffer : 4; // input/output/scratch
  int8_t hasDependence;
  uint8_t type;
  uint8_t count;
}; // 16 bytes

static_assert((1ULL << (8*sizeof(mscclTransmission::count))) - 1 > MSCCL_MAX_COUNT, "MSCCL_MAX_COUNT must representable by datatype of count");

struct alignas(16) mscclThreadBlock {
  // step is used to index into these arrays
  alignas(16) struct mscclTransmission transmissions[MSCCL_MAX_NUM_STEPS]; // 4KB
  int8_t dependentBid[MSCCL_MAX_NUM_STEPS]; // -1 if not dependent on any thread block, 256 bytes
  int16_t dependentStep[MSCCL_MAX_NUM_STEPS]; // 512 bytes
  int16_t reductionSrcOffsets[MSCCL_MAX_NUM_STEPS]; // 512 bytes
  int16_t sendPeer;
  int16_t recvPeer;
  uint16_t nSteps;
  int16_t channelId; // associated channel. -1 indicates a thread block with only local copies
}; // 5384 bytes

static_assert(sizeof(struct mscclThreadBlock) % sizeof(uint64_t) == 0, "Sanity check: sizeof(struct mscclThreadBlock) \
  % sizeof(uint64_t) != 0");

struct mscclFlag {
  uint64_t flag;
  uint64_t align[3]; // to avoid false sharing
};

struct mscclChannelPeerInfo {
  int peer;
  // nTransmissionsOfCount[i]: number of transmissions with count i (in terms of msccl chunks)
  int nTransmissionsOfCount[MSCCL_MAX_COUNT + 1];
  int existingCounts[MSCCL_MAX_COUNT + 1];
  int nExistingCounts;
};

struct mscclChannelInfo {
  struct mscclChannelPeerInfo sendPeerInfo[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  int nSendPeers;
  struct mscclChannelPeerInfo recvPeerInfo[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  int nRecvPeers;
};

struct mscclAlgoMeta {
  // Path to algorithm file
  std::string filePath;
  // number of chunks of input/output in each MSCCL algorithm loop
  int nChunksPerLoop;
  // number of ranks required by this algorithm
  int nRanks;
  // need to times nRanks for all-gather, reduce-scatter and all-to-all
  int sizeMultiplier;
  // MSCCL function type
  mscclFunc_t func;
  // Min message size allowed for this algorithm.
  int64_t minBytes;
  // Max message size allowed for this algorithm, 0 for no limit.
  int64_t maxBytes;
  // Whether this algorithm is suitable for in-place.
  bool inPlace;
  // Whether this algorithm is suitable for out-of-place.
  bool outOfPlace;
};

struct mscclAlgo {
  // number of chunks of input/output in each MSCCL algorithm loop
  int nChunksPerLoop;
  // the protocol that the algorithm needs to use
  int protocol;
  // number of channels needed by MSCCL algorithm
  int nChannels;
  // number of ranks required by this algorithm
  int nRanks;
  // number of necessary thread blocks
  int nBlocks;
  // number of scratch chunks that MSCCL will use
  int nScratchChunks;
  // need to times nRanks for all-gather, reduce-scatter and all-to-all
  int sizeMultiplier;
  // number of steps per chunk for this algorithm
  int chunkSteps;
  // number of steps per slice for this algorithm
  int sliceSteps;
  // bid is used as an index into this array
  alignas(16) struct mscclThreadBlock mscclTBs[MSCCL_MAX_NUM_THREAD_BLOCKS];
  // used to calculate proxy info
  struct mscclChannelInfo mscclChannels[MAXCHANNELS];
  // Whether the algorithm requires reduce operation
  bool hasReduce;
  // MSCCL function type
  mscclFunc_t func;
  // Min message size allowed for this algorithm.
  int64_t minBytes;
  // Max message size allowed for this algorithm, 0 for no limit.
  int64_t maxBytes;
  // Whether this algorithm is suitable for in-place.
  bool inPlace;
  // Whether this algorithm is suitable for out-of-place.
  bool outOfPlace;
};

enum mscclGroupStatus {
  mscclNoGroup,
  mscclGroupSupportedOp,
  mscclGroupUnsupportedOp
};

struct mscclSavedSchedulerParam {
  struct mscclSchedulerParam p;
  std::vector<size_t> savedSendCounts;
  std::vector<size_t> savedSDisPls;
  std::vector<size_t> savedRecvCounts;
  std::vector<size_t> savedRDisPls;
  ncclComm_t comm;
  cudaStream_t stream;
};

enum mscclCaptureStatus {
  mscclNoCapture,
  mscclNewCapture,
  mscclExistingCapture
};

// struct mscclProxyArg {
//   struct ncclChannel* channel;
//   int opType;
//   int peer;
//   struct ncclProxyOp* proxyOp;
//   int connIndex;
//   mscclProxyArg(struct ncclChannel* channel, int opType, int peer, struct ncclProxyOp* proxyOp, int connIndex) 
//     : channel(channel), opType(opType), peer(peer), proxyOp(proxyOp), connIndex(connIndex) {}
// };

struct mscclProxyArg {
  struct mscclAlgo* hostAlgo;
  ncclComm_t comm;
  mscclProxyArg(struct mscclAlgo* hostAlgo, ncclComm_t comm) 
    : hostAlgo(hostAlgo), comm(comm) {}
};

// typedef std::map<ncclComm_t, std::vector<struct mscclProxyArg>> mscclSavedCudaHostNodeParams;

// typedef std::map<unsigned long long, std::vector<mscclSavedCudaHostNodeParams>> mscclSavedProxyArgs;

typedef std::map<unsigned long long, std::vector<struct mscclProxyArg>> mscclSavedProxyArgs;

struct mscclThreadLocalStatus {
  bool mscclIsCallerFlag;
  mscclGroupStatus groupStatus;
  int groupDepth;
  std::vector<struct mscclSavedSchedulerParam> savedSchedulerParams;
  unsigned long long captureId;
  mscclCaptureStatus captureStatus;
  cudaGraph_t graph;
};

struct mscclStatus {
  std::vector<mscclAlgoHandle_t> freeAlgoHandles;
  std::map<mscclAlgoHandle_t, mscclAlgo *> hostAlgos;
  std::map<mscclAlgoHandle_t, mscclAlgo *> devAlgos;
  struct mscclFlag* syncFlags;
  void *scratchBuffer;
  uint64_t scratchBufferSize;
  size_t nBytes;
  int stepSize;
  int chunkSteps;
  int sliceSteps;
  int chunkSize;
  int chunkEffectiveSize;
  uint32_t workIndex;
  uint32_t maxAllowedCount;
  ncclDataType_t dataType;
  std::map<ncclComm_t, std::set<mscclAlgoHandle_t>> connectedAlgos;
  cudaStream_t lastStream;
  void* mscclSchedulerLib;
  mscclSchedulerInterface* mscclSchedulerPtr;
  std::vector<mscclAlgoMeta> algoMetas;
  std::vector<std::map<int, mscclAlgoHandle_t>> rankToAlgoHandles;
  bool graphEnabled;
  bool graphFirstKernel;
};

struct alignas(16) mscclWork {
  volatile struct mscclFlag *syncFlags;
  void *scratchBuffer;
  const void *sendBuff;
  void *recvBuff;
  size_t count;
  uint64_t redOpArg;
  uint32_t workIndex;
  int nChunksPerLoop;
  uint32_t maxAllowedCount;
  bool hasReduce;
  bool redOpArgIsPtr;
};

struct mscclShmemData {
  alignas(16) struct mscclThreadBlock mscclTB;
  alignas(16) struct mscclWork work;
};
static_assert(offsetof(struct mscclShmemData, work) % 16 == 0, "mscclShmemData.work needs to be 16B aligned");

#endif
