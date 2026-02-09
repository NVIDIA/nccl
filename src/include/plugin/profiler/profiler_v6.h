/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef PROFILER_V6_H_
#define PROFILER_V6_H_

#include "profiler_v5.h"

// Extend v5 descriptor with CE-specific fields
typedef struct {
  uint64_t type;                // event type descriptor
  void* parentObj;              // pointer to the profiler parent object
  int rank;                     // originating rank
  union {
    // All v5 descriptors (groupApi, collApi, p2pApi, kernelLaunch, coll, p2p, proxyOp, proxyStep, kernelCh, netPlugin)
    struct {
      bool graphCaptured;
      int groupDepth;
    } groupApi;

    struct {
      const char* func;
      size_t count;
      const char* datatype;
      int root;
      void* stream;
      bool graphCaptured;
    } collApi;

    struct {
      const char* func;
      size_t count;
      const char* datatype;
      void* stream;
      bool graphCaptured;
    } p2pApi;

    struct {
      void* stream;
    } kernelLaunch;

    struct {
      uint64_t seqNumber;
      const char* func;
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      const char* datatype;
      uint8_t nChannels;
      uint8_t nWarps;
      const char* algo;
      const char* proto;
      void* parentGroup;
    } coll;

    struct {
      const char* func;
      void* buff;
      const char* datatype;
      size_t count;
      int peer;
      uint8_t nChannels;
      void* parentGroup;
    } p2p;

    struct {
      pid_t pid;
      uint8_t channelId;
      int peer;
      int nSteps;
      int chunkSize;
      int isSend;
    } proxyOp;

    struct {
      int step;
    } proxyStep;

    struct {
      uint8_t channelId;
      uint64_t pTimer;
    } kernelCh;

    struct {
      int64_t id;
      void* data;
    } netPlugin;

    // v6 CE-specific descriptors
    struct {
      uint64_t seqNumber;
      const char* func;
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      const char* datatype;
      const char* syncStrategy;
      bool intraBatchSync;
      uint32_t batchSize;
      uint32_t numBatches;
      uint32_t ceSeqNum;
      void* stream;
    } ceColl;

    struct {
      bool isComplete;
      int nRanks;
    } ceCollSync;

    struct {
      int numOps;
      size_t totalBytes;
      bool useIntraSync;
    } ceCollBatch;
  };
} ncclProfilerEventDescr_v6_t;

// v6 uses same state args as v5 (no CE-specific state args needed)
// CE events don't use recordEventState - plugin manages all timing internally
typedef ncclProfilerEventStateArgs_v5_t ncclProfilerEventStateArgs_v6_t;

typedef struct {
  const char* name;

  // init - initialize the profiler plugin
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes, int nranks, int rank, ncclDebugLogger_t logfn);

  // startEvent - initialize and start a new event
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr);

  // stopEvent - stop/finalize an event
  ncclResult_t (*stopEvent)(void* eHandle);

  // recordEventState - record event state transitions and updates
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v6_t eState, ncclProfilerEventStateArgs_v6_t* eStateArgs);

  // finalize - finalize the profiler plugin
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v6_t;

#endif // PROFILER_V6_H_

