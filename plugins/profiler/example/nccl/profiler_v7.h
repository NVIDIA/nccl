/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Poolside Inc & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PROFILER_V7_H_
#define PROFILER_V7_H_

#include "profiler_v6.h"

// Extend v6 descriptor with UserTag support
typedef struct {
  uint64_t type;                // event type descriptor
  void* parentObj;              // pointer to the profiler parent object
  int rank;                     // originating rank
  union {
    // All v6 descriptors
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

    // v7 UserTag descriptor
    struct {
      const char* tag;  // Pointer valid only during startEvent/stopEvent
    } userTag;
  };
} ncclProfilerEventDescr_v7_t;

// v7 uses same state args as v6
typedef ncclProfilerEventStateArgs_v6_t ncclProfilerEventStateArgs_v7_t;

typedef struct {
  const char* name;

  // init - initialize the profiler plugin
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes, int nranks, int rank, ncclDebugLogger_t logfn);

  // startEvent - initialize and start a new event
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v7_t* eDescr);

  // stopEvent - stop/finalize an event
  ncclResult_t (*stopEvent)(void* eHandle);

  // recordEventState - record event state transitions and updates
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v7_t eState, ncclProfilerEventStateArgs_v7_t* eStateArgs);

  // finalize - finalize the profiler plugin
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v7_t;

#endif // PROFILER_V7_H_
