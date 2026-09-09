/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PROFILER_V7_H_
#define PROFILER_V7_H_

#include "profiler_v6.h"

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
      uint64_t userTag;         // v7: per-call user profiler tag (0 == untagged)
    } collApi;

    struct {
      const char* func;
      size_t count;
      const char* datatype;
      void* stream;
      bool graphCaptured;
      uint64_t userTag;         // v7: per-call user profiler tag (0 == untagged)
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
      const char* kernelVariant;
      bool isSymColl;
      uint64_t userTag;         // v7: per-call user profiler tag (0 == untagged)
    } coll;

    struct {
      const char* func;
      void* buff;
      const char* datatype;
      size_t count;
      int peer;
      uint8_t nChannels;
      void* parentGroup;
      uint64_t userTag;         // v7: per-call user profiler tag (0 == untagged)
    } p2p;

    struct {
      ncclPid_t pid;
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
      uint64_t userTag;         // v7: per-call user profiler tag (0 == untagged)
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

    // v7: kernel barrier phase sub-event
    struct {
      uint8_t channelId;
      uint8_t phaseId;          // 0=initial_sync, 1=compute, 2=final_sync
      const char* phaseName;
      uint64_t pTimer;          // start timestamp (GPU globaltimer)
    } kernelPhase;
  };
} ncclProfilerEventDescr_v7_t;

typedef union {
  struct {
    size_t transSize;
  } proxyStep;

  struct {
    int appendedProxyOps;
  } proxyCtrl;

  struct {
    void* data;
  } netPlugin;

  // Shared by ncclProfileKernelCh and the v7 ncclProfileKernelPhase sub-event: both
  // record a single stop timestamp (GPU globaltimer) of identical shape, so the phase
  // stop reuses this member rather than a dedicated one.
  struct {
    uint64_t pTimer;
  } kernelCh;
} ncclProfilerEventStateArgs_v7_t;

typedef struct {
  const char* name;

  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes,
                       int nranks, int rank, ncclDebugLogger_t logfn);

  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v7_t* eDescr);

  ncclResult_t (*stopEvent)(void* eHandle);

  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v7_t eState,
                                   ncclProfilerEventStateArgs_v7_t* eStateArgs);

  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v7_t;

#endif // PROFILER_V7_H_
