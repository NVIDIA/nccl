/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef PROFILER_V1_H_
#define PROFILER_V1_H_

typedef struct {
  uint8_t type;                 // event type descriptor: ncclProfileColl, ...
  void* parentObj;              // pointer to the profiler parent object (for coll is the group)
  int rank;                     // originating rank
  union {
    struct {
      const char* name;
      uint64_t commHash;
      uint64_t seqNumber;
      uint8_t func;
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      uint8_t datatype;
      uint32_t op;
      size_t trafficBytes;
      uint8_t nMaxChannels;
      uint8_t nWarps;
      uint8_t algo;
      uint8_t proto;
      int isCollnet;
      int isNvls;
    } coll;

    struct {
      const char* name;
      uint64_t commHash;
      uint8_t func;
      void* buff;
      uint8_t datatype;
      size_t count;
      int peer;
    } p2p;

    struct {
      pid_t pid;                // pid of the originating process
      uint8_t channelId;        // channel id for this proxy operation
      int peer;                 // remote rank for send/recv
      int nSteps;               // number of steps for this proxy operation
      int chunkSize;            // amount of data transferred by this proxy operation
      int isSend;
    } proxyOp;

    struct {
      int step;
    } proxyStep;
  };
} ncclProfilerEventDescr_v1_t;

typedef union {
  struct {
    size_t transSize;
    int steps;
  } proxyOp;

  struct {
    int appendedProxyOps;
  } proxyCtrl;
} ncclProfilerEventStateArgs_v1_t;

typedef struct {
  const char* name;

  // init - initialize the profiler plugin
  // Input
  //  - context        : opaque profiler context object for separating profiler behavior across comms
  // Output
  //  - eActivationMask: bitmask of active events set by the plugin
  ncclResult_t (*init)(void** context, int* eActivationMask);

  // startEvent - initialize and start a new event for the supplied event descriptor inside the eventset
  // Input
  //  - context: opaque profiler context object
  //  - eDescr : pointer to ncclProfilerEventDescr_t object
  // Output
  //  - eHandle: return event handle for supplied event descriptor object
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v1_t* eDescr);

  // stopEvent - stop/finalize an event inside and event set
  // Input
  //  - eHandle: handle to event object
  ncclResult_t (*stopEvent)(void* eHandle);

  // recordEventState - record event state transitions and event attribute updates
  // Input
  //  - eHandle   : handle to event object created through startEvent
  //  - eStateArgs: optional argument used to capture event attribute updates associated with the state transition
  //  - eState    : event state transition
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v1_t eState, ncclProfilerEventStateArgs_v1_t* eStateArgs);

  // finalize - finalize the profiler plugin
  // Input
  //  - context: opaque profiler context object
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v1_t;

#endif
