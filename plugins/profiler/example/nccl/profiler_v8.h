/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PROFILER_V8_H_
#define PROFILER_V8_H_

#include "profiler_v7.h"

// v8 extends v7 by letting NCCL stamp authoritative timestamps on every
// event (start via the descriptor, stop via a new stop descriptor) so the
// plugin does not have to compute its own CPU clocks. That matters for events
// NCCL reconstructs after the fact, where a host clock read at emission time
// would describe when the plugin was called rather than when the work ran.
//
// This is opt-in and off by default: a plugin asks for it by setting
// ncclProfilerConfig_v8_t::ncclSourcedTimings during init (see below). Until
// it does, every timestamp below is NCCL_PROFILER_TS_NONE and the plugin times
// events itself exactly as it did in v7. Timestamps are also
// NCCL_PROFILER_TS_NONE whenever a particular clock is unavailable for a
// particular event.
//
// Clocks:
//  - tsGpu     : GPU globaltimer nanoseconds (arbitrary epoch). Set for
//                events whose timing is GPU-derived (KernelCh and KernelPhase
//                always; Coll/P2p in lossy mode, derived from their first/last
//                KernelCh). 0 for purely host-side events.
//  - tsCpuMono : CLOCK_MONOTONIC nanoseconds. Canonical timebase for
//                ordering/durations. For host-side events it is sampled at the
//                call site; for GPU-derived events it is tsGpu mapped through
//                the profiler thread's GPU->CPU calibration.
//  - tsCpuReal : CLOCK_REALTIME nanoseconds. Wall-clock anchor for human
//                readability only (may jump); never use for durations.
//
// CE events (ncclProfileCeColl/CeSync/CeBatch) are intentionally left with
// zeroed timestamps here: their completion is tracked by CUDA events inside
// the plugin, so NCCL does not stamp them.

#define NCCL_PROFILER_TS_NONE (0ULL)

typedef struct {
  uint64_t type;                // event type descriptor
  void* parentObj;              // pointer to the profiler parent object
  int rank;                     // originating rank

  // NCCL-stamped start timestamps (see clock notes above). 0 == unavailable.
  uint64_t tsGpu;               // GPU globaltimer ns (GPU-derived events only)
  uint64_t tsCpuMono;           // CLOCK_MONOTONIC ns (canonical)
  uint64_t tsCpuReal;           // CLOCK_REALTIME ns (wall-clock anchor)

  union {
    // All v7 descriptors
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
      uint64_t userTag;         // per-call user profiler tag (0 == untagged)
    } collApi;

    struct {
      const char* func;
      size_t count;
      const char* datatype;
      void* stream;
      bool graphCaptured;
      uint64_t userTag;         // per-call user profiler tag (0 == untagged)
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
      uint64_t userTag;         // per-call user profiler tag (0 == untagged)
    } coll;

    struct {
      const char* func;
      void* buff;
      const char* datatype;
      size_t count;
      int peer;
      uint8_t nChannels;
      void* parentGroup;
      uint64_t userTag;         // per-call user profiler tag (0 == untagged)
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

    // v6 CE-specific descriptors (timestamps above stay 0 for these)
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
      uint64_t userTag;         // per-call user profiler tag (0 == untagged)
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
} ncclProfilerEventDescr_v8_t;

// New in v8: the stop timestamps carried by stopEvent. Same clock semantics
// as the descriptor's start timestamps. Passing NULL (or zeroed fields) means
// "no NCCL-stamped stop time available"; the plugin should fall back to its
// previous behavior in that case.
typedef struct {
  uint64_t tsGpu;               // GPU globaltimer ns (GPU-derived events only)
  uint64_t tsCpuMono;           // CLOCK_MONOTONIC ns (canonical)
  uint64_t tsCpuReal;           // CLOCK_REALTIME ns (wall-clock anchor)
} ncclProfilerEventStopDescr_v8_t;

// v8 uses the same state args structure as v7.
typedef ncclProfilerEventStateArgs_v7_t ncclProfilerEventStateArgs_v8_t;

// New in v8: the configuration handshake between NCCL and the plugin.
//
// NCCL owns one of these per communicator. It fills in the header fields and
// the defaults, then passes a pointer to init(). The plugin opts into optional
// behavior by writing the toggles before it returns, and NCCL reads the struct
// back when init() returns and keeps its own copy for the lifetime of the
// communicator. The pointer is only valid for the duration of the init() call,
// and may be NULL, in which case every default applies.
//
// This is append-only: new members are added at the end, and existing ones are
// never removed or reordered. A plugin built against an older v8 header
// therefore stays source- and binary-compatible; check `size` before reading a
// member that the running NCCL may predate.
#define NCCL_PROFILER_CONFIG_MAGIC   (0x50524f46u)  // 'PROF'
#define NCCL_PROFILER_CONFIG_VERSION (1u)

typedef struct {
  /* Set by NCCL. The plugin must not write these. */
  size_t size;                  // sizeof(ncclProfilerConfig_v8_t) as NCCL built it
  unsigned int magic;           // NCCL_PROFILER_CONFIG_MAGIC
  unsigned int version;         // NCCL_PROFILER_CONFIG_VERSION

  /* Toggles the plugin may set. All default to 0, meaning disabled. */

  // Opt in to NCCL-sourced timings. While this is 0, NCCL reads no clocks,
  // runs no GPU clock calibration and leaves every tsGpu/tsCpuMono/tsCpuReal
  // at NCCL_PROFILER_TS_NONE, so the feature costs nothing when unused and the
  // plugin derives its own timing as it did in v7 and earlier.
  //
  // Set it non-zero and NCCL stamps the timestamps documented at the top of
  // this header. For GPU-derived events (KernelCh, KernelPhase) that requires
  // a periodic GPU->CPU clock calibration -- a one-thread probe kernel on a
  // private stream, from which NCCL fits the GPU globaltimer onto
  // CLOCK_MONOTONIC and CLOCK_REALTIME -- so the opt-in is not free.
  int ncclSourcedTimings;
} ncclProfilerConfig_v8_t;

typedef struct {
  const char* name;

  // init - initialize the profiler plugin. config carries the NCCL/plugin
  // feature handshake (see ncclProfilerConfig_v8_t); it may be NULL, and any
  // toggle the plugin sets takes effect when init returns.
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes,
                       int nranks, int rank, ncclDebugLogger_t logfn, ncclProfilerConfig_v8_t* config);

  // startEvent - initialize and start a new event. eDescr carries NCCL-stamped
  // start timestamps (see ncclProfilerEventDescr_v8_t).
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v8_t* eDescr);

  // stopEvent - stop/finalize an event. eStopDescr carries NCCL-stamped stop
  // timestamps and may be NULL when none are available.
  ncclResult_t (*stopEvent)(void* eHandle, ncclProfilerEventStopDescr_v8_t* eStopDescr);

  // recordEventState - record event state transitions and updates
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v8_t eState,
                                   ncclProfilerEventStateArgs_v8_t* eStateArgs);

  // finalize - finalize the profiler plugin
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v8_t;

#endif // PROFILER_V8_H_
