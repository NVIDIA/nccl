/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PROFILER_V8_H_
#define PROFILER_V8_H_

#include "profiler_v7.h"

// v8 extends v7 by letting NCCL stamp an authoritative CPU timestamp on every
// event, so the plugin does not have to read its own clock. That matters for
// events NCCL reconstructs after the fact, where a host clock read at emission
// time would describe when the plugin was called rather than when the work
// ran. The same descriptor type carries the start timestamp to startEvent and
// the stop timestamp to stopEvent, and ncclProfilerEventStateArgs_v8_t carries
// one for every state transition in between.
//
// GPU-timed events are unaffected: KernelCh and KernelPhase continue to carry
// the raw GPU globaltimer in their existing pTimer fields -- in the descriptor
// union on start, and in ncclProfilerEventStateArgs_t on the corresponding
// recordEventState stop. v8 adds no second path for those.
//
// This is opt-in and off by default. A plugin asks for it by setting
// ncclProfilerConfig_v8_t::cpuClock during init (see below) and, separately,
// gpuClockSync if it also wants GPU-timed events mapped onto that clock. Until
// it does, NCCL reads no clocks, runs no calibration, and leaves tsCpu at
// NCCL_PROFILER_TS_NONE -- the plugin times events itself exactly as it did in
// v7. tsCpu is also NCCL_PROFILER_TS_NONE whenever the clock is unavailable
// for a particular event.
//
// Which clock tsCpu names is the plugin's choice, not a property of the event:
// whatever cpuClock selects, every tsCpu in the run is on that timebase. A
// consumer that needs both a monotonic timebase and wall-clock can pick
// MONOTONIC and apply a CLOCK_REALTIME - CLOCK_MONOTONIC offset it samples
// itself, which moves only with NTP slew.
//
// Accuracy: tsCpuUncertainty carries a symmetric +/- bound on tsCpu. Host-side
// events read the clock at the call site and report 0. GPU-timed events, when
// gpuClockSync is on, are mapped through a calibration anchor NCCL builds by
// reading the GPU clock from the host: the GPU read landed somewhere inside a
// CPU interval of width w, and NCCL reports that interval's midpoint, so the
// bound is w/2. NCCL probes several times per anchor and keeps the tightest,
// but it cannot drive w to zero -- a descheduled probe thread or a busy GPU
// widens it. The bound covers the anchor only, not drift accumulated between
// anchors. Plugins needing a confidence interval should use it (and may drop
// samples whose bound is too wide for their purpose); the rest can ignore it.
//
// CE events (ncclProfileCeColl/CeSync/CeBatch) are intentionally left with a
// zeroed tsCpu: their completion is tracked by CUDA events inside the plugin,
// so NCCL does not stamp them.

#define NCCL_PROFILER_TS_NONE (0ULL)

typedef struct {
  uint64_t type;                // event type descriptor
  void* parentObj;              // pointer to the profiler parent object
  int rank;                     // originating rank

  // NCCL-stamped CPU timestamp (see clock notes above). This is the event's
  // start time in startEvent and its stop time in stopEvent; every field below
  // it is meaningful in startEvent only.
  uint64_t tsCpu;               // ns on the clock cpuClock selected
  uint64_t tsCpuUncertainty;    // symmetric +/- bound on tsCpu, ns

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

// State transitions are timestamped on the same terms as start and stop: an
// event that lives through several updates (a proxy step moving through its
// wait states, a net plugin reporting progress) would otherwise have NCCL
// timing only at its two ends, leaving the plugin to time the middle from its
// own clock on a thread that may be running well after the transition.
//
// Laid out like ncclProfilerEventDescr_v8_t, with the timestamp ahead of the
// union, so the two read the same way. This is what makes v8 state args
// layout-incompatible with v7; adapters field-copy rather than cast.
typedef struct {
  // NCCL-stamped CPU timestamp of the transition, on the clock cpuClock
  // selected, and its symmetric +/- bound. Same rules as the descriptor's:
  // NCCL_PROFILER_TS_NONE when the plugin did not opt in or the clock is
  // unavailable, and uncertainty 0 for host-side transitions. The
  // KernelCh/KernelPhase stop transitions are the GPU-timed ones, so they
  // carry a non-zero bound when gpuClockSync is on, matching the tsCpu on the
  // stopEvent that follows.
  uint64_t tsCpu;
  uint64_t tsCpuUncertainty;

  union {
    struct {
      size_t transSize;
    } proxyStep;

    struct {
      int appendedProxyOps;
    } proxyCtrl;

    struct {
      void* data;
    } netPlugin;

    // Shared by ncclProfileKernelCh and the ncclProfileKernelPhase sub-event:
    // both record a single stop timestamp (GPU globaltimer) of identical
    // shape, so the phase stop reuses this member rather than a dedicated one.
    struct {
      uint64_t pTimer;
    } kernelCh;
  };
} ncclProfilerEventStateArgs_v8_t;

// New in v8: the configuration handshake between NCCL and the plugin.
//
// NCCL owns one of these per communicator. It zeroes it, then passes a pointer
// to init(). The plugin opts into optional behavior by writing the fields
// before it returns, and NCCL reads the struct back when init() returns and
// keeps its own copy for the lifetime of the communicator. The pointer is only
// valid for the duration of the init() call, and may be NULL, in which case
// every default applies. Because NCCL reads it back only once, this is
// init-time configuration: a plugin cannot change these at runtime (unlike
// eActivationMask, which stays where it is for exactly that reason).
//
// The layout is fixed, like every other struct on this interface; there is no
// per-struct size/magic/version. Both sides already agree on the layout by
// agreeing on the ncclProfiler_v8 symbol name, and anything incompatible rides
// the next interface version bump.

// Clock backing ncclProfilerEventDescr_v8_t::tsCpu. Stored as uint32_t in the
// config so the ABI does not depend on the compiler's enum width.
typedef enum {
  ncclProfilerCpuClockNone      = 0,  // no CPU timestamps (default)
  ncclProfilerCpuClockMonotonic = 1,  // CLOCK_MONOTONIC
  ncclProfilerCpuClockRealtime  = 2,  // CLOCK_REALTIME
} ncclProfilerCpuClock_v8_t;

typedef struct {
  // Which clock NCCL should stamp into tsCpu, as an ncclProfilerCpuClock_v8_t.
  // Left at ncclProfilerCpuClockNone, NCCL reads no clocks and leaves every
  // tsCpu at NCCL_PROFILER_TS_NONE, so the feature costs nothing when unused
  // and the plugin derives its own timing as it did in v7 and earlier. Set to
  // Monotonic or Realtime, NCCL samples that clock once per event at the call
  // site and stamps it into tsCpu, with tsCpuUncertainty 0.
  //
  // A value NCCL does not recognize is treated as None.
  uint32_t cpuClock;

  // Also map GPU-timed events (KernelCh, KernelPhase) onto cpuClock, so they
  // carry tsCpu in addition to the raw globaltimer they already report in
  // pTimer. This is the expensive half of the feature: it requires a periodic
  // GPU->CPU calibration -- a one-thread probe kernel on a private stream --
  // from which NCCL fits the GPU globaltimer onto cpuClock, and the resulting
  // tsCpu is an estimate with a non-zero tsCpuUncertainty.
  //
  // Ignored when cpuClock is None, since there is no clock to map onto.
  uint32_t gpuClockSync;
} ncclProfilerConfig_v8_t;

typedef struct {
  const char* name;

  // init - initialize the profiler plugin. config carries the NCCL/plugin
  // feature handshake (see ncclProfilerConfig_v8_t); it may be NULL, and any
  // toggle the plugin sets takes effect when init returns.
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes,
                       int nranks, int rank, ncclDebugLogger_t logfn, ncclProfilerConfig_v8_t* config);

  // startEvent - initialize and start a new event. eDescr carries the
  // NCCL-stamped start timestamp (see ncclProfilerEventDescr_v8_t).
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v8_t* eDescr);

  // stopEvent - stop/finalize an event. eDescr reuses the start descriptor
  // type to carry the NCCL-stamped stop timestamp; only the tsCpu and
  // tsCpuUncertainty fields are meaningful here, and the rest (type,
  // parentObj, rank, union) are zeroed. The event being stopped is identified
  // by eHandle alone. eDescr may be NULL when no timestamp is available.
  ncclResult_t (*stopEvent)(void* eHandle, ncclProfilerEventDescr_v8_t* eDescr);

  // recordEventState - record event state transitions and updates. eStateArgs
  // carries the NCCL-stamped timestamp of the transition (see
  // ncclProfilerEventStateArgs_v8_t) alongside the per-state payload, and is
  // never NULL in v8.
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v8_t eState,
                                   ncclProfilerEventStateArgs_v8_t* eStateArgs);

  // finalize - finalize the profiler plugin
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v8_t;

#endif // PROFILER_V8_H_
