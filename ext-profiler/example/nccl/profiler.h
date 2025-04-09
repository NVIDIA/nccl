/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef PROFILER_H_
#define PROFILER_H_

#include <stdint.h>
#include <stdlib.h>

#include "common.h"
#include "err.h"

enum {
  ncclProfileGroup     = (1 << 0),  // group event type
  ncclProfileColl      = (1 << 1),  // host collective call event type
  ncclProfileP2p       = (1 << 2),  // host point-to-point call event type
  ncclProfileProxyOp   = (1 << 3),  // proxy operation event type
  ncclProfileProxyStep = (1 << 4),  // proxy step event type
  ncclProfileProxyCtrl = (1 << 5),  // proxy control event type
  ncclProfileKernelCh  = (1 << 6),  // kernel channel event type
  ncclProfileNetPlugin = (1 << 7),  // network plugin-defined, events
};

typedef enum {
  ncclProfilerProxyOpSendPosted,
  ncclProfilerProxyOpSendRemFifoWait,
  ncclProfilerProxyOpSendTransmitted,
  ncclProfilerProxyOpSendDone,
  ncclProfilerProxyOpRecvPosted,
  ncclProfilerProxyOpRecvReceived,
  ncclProfilerProxyOpRecvTransmitted,
  ncclProfilerProxyOpRecvDone,

  /* Legacy proxy profiler states */
  ncclProfilerProxyStepSendGPUWait,
  ncclProfilerProxyStepSendWait,
  ncclProfilerProxyStepRecvWait,
  ncclProfilerProxyStepRecvFlushWait,
  ncclProfilerProxyStepRecvGPUWait,

  /* Legacy proxy control states */
  ncclProfilerProxyCtrlIdle,
  ncclProfilerProxyCtrlActive,
  ncclProfilerProxyCtrlSleep,
  ncclProfilerProxyCtrlWakeup,
  ncclProfilerProxyCtrlAppend,
  ncclProfilerProxyCtrlAppendEnd,
} ncclProfilerEventState_t;

typedef ncclProfilerEventState_t ncclProfilerEventState_v1_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v2_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v3_t;

#include "profiler_v3.h"
#include "profiler_v2.h"
#include "profiler_v1.h"
#include "profiler_net.h"

typedef ncclProfiler_v3_t ncclProfiler_t;
typedef ncclProfilerEventDescr_v3_t ncclProfilerEventDescr_t;
typedef ncclProfilerEventStateArgs_v3_t ncclProfilerEventStateArgs_t;

#endif // end include guard
