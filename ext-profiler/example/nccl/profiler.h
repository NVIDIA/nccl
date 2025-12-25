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

enum {
  ncclProfileGroup          = (1 << 0),  // group event type
  ncclProfileColl           = (1 << 1),  // host collective call event type
  ncclProfileP2p            = (1 << 2),  // host point-to-point call event type
  ncclProfileProxyOp        = (1 << 3),  // proxy operation event type
  ncclProfileProxyStep      = (1 << 4),  // proxy step event type
  ncclProfileProxyCtrl      = (1 << 5),  // proxy control event type
  ncclProfileKernelCh       = (1 << 6),  // kernel channel event type
  ncclProfileNetPlugin      = (1 << 7),  // network plugin-defined, events
  ncclProfileGroupApi       = (1 << 8),  // Group API events
  ncclProfileCollApi        = (1 << 9),  // Collective API events
  ncclProfileP2pApi         = (1 << 10), // Point-to-Point API events
  ncclProfileKernelLaunch   = (1 << 11), // Kernel launch events
  // CE events (v6)
  ncclProfileCeColl         = (1 << 12), // CE collective operation
  ncclProfileCeSync         = (1 << 13), // CE synchronization operation
  ncclProfileCeBatch        = (1 << 14), // CE batch operation
};

typedef enum {
  ncclProfilerProxyOpSendPosted        = 0,  // deprecated in v4
  ncclProfilerProxyOpSendRemFifoWait   = 1,  // deprecated in v4
  ncclProfilerProxyOpSendTransmitted   = 2,  // deprecated in v4
  ncclProfilerProxyOpSendDone          = 3,  // deprecated in v4
  ncclProfilerProxyOpRecvPosted        = 4,  // deprecated in v4
  ncclProfilerProxyOpRecvReceived      = 5,  // deprecated in v4
  ncclProfilerProxyOpRecvTransmitted   = 6,  // deprecated in v4
  ncclProfilerProxyOpRecvDone          = 7,  // deprecated in v4
  ncclProfilerProxyOpInProgress_v4     = 19,

  /* Legacy proxy profiler states */
  ncclProfilerProxyStepSendGPUWait     = 8,
  ncclProfilerProxyStepSendPeerWait_v4 = 20,
  ncclProfilerProxyStepSendWait        = 9,
  ncclProfilerProxyStepRecvWait        = 10,
  ncclProfilerProxyStepRecvFlushWait   = 11,
  ncclProfilerProxyStepRecvGPUWait     = 12,

  /* Legacy proxy control states */
  ncclProfilerProxyCtrlIdle            = 13,
  ncclProfilerProxyCtrlActive          = 14,
  ncclProfilerProxyCtrlSleep           = 15,
  ncclProfilerProxyCtrlWakeup          = 16,
  ncclProfilerProxyCtrlAppend          = 17,
  ncclProfilerProxyCtrlAppendEnd       = 18,

  /* Network defined event states */
  ncclProfilerNetPluginUpdate          = 21,

  /* Kernel event states */
  ncclProfilerKernelChStop             = 22,

  /* Group API States */
  ncclProfilerGroupStartApiStop        = 23,
  ncclProfilerGroupEndApiStart         = 24,

  /* CE-specific states (v6) */
  ncclProfilerCeCollStart              = 25,  // CE collective operation begins
  ncclProfilerCeCollComplete           = 26,  // CE collective operation completes
  ncclProfilerCeSyncStart              = 27,  // CE synchronization begins
  ncclProfilerCeSyncComplete           = 28,  // CE synchronization completes
  ncclProfilerCeBatchStart             = 29,  // CE batch operation begins
  ncclProfilerCeBatchComplete          = 30,  // CE batch operation completes
} ncclProfilerEventState_t;

typedef ncclProfilerEventState_t ncclProfilerEventState_v1_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v2_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v3_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v4_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v5_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v6_t;

#include "profiler_v6.h"
#include "profiler_v5.h"
#include "profiler_v4.h"
#include "profiler_v3.h"
#include "profiler_v2.h"
#include "profiler_v1.h"
#include "profiler_net.h"

// Use v6 as default to support CE events
// v5 and earlier versions are still supported for backward compatibility
typedef ncclProfiler_v6_t ncclProfiler_t;
typedef ncclProfilerEventDescr_v6_t ncclProfilerEventDescr_t;
typedef ncclProfilerEventStateArgs_v6_t ncclProfilerEventStateArgs_t;

#endif // end include guard
