/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef PROFILER_PLUGIN_CE_H_
#define PROFILER_PLUGIN_CE_H_

#include "event.h"
#include "nccl/profiler_v6.h"

// CE profiler initialization and cleanup
ncclResult_t ceProfilerInitGlobal(void);
ncclResult_t ceProfilerFinalizeGlobal(FILE* fh);

// CE context management
void ceProfilerRegisterContext(struct context* ctx);
void ceProfilerDeregisterContext(struct context* ctx);

// CE event cleanup
void ceProfilerCleanupPendingEvents(struct context* ctx);

// CE event start/stop functions
ncclResult_t ceProfilerStartCeCollEvent(struct context* ctx, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr, double startTime);
ncclResult_t ceProfilerStopCeCollEvent(void* eHandle);

ncclResult_t ceProfilerStartCeSyncEvent(struct context* ctx, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr, double startTime);
ncclResult_t ceProfilerStopCeSyncEvent(void* eHandle);

ncclResult_t ceProfilerStartCeBatchEvent(struct context* ctx, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr, double startTime);
ncclResult_t ceProfilerStopCeBatchEvent(void* eHandle);

// Get CE timing mode for context initialization
CeTimingMode_t ceProfilerGetTimingMode(void);

#endif // PROFILER_PLUGIN_CE_H_

