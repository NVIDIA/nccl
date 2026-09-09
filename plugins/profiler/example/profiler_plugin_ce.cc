/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdio.h>
#include <pthread.h>
#include <cstring>
#include <unistd.h>
#include <time.h>
#include "profiler_plugin_ce.h"
#include "event.h"
#include "print_event.h"

#define __hidden __attribute__ ((visibility("hidden")))

// External reference to gettime() from plugin.cc
extern double gettime(void);

// External reference to startTime from plugin.cc
extern double getProfilerStartTime(void);

// CE profiler global state
static struct {
  pthread_t pollerThread;
  bool pollerStarted;   // poller thread exists, so CE global state is live
  bool pollerRunning;
  pthread_mutex_t mutex;
  struct context** contextRegistry;
  int contextCount;
  int contextCapacity;
  CeTimingMode_t timingMode;
  int pollerIntervalUs;
} ceProfilerCtxt = {
  .pollerThread = 0,
  .pollerStarted = false,
  .pollerRunning = false,
  .mutex = PTHREAD_MUTEX_INITIALIZER,
  .contextRegistry = NULL,
  .contextCount = 0,
  .contextCapacity = 0,
  .timingMode = CE_TIMING_CPU,
  .pollerIntervalUs = 500
};

// Poll CE Coll events for a context
static void pollCeCollEvents(struct context* ctx) {
  if (ctx->ceCollPoolSize == 0 || ctx->ceCollPool == NULL) return;

  double startTime = getProfilerStartTime();
  struct ceColl** ceCollPtr = &ctx->ceEvents.ceCollHead;
  while (*ceCollPtr) {
    struct ceColl* event = *ceCollPtr;

    if (!event->startCompleted && cudaEventQuery(event->startEvent) == cudaSuccess) {
      event->startCompleted = true;
      event->cpuStartTime = gettime();
      event->base.startTs = gettime() - startTime;
    }

    if (event->startCompleted && !event->stopCompleted && cudaEventQuery(event->stopEvent) == cudaSuccess) {
      event->stopCompleted = true;
      event->cpuStopTime = gettime();
      event->base.stopTs = gettime() - startTime;

      event->cpuDuration = event->cpuStopTime - event->cpuStartTime;
      if (event->timingMode == CE_TIMING_GPU) {
        float elapsedMs;
        if (cudaEventElapsedTime(&elapsedMs, event->startEvent, event->stopEvent) == cudaSuccess) {
          event->elapsedTime = (uint64_t)(elapsedMs * 1000);
        } else {
          event->elapsedTime = (uint64_t)event->cpuDuration;
        }
      } else {
        event->elapsedTime = (uint64_t)event->cpuDuration;
      }

      // Decrement parent refCount when complete
      if (event->parent) {
        __atomic_sub_fetch(&event->parent->refCount, 1, __ATOMIC_ACQ_REL);
      }

      // Return event to pool for reuse (ring buffer behavior)
      __atomic_fetch_add(&ctx->ceCollPoolBase, 1, __ATOMIC_RELAXED);

      *ceCollPtr = event->pollerNext;
      // Release only after unlinking, so the slot cannot be recycled while still
      // reachable from ceCollHead.
      __atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_ACQ_REL);
      continue;
    }

    ceCollPtr = &event->pollerNext;
  }
}

// Poll CE Sync events for a context
static void pollCeSyncEvents(struct context* ctx) {
  if (ctx->ceSyncPoolSize == 0 || ctx->ceSyncPool == NULL) return;

  double startTime = getProfilerStartTime();
  struct ceSync** ceSyncPtr = &ctx->ceEvents.ceSyncHead;
  while (*ceSyncPtr) {
    struct ceSync* event = *ceSyncPtr;

    if (!event->startCompleted && cudaEventQuery(event->startEvent) == cudaSuccess) {
      event->startCompleted = true;
      event->cpuStartTime = gettime();
      event->base.startTs = gettime() - startTime;
    }

    if (event->startCompleted && !event->stopCompleted && cudaEventQuery(event->stopEvent) == cudaSuccess) {
      event->stopCompleted = true;
      event->cpuStopTime = gettime();
      event->base.stopTs = gettime() - startTime;

      event->cpuDuration = event->cpuStopTime - event->cpuStartTime;
      if (event->timingMode == CE_TIMING_GPU) {
        float elapsedMs;
        if (cudaEventElapsedTime(&elapsedMs, event->startEvent, event->stopEvent) == cudaSuccess) {
          event->elapsedTime = (uint64_t)(elapsedMs * 1000);
        } else {
          event->elapsedTime = (uint64_t)event->cpuDuration;
        }
      } else {
        event->elapsedTime = (uint64_t)event->cpuDuration;
      }

      // Decrement parent refCount when complete
      if (event->parent) {
        __atomic_sub_fetch(&event->parent->base.refCount, 1, __ATOMIC_ACQ_REL);
      }

      // Return event to pool for reuse (ring buffer behavior)
      __atomic_fetch_add(&ctx->ceSyncPoolBase, 1, __ATOMIC_RELAXED);

      *ceSyncPtr = event->pollerNext;
      __atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_ACQ_REL);
      continue;
    }

    ceSyncPtr = &event->pollerNext;
  }
}

// Poll CE Batch events for a context
static void pollCeBatchEvents(struct context* ctx) {
  if (ctx->ceBatchPoolSize == 0 || ctx->ceBatchPool == NULL) return;

  double startTime = getProfilerStartTime();
  struct ceBatch** ceBatchPtr = &ctx->ceEvents.ceBatchHead;
  while (*ceBatchPtr) {
    struct ceBatch* event = *ceBatchPtr;

    if (!event->startCompleted && cudaEventQuery(event->startEvent) == cudaSuccess) {
      event->startCompleted = true;
      event->cpuStartTime = gettime();
      event->base.startTs = gettime() - startTime;
    }

    if (event->startCompleted && !event->stopCompleted && cudaEventQuery(event->stopEvent) == cudaSuccess) {
      event->stopCompleted = true;
      event->cpuStopTime = gettime();
      event->base.stopTs = gettime() - startTime;

      event->cpuDuration = event->cpuStopTime - event->cpuStartTime;
      if (event->timingMode == CE_TIMING_GPU) {
        float elapsedMs;
        if (cudaEventElapsedTime(&elapsedMs, event->startEvent, event->stopEvent) == cudaSuccess) {
          event->elapsedTime = (uint64_t)(elapsedMs * 1000);
        } else {
          event->elapsedTime = (uint64_t)event->cpuDuration;
        }
      } else {
        event->elapsedTime = (uint64_t)event->cpuDuration;
      }

      // Decrement parent refCount when complete
      if (event->parent) {
        __atomic_sub_fetch(&event->parent->base.refCount, 1, __ATOMIC_ACQ_REL);
      }

      // Return event to pool for reuse (ring buffer behavior)
      __atomic_fetch_add(&ctx->ceBatchPoolBase, 1, __ATOMIC_RELAXED);

      *ceBatchPtr = event->pollerNext;
      __atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_ACQ_REL);
      continue;
    }

    ceBatchPtr = &event->pollerNext;
  }
}

// CE poller thread main function
static void* cePollerThreadMain(void* arg) {
  while (__atomic_load_n(&ceProfilerCtxt.pollerRunning, __ATOMIC_RELAXED)) {
    if (pthread_mutex_trylock(&ceProfilerCtxt.mutex) != 0) {
      usleep(ceProfilerCtxt.pollerIntervalUs);
      continue;
    }

    for (int i = 0; i < ceProfilerCtxt.contextCount; i++) {
      struct context* ctx = ceProfilerCtxt.contextRegistry[i];
      if (!ctx) continue;

      if (pthread_mutex_trylock(&ctx->ceEvents.mutex) != 0) {
        continue;
      }

      pollCeCollEvents(ctx);
      pollCeSyncEvents(ctx);
      pollCeBatchEvents(ctx);
      pthread_mutex_unlock(&ctx->ceEvents.mutex);
    }

    pthread_mutex_unlock(&ceProfilerCtxt.mutex);
    usleep(ceProfilerCtxt.pollerIntervalUs);
  }

  return NULL;
}

// Initialize CE profiler global state
ncclResult_t ceProfilerInitGlobal(void) {
  const char* ceTimingStr = getenv("NCCL_PROFILER_CE_TIMING");
  if (ceTimingStr && strcasecmp(ceTimingStr, "gpu") == 0) {
    ceProfilerCtxt.timingMode = CE_TIMING_GPU;
  } else {
    ceProfilerCtxt.timingMode = CE_TIMING_CPU;
  }

  const char* intervalStr = getenv("NCCL_PROFILER_CE_POLLER_INTERVAL_MICROSECONDS");
  if (intervalStr) {
    ceProfilerCtxt.pollerIntervalUs = atoi(intervalStr);
  }

  ceProfilerCtxt.contextCapacity = 16;
  ceProfilerCtxt.contextRegistry
    = (struct context**)calloc(ceProfilerCtxt.contextCapacity,
                               sizeof(struct context*));
  if (!ceProfilerCtxt.contextRegistry) {
    ceProfilerCtxt.contextCapacity = 0;
    return ncclSystemError;
  }

  ceProfilerCtxt.pollerRunning = true;
  if (pthread_create(&ceProfilerCtxt.pollerThread, NULL, cePollerThreadMain, NULL) != 0) {
    ceProfilerCtxt.pollerRunning = false;
    free(ceProfilerCtxt.contextRegistry);
    ceProfilerCtxt.contextRegistry = NULL;
    ceProfilerCtxt.contextCapacity = 0;
    return ncclSystemError;
  }

  ceProfilerCtxt.pollerStarted = true;
  return ncclSuccess;
}

// Finalize CE profiler global state
ncclResult_t ceProfilerFinalizeGlobal(FILE* fh) {
  // With CE events disabled the poller was never created, and pthread_join(0) segfaults.
  if (!ceProfilerCtxt.pollerStarted) return ncclSuccess;

  __atomic_store_n(&ceProfilerCtxt.pollerRunning, false, __ATOMIC_RELAXED);
  pthread_join(ceProfilerCtxt.pollerThread, NULL);
  ceProfilerCtxt.pollerStarted = false;

  free(ceProfilerCtxt.contextRegistry);
  ceProfilerCtxt.contextRegistry = NULL;
  ceProfilerCtxt.contextCount = 0;
  ceProfilerCtxt.contextCapacity = 0;
  return ncclSuccess;
}

// Register context with CE poller for tracking
void ceProfilerRegisterContext(struct context* ctx) {
  if (pthread_mutex_init(&ctx->ceEvents.mutex, NULL) != 0) {
    return;
  }

  ctx->ceEvents.ceCollHead = NULL;
  ctx->ceEvents.ceSyncHead = NULL;
  ctx->ceEvents.ceBatchHead = NULL;

  // Nothing polls this context, and growing from zero capacity would leave a
  // registry that makes teardown think CE is live.
  if (!ceProfilerCtxt.pollerStarted) return;

  // Must not be skipped: an unregistered context is never polled.
  pthread_mutex_lock(&ceProfilerCtxt.mutex);

  // Check if context with this commHash+rank already exists
  for (int i = 0; i < ceProfilerCtxt.contextCount; i++) {
    if (ceProfilerCtxt.contextRegistry[i] &&
        ceProfilerCtxt.contextRegistry[i]->commHash == ctx->commHash &&
        ceProfilerCtxt.contextRegistry[i]->rank == ctx->rank) {
      pthread_mutex_unlock(&ceProfilerCtxt.mutex);
      return;
    }
  }

  // Resize registry if needed
  if (ceProfilerCtxt.contextCount >= ceProfilerCtxt.contextCapacity) {
    int newCapacity = ceProfilerCtxt.contextCapacity * 2;
    struct context** newRegistry = (struct context**)calloc(newCapacity, sizeof(struct context*));
    if (newRegistry) {
      memcpy(newRegistry, ceProfilerCtxt.contextRegistry, ceProfilerCtxt.contextCount * sizeof(struct context*));
      free(ceProfilerCtxt.contextRegistry);
      ceProfilerCtxt.contextRegistry = newRegistry;
      ceProfilerCtxt.contextCapacity = newCapacity;
    }
  }

  if (ceProfilerCtxt.contextCount < ceProfilerCtxt.contextCapacity) {
    ceProfilerCtxt.contextRegistry[ceProfilerCtxt.contextCount++] = ctx;
  }
  pthread_mutex_unlock(&ceProfilerCtxt.mutex);
}

// Deregister context from CE poller
void ceProfilerDeregisterContext(struct context* ctx) {
  if (!ceProfilerCtxt.pollerStarted) return;

  // Must not be skipped: the poller would walk this context after it is freed.
  pthread_mutex_lock(&ceProfilerCtxt.mutex);

  for (int i = 0; i < ceProfilerCtxt.contextCount; i++) {
    if (ceProfilerCtxt.contextRegistry[i] &&
        ceProfilerCtxt.contextRegistry[i]->commHash == ctx->commHash &&
        ceProfilerCtxt.contextRegistry[i]->rank == ctx->rank) {
      ceProfilerCtxt.contextRegistry[i] = ceProfilerCtxt.contextRegistry[ceProfilerCtxt.contextCount - 1];
      ceProfilerCtxt.contextCount--;
      break;
    }
  }
  pthread_mutex_unlock(&ceProfilerCtxt.mutex);
}

void ceProfilerCleanupPendingEvents(struct context* ctx) {
  // Must not be skipped: it would leak this context's CUDA events.
  pthread_mutex_lock(&ctx->ceEvents.mutex);

  struct ceColl* ceColl = ctx->ceEvents.ceCollHead;
  while (ceColl) {
    if (ceColl->startEvent) cudaEventDestroy(ceColl->startEvent);
    if (ceColl->stopEvent) cudaEventDestroy(ceColl->stopEvent);
    ceColl = ceColl->pollerNext;
  }

  struct ceSync* ceSync = ctx->ceEvents.ceSyncHead;
  while (ceSync) {
    if (ceSync->startEvent) cudaEventDestroy(ceSync->startEvent);
    if (ceSync->stopEvent) cudaEventDestroy(ceSync->stopEvent);
    ceSync = ceSync->pollerNext;
  }

  struct ceBatch* ceBatch = ctx->ceEvents.ceBatchHead;
  while (ceBatch) {
    if (ceBatch->startEvent) cudaEventDestroy(ceBatch->startEvent);
    if (ceBatch->stopEvent) cudaEventDestroy(ceBatch->stopEvent);
    ceBatch = ceBatch->pollerNext;
  }

  pthread_mutex_unlock(&ctx->ceEvents.mutex);
}

// Get CE timing mode
CeTimingMode_t ceProfilerGetTimingMode(void) {
  return ceProfilerCtxt.timingMode;
}

// Start CE Coll event
ncclResult_t ceProfilerStartCeCollEvent(struct context* ctx, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr, double startTime) {
  // Core reads the handle back to decide whether to stop this event and what to
  // parent CeSync/CeBatch to, so a path that creates no event must leave it NULL.
  *eHandle = NULL;
  if (ctx->ceCollPoolSize <= 0) return ncclSuccess;

  struct ceColl* event;
  int ceCollId = __atomic_fetch_add(&ctx->ceCollPoolIndex, 1, __ATOMIC_RELAXED);
  event = &ctx->ceCollPool[ceCollId % ctx->ceCollPoolSize];
  // Recycle a slot only once its previous owner retired it (refCount==0). CE
  // events complete out of order, so the ring window alone does not prove the
  // slot is free, and reusing a linked one spins the poller forever.
  if ((ceCollId - __atomic_load_n(&ctx->ceCollPoolBase, __ATOMIC_RELAXED)) < ctx->ceCollPoolSize &&
      __atomic_load_n(&event->base.refCount, __ATOMIC_ACQUIRE) == 0) {
    event->parent = (struct collApi*)eDescr->parentObj;
    event->ceCollId = ceCollId;
    event->seqNumber = eDescr->ceColl.seqNumber;
    event->count = eDescr->ceColl.count;
    event->datatype = eDescr->ceColl.datatype;
    event->root = eDescr->ceColl.root;
    event->syncStrategy = eDescr->ceColl.syncStrategy;
    event->stream = (cudaStream_t)eDescr->ceColl.stream;
    event->eventId = ceCollId;
    event->timingMode = ceProfilerCtxt.timingMode;
    event->startCompleted = false;
    event->stopCompleted = false;

    // Initialize taskEventBase fields
    event->base.type = eDescr->type;
    event->base.rank = eDescr->rank;
    event->base.func = eDescr->ceColl.func;
    event->base.refCount = 1;
    event->base.parent = event->parent;
    event->base.next = NULL;
    event->base.startTs = -1;
    event->base.stopTs = -1;

    // Initialize child event queue for CeSync/CeBatch
    event->eventHead = NULL;
    event->eventTail = NULL;

    // Add to parent CollApi's task event queue
    if (event->parent) {
      taskEventQueueEnqueue(event->parent, &event->base);
      __atomic_fetch_add(&event->parent->refCount, 1, __ATOMIC_RELAXED);
    }

    // Create CUDA events with appropriate flags
    if (ceProfilerCtxt.timingMode == CE_TIMING_GPU) {
      cudaEventCreate(&event->startEvent, 0);
      cudaEventCreate(&event->stopEvent, 0);
    } else {
      cudaEventCreateWithFlags(&event->startEvent, cudaEventDisableTiming);
      cudaEventCreateWithFlags(&event->stopEvent, cudaEventDisableTiming);
    }

    // Record start event to stream
    cudaEventRecord(event->startEvent, event->stream);

    // Must not be skipped: an unlinked event is never polled, so its slot would
    // never retire.
    pthread_mutex_lock(&ctx->ceEvents.mutex);
    event->pollerNext = ctx->ceEvents.ceCollHead;
    ctx->ceEvents.ceCollHead = event;
    pthread_mutex_unlock(&ctx->ceEvents.mutex);

    *eHandle = event;
    debugEvent(*eHandle, "CeCollStartEvent");
    return ncclSuccess;
  } else {
    __atomic_fetch_sub(&ctx->ceCollPoolIndex, 1, __ATOMIC_RELAXED);
    return ncclSuccess;
  }
}

// Stop CE Coll event
ncclResult_t ceProfilerStopCeCollEvent(void* eHandle) {
  struct ceColl* event = (struct ceColl*)eHandle;
  cudaEventRecord(event->stopEvent, event->stream);
  debugEvent(eHandle, "CeCollStopEvent");
  return ncclSuccess;
}

// Start CE Sync event
ncclResult_t ceProfilerStartCeSyncEvent(struct context* ctx, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr, double startTime) {
  *eHandle = NULL;
  if (ctx->ceSyncPoolSize <= 0) return ncclSuccess;
  if (eDescr->parentObj == NULL) return ncclSuccess;

  struct ceSync* event;
  int ceSyncId = __atomic_fetch_add(&ctx->ceSyncPoolIndex, 1, __ATOMIC_RELAXED);
  event = &ctx->ceSyncPool[ceSyncId % ctx->ceSyncPoolSize];
  // See ceProfilerStartCeCollEvent.
  if ((ceSyncId - __atomic_load_n(&ctx->ceSyncPoolBase, __ATOMIC_RELAXED)) < ctx->ceSyncPoolSize &&
      __atomic_load_n(&event->base.refCount, __ATOMIC_ACQUIRE) == 0) {
    event->parent = (struct ceColl*)eDescr->parentObj;
    event->ceSyncId = ceSyncId;
    event->isComplete = eDescr->ceCollSync.isComplete;
    event->nRanks = eDescr->ceCollSync.nRanks;
    // Get seqNumber and stream from parent CeColl event
    event->seqNumber = event->parent->seqNumber;
    event->stream = event->parent->stream;
    event->eventId = ceSyncId;
    event->timingMode = ceProfilerCtxt.timingMode;
    event->startCompleted = false;
    event->stopCompleted = false;

    // Initialize taskEventBase fields
    event->base.type = eDescr->type;
    event->base.rank = eDescr->rank;
    event->base.func = "CeSync";
    event->base.refCount = 1;
    event->base.parent = event->parent;
    event->base.next = NULL;
    event->base.startTs = -1;
    event->base.stopTs = -1;

    // Add to parent CeColl's event queue if it exists
    if (event->parent) {
      taskEventQueueEnqueue(event->parent, &event->base);
      __atomic_fetch_add(&event->parent->base.refCount, 1, __ATOMIC_RELAXED);
    }

    // Create CUDA events with appropriate flags
    if (ceProfilerCtxt.timingMode == CE_TIMING_GPU) {
      cudaEventCreate(&event->startEvent, 0);
      cudaEventCreate(&event->stopEvent, 0);
    } else {
      cudaEventCreateWithFlags(&event->startEvent, cudaEventDisableTiming);
      cudaEventCreateWithFlags(&event->stopEvent, cudaEventDisableTiming);
    }

    // Record start event to stream
    cudaEventRecord(event->startEvent, event->stream);

    pthread_mutex_lock(&ctx->ceEvents.mutex);
    event->pollerNext = ctx->ceEvents.ceSyncHead;
    ctx->ceEvents.ceSyncHead = event;
    pthread_mutex_unlock(&ctx->ceEvents.mutex);

    *eHandle = event;
    debugEvent(*eHandle, "CeSyncStartEvent");
    return ncclSuccess;
  } else {
    __atomic_fetch_sub(&ctx->ceSyncPoolIndex, 1, __ATOMIC_RELAXED);
    return ncclSuccess;
  }
}

// Stop CE Sync event
ncclResult_t ceProfilerStopCeSyncEvent(void* eHandle) {
  struct ceSync* event = (struct ceSync*)eHandle;
  cudaEventRecord(event->stopEvent, event->stream);
  debugEvent(eHandle, "CeSyncStopEvent");
  return ncclSuccess;
}

// Start CE Batch event
ncclResult_t ceProfilerStartCeBatchEvent(struct context* ctx, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr, double startTime) {
  *eHandle = NULL;
  if (ctx->ceBatchPoolSize <= 0) return ncclSuccess;
  if (eDescr->parentObj == NULL) return ncclSuccess;

  struct ceBatch* event;
  int ceBatchId = __atomic_fetch_add(&ctx->ceBatchPoolIndex, 1, __ATOMIC_RELAXED);
  event = &ctx->ceBatchPool[ceBatchId % ctx->ceBatchPoolSize];
  // See ceProfilerStartCeCollEvent.
  if ((ceBatchId - __atomic_load_n(&ctx->ceBatchPoolBase, __ATOMIC_RELAXED)) < ctx->ceBatchPoolSize &&
      __atomic_load_n(&event->base.refCount, __ATOMIC_ACQUIRE) == 0) {
    event->parent = (struct ceColl*)eDescr->parentObj;
    event->ceBatchId = ceBatchId;
    event->numOps = eDescr->ceCollBatch.numOps;
    event->totalBytes = eDescr->ceCollBatch.totalBytes;
    event->useIntraSync = eDescr->ceCollBatch.useIntraSync;
    // Get stream from parent CeColl event
    event->stream = event->parent->stream;
    event->eventId = ceBatchId;
    event->timingMode = ceProfilerCtxt.timingMode;
    event->startCompleted = false;
    event->stopCompleted = false;

    // Initialize taskEventBase fields
    event->base.type = eDescr->type;
    event->base.rank = eDescr->rank;
    event->base.func = "CeBatch";
    event->base.refCount = 1;
    event->base.parent = event->parent;
    event->base.next = NULL;
    event->base.startTs = -1;
    event->base.stopTs = -1;

    // Add to parent CeColl's event queue if it exists
    if (event->parent) {
      taskEventQueueEnqueue(event->parent, &event->base);
      __atomic_fetch_add(&event->parent->base.refCount, 1, __ATOMIC_RELAXED);
    }

    // Create CUDA events with appropriate flags
    if (ceProfilerCtxt.timingMode == CE_TIMING_GPU) {
      cudaEventCreate(&event->startEvent, 0);
      cudaEventCreate(&event->stopEvent, 0);
    } else {
      cudaEventCreateWithFlags(&event->startEvent, cudaEventDisableTiming);
      cudaEventCreateWithFlags(&event->stopEvent, cudaEventDisableTiming);
    }

    // Record start event to stream
    cudaEventRecord(event->startEvent, event->stream);

    pthread_mutex_lock(&ctx->ceEvents.mutex);
    event->pollerNext = ctx->ceEvents.ceBatchHead;
    ctx->ceEvents.ceBatchHead = event;
    pthread_mutex_unlock(&ctx->ceEvents.mutex);

    *eHandle = event;
    debugEvent(*eHandle, "CeBatchStartEvent");
    return ncclSuccess;
  } else {
    __atomic_fetch_sub(&ctx->ceBatchPoolIndex, 1, __ATOMIC_RELAXED);
    return ncclSuccess;
  }
}

// Stop CE Batch event
ncclResult_t ceProfilerStopCeBatchEvent(void* eHandle) {
  struct ceBatch* event = (struct ceBatch*)eHandle;
  cudaEventRecord(event->stopEvent, event->stream);
  debugEvent(eHandle, "CeBatchStopEvent");
  return ncclSuccess;
}

