/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdio.h>
#include <pthread.h>
#include <cstring>
#include <chrono>
#include <linux/limits.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "event.h"
#include "print_event.h"
#include "profiler_plugin_ce.h"

#define __hidden __attribute__ ((visibility("hidden")))

static int initialized;             // initialization counter for profiler
static double startTime;            // profiler start time

static const int defaultEActivationMask = ncclProfileColl | ncclProfileP2p;
static const int defaultGroupApiPoolSize = 8;
static const int defaultCollApiPoolSize = 8;
static const int defaultP2pApiPoolSize = 8;
static const int defaultKernelLaunchPoolSize = 8;
static const int defaultGroupPoolSize = 8;
static const int defaultCeCollPoolSize = 8;
static const int defaultCeSyncPoolSize = 8;
static const int defaultCeBatchPoolSize = 8;
static const int defaultCollPoolSize = 8;
static const int defaultP2pPoolSize = 8;
static const int defaultProxyCtrlPoolSize = 16;
static const int defaultDetachPoolSize = 8;

static int groupApiPoolSize;
static int collApiPoolSize;
static int p2pApiPoolSize;
static int kernelLaunchPoolSize;
static int groupPoolSize;
static int collPoolSize;
static int p2pPoolSize;
static int proxyCtrlPoolSize;
static int ceCollPoolSize;
static int ceSyncPoolSize;
static int ceBatchPoolSize;
static int detachPoolSize;
static int detachPoolBase;
static int detachPoolIndex;
static int detachPoolDone;
static struct proxyOp* detachPool;

ncclDebugLogger_t logFn;
#define INFO(FLAGS, ...) logFn(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

__hidden double gettime(void) {
  using namespace std::chrono;
  auto now = steady_clock::now();
  return duration_cast<duration<double, std::micro>>(now.time_since_epoch()).count();
}

// Timestamp for the callback in flight, in this profiler's timeline (us since
// startTime). The v8 entry points fill it in from the NCCL-stamped tsCpu; 0
// means we were not given one and eventTime() reads the clock here instead,
// which dates the event to when the plugin was called rather than when it
// actually happened. Per-thread because proxy threads and the profiler thread
// deliver events concurrently.
static __thread double ncclStampedTs;

static inline double eventTime(void) {
  return ncclStampedTs != 0.0 ? ncclStampedTs : gettime() - startTime;
}

// Convert an NCCL tsCpu (ns on the clock we selected at init) into this
// profiler's timeline. Valid because we ask for Monotonic, the same base
// gettime() uses. NCCL_PROFILER_TS_NONE maps to 0, which makes eventTime()
// fall back.
static inline double ncclTsToLocal(uint64_t tsCpu) {
  return tsCpu != 0 ? (double)tsCpu / 1000.0 - startTime : 0.0;
}

// Export startTime for CE profiler
double getProfilerStartTime(void) {
  return startTime;
}

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pid_t pid;
static int* eActivationMaskPtr;
static void freeContextPools(struct context* ctx);
static pthread_mutex_t deferredCtxLock = PTHREAD_MUTEX_INITIALIZER;
static struct context* deferredCtxList[1024];
static int deferredCtxCount = 0;

static void deferContextFree(struct context* ctx) {
  pthread_mutex_lock(&deferredCtxLock);
  if (deferredCtxCount < 1024) {
    deferredCtxList[deferredCtxCount++] = ctx;
  }
  pthread_mutex_unlock(&deferredCtxLock);
}

static void freeDeferredContexts() {
  pthread_mutex_lock(&deferredCtxLock);
  for (int i = 0; i < deferredCtxCount; i++) {
    struct context* ctx = deferredCtxList[i];
    if (!ctx) continue;
    freeContextPools(ctx);
    free(ctx);
    deferredCtxList[i] = nullptr;
  }
  deferredCtxCount = 0;
  pthread_mutex_unlock(&deferredCtxLock);
}

static inline struct context* getTaskEventCtx(struct taskEventBase* base) {
  if (!base || !base->parent) return nullptr;
  if (base->type == ncclProfileColl) return ((struct collApi*)base->parent)->ctx;
  if (base->type == ncclProfileP2p) return ((struct p2pApi*)base->parent)->ctx;
  return nullptr;
}

static struct context* contextFromEventHandle(void* eHandle) {
  if (!eHandle) return nullptr;
  uint64_t type = *(uint64_t*)eHandle;
  switch (type) {
  case ncclProfileGroupApi: return ((struct groupApi*)eHandle)->ctx;
  case ncclProfileCollApi: return ((struct collApi*)eHandle)->ctx;
  case ncclProfileP2pApi: return ((struct p2pApi*)eHandle)->ctx;
  case ncclProfileKernelLaunch: {
    struct kernelLaunch* ev = (struct kernelLaunch*)eHandle;
    return ev->parent ? ev->parent->ctx : nullptr;
  }
  case ncclProfileGroup: return ((struct group*)eHandle)->ctx;
  case ncclProfileColl:
  case ncclProfileP2p: return getTaskEventCtx((struct taskEventBase*)eHandle);
  case ncclProfileProxyCtrl: return ((struct proxyCtrl*)eHandle)->ctx;
  case ncclProfileProxyOp: {
    struct proxyOp* ev = (struct proxyOp*)eHandle;
    return ev->parent ? getTaskEventCtx(ev->parent) : nullptr;
  }
  case ncclProfileProxyStep: {
    struct proxyStep* ev = (struct proxyStep*)eHandle;
    return (ev->parent && ev->parent->parent) ? getTaskEventCtx(ev->parent->parent) : nullptr;
  }
  case ncclProfileNetPlugin: {
    struct netPlugin* ev = (struct netPlugin*)eHandle;
    return (ev->parent && ev->parent->parent && ev->parent->parent->parent)
             ? getTaskEventCtx(ev->parent->parent->parent)
             : nullptr;
  }
  case ncclProfileKernelCh: {
    struct kernelCh* ev = (struct kernelCh*)eHandle;
    return getTaskEventCtx(ev->parent);
  }
  case ncclProfileCeColl: {
    struct ceColl* ev = (struct ceColl*)eHandle;
    return ev->parent ? ev->parent->ctx : nullptr;
  }
  case ncclProfileCeSync: {
    struct ceSync* ev = (struct ceSync*)eHandle;
    return (ev->parent && ev->parent->parent) ? ev->parent->parent->ctx : nullptr;
  }
  case ncclProfileCeBatch: {
    struct ceBatch* ev = (struct ceBatch*)eHandle;
    return (ev->parent && ev->parent->parent) ? ev->parent->parent->ctx : nullptr;
  }
  default: return nullptr;
  }
}

// Initialize pool sizes from environment variables
static void initPoolSizes(void) {
  const char* str;

  str = getenv("NCCL_PROFILE_GROUP_API_POOL_SIZE");
  groupApiPoolSize = str ? atoi(str) : defaultGroupApiPoolSize;

  str = getenv("NCCL_PROFILE_COLL_API_POOL_SIZE");
  collApiPoolSize = str ? atoi(str) : defaultCollApiPoolSize;

  str = getenv("NCCL_PROFILE_P2P_API_POOL_SIZE");
  p2pApiPoolSize = str ? atoi(str) : defaultP2pApiPoolSize;

  str = getenv("NCCL_PROFILE_KERNEL_LAUNCH_POOL_SIZE");
  kernelLaunchPoolSize = str ? atoi(str) : defaultKernelLaunchPoolSize;

  str = getenv("NCCL_PROFILE_GROUP_POOL_SIZE");
  groupPoolSize = str ? atoi(str) : defaultGroupPoolSize;

  str = getenv("NCCL_PROFILE_COLL_POOL_SIZE");
  collPoolSize = str ? atoi(str) : defaultCollPoolSize;

  str = getenv("NCCL_PROFILE_P2P_POOL_SIZE");
  p2pPoolSize = str ? atoi(str) : defaultP2pPoolSize;

  str = getenv("NCCL_PROFILE_PROXY_CTRL_POOL_SIZE");
  proxyCtrlPoolSize = str ? atoi(str) : defaultProxyCtrlPoolSize;

  str = getenv("NCCL_PROFILE_CE_COLL_POOL_SIZE");
  ceCollPoolSize = str ? atoi(str) : defaultCeCollPoolSize;

  str = getenv("NCCL_PROFILE_CE_SYNC_POOL_SIZE");
  ceSyncPoolSize = str ? atoi(str) : defaultCeSyncPoolSize;

  str = getenv("NCCL_PROFILE_CE_BATCH_POOL_SIZE");
  ceBatchPoolSize = str ? atoi(str) : defaultCeBatchPoolSize;

  str = getenv("NCCL_PROFILE_PROXY_DETACH_POOL_SIZE");
  detachPoolSize = str ? atoi(str) : defaultDetachPoolSize;
}

// Allocate global shared pools
static ncclResult_t allocateGlobalPools(void) {
  detachPool = (struct proxyOp *)calloc(detachPoolSize, sizeof(*detachPool));
  if (detachPool == NULL) {
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Allocate event pools for a context
static ncclResult_t allocateContextPools(struct context* ctx) {
  ctx->groupApiPool = (struct groupApi *)calloc(groupApiPoolSize, sizeof(*ctx->groupApiPool));
  if (!ctx->groupApiPool) goto fail;

  ctx->collApiPool = (struct collApi *)calloc(collApiPoolSize, sizeof(*ctx->collApiPool));
  if (!ctx->collApiPool) goto fail;

  ctx->p2pApiPool = (struct p2pApi *)calloc(p2pApiPoolSize, sizeof(*ctx->p2pApiPool));
  if (!ctx->p2pApiPool) goto fail;

  ctx->kernelLaunchPool = (struct kernelLaunch *)calloc(kernelLaunchPoolSize, sizeof(*ctx->kernelLaunchPool));
  if (!ctx->kernelLaunchPool) goto fail;

  ctx->groupPool = (struct group *)calloc(groupPoolSize, sizeof(*ctx->groupPool));
  if (!ctx->groupPool) goto fail;

  ctx->collPool = (struct collective *)calloc(collPoolSize, sizeof(*ctx->collPool));
  if (!ctx->collPool) goto fail;

  ctx->p2pPool = (struct p2p *)calloc(p2pPoolSize, sizeof(*ctx->p2pPool));
  if (!ctx->p2pPool) goto fail;

  ctx->proxyCtrlPool = (struct proxyCtrl *)calloc(proxyCtrlPoolSize, sizeof(*ctx->proxyCtrlPool));
  if (!ctx->proxyCtrlPool) goto fail;

  ctx->ceCollPool = (struct ceColl *)calloc(ceCollPoolSize, sizeof(*ctx->ceCollPool));
  if (!ctx->ceCollPool) goto fail;
  ctx->ceCollPoolSize = ceCollPoolSize;
  ctx->ceCollPoolBase = 0;
  ctx->ceCollPoolIndex = 0;

  ctx->ceSyncPool = (struct ceSync *)calloc(ceSyncPoolSize, sizeof(*ctx->ceSyncPool));
  if (!ctx->ceSyncPool) goto fail;
  ctx->ceSyncPoolSize = ceSyncPoolSize;
  ctx->ceSyncPoolBase = 0;
  ctx->ceSyncPoolIndex = 0;

  ctx->ceBatchPool = (struct ceBatch *)calloc(ceBatchPoolSize, sizeof(*ctx->ceBatchPool));
  if (!ctx->ceBatchPool) goto fail;
  ctx->ceBatchPoolSize = ceBatchPoolSize;
  ctx->ceBatchPoolBase = 0;
  ctx->ceBatchPoolIndex = 0;
  ctx->finalizing = 0;

  return ncclSuccess;

fail:
  if (ctx->ceBatchPool) free(ctx->ceBatchPool);
  if (ctx->ceSyncPool) free(ctx->ceSyncPool);
  if (ctx->ceCollPool) free(ctx->ceCollPool);
  if (ctx->proxyCtrlPool) free(ctx->proxyCtrlPool);
  if (ctx->p2pPool) free(ctx->p2pPool);
  if (ctx->collPool) free(ctx->collPool);
  if (ctx->groupPool) free(ctx->groupPool);
  if (ctx->collApiPool) free(ctx->collApiPool);
  if (ctx->p2pApiPool) free(ctx->p2pApiPool);
  if (ctx->kernelLaunchPool) free(ctx->kernelLaunchPool);
  if (ctx->groupApiPool) free(ctx->groupApiPool);
  return ncclSystemError;
}

// One-time global profiler initialization
static ncclResult_t initGlobalProfiler(int* eActivationMask) {
  const char* str = getenv("NCCL_PROFILE_EVENT_MASK");
  __atomic_store_n(eActivationMask, str ? atoi(str) : 0, __ATOMIC_RELAXED);

  initPoolSizes();

  ncclResult_t ret = allocateGlobalPools();
  if (ret != ncclSuccess) {
    return ret;
  }

  pid = getpid();
  startTime = gettime();

  // Only start CE poller thread if CE events are enabled AND at least one CE pool is allocated
  if ((*eActivationMask & (ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch)) &&
      (ceCollPoolSize > 0 || ceSyncPoolSize > 0 || ceBatchPoolSize > 0)) {
    ncclResult_t ret = ceProfilerInitGlobal();
    if (ret != ncclSuccess) {
      return ret;
    }
  }

  return ncclSuccess;
}

__hidden ncclResult_t exampleProfilerInit(void** context, uint64_t commId,
                                          int* eActivationMask,
                                          const char* commName, int nNodes,
                                          int nranks, int rank,
                                          ncclDebugLogger_t logfn) {
  // Must not be skipped: a comm left without a context produces no trace.
  pthread_mutex_lock(&lock);

  if (__atomic_fetch_add(&initialized, 1, __ATOMIC_RELAXED) == 0) {
    ncclResult_t ret = initGlobalProfiler(eActivationMask);
    if (ret != ncclSuccess) {
      // Init failed, so finalize is never called for this comm.
      __atomic_fetch_sub(&initialized, 1, __ATOMIC_RELAXED);
      pthread_mutex_unlock(&lock);
      return ret;
    }
  }
  pthread_mutex_unlock(&lock);

  eActivationMaskPtr = eActivationMask;

  struct context* ctx = (struct context *)calloc(1, sizeof(*ctx));
  if (!ctx) {
    __atomic_fetch_sub(&initialized, 1, __ATOMIC_RELAXED);
    return ncclSystemError;
  }

  ctx->commName = commName;
  ctx->commHash = commId;
  ctx->nranks = nranks;
  ctx->rank = rank;
  logFn = logfn;
  INFO(NCCL_INIT,
       "PROFILER/Plugin: init commName: %s commHash: %lu nranks: %d rank: %d",
       commName ? commName : "", commId, nranks, rank);

  ncclResult_t ret = allocateContextPools(ctx);
  if (ret != ncclSuccess) {
    free(ctx);
    __atomic_fetch_sub(&initialized, 1, __ATOMIC_RELAXED);
    return ret;
  }

  ceProfilerRegisterContext(ctx);

  *context = ctx;
  return ncclSuccess;
}

static const char* profilerDumpFile;

// Open trace file for writing
static FILE* openTraceFile(struct context* ctx, char* filename, size_t filenameSize) {
  const char* dump = profilerDumpFile ? profilerDumpFile : getenv("NCCL_PROFILE_DUMP_FILE");
  if (!dump) return NULL;

  snprintf(filename, filenameSize, "%s_%lu_%d.json", dump, ctx->commHash, ctx->rank);
  FILE* fh = fopen(filename, "w");
  if (fh) {
    fprintf(fh, "[\n");
  }
  return fh;
}

// Print all events to trace file
static void printAllEvents(FILE* fh, struct context* ctx) {
  if (!fh) return;

  int start, end;

  start = (ctx->groupApiPoolIndex - groupApiPoolSize >= 0) ? ctx->groupApiPoolIndex - groupApiPoolSize : 0;
  end = ctx->groupApiPoolIndex;
  for (int i = start; i < end; i++) {
    printEvent(fh, &ctx->groupApiPool[i % groupApiPoolSize]);
  }

  // Only a group's first comm owns the group API event, so every other comm's
  // events hang off a foreign root. Emit them here or its trace comes out empty.
  start = (ctx->collApiPoolIndex - collApiPoolSize >= 0) ? ctx->collApiPoolIndex - collApiPoolSize : 0;
  end = ctx->collApiPoolIndex;
  for (int i = start; i < end; i++) {
    struct collApi* event = &ctx->collApiPool[i % collApiPoolSize];
    if (event->parent == NULL || event->parent->ctx != ctx) printEvent(fh, event);
  }

  start = (ctx->p2pApiPoolIndex - p2pApiPoolSize >= 0) ? ctx->p2pApiPoolIndex - p2pApiPoolSize : 0;
  end = ctx->p2pApiPoolIndex;
  for (int i = start; i < end; i++) {
    struct p2pApi* event = &ctx->p2pApiPool[i % p2pApiPoolSize];
    if (event->parent == NULL || event->parent->ctx != ctx) printEvent(fh, event);
  }

  start = (ctx->kernelLaunchPoolIndex - kernelLaunchPoolSize >= 0)
            ? ctx->kernelLaunchPoolIndex - kernelLaunchPoolSize : 0;
  end = ctx->kernelLaunchPoolIndex;
  for (int i = start; i < end; i++) {
    struct kernelLaunch* event = &ctx->kernelLaunchPool[i % kernelLaunchPoolSize];
    if (event->parent == NULL || event->parent->ctx != ctx) printEvent(fh, event);
  }

  start = (ctx->proxyCtrlPoolIndex - proxyCtrlPoolSize >= 0) ? ctx->proxyCtrlPoolIndex - proxyCtrlPoolSize : 0;
  end = ctx->proxyCtrlPoolIndex;
  for (int i = start; i < end; i++) {
    printEvent(fh, &ctx->proxyCtrlPool[i % proxyCtrlPoolSize]);
  }

  // Print orphan CeColl events (those without CollApi parent)
  start = (ctx->ceCollPoolIndex - ctx->ceCollPoolSize >= 0) ? ctx->ceCollPoolIndex - ctx->ceCollPoolSize : 0;
  end = ctx->ceCollPoolIndex;
  for (int i = start; i < end; i++) {
    struct ceColl* event = &ctx->ceCollPool[i % ctx->ceCollPoolSize];
    // Only print if no parent (orphan) AND completed
    if (!event->parent && event->stopCompleted) {
      printEvent(fh, event);
    }
  }

  // CeSync and CeBatch are printed via their CeColl parent
}

// Free all context pools
static void freeContextPools(struct context* ctx) {
  free(ctx->groupPool);
  free(ctx->collApiPool);
  free(ctx->p2pApiPool);
  free(ctx->kernelLaunchPool);
  free(ctx->groupApiPool);
  free(ctx->collPool);
  free(ctx->p2pPool);
  free(ctx->proxyCtrlPool);
  free(ctx->ceCollPool);
  free(ctx->ceSyncPool);
  free(ctx->ceBatchPool);
}

// Global cleanup on last thread
static void finalizeGlobalProfiler(FILE* fh) {
  if (fh) {
    int start = (detachPoolIndex - detachPoolSize >= 0) ? detachPoolIndex - detachPoolSize : 0;
    int end = detachPoolIndex;
    for (int i = start; i < end; i++) {
      printEvent(fh, &detachPool[i % detachPoolSize]);
    }
  }
  free(detachPool);

  ceProfilerFinalizeGlobal(fh);
}

__hidden ncclResult_t exampleProfilerFinalize(void* context) {
  struct context* ctx = (struct context *)context;

  if (ctx == NULL) {
    return ncclSuccess;
  }
  char filename[PATH_MAX] = { 0 };
  FILE* fh = openTraceFile(ctx, filename, sizeof(filename));

  INFO(NCCL_DESTROY, "PROFILER/Plugin: finalize commName: %s commHash: %lu nranks: %d rank: %d traceFile: %s",
       ctx->commName ? ctx->commName : "", ctx->commHash, ctx->nranks, ctx->rank,
       filename[0] ? filename : "none");

  // Stop accepting any further updates for this context while dumping.
  __atomic_store_n(&ctx->finalizing, 1, __ATOMIC_RELAXED);

  // Wait for poller to complete pending events
  usleep(10000);

  // Check how many completed events we have
  int completedCount = 0;
  int totalCount = 0;
  for (int i = 0; i < ctx->ceCollPoolSize && i < ctx->ceCollPoolIndex; i++) {
    totalCount++;
    if (ctx->ceCollPool[i].stopCompleted) completedCount++;
  }
  INFO(NCCL_INIT, "PROFILER/Plugin: CeColl events - total=%d completed=%d", totalCount, completedCount);

  // Print events first (while pools are still valid)
  printAllEvents(fh, ctx);

  // Then cleanup and free resources
  ceProfilerCleanupPendingEvents(ctx);
  ceProfilerDeregisterContext(ctx);
  deferContextFree(ctx);

  if (__atomic_sub_fetch(&initialized, 1, __ATOMIC_RELAXED) == 0) {
    finalizeGlobalProfiler(fh);
    freeDeferredContexts();
  }

  if (fh) fprintf(fh, "{}]\n");
  if (fh) fclose(fh);

  return ncclSuccess;
}

__hidden void updateEvent(void* handle);

__hidden ncclResult_t exampleProfilerStartEvent(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  *eHandle = NULL;
  struct context* ctx = (struct context *)context;
  if (ctx == NULL) {
    return ncclSuccess;
  }
  if (__atomic_load_n(&ctx->finalizing, __ATOMIC_RELAXED)) {
    return ncclSuccess;
  }
  if (eDescr->type == ncclProfileGroupApi) {
    struct groupApi* event;
    int groupApiId = __atomic_fetch_add(&ctx->groupApiPoolIndex, 1, __ATOMIC_RELAXED);
    event = &ctx->groupApiPool[groupApiId%groupApiPoolSize];
    // Recycle the slot only if it is within the ring window AND its previous
    // owner has fully retired it (refCount==0). The acquire load pairs with the
    // ACQ_REL decrement in updateEvent, so out-of-order, cross-thread completion
    // can never make us reset/reuse a slot that is still live.
    if ((groupApiId - __atomic_load_n(&ctx->groupApiPoolBase, __ATOMIC_RELAXED)) < groupApiPoolSize &&
        __atomic_load_n(&event->refCount, __ATOMIC_ACQUIRE) == 0) {
      // if there are available group API events grab one
      // Make sure all child events of the picked group API event are cleared.
      // A group can span comms, so each child returns to its own context's pool.
      while (!profilerQueueEmpty(&event->collApiEvents)) {
        struct collApi *collApiEvent = profilerQueueDequeue(&event->collApiEvents);
        resetTaskEvents(collApiEvent, collApiEvent->ctx);
        __atomic_fetch_add(&collApiEvent->ctx->collApiPoolBase, 1, __ATOMIC_RELAXED);
      }
      while (!profilerQueueEmpty(&event->p2pApiEvents)) {
        struct p2pApi *p2pApiEvent = profilerQueueDequeue(&event->p2pApiEvents);
        resetTaskEvents(p2pApiEvent, p2pApiEvent->ctx);
        __atomic_fetch_add(&p2pApiEvent->ctx->p2pApiPoolBase, 1, __ATOMIC_RELAXED);
      }
      while (!profilerQueueEmpty(&event->kernelLaunchEvents)) {
        struct kernelLaunch *kernelLaunchEvent = profilerQueueDequeue(&event->kernelLaunchEvents);
        __atomic_fetch_add(&kernelLaunchEvent->ctx->kernelLaunchPoolBase, 1, __ATOMIC_RELAXED);
      }
    } else {
      // else drop this event
      __atomic_fetch_sub(&ctx->groupApiPoolIndex, 1, __ATOMIC_RELAXED);
      return ncclSuccess;
    }
    event->type = ncclProfileGroupApi;
    event->ctx = ctx;
    event->groupApiId = groupApiId;
    event->graphCaptured = eDescr->groupApi.graphCaptured;
    event->groupDepth = eDescr->groupApi.groupDepth;
    event->startTs = eventTime();
    *eHandle = event;
  } else if (eDescr->type == ncclProfileCollApi) {
    if (eDescr->parentObj == NULL) return ncclSuccess;
    struct collApi* event;
    int collApiId = __atomic_fetch_add(&ctx->collApiPoolIndex, 1, __ATOMIC_RELAXED);
    event = &ctx->collApiPool[collApiId%collApiPoolSize];
    // Recycle only a fully-retired slot (refCount==0, acquire). See ncclProfileGroupApi.
    if ((collApiId - __atomic_load_n(&ctx->collApiPoolBase, __ATOMIC_RELAXED)) < collApiPoolSize &&
        __atomic_load_n(&event->refCount, __ATOMIC_ACQUIRE) == 0) {
      // if there are available Coll API events grab one
      resetTaskEvents(event, ctx);
    } else {
      // else drop this event
      __atomic_fetch_sub(&ctx->collApiPoolIndex, 1, __ATOMIC_RELAXED);
      return ncclSuccess;
    }
    event->type = ncclProfileCollApi;
    event->collApiId = collApiId;
    event->ctx = ctx;
    event->func = eDescr->collApi.func;
    event->stream = (cudaStream_t) eDescr->collApi.stream;
    event->count = eDescr->collApi.count;
    event->datatype = eDescr->collApi.datatype;
    event->root = eDescr->collApi.root;
    event->graphCaptured = eDescr->collApi.graphCaptured;
    struct groupApi* parent = (struct groupApi *) eDescr->parentObj;
    event->parent = parent;
    profilerQueueEnqueue(&parent->collApiEvents, event);
    __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
    *eHandle = event;
  } else if (eDescr->type == ncclProfileP2pApi) {
    if (eDescr->parentObj == NULL) return ncclSuccess;
    struct p2pApi* event;
    int p2pApiId = __atomic_fetch_add(&ctx->p2pApiPoolIndex, 1, __ATOMIC_RELAXED);
    event = &ctx->p2pApiPool[p2pApiId%p2pApiPoolSize];
    // Recycle only a fully-retired slot (refCount==0, acquire). See ncclProfileGroupApi.
    if ((p2pApiId - __atomic_load_n(&ctx->p2pApiPoolBase, __ATOMIC_RELAXED)) < p2pApiPoolSize &&
        __atomic_load_n(&event->refCount, __ATOMIC_ACQUIRE) == 0) {
      // if there are available p2p API events grab one
      resetTaskEvents(event, ctx);
    } else {
      // else drop this event
      __atomic_fetch_sub(&ctx->p2pApiPoolIndex, 1, __ATOMIC_RELAXED);
      return ncclSuccess;
    }
    event->type = ncclProfileP2pApi;
    event->p2pApiId = p2pApiId;
    event->ctx = ctx;
    event->func = eDescr->p2pApi.func;
    event->stream = (cudaStream_t) eDescr->p2pApi.stream;
    event->count = eDescr->p2pApi.count;
    event->datatype = eDescr->p2pApi.datatype;
    event->graphCaptured = eDescr->p2pApi.graphCaptured;
    struct groupApi* parent = (struct groupApi *) eDescr->parentObj;
    event->parent = parent;
    profilerQueueEnqueue(&parent->p2pApiEvents, event);
    __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
    *eHandle = event;
  } else if (eDescr->type == ncclProfileKernelLaunch) {
    if (eDescr->parentObj == NULL) return ncclSuccess;
    struct kernelLaunch* event;
    int kernelLaunchId = __atomic_fetch_add(&ctx->kernelLaunchPoolIndex, 1, __ATOMIC_RELAXED);
    if ((kernelLaunchId - __atomic_load_n(&ctx->kernelLaunchPoolBase, __ATOMIC_RELAXED)) < kernelLaunchPoolSize) {
      // if there are available kernel API events grab one
      event = &ctx->kernelLaunchPool[kernelLaunchId%kernelLaunchPoolSize];
    } else {
      // else drop this event
      __atomic_fetch_sub(&ctx->kernelLaunchPoolIndex, 1, __ATOMIC_RELAXED);
      return ncclSuccess;
    }
    event->type = ncclProfileKernelLaunch;
    event->ctx = ctx;
    event->stream = (cudaStream_t) eDescr->kernelLaunch.stream;
    struct groupApi* parent = (struct groupApi *) eDescr->parentObj;
    event->parent = parent;
    profilerQueueEnqueue(&parent->kernelLaunchEvents, event);
    __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
    *eHandle = event;
  } else if (eDescr->type == ncclProfileGroup) {
    if (eDescr->parentObj == NULL) return ncclSuccess;
    struct group* event;
    int groupId = __atomic_fetch_add(&ctx->groupPoolIndex, 1, __ATOMIC_RELAXED);
    if ((groupId - __atomic_load_n(&ctx->groupPoolBase, __ATOMIC_RELAXED)) < groupPoolSize) {
      // if there are available group events grab one
      event = &ctx->groupPool[groupId%groupPoolSize];
      while (!taskEventQueueEmpty(event)) {
        struct taskEventBase* base = taskEventQueueDequeue(event);
        if (base->type == ncclProfileColl) {
          struct collective* c = (struct collective *)base;
          // reset event proxyOps & proxySteps
          memset(c->nProxyOps, 0, sizeof(int)*MAX_CHANNELS);
          // release collective events in the group and return them to the collective pool
          __atomic_fetch_add(&ctx->collPoolBase, 1, __ATOMIC_RELAXED);
        } else if (base->type == ncclProfileP2p) {
          struct p2p* p = (struct p2p *)base;
          // reset event proxyOp and proxySteps
          memset(&p->op, 0, sizeof(struct proxyOp)*MAX_CHANNELS);
          // release p2p events in the group and return them to the p2p pool
          __atomic_fetch_add(&ctx->p2pPoolBase, 1, __ATOMIC_RELAXED);
        }
      }
    } else {
      // else drop this event
      __atomic_fetch_sub(&ctx->groupPoolIndex, 1, __ATOMIC_RELAXED);
      return ncclSuccess;
    }
    event->type = ncclProfileGroup;
    event->ctx = ctx;
    event->groupId = groupId;
    event->startTs = eventTime();
    *eHandle = event;
    debugEvent(event, "GroupStart");
  } else if (eDescr->type == ncclProfileColl) {
    // the parent might be null if we run out of events
    struct collApi* parent = (struct collApi *)eDescr->parentObj;
    if (parent == NULL) return ncclSuccess;

    struct collective* event;
    int collId = __atomic_fetch_add(&ctx->collPoolIndex, 1, __ATOMIC_RELAXED);
    if ((collId - __atomic_load_n(&ctx->collPoolBase, __ATOMIC_RELAXED)) < collPoolSize) {
      // if there are available collective events grab one
      event = &ctx->collPool[collId%collPoolSize];
    } else {
      // else drop this event
      __atomic_fetch_sub(&ctx->collPoolIndex, 1, __ATOMIC_RELAXED);
      return ncclSuccess;
    }

    event->base.type = ncclProfileColl;
    event->base.rank = eDescr->rank;
    event->base.func = eDescr->coll.func;
    event->base.startTs = eventTime();
    event->base.parent = parent;
    event->seqNumber = eDescr->coll.seqNumber;
    event->sendBuff = eDescr->coll.sendBuff;
    event->recvBuff = eDescr->coll.recvBuff;
    event->count = eDescr->coll.count;
    event->root = eDescr->coll.root;
    event->datatype = eDescr->coll.datatype;
    event->nChannels = eDescr->coll.nChannels;
    event->nWarps = eDescr->coll.nWarps;
    event->algo = eDescr->coll.algo;
    event->proto = eDescr->coll.proto;
    event->kernelVariant = NULL;   // filled by the v7 startEvent when a v7 core is loaded
    event->isSymColl = false;
    *eHandle = event;
    taskEventQueueEnqueue(parent, (struct taskEventBase *)event);
    // increment the group ref counter so the event will stay open
    __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
    debugEvent(event, "CollStart");
  } else if (eDescr->type == ncclProfileP2p) {
    // the parent might be null if we run out of events
    struct p2pApi* parent = (struct p2pApi*) eDescr->parentObj;
    if (parent == NULL) return ncclSuccess;

    struct p2p* event;
    int p2pId = __atomic_fetch_add(&ctx->p2pPoolIndex, 1, __ATOMIC_RELAXED);
    if ((p2pId - __atomic_load_n(&ctx->p2pPoolBase, __ATOMIC_RELAXED)) < p2pPoolSize) {
      // if there are available p2p events grab one
      event = &ctx->p2pPool[p2pId%p2pPoolSize];
    } else {
      // else drop this event
      __atomic_fetch_sub(&ctx->p2pPoolIndex, 1, __ATOMIC_RELAXED);
      return ncclSuccess;
    }

    event->base.type = ncclProfileP2p;
    event->base.rank = eDescr->rank;
    event->base.func = eDescr->p2p.func;
    event->base.startTs = eventTime();
    event->base.parent = parent;
    event->buff = eDescr->p2p.buff;
    event->count = eDescr->p2p.count;
    event->datatype = eDescr->p2p.datatype;
    event->peer = eDescr->p2p.peer;
    event->nChannels = eDescr->p2p.nChannels;
    *eHandle = event;
    // increment the group ref counter so the event will staty open
    taskEventQueueEnqueue(parent, (struct taskEventBase *)event);
    __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
    debugEvent(event, "P2pStart");
  } else if (eDescr->type == ncclProfileProxyCtrl) {
    int proxyCtrlId = __atomic_fetch_add(&ctx->proxyCtrlPoolIndex, 1, __ATOMIC_RELAXED);
    struct proxyCtrl* event = &ctx->proxyCtrlPool[proxyCtrlId%proxyCtrlPoolSize];
    event->type = ncclProfileProxyCtrl;
    event->ctx = ctx;
    event->startTs = eventTime();
    *eHandle = event;
  } else if (eDescr->type == ncclProfileProxyOp) {
    // the eventBase might be null if we run out of events
    struct taskEventBase* eventBase = (struct taskEventBase *)eDescr->parentObj;
    if (eventBase == NULL) return ncclSuccess;

    if (eDescr->proxyOp.pid != pid) {
      // PXN captured proxyOp events
      struct proxyOp* event;
      int detachId = __atomic_fetch_add(&detachPoolIndex, 1, __ATOMIC_RELAXED);
      if ((detachId - detachPoolBase) < detachPoolSize) {
        // if there are available detached proxyOp events grab one
        event = &detachPool[detachId%detachPoolSize];
      } else {
        // else drop this event
        __atomic_fetch_sub(&detachPoolIndex, 1, __ATOMIC_RELAXED);
        return ncclSuccess;
      }

      event->type = ncclProfileProxyOp;
      event->channelId = eDescr->proxyOp.channelId;
      event->pid = eDescr->proxyOp.pid;
      event->rank = eDescr->rank;
      event->peer = eDescr->proxyOp.peer;
      event->nSteps = eDescr->proxyOp.nSteps;
      event->chunkSize = eDescr->proxyOp.chunkSize;
      event->isSend = eDescr->proxyOp.isSend;
      event->startTs = eventTime();
      event->parent = NULL;
      event->stepCount = 0;
      *eHandle = event;
      debugEvent(event, "PxnProxyOpStart");
      return ncclSuccess;
    }

    if (eventBase->type == ncclProfileColl) {
      struct collective* parent = (struct collective *)eDescr->parentObj;
      int channelId = eDescr->proxyOp.channelId;
      if (channelId >= MAX_CHANNELS) return ncclSuccess;
      struct proxyOp* event = &parent->op[channelId][parent->nProxyOps[channelId]++];

      event->type = ncclProfileProxyOp;
      event->channelId = channelId;
      event->pid = eDescr->proxyOp.pid;
      event->rank = eDescr->rank;
      event->peer = eDescr->proxyOp.peer;
      event->nSteps = eDescr->proxyOp.nSteps;
      event->chunkSize = eDescr->proxyOp.chunkSize;
      event->isSend = eDescr->proxyOp.isSend;
      event->parent = eventBase;
      event->startTs = eventTime();
      event->stepCount = 0;
      *eHandle = event;
      __atomic_fetch_add(&parent->base.refCount, 1, __ATOMIC_RELAXED);
      debugEvent(event, "ProxyOpStart");
    } else { // ncclProfileP2p
      struct p2p* parent = (struct p2p *)eDescr->parentObj;
      int channelId = eDescr->proxyOp.channelId;
      if (channelId >= MAX_CHANNELS) return ncclSuccess;
      struct proxyOp* event = &parent->op[channelId];
      event->type = ncclProfileProxyOp;
      event->channelId = channelId;
      event->pid = eDescr->proxyOp.pid;
      event->rank = eDescr->rank;
      event->peer = eDescr->proxyOp.peer;
      event->nSteps = eDescr->proxyOp.nSteps;
      event->chunkSize = eDescr->proxyOp.chunkSize;
      event->isSend = eDescr->proxyOp.isSend;
      event->parent = eventBase;
      event->startTs = eventTime();
      event->stepCount = 0;
      *eHandle = event;
      __atomic_fetch_add(&parent->base.refCount, 1, __ATOMIC_RELAXED);
      debugEvent(event, "ProxyOpStart");
    }
  } else if (eDescr->type == ncclProfileProxyStep) {
    // the parent might be null if we run out of events
    struct proxyOp* parent = (struct proxyOp *)eDescr->parentObj;
    if (parent == NULL) return ncclSuccess;

    int s = parent->stepCount++ % MAX_STEPS;
    struct proxyStep* event = &parent->step[s];
    event->type = ncclProfileProxyStep;
    event->state = 0;
    event->step = eDescr->proxyStep.step;
    event->parent = parent;
    event->isSend = parent->isSend;
    event->startTs = eventTime();
    event->nNetEvents = 0;
    *eHandle = event;
    debugEvent(event, "ProxyStepStart");
  } else if (eDescr->type == ncclProfileKernelCh) {
    struct taskEventBase* eventBase = (struct taskEventBase *)eDescr->parentObj;
    if (eventBase == NULL) return ncclSuccess;
    if (eDescr->kernelCh.channelId >= MAX_CHANNELS) return ncclSuccess;
    if (eventBase->type == ncclProfileColl) {
      struct collective* parent = (struct collective *)eDescr->parentObj;
      struct kernelCh* event = &parent->kernel[eDescr->kernelCh.channelId];
      event->type = ncclProfileKernelCh;
      event->channelId = eDescr->kernelCh.channelId;
      event->startGpuClk = eDescr->kernelCh.pTimer;
      event->parent = eventBase;
      event->startTs = eventTime();
      memset(event->phases, 0, sizeof(event->phases)); // clear stale v7 phase data on slot reuse
      *eHandle = event;
      __atomic_fetch_add(&parent->base.refCount, 1, __ATOMIC_RELAXED);
      debugEvent(event, "KernelChStart");
    } else { // ncclProfileP2p
      struct p2p* parent = (struct p2p *)eDescr->parentObj;
      struct kernelCh* event = &parent->kernel[eDescr->kernelCh.channelId];
      event->type = ncclProfileKernelCh;
      event->channelId = eDescr->kernelCh.channelId;
      event->startGpuClk = eDescr->kernelCh.pTimer;
      event->parent = eventBase;
      event->startTs = eventTime();
      memset(event->phases, 0, sizeof(event->phases)); // clear stale v7 phase data on slot reuse
      *eHandle = event;
      __atomic_fetch_add(&parent->base.refCount, 1, __ATOMIC_RELAXED);
      debugEvent(event, "KernelChStart");
    }
  } else if (eDescr->type == ncclProfileNetPlugin) {
    struct proxyStep* parent = (struct proxyStep *)eDescr->parentObj;
    if (parent == NULL) return ncclSuccess;

    int64_t pluginId = eDescr->netPlugin.id;
    int64_t type = pluginId & NCCL_PROFILER_NET_TYPE_MASK;
    int64_t ver = pluginId & NCCL_PROFILER_NET_VER_MASK;
    if (type == NCCL_PROFILER_NET_TYPE_IB) {
      if (ver == 1) {
        ncclProfilerNetIbDescr_v1_t* descr = (ncclProfilerNetIbDescr_v1_t *)eDescr->netPlugin.data;
        struct netPlugin* event = parent->net + __atomic_fetch_add(&parent->nNetEvents, 1, __ATOMIC_RELAXED);
        event->type = ncclProfileNetPlugin;
        event->pluginType = type;
        event->pluginVer = ver;
        if (descr->type == ncclProfileQp) {
          event->pluginEvent = ncclProfileQp;
          event->qp.device = descr->qp.device;
          event->qp.wr_id = descr->qp.wr_id;
          event->qp.opcode = descr->qp.opcode;
          event->qp.qpNum = descr->qp.qpNum;
          event->qp.length = descr->qp.length;
        }
        event->startTs = eventTime();
        *eHandle = event;
        debugEvent(event, "NetPluginStart");
      }
    } else if (type == NCCL_PROFILER_NET_TYPE_SOCK) {
      if (ver == 1) {
        ncclProfilerNetSockDescr_v1_t* descr = (ncclProfilerNetSockDescr_v1_t *)eDescr->netPlugin.data;
        struct netPlugin* event = parent->net + __atomic_fetch_add(&parent->nNetEvents, 1, __ATOMIC_RELAXED);
        event->type = ncclProfileNetPlugin;
        event->pluginType = type;
        event->pluginVer = ver;
        if (descr->type == ncclProfileSocket) {
          event->pluginEvent = ncclProfileSocket;
          event->sock.fd = descr->sock.fd;
          event->sock.op = descr->sock.op;
          event->sock.length = descr->sock.length;
        }
        event->startTs = eventTime();
        *eHandle = event;
        debugEvent(event, "NetPluginStart");
      }
    }
  }
  return ncclSuccess;
}

void updateEvent(void* handle) {
  uint64_t type = *(uint64_t *)handle;
  if (type == ncclProfileGroupApi) {
    struct groupApi* event = (struct groupApi*) handle;
    if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_ACQ_REL) == 0) {
      event->stopTs = eventTime();
      __atomic_fetch_add(&event->ctx->groupApiPoolBase, 1, __ATOMIC_RELAXED);
    }
  } else if (type == ncclProfileCollApi) {
    struct collApi* event = (struct collApi*) handle;
    if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_ACQ_REL) == 0) {
      event->stopTs = eventTime();
      __atomic_fetch_add(&event->ctx->collApiPoolBase, 1, __ATOMIC_RELAXED);
    }
    updateEvent(event->parent);
    return;
  } else if (type == ncclProfileP2pApi) {
    struct p2pApi* event = (struct p2pApi*) handle;
    if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_ACQ_REL) == 0) {
      event->stopTs = eventTime();
      __atomic_fetch_add(&event->ctx->p2pApiPoolBase, 1, __ATOMIC_RELAXED);
    }
    updateEvent(event->parent);
    event->stopTs = eventTime();
  } else if (type == ncclProfileKernelLaunch) {
    struct kernelLaunch* event = (struct kernelLaunch*) handle;
    event->stopTs = eventTime();
    updateEvent(event->parent);
  } else if (type == ncclProfileGroup) {
    struct group* event = (struct group *)handle;
    if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_ACQ_REL) == 0) {
      event->stopTs = eventTime();
      // return group event to the pool
      __atomic_fetch_add(&event->ctx->groupPoolBase, 1, __ATOMIC_RELAXED);
    }
    debugEvent(event, "GroupStop");
  } else if (type == ncclProfileColl) {
    struct collective* event = (struct collective *)handle;
    if (__atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_ACQ_REL) == 0) {
      event->base.stopTs = eventTime();
      debugEvent(event, "CollStop");
      updateEvent(event->base.parent);
      return;
    }
    debugEvent(event, "CollStop");
  } else if (type == ncclProfileP2p) {
    struct p2p* event = (struct p2p *)handle;
    if (__atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_ACQ_REL) == 0) {
      event->base.stopTs = eventTime();
      debugEvent(event, "P2pStop");
      updateEvent(event->base.parent);
      return;
    }
    debugEvent(event, "P2pStop");
  } else if (type == ncclProfileProxyOp) {
    struct proxyOp* event = (struct proxyOp *)handle;
    event->stopTs = eventTime();
    if (event->pid != pid) {
      // only for proxyOps that don't have a parent collective/p2p (i.e., PXN)
      int done = __atomic_add_fetch(&detachPoolDone, 1, __ATOMIC_RELAXED);
      if (done == detachPoolSize) {
        // reset the event completed (done) counter
        __atomic_store_n(&detachPoolDone, 0, __ATOMIC_RELAXED);
        // update the base pointer to the top of the pool
        int index = __atomic_load_n(&detachPoolIndex, __ATOMIC_RELAXED);
        __atomic_store_n(&detachPoolBase, index, __ATOMIC_RELAXED);
      }
      debugEvent(event, "ProxyOpStop");
      return;
    }
    updateEvent(event->parent);
    debugEvent(event, "ProxyOpStop");
  } else if (type == ncclProfileProxyStep) {
    struct proxyStep* event = (struct proxyStep *)handle;
    event->stopTs = eventTime();
    debugEvent(event, "ProxyStepStop");
  } else if (type == ncclProfileProxyCtrl) {
    struct proxyCtrl* event = (struct proxyCtrl *)handle;
    event->stopTs = eventTime();
    debugEvent(event, "ProxyCtrlStop");
  } else if (type == ncclProfileKernelCh) {
    struct kernelCh* event = (struct kernelCh *)handle;
    event->stopTs = eventTime();
    updateEvent(event->parent);
    debugEvent(event, "KernelChStop");
  } else if (type == ncclProfileNetPlugin) {
    struct netPlugin* event = (struct netPlugin *)handle;
    event->stopTs = eventTime();
    debugEvent(event, "NetPluginStop");
  }
}

__hidden ncclResult_t exampleProfilerStopEvent(void* eHandle) {
  // the event handle might be null if we run out of events
  if (eHandle == NULL) return ncclSuccess;
  struct context* eventCtx = contextFromEventHandle(eHandle);
  if (eventCtx && __atomic_load_n(&eventCtx->finalizing, __ATOMIC_RELAXED)) {
    return ncclSuccess;
  }

  uint64_t type = *(uint64_t *)eHandle;
  // Stopping API events, Kernel Launch events, collective/p2p task events
  // in NCCL core do not mean that they are complete. It means that the
  // operation was enqueued so we need to keep the events open
  if (type == ncclProfileGroupApi) {
    struct groupApi* event = (struct groupApi*) eHandle;
    event->stopTs = eventTime();
    return ncclSuccess;
  } else if (type == ncclProfileCollApi) {
    struct collApi* event = (struct collApi*) eHandle;
    event->stopTs = eventTime();
    return ncclSuccess;
  } else if (type == ncclProfileP2pApi) {
    struct p2pApi* event = (struct p2pApi*) eHandle;
    event->stopTs = eventTime();
    return ncclSuccess;
  } else if (type == ncclProfileKernelLaunch) {
    struct kernelLaunch* event = (struct kernelLaunch*) eHandle;
    event->stopTs = eventTime();
    return ncclSuccess;
  } else if (type == ncclProfileGroup) {
    struct group* event = (struct group *)eHandle;
    event->stopTs = eventTime();
    return ncclSuccess;
  } else if (type == ncclProfileColl) {
    struct collective* event = (struct collective *)eHandle;
    event->base.stopTs = eventTime();
    return ncclSuccess;
  } else if (type == ncclProfileP2p) {
    struct p2p* event = (struct p2p *)eHandle;
    event->base.stopTs = eventTime();
    return ncclSuccess;
  }

  updateEvent(eHandle);
  return ncclSuccess;
}

__hidden ncclResult_t exampleProfilerRecordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) {
  // the event handle might be null if we run out of events
  if (eHandle == NULL) return ncclSuccess;
  struct context* eventCtx = contextFromEventHandle(eHandle);
  if (eventCtx && __atomic_load_n(&eventCtx->finalizing, __ATOMIC_RELAXED)) {
    return ncclSuccess;
  }

  uint64_t type = *(uint64_t *)eHandle;
  if (type == ncclProfileGroupApi) {
    struct groupApi* event = (struct groupApi*) eHandle;
    if (eState == ncclProfilerGroupEndApiStart) {
      event->endOfncclGroupStartTs = eventTime();
    } else if (eState == ncclProfilerGroupStartApiStop) {
      event->startOfncclGroupEndTs = eventTime();
    }
  } else if (type == ncclProfileProxyOp) {
    struct proxyOp* event = (struct proxyOp *)eHandle;
    if (eState == ncclProfilerProxyOpInProgress_v4) {
      event->progrTs = eventTime();
    }
  } else if (type == ncclProfileProxyStep) {
    struct proxyStep* event = (struct proxyStep *)eHandle;
    struct proxyOp* parent = event->parent;
    switch (eState) {
    case ncclProfilerProxyStepSendGPUWait:
      event->timestamp[PROXY_STEP_SEND_GPU_WAIT] = eventTime();
      break;
    case ncclProfilerProxyStepSendPeerWait_v4:
      // do not update step event if in SendPeerWait
      if (event->state == ncclProfilerProxyStepSendPeerWait_v4) break;
      event->timestamp[PROXY_STEP_SEND_PEER_WAIT] = eventTime();
      event->state = ncclProfilerProxyStepSendPeerWait_v4;
      break;
    case ncclProfilerProxyStepSendWait:
      event->timestamp[PROXY_STEP_SEND_WAIT] = eventTime();
      parent->transSize += eStateArgs->proxyStep.transSize;
      break;
    case ncclProfilerProxyStepRecvWait:
      event->timestamp[PROXY_STEP_RECV_WAIT] = eventTime();
      break;
    case ncclProfilerProxyStepRecvFlushWait:
      event->timestamp[PROXY_STEP_RECV_FLUSH_WAIT] = eventTime();
      parent->transSize += eStateArgs->proxyStep.transSize;
      break;
    case ncclProfilerProxyStepRecvGPUWait:
      event->timestamp[PROXY_STEP_RECV_GPU_WAIT] = eventTime();
      break;
    default:
      break;
    }
  } else if (type == ncclProfileProxyCtrl) {
    struct proxyCtrl* event = (struct proxyCtrl *)eHandle;
    if (eState == ncclProfilerProxyCtrlAppendEnd) {
      event->appended = eStateArgs->proxyCtrl.appendedProxyOps;
    }
    event->state = eState;
  } else if (type == ncclProfileKernelCh) {
    struct kernelCh* event = (struct kernelCh *)eHandle;
    if (eState == ncclProfilerKernelChStop) {
      event->stopGpuClk = eStateArgs->kernelCh.pTimer;
    }
  }
  debugEvent(eHandle, "RecordEventState");
  return ncclSuccess;
}

ncclProfiler_t ncclProfiler_v5 = {
  "Example-profiler",
  exampleProfilerInit,
  exampleProfilerStartEvent,
  exampleProfilerStopEvent,
  exampleProfilerRecordEventState,
  exampleProfilerFinalize,
};

__attribute__((visibility("default"))) int exampleProfilerStart(int eActivationMask, const char* name) {
  profilerDumpFile = name;
  if (__atomic_load_n(&initialized, __ATOMIC_RELAXED)) {
    __atomic_store_n(eActivationMaskPtr, eActivationMask, __ATOMIC_RELAXED);
  }
  return ncclSuccess;
}

__attribute__((visibility("default"))) int exampleProfilerStop(void) {
  if (__atomic_load_n(&initialized, __ATOMIC_RELAXED)) {
    __atomic_store_n(eActivationMaskPtr, 0, __ATOMIC_RELAXED);
  }
  return ncclSuccess;
}

// ============================================================================
// v6 implementation with CE events support
// ============================================================================

#include "nccl/profiler_v6.h"

__hidden ncclResult_t exampleProfilerStartEvent_v6(void* context, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr) {
  struct context* ctx = (struct context*)context;

  if (ctx == NULL) {
    *eHandle = NULL;
    return ncclSuccess;
  }
  if (eDescr->type == ncclProfileCeColl) {
    return ceProfilerStartCeCollEvent(ctx, eHandle, eDescr, startTime);
  }

  if (eDescr->type == ncclProfileCeSync) {
    return ceProfilerStartCeSyncEvent(ctx, eHandle, eDescr, startTime);
  }

  if (eDescr->type == ncclProfileCeBatch) {
    return ceProfilerStartCeBatchEvent(ctx, eHandle, eDescr, startTime);
  }

  return exampleProfilerStartEvent(context, eHandle, (ncclProfilerEventDescr_t*)eDescr);
}

__hidden ncclResult_t exampleProfilerStopEvent_v6(void* eHandle) {
  if (!eHandle) return ncclSuccess;

  uint64_t type = *(uint64_t*)eHandle;

  // Handle CE events - record stop event to stream
  if (type == ncclProfileCeColl) {
    return ceProfilerStopCeCollEvent(eHandle);
  }

  if (type == ncclProfileCeSync) {
    return ceProfilerStopCeSyncEvent(eHandle);
  }

  if (type == ncclProfileCeBatch) {
    return ceProfilerStopCeBatchEvent(eHandle);
  }

  // Fall through to v5 handling for non-CE events
  return exampleProfilerStopEvent(eHandle);
}

__hidden ncclResult_t exampleProfilerRecordEventState_v6(void* eHandle, ncclProfilerEventState_v6_t eState, ncclProfilerEventStateArgs_v6_t* eStateArgs) {
  // CE events don't use recordEventState - poller handles all timing
  // Just fall through to v5 for non-CE events
  return exampleProfilerRecordEventState(eHandle, (ncclProfilerEventState_t)eState, (ncclProfilerEventStateArgs_t*)eStateArgs);
}

ncclProfiler_v6_t ncclProfiler_v6 = {
  "Example-profiler-v6",
  exampleProfilerInit,
  exampleProfilerStartEvent_v6,
  exampleProfilerStopEvent_v6,
  exampleProfilerRecordEventState_v6,
  exampleProfilerFinalize,
};

// ============================================================================
// v7 implementation: kernel phase sub-events + symmetric-kernel coll metadata
// ============================================================================

#include "nccl/profiler_v7.h"

__hidden ncclResult_t exampleProfilerStartEvent_v7(void* context, void** eHandle, ncclProfilerEventDescr_v7_t* eDescr) {
  struct context* ctx = (struct context*)context;
  if (ctx == NULL) { *eHandle = NULL; return ncclSuccess; }

  // v7 adds the kernel phase sub-event: a child of a kernelCh, stored in the parent.
  if (eDescr->type == ncclProfileKernelPhase) {
    *eHandle = NULL;
    if (__atomic_load_n(&ctx->finalizing, __ATOMIC_RELAXED)) return ncclSuccess;
    struct kernelCh* parent = (struct kernelCh*)eDescr->parentObj;
    if (parent == NULL || parent->type != ncclProfileKernelCh) return ncclSuccess;
    int idx = eDescr->kernelPhase.phaseId;
    if (idx < 0 || idx >= MAX_KERNEL_PHASES) return ncclSuccess;
    struct kernelPhase* phase = &parent->phases[idx];
    phase->type = ncclProfileKernelPhase;
    phase->channelId = eDescr->kernelPhase.channelId;
    phase->phaseId = eDescr->kernelPhase.phaseId;
    phase->phaseName = eDescr->kernelPhase.phaseName;
    phase->startGpuClk = eDescr->kernelPhase.pTimer;
    phase->stopGpuClk = 0;
    phase->startTs = eventTime();
    phase->parent = parent;
    *eHandle = phase;
    return ncclSuccess;
  }

  // Other events: the v6 coll layout is a prefix of v7, so the v6 handler reads it
  // safely. Capture the v7-only coll metadata on the created event afterwards.
  ncclResult_t ret = exampleProfilerStartEvent_v6(context, eHandle, (ncclProfilerEventDescr_v6_t*)eDescr);
  if (eDescr->type == ncclProfileColl && *eHandle) {
    struct collective* c = (struct collective*)*eHandle;
    c->kernelVariant = eDescr->coll.kernelVariant;
    c->isSymColl = eDescr->coll.isSymColl;
  }
  return ret;
}

__hidden ncclResult_t exampleProfilerStopEvent_v7(void* eHandle) {
  if (eHandle == NULL) return ncclSuccess;
  if (*(uint64_t*)eHandle == ncclProfileKernelPhase) {
    struct kernelPhase* phase = (struct kernelPhase*)eHandle;
    phase->stopTs = eventTime();
    return ncclSuccess;
  }
  return exampleProfilerStopEvent_v6(eHandle);
}

__hidden ncclResult_t exampleProfilerRecordEventState_v7(void* eHandle, ncclProfilerEventState_v7_t eState, ncclProfilerEventStateArgs_v7_t* eStateArgs) {
  if (eHandle == NULL) return ncclSuccess;
  if (*(uint64_t*)eHandle == ncclProfileKernelPhase) {
    if (eState == ncclProfilerKernelPhaseStop) {
      struct kernelPhase* phase = (struct kernelPhase*)eHandle;
      phase->stopGpuClk = eStateArgs->kernelCh.pTimer;
    }
    return ncclSuccess;
  }
  return exampleProfilerRecordEventState_v6(eHandle, (ncclProfilerEventState_v6_t)eState, (ncclProfilerEventStateArgs_v6_t*)eStateArgs);
}

ncclProfiler_v7_t ncclProfiler_v7 = {
  "Example-profiler-v7",
  exampleProfilerInit,
  exampleProfilerStartEvent_v7,
  exampleProfilerStopEvent_v7,
  exampleProfilerRecordEventState_v7,
  exampleProfilerFinalize,
};

// ============================================================================
// v8 implementation: NCCL stamps authoritative timestamps on every event.
//
// This example forwards to the v7 handlers above. v8 inserts timestamp fields
// ahead of the descriptor union, so it is NOT layout-compatible with v7 -- we
// field-copy v8 -> v7 here rather than reinterpret-cast. (A real plugin would
// consume eDescr->tsCpu and the stop descriptor; the inspector plugin does.)
// ============================================================================

#include "nccl/profiler_v8.h"

__hidden ncclResult_t exampleProfilerStartEvent_v8(void* context, void** eHandle, ncclProfilerEventDescr_v8_t* eDescr) {
  ncclProfilerEventDescr_v7_t d7 = { };
  d7.type = eDescr->type;
  d7.parentObj = eDescr->parentObj;
  d7.rank = eDescr->rank;
  switch (eDescr->type) {
  case ncclProfileGroup:
  case ncclProfileProxyCtrl:
    break;
  case ncclProfileGroupApi:
    d7.groupApi.graphCaptured = eDescr->groupApi.graphCaptured;
    d7.groupApi.groupDepth = eDescr->groupApi.groupDepth;
    break;
  case ncclProfileCollApi:
    d7.collApi.func = eDescr->collApi.func;
    d7.collApi.count = eDescr->collApi.count;
    d7.collApi.datatype = eDescr->collApi.datatype;
    d7.collApi.root = eDescr->collApi.root;
    d7.collApi.stream = eDescr->collApi.stream;
    d7.collApi.graphCaptured = eDescr->collApi.graphCaptured;
    d7.collApi.userTag = eDescr->collApi.userTag;
    break;
  case ncclProfileP2pApi:
    d7.p2pApi.func = eDescr->p2pApi.func;
    d7.p2pApi.count = eDescr->p2pApi.count;
    d7.p2pApi.datatype = eDescr->p2pApi.datatype;
    d7.p2pApi.stream = eDescr->p2pApi.stream;
    d7.p2pApi.graphCaptured = eDescr->p2pApi.graphCaptured;
    d7.p2pApi.userTag = eDescr->p2pApi.userTag;
    break;
  case ncclProfileKernelLaunch:
    d7.kernelLaunch.stream = eDescr->kernelLaunch.stream;
    break;
  case ncclProfileColl:
    d7.coll.seqNumber = eDescr->coll.seqNumber;
    d7.coll.func = eDescr->coll.func;
    d7.coll.sendBuff = eDescr->coll.sendBuff;
    d7.coll.recvBuff = eDescr->coll.recvBuff;
    d7.coll.count = eDescr->coll.count;
    d7.coll.root = eDescr->coll.root;
    d7.coll.datatype = eDescr->coll.datatype;
    d7.coll.nChannels = eDescr->coll.nChannels;
    d7.coll.nWarps = eDescr->coll.nWarps;
    d7.coll.algo = eDescr->coll.algo;
    d7.coll.proto = eDescr->coll.proto;
    d7.coll.parentGroup = eDescr->coll.parentGroup;
    d7.coll.kernelVariant = eDescr->coll.kernelVariant;
    d7.coll.isSymColl = eDescr->coll.isSymColl;
    d7.coll.userTag = eDescr->coll.userTag;
    break;
  case ncclProfileP2p:
    d7.p2p.func = eDescr->p2p.func;
    d7.p2p.buff = eDescr->p2p.buff;
    d7.p2p.datatype = eDescr->p2p.datatype;
    d7.p2p.count = eDescr->p2p.count;
    d7.p2p.peer = eDescr->p2p.peer;
    d7.p2p.nChannels = eDescr->p2p.nChannels;
    d7.p2p.parentGroup = eDescr->p2p.parentGroup;
    d7.p2p.userTag = eDescr->p2p.userTag;
    break;
  case ncclProfileProxyOp:
    d7.proxyOp.pid = eDescr->proxyOp.pid;
    d7.proxyOp.channelId = eDescr->proxyOp.channelId;
    d7.proxyOp.peer = eDescr->proxyOp.peer;
    d7.proxyOp.nSteps = eDescr->proxyOp.nSteps;
    d7.proxyOp.chunkSize = eDescr->proxyOp.chunkSize;
    d7.proxyOp.isSend = eDescr->proxyOp.isSend;
    break;
  case ncclProfileProxyStep:
    d7.proxyStep.step = eDescr->proxyStep.step;
    break;
  case ncclProfileKernelCh:
    d7.kernelCh.channelId = eDescr->kernelCh.channelId;
    d7.kernelCh.pTimer = eDescr->kernelCh.pTimer;
    break;
  case ncclProfileKernelPhase:
    d7.kernelPhase.channelId = eDescr->kernelPhase.channelId;
    d7.kernelPhase.phaseId = eDescr->kernelPhase.phaseId;
    d7.kernelPhase.phaseName = eDescr->kernelPhase.phaseName;
    d7.kernelPhase.pTimer = eDescr->kernelPhase.pTimer;
    break;
  case ncclProfileNetPlugin:
    d7.netPlugin.id = eDescr->netPlugin.id;
    d7.netPlugin.data = eDescr->netPlugin.data;
    break;
  case ncclProfileCeColl:
    d7.ceColl.seqNumber = eDescr->ceColl.seqNumber;
    d7.ceColl.func = eDescr->ceColl.func;
    d7.ceColl.sendBuff = eDescr->ceColl.sendBuff;
    d7.ceColl.recvBuff = eDescr->ceColl.recvBuff;
    d7.ceColl.count = eDescr->ceColl.count;
    d7.ceColl.root = eDescr->ceColl.root;
    d7.ceColl.datatype = eDescr->ceColl.datatype;
    d7.ceColl.syncStrategy = eDescr->ceColl.syncStrategy;
    d7.ceColl.intraBatchSync = eDescr->ceColl.intraBatchSync;
    d7.ceColl.batchSize = eDescr->ceColl.batchSize;
    d7.ceColl.numBatches = eDescr->ceColl.numBatches;
    d7.ceColl.ceSeqNum = eDescr->ceColl.ceSeqNum;
    d7.ceColl.stream = eDescr->ceColl.stream;
    break;
  case ncclProfileCeSync:
    d7.ceCollSync.isComplete = eDescr->ceCollSync.isComplete;
    d7.ceCollSync.nRanks = eDescr->ceCollSync.nRanks;
    break;
  case ncclProfileCeBatch:
    d7.ceCollBatch.numOps = eDescr->ceCollBatch.numOps;
    d7.ceCollBatch.totalBytes = eDescr->ceCollBatch.totalBytes;
    d7.ceCollBatch.useIntraSync = eDescr->ceCollBatch.useIntraSync;
    break;
  default:
    *eHandle = NULL;
    return ncclSuccess;
  }
  ncclStampedTs = ncclTsToLocal(eDescr->tsCpu);
  ncclResult_t res = exampleProfilerStartEvent_v7(context, eHandle, &d7);
  ncclStampedTs = 0.0;
  return res;
}

__hidden ncclResult_t exampleProfilerStopEvent_v8(void* eHandle, ncclProfilerEventDescr_v8_t* eStopDescr) {
  ncclStampedTs = eStopDescr ? ncclTsToLocal(eStopDescr->tsCpu) : 0.0;
  ncclResult_t res = exampleProfilerStopEvent_v7(eHandle);
  ncclStampedTs = 0.0;
  return res;
}

__hidden ncclResult_t exampleProfilerRecordEventState_v8(void* eHandle, ncclProfilerEventState_v8_t eState,
                                                         ncclProfilerEventStateArgs_v8_t* eStateArgs) {
  // v8 puts the timestamp ahead of the union, so these are not layout
  // compatible: field-copy the payload rather than reinterpret-cast.
  ncclProfilerEventStateArgs_v7_t a7 = { };
  if (eStateArgs) {
    switch (eState) {
    case ncclProfilerProxyStepSendWait:
    case ncclProfilerProxyStepRecvFlushWait:
      a7.proxyStep.transSize = eStateArgs->proxyStep.transSize;
      break;
    case ncclProfilerProxyCtrlAppendEnd:
      a7.proxyCtrl.appendedProxyOps = eStateArgs->proxyCtrl.appendedProxyOps;
      break;
    case ncclProfilerNetPluginUpdate:
      a7.netPlugin.data = eStateArgs->netPlugin.data;
      break;
    case ncclProfilerKernelChStop:
    case ncclProfilerKernelPhaseStop:
      a7.kernelCh.pTimer = eStateArgs->kernelCh.pTimer;
      break;
    default:
      break;
    }
    ncclStampedTs = ncclTsToLocal(eStateArgs->tsCpu);
  }
  ncclResult_t res = exampleProfilerRecordEventState_v7(eHandle, (ncclProfilerEventState_v7_t)eState, &a7);
  ncclStampedTs = 0.0;
  return res;
}

__hidden ncclResult_t exampleProfilerInit_v8(void** context, uint64_t commId, int* eActivationMask,
                                             const char* commName, int nNodes, int nranks, int rank,
                                             ncclDebugLogger_t logfn, ncclProfilerConfig_v8_t* config) {
  // The feature handshake. Monotonic because this profiler's timeline is
  // steady_clock (see gettime()), so an NCCL tsCpu needs nothing but ns -> us
  // and rebasing; picking Realtime here would put the two on different bases.
  // gpuClockSync additionally lands KernelCh and KernelPhase on that same
  // timeline, which is what lets kernel events be compared against host events
  // instead of living in raw globaltimer units.
  //
  // Opting in only changes where the numbers come from: every timestamp still
  // goes through eventTime(), which falls back to reading the clock here for
  // anything NCCL leaves at NCCL_PROFILER_TS_NONE (CE events, or any event
  // NCCL could not stamp).
  if (config) {
    config->cpuClock = ncclProfilerCpuClockMonotonic;
    config->gpuClockSync = 1;
  }
  return exampleProfilerInit(context, commId, eActivationMask, commName, nNodes, nranks, rank, logfn);
}

ncclProfiler_v8_t ncclProfiler_v8 = {
  "Example-profiler-v8",
  exampleProfilerInit_v8,
  exampleProfilerStartEvent_v8,
  exampleProfilerStopEvent_v8,
  exampleProfilerRecordEventState_v8,
  exampleProfilerFinalize,
};

