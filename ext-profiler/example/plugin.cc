/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include <pthread.h>
#include <cstring>
#include <linux/limits.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
#include "event.h"
#include "print_event.h"

#define __hidden __attribute__ ((visibility("hidden")))

static int initialized;             // initialization counter for profiler
static double startTime;            // profiler start time

static const int defaultEActivationMask = ncclProfileColl | ncclProfileP2p;
static const int defaultGroupApiPoolSize = 256;
static const int defaultCollApiPoolSize = 256;
static const int defaultP2pApiPoolSize = 256;
static const int defaultKernelLaunchPoolSize = 256;
static const int defaultGroupPoolSize = 256;
static const int defaultCollPoolSize = 256;
static const int defaultP2pPoolSize = 256;
static const int defaultProxyCtrlPoolSize = 16;
static const int defaultDetachPoolSize = 256;

static int groupApiPoolSize;
static int collApiPoolSize;
static int p2pApiPoolSize;
static int kernelLaunchPoolSize;
static int groupPoolSize;
static int collPoolSize;
static int p2pPoolSize;
static int proxyCtrlPoolSize;
static int detachPoolSize;
static int detachPoolBase;
static int detachPoolIndex;
static int detachPoolDone;
static struct proxyOp* detachPool;

ncclDebugLogger_t logFn;
#define INFO(FLAGS, ...) logFn(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

__hidden double gettime(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (t.tv_sec*1e6 + (t.tv_nsec*1e-3));
}

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pid_t pid;
static int* eActivationMaskPtr;

__hidden ncclResult_t exampleProfilerInit(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes, int nranks, int rank, ncclDebugLogger_t logfn) {
  pthread_mutex_lock(&lock);
  if (__atomic_fetch_add(&initialized, 1, __ATOMIC_RELAXED) == 0) {
    // first thread initializes event mask, environment and detach pool
    const char* str;
    str = getenv("NCCL_PROFILE_EVENT_MASK");
    __atomic_store_n(eActivationMask, str ? atoi(str) : 0, __ATOMIC_RELAXED);

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

    str = getenv("NCCL_PROFILE_PROXY_DETACH_POOL_SIZE");
    detachPoolSize = str ? atoi(str) : defaultDetachPoolSize;

    // detach pool is used to store PXN proxyOps and is shared among threads
    detachPool = (struct proxyOp *)calloc(detachPoolSize, sizeof(*detachPool));
    if (detachPool == NULL) {
      pthread_mutex_unlock(&lock);
      return ncclSystemError;
    }
    // Pid of the process initializing the profiler first.
    // This is compared against the pid of proxyOp events
    // to figure out if they have a parent event in this
    // process address space.
    pid = getpid();

    startTime = gettime();
  }
  pthread_mutex_unlock(&lock);

  // store pointer to activation mask globally
  eActivationMaskPtr = eActivationMask;

  // pre-allocate memory for event object pools in dedicated profiler context
  struct context* ctx = (struct context *)calloc(1, sizeof(*ctx));
  ctx->commName = commName;
  ctx->commHash = commId;
  ctx->nranks = nranks;
  ctx->rank = rank;
  logFn = logfn;
  INFO(NCCL_INIT, "PROFILER/Plugin: init commName: %s commHash: %lu nranks: %d rank: %d", commName ? commName : "", commId, nranks, rank);

  ctx->groupApiPool = (struct groupApi *)calloc(groupApiPoolSize, sizeof(*ctx->groupApiPool));
  if (ctx->groupApiPool == NULL) goto fail;

  ctx->collApiPool = (struct collApi *)calloc(collApiPoolSize, sizeof(*ctx->collApiPool));
  if (ctx->collApiPool == NULL) goto fail;

  ctx->p2pApiPool = (struct p2pApi *)calloc(p2pApiPoolSize, sizeof(*ctx->p2pApiPool));
  if (ctx->p2pApiPool == NULL) goto fail;

  ctx->kernelLaunchPool = (struct kernelLaunch *)calloc(kernelLaunchPoolSize, sizeof(*ctx->kernelLaunchPool));
  if (ctx->kernelLaunchPool == NULL) goto fail;

  ctx->groupPool = (struct group *)calloc(groupPoolSize, sizeof(*ctx->groupPool));
  if (ctx->groupPool == NULL) goto fail;

  ctx->collPool = (struct collective *)calloc(collPoolSize, sizeof(*ctx->collPool));
  if (ctx->collPool == NULL) goto fail;

  ctx->p2pPool = (struct p2p *)calloc(p2pPoolSize, sizeof(*ctx->p2pPool));
  if (ctx->p2pPool == NULL) goto fail;

  ctx->proxyCtrlPool = (struct proxyCtrl *)calloc(proxyCtrlPoolSize, sizeof(*ctx->proxyCtrlPool));
  if (ctx->proxyCtrlPool == NULL) goto fail;

  // Print event pool sizes for debugging
  //fprintf(stdout, "Profiler: Group pool size (bytes): %lu\n", sizeof(struct group)*groupPoolSize);
  //fprintf(stdout, "Profiler: Coll  pool size (bytes): %lu\n", sizeof(struct collective)*collPoolSize);
  //fprintf(stdout, "Profiler: P2p   pool size (bytes): %lu\n", sizeof(struct p2p)*p2pPoolSize);
  //fprintf(stdout, "Profiler: Proxy pool size (bytes): %lu\n", sizeof(struct proxyCtrl)*proxyCtrlPoolSize);
  //fprintf(stdout, "Profiler: PXN   pool size (bytes): %lu\n", sizeof(struct proxyOp)*detachPoolSize);

  *context = ctx;
  return ncclSuccess;

fail:
  // cleanup resources
  if (ctx->proxyCtrlPool) free(ctx->proxyCtrlPool);
  if (ctx->p2pPool) free(ctx->p2pPool);
  if (ctx->collPool) free(ctx->collPool);
  if (ctx->groupPool) free(ctx->groupPool);
  if (ctx->collApiPool) free(ctx->collApiPool);
  if (ctx->p2pApiPool) free(ctx->p2pApiPool);
  if (ctx->kernelLaunchPool) free(ctx->kernelLaunchPool);
  if (ctx->groupApiPool) free(ctx->groupApiPool);
  free(ctx);
  if (detachPool) free(detachPool);
  return ncclSystemError;
}

static const char* profilerDumpFile;

__hidden ncclResult_t exampleProfilerFinalize(void* context) {
  FILE* fh = NULL;
  char filename[PATH_MAX] = { 0 };
  struct context* ctx = (struct context *)context;
  const char* dump = profilerDumpFile ? profilerDumpFile : getenv("NCCL_PROFILE_DUMP_FILE");
  if (dump) {
    sprintf(filename, "%s_%lu_%d.json", dump, ctx->commHash, ctx->rank);
    fh = fopen(filename, "w");
    fprintf(fh, "[\n");
  }
  INFO(NCCL_INIT, "PROFILER/Plugin: finalize commName: %s commHash: %lu nranks: %d rank: %d", ctx->commName ? ctx->commName : "", ctx->commHash, ctx->nranks, ctx->rank);

  // print last N groups/collectives/p2ps
  // Note that since the v5 version of the profiler, group API events are now at the top of the hierarchy.
  // Legacy Group events from v4 are still emitted for compatibility purposes when using the v4 profiler but excluded from this example.
  int start = (ctx->groupApiPoolIndex - groupApiPoolSize >= 0) ? ctx->groupApiPoolIndex - groupApiPoolSize : 0;
  int end = ctx->groupApiPoolIndex;
  for (int i = start; i < end; i++) {
    printEvent(fh, &ctx->groupApiPool[i%groupApiPoolSize]);
  }

  start = (ctx->proxyCtrlPoolIndex - proxyCtrlPoolSize >= 0) ? ctx->proxyCtrlPoolIndex - proxyCtrlPoolSize : 0;
  end = ctx->proxyCtrlPoolIndex;
  for (int i = start; i < end; i++) {
    printEvent(fh, &ctx->proxyCtrlPool[i%proxyCtrlPoolSize]);
  }

  free(ctx->groupPool);
  free(ctx->collApiPool);
  free(ctx->p2pApiPool);
  free(ctx->kernelLaunchPool);
  free(ctx->groupApiPool);
  free(ctx->collPool);
  free(ctx->p2pPool);
  free(ctx->proxyCtrlPool);
  free(ctx);

  // last thread cleans up shared detach pool
  if (__atomic_sub_fetch(&initialized, 1, __ATOMIC_RELAXED) == 0) {
    start = (detachPoolIndex - detachPoolSize >= 0) ? detachPoolIndex - detachPoolSize : 0;
    end = detachPoolIndex;
    for (int i = start; i < end; i++) {
      printEvent(fh, &detachPool[i%detachPoolSize]);
    }
    free(detachPool);
  }

  if (fh) fprintf(fh, "{}]\n");
  if (fh) fclose(fh);

  return ncclSuccess;
}

__hidden void updateEvent(void* handle);

__hidden ncclResult_t exampleProfilerStartEvent(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  *eHandle = NULL;
  struct context* ctx = (struct context *)context;
  if (eDescr->type == ncclProfileGroupApi) {
    struct groupApi* event;
    int groupApiId = __atomic_fetch_add(&ctx->groupApiPoolIndex, 1, __ATOMIC_RELAXED);
    if ((groupApiId - __atomic_load_n(&ctx->groupApiPoolBase, __ATOMIC_RELAXED)) < groupApiPoolSize) {
      // if there are available group API events grab one
      event = &ctx->groupApiPool[groupApiId%groupApiPoolSize];
      // Make sure all child events of the picked group API event are cleared
      while (!profilerQueueEmpty(&event->collApiEvents)) {
        struct collApi *collApiEvent = profilerQueueDequeue(&event->collApiEvents);
        resetTaskEvents(collApiEvent, ctx);
        __atomic_fetch_add(&ctx->collApiPoolBase, 1, __ATOMIC_RELAXED);
      }
      while (!profilerQueueEmpty(&event->p2pApiEvents)) {
        struct p2pApi *p2pApiEvent = profilerQueueDequeue(&event->p2pApiEvents);
        resetTaskEvents(p2pApiEvent, ctx);
        __atomic_fetch_add(&ctx->p2pApiPoolBase, 1, __ATOMIC_RELAXED);
      }
      while (!profilerQueueEmpty(&event->kernelLaunchEvents)) {
        profilerQueueDequeue(&event->kernelLaunchEvents);
        __atomic_fetch_add(&ctx->kernelLaunchPoolBase, 1, __ATOMIC_RELAXED);
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
    event->startTs = gettime() - startTime;
    *eHandle = event;
  } else if (eDescr->type == ncclProfileCollApi) {
    if (eDescr->parentObj == NULL) return ncclSuccess;
    struct collApi* event;
    int collApiId = __atomic_fetch_add(&ctx->collApiPoolIndex, 1, __ATOMIC_RELAXED);
    if ((collApiId - __atomic_load_n(&ctx->collApiPoolBase, __ATOMIC_RELAXED)) < collApiPoolSize) {
      // if there are available Coll API events grab one
      event = &ctx->collApiPool[collApiId%collApiPoolSize];
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
    if ((p2pApiId - __atomic_load_n(&ctx->p2pApiPoolBase, __ATOMIC_RELAXED)) < p2pApiPoolSize) {
      // if there are available p2p API events grab one
      event = &ctx->p2pApiPool[p2pApiId%p2pApiPoolSize];
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
    event->startTs = gettime() - startTime;
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
    event->base.startTs = gettime() - startTime;
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
    event->base.next = parent->eventHead;
    event->base.startTs = gettime() - startTime;
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
    event->startTs = gettime() - startTime;
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
      event->startTs = gettime() - startTime;
      event->parent = NULL;
      event->stepCount = 0;
      *eHandle = event;
      debugEvent(event, "PxnProxyOpStart");
      return ncclSuccess;
    }

    if (eventBase->type == ncclProfileColl) {
      struct collective* parent = (struct collective *)eDescr->parentObj;
      int channelId = eDescr->proxyOp.channelId;
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
      event->startTs = gettime() - startTime;
      event->stepCount = 0;
      *eHandle = event;
      __atomic_fetch_add(&parent->base.refCount, 1, __ATOMIC_RELAXED);
      debugEvent(event, "ProxyOpStart");
    } else { // ncclProfileP2p
      struct p2p* parent = (struct p2p *)eDescr->parentObj;
      int channelId = eDescr->proxyOp.channelId;
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
      event->startTs = gettime() - startTime;
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
    event->startTs = gettime() - startTime;
    event->nNetEvents = 0;
    *eHandle = event;
    debugEvent(event, "ProxyStepStart");
  } else if (eDescr->type == ncclProfileKernelCh) {
    struct taskEventBase* eventBase = (struct taskEventBase *)eDescr->parentObj;
    if (eventBase == NULL) return ncclSuccess;
    if (eventBase->type == ncclProfileColl) {
      struct collective* parent = (struct collective *)eDescr->parentObj;
      struct kernelCh* event = &parent->kernel[eDescr->kernelCh.channelId];
      event->type = ncclProfileKernelCh;
      event->channelId = eDescr->kernelCh.channelId;
      event->startGpuClk = eDescr->kernelCh.pTimer;
      event->parent = eventBase;
      event->startTs = gettime() - startTime;
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
      event->startTs = gettime() - startTime;
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
        event->startTs = gettime() - startTime;
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
        event->startTs = gettime() - startTime;
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
    if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_RELAXED) == 0) {
      event->stopTs = gettime() - startTime;
      __atomic_fetch_add(&event->ctx->groupApiPoolBase, 1, __ATOMIC_RELAXED);
    }
  } else if (type == ncclProfileCollApi) {
    struct collApi* event = (struct collApi*) handle;
    if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_RELAXED) == 0) {
      event->stopTs = gettime() - startTime;
      __atomic_fetch_add(&event->ctx->collApiPoolBase, 1, __ATOMIC_RELAXED);
    }
    updateEvent(event->parent);
    return;
  } else if (type == ncclProfileP2pApi) {
    struct p2pApi* event = (struct p2pApi*) handle;
    if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_RELAXED) == 0) {
      event->stopTs = gettime() - startTime;
      __atomic_fetch_add(&event->ctx->p2pApiPoolBase, 1, __ATOMIC_RELAXED);
    }
    updateEvent(event->parent);
    event->stopTs = gettime() - startTime;
  } else if (type == ncclProfileKernelLaunch) {
    struct kernelLaunch* event = (struct kernelLaunch*) handle;
    event->stopTs = gettime() - startTime;
    updateEvent(event->parent);
  } else if (type == ncclProfileGroup) {
    struct group* event = (struct group *)handle;
    if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_RELAXED) == 0) {
      event->stopTs = gettime() - startTime;
      // return group event to the pool
      __atomic_fetch_add(&event->ctx->groupPoolBase, 1, __ATOMIC_RELAXED);
    }
    debugEvent(event, "GroupStop");
  } else if (type == ncclProfileColl) {
    struct collective* event = (struct collective *)handle;
    if (__atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_RELAXED) == 0) {
      event->base.stopTs = gettime() - startTime;
      debugEvent(event, "CollStop");
      updateEvent(event->base.parent);
      return;
    }
    debugEvent(event, "CollStop");
  } else if (type == ncclProfileP2p) {
    struct p2p* event = (struct p2p *)handle;
    if (__atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_RELAXED) == 0) {
      event->base.stopTs = gettime() - startTime;
      debugEvent(event, "P2pStop");
      updateEvent(event->base.parent);
      return;
    }
    debugEvent(event, "P2pStop");
  } else if (type == ncclProfileProxyOp) {
    struct proxyOp* event = (struct proxyOp *)handle;
    event->stopTs = gettime() - startTime;
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
    event->stopTs = gettime() - startTime;
    debugEvent(event, "ProxyStepStop");
  } else if (type == ncclProfileProxyCtrl) {
    struct proxyCtrl* event = (struct proxyCtrl *)handle;
    event->stopTs = gettime() - startTime;
    debugEvent(event, "ProxyCtrlStop");
  } else if (type == ncclProfileKernelCh) {
    struct kernelCh* event = (struct kernelCh *)handle;
    event->stopTs = gettime() - startTime;
    updateEvent(event->parent);
    debugEvent(event, "KernelChStop");
  } else if (type == ncclProfileNetPlugin) {
    struct netPlugin* event = (struct netPlugin *)handle;
    event->stopTs = gettime() - startTime;
    debugEvent(event, "NetPluginStop");
  }
}

__hidden ncclResult_t exampleProfilerStopEvent(void* eHandle) {
  // the event handle might be null if we run out of events
  if (eHandle == NULL) return ncclSuccess;

  uint64_t type = *(uint64_t *)eHandle;
  // Stopping API events, Kernel Launch events, collective/p2p task events
  // in NCCL core do not mean that they are complete. It means that the
  // operation was enqueued so we need to keep the events open
  if (type == ncclProfileGroupApi) {
    struct groupApi* event = (struct groupApi*) eHandle;
    event->stopTs = gettime() - startTime;
    return ncclSuccess;
  } else if (type == ncclProfileCollApi) {
    struct collApi* event = (struct collApi*) eHandle;
    event->stopTs = gettime() - startTime;
    return ncclSuccess;
  } else if (type == ncclProfileP2pApi) {
    struct p2pApi* event = (struct p2pApi*) eHandle;
    event->stopTs = gettime() - startTime;
    return ncclSuccess;
  } else if (type == ncclProfileKernelLaunch) {
    struct kernelLaunch* event = (struct kernelLaunch*) eHandle;
    event->stopTs = gettime() - startTime;
    return ncclSuccess;
  } else if (type == ncclProfileGroup) {
    struct group* event = (struct group *)eHandle;
    event->stopTs = gettime() - startTime;
    return ncclSuccess;
  } else if (type == ncclProfileColl) {
    struct collective* event = (struct collective *)eHandle;
    event->base.stopTs = gettime() - startTime;
    return ncclSuccess;
  } else if (type == ncclProfileP2p) {
    struct p2p* event = (struct p2p *)eHandle;
    event->base.stopTs = gettime() - startTime;
    return ncclSuccess;
  }

  updateEvent(eHandle);
  return ncclSuccess;
}

__hidden ncclResult_t exampleProfilerRecordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) {
  // the event handle might be null if we run out of events
  if (eHandle == NULL) return ncclSuccess;

  uint64_t type = *(uint64_t *)eHandle;
  if (type == ncclProfileGroupApi) {
    struct groupApi* event = (struct groupApi*) eHandle;
    if (eState == ncclProfilerEndGroupApiStart) {
      event->endOfncclGroupStartTs = gettime() - startTime;
    } else if (eState == ncclProfilerBeginGroupApiEnd) {
      event->startOfncclGroupEndTs = gettime() - startTime;
    }
  } else if (type == ncclProfileProxyOp) {
    struct proxyOp* event = (struct proxyOp *)eHandle;
    if (eState == ncclProfilerProxyOpInProgress_v4) {
      event->progrTs = gettime() - startTime;
    }
  } else if (type == ncclProfileProxyStep) {
    struct proxyStep* event = (struct proxyStep *)eHandle;
    struct proxyOp* parent = event->parent;
    switch (eState) {
      case ncclProfilerProxyStepSendGPUWait:
        event->timestamp[PROXY_STEP_SEND_GPU_WAIT] = gettime() - startTime;
        break;
      case ncclProfilerProxyStepSendPeerWait_v4:
        // do not update step event if in SendPeerWait
        if (event->state == ncclProfilerProxyStepSendPeerWait_v4) break;
        event->timestamp[PROXY_STEP_SEND_PEER_WAIT] = gettime() - startTime;
        event->state = ncclProfilerProxyStepSendPeerWait_v4;
        break;
      case ncclProfilerProxyStepSendWait:
        event->timestamp[PROXY_STEP_SEND_WAIT] = gettime() - startTime;
        parent->transSize += eStateArgs->proxyStep.transSize;
        break;
      case ncclProfilerProxyStepRecvWait:
        event->timestamp[PROXY_STEP_RECV_WAIT] = gettime() - startTime;
        break;
      case ncclProfilerProxyStepRecvFlushWait:
        event->timestamp[PROXY_STEP_RECV_FLUSH_WAIT] = gettime() - startTime;
        parent->transSize += eStateArgs->proxyStep.transSize;
        break;
      case ncclProfilerProxyStepRecvGPUWait:
        event->timestamp[PROXY_STEP_RECV_GPU_WAIT] = gettime() - startTime;
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
