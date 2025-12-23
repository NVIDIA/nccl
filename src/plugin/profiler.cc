/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "checks.h"
#include "comm.h"
#include "enqueue.h"
#include "utils.h"
#include "proxy.h"
#include "profiler.h"
#include "transport.h"
#include "plugin.h"
#include "compiler.h"
#include <mutex>
#include "os.h"

extern ncclProfiler_t* getNcclProfiler_v1(void* lib);
extern ncclProfiler_t* getNcclProfiler_v2(void* lib);
extern ncclProfiler_t* getNcclProfiler_v3(void* lib);
extern ncclProfiler_t* getNcclProfiler_v4(void* lib);
extern ncclProfiler_t* getNcclProfiler_v5(void* lib);
extern ncclProfiler_t* getNcclProfiler_v6(void* lib);

static std::mutex profilerMutex;
static int profilerPluginRefCount;
static void* profilerPluginLib;
static ncclProfiler_t* ncclProfiler;

extern thread_local int ncclGroupDepth;
thread_local ncclProfilerApiState_t ncclProfilerApiState;

#define MAX_STR_LEN 256

enum {
  profilerPluginLoadFailed = -1,
  profilerPluginLoadReady = 0,
  profilerPluginLoadSuccess = 1,
};
static int profilerPluginStatus = profilerPluginLoadReady;
static pid_t pid;

static ncclResult_t ncclProfilerPluginLoad(void) {
  const char* profilerName;
  if (profilerPluginLoadFailed == profilerPluginStatus) {
    return ncclSuccess;
  }

  std::lock_guard<std::mutex> lock(profilerMutex);
  if (profilerPluginLoadSuccess == profilerPluginStatus) {
    ++profilerPluginRefCount;
    goto exit;
  }

  if ((profilerName = ncclGetEnv("NCCL_PROFILER_PLUGIN")) != nullptr) {
    INFO(NCCL_ENV, "NCCL_PROFILER_PLUGIN set by environment to %s", profilerName);
    if (strcasecmp(profilerName, "none") == 0)
      goto fail;
  }
  profilerPluginLib = ncclOpenProfilerPluginLib(profilerName);
  if (profilerPluginLib == nullptr) {
    profilerPluginLib = ncclGetNetPluginLib(ncclPluginTypeProfiler);
    if (nullptr == profilerPluginLib) {
      goto fail;
    }
    profilerName = nullptr;
  } else if (ncclPluginLibPaths[ncclPluginTypeProfiler]) {
    profilerName = ncclPluginLibPaths[ncclPluginTypeProfiler];
  }

  ncclProfiler = getNcclProfiler_v6(profilerPluginLib);
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v5(profilerPluginLib);
  }
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v4(profilerPluginLib);
  }
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v3(profilerPluginLib);
  }
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v2(profilerPluginLib);
  }
  if (ncclProfiler == NULL) {
    ncclProfiler = getNcclProfiler_v1(profilerPluginLib);
  }
  if (ncclProfiler == NULL) {
    if (profilerName) INFO(NCCL_INIT, "External profiler plugin %s is unsupported", profilerName);
    goto fail;
  }
  if (profilerName) INFO(NCCL_INIT, "Successfully loaded external profiler plugin %s", profilerName);

  ++profilerPluginRefCount;
  profilerPluginStatus = profilerPluginLoadSuccess;

  // Store the pid of the process loading the profiler.
  // This is attached to the proxyOp event descriptor
  // so the plugin can figure out if the parent event
  // is in the same address space or not
  pid = ncclOsGetpid();

exit:
  return ncclSuccess;
fail:
  if (profilerPluginLib) NCCLCHECK(ncclClosePluginLib(profilerPluginLib, ncclPluginTypeProfiler));
  profilerPluginLib = nullptr;
  profilerPluginStatus = profilerPluginLoadFailed;
  goto exit;
}

static ncclResult_t ncclProfilerPluginUnload(void) {
  std::lock_guard<std::mutex> lock(profilerMutex);
  if (0 == (--profilerPluginRefCount)) {
    if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
      INFO(NCCL_INIT, "PROFILER/Plugin: Closing profiler plugin %s", ncclProfiler->name);
    }
    NCCLCHECK(ncclClosePluginLib(profilerPluginLib, ncclPluginTypeProfiler));
    profilerPluginLib = nullptr;
    ncclProfiler = nullptr;
    profilerPluginStatus = profilerPluginLoadReady;
  }
  return ncclSuccess;
}

#define ENABLE_TIMER 0
#include "timer.h"

#if ENABLE_TIMER
// These counters are used to measure profiler overheads for different part of the code
// These counters are only useful/meaningful in controlled test environments where there
// is only one thread updating each set of counters, i.e., every communicator has its
// own proxy thread and the network uses only one thread to make progress (this is true
// for net_ib plugin but might not be true for net_socket plugin).
static int64_t elapsedCount;
static int64_t initCount, finalizeCount;
static int64_t groupStartCount, groupStopCount;
static int64_t taskStartCount, taskStopCount;
static int64_t proxyOpStartCount, proxyOpStopCount;
static int64_t proxyStepStartCount, proxyStepStopCount;
static int64_t proxyCtrlStartCount, proxyCtrlStopCount;
static int64_t proxyOpRecordCount, proxyStepRecordCount, proxyCtrlRecordCount;

static double elapsedTs[2];
static double initTs[2], finalizeTs[2];
static double groupStartTs[2], groupStopTs[2];
static double taskStartTs[2], taskStopTs[2];
static double proxyOpStartTs[2], proxyOpStopTs[2];
static double proxyStepStartTs[2], proxyStepStopTs[2];
static double proxyCtrlStartTs[2], proxyCtrlStopTs[2];
static double proxyOpRecordTs[2], proxyStepRecordTs[2], proxyCtrlRecordTs[2];

#define TIME_START_EVENT(event) do {            \
    (event ## Count)++;                         \
    (event ## Ts)[0] = gettime();               \
  } while(0)

#define TIME_STOP_EVENT(event) do {             \
    double val = gettime() - (event ## Ts)[0];  \
    (event ## Ts)[1] += val;                    \
  } while(0)

#define TIME_PRINT_EVENTS(name) do {                                    \
    printf("%s ", name);                                                \
    if (elapsedCount)         printf("[elapsed] %g/%ld = %g ", elapsedTs[1], elapsedCount, elapsedTs[1]/elapsedCount); \
    if (initCount)            printf("[init] %g/%ld = %g ", initTs[1], initCount, initTs[1]/initCount); \
    if (finalizeCount)        printf("[finalize] %g/%ld = %g ", finalizeTs[1], finalizeCount, finalizeTs[1]/finalizeCount); \
    if (groupStartCount)      printf("[groupStart] %g/%ld = %g ", groupStartTs[1], groupStartCount, groupStartTs[1]/groupStartCount); \
    if (groupStopCount)       printf("[groupStop] %g/%ld = %g ", groupStopTs[1], groupStopCount, groupStopTs[1]/groupStopCount); \
    if (taskStartCount)       printf("[taskStart] %g/%ld = %g ", taskStartTs[1], taskStartCount, taskStartTs[1]/taskStartCount); \
    if (taskStopCount)        printf("[taskStop] %g/%ld = %g ", taskStopTs[1], taskStopCount, taskStopTs[1]/taskStopCount); \
    if (proxyOpStartCount)    printf("[proxyOpStart] %g/%ld = %g ", proxyOpStartTs[1], proxyOpStartCount, proxyOpStartTs[1]/proxyOpStartCount); \
    if (proxyOpStopCount)     printf("[proxyOpStop] %g/%ld = %g ", proxyOpStopTs[1], proxyOpStopCount, proxyOpStopTs[1]/proxyOpStopCount); \
    if (proxyStepStartCount)  printf("[proxyStepStart] %g/%ld = %g ", proxyStepStartTs[1], proxyStepStartCount, proxyStepStartTs[1]/proxyStepStartCount); \
    if (proxyStepStopCount)   printf("[proxyStepStop] %g/%ld = %g ", proxyStepStopTs[1], proxyStepStopCount, proxyStepStopTs[1]/proxyStepStopCount); \
    if (proxyCtrlStartCount)  printf("[proxyCtrlStart] %g/%ld = %g ", proxyCtrlStartTs[1], proxyCtrlStartCount, proxyCtrlStartTs[1]/proxyCtrlStartCount); \
    if (proxyCtrlStopCount)   printf("[proxyCtrlStop] %g/%ld = %g ", proxyCtrlStopTs[1], proxyCtrlStopCount, proxyCtrlStopTs[1]/proxyCtrlStopCount); \
    if (proxyOpRecordCount)   printf("[proxyOpRecord] %g/%ld = %g ", proxyOpRecordTs[1], proxyOpRecordCount, proxyOpRecordTs[1]/proxyOpRecordCount); \
    if (proxyStepRecordCount) printf("[proxyStepRecord] %g/%ld = %g ", proxyStepRecordTs[1], proxyStepRecordCount, proxyStepRecordTs[1]/proxyStepRecordCount); \
    if (proxyCtrlRecordCount) printf("[proxyCtrlRecord] %g/%ld = %g", proxyCtrlRecordTs[1], proxyCtrlRecordCount, proxyCtrlRecordTs[1]/proxyCtrlRecordCount); \
    printf("\n");                                                       \
  } while(0)
#else
#define TIME_START_EVENT(event) do {} while(0)
#define TIME_STOP_EVENT(event)  do {} while(0)
#define TIME_PRINT_EVENTS(name) do {} while(0)
#endif


int ncclProfilerEventMask;       // Set by profiler

// Print enabled profiler event types
static void printProfilerEventMask(int mask) {
  if (!mask) return;

  char enabled[512] = {0};
  int pos = 0;
  if (mask & ncclProfileGroup)        pos += sprintf(enabled + pos, "Group ");
  if (mask & ncclProfileColl)         pos += sprintf(enabled + pos, "Coll ");
  if (mask & ncclProfileP2p)          pos += sprintf(enabled + pos, "P2p ");
  if (mask & ncclProfileProxyOp)      pos += sprintf(enabled + pos, "ProxyOp ");
  if (mask & ncclProfileProxyStep)    pos += sprintf(enabled + pos, "ProxyStep ");
  if (mask & ncclProfileProxyCtrl)    pos += sprintf(enabled + pos, "ProxyCtrl ");
  if (mask & ncclProfileKernelCh)     pos += sprintf(enabled + pos, "KernelCh ");
  if (mask & ncclProfileNetPlugin)    pos += sprintf(enabled + pos, "NetPlugin ");
  if (mask & ncclProfileGroupApi)     pos += sprintf(enabled + pos, "GroupApi ");
  if (mask & ncclProfileCollApi)      pos += sprintf(enabled + pos, "CollApi ");
  if (mask & ncclProfileP2pApi)       pos += sprintf(enabled + pos, "P2pApi ");
  if (mask & ncclProfileKernelLaunch) pos += sprintf(enabled + pos, "KernelLaunch ");
  if (mask & ncclProfileCeColl)       pos += sprintf(enabled + pos, "CeColl ");
  if (mask & ncclProfileCeSync)       pos += sprintf(enabled + pos, "CeSync ");
  if (mask & ncclProfileCeBatch)      pos += sprintf(enabled + pos, "CeBatch ");
  INFO(NCCL_INIT, "Profiler event mask: 0x%x (%d) - Enabled: %s", mask, mask, enabled);
}

ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) {
  TIME_START_EVENT(elapsed);
  TIME_START_EVENT(init);
  ncclProfilerPluginLoad();
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    int err = ncclProfiler->init(&comm->profilerContext, comm->commHash, &ncclProfilerEventMask, comm->config.commName, comm->nNodes, comm->nRanks, comm->rank, ncclDebugLog);
    if (err) {
      INFO(NCCL_INIT, "Profiler init failed with error '%d': %s. Continue without profiler.", err, strerror(errno));
    }

    printProfilerEventMask(ncclProfilerEventMask);
  }
  TIME_STOP_EVENT(init);
  return ncclSuccess;
}

ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm) {
  TIME_START_EVENT(finalize);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && comm->profilerContext) {
    ncclProfiler->finalize(comm->profilerContext);
  }
  ncclProfilerPluginUnload();
  TIME_STOP_EVENT(finalize);
  TIME_STOP_EVENT(elapsed);
  TIME_PRINT_EVENTS("Profiler");
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = { 0 };
  eDescr.type = ncclProfileGroupApi;
  eDescr.groupApi.graphCaptured = isGraphCaptured;

  ncclProfilerApiState.eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
  int groupApiMask = ncclProfileGroupApi | ncclProfileP2pApi | ncclProfileCollApi | ncclProfileKernelLaunch | ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin | ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;

  // Only count outermost groups when emitting group API events
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & groupApiMask)) {
    if (ncclProfilerApiState.profilerGroupDepth == 0) {
      eDescr.groupApi.groupDepth = ncclGroupDepth;
      ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.groupApiEventHandle, &eDescr);
      ncclProfilerApiState.profilerGroupDepth = ncclGroupDepth;
      ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStarted;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupApiEvent() {
  void* groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && groupApiEventHandle && ncclProfilerApiState.profilerGroupDepth == 0) {
    ncclProfiler->stopEvent(groupApiEventHandle);
    ncclProfilerApiState.groupApiEventHandle = nullptr;
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState) {
  void* groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  bool shouldRecord = false;
  if (eState == ncclProfilerGroupStartApiStop && ncclProfilerApiState.state == ncclProfilerGroupApiStartStateStarted) {
    ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStopped;
    shouldRecord = true;
  } else if (eState == ncclProfilerGroupEndApiStart && ncclProfilerApiState.state == ncclProfilerGroupApiStartStateStopped) {
    ncclProfilerApiState.state = ncclProfilerGroupApiStartStateReset;
    shouldRecord = true;
  }

  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && groupApiEventHandle && shouldRecord) {
    ncclProfiler->recordEventState(groupApiEventHandle, eState, NULL);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo *info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = { 0 };
  eDescr.type = ncclProfileP2pApi;
  eDescr.parentObj = ncclProfilerApiState.groupApiEventHandle;
  eDescr.p2pApi.func = ncclFuncToString(info->coll);
  eDescr.p2pApi.count = info->count;
  eDescr.p2pApi.datatype = ncclDatatypeToString(info->datatype);
  eDescr.p2pApi.stream = (void *) info->stream;
  eDescr.p2pApi.graphCaptured = isGraphCaptured;
  int p2pApiMask = ncclProfileP2pApi | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin;
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & p2pApiMask)) {
    ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.p2pApiEventHandle, &eDescr);
  }
  if (isGraphCaptured && info->comm->config.graphUsageMode == 0) {
    INFO(NCCL_P2P | NCCL_ENV, "Comm config graphUsageMode is set to %d but the user is capturing graphs on the stream. Violating graphUsageMode semantics can lead to hangs!", info->comm->config.graphUsageMode);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopP2pApiEvent() {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && ncclProfilerApiState.p2pApiEventHandle) {
    ncclProfiler->stopEvent(ncclProfilerApiState.p2pApiEventHandle);
    ncclProfilerApiState.p2pApiEventHandle = nullptr;
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo *info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = { 0 };
  eDescr.type = ncclProfileCollApi;
  eDescr.parentObj = ncclProfilerApiState.groupApiEventHandle;
  eDescr.collApi.func = ncclFuncToString(info->coll);
  eDescr.collApi.count = info->count;
  eDescr.collApi.datatype = ncclDatatypeToString(info->datatype);
  eDescr.collApi.stream = (void *) info->stream;
  eDescr.collApi.root = info->root;
  eDescr.collApi.graphCaptured = isGraphCaptured;
  int collApiMask = ncclProfileCollApi | ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin | ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & collApiMask)) {
    ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.collApiEventHandle, &eDescr);
  }
  if (isGraphCaptured && info->comm->config.graphUsageMode == 0) {
    INFO(NCCL_COLL | NCCL_ENV, "Comm config graphUsageMode is set to %d but the user is capturing graphs on the stream. Violating graphUsageMode semantics can lead to hangs!", info->comm->config.graphUsageMode);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCollApiEvent() {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && ncclProfilerApiState.collApiEventHandle) {
    ncclProfiler->stopEvent(ncclProfilerApiState.collApiEventHandle);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclProfilerEventDescr_t eDescr = { 0 };
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    void* groupApiEventHandle = NULL;
    // Check if any collective in the plan has a set event activation mask
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    int eActivationMask_ = 0;
    while (ct) {
      if (ct->eActivationMask) {
        eActivationMask_ = ct->eActivationMask;
        groupApiEventHandle = ct->groupApiEventHandle;
        goto startKernelLaunchEvent;
      }
      ct = ct->next;
    }
    // Check if any pt2pt in the plan has a set event activation mask
    while (pt) {
      if (pt->eActivationMask) {
        eActivationMask_ = pt->eActivationMask;
        groupApiEventHandle = pt->groupApiEventHandle;
        goto startKernelLaunchEvent;
      }
      pt = pt->next;
    }

  startKernelLaunchEvent:
    if (eActivationMask_ & ncclProfileKernelLaunch) {
      eDescr.type = ncclProfileKernelLaunch;
      eDescr.parentObj = groupApiEventHandle;
      eDescr.kernelLaunch.stream = (void *) stream;
      ncclProfiler->startEvent(plan->comm->profilerContext, &plan->kernelLaunchEventHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && plan->kernelLaunchEventHandle) {
    ncclProfiler->stopEvent(plan->kernelLaunchEventHandle);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(groupStart);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    // Check if any collective in the plan has a set event activation mask
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    int eActivationMask_ = 0;
    while (ct) {
      if (ct->eActivationMask) {
        eActivationMask_ = ct->eActivationMask;
        goto startGroup;
      }
      ct = ct->next;
    }
    // Check if any pt2pt in the plan has a set event activation mask
    while (pt) {
      if (pt->eActivationMask) {
        eActivationMask_ = pt->eActivationMask;
        goto startGroup;
      }
      pt = pt->next;
    }

  startGroup:
    if (eActivationMask_ & (ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileGroup;
      ncclProfiler->startEvent(plan->comm->profilerContext, &plan->groupEventHandle, &eDescr);
    }
  }
  TIME_STOP_EVENT(groupStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(groupStop);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && plan->groupEventHandle) {
    ncclProfiler->stopEvent(plan->groupEventHandle);
  }
  TIME_STOP_EVENT(groupStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(taskStart);
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct) {
    if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
      int enable = ct->eActivationMask & (ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin);
      if (enable) {
        ncclProfilerEventDescr_t eDescr = { 0 };
        eDescr.type = ncclProfileColl;
        eDescr.coll.parentGroup = plan->groupEventHandle;
        eDescr.parentObj = ct->collApiEventHandle;
        eDescr.rank = plan->comm->rank;
        eDescr.coll.seqNumber = plan->comm->seqNumber[ct->func];
        eDescr.coll.func = ncclFuncToString(ct->func);
        eDescr.coll.sendBuff = ct->sendbuff;
        eDescr.coll.recvBuff = ct->recvbuff;
        eDescr.coll.count = ct->count;
        eDescr.coll.root = ct->root;
        eDescr.coll.datatype = ncclDatatypeToString(ct->datatype);
        eDescr.coll.nChannels = ct->nChannels;
        eDescr.coll.nWarps = ct->nWarps;
        eDescr.coll.algo = ncclAlgoToString(ct->algorithm);
        eDescr.coll.proto = ncclProtoToString(ct->protocol);
        ncclProfiler->startEvent(plan->comm->profilerContext, &ct->eventHandle, &eDescr);
      }
    }
    // comm->seqNumber values are updated even if the plugin is not active, since they are used by RAS as well.
    // The test for "persistent" is a workaround for graph-captured collectives.  In their case this function may not be
    // consistently invoked on all the ranks, which would lead to mismatched counter values and thus false-positive
    // reports from RAS.  Instead, we choose not to include graph-captured collectives in our counts.  An exception is
    // made if ncclProfileKernelCh profiler events are active, as they result in proxy events always being added, which
    // gives the consistency.
    if (!plan->persistent || (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (plan->groupEventHandle || ct->collApiEventHandle) &&
                              (ct->eActivationMask & ncclProfileKernelCh)))
      COMPILER_ATOMIC_FETCH_ADD(&plan->comm->seqNumber[ct->func], 1, std::memory_order_relaxed);
    ct = ct->next;
  }
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    while (pt) {
      int enable = pt->eActivationMask & (ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin);
      if (enable) {
        ncclProfilerEventDescr_t eDescr = { 0 };
        eDescr.type = ncclProfileP2p;
        eDescr.p2p.parentGroup = plan->groupEventHandle;
        eDescr.parentObj = pt->p2pApiEventHandle;
        eDescr.rank = plan->comm->rank;
        eDescr.p2p.func = ncclFuncToString(pt->func);
        eDescr.p2p.buff = pt->buff;
        eDescr.p2p.count = pt->count;
        eDescr.p2p.datatype = ncclDatatypeToString(pt->datatype);
        eDescr.p2p.peer = pt->root;
        eDescr.p2p.nChannels = pt->nChannels;
        ncclProfiler->startEvent(plan->comm->profilerContext, &pt->eventHandle, &eDescr);
      }
      pt = pt->next;
    }
  }
  TIME_STOP_EVENT(taskStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(taskStop);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    while (ct) {
      if (ct->eventHandle) ncclProfiler->stopEvent(ct->eventHandle);
      ct = ct->next;
    }
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    while (pt) {
      if (pt->eventHandle) ncclProfiler->stopEvent(pt->eventHandle);
      pt = pt->next;
    }
  }
  TIME_STOP_EVENT(taskStop);
  return ncclSuccess;
}

// Bellow we set the proxy descriptor step number to DIVUP(step, args->sliceSteps).
// The reason is that for some ncclOp (e.g. AllReduce) one network transfer is
// made of sliceSteps steps rather than one step. In the profiler we are still
// interested in whole network transfers though, so we account for this when
// computing the actual network step number.
ncclResult_t ncclProfilerStartProxyOpEvent(int s, struct ncclProxyArgs* args) {
  TIME_START_EVENT(proxyOpStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (sub->eActivationMask & (ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileProxyOp;
      eDescr.parentObj = sub->taskEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyOp.pid = sub->pid;
      eDescr.proxyOp.channelId = sub->channelId;
      eDescr.proxyOp.peer = sub->peer;
      eDescr.proxyOp.nSteps = DIVUP(sub->nsteps, args->sliceSteps);
      eDescr.proxyOp.chunkSize = args->chunkSize * args->sliceSteps;
      eDescr.proxyOp.isSend = args->progress == ncclTransports[TRANSPORT_NET]->send.proxyProgress ? 1 : 0;
      ncclProfiler->startEvent(sub->profilerContext, &sub->opEventHandle, &eDescr);
    }
  }
  TIME_STOP_EVENT(proxyOpStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyOpEvent(int s, struct ncclProxyArgs* args) {
  TIME_START_EVENT(proxyOpStop);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    ncclProfiler->stopEvent(sub->opEventHandle);
    sub->opEventHandle = NULL;
  }
  TIME_STOP_EVENT(proxyOpStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartSendProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  int step_ = DIVUP(stepId, args->sliceSteps);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileProxyStep;
      eDescr.parentObj = sub->opEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyStep.step = step_;
      ncclProfiler->startEvent(sub->profilerContext, &sub->pHandles[step_%NCCL_STEPS].stepEventHandle, &eDescr);
    }
  }
  sub->pHandles[step_%NCCL_STEPS].subArgPtr = sub;
  TIME_STOP_EVENT(proxyStepStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartRecvProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  int step_ = DIVUP(stepId, args->sliceSteps);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileProxyStep;
      eDescr.parentObj = sub->opEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyStep.step = step_;
      ncclProfiler->startEvent(sub->profilerContext, &sub->pHandles[step_%NCCL_STEPS].stepEventHandle, &eDescr);
    }
  }
  sub->pHandles[step_%NCCL_STEPS].subArgPtr = sub;
  TIME_STOP_EVENT(proxyStepStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStop);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    int step_ = DIVUP(stepId, args->sliceSteps);
    if (sub->pHandles[step_%NCCL_STEPS].stepEventHandle) {
      ncclProfiler->stopEvent(sub->pHandles[step_%NCCL_STEPS].stepEventHandle);
      sub->pHandles[step_%NCCL_STEPS].stepEventHandle = NULL;
    }
  }
  TIME_STOP_EVENT(proxyStepStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle) {
  TIME_START_EVENT(proxyCtrlStart);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    // for proxy control events we allow profiling mode to change on a per event basis
    int eActivationMaskProxy = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
    if (eActivationMaskProxy & ncclProfileProxyCtrl) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileProxyCtrl;
      ncclProfiler->startEvent(profilerContext, eHandle, &eDescr);
      TIME_STOP_EVENT(proxyCtrlStart);
      return ncclSuccess;
    }
  }
  *eHandle = NULL;
  TIME_STOP_EVENT(proxyCtrlStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyCtrlEvent(void* eHandle) {
  TIME_START_EVENT(proxyCtrlStop);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && eHandle) {
    ncclProfiler->stopEvent(eHandle);
  }
  TIME_STOP_EVENT(proxyCtrlStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t start) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    struct ncclProxySubArgs* sub = &args->subs[s];
    if (sub->eActivationMask & ncclProfileKernelCh) {
      ncclProfilerEventDescr_t eDescr = { };
      eDescr.type = ncclProfileKernelCh;
      eDescr.parentObj = sub->taskEventHandle;
      eDescr.kernelCh.channelId = sub->channelId;
      eDescr.kernelCh.pTimer = start;
      ncclProfiler->startEvent(sub->profilerContext, &sub->kernelEventHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t stop) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    struct ncclProxySubArgs* sub = &args->subs[s];
    if (sub->kernelEventHandle) {
      ncclProfilerEventStateArgs_t a = { };
      a.kernelCh.pTimer = stop;
      ncclProfiler->recordEventState(sub->kernelEventHandle, ncclProfilerKernelChStop, &a);
      ncclProfiler->stopEvent(sub->kernelEventHandle);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyOpEventState(int s, struct ncclProxyArgs* args, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyOpRecord);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    ncclProfilerEventStateArgs_t a = { };
    ncclProfiler->recordEventState(sub->opEventHandle, eState, &a);
  }
  TIME_STOP_EVENT(proxyOpRecord);
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyStepEventState(int s, struct ncclProxyArgs* args, int stepId, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyStepRecord);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    int step_ = DIVUP(stepId, args->sliceSteps);
    if (sub->pHandles[step_%NCCL_STEPS].stepEventHandle) {
      ncclProfilerEventStateArgs_t a = { };
      a.proxyStep.transSize = sub->transSize;
      ncclProfiler->recordEventState(sub->pHandles[step_%NCCL_STEPS].stepEventHandle, eState, &a);
    }
  }
  TIME_STOP_EVENT(proxyStepRecord);
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyCtrlEventState(void* eHandle, int appended, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyCtrlRecord);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && eHandle && COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed) & ncclProfileProxyCtrl) {
    ncclProfilerEventStateArgs_t args = { };
    args.proxyCtrl.appendedProxyOps = appended;
    ncclProfiler->recordEventState(eHandle, eState, &args);
  }
  TIME_STOP_EVENT(proxyCtrlRecord);
  return ncclSuccess;
}

ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op) {
  op->pid = pid;
  return ncclSuccess;
}

static std::mutex proxyProfilerConnectMutex;

static ncclResult_t proxyProfilerConnect(struct ncclComm* comm, struct ncclProxyOp* op) {
  ncclResult_t ret = ncclSuccess;
  std::lock_guard<std::mutex> lock(proxyProfilerConnectMutex);
  if (comm->profiler.initialized) goto exit;
  for (int c = 0; c < MAXCHANNELS; c++) {
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_PROFILER, 0, comm->rank, &comm->profiler.sendProxyConn[c]), ret, exit);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &comm->profiler.sendProxyConn[c], ncclProxyMsgConnect, NULL, 0, NULL, 0), ret, exit);
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_PROFILER, 0, comm->rank, &comm->profiler.recvProxyConn[c]), ret, exit);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &comm->profiler.recvProxyConn[c], ncclProxyMsgConnect, NULL, 0, NULL, 0), ret, exit);
  }
  comm->profiler.initialized = true;
exit:
  return ret;
}

bool ncclProfilerNeedsProxy(struct ncclComm* comm, struct ncclProxyOp* op) {
  bool enabled = ncclProfilerPluginLoaded() && (op->eActivationMask & ncclProfileKernelCh);
  if (enabled && !comm->profiler.initialized) (void)proxyProfilerConnect(comm, op);
  return enabled;
}

bool ncclProfilerPluginLoaded(void) {
  return (COMPILER_EXPECT(ncclProfiler != NULL, 0));
}

ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (type == ncclProfilerNetEventStart) { // start
      struct ncclProxyEventHandle* p = (struct ncclProxyEventHandle*)pHandle;
      struct ncclProxySubArgs* sub = p->subArgPtr;
      if (sub->eActivationMask & ncclProfileNetPlugin) {
        ncclProfilerEventDescr_t eDescr = { 0 };
        eDescr.type = ncclProfileNetPlugin;
        eDescr.parentObj = p->stepEventHandle;
        eDescr.rank = sub->rank;
        eDescr.netPlugin.id = pluginId;
        eDescr.netPlugin.data = extData;
        ncclProfiler->startEvent(sub->profilerContext, eHandle, &eDescr);
      }
    } else if (type == ncclProfilerNetEventStop) { // stop
      ncclProfiler->stopEvent(*eHandle);
    } else if (type == ncclProfilerNetEventUpdate) { // update
      ncclProfilerEventStateArgs_t args = { };
      args.netPlugin.data = extData;
      ncclProfiler->recordEventState(*eHandle, ncclProfilerNetPluginUpdate, &args);
    } else { // update and stop
      ncclProfilerEventStateArgs_t args = { };
      args.netPlugin.data = extData;
      ncclProfiler->recordEventState(*eHandle, ncclProfilerNetPluginUpdate, &args);
      ncclProfiler->stopEvent(*eHandle);
    }
  }
  return ncclSuccess;
}

// ============================================================================
// CE Profiler Functions (Simple Wrappers)
// ============================================================================

/*
 * CE Collective start event - calls plugin startEvent callback
 */
ncclResult_t ncclProfilerStartCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    // Check if CE Coll events are enabled (or child events CeSync/CeBatch which need CeColl)
    int ceCollMask = ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
    if (__atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ceCollMask) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileCeColl;
      eDescr.parentObj = args->collApiEventHandle;
      eDescr.rank = comm->rank;

      eDescr.ceColl.seqNumber = comm->ceColl.ceSeqNum;
      eDescr.ceColl.func = ncclFuncToString(args->func);
      eDescr.ceColl.sendBuff = args->sendBuff;
      eDescr.ceColl.recvBuff = args->recvBuff;
      eDescr.ceColl.count = args->nElts;
      eDescr.ceColl.root = args->rootRank;
      eDescr.ceColl.datatype = ncclDatatypeToString(args->datatype);
      eDescr.ceColl.syncStrategy = comm->nvlsSupport ? "MC" : "UC";
      eDescr.ceColl.intraBatchSync = false;
      eDescr.ceColl.batchSize = 0;
      eDescr.ceColl.numBatches = 0;
      eDescr.ceColl.ceSeqNum = comm->ceColl.ceSeqNum;
      eDescr.ceColl.stream = (void*)stream;

      ncclProfiler->startEvent(comm->profilerContext, &args->ceCollProfHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

/*
 * CE Collective stop event - calls plugin stopEvent callback
 */
ncclResult_t ncclProfilerStopCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (args && args->ceCollProfHandle) {
      ncclProfiler->stopEvent(args->ceCollProfHandle);
    }
  }
  return ncclSuccess;
}

/*
 * CE Sync start event - calls plugin startEvent callback
 */
ncclResult_t ncclProfilerStartCeSyncEvent(struct ncclComm* comm, struct ncclCeCollArgs* args,
                                          cudaStream_t stream, void** ceSyncHandle) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (args && args->ceCollProfHandle && (__atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ncclProfileCeSync)) {
      // CeSync only needs to check if it's enabled; parent CeColl is implicitly started via ceCollMask
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileCeSync;
      eDescr.parentObj = args->ceCollProfHandle;
      eDescr.rank = comm->rank;

      eDescr.ceCollSync.isComplete = comm->ceColl.useCompletePtr;
      eDescr.ceCollSync.nRanks = comm->nRanks;

      ncclProfiler->startEvent(comm->profilerContext, ceSyncHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

/*
 * CE Sync stop event - calls plugin stopEvent callback
 */
ncclResult_t ncclProfilerStopCeSyncEvent(struct ncclComm* comm, void* ceSyncHandle,
                                         cudaStream_t stream) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && ceSyncHandle) {
    ncclProfiler->stopEvent(ceSyncHandle);
  }
  return ncclSuccess;
}

/*
 * CE Batch start event - calls plugin startEvent callback
 */
ncclResult_t ncclProfilerStartCeBatchEvent(struct ncclComm* comm,
                                           struct ncclCeCollArgs* args,
                                           struct ncclCeBatchOpsParams* params,
                                           cudaStream_t stream,
                                           void** ceBatchHandle) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (args && args->ceCollProfHandle) {
      // CeBatch only needs to check if it's enabled; parent CeColl is implicitly started via ceCollMask
      if (__atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ncclProfileCeBatch) {
        ncclProfilerEventDescr_t eDescr = { 0 };
        eDescr.type = ncclProfileCeBatch;
        eDescr.parentObj = args->ceCollProfHandle;
        eDescr.rank = comm->rank;

        eDescr.ceCollBatch.numOps = params->numOps;

        size_t totalBytes = 0;
        for (int i = 0; i < params->numOps; i++) {
          totalBytes += params->sizes[i];
        }
        eDescr.ceCollBatch.totalBytes = totalBytes;
        eDescr.ceCollBatch.useIntraSync = params->intraBatchSync;

        ncclProfiler->startEvent(comm->profilerContext, ceBatchHandle, &eDescr);
      }
    }
  }
  return ncclSuccess;
}

/*
 * CE Batch stop event - calls plugin stopEvent callback
 */
ncclResult_t ncclProfilerStopCeBatchEvent(struct ncclComm* comm, void* ceBatchHandle, cudaStream_t stream) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && ceBatchHandle) {
    ncclProfiler->stopEvent(ceBatchHandle);
  }
  return ncclSuccess;
}

