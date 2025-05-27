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

extern ncclProfiler_t* getNcclProfiler_v1(void* lib);
extern ncclProfiler_t* getNcclProfiler_v2(void* lib);
extern ncclProfiler_t* getNcclProfiler_v3(void* lib);

static pthread_mutex_t profilerLock = PTHREAD_MUTEX_INITIALIZER;
static int profilerPluginRefCount;
static void* profilerPluginLib;
static ncclProfiler_t* ncclProfiler;

#define MAX_STR_LEN 256

enum {
  profilerPluginLoadFailed = -1,
  profilerPluginLoadReady = 0,
  profilerPluginLoadSuccess = 1,
};
static int profilerPluginStatus = profilerPluginLoadReady;
static pid_t pid;

static ncclResult_t ncclProfilerPluginLoad(void) {
  if (profilerPluginLoadFailed == profilerPluginStatus) {
    return ncclSuccess;
  }

  pthread_mutex_lock(&profilerLock);
  if (profilerPluginLoadSuccess == profilerPluginStatus) {
    ++profilerPluginRefCount;
    goto exit;
  }

  profilerPluginLib = ncclOpenProfilerPluginLib(ncclGetEnv("NCCL_PROFILER_PLUGIN"));
  if (profilerPluginLib == nullptr) {
    goto fail;
  }

  ncclProfiler = getNcclProfiler_v3(profilerPluginLib);
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v2(profilerPluginLib);
  }
  if (ncclProfiler == NULL) {
    ncclProfiler = getNcclProfiler_v1(profilerPluginLib);
  }
  if (ncclProfiler == NULL) {
    goto fail;
  }

  ++profilerPluginRefCount;
  profilerPluginStatus = profilerPluginLoadSuccess;

  // Store the pid of the process loading the profiler.
  // This is attached to the proxyOp event descriptor
  // so the plugin can figure out if the parent event
  // is in the same address space or not
  pid = getpid();

exit:
  pthread_mutex_unlock(&profilerLock);
  return ncclSuccess;
fail:
  if (profilerPluginLib) NCCLCHECK(ncclClosePluginLib(profilerPluginLib));
  profilerPluginStatus = profilerPluginLoadFailed;
  goto exit;
}

static ncclResult_t ncclProfilerPluginUnload(void) {
  pthread_mutex_lock(&profilerLock);
  if (0 == (--profilerPluginRefCount)) {
    INFO(NCCL_ENV, "PROFILER/Plugin: Closing profiler plugin %s", ncclProfiler->name);
    NCCLCHECK(ncclClosePluginLib(profilerPluginLib));
    profilerPluginLib = nullptr;
    ncclProfiler = nullptr;
    profilerPluginStatus = profilerPluginLoadReady;
  }
  pthread_mutex_unlock(&profilerLock);
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

#define TIME_START_EVENT(event) do { \
  (event ## Count)++; \
  (event ## Ts)[0] = gettime(); \
} while(0)

#define TIME_STOP_EVENT(event) do { \
  double val = gettime() - (event ## Ts)[0]; \
  (event ## Ts)[1] += val; \
} while(0)

#define TIME_PRINT_EVENTS(name) do { \
  printf("%s ", name); \
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
  printf("\n"); \
} while(0)
#else
#define TIME_START_EVENT(event) do {} while(0)
#define TIME_STOP_EVENT(event)  do {} while(0)
#define TIME_PRINT_EVENTS(name) do {} while(0)
#endif


int ncclProfilerEventMask;       // Set by profiler

ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) {
  TIME_START_EVENT(elapsed);
  TIME_START_EVENT(init);
  ncclProfilerPluginLoad();
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    int err = ncclProfiler->init(&comm->profilerContext, &ncclProfilerEventMask);
    if (err) {
      WARN("Profiler init failed with error (%d). Continue without profiler.", err);
      ncclProfiler = NULL;
    }
  }
  TIME_STOP_EVENT(init);
  return ncclSuccess;
}

ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm) {
  TIME_START_EVENT(finalize);
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    ncclProfiler->finalize(comm->profilerContext);
  }
  ncclProfilerPluginUnload();
  TIME_STOP_EVENT(finalize);
  TIME_STOP_EVENT(elapsed);
  TIME_PRINT_EVENTS("Profiler");
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(groupStart);
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
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
  if (__builtin_expect(ncclProfiler != NULL, 0) && plan->groupEventHandle) {
    ncclProfiler->stopEvent(plan->groupEventHandle);
  }
  TIME_STOP_EVENT(groupStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(taskStart);
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct) {
    if (__builtin_expect(ncclProfiler != NULL, 0)) {
      if (plan->groupEventHandle) {
        int enable = ct->eActivationMask & (ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin);
        if (enable) {
          ncclProfilerEventDescr_t eDescr = { 0 };
          eDescr.type = ncclProfileColl;
          eDescr.parentObj = plan->groupEventHandle;
          eDescr.rank = plan->comm->rank;
          eDescr.coll.name = plan->comm->commName;
          eDescr.coll.commHash = plan->comm->commHash;
          eDescr.coll.seqNumber = plan->comm->seqNumber[ct->func];
          eDescr.coll.func = ncclFuncToString(ct->func);
          eDescr.coll.sendBuff = ct->sendbuff;
          eDescr.coll.recvBuff = ct->recvbuff;
          eDescr.coll.count = ct->count;
          eDescr.coll.root = ct->root;
          eDescr.coll.datatype = ncclDatatypeToString(ct->datatype);
          eDescr.coll.nMaxChannels = ct->nMaxChannels;
          eDescr.coll.nWarps = ct->nWarps;
          eDescr.coll.algo = ncclAlgoToString(ct->algorithm);
          eDescr.coll.proto = ncclProtoToString(ct->protocol);
          ncclProfiler->startEvent(plan->comm->profilerContext, &ct->eventHandle, &eDescr);
        }
      }
    }
    // comm->seqNumber values are updated even if the plugin is not active, since they are used by RAS as well.
    // The test for "persistent" is a workaround for graph-captured collectives.  In their case this function may not be
    // consistently invoked on all the ranks, which would lead to mismatched counter values and thus false-positive
    // reports from RAS.  Instead, we choose not to include graph-captured collectives in our counts.  An exception is
    // made if ncclProfileKernelCh profiler events are active, as they result in proxy events always being added, which
    // gives the consistency.
    if (!plan->persistent || (__builtin_expect(ncclProfiler != NULL, 0) && plan->groupEventHandle &&
                              (ct->eActivationMask & ncclProfileKernelCh)))
      plan->comm->seqNumber[ct->func]++;
    ct = ct->next;
  }
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (plan->groupEventHandle) {
      struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
      while (pt) {
        int enable = pt->eActivationMask & (ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh);
        if (enable) {
          ncclProfilerEventDescr_t eDescr = { 0 };
          eDescr.type = ncclProfileP2p;
          eDescr.parentObj = plan->groupEventHandle;
          eDescr.rank = plan->comm->rank;
          eDescr.p2p.name = plan->comm->commName;
          eDescr.p2p.commHash = plan->comm->commHash;
          eDescr.p2p.func = ncclFuncToString(pt->func);
          eDescr.p2p.buff = pt->buff;
          eDescr.p2p.count = pt->count;
          eDescr.p2p.datatype = ncclDatatypeToString(pt->datatype);
          eDescr.p2p.peer = pt->root;
          ncclProfiler->startEvent(plan->comm->profilerContext, &pt->eventHandle, &eDescr);
        }
        pt = pt->next;
      }
    }
  }
  TIME_STOP_EVENT(taskStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(taskStop);
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (plan->groupEventHandle) {
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
  }
  TIME_STOP_EVENT(taskStop);
  return ncclSuccess;
}

// Bellow we set the proxy descriptor step number to DIVUP(step, args->sliceSteps).
// The reason is that for some ncclOp (e.g. AllReduce) one network transfer is
// made of sliceSteps steps rather than one step. In the profiler we are still
// interested in whole network transfers though, so we account for this when
// computing the actual network step number.
ncclResult_t ncclProfilerStartSendProxyOpEvent(int s, struct ncclProxyArgs* args) {
  TIME_START_EVENT(proxyOpStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
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
      eDescr.proxyOp.isSend = 1;
      ncclProfiler->startEvent(sub->profilerContext, &sub->opEventHandle, &eDescr);
    }
  }
  TIME_STOP_EVENT(proxyOpStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartRecvProxyOpEvent(int s, struct ncclProxyArgs* args) {
  TIME_START_EVENT(proxyOpStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
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
      eDescr.proxyOp.isSend = 0;
      ncclProfiler->startEvent(sub->profilerContext, &sub->opEventHandle, &eDescr);
    }
  }
  TIME_STOP_EVENT(proxyOpStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyOpEvent(int s, struct ncclProxyArgs* args) {
  TIME_START_EVENT(proxyOpStop);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    ncclProfiler->stopEvent(sub->opEventHandle);
    sub->opEventHandle = NULL;
  }
  TIME_STOP_EVENT(proxyOpStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartSendProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (sub->opEventHandle && (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileNetPlugin))) {
      int step_ = DIVUP(stepId, args->sliceSteps);
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileProxyStep;
      eDescr.parentObj = sub->opEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyStep.step = step_;
      ncclProfiler->startEvent(sub->profilerContext, &sub->stepEventHandles[step_%NCCL_STEPS], &eDescr);
    }
  }
  TIME_STOP_EVENT(proxyStepStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartRecvProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (sub->opEventHandle && (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileNetPlugin))) {
      int step_ = DIVUP(stepId, args->sliceSteps);
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileProxyStep;
      eDescr.parentObj = sub->opEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyStep.step = step_;
      ncclProfiler->startEvent(sub->profilerContext, &sub->stepEventHandles[step_%NCCL_STEPS], &eDescr);
    }
  }
  TIME_STOP_EVENT(proxyStepStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStop);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    int step_ = DIVUP(stepId, args->sliceSteps);
    if (sub->stepEventHandles[step_%NCCL_STEPS]) {
      ncclProfiler->stopEvent(sub->stepEventHandles[step_%NCCL_STEPS]);
      sub->stepEventHandles[step_%NCCL_STEPS] = NULL;
    }
  }
  TIME_STOP_EVENT(proxyStepStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle) {
  TIME_START_EVENT(proxyCtrlStart);
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    // for proxy control events we allow profiling mode to change on a per event basis
    int eActivationMaskProxy = __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED);
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
  if (__builtin_expect(ncclProfiler != NULL, 0) && eHandle) {
    ncclProfiler->stopEvent(eHandle);
  }
  TIME_STOP_EVENT(proxyCtrlStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProxyArgs* args, int s) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    struct ncclProxySubArgs* sub = &args->subs[s];
    if (sub->eActivationMask & ncclProfileKernelCh) {
      ncclProfilerEventDescr_t eDescr = { };
      eDescr.type = ncclProfileKernelCh;
      eDescr.parentObj = sub->taskEventHandle;
      eDescr.kernelCh.channelId = sub->channelId;
      ncclProfiler->startEvent(sub->profilerContext, &sub->kernelEventHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProxyArgs* args, int s) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    struct ncclProxySubArgs* sub = &args->subs[s];
    if (sub->kernelEventHandle) {
      ncclProfiler->stopEvent(sub->kernelEventHandle);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyOpEventState(int s, struct ncclProxyArgs* args, int steps, size_t transSize, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyOpRecord);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    ncclProfilerEventStateArgs_t a = { };
    a.proxyOp.steps = DIVUP(steps, args->sliceSteps);
    a.proxyOp.transSize = transSize;
    ncclProfiler->recordEventState(sub->opEventHandle, eState, &a);
  }
  TIME_STOP_EVENT(proxyOpRecord);
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyStepEventState(int s, struct ncclProxyArgs* args, int stepId, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyStepRecord);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    int step_ = DIVUP(stepId, args->sliceSteps);
    if (sub->stepEventHandles[step_%NCCL_STEPS]) {
      ncclProfiler->recordEventState(sub->stepEventHandles[step_%NCCL_STEPS], eState, 0);
    }
  }
  TIME_STOP_EVENT(proxyStepRecord);
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyCtrlEventState(void* eHandle, int appended, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyCtrlRecord);
  if (__builtin_expect(ncclProfiler != NULL, 0) && eHandle && __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ncclProfileProxyCtrl) {
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

static pthread_mutex_t proxyProfilerConnectLock = PTHREAD_MUTEX_INITIALIZER;

static ncclResult_t proxyProfilerConnect(struct ncclComm* comm, struct ncclProxyOp* op) {
  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&proxyProfilerConnectLock);
  if (comm->profiler.initialized) goto exit;
  for (int c = 0; c < MAXCHANNELS; c++) {
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_PROFILER, 0, comm->rank, &comm->profiler.sendProxyConn[c]), ret, exit);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &comm->profiler.sendProxyConn[c], ncclProxyMsgConnect, NULL, 0, NULL, 0), ret, exit);
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_PROFILER, 0, comm->rank, &comm->profiler.recvProxyConn[c]), ret, exit);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &comm->profiler.recvProxyConn[c], ncclProxyMsgConnect, NULL, 0, NULL, 0), ret, exit);
  }
  comm->profiler.initialized = true;
exit:
  pthread_mutex_unlock(&proxyProfilerConnectLock);
  return ret;
}

bool ncclProfilerNeedsProxy(struct ncclComm* comm, struct ncclProxyOp* op) {
  bool enabled = ncclProfilerPluginLoaded() && (op->eActivationMask & ncclProfileKernelCh);
  if (enabled && !comm->profiler.initialized) (void)proxyProfilerConnect(comm, op);
  return enabled;
}

bool ncclProfilerPluginLoaded(void) {
  return (__builtin_expect(ncclProfiler != NULL, 0));
}

ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    struct ncclProxySubArgs* sub = (struct ncclProxySubArgs*)pHandle;
    if (type == 0) { // start
      if (sub->eActivationMask & ncclProfileNetPlugin) {
        ncclProfilerEventDescr_t eDescr = { 0 };
        eDescr.type = ncclProfileNetPlugin;
        eDescr.parentObj = sub->stepEventHandles[sub->profilerSteps%NCCL_STEPS];
        eDescr.rank = sub->rank;
        eDescr.netPlugin.id = pluginId;
        eDescr.netPlugin.data = extData;
        ncclProfiler->startEvent(sub->profilerContext, eHandle, &eDescr);
      }
    } else { // stop
      ncclProfiler->stopEvent(*eHandle);
    }
  }
  return ncclSuccess;
}
