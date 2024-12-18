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

static pthread_mutex_t profilerLock = PTHREAD_MUTEX_INITIALIZER;
static int profilerPluginRefCount;
static void* profilerPluginLib;
static ncclProfiler_t* ncclProfiler;
static ncclProfiler_v2_t ncclProfiler_v1_as_v2;
static ncclProfiler_v1_t* ncclProfiler_v1;

static uint8_t ncclStringToFunc(const char* func) {
  if (0 == strcmp(func, "AllGather")) return ncclFuncAllGather;
  if (0 == strcmp(func, "AllReduce")) return ncclFuncAllReduce;
  if (0 == strcmp(func, "Broadcast")) return ncclFuncBroadcast;
  if (0 == strcmp(func, "Recv")) return ncclFuncRecv;
  if (0 == strcmp(func, "Reduce")) return ncclFuncReduce;
  if (0 == strcmp(func, "ReduceScatter")) return ncclFuncReduceScatter;
  if (0 == strcmp(func, "SendRecv")) return ncclFuncSendRecv;
  return ncclFuncSend;
}

static uint8_t ncclStringToAlgo(const char* algo) {
  if (0 == strcmp(algo, "TREE")) return NCCL_ALGO_TREE;
  if (0 == strcmp(algo, "RING")) return NCCL_ALGO_RING;
  if (0 == strcmp(algo, "COLLNET_DIRECT")) return NCCL_ALGO_COLLNET_DIRECT;
  if (0 == strcmp(algo, "COLLNET_CHAIN")) return NCCL_ALGO_COLLNET_CHAIN;
  if (0 == strcmp(algo, "NVLS")) return NCCL_ALGO_NVLS;
  if (0 == strcmp(algo, "NVLS_TREE")) return NCCL_ALGO_NVLS_TREE;
  return NCCL_ALGO_PAT;
}

static uint8_t ncclStringToProto(const char* proto) {
  if (0 == strcmp(proto, "LL")) return NCCL_PROTO_LL;
  if (0 == strcmp(proto, "LL128")) return NCCL_PROTO_LL128;
  return NCCL_PROTO_SIMPLE;
}

static uint8_t ncclStringToDatatype(const char* dt) {
  if (0 == strcmp(dt, "ncclInt8")) return ncclInt8;
  if (0 == strcmp(dt, "ncclInt32")) return ncclInt32;
  if (0 == strcmp(dt, "ncclUint32")) return ncclUint32;
  if (0 == strcmp(dt, "ncclInt64")) return ncclInt64;
  if (0 == strcmp(dt, "ncclUint64")) return ncclUint64;
  if (0 == strcmp(dt, "ncclFloat16")) return ncclFloat16;
  if (0 == strcmp(dt, "ncclFloat32")) return ncclFloat32;
#if defined(__CUDA_BF16_TYPES_EXIST__)
  if (0 == strcmp(dt, "ncclBfloat16")) return ncclBfloat16;
#endif
  return ncclFloat64;
}

static ncclResult_t ncclProfiler_v1_as_v2_startEvent(void* context, void** eHandle, ncclProfilerEventDescr_v2_t* eDescr) {
  ncclProfilerEventDescr_v1_t eDescr_v1 = { 0 };
  eDescr_v1.type = eDescr->type;
  eDescr_v1.parentObj = eDescr->parentObj;
  eDescr_v1.rank = eDescr->rank;
  switch(eDescr->type) {
    case ncclProfileGroup: break;
    case ncclProfileColl: {
      eDescr_v1.coll.name = eDescr->coll.name;
      eDescr_v1.coll.commHash = eDescr->coll.commHash;
      eDescr_v1.coll.seqNumber = eDescr->coll.seqNumber;
      eDescr_v1.coll.func = ncclStringToFunc(eDescr->coll.func);
      eDescr_v1.coll.sendBuff = eDescr->coll.sendBuff;
      eDescr_v1.coll.recvBuff = eDescr->coll.recvBuff;
      eDescr_v1.coll.count = eDescr->coll.count;
      eDescr_v1.coll.root = eDescr->coll.root;
      eDescr_v1.coll.datatype = ncclStringToDatatype(eDescr->coll.datatype);
      eDescr_v1.coll.op = 0; // removed in v2
      eDescr_v1.coll.trafficBytes = eDescr->coll.trafficBytes;
      eDescr_v1.coll.nMaxChannels = eDescr->coll.nMaxChannels;
      eDescr_v1.coll.nWarps = eDescr->coll.nWarps;
      eDescr_v1.coll.algo = ncclStringToAlgo(eDescr->coll.algo);
      eDescr_v1.coll.proto = ncclStringToProto(eDescr->coll.proto);
    } break;
    case ncclProfileP2p: {
      eDescr_v1.p2p.name = eDescr->p2p.name;
      eDescr_v1.p2p.commHash = eDescr->p2p.commHash;
      eDescr_v1.p2p.func = ncclStringToFunc(eDescr->p2p.func);
      eDescr_v1.p2p.buff = eDescr->p2p.buff;
      eDescr_v1.p2p.count = eDescr->p2p.count;
      eDescr_v1.p2p.datatype = ncclStringToDatatype(eDescr->p2p.datatype);
      eDescr_v1.p2p.peer = eDescr->p2p.peer;
    } break;
    case ncclProfileProxyOp: {
      eDescr_v1.proxyOp.pid = eDescr->proxyOp.pid;
      eDescr_v1.proxyOp.channelId = eDescr->proxyOp.channelId;
      eDescr_v1.proxyOp.peer = eDescr->proxyOp.peer;
      eDescr_v1.proxyOp.nSteps = eDescr->proxyOp.nSteps;
      eDescr_v1.proxyOp.chunkSize = eDescr->proxyOp.chunkSize;
      eDescr_v1.proxyOp.isSend = eDescr->proxyOp.isSend;
    } break;
    case ncclProfileProxyStep: {
      eDescr_v1.proxyStep.step = eDescr->proxyStep.step;
    } break;
    case ncclProfileProxyCtrl: break;
    default:;
  }
  return ncclProfiler_v1->startEvent(context, eHandle, &eDescr_v1);
}

static ncclResult_t ncclProfiler_v1_as_v2_init(void** context, int* eActivationMask) {
  ncclProfiler_v1->init(context, eActivationMask);
  ncclProfiler_v1_as_v2.startEvent = ncclProfiler_v1_as_v2_startEvent;
  ncclProfiler_v1_as_v2.stopEvent = ncclProfiler_v1->stopEvent;
  ncclProfiler_v1_as_v2.recordEventState = ncclProfiler_v1->recordEventState;
  ncclProfiler_v1_as_v2.finalize = ncclProfiler_v1->finalize;
  return ncclSuccess;
}

#define MAX_STR_LEN 256

static void* tryOpenLib(char* name, int *err, char* errStr) {
  if (nullptr == name || strlen(name) == 0) {
    return nullptr;
  }

  if (strncasecmp(name, "STATIC_PLUGIN", strlen(name)) == 0) {
    name = nullptr;
  }

  void *handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (nullptr == handle) {
    strncpy(errStr, dlerror(), MAX_STR_LEN);
    errStr[MAX_STR_LEN] = 0;
    if (name && strstr(errStr, name) && strstr(errStr, "No such file or directory")) {
      *err = ENOENT;
    }
  }

  return handle;
}

static char* tryOpenLibCheck(int openErr, char* openErrStr, char* nameList, int *nameListLen, char* name) {
  if (openErr == ENOENT) {
    snprintf(nameList, *nameListLen, " %s", name);
    nameList += strlen(name) + 1;
    *nameListLen -= strlen(name) + 1;
    return nameList;
  }
  INFO(NCCL_ENV, "PROFILER/Plugin: %s", openErrStr);
  return nameList;
}

static void* openProfilerPluginLib(char* couldNotFindNames, int len) {
  int openErr;
  void *pluginLib;
  char profilerPluginLibName[PATH_MAX];
  char openErrStr[MAX_STR_LEN + 1] = { 0 };

  const char *envProfilerPluginName = getenv("NCCL_PROFILER_PLUGIN");
  if (envProfilerPluginName && strlen(envProfilerPluginName)) {
    snprintf(profilerPluginLibName, PATH_MAX, "%s", envProfilerPluginName);
    pluginLib = tryOpenLib(profilerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: Plugin name set by env to %s", profilerPluginLibName);
      return pluginLib;
    }

    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, profilerPluginLibName);
    pluginLib = tryOpenLib(profilerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: Plugin name set by env to %s", profilerPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, profilerPluginLibName);
  } else {
    snprintf(profilerPluginLibName, PATH_MAX, "libnccl-profiler.so");
    pluginLib = tryOpenLib(profilerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, profilerPluginLibName);
  }

  return nullptr;
}

enum {
  profilerPluginLoadFailed = -1,
  profilerPluginLoadReady = 0,
  profilerPluginLoadSuccess = 1,
};
static int profilerPluginStatus = profilerPluginLoadReady;
static pid_t pid;

#define MAX_PLUGIN_LOAD 2

static ncclResult_t ncclProfilerPluginLoad(void) {
  if (profilerPluginLoadFailed == profilerPluginStatus) {
    return ncclSuccess;
  }

  char couldNotFindNames[MAX_PLUGIN_LOAD * PATH_MAX] = { 0 };
  pthread_mutex_lock(&profilerLock);
  if (profilerPluginLoadSuccess == profilerPluginStatus) {
    ++profilerPluginRefCount;
    goto exit;
  }

  profilerPluginLib = openProfilerPluginLib(couldNotFindNames, MAX_PLUGIN_LOAD * PATH_MAX);
  if (profilerPluginLib == nullptr) {
    if (strlen(couldNotFindNames)) {
      INFO(NCCL_ENV, "PROFILER/Plugin: Could not find:%s.", couldNotFindNames);
    }
    goto fail;
  }

  ncclProfiler = (ncclProfiler_v2_t*)dlsym(profilerPluginLib, "ncclProfiler_v2");
  if (ncclProfiler == nullptr) {
    INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: failed to find ncclProfiler_v2.");
    ncclProfiler_v1 = (ncclProfiler_v1_t*)dlsym(profilerPluginLib, "ncclProfiler_v1");
    if (ncclProfiler_v1 == nullptr) {
      INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: failed to find ncclProfiler_v1.");
      goto fail;
    } else {
      ncclProfiler = &ncclProfiler_v1_as_v2;
      ncclProfiler_v1_as_v2.name = ncclProfiler_v1->name;
      ncclProfiler_v1_as_v2.init = ncclProfiler_v1_as_v2_init;
      INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: loaded ncclProfiler_v1.");
    }
  } else {
    INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: loaded ncclProfiler_v2.");
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
  if (profilerPluginLib) dlclose(profilerPluginLib);
  profilerPluginStatus = profilerPluginLoadFailed;
  goto exit;
}

static ncclResult_t ncclProfilerPluginUnload(void) {
  pthread_mutex_lock(&profilerLock);
  if (0 == (--profilerPluginRefCount)) {
    INFO(NCCL_ENV, "PROFILER/Plugin: Closing profiler plugin %s", ncclProfiler->name);
    dlclose(profilerPluginLib);
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


static int eActivationMask;       // Set by profiler
static int eActivationMaskGroup;  // Cached for current group

ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) {
  TIME_START_EVENT(elapsed);
  TIME_START_EVENT(init);
  ncclProfilerPluginLoad();
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    int err = ncclProfiler->init(&comm->profilerContext, &eActivationMask);
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
  eActivationMaskGroup = __atomic_load_n(&eActivationMask, __ATOMIC_RELAXED);
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (eActivationMaskGroup & (ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep)) {
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
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    int enable = eActivationMaskGroup & (ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileColl);
    if (plan->groupEventHandle && enable) {
      struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
      while (ct) {
        ncclProfilerEventDescr_t eDescr = { 0 };
        eDescr.type = ncclProfileColl;
        eDescr.parentObj = plan->groupEventHandle;
        eDescr.rank = plan->comm->rank;
        eDescr.coll.name = plan->comm->commName;
        eDescr.coll.commHash = plan->comm->commHash;
        eDescr.coll.seqNumber = plan->comm->seqNumber[ct->func]++;
        eDescr.coll.func = ncclFuncToString(ct->func);
        eDescr.coll.sendBuff = ct->sendbuff;
        eDescr.coll.recvBuff = ct->recvbuff;
        eDescr.coll.count = ct->count;
        eDescr.coll.root = ct->root;
        eDescr.coll.datatype = ncclDatatypeToString(ct->datatype);
        eDescr.coll.trafficBytes = ct->trafficBytes;
        eDescr.coll.nMaxChannels = ct->nMaxChannels;
        eDescr.coll.nWarps = ct->nWarps;
        eDescr.coll.algo = ncclAlgoToString(ct->algorithm);
        eDescr.coll.proto = ncclProtoToString(ct->protocol);
        ncclProfiler->startEvent(plan->comm->profilerContext, &ct->eventHandle, &eDescr);

        // update collective task with group event activation mask
        ct->eActivationMask = eActivationMaskGroup;
        ct = ct->next;
      }
      struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
      while (pt) {
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

        // update collective task with group event activation mask
        pt->eActivationMask = eActivationMaskGroup;
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
    int enable = eActivationMaskGroup & (ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileColl);
    if (plan->groupEventHandle && enable) {
      struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
      while (ct) {
        ncclProfiler->stopEvent(ct->eventHandle);
        ct = ct->next;
      }
      struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
      while (pt) {
        ncclProfiler->stopEvent(pt->eventHandle);
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
    if (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileProxyOp)) {
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
    if (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileProxyOp)) {
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
    if (sub->opEventHandle && (sub->eActivationMask & ncclProfileProxyStep)) {
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
    if (sub->opEventHandle && (sub->eActivationMask & ncclProfileProxyStep)) {
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
    int eActivationMaskProxy = __atomic_load_n(&eActivationMask, __ATOMIC_RELAXED);
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
  if (__builtin_expect(ncclProfiler != NULL, 0) && eHandle && __atomic_load_n(&eActivationMask, __ATOMIC_RELAXED) & ncclProfileProxyCtrl) {
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
