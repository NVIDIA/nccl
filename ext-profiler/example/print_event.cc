/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include "err.h"
#include "profiler.h"
#include "event.h"
#include "print_event.h"
#include <cuda_runtime.h>

#define __hidden __attribute__ ((visibility("hidden")))

// FIXME: chrome tracing asynchronous events (following used) allow event nesting for events that have same id and category
// It appears that nesting more than three events causes issues. Therefore, every event is given an increasing id and a
// category that matches the type of event (GROUP API, COLL API, P2P API, GROUP, COLL, P2P, PROXY, NET)
static __thread int groupApiId;
__hidden void printGroupApiEventHeader(FILE* fh, struct groupApi* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"GROUP_API\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"groupApiId\": %d, \"groupDepth\": %d}},\n",
          "Group API", groupApiId, getpid(), 1, event->startTs, event->groupApiId, event->groupDepth);
}

__hidden void printGroupApiEventTrailer(FILE* fh, struct groupApi* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"GROUP_API\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          "Group API", groupApiId++, getpid(), 1, event->stopTs);
}

static __thread int p2pApiId;
__hidden void printP2pApiEventHeader(FILE* fh, struct p2pApi* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"P2P_API\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"count\": %lu, \"datatype\": \"%s\", \"GraphCaptured\": %d, \"Stream\": \"%p\"}},\n",
          event->func, p2pApiId, getpid(), 1, event->startTs, event->count, event->datatype, event->graphCaptured, event->stream);
}

__hidden void printP2pApiEventTrailer(FILE* fh, struct p2pApi* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"P2P_API\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          event->func, p2pApiId++, getpid(), 1, event->stopTs);
}

static __thread int collApiId;
__hidden void printCollApiEventHeader(FILE* fh, struct collApi* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"COLL_API\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"count\": %lu, \"datatype\": \"%s\", \"root\": %d, \"GraphCaptured\": %d, \"Stream\": \"%p\"}},\n",
          event->func, collApiId, getpid(), 1, event->startTs, event->count, event->datatype, event->root, event->graphCaptured, event->stream);
}

__hidden void printCollApiEventTrailer(FILE* fh, struct collApi* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"COLL_API\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          event->func, collApiId++, getpid(), 1, event->stopTs);
}

static __thread int kernelLaunchId;
__hidden void printKernelLaunchEventHeader(FILE* fh, struct kernelLaunch* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"KERNEL_LAUNCH\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"groupId\": %d, \"Stream\": \"%p\"}},\n", "KernelLaunch", kernelLaunchId, getpid(), 1, event->startTs, event->kernelLaunchId, event->stream);
}

__hidden void printKernelLaunchEventTrailer(FILE* fh, struct kernelLaunch* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"KERNEL_LAUNCH\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n", "KernelLaunch", kernelLaunchId++, getpid(), 1, event->stopTs);
}

static __thread int groupId;
__hidden void printGroupEventHeader(FILE* fh, struct group* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"GROUP\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"groupId\": %d}},\n",
          "Group", groupId, getpid(), 1, event->startTs, event->groupId);
}

__hidden void printGroupEventTrailer(FILE* fh, struct group* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"GROUP\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          "Group", groupId++, getpid(), 1, event->stopTs);
}

static __thread int collId;
__hidden void printCollEventHeader(FILE* fh, struct collective* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"COLL\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"SeqNum\": %lu, \"CommHash\": %lu, \"Rank\": %d, \"Count\": %lu, \"Datatype\": \"%s\", \"Algorithm\": \"%s\", \"Protocol\": \"%s\", \"nChannels\": %d}},\n",
          event->base.func, collId, getpid(), 1, event->base.startTs, event->seqNumber, ((struct collApi*)event->base.parent)->ctx->commHash, event->base.rank, event->count, event->datatype, event->algo, event->proto, event->nChannels);
}

__hidden void printCollEventTrailer(FILE* fh, struct collective* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"COLL\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          event->base.func, collId++, getpid(), 1, event->base.stopTs);
}

static __thread int p2pId;
__hidden void printP2pEventHeader(FILE* fh, struct p2p* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"P2P\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"CommHash\": %lu, \"Rank\": %d, \"Peer\": %d, \"Count\": %lu, \"Datatype\": \"%s\", \"nChannels\": %d}},\n",
          event->base.func, p2pId, getpid(), 1, event->base.startTs, ((struct p2pApi*)event->base.parent)->ctx->commHash, event->base.rank, event->peer, event->count, event->datatype, event->nChannels);
}

__hidden void printP2pEventTrailer(FILE* fh, struct p2p* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"P2P\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          event->base.func, p2pId++, getpid(), 1, event->base.stopTs);
}

static __thread int proxyOpId;
__hidden void printProxyOpEventHeader(FILE* fh, struct proxyOp* event) {
  if (event->isSend) {
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"PROXY\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Channel\": %d, \"Peer\": %d, \"Steps\": %d, \"ChunkSize\": %d, \"transSize\": %lu}},\n",
            "ScheduleSend", proxyOpId, getpid(), 1, event->startTs, event->channelId, event->peer, event->nSteps, event->chunkSize, event->transSize);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"PROXY\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
            "ScheduleSend", proxyOpId, getpid(), 1, event->progrTs);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"PROXY\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Channel\": %d, \"Peer\": %d, \"Steps\": %d, \"ChunkSize\": %d, \"transSize\": %lu}},\n",
            "ProgressSend", proxyOpId, getpid(), 1, event->progrTs, event->channelId, event->peer, event->nSteps, event->chunkSize, event->transSize);
  } else {
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"PROXY\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Channel\": %d, \"Peer\": %d, \"Steps\": %d, \"ChunkSize\": %d, \"transSize\": %lu}},\n",
            "ScheduleRecv", proxyOpId, getpid(), 1, event->startTs, event->channelId, event->peer, event->nSteps, event->chunkSize, event->transSize);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"PROXY\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
            "ScheduleRecv", proxyOpId, getpid(), 1, event->progrTs);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"PROXY\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Channel\": %d, \"Peer\": %d, \"Steps\": %d, \"ChunkSize\": %d, \"transSize\": %lu}},\n",
            "ProgressRecv", proxyOpId, getpid(), 1, event->progrTs, event->channelId, event->peer, event->nSteps, event->chunkSize, event->transSize);
  }
}

__hidden void printProxyOpEventTrailer(FILE* fh, struct proxyOp* event) {
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"PROXY\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          event->isSend ? "ProgressSend" : "ProgressRecv", proxyOpId++, getpid(), 1, event->stopTs);
}

static __thread int proxyStepId;
__hidden void printProxyStepEventHeader(FILE* fh, struct proxyStep* event) {
  if (event->isSend) {
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Step\": %d}},\n",
            "SendGpuWait", proxyStepId, getpid(), 1, event->timestamp[PROXY_STEP_SEND_GPU_WAIT], event->step);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
            "SendGpuWait", proxyStepId, getpid(), 1, event->timestamp[PROXY_STEP_SEND_PEER_WAIT]);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Step\": %d}},\n",
            "SendPeerWait", proxyStepId, getpid(), 1, event->timestamp[PROXY_STEP_SEND_PEER_WAIT], event->step);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
            "SendPeerWait", proxyStepId, getpid(), 1, event->timestamp[PROXY_STEP_SEND_WAIT]);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Step\": %d}},\n",
            "SendWait", proxyStepId, getpid(), 1, event->timestamp[PROXY_STEP_SEND_WAIT], event->step);
  } else {
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Step\": %d}},\n",
            "RecvWait", proxyStepId, getpid(), 1, event->timestamp[PROXY_STEP_RECV_WAIT], event->step);
  }
}

__hidden void printProxyStepEventTrailer(FILE* fh, struct proxyStep* event) {
  if (event->isSend) {
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
            "SendWait", proxyStepId++, getpid(), 1, event->stopTs);
  } else {
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
            "RecvWait", proxyStepId, getpid(), 1, event->timestamp[PROXY_STEP_RECV_FLUSH_WAIT]);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Step\": %d}},\n",
            "RecvFlushWait", proxyStepId, getpid(), 1, event->timestamp[PROXY_STEP_RECV_FLUSH_WAIT], event->step);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
            "RecvFlushWait", proxyStepId, getpid(), 1, event->timestamp[PROXY_STEP_RECV_GPU_WAIT]);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Step\": %d}},\n",
            "RecvGpuWait", proxyStepId, getpid(), 1, event->timestamp[PROXY_STEP_RECV_GPU_WAIT], event->step);
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
            "RecvGpuWait", proxyStepId++, getpid(), 1, event->stopTs);
  }
}

static __thread int kernelId;
__hidden void printKernelChEventHeader(FILE* fh, struct kernelCh* event) {
  if (event->type != ncclProfileKernelCh) return;
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"GPU\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"Channel\": %d, \"StartGpuClk\": %lu, \"StopGpuClk\": %lu}},\n",
          "KernelCh", kernelId, getpid(), 1, event->startTs, event->channelId, event->startGpuClk, event->stopGpuClk);
}

__hidden void printKernelChEventTrailer(FILE* fh, struct kernelCh* event) {
  if (event->type != ncclProfileKernelCh) return;
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"GPU\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          "KernelCh", kernelId, getpid(), 1, event->stopTs);
}

static __thread int proxyCtrlId;
__hidden void printProxyCtrlEvent(FILE* fh, struct proxyCtrl* event) {
  const char* str;
  if (event->state == ncclProfilerProxyCtrlIdle || event->state == ncclProfilerProxyCtrlActive) {
    str = "Idle";
  } else if (event->state == ncclProfilerProxyCtrlSleep || event->state == ncclProfilerProxyCtrlWakeup) {
    str = "Sleep";
  } else if (event->state == ncclProfilerProxyCtrlAppend || event->state == ncclProfilerProxyCtrlAppendEnd) {
    str = "Append";
  } else {
    return;
  }
  if (event->state == ncclProfilerProxyCtrlAppendEnd) {
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"PROXY\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"appended\": %d}},\n",
            str, proxyCtrlId, getpid(), 1, event->startTs, event->appended);
  } else {
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"PROXY\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
            str, proxyCtrlId, getpid(), 1, event->startTs);
  }
  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"PROXY\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          str, proxyCtrlId++, getpid(), 1, event->stopTs);
}

static __thread int ibQpId, sockId;
__hidden void printNetPluginEvent(FILE* fh, struct netPlugin* event) {
  if (event->pluginType == NCCL_PROFILER_NET_TYPE_IB) {
    if (event->pluginVer == 1) {
      if (event->pluginEvent == ncclProfileQp) {
        fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET_IB\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"device\": %d, \"qp_num\": %d, \"opcode\": %d, \"wr_id\": %lu, \"size\": %lu}},\n",
                "Qp", ibQpId, getpid(), 1, event->startTs, event->qp.device, event->qp.qpNum, event->qp.opcode, event->qp.wr_id, event->qp.length);
        fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET_IB\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
                "Qp", ibQpId++, getpid(), 1, event->stopTs);
      }
    }
  } else if (event->pluginType == NCCL_PROFILER_NET_TYPE_SOCK) {
    if (event->pluginVer == 1) {
      if (event->pluginEvent == ncclProfileSocket) {
        fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET_SOCK\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"sock\": %d, \"op\": %d, \"size\": %lu}},\n",
                "Sock", sockId, getpid(), 1, event->startTs, event->sock.fd, event->sock.op, event->sock.length);
        fprintf(fh, "{\"name\": \"%s\", \"cat\": \"NET_SOCK\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
                "Sock", sockId++, getpid(), 1, event->stopTs);
      }
    }
  }
}

//#define DEBUG_EVENTS
// Debug helper functions for CE events
#ifdef DEBUG_EVENTS
static void debugCeCollEvent(FILE* fh, struct ceColl* event, const char* tag) {
  fprintf(fh, "CeColl event %p tag = %s {\n", event, tag);
  fprintf(fh, "  func              = %s\n", event->base.func);
  fprintf(fh, "  eventId           = %lu\n", event->eventId);
  fprintf(fh, "  timingMode        = %s\n", event->timingMode == CE_TIMING_GPU ? "gpu" : "cpu");
  fprintf(fh, "  startCompleted    = %d\n", event->startCompleted);
  fprintf(fh, "  stopCompleted     = %d\n", event->stopCompleted);
  fprintf(fh, "  startTs           = %f\n", event->base.startTs);
  fprintf(fh, "  stopTs            = %f\n", event->base.stopTs);
  fprintf(fh, "  cpuDuration       = %lu us\n", event->cpuDuration);
  fprintf(fh, "  elapsedTime       = %lu us\n", event->elapsedTime);
  fprintf(fh, "}\n");
}

static void debugCeSyncEvent(FILE* fh, struct ceSync* event, const char* tag) {
  fprintf(fh, "CeSync event %p tag = %s {\n", event, tag);
  fprintf(fh, "  syncType          = %s\n", event->isComplete ? "Complete" : "Ready");
  fprintf(fh, "  eventId           = %lu\n", event->eventId);
  fprintf(fh, "  timingMode        = %s\n", event->timingMode == CE_TIMING_GPU ? "gpu" : "cpu");
  fprintf(fh, "  startCompleted    = %d\n", event->startCompleted);
  fprintf(fh, "  stopCompleted     = %d\n", event->stopCompleted);
  fprintf(fh, "  startTs           = %f\n", event->base.startTs);
  fprintf(fh, "  stopTs            = %f\n", event->base.stopTs);
  fprintf(fh, "  cpuDuration       = %lu us\n", event->cpuDuration);
  fprintf(fh, "}\n");
}

static void debugCeBatchEvent(FILE* fh, struct ceBatch* event, const char* tag) {
  fprintf(fh, "CeBatch event %p tag = %s {\n", event, tag);
  fprintf(fh, "  numOps            = %d\n", event->numOps);
  fprintf(fh, "  totalBytes        = %lu\n", event->totalBytes);
  fprintf(fh, "  eventId           = %lu\n", event->eventId);
  fprintf(fh, "  timingMode        = %s\n", event->timingMode == CE_TIMING_GPU ? "gpu" : "cpu");
  fprintf(fh, "  startCompleted    = %d\n", event->startCompleted);
  fprintf(fh, "  stopCompleted     = %d\n", event->stopCompleted);
  fprintf(fh, "  startTs           = %f\n", event->base.startTs);
  fprintf(fh, "  stopTs            = %f\n", event->base.stopTs);
  fprintf(fh, "  cpuDuration       = %lu us\n", event->cpuDuration);
  fprintf(fh, "}\n");
}
#endif

void debugEvent(void* eHandle, const char* tag) {
#ifdef DEBUG_EVENTS
  char filename[64] = { 0 };
  sprintf(filename, "EventDebug-%d", getpid());
  FILE* fh = fopen(filename, "a+");
  uint64_t type = *(uint64_t *)eHandle;
  if (type == ncclProfileGroup) {
    struct group* event = (struct group *)eHandle;
    fprintf(fh, "Group event %p tag = %s {\n", event, tag);
    fprintf(fh, "  refCount          = %d\n", __atomic_load_n(&event->refCount, __ATOMIC_RELAXED));
    fprintf(fh, "  startTs           = %f\n", event->startTs);
    fprintf(fh, "  stopTs            = %f\n", event->stopTs);
    fprintf(fh, "}\n");
  } else if (type == ncclProfileColl) {
    struct collective* event = (struct collective *)eHandle;
    fprintf(fh, "Collective event %p tag = %s {\n", event, tag);
    fprintf(fh, "  refCount          = %d\n", __atomic_load_n(&event->base.refCount, __ATOMIC_RELAXED));
    fprintf(fh, "  parent            = %p\n", event->base.parent);
    for (int j = 0; j < 2*MAX_OPS; j++) {
      for (int i = 0; i < MAX_CHANNELS; i++) if (event->op[i][j].type == ncclProfileProxyOp) fprintf(fh, "  op[%d]           = %p\n", i, &event->op[i]);
    }
    fprintf(fh, "  startTs           = %f\n", event->base.startTs);
    fprintf(fh, "  stopTs            = %f\n", event->base.stopTs);
    fprintf(fh, "}\n");
  } else if (type == ncclProfileP2p) {
    struct p2p* event = (struct p2p *)eHandle;
    fprintf(fh, "P2p event %p tag = %s {\n", event, tag);
    fprintf(fh, "  refCount          = %d\n", __atomic_load_n(&event->base.refCount, __ATOMIC_RELAXED));
    fprintf(fh, "  parent            = %p\n", event->base.parent);
    fprintf(fh, "  op                = %p\n", &event->op);
    fprintf(fh, "  startTs           = %f\n", event->base.startTs);
    fprintf(fh, "  stopTs            = %f\n", event->base.stopTs);
    fprintf(fh, "}\n");
  } else if (type == ncclProfileProxyOp) {
    struct proxyOp* event = (struct proxyOp *)eHandle;
    fprintf(fh, "ProxyOp event %p tag = %s {\n", event, tag);
    fprintf(fh, "  type              = %s\n", event->isSend < 0 ? "Unknown" : event->isSend ? "Send" : "Recv");
    fprintf(fh, "  channel           = %d\n", event->channelId);
    fprintf(fh, "  parent            = %p\n", event->parent);
    fprintf(fh, "  rank              = %d\n", event->rank);
    fprintf(fh, "  startTs           = %f\n", event->startTs);
    fprintf(fh, "  progrTs           = %f\n", event->progrTs);
    fprintf(fh, "  stopTs            = %f\n", event->stopTs);
    fprintf(fh, "}\n");
  } else if (type == ncclProfileProxyStep) {
    struct proxyStep* event = (struct proxyStep *)eHandle;
    fprintf(fh, "ProxyStep event %p tag = %s {\n", event, tag);
    fprintf(fh, "  type              = %s\n", event->isSend < 0 ? "Unknown" : event->isSend ? "Send" : "Recv");
    fprintf(fh, "  parent            = %p\n", event->parent);
    fprintf(fh, "  startTs           = %f\n", event->startTs);
    fprintf(fh, "  stopTs            = %f\n", event->stopTs);
    fprintf(fh, "}\n");
  } else if (type == ncclProfileKernelCh) {
    struct kernelCh* event = (struct kernelCh *)eHandle;
    fprintf(fh, "KernelCh event %p tag = %s {\n", event, tag);
    fprintf(fh, "  parent            = %p\n", event->parent);
    fprintf(fh, "  channel           = %d\n", event->channelId);
  } else if (type == ncclProfileNetPlugin) {
    struct netPlugin* event = (struct netPlugin *)eHandle;
    fprintf(fh, "NetPlugin event %p tag = %s {\n", event, tag);
    fprintf(fh, "  pluginType        = %d\n", event->pluginType);
    fprintf(fh, "  pluginVer         = %d\n", event->pluginVer);
    fprintf(fh, "  pluginEvent       = %d\n", event->pluginEvent);
    fprintf(fh, "  startTs           = %f\n", event->startTs);
    fprintf(fh, "  stopTs            = %f\n", event->stopTs);
    fprintf(fh, "}\n");
  } else if (type == ncclProfileCeColl) {
    debugCeCollEvent(fh, (struct ceColl*)eHandle, tag);
  } else if (type == ncclProfileCeSync) {
    debugCeSyncEvent(fh, (struct ceSync*)eHandle, tag);
  } else if (type == ncclProfileCeBatch) {
    debugCeBatchEvent(fh, (struct ceBatch*)eHandle, tag);
  }
  fclose(fh);
#endif
}

// CE event print functions
static int ceCollId = 0;
static void printCeCollEvent(FILE* fh, struct ceColl* event) {
  if (event->timingMode == CE_TIMING_GPU) {
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"CE_COLL\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"eventId\": %lu, \"count\": %lu, \"datatype\": \"%s\", \"strategy\": \"%s\", \"start_ts_cpu\": %f, \"stop_ts_cpu\": %f, \"duration_cpu_us\": %f, \"duration_gpu_us\": %lu}},\n",
            event->base.func, ceCollId, getpid(), 1, event->base.startTs, event->eventId, event->count, event->datatype, event->syncStrategy, event->cpuStartTime, event->cpuStopTime, event->cpuDuration, event->elapsedTime);
  } else {
    fprintf(fh, "{\"name\": \"%s\", \"cat\": \"CE_COLL\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"eventId\": %lu, \"count\": %lu, \"datatype\": \"%s\", \"strategy\": \"%s\", \"start_ts_cpu\": %f, \"stop_ts_cpu\": %f, \"duration_cpu_us\": %f}},\n",
            event->base.func, ceCollId, getpid(), 1, event->base.startTs, event->eventId, event->count, event->datatype, event->syncStrategy, event->cpuStartTime, event->cpuStopTime, event->cpuDuration);
  }

  // Print child events (CeSync and CeBatch)
  struct taskEventBase* child = taskEventQueueHead(event);
  while (child) {
    struct taskEventBase* next = child->next;
    printEvent(fh, child);
    child = next;
  }

  fprintf(fh, "{\"name\": \"%s\", \"cat\": \"CE_COLL\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          event->base.func, ceCollId++, getpid(), 1, event->base.stopTs);
}

static int ceSyncId = 0;
static void printCeSyncEvent(FILE* fh, struct ceSync* event) {
  const char* syncTypeStr = event->isComplete ? "Complete" : "Ready";
  if (event->timingMode == CE_TIMING_GPU) {
    fprintf(fh, "{\"name\": \"CeSync\", \"cat\": \"CE_SYNC\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"eventId\": %lu, \"type\": \"%s\", \"strategy\": \"%s\", \"nRanks\": %d, \"start_ts_cpu\": %f, \"stop_ts_cpu\": %f, \"duration_cpu_us\": %f, \"duration_gpu_us\": %lu}},\n",
            ceSyncId, getpid(), 1, event->base.startTs, event->eventId, syncTypeStr, event->parent->syncStrategy, event->nRanks, event->cpuStartTime, event->cpuStopTime, event->cpuDuration, event->elapsedTime);
  } else {
    fprintf(fh, "{\"name\": \"CeSync\", \"cat\": \"CE_SYNC\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"eventId\": %lu, \"type\": \"%s\", \"strategy\": \"%s\", \"nRanks\": %d, \"start_ts_cpu\": %f, \"stop_ts_cpu\": %f, \"duration_cpu_us\": %f}},\n",
            ceSyncId, getpid(), 1, event->base.startTs, event->eventId, syncTypeStr, event->parent->syncStrategy, event->nRanks, event->cpuStartTime, event->cpuStopTime, event->cpuDuration);
  }
  fprintf(fh, "{\"name\": \"CeSync\", \"cat\": \"CE_SYNC\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          ceSyncId++, getpid(), 1, event->base.stopTs);
}

static int ceBatchId = 0;
static void printCeBatchEvent(FILE* fh, struct ceBatch* event) {
  if (event->timingMode == CE_TIMING_GPU) {
    fprintf(fh, "{\"name\": \"CeBatch\", \"cat\": \"CE_BATCH\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"eventId\": %lu, \"numOps\": %d, \"totalBytes\": %lu, \"start_ts_cpu\": %f, \"stop_ts_cpu\": %f, \"duration_cpu_us\": %f, \"duration_gpu_us\": %lu}},\n",
            ceBatchId, getpid(), 1, event->base.startTs, event->eventId, event->numOps, event->totalBytes, event->cpuStartTime, event->cpuStopTime, event->cpuDuration, event->elapsedTime);
  } else {
    fprintf(fh, "{\"name\": \"CeBatch\", \"cat\": \"CE_BATCH\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f, \"args\": {\"eventId\": %lu, \"numOps\": %d, \"totalBytes\": %lu, \"start_ts_cpu\": %f, \"stop_ts_cpu\": %f, \"duration_cpu_us\": %f}},\n",
            ceBatchId, getpid(), 1, event->base.startTs, event->eventId, event->numOps, event->totalBytes, event->cpuStartTime, event->cpuStopTime, event->cpuDuration);
  }
  fprintf(fh, "{\"name\": \"CeBatch\", \"cat\": \"CE_BATCH\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": %d, \"ts\": %f},\n",
          ceBatchId++, getpid(), 1, event->base.stopTs);
}

void printEvent(FILE* fh, void* handle) {
  if (handle == NULL || fh == NULL) return;
  uint64_t type = *(uint64_t *)handle;
  if (type == ncclProfileGroupApi) {
    struct groupApi* g = (struct groupApi*) handle;
    printGroupApiEventHeader(fh, g);
    struct kernelLaunch* kernelLaunchHead = profilerQueueHead(&g->kernelLaunchEvents);
    while (kernelLaunchHead != NULL) {
      printEvent(fh, kernelLaunchHead);
      kernelLaunchHead = kernelLaunchHead->next;
    }
    struct collApi* collApiHead = profilerQueueHead(&g->collApiEvents);
    while (collApiHead != NULL) {
      printEvent(fh, collApiHead);
      collApiHead = collApiHead->next;
    }
    struct p2pApi* p2pApiHead = profilerQueueHead(&g->p2pApiEvents);
    while (p2pApiHead != NULL) {
      printEvent(fh, p2pApiHead);
      p2pApiHead = p2pApiHead->next;
    }
    printGroupApiEventTrailer(fh, g);
  } else if (type == ncclProfileCollApi) {
    struct collApi* collApiEvent = (struct collApi *) handle;
    printCollApiEventHeader(fh, collApiEvent);
    struct taskEventBase* base = taskEventQueueHead(collApiEvent);
    while (base) {
      struct taskEventBase* next = base->next;
      printEvent(fh, base);
      base = next;
    }
    printCollApiEventTrailer(fh, collApiEvent);
  } else if (type == ncclProfileP2pApi) {
    struct p2pApi* p2pApiEvent = (struct p2pApi *) handle;
    printP2pApiEventHeader(fh, p2pApiEvent);
    struct taskEventBase* base = taskEventQueueHead(p2pApiEvent);
    while (base) {
      struct taskEventBase* next = base->next;
      printEvent(fh, base);
      base = next;
    }
    printP2pApiEventTrailer(fh, p2pApiEvent);
  } else if (type == ncclProfileKernelLaunch) {
    struct kernelLaunch* kernelLaunchEvent = (struct kernelLaunch *) handle;
    printKernelLaunchEventHeader(fh, kernelLaunchEvent);
    printKernelLaunchEventTrailer(fh, kernelLaunchEvent);
  } else if (type == ncclProfileGroup) {
    struct group* g = (struct group *)handle;
    printGroupEventHeader(fh, g);
    struct taskEventBase* base = taskEventQueueHead(g);
    while (base) {
      struct taskEventBase* next = base->next;
      printEvent(fh, base);
      base = next;
    }
    printGroupEventTrailer(fh, g);
  } else if (type == ncclProfileColl) {
    struct collective* c = (struct collective *)handle;
    printCollEventHeader(fh, c);
    for (int i = 0; i < MAX_CHANNELS; i++) {
      printKernelChEventHeader(fh, &c->kernel[i]);
      for (int j = 0; j < c->nProxyOps[i]; j++) {
        printEvent(fh, &c->op[i][j]);
      }
      printKernelChEventTrailer(fh, &c->kernel[i]);
    }
    printCollEventTrailer(fh, c);
  } else if (type == ncclProfileP2p) {
    struct p2p* p = (struct p2p *)handle;
    printP2pEventHeader(fh, p);
    for (int i = 0; i < MAX_CHANNELS; i++) {
      printKernelChEventHeader(fh, &p->kernel[i]);
      printEvent(fh, &p->op[i]);
      printKernelChEventTrailer(fh, &p->kernel[i]);
    }
    printP2pEventTrailer(fh, p);
  } else if (type == ncclProfileProxyOp) {
    struct proxyOp* p = (struct proxyOp *)handle;
    printProxyOpEventHeader(fh, p);
    for (int i = 0; i < MAX_STEPS; i++) {
      printEvent(fh, &p->step[i]);
    }
    printProxyOpEventTrailer(fh, p);
  } else if (type == ncclProfileProxyStep) {
    struct proxyStep* p = (struct proxyStep *)handle;
    printProxyStepEventHeader(fh, p);
    for (int q = 0; q < p->nNetEvents; q++) {
      printNetPluginEvent(fh, &p->net[q]);
    }
    printProxyStepEventTrailer(fh, p);
  } else if (type == ncclProfileProxyCtrl) {
    struct proxyCtrl* p = (struct proxyCtrl *)handle;
    printProxyCtrlEvent(fh, p);
  } else if (type == ncclProfileCeColl) {
    struct ceColl* ce = (struct ceColl*)handle;
    printCeCollEvent(fh, ce);
  } else if (type == ncclProfileCeSync) {
    struct ceSync* ce = (struct ceSync*)handle;
    printCeSyncEvent(fh, ce);
  } else if (type == ncclProfileCeBatch) {
    struct ceBatch* ce = (struct ceBatch*)handle;
    printCeBatchEvent(fh, ce);
  }
  return;
}
