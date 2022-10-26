/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_profiler.h"

#include <stdlib.h>
#include <stdio.h>
#include "timer.h"

/* Profiler attribute definitions internal to this profiler implementation */

static const char* profilingStateSendStr[] = { "BufferWait", "GPUWait", "SendWait", "", "End" };
static const char* profilingStateRecvStr[] = { "BufferWait", "RecvWait", "FlushWait", "GPUWait", "End" };
static const char* profilingEventStr[] = { "SendRecv", "Sleep", "Idle", "Append" };
struct ncclProxyProfileEvent {
  uint64_t opCount;
  int peer;
  int step;
  uint16_t channel;
  ncclProxyPattern_t type; // send / recv
  uint8_t opIndex;
  double timestamp[6];
};

struct ncclProxyProfileEvent* profilingEvents = NULL;
int profilingIndex = 0;
double profilingStart = 0;
#define MAX_EVENTS 200000

/* Implementation of NCCL profiler APIs */

#define __hidden __attribute__ ((visibility("hidden")))

__hidden ncclResult_t profilingRecord(ncclProxyProfileInfo_t* args, int state, void** profileEvent) {
  if (profilingEvents == NULL) {
    profilingEvents = (struct ncclProxyProfileEvent*)calloc(MAX_EVENTS, sizeof(ncclProxyProfileEvent));
    profilingStart = gettime();
  }
  struct ncclProxyProfileEvent* event = NULL;
  if (state%8 == 0) {
    if (profilingIndex == MAX_EVENTS) return ncclSuccess;
    *profileEvent = event = profilingEvents+profilingIndex++;
    if (state == ncclProxyProfileBegin) {
      // Proxy operation information
      event->opCount = args->opCount;
      event->channel = args->channel;
      event->peer = args->peer;
      event->type = args->type;
      event->step = args->step;
      event->opIndex = args->opIndex;
    } else event->peer = -state;
  } else {
    event = (struct ncclProxyProfileEvent*)(*profileEvent);
    if (state == ncclProxyProfileEnd) *profileEvent = NULL;
    if (state == ncclProxyProfileAppendEnd) event->opCount = args->opCount;
  }
  // Timestamp
  event->timestamp[state%8] = gettime()-profilingStart;
  return ncclSuccess;
}

__hidden ncclResult_t profilingDump() {
  static int dumpDone = 0;
  if (dumpDone)
    return ncclSuccess;
  dumpDone = 1;
  const char* str = getenv("NCCL_PROXY_PROFILE");
  if (!str) {
    printf("Empty env NCCL_PROXY_PROFILE.\n");
    free(profilingEvents);
    return ncclSuccess;
  }
  FILE* f = fopen(str, "w");
  fprintf(f, "[\n");

  for (int i=0; i<profilingIndex; i++) {
    struct ncclProxyProfileEvent* e = profilingEvents+i;
    const int sendrecv = e->peer >= 0;
    const char* typeStr = sendrecv ? (e->type == ncclProxySend ? "Send" : "Recv") :
      profilingEventStr[-(e->peer/8)];


    if (sendrecv) {
      int state = ncclProxyProfileBegin;
      const char** stateStr = e->type == ncclProxySend ? profilingStateSendStr : profilingStateRecvStr;
      fprintf(f, "{\"name\": \"%s-%d-%d\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f, \"args\": { \"opCount\": %ld, \"proxyOpIndex\":%d } },\n",
          typeStr, e->peer, e->step, i, e->channel, e->timestamp[state], e->opCount, e->opIndex);

      while (state<ncclProxyProfileEnd) {
        if (e->timestamp[state]) {
          const char* name = stateStr[state];
          fprintf(f, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f },\n",
              name, i, e->channel, e->timestamp[state]);
          state++;
          while (e->timestamp[state] == 0) state++;
          fprintf(f, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f },\n",
              name, i, e->channel, e->timestamp[state]);
        }
      }

      fprintf(f, "{\"name\": \"%s-%d-%d\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f },\n",
          typeStr, e->peer, e->step, i, e->channel, e->timestamp[state]);
    } else {
      if (e->peer == -ncclProxyProfileAppend) {
      fprintf(f, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": -1, \"tid\": 1, \"ts\": %f, \"args\": { \"added\": %ld } },\n",
          typeStr, i, e->timestamp[0], e->opCount);
      } else {
        fprintf(f, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": -1, \"tid\": 1, \"ts\": %f },\n",
          typeStr, i, e->timestamp[0]);
      }
      fprintf(f, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": -1, \"tid\": 1, \"ts\": %f },\n",
          typeStr, i, e->timestamp[1]);
    }
  }
  fprintf(f, "{} ]\n");
  fclose(f);
  free(profilingEvents);
  return ncclSuccess;
}

// Instantiate plugin symbol
ncclProfiler_t NCCL_PROFILER_SYMBOL = {
  "Dummy",
  profilingRecord,
  profilingDump
};
