/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "profiler.h"

#define PROFILE_PROXY 1
#ifdef PROFILE_PROXY
#define ENABLE_TIMER  1
#include "timer.h"
#include "alloc.h"
#include <libgen.h>

static const char* profilingStateSendStr[] = { "BufferWait", "GPUWait", "SendWait", "", "End" };
static const char* profilingStateRecvStr[] = { "BufferWait", "RecvWait", "FlushWait", "GPUWait", "End" };
static const char* profilingEventStr[] = { "SendRecv", "Sleep", "Idle", "Append" };
struct ncclProxyProfileEvent {
  double timestamp[6];
  uint64_t opCount;
  int peer;
  int step;
  uint16_t channel;
  uint8_t type; // send / recv / ring / trees
  uint8_t opIndex;
  uint8_t direction; // send / recv
  size_t size; // size of network transfer
};

struct ncclProxyProfileEvent* profilingEvents = NULL;
int profilingIndex = 0;
double profilingStart = 0;

ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state) {
  if (profilingEvents == NULL) {
    NCCLCHECK(ncclCalloc(&profilingEvents, MAX_EVENTS));
    profilingStart = gettime();
  }
  struct ncclProxyProfileEvent* event = NULL;
  if (state%8 == 0) {
    if (profilingIndex == MAX_EVENTS) return ncclSuccess;
    args->subs[sub].profilingEvents[step] = event = profilingEvents+profilingIndex++;
    if (state == ncclProxyProfileBegin) {
      // Proxy operation information
      event->opCount = args->opCount;
      event->channel = args->subs[sub].channelId;
      event->peer = args->subs[sub].peer;
      event->type = args->pattern;
      if ((event->type == ncclPatternRing) || (event->type == ncclPatternTreeUpDown)) {
        event->direction = args->direction;
        event->size = (event->direction == ncclDirectionSend) ? args->sendSize : args->recvSize;
      }
      event->step = step;
      event->opIndex = (((uint64_t)args)/sizeof(struct ncclProxyArgs))%256;
    } else event->peer = -state;
  } else {
    event = (struct ncclProxyProfileEvent*)args->subs[sub].profilingEvents[step];
    if (((event->type == ncclPatternRing) || (event->type == ncclPatternTreeUpDown)) &&
        (event->direction == ncclDirectionSend) &&
        (state == ncclProxyProfileSendWait)) {
      /* Update size for sends as we have the update size to transfer from GPUs later in the process */
      event->size = args->sendSize;
    }
    if (state == ncclProxyProfileEnd) args->subs[sub].profilingEvents[step] = NULL;
    if (state == ncclProxyProfileAppendEnd) event->opCount = args->opCount;
  }
  // Timestamp
  event->timestamp[state%8] = gettime()-profilingStart;
  return ncclSuccess;
}

void ncclProfilingDump(int rank) {
  static int dumpDone = 0;
  if (dumpDone) return;
  dumpDone = 1;
  char* str = getenv("NCCL_PROXY_PROFILE");
  if (!str) { free(profilingEvents); return; }
  int str_size = strlen(str);

  /* Extract filename and dirname */
  const char *filename = basename(str);
  const char *dname = dirname(str);

  /* Create files for each rank to avoid data corruption when multiple processes write to the same file */
  int fname_size = str_size + sizeof(int) + 10;
  char* fname = (char *)malloc(fname_size*sizeof(char));

  /* FIXME: Filename can have extension which needs to be removed too */
  int rc = snprintf(fname, fname_size, "%s/%s_%d.json", dname, filename, rank);
  if (rc < 0) {
    printf("Error occured when forming the filename from %s and %d\n", str, rank);
  }

  FILE* f = fopen(fname, "w");
  fprintf(f, "[\n");

  for (int i=0; i<profilingIndex; i++) {
    struct ncclProxyProfileEvent* e = profilingEvents+i;
    const int sendrecv = e->peer >= 0;
    const char* typeStr = sendrecv ? (e->type == ncclPatternSend ? "Send" : "Recv") :
      profilingEventStr[-(e->peer/8)];
    if (((e->type == ncclPatternRing) || (e->type == ncclPatternTreeUpDown)) && sendrecv) {
      typeStr = (e->direction == ncclDirectionSend) ? "Send" : "Recv";
    }


    if (sendrecv) {
      int state = ncclProxyProfileBegin;
      const char** stateStr = e->type == ncclPatternSend ? profilingStateSendStr : profilingStateRecvStr;

      if (((e->type == ncclPatternRing) || (e->type == ncclPatternTreeUpDown)) && sendrecv) {
        stateStr = (e->direction == ncclDirectionSend) ? profilingStateSendStr : profilingStateRecvStr;
      }

      fprintf(f, "{\"name\": \"%s-%d-%d\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f, \"args\": { \"opCount\": %ld, \"proxyOpIndex\":%d , \"xferSize\": %zu} },\n",
          typeStr, e->peer, e->step, i, e->channel, e->timestamp[state], e->opCount, e->opIndex, e->size);

      while (state<ncclProxyProfileEnd) {
        if (e->timestamp[state]) {
          const char* name = stateStr[state];
          fprintf(f, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f, \"args\": { \"opCount\": %ld, \"proxyOpIndex\":%d , \"xferSize\": %zu} },\n",
              name, i, e->channel, e->timestamp[state], e->opCount, e->opIndex, e->size);
          state++;
          while (e->timestamp[state] == 0) state++;
          fprintf(f, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f, \"args\": { \"opCount\": %ld, \"proxyOpIndex\":%d , \"xferSize\": %zu} },\n",
              name, i, e->channel, e->timestamp[state], e->opCount, e->opIndex, e->size);
        }
      }

      fprintf(f, "{\"name\": \"%s-%d-%d\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f, \"args\": { \"opCount\": %ld, \"proxyOpIndex\":%d , \"xferSize\": %zu} },\n",
          typeStr, e->peer, e->step, i, e->channel, e->timestamp[state], e->opCount, e->opIndex, e->size);
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
}
#else
ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state) { return ncclSuccess; }
void ncclProfilingDump(int rank) {}
#endif
