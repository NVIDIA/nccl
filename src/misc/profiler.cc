/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "profiler.h"

//#define PROFILE_PROXY 1
#ifdef PROFILE_PROXY
#include "timer.h"
#include "alloc.h"
#include <linux/bpf.h>
#include <sys/syscall.h>

NCCL_PARAM(ProfilerSamplingEnabled, "PROFILER_SAMPLING_ENABLED", false);
NCCL_PARAM(ProfilerSamplingWeight, "PROFILER_SAMPLING_WEIGHT", 100);

static const char* profilingStateSendStr[] = { "BufferWait", "GPUWait", "SendWait", "", "End" };
static const char* profilingStateRecvStr[] = { "BufferWait", "RecvWait", "FlushWait", "GPUWait", "End" };
static const char* profilingEventStr[] = { "SendRecv", "Sleep", "Idle", "Append" };
struct ncclProxyProfileEvent {
  double timestamp[6];
  uint64_t opCount;
  int peer;
  int step;
  uint16_t channel;
  uint8_t type; // send / recv
  uint8_t opIndex;
};
struct bpf_map_attr {
    __u32 map_type;
    __u32 key_size;
    __u32 value_size;
    __u32 max_entries;
    __u32 map_flags;
  };

struct ncclProxyProfileEvent* profilingEvents = NULL;
struct ncclProxyProfileEvent* sampledEvent = NULL;
bool sampledEventAllocated = false;
int profilingIndex = 0;
int samplingProfilingIndex = 0;
double profilingStart = 0;
int samplingProfilerMapFd = -1;
#define MAX_EVENTS 200000
#define MAX_SAMPLED_BUFFER_SIZE 100000

bool shouldSample() {
  double sampling_rate = 1.0 / (double)ncclParamProfilerSamplingWeight();
  double r = (double)rand() / (double)RAND_MAX ;
  return r <= sampling_rate;
}

void allocateProfilingBuffer() {
  if (profilingEvents != NULL) {
    return;
  }
  ncclCalloc(&profilingEvents, MAX_EVENTS);
  profilingStart = gettime();
}

ncclResult_t allocateSamplingProfilerBuffer() {
  struct bpf_map_attr mapAttr = {
      .map_type = BPF_MAP_TYPE_ARRAY,
      .key_size = sizeof(int),
      .value_size = sizeof(struct ncclProxyProfileEvent),
      .max_entries = MAX_SAMPLED_BUFFER_SIZE,
      .map_flags = 0,
  };
  samplingProfilerMapFd =
      syscall(__NR_bpf, BPF_MAP_CREATE, &mapAttr, sizeof(mapAttr));
  // Check if the map is created successfully
  if (samplingProfilerMapFd < 0) {
    INFO(NCCL_ALL, "Failed to create sampling profiler buffer");
  }
  profilingStart = gettime();
  return ncclSuccess;
}

struct ncclProxyProfileEvent* ncclProfilingEventCreate(int state) {
  struct ncclProxyProfileEvent* event = NULL;
  // If there is still space in the profiling buffer, we save the event there
  if (profilingIndex < MAX_EVENTS) {
    event = profilingEvents + profilingIndex++;
  }
  // Otherwise, we save the event only if:
  // 1. sampling is enabled
  // 2. the sampling decision is true
  // 3. there is no other event currently being sampled (i.e., the previously sampled event has completed)
  // 4. if the event is a send/recv event (i.e., not idle, sleep)
  else {
    if (ncclParamProfilerSamplingEnabled() && shouldSample() && !sampledEvent && state == ncclProxyProfileBegin) {
    // Then, we decide where to store the event being sampled.
    // If the profiler already records the events in the profilingEvents buffer, we use that location, 
    // otherwise we allocate the sampledEvent variable
    if (event) {
      sampledEvent = event;
    }
    else {
      ncclCalloc(&sampledEvent, 1);
      sampledEventAllocated = true;
      event = sampledEvent;
    }
  }
  }
  
  return event;
}

ncclResult_t ncclProfilingEventPopulateMetadata(struct ncclProxyProfileEvent* event, struct ncclProxyArgs* args, int sub, int step, int state) {
  
  // Proxy operation information
  if(!event || state%8 != 0) {
    return ncclSuccess;
  }
  if (state == ncclProxyProfileBegin) {
    // Proxy operation information
    event->opCount = args->opCount;
    event->channel = args->subs[sub].channelId;
    event->peer = args->subs[sub].peer;
    event->type = args->pattern;
    event->step = step;
    event->opIndex = (((uint64_t)args)/sizeof(struct ncclProxyArgs))%256;
  }
  else {
    event->peer = -state;
  }

  return ncclSuccess;
}

ncclResult_t ncclProfilingEventPopulateTimestamp(struct ncclProxyProfileEvent* event, int state) {
  // Timestamp
  if (event) {
    event->timestamp[state % 8] = gettime() - profilingStart;
  }

  return ncclSuccess;
}

ncclResult_t ncclProfilingSampledEventSave(struct ncclProxyProfileEvent* event) {
  if (samplingProfilerMapFd < 0) {
    INFO(NCCL_ALL, "No BPF map to dump event\n");
    return ncclSuccess;
  }
  if (!event) {
    INFO(NCCL_ALL, "No event to save\n");
    return ncclSuccess;
  }
  union bpf_attr attr = {
        .map_fd = (__u32)samplingProfilerMapFd,
        .key = (__u64)(unsigned long)(&samplingProfilingIndex),
        .value = (__u64)(unsigned long)(event),
        .flags = 0,
  };
  const int ret = syscall(__NR_bpf, BPF_MAP_UPDATE_ELEM, &attr, sizeof(attr));
  if (ret < 0) {
    INFO(NCCL_ALL, "Failed to update bpf with error %d %d\n", ret, errno);
  }
  samplingProfilingIndex++;
  return ncclSuccess;
}

ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state) {
  // INFO(NCCL_ALL,"NCCL profiling record called: sub %d step %d state %d", sub, step, state);
  if (profilingEvents == NULL) {
    allocateProfilingBuffer();
  }
  else {
    if (ncclParamProfilerSamplingEnabled() && samplingProfilerMapFd < 0) {
      allocateSamplingProfilerBuffer();
    }
  }

  struct ncclProxyProfileEvent* event = NULL;
  // if this is a new event
  if (state % 8 == 0) {
    event = ncclProfilingEventCreate(state);
    if (!event) { 
      return ncclSuccess;
    }
    args->subs[sub].profilingEvents[step%NCCL_STEPS] = event;
    NCCLCHECK(ncclProfilingEventPopulateMetadata(event, args, sub, step, state));
  }
  else {
    event = (struct ncclProxyProfileEvent*)args->subs[sub].profilingEvents[step%NCCL_STEPS];
    if (state == ncclProxyProfileEnd) args->subs[sub].profilingEvents[step%NCCL_STEPS] = NULL;
    if (event && state == ncclProxyProfileAppendEnd) event->opCount = args->opCount;
  }

  NCCLCHECK(ncclProfilingEventPopulateTimestamp(event, state));

  // only when we are sampling: if we reach the end of the event, we save it and free the memory used to store the event
  if (event && ncclParamProfilerSamplingEnabled() && state == ncclProxyProfileEnd && sampledEvent) {
    ncclProfilingSampledEventSave(sampledEvent);
    if (sampledEventAllocated) free(sampledEvent);
    sampledEventAllocated = false;
    sampledEvent = NULL;
  }

  return ncclSuccess;
}

void ncclProfilingDump() {
  static int dumpDone = 0;
  if (dumpDone) return;
  dumpDone = 1;
  const char* str = ncclGetEnv("NCCL_PROXY_PROFILE");
  if (!str) { free(profilingEvents); return; }
  FILE* f = fopen(str, "w");
  fprintf(f, "[\n");

  for (int i=0; i<profilingIndex; i++) {
    struct ncclProxyProfileEvent* e = profilingEvents+i;
    const int sendrecv = e->peer >= 0;
    const char* typeStr = sendrecv ? (e->type == ncclPatternSend ? "Send" : "Recv") :
      profilingEventStr[-(e->peer/8)];


    if (sendrecv) {
      int state = ncclProxyProfileBegin;
      const char** stateStr = e->type == ncclPatternSend ? profilingStateSendStr : profilingStateRecvStr;
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
}
#else
ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state) { return ncclSuccess; }
void ncclProfilingDump() {}
#endif
