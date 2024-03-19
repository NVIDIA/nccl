/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <atomic>
#include <mutex>
#include "profiler.h"

//#define PROFILE_PROXY 1
#ifdef NCCL_PROXY_PROFILER_ENABLED
#define ENABLE_TIMER 1
#include "timer.h"
#include "alloc.h"

#include <linux/bpf.h>
#include <sys/syscall.h>

NCCL_PARAM(ProfilerSamplingEnabled, "PROFILER_SAMPLING_ENABLED", true);
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
  int chunkSize;
  int nsteps;
  int nChannels;
  int collectiveID;
  int nbytes;
  uint8_t protocol;
};

struct ncclCollectiveEvent {
  double timestamp;
  double timestamp_end;
  std::string collectiveName;
  int collectiveCount;
  char type;
};
struct ncclCollectiveEvent* collectiveEvents = NULL;

struct bpf_map_attr {
    __u32 map_type;
    __u32 key_size;
    __u32 value_size;
    __u32 max_entries;
    __u32 map_flags;
  };

std::atomic<bool> enable_profiler_trace{true};
std::atomic<bool> dumpDone{false};
std::mutex profiler_dump_mutex;
struct ncclProxyProfileEvent* profilingEvents = NULL;
int profilingIndex = 0;
double profilingStart = 0;

struct ncclProxyProfileEvent* sampledEvent = NULL;
bool sampledEventAllocated = false;
int samplingProfilingIndex = 0;
int samplingProfilerMapFd = -1;

#define MAX_EVENTS 200000
/* Reduce buffer for collective traces to 500.
Too large buffer for collective may resulting in failure to allocate memory.
Example: If on average there is 50 collective calls in 500ms profiling window. A buffer of 500
can accomodate 10 such windows */
#define MAX_COLLECTIVES 500
#define MAX_SAMPLED_BUFFER_SIZE 100
int collectiveIndex = 0;
volatile double lastEventTimestamp = 0;

void allocateProfilingBuffer() {
  std::unique_lock<std::mutex> lg(profiler_dump_mutex);
  if (profilingEvents != NULL) {
    return;
  }
  ncclCalloc(&profilingEvents, MAX_EVENTS);
  profilingStart = gettime();
}

void allocateCollectiveBuffer() {
  std::unique_lock<std::mutex> lg(profiler_dump_mutex);
  if (collectiveEvents != NULL) {
    return;
  }
  ncclCalloc(&collectiveEvents, MAX_COLLECTIVES);
  profilingStart = gettime();
}

bool shouldSample() {
  double sampling_rate = 1.0 / (double)ncclParamProfilerSamplingWeight();
  double r = (double)rand() / (double)RAND_MAX ;
  return r <= sampling_rate;
  }

  ncclResult_t allocateSamplingProfilerBuffer() {
  struct bpf_map_attr mapAttr = {
      .map_type = BPF_MAP_TYPE_ARRAY,
      .key_size = sizeof(int),
      .value_size = sizeof(struct ncclProxyProfileEvent*),
      .max_entries = MAX_SAMPLED_BUFFER_SIZE,
      .map_flags = 0,
  };
  samplingProfilerMapFd =
      syscall(__NR_bpf, BPF_MAP_CREATE, &mapAttr, sizeof(mapAttr));
  // Check if the map is created successfully
  if (samplingProfilerMapFd < 0) {
    // INFO(NCCL_ALL, "Failed to create sampling profiler buffer");
  }

  profilingStart = gettime();
  return ncclSuccess;
}

struct ncclProxyProfileEvent* ncclProfilingEventCreate(int state) {
  struct ncclProxyProfileEvent* event = NULL;
  // If in trace mode and there is still space in the profiling buffer, we save the event there
  if (enable_profiler_trace.load(std::memory_order_relaxed) && profilingIndex < MAX_EVENTS) {
    event = profilingEvents + profilingIndex++;
  }

  // If in sampling mode, we first check whether this event should be sampled.
  // An event should be sampled if:
  // 1. sampling is enabled
  // 2. the sampling decision is true
  // 3. there is no other event currently being sampled (i.e., the previously sampled event has completed)
  // 4. if the event is a send/recv event (i.e., not idle, sleep)

  if (ncclParamProfilerSamplingEnabled() && shouldSample() && !sampledEvent && state == ncclProxyProfileBegin) {
    // Then, we decide where to store the event being sampled.
    // If the profiler already records the events in the profilingEvents buffer (i.e., is in trace mode), we use that location,
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
  return event;
}

ncclResult_t ncclProfilingEventPopulateMetadata(struct ncclProxyProfileEvent* event, struct ncclProxyArgs* args, int sub, int step, int state) {
  // Proxy operation information
  if(!event || state%8 != 0) {
    return ncclSuccess;
  }
  if (state == ncclProxyProfileBegin) {
      event->opCount = args->opCount;
      event->channel = args->subs[sub].channelId;
      event->peer = args->subs[sub].peer;
      event->type = args->pattern;
      event->step = step;
      event->opIndex = (((uint64_t)args)/sizeof(struct ncclProxyArgs))%256;
      event->chunkSize = args->chunkSize;
      event->nsteps = args->subs[sub].nsteps;
      event->nChannels = args->subs[sub].nChannels;
      event->nbytes = args->subs[sub].nbytes;
      event->protocol = args->protocol;
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
    if (event->peer >= 0) {
      if (lastEventTimestamp < event->timestamp[state % 8]) {
        lastEventTimestamp = event->timestamp[state % 8];
      }
      event->collectiveID = collectiveIndex;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilingSampledEventSave(struct ncclProxyProfileEvent* event) {
  if (samplingProfilerMapFd < 0) {
    // INFO(NCCL_ALL, "No BPF map to dump event\n");
    return ncclSuccess;
  }
  if (!event) {
    // INFO(NCCL_ALL, "No event to save\n");
    return ncclSuccess;
  }
  union bpf_attr attr = {
        .map_fd = (__u32)samplingProfilerMapFd,
        .key = (__u64)(unsigned long)(&samplingProfilingIndex),
        .value = (__u64)(unsigned long)(event),
        .flags = {},
  };
  const int ret = syscall(__NR_bpf, BPF_MAP_UPDATE_ELEM, &attr, sizeof(attr));
  if (ret < 0) {
    // INFO(NCCL_ALL, "Failed to update bpf map with error %d\n", ret);
  }
  samplingProfilingIndex++;
  return ncclSuccess;
}

ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state) {
  // INFO(NCCL_ALL,"NCCL profiling record called: sub %d step %d state %d", sub, step, state);
  if (!enable_profiler_trace.load(std::memory_order_relaxed) && !ncclParamProfilerSamplingEnabled()) {
    return ncclSuccess;
  }

  // allocate buffers if necessary
  if (enable_profiler_trace.load(std::memory_order_relaxed)) {
    if (profilingEvents == NULL) {
      allocateProfilingBuffer();
    }
  }
  if (ncclParamProfilerSamplingEnabled()) {
    if (samplingProfilerMapFd < 0) {
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
    if (state == ncclProxyProfileAppendEnd) event->opCount = args->opCount;
    if (event && state == ncclProxyProfileAppendEnd) event->opCount = args->opCount;
  }

  NCCLCHECK(ncclProfilingEventPopulateTimestamp(event, state));

  if (event && state == ncclProxyProfileEnd && sampledEvent) {
    // INFO(NCCL_ALL, "Saving sampled event");
    ncclProfilingSampledEventSave(sampledEvent);
    if (sampledEventAllocated) free(sampledEvent);
    sampledEvent = NULL;
  }

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCollectiveRecord, const char*, char);
ncclResult_t ncclCollectiveRecord(const char* name, char type) {
  if (!enable_profiler_trace.load(std::memory_order_relaxed)) {
    return ncclSuccess;
  }
  if (collectiveIndex >= MAX_COLLECTIVES) {
    return ncclSuccess;
  }
  // INFO(NCCL_ALL,"NCCL Collective Record name = %s, event = %c, collectiveindex=%d, max_collectives=%d", name, type, collectiveIndex, MAX_COLLECTIVES);
  if (collectiveEvents == NULL) {
    allocateCollectiveBuffer();
  }
  struct ncclCollectiveEvent* e_prev = collectiveEvents + collectiveIndex - 1;
  if (e_prev){
    if (e_prev->type == 'b' && collectiveIndex > 0) {
      e_prev->timestamp_end = lastEventTimestamp;
      e_prev->type = 'e';
    }
  }
  struct ncclCollectiveEvent* event = NULL;
  event = collectiveEvents + collectiveIndex++;
  if(event){
    event->collectiveName = name;
    event->collectiveCount = collectiveIndex;
    event->timestamp = gettime() - profilingStart;
    event->timestamp_end = 0;
    event->type = type;
  }
  // INFO(NCCL_ALL, "The collective name is %s\n", name);
  return ncclSuccess;
}

const char* getCollectiveForEvent(struct ncclProxyProfileEvent* e) {
  if (collectiveEvents == NULL || e->collectiveID < 1) {
    return "N/A";
  }
  struct ncclCollectiveEvent* e_coll = collectiveEvents + e->collectiveID - 1;
  return e_coll->collectiveName.c_str();
}

void profilerCleanup() {
  if (profilingEvents != NULL) {
    free(profilingEvents);
    profilingEvents = NULL;
  }

  if (collectiveEvents != NULL) {
    free(collectiveEvents);
    collectiveEvents = NULL;
  }
  dumpDone = true;
}

NCCL_API(void, ncclProfilingDump, const char*);
void ncclProfilingDump(const char* filename) {
  // INFO(NCCL_ALL,"Dumping proxy profiler trace");
  std::unique_lock<std::mutex> lg(profiler_dump_mutex);
  if (dumpDone) return;

  const char* str = filename;
  if (str == "//") { str = getenv("NCCL_PROXY_PROFILE"); }
  if (!str) {
    profilerCleanup();
    return;
  }

  FILE* f = fopen(str, "w");
  fprintf(f, "[\n");

  for (int i=0; i<profilingIndex; i++) {
    if(profilingEvents == NULL){
      break;
    }
    struct ncclProxyProfileEvent* e = profilingEvents+i;
    const int sendrecv = e->peer >= 0;
    const char* typeStr = sendrecv ? (e->type == ncclPatternSend ? "Send" : "Recv") :
      profilingEventStr[-(e->peer/8)];


    if (sendrecv) {
      int state = ncclProxyProfileBegin;
      const char** stateStr = e->type == ncclPatternSend ? profilingStateSendStr : profilingStateRecvStr;
      const char* collectiveName = getCollectiveForEvent(e);
      fprintf(f, "{\"name\": \"%s-%d-%d\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f, \"args\": { \"opCount\": %ld, \"proxyOpIndex\":%d }, \"chunkSize\": %d, \"totalSteps\": %d, \"totalChannels\": %d, \"collectiveName\": \"%s-%d\" , \"totalbytes\": %d ,  \"proto\": %d },\n",
          typeStr, e->peer, e->step, i, e->channel, e->timestamp[state], e->opCount, e->opIndex, e->chunkSize,  e->nsteps, e->nChannels, collectiveName, e->collectiveID, e->nbytes, e->protocol);

      while (state<ncclProxyProfileEnd) {
        if (e->timestamp[state]) {
          const char* name = stateStr[state];
          fprintf(f, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f, \"chunkSize\": %d, \"totalSteps\": %d, \"collectiveName\": \"%s-%d\", \"totalbytes\": %d ,  \"proto\": %d},\n",
              name, i, e->channel, e->timestamp[state], e->chunkSize,  e->nsteps, collectiveName, e->collectiveID, e->nbytes, e->protocol);
          state++;
          while (e->timestamp[state] == 0) state++;
          fprintf(f, "{\"name\": \"%s\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f, \"chunkSize\": %d, \"totalSteps\": %d, \"collectiveName\": \"%s-%d\", \"totalbytes\": %d ,  \"proto\": %d},\n",
              name, i, e->channel, e->timestamp[state], e->chunkSize,  e->nsteps, collectiveName, e->collectiveID, e->nbytes, e->protocol);
        }
      }

      fprintf(f, "{\"name\": \"%s-%d-%d\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %d, \"pid\": %d, \"tid\": 1, \"ts\": %f , \"chunkSize\": %d, \"totalSteps\": %d , \"collectiveName\": \"%s-%d\", \"totalbytes\": %d ,  \"proto\": %d},\n",
          typeStr, e->peer, e->step, i, e->channel, e->timestamp[state], e->chunkSize,  e->nsteps, collectiveName, e->collectiveID, e->nbytes, e->protocol);
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
  for (int i = 0; i < collectiveIndex; i++) {
    struct ncclCollectiveEvent* e = collectiveEvents + i;
    if (e) {
      fprintf(
          f,
          "{\"name\": \"%s-%d\", \"cat\": \"COL\", \"id\": %d, \"ph\": \"b\", \"pid\": -1, \"tid\": 1, \"ts\": %f },\n",
          e->collectiveName.c_str(),
          e->collectiveCount,
          e->collectiveCount,
          e->timestamp);
      if (i == collectiveIndex - 1) {
        fprintf(
            f,
            "{\"name\": \"%s-%d\", \"cat\": \"COL\", \"id\": %d, \"ph\": \"e\", \"pid\": -1, \"tid\": 1, \"ts\": %f },\n",
            e->collectiveName.c_str(),
            e->collectiveCount,
            e->collectiveCount,
            lastEventTimestamp);
      } else {
        fprintf(
            f,
            "{\"name\": \"%s-%d\", \"cat\": \"COL\", \"id\": %d, \"ph\": \"e\", \"pid\": -1, \"tid\": 1, \"ts\": %f },\n",
            e->collectiveName.c_str(),
            e->collectiveCount,
            e->collectiveCount,
            e->timestamp_end);
      }
      fflush(stdout);
    }
  }
  fprintf(f, "{} ]\n");
  fclose(f);
  profilerCleanup();
}

// returns true if it was previously disabled
NCCL_API(ncclResult_t, ncclProfilerEnable);
ncclResult_t ncclProfilerEnable() {
  // INFO(NCCL_ALL,"Enabling proxy profiler\n");
  // only if it was previously disabled
  if (!enable_profiler_trace.exchange(true)) {
    std::unique_lock<std::mutex> lg(profiler_dump_mutex);
    profilingIndex = 0;
    collectiveIndex = 0;
    dumpDone = false;
    return ncclSuccess;
  }
  return ncclInternalError;
};

// returns true if it was previously enabled
NCCL_API(ncclResult_t, ncclProfilerDisable);
ncclResult_t ncclProfilerDisable() {
  // INFO(NCCL_ALL,"Disabling proxy profiler\n");
  if (enable_profiler_trace.exchange(false)) {
    return ncclSuccess;
  }
  return ncclInternalError;
};

#else
ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state) { return ncclSuccess; }
void ncclProfilingDump(const char* filename) {}
#endif
