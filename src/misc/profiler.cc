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

std::atomic<bool> enable_profiler{false};
std::atomic<bool> dumpDone{false};
std::mutex profiler_dump_mutex;

struct ncclProxyProfileEvent* profilingEvents = NULL;
int profilingIndex = 0;
double profilingStart = 0;
#define MAX_EVENTS 200000
/* Reduce buffer for collective traces to 500.
Too large buffer for collective may resulting in failure to allocate memory.
Example: If on average there is 50 collective calls in 500ms profiling window. A buffer of 500
can accomodate 10 such windows */
#define MAX_COLLECTIVES 500
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

ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state) {
  if (!enable_profiler.load(std::memory_order_relaxed)) {
    return ncclSuccess;
  }
  if (profilingEvents == NULL) {
    allocateProfilingBuffer();
  }
  struct ncclProxyProfileEvent* event = NULL;
  if (state%8 == 0) {
    if (profilingIndex == MAX_EVENTS) return ncclSuccess;
    args->subs[sub].profilingEvents[step%NCCL_STEPS] = event = profilingEvents+profilingIndex++;
    if (state == ncclProxyProfileBegin) {
      // Proxy operation information
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
    } else event->peer = -state;
  } else {
    event = (struct ncclProxyProfileEvent*)args->subs[sub].profilingEvents[step%NCCL_STEPS];
    if (state == ncclProxyProfileEnd) args->subs[sub].profilingEvents[step%NCCL_STEPS] = NULL;
    if (state == ncclProxyProfileAppendEnd) event->opCount = args->opCount;
    if (event && state == ncclProxyProfileAppendEnd) event->opCount = args->opCount;
  }
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

NCCL_API(ncclResult_t, ncclCollectiveRecord, const char*, char);
ncclResult_t ncclCollectiveRecord(const char* name, char type) {
  if (!enable_profiler.load(std::memory_order_relaxed)) {
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
  if (!enable_profiler.exchange(true)) {
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
  if (enable_profiler.exchange(false)) {
    return ncclSuccess;
  }
  return ncclInternalError;
};

#else
ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state) { return ncclSuccess; }
void ncclProfilingDump(const char* filename) {}
#endif
