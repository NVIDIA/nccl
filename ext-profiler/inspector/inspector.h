#ifndef INSPECTOR_INSPECTOR_H_
#define INSPECTOR_INSPECTOR_H_

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "json.h"
#include "common.h"
#include "version.h"

#define MAX_CHANNELS                     64

#define INS_CHK(call)                                                   \
  do {                                                                  \
    inspectorResult_t res = call;                                       \
    if (inspectorSuccess != res) {                                      \
      INFO_INSPECTOR("%s:%d -> error %d: %s", __FILE__, __LINE__, res,  \
                     inspectorErrorString(res));                        \
      return res;                                                       \
    }                                                                   \
  } while (0);

#define INS_CHK_GOTO(call, res, label)                                  \
  do {                                                                  \
    res = call;                                                         \
    if (inspectorSuccess != res) {                                      \
      INFO_INSPECTOR("%s:%d -> error %d: %s", __FILE__, __LINE__, res,  \
                     inspectorErrorString(res));                        \
      goto label;                                                       \
    }                                                                   \
  } while (0);

// Lock convenience macros
#define INSPECTOR_LOCK_RD_FLAG(lockRef, lockFlag, debug)        \
  do {                                                          \
    if (!lockFlag) {                                            \
      INS_CHK(inspectorLockRd(lockRef));                        \
    }                                                           \
    lockFlag = true;                                            \
  } while (0);

#define INSPECTOR_LOCK_WR_FLAG(lockRef, lockFlag, debug)        \
  do {                                                          \
    if (!lockFlag) {                                            \
      INS_CHK(inspectorLockWr(lockRef));                        \
    }                                                           \
    lockFlag = true;                                            \
  } while (0);

#define INSPECTOR_UNLOCK_RW_LOCK_FLAG(lockRef, lockFlag, debug) \
  do {                                                          \
    if (lockFlag) {                                             \
      INS_CHK(inspectorUnlockRWLock(lockRef));                  \
    }                                                           \
    lockFlag = false;                                           \
  } while (0);


typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncSendRecv = 5,
  ncclFuncSend = 6,
  ncclFuncRecv = 7,
  ncclNumFuncs = 8
} ncclFunc_t;

typedef enum {
  inspectorSuccess = 0,
  inspectorUninitializedError,
  inspectorMemoryError,
  inspectorFileOpenError,
  inspectorDisabledError,
  inspectorLockError,
  inspectorPthreadError,
  inspectorJsonError,
  inspectorCudaError,
  inspectorBadHash,
  inspectorDeleteUnknownCommError,
  inspectorAddDuplicateCommError,
  inspectorNop,
  inspectorNullTally,
  inspectorGlobalInitError,
  inspectorReturn,
} inspectorResult_t;

typedef enum {
  inspectorTimingSourceKernelGpu = 0,
  inspectorTimingSourceKernelCpu = 1,
  inspectorTimingSourceCollectiveCpu = 2,
} inspectorTimingSource_t;

struct inspectorEventTraceInfo {
  uint64_t ts;
  uint64_t sn;
};

typedef enum {
  NCCL_INSP_EVT_TRK_COLL_START = 0,
  NCCL_INSP_EVT_TRK_COLL_STOP = 1,
  NCCL_INSP_EVT_TRK_COLL_NEVT = 2,
} inspectorEventTrkColl_t;

typedef enum {
  NCCL_INSP_EVT_TRK_KERNEL_START = 0,
  NCCL_INSP_EVT_TRK_KERNEL_STOP = 1,
  NCCL_INSP_EVT_TRK_KERNEL_RECORD = 2,
  NCCL_INSP_EVT_TRK_KERNEL_NEVT = 3,
} inspectorEventTrkKernel_t;

struct inspectorEventTrkKernelInfo {
  struct inspectorEventTraceInfo evntTrace[NCCL_INSP_EVT_TRK_KERNEL_NEVT];
};

struct inspectorEventTrkCollInfo {
  int sn;
  uint32_t nChannels;
  struct inspectorEventTraceInfo evntTrace[NCCL_INSP_EVT_TRK_COLL_NEVT];
  struct inspectorEventTrkKernelInfo kernelCh[MAX_CHANNELS];
};

struct inspectorCompletedCollInfo {
  ncclFunc_t func;
  uint64_t sn;
  size_t msgSizeBytes;
  uint64_t execTimeUsecs;
  inspectorTimingSource_t timingSource;
  double algoBwGbs;
  double busBwGbs;
  // Event trace information
  struct inspectorEventTrkCollInfo collEvtTrk;
};

enum {
  NCCL_COMM_HASH_LENGTH = 17
};

struct inspectorCommInfo {
  struct inspectorCommInfo* next;

  const char* commName;
  uint64_t commHash;
  char commHashStr[NCCL_COMM_HASH_LENGTH];
  int rank;
  int nranks;
  int nnodes;
  int cudaDeviceId;     // CUDA device ID for this communicator
  char deviceUuidStr[37]; // Pre-computed device UUID string for filename generation
  char cachedStaticLabels[256]; // Cached static parts of Prometheus labels (hostname, job, comm, rank, etc.)

  bool dump;
  struct inspectorCompletedCollInfo completedCollInfo;
  pthread_rwlock_t guard;
};

// Structure to track flush times and file handles per device UUID
struct deviceFlushInfo {
  char deviceUuidStr[37];
  uint64_t lastFlushTime;
  FILE* fileHandle;           // Open file handle for this device
  bool needsCreation;         // Whether file needs to be created/flushed
  char filename[1024];        // Store filename for cleanup
};

struct inspectorDumpThread {
  bool run{false};
  jsonFileOutput* jfo;
  char* outputRoot;
  uint64_t sampleIntervalUsecs;
  std::vector<deviceFlushInfo> deviceFlushEntries;
  pthread_t pthread;
  pthread_rwlock_t guard;

  // Constructor and destructor implemented in inspector.cc where dependencies are available
  inspectorDumpThread(const char* _outputRoot, uint64_t _sampleIntervalUsecs);
  ~inspectorDumpThread();

  /*
   * Gets or creates a file handle for a device UUID, handling flushing as needed.
   */
  FILE* getOrCreateFileHandle(const char* deviceUuidStr,
                              const char* filename,
                              uint64_t currentTime);

  void startThread();
  void stopThread();
  inspectorResult_t inspectorStateDump(const char* output_root);
  inspectorResult_t inspectorStateDumpJSON(const char* output_root);
  inspectorResult_t inspectorStateDumpProm(const char* output_root);
  static void* dumpMain(void* arg);
};

struct inspectorKernelChInfo {
  uint64_t type;
  int refCount; /*unused*/
  struct inspectorCollInfo *collInfo;
  uint8_t channelId;
  uint64_t tsStartUsec;
  uint64_t tsCompletedUsec;
  uint64_t startGpuClk;
  uint64_t stopGpuClk;
};

struct inspectorCollInfo {
  uint64_t type;
  int refCount;
  struct inspectorCommInfo *commInfo;
  const char* func;
  uint64_t sn;
  size_t msgSizeBytes;
  uint64_t tsStartUsec;
  uint64_t tsCompletedUsec;
  uint32_t nChannels;
  uint32_t nKernelChStarted;
  uint32_t nKernelChCompleted;
  pthread_rwlock_t guard;
  struct inspectorKernelChInfo kernelCh[MAX_CHANNELS];
  struct inspectorEventTrkCollInfo collEvtTrk;
};

struct inspectorCommInfoList {
  struct inspectorCommInfo* comms;
  uint32_t ncomms;
  pthread_rwlock_t guard;
};

struct inspectorState {
  struct inspectorCommInfoList liveComms;
  struct inspectorCommInfoList deletedComms;
};



extern ncclDebugLogger_t logFn;
#define VERSION(...) logFn(NCCL_LOG_VERSION, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) logFn(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

// Use NCCL_PROFILE for inspector messages so they can be filtered with NCCL_DEBUG_SUBSYS=PROFILE
#define INFO_INSPECTOR(...) logFn(NCCL_LOG_INFO, NCCL_PROFILE, __func__, __LINE__, __VA_ARGS__)
#define TRACE_INSPECTOR(...) logFn(NCCL_LOG_TRACE, NCCL_PROFILE, __func__, __LINE__, __VA_ARGS__)

inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    return 1;
  case ncclFloat16:
  case ncclBfloat16:
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}

const char* inspectorErrorString(inspectorResult_t result);

inspectorResult_t inspectorLockInit(pthread_rwlock_t* lockRef);
inspectorResult_t inspectorLockDestroy(pthread_rwlock_t* lockRef);
inspectorResult_t inspectorLockRd(pthread_rwlock_t* lockRef);
inspectorResult_t inspectorLockWr(pthread_rwlock_t* lockRef);
inspectorResult_t inspectorUnlockRWLock(pthread_rwlock_t* lockRef);
inspectorResult_t inspectorGlobalInit(int rank);
inspectorResult_t inspectorGlobalFinalize();
uint64_t inspectorGetTime();
inspectorResult_t inspectorGetTimeUTC(char* buffer, size_t bufferSize);
inspectorResult_t inspectorAddComm(struct inspectorCommInfo **commInfo,
                                   const char* commName, uint64_t commHash,
                                   int nNodes, int nranks, int rank);
inspectorResult_t inspectorDelComm(struct inspectorCommInfo *commInfo);

void inspectorUpdateCollPerf(struct inspectorCompletedCollInfo *completedColl,
                             struct inspectorCollInfo *collInfo);
ncclDataType_t inspectorStringToDatatype(const char* str);

void inspectorComputeCollBw(struct inspectorCommInfo *commInfo,
                            struct inspectorCompletedCollInfo *completedColl,
                            ncclFunc_t collType);

// Utility functions exposed for Prometheus module
const char* inspectorTimingSourceToString(inspectorTimingSource_t timingSource);
inspectorResult_t inspectorCommInfoListFinalize(struct inspectorCommInfoList* commList);
const char* ncclFuncToString(ncclFunc_t fn);

// Global state
extern struct inspectorState g_state;

#endif  // INSPECTOR_INSPECTOR_H_
