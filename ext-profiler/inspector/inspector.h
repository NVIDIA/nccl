#pragma once

#include <pthread.h>

#include "json.h"
#include "common.h"
#include "version.h"

#define MAX_CHANNELS                     64

#define INS_CHK_GOTO(call, res, label)                                  \
  do {                                                                  \
    res = call;                                                         \
    if (inspectorSuccess != res) {                                      \
      INFO(NCCL_INSPECTOR, "%s:%d -> error %d: %s", __FILE__, __LINE__, res, \
           inspectorErrorString(res));                                  \
      goto label;                                                       \
    }                                                                   \
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

  bool dump;
  struct inspectorCompletedCollInfo completedCollInfo;
  pthread_rwlock_t guard;
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



extern ncclDebugLogger_t logFn;
#define VERSION(...) logFn(NCCL_LOG_VERSION, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) logFn(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#define WARN(...) logFn(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)

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
