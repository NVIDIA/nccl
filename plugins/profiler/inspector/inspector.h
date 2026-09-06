/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef INSPECTOR_INSPECTOR_H_
#define INSPECTOR_INSPECTOR_H_

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <vector>

#include "json.h"
#include "common.h"
#include "version.h"

#define MAX_CHANNELS                     64

// Bump when ncclProfiler_t alias changes to a new interface version.
#define NCCL_PROFILER_INTERFACE_VERSION 5

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
  ncclFuncAll2All = 8,
  ncclFuncAllGatherV = 9,
  ncclNumFuncs = 10
} ncclFunc_t;

enum inspectorResult_t : int {
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
};

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
  NCCL_INSP_EVT_TRK_OP_START = 0,
  NCCL_INSP_EVT_TRK_OP_STOP  = 1,
  NCCL_INSP_EVT_TRK_OP_NEVT  = 2,
} inspectorEventTrkOp_t;

typedef enum {
  NCCL_INSP_EVT_TRK_KERNEL_START = 0,
  NCCL_INSP_EVT_TRK_KERNEL_STOP = 1,
  NCCL_INSP_EVT_TRK_KERNEL_RECORD = 2,
  NCCL_INSP_EVT_TRK_KERNEL_NEVT = 3,
} inspectorEventTrkKernel_t;

typedef enum {
  NCCL_INSP_EVT_TRK_PROXY_OP_START = 0,
  NCCL_INSP_EVT_TRK_PROXY_OP_IN_PROGRESS,
  NCCL_INSP_EVT_TRK_PROXY_OP_STOP,
  NCCL_INSP_EVT_TRK_PROXY_OP_NEVT,
} inspectorEventTrkProxyOp_t;

typedef enum {
  NCCL_INSP_EVT_TRK_PROXY_STEP_START = 0,
  NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_GPU_WAIT,
  NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_PEER_WAIT,
  NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_WAIT,
  NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_WAIT,
  NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_FLUSH_WAIT,
  NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_GPU_WAIT,
  NCCL_INSP_EVT_TRK_PROXY_STEP_STOP,
  NCCL_INSP_EVT_TRK_PROXY_STEP_NEVT,
} inspectorEventTrkProxyStep_t;

typedef enum {
  NCCL_INSP_PROXY_RECORD_OP = 0,
  NCCL_INSP_PROXY_RECORD_STEP,
} inspectorProxyRecordType_t;

struct inspectorEventTrkKernelInfo {
  struct inspectorEventTraceInfo evntTrace[NCCL_INSP_EVT_TRK_KERNEL_NEVT];
};

// Unified event tracking for both collective and P2P operations.
struct inspectorEventTrkOpInfo {
  int sn;
  uint32_t nChannels;
  struct inspectorEventTraceInfo evntTrace[NCCL_INSP_EVT_TRK_OP_NEVT];
  struct inspectorEventTrkKernelInfo kernelCh[MAX_CHANNELS];
};

// Unified record stored in the completed ring buffer for both collective and
// P2P operations.  The isP2p discriminator selects which op-type-specific
// fields (algo/proto vs peer) are valid.
struct inspectorCompletedOpInfo {
  bool isP2p;
  ncclFunc_t func;
  uint64_t sn;
  uint64_t timestampUsec;
  size_t msgSizeBytes;
  uint64_t execTimeUsecs;
  inspectorTimingSource_t timingSource;
  double algoBwGbs;
  double busBwGbs;
  const char* algo;   // coll only (nullptr for P2P)
  const char* proto;  // coll only (nullptr for P2P)
  int peer;           // P2P only (unused for coll)
  struct inspectorEventTrkOpInfo evtTrk;
};

// Pointer-free metadata shared by active and completed Proxy records.
struct inspectorProxyOpMetadata {
  uint64_t parentType;
  uint64_t parentSn;
  uint64_t proxyOpSn;
  int rank;
  int peer;
  uint8_t channelId;
  bool isSend;
};

// Self-contained record copied into the completed proxy ring. It must not
// retain pointers to active events because those objects return to their pools.
struct inspectorCompletedProxyRecord {
  inspectorProxyRecordType_t recordType;
  uint64_t recordSn;
  struct inspectorProxyOpMetadata metadata;
  union {
    struct {
      int nSteps;
      int chunkSize;
      uint32_t nStepsStarted;
      uint32_t nStepsCompleted;
      uint32_t nStepsDropped;
      size_t transSizeBytes;
      struct inspectorEventTraceInfo evntTrace[NCCL_INSP_EVT_TRK_PROXY_OP_NEVT];
    } proxyOp;
    struct {
      uint64_t proxyStepSn;
      int step;
      size_t transSizeBytes;
      struct inspectorEventTraceInfo evntTrace[NCCL_INSP_EVT_TRK_PROXY_STEP_NEVT];
    } proxyStep;
  };
};

#include "inspector_ring.h"

enum {
  NCCL_COMM_HASH_LENGTH = 17,
  NCCL_COMM_NAME_MAX = 256
};

struct inspectorCommInfo {
  struct inspectorCommInfo* next;

  const char* commName;
  char commNameStr[NCCL_COMM_NAME_MAX];
  uint64_t commHash;
  char commHashStr[NCCL_COMM_HASH_LENGTH];
  int rank;
  int nranks;
  int nnodes;
  int cudaDeviceId;     // CUDA device ID for this communicator
  char deviceUuidStr[37]; // Pre-computed device UUID string for filename generation

  bool dump_coll;
  bool dump_p2p;
  bool dump_proxy;
  struct inspectorCompletedRing completedCollRing;
  struct inspectorCompletedRing completedP2pRing;
  struct inspectorCompletedRing completedProxyRing;
  uint64_t p2pSeqNum;
  uint64_t nextProxyOpSn;
  uint64_t nextProxyRecordSn;
  uint64_t proxyOpsDropped;
  uint64_t proxyOpsDroppedReported;
  // Last observed process-wide PXN skip count; not a per-comm loss counter.
  uint64_t proxyPxnSkippedReported;
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
  bool threadStarted{false};
  jsonFileOutput* jfo;
  char* outputRoot;
  int64_t sampleIntervalUsecs;
  std::vector<deviceFlushInfo> deviceFlushEntries;
  pthread_t pthread;
  pthread_rwlock_t guard;

  // Constructor and destructor implemented in inspector.cc where dependencies are available
  inspectorDumpThread(const char* _outputRoot, int64_t _sampleIntervalUsecs);
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
  inspectorResult_t inspectorStateDumpOtel(const char* output_root);
  static void* dumpMain(void* arg);
};

struct inspectorKernelChInfo {
  uint64_t type;
  int refCount; /*unused*/
  uint64_t parentType;  // ncclProfileColl or ncclProfileP2p
  void* parentObj;      // Pointer to either inspectorCollInfo or inspectorP2pInfo
  uint8_t channelId;
  uint64_t tsStartUsec;
  uint64_t tsCompletedUsec;
  uint64_t startGpuClk;
  uint64_t stopGpuClk;
};

// Live ProxyOp handle. Parent Coll/P2P correlation is copied at start;
// completed records copy the pointer-free metadata instead.
struct inspectorProxyOpInfo {
  uint64_t type;
  int refCount;
  struct inspectorCommInfo* commInfo;
  struct inspectorProxyOpMetadata metadata;
  int nSteps;
  int chunkSize;
  bool stopped;
  uint32_t nStepsStarted;
  uint32_t nStepsCompleted;
  uint32_t nStepsDropped;
  uint64_t nextProxyStepSn;
  uint64_t eventSeqNum;
  size_t transSizeBytes;
  struct inspectorEventTraceInfo evntTrace[NCCL_INSP_EVT_TRK_PROXY_OP_NEVT];
  pthread_rwlock_t guard;
};

// Live ProxyStep handle. Its parent reference keeps the ProxyOp alive until
// the step has produced a pointer-free completed record.
struct inspectorProxyStepInfo {
  uint64_t type;
  struct inspectorProxyOpInfo* parent;
  uint64_t proxyStepSn;
  int step;
  size_t transSizeBytes;
  bool transSizeRecorded;
  struct inspectorEventTraceInfo evntTrace[NCCL_INSP_EVT_TRK_PROXY_STEP_NEVT];
};

struct inspectorCollInfo {
  uint64_t type;
  int refCount;
  struct inspectorCommInfo *commInfo;
  const char* func;
  const char* algo;
  const char* proto;
  uint64_t sn;
  size_t msgSizeBytes;
  uint64_t tsStartUsec;
  uint64_t tsCompletedUsec;
  uint32_t nChannels;
  uint32_t nKernelChStarted;
  uint32_t nKernelChCompleted;
  pthread_rwlock_t guard;
  struct inspectorKernelChInfo kernelCh[MAX_CHANNELS];
  struct inspectorEventTrkOpInfo collEvtTrk;
};

struct inspectorP2pInfo {
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
  struct inspectorEventTrkOpInfo p2pEvtTrk;
  int peer;
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
// Set NCCL_INSPECTOR_ENABLE_WARN=1 to enable WARN-level inspector logging (default: 0 -> INFO)
#if NCCL_INSPECTOR_ENABLE_WARN
#define WARN_INSPECTOR(...) logFn(NCCL_LOG_WARN, NCCL_PROFILE, __func__, __LINE__, __VA_ARGS__)
#else
#define WARN_INSPECTOR(...) INFO_INSPECTOR(__VA_ARGS__)
#endif

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

// Global flag to control P2P tracking
extern bool enableNcclInspectorP2p;
// Global flag to control ProxyOp/ProxyStep tracking
extern bool enableNcclInspectorProxy;
// Controls completed Step records only; callbacks still feed Op statistics.
extern bool enableNcclInspectorProxyStepDump;
extern pid_t ncclInspectorPid;
// Foreign-PID descriptors may also carry a foreign profiler context, so these
// skips cannot be safely attributed to a communicator. Access atomically.
extern uint64_t ncclInspectorProxyPxnSkipped;
extern bool requireKernelTiming;
// Opt-in flag for extra per-device Prometheus stats metrics (totals + drops)
extern bool enableNcclInspectorPromStats;
// Minimum message size (bytes) to be `tracked by inspector
extern size_t ncclInspectorDumpMinSizeBytes;

bool inspectorIsDumpVerboseEnabled();

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

void inspectorUpdateCollPerf(struct inspectorCompletedOpInfo *completedOp,
                             struct inspectorCollInfo *collInfo);
void inspectorUpdateP2pPerf(struct inspectorCompletedOpInfo *completedOp,
                            struct inspectorP2pInfo *p2pInfo);
ncclDataType_t inspectorStringToDatatype(const char* str);

void inspectorComputeOpBw(struct inspectorCommInfo *commInfo,
                          struct inspectorCompletedOpInfo *op);

// Utility functions exposed for Prometheus module
const char* inspectorTimingSourceToString(inspectorTimingSource_t timingSource);
inspectorResult_t inspectorCommInfoListFinalize(struct inspectorCommInfoList* commList);
const char* ncclFuncToString(ncclFunc_t fn);
ncclFunc_t ncclStringToFunc(const char* str);

// Global state
extern struct inspectorState g_state;

#endif  // INSPECTOR_INSPECTOR_H_
