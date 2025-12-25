#include "inspector.h"
#include "inspector_prom.h"
#include "inspector_cudawrap.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <cuda_runtime.h>

#include "common.h"

#define JSON_CHK(expr)                                          \
  do {                                                          \
    const jsonResult_t res = (expr);                            \
    if (res != jsonSuccess) {                                   \
      INFO_INSPECTOR("jsonError: %s\n", jsonErrorString(res));  \
      return inspectorJsonError;                                \
    }                                                           \
  } while (0)


#define JSON_CHK_GOTO(expr, res, label)                                 \
  do {                                                                  \
    const jsonResult_t macro_res = (expr);                              \
    if (macro_res != jsonSuccess) {                                     \
      INFO_INSPECTOR("jsonError: %s\n", jsonErrorString(macro_res));    \
      res = inspectorJsonError;                                         \
      goto label;                                                       \
    }                                                                   \
  } while (0)

#define INS_CUDA_CHK(cmd)                                               \
  do {                                                                  \
    cudaError_t err = cmd;                                              \
    if (err != cudaSuccess) {                                           \
      INFO_INSPECTOR("Cuda failure '%s'", cudaGetErrorString(err));     \
      return inspectorCudaError;                                        \
    }                                                                   \
  } while (false)


// Global flag to control inspector use
static bool enableNcclInspector = false;
// Global flag to control starting internal dump thread
static bool enableNcclInspectorDumpThread = false;
// Global flag to control verbose dumping (event_trace)
static bool enableNcclInspectorDumpVerbose = false;
// Global flag to control prometheus format dumping
static bool enableNcclInspectorPromDump = false;
// Global dump interval in microseconds
static uint64_t ncclInspectorDumpIntervalUsecs = 0;
// Extra guard to prevent spurious messages for eager pollers that try to dump
// out results before we have initialized
static bool ncclInspectorInit = false;

// Define the global logFn variable
ncclDebugLogger_t logFn = nullptr;

/*
 * Description:
 *
 *   Returns the current time in microseconds since the epoch.
 *
 * Thread Safety:
 *
 *   Thread-safe (uses gettimeofday).
 *
 * Input:
 *
 *   None.
 *
 * Output:
 *
 *   None.
 *
 * Return:
 *   uint64_t - current time in microseconds.
 *
 * Error Handling:
 *   This function uses gettimeofday() which rarely fails. In case of
 *   failure, the function returns 0. Callers should check for 0 return
 *   value if precise error handling is required.
 *
 */
uint64_t inspectorGetTime() {
  uint64_t ts = 0;
  timeval tv;

  gettimeofday(&tv, 0);
  ts = tv.tv_sec * 1000000 + tv.tv_usec;
  return ts;
}

/*
 * Description:
 *
 *   Wrapper around inspectorGetTime() that returns formatted UTC datetime string.
 *
 * Thread Safety:
 *
 *   Not thread-safe. Onus of thread safety is on the caller/owner of
 *   the buffer.
 *
 * Input:
 *   char* buffer - output buffer for datetime string.
 *   size_t bufferSize - size of output buffer.
 *
 * Output:
 *   buffer contains UTC datetime string in ISO 8601 format.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorGetTimeUTC(char* buffer, size_t bufferSize) {
  if (!buffer || bufferSize < 21) {  // Need at least 20 chars for "YYYY-MM-DDTHH:MM:SSZ"
    return inspectorMemoryError;
  }

  uint64_t timestampUsec = inspectorGetTime();
  time_t timestampSec = timestampUsec / 1000000;  // Convert microseconds to seconds
  struct tm* utc_tm = gmtime(&timestampSec);

  if (utc_tm) {
    // Format as ISO 8601 datetime: YYYY-MM-DDTHH:MM:SSZ
    if (strftime(buffer, bufferSize, "%Y-%m-%dT%H:%M:%SZ", utc_tm) == 0) {
      return inspectorMemoryError;  // Buffer too small
    }
  } else {
    // Fallback if gmtime fails
    snprintf(buffer, bufferSize, "unknown");
  }

  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Converts a string to the corresponding ncclDataType_t enum value.
 *
 * Thread Safety:
 *   Thread-safe (read-only string input).
 *
 * Input:
 *
 *   const char* str - string representation of the datatype.
 *
 * Output:
 *
 *   None.
 *
 * Return:
 *
 *   ncclDataType_t - corresponding enum value, or -1 if unknown.
 *
 */
ncclDataType_t inspectorStringToDatatype(const char* str) {
  if (strcmp(str, "ncclInt8") == 0) return ncclInt8;
  if (strcmp(str, "ncclInt32") == 0) return ncclInt32;
  if (strcmp(str, "ncclUint32") == 0) return ncclUint32;
  if (strcmp(str, "ncclInt64") == 0) return ncclInt64;
  if (strcmp(str, "ncclUint64") == 0) return ncclUint64;
  if (strcmp(str, "ncclFloat16") == 0) return ncclFloat16;
  if (strcmp(str, "ncclFloat32") == 0) return ncclFloat32;
  if (strcmp(str, "ncclFloat64") == 0) return ncclFloat64;
  if (strcmp(str, "ncclBfloat16") == 0) return ncclBfloat16;
  if (strcmp(str, "ncclFloat8e4m3") == 0) return ncclFloat8e4m3;
  if (strcmp(str, "ncclFloat8e5m2") == 0) return ncclFloat8e5m2;
  return (ncclDataType_t)-1;  // Or handle error as appropriate
}

/*
 * Description:
 *
 *   Converts a string to the corresponding ncclFunc_t enum value.
 *
 * Thread Safety:
 *   Thread-safe (read-only string input).
 *
 * Input:
 *   const char* str - string representation of the function (must not be NULL).
 *
 * Output:
 *   None.
 *
 * Return:
 *   ncclFunc_t - corresponding enum value, or ncclNumFuncs if unknown.
 *
 * Preconditions:
 *   - str must not be NULL
 */
ncclFunc_t ncclStringToFunc(const char* str) {
  if (strcmp(str, "AllGather") == 0) return ncclFuncAllGather;
  if (strcmp(str, "AllReduce") == 0) return ncclFuncAllReduce;
  if (strcmp(str, "Broadcast") == 0) return ncclFuncBroadcast;
  if (strcmp(str, "Recv") == 0) return ncclFuncRecv;
  if (strcmp(str, "Reduce") == 0) return ncclFuncReduce;
  if (strcmp(str, "ReduceScatter") == 0) return ncclFuncReduceScatter;
  if (strcmp(str, "SendRecv") == 0) return ncclFuncSendRecv;
  if (strcmp(str, "Send") == 0) return ncclFuncSend;
  return ncclNumFuncs; // Invalid / unknown
}

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  case ncclFuncSendRecv: return "SendRecv";
  case ncclFuncSend: return "Send";
  default: return "Invalid";
  }
}

struct inspectorDumpThread;
static inspectorDumpThread* dumper = nullptr;

#define UNUSED(x) (void)(x)

inspectorResult_t inspectorLockInit(pthread_rwlock_t* lockRef) {
  if (0 != pthread_rwlock_init(lockRef, nullptr)) {
    return inspectorLockError;
  } else {
    return inspectorSuccess;
  }
}

inspectorResult_t inspectorLockDestroy(pthread_rwlock_t* lockRef) {
  if (0 != pthread_rwlock_destroy(lockRef)) {
    return inspectorLockError;
  } else {
    return inspectorSuccess;
  }
}

inspectorResult_t inspectorLockRd(pthread_rwlock_t* lockRef) {
  if (0 != pthread_rwlock_rdlock(lockRef)) {
    return inspectorLockError;
  } else {
    return inspectorSuccess;
  }
}

inspectorResult_t inspectorLockWr(pthread_rwlock_t* lockRef) {
  if (0 != pthread_rwlock_wrlock(lockRef)) {
    return inspectorLockError;
  } else {
    return inspectorSuccess;
  }
}

inspectorResult_t inspectorUnlockRWLock(pthread_rwlock_t* lockRef) {
  if (0 != pthread_rwlock_unlock(lockRef)) {
    return inspectorLockError;
  } else {
    return inspectorSuccess;
  }
}




inspectorState g_state;

static inspectorResult_t inspectorCommInfoListInit(struct inspectorCommInfoList* commList) {
  if (commList->comms) {
    return inspectorGlobalInitError;
  }
  commList->comms = nullptr;
  commList->ncomms = 0;
  INS_CHK(inspectorLockInit(&commList->guard));
  return inspectorSuccess;
}

static inspectorResult_t inspectorGlobalStateInit() {
  memset(&g_state, 0, sizeof(struct inspectorState));
  INS_CHK(inspectorCommInfoListInit(&g_state.liveComms));
  INS_CHK(inspectorCommInfoListInit(&g_state.deletedComms));
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Converts inspectorTimingSource_t enum to a string representation.
 *
 * Thread Safety:
 *   Thread-safe (read-only operation).
 *
 * Input:
 *   inspectorTimingSource_t timingSource - timing source enum value.
 *
 * Output:
 *   None.
 *
 * Return:
 *   const char* - string representation of the timing source.
 */
const char* inspectorTimingSourceToString(inspectorTimingSource_t timingSource) {
  switch (timingSource) {
  case inspectorTimingSourceKernelGpu:
    return "kernel_gpu";
  case inspectorTimingSourceKernelCpu:
    return "kernel_cpu";
  case inspectorTimingSourceCollectiveCpu:
    return "collective_cpu";
  default:
    return "unknown";
  }
}

/*
 * Description:
 *
 *   Writes the header information for a communicator to the JSON output.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *   struct inspectorCommInfo* commInfo - communicator info.
 *
 * Output:
 *   Header is written to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inspectorResult_t inspectorCommInfoHeader(jsonFileOutput* jfo,
                                                 struct inspectorCommInfo* commInfo) {
  JSON_CHK(jsonStartObject(jfo));
  JSON_CHK(jsonKey(jfo, "id")); JSON_CHK(jsonStr(jfo, commInfo->commHashStr));
  JSON_CHK(jsonKey(jfo, "rank")); JSON_CHK(jsonInt(jfo, commInfo->rank));
  JSON_CHK(jsonKey(jfo, "n_ranks")); JSON_CHK(jsonInt(jfo, commInfo->nranks));
  JSON_CHK(jsonKey(jfo, "nnodes")); JSON_CHK(jsonUint64(jfo, commInfo->nnodes));
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Writes metadata header information to the JSON output.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *
 * Output:
 *   Metadata header is written to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inspectorResult_t inspectorCommInfoMetaHeader(jsonFileOutput* jfo) {
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "inspector_output_format_version")); JSON_CHK(jsonStr(jfo, "v4.0"));
    JSON_CHK(jsonKey(jfo, "git_rev")); JSON_CHK(jsonStr(jfo, get_git_version_info()));
    JSON_CHK(jsonKey(jfo, "rec_mechanism")); JSON_CHK(jsonStr(jfo, "nccl_profiler_interface"));
    JSON_CHK(jsonKey(jfo, "dump_timestamp_us")); JSON_CHK(jsonUint64(jfo, inspectorGetTime()));
    char hostname[256];
    gethostname(hostname, 255);
    JSON_CHK(jsonKey(jfo, "hostname")); JSON_CHK(jsonStr(jfo, hostname));
    JSON_CHK(jsonKey(jfo, "pid")); JSON_CHK(jsonUint64(jfo, getpid()));
  }
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Writes verbose information (event_trace) for a completed
 *   collective operation to the JSON output.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *   const struct inspectorCompletedCollInfo* collInfo - completed
 *   collective info.
 *
 * Output:
 *   Verbose collective info is written to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inline inspectorResult_t inspectorCompletedCollVerbose(jsonFileOutput* jfo,
                                                              struct inspectorCompletedCollInfo* collInfo) {
  // Add event trace information
  JSON_CHK(jsonKey(jfo, "event_trace_sn"));
  JSON_CHK(jsonStartObject(jfo));
  {
    // Collective events
    JSON_CHK(jsonKey(jfo, "coll_start_sn")); JSON_CHK(jsonUint64(jfo, collInfo->collEvtTrk.evntTrace[NCCL_INSP_EVT_TRK_COLL_START].sn));
    JSON_CHK(jsonKey(jfo, "coll_stop_sn")); JSON_CHK(jsonUint64(jfo, collInfo->collEvtTrk.evntTrace[NCCL_INSP_EVT_TRK_COLL_STOP].sn));

    // Kernel events
    JSON_CHK(jsonKey(jfo, "kernel_events"));
    JSON_CHK(jsonStartList(jfo));
    for (uint32_t ch = 0; ch < collInfo->collEvtTrk.nChannels; ch++) {
      JSON_CHK(jsonStartObject(jfo));
      JSON_CHK(jsonKey(jfo, "channel_id")); JSON_CHK(jsonInt(jfo, ch));
      JSON_CHK(jsonKey(jfo, "kernel_start_sn")); JSON_CHK(jsonUint64(jfo, collInfo->collEvtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_START].sn));
      JSON_CHK(jsonKey(jfo, "kernel_stop_sn")); JSON_CHK(jsonUint64(jfo, collInfo->collEvtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_STOP].sn));
      JSON_CHK(jsonKey(jfo, "kernel_record_sn")); JSON_CHK(jsonUint64(jfo, collInfo->collEvtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_RECORD].sn));
      JSON_CHK(jsonFinishObject(jfo));
    }
    JSON_CHK(jsonFinishList(jfo));
  }
  JSON_CHK(jsonFinishObject(jfo));

  JSON_CHK(jsonKey(jfo, "event_trace_ts"));
  JSON_CHK(jsonStartObject(jfo));
  {
    // Collective events
    JSON_CHK(jsonKey(jfo, "coll_start_ts")); JSON_CHK(jsonUint64(jfo, collInfo->collEvtTrk.evntTrace[NCCL_INSP_EVT_TRK_COLL_START].ts));
    JSON_CHK(jsonKey(jfo, "coll_stop_ts")); JSON_CHK(jsonUint64(jfo, collInfo->collEvtTrk.evntTrace[NCCL_INSP_EVT_TRK_COLL_STOP].ts));

    // Kernel events
    JSON_CHK(jsonKey(jfo, "kernel_events"));
    JSON_CHK(jsonStartList(jfo));
    for (uint32_t ch = 0; ch < collInfo->collEvtTrk.nChannels; ch++) {
      JSON_CHK(jsonStartObject(jfo));
      JSON_CHK(jsonKey(jfo, "channel_id")); JSON_CHK(jsonInt(jfo, ch));
      JSON_CHK(jsonKey(jfo, "kernel_start_ts")); JSON_CHK(jsonUint64(jfo, collInfo->collEvtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_START].ts));
      JSON_CHK(jsonKey(jfo, "kernel_stop_ts")); JSON_CHK(jsonUint64(jfo, collInfo->collEvtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_STOP].ts));
      JSON_CHK(jsonKey(jfo, "kernel_record_ts")); JSON_CHK(jsonUint64(jfo, collInfo->collEvtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_RECORD].ts));
      JSON_CHK(jsonFinishObject(jfo));
    }
    JSON_CHK(jsonFinishList(jfo));
  }
  JSON_CHK(jsonFinishObject(jfo));

  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Writes completed collective operation information to the JSON
 *   output.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *   const struct inspectorCompletedCollInfo* collInfo - completed
 *   collective info.
 *
 * Output:
 *   Collective info is written to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inline inspectorResult_t inspectorCompletedColl(jsonFileOutput* jfo,
                                                       struct inspectorCompletedCollInfo* collInfo) {
  JSON_CHK(jsonStartObject(jfo));
  {

    JSON_CHK(jsonKey(jfo, "coll")); JSON_CHK(jsonStr(jfo, ncclFuncToString(collInfo->func)));

    JSON_CHK(jsonKey(jfo, "coll_sn")); JSON_CHK(jsonUint64(jfo, collInfo->sn));

    JSON_CHK(jsonKey(jfo, "coll_msg_size_bytes")); JSON_CHK(jsonUint64(jfo, collInfo->msgSizeBytes));

    JSON_CHK(jsonKey(jfo, "coll_exec_time_us")); JSON_CHK(jsonUint64(jfo, collInfo->execTimeUsecs));

    JSON_CHK(jsonKey(jfo, "coll_timing_source")); JSON_CHK(jsonStr(jfo, inspectorTimingSourceToString(collInfo->timingSource)));

    JSON_CHK(jsonKey(jfo, "coll_algobw_gbs")); JSON_CHK(jsonDouble(jfo, collInfo->algoBwGbs));

    JSON_CHK(jsonKey(jfo, "coll_busbw_gbs")); JSON_CHK(jsonDouble(jfo, collInfo->busBwGbs));

    if (enableNcclInspectorDumpVerbose) {
      INS_CHK(inspectorCompletedCollVerbose(jfo, collInfo));
    }
  }
  JSON_CHK(jsonFinishObject(jfo));

  return inspectorSuccess;
}


/*
 * Description:
 *
 *   Dumps the state of a communicator to the JSON output if needed.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *   inspectorCommInfo* commInfo - communicator info.
 *   bool* needs_writing - set to true if output was written.
 *
 * Output:
 *   State is dumped to JSON output if needed.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inspectorResult_t inspectorCommInfoDump(jsonFileOutput* jfo,
                                               inspectorCommInfo* commInfo,
                                               bool* needs_writing) {
  *needs_writing = false;

  if (commInfo == nullptr)
    return inspectorSuccess;

  struct inspectorCompletedCollInfo collInfo;
  memset(&collInfo, 0, sizeof(struct inspectorCompletedCollInfo));

  inspectorLockWr(&commInfo->guard);
  if (commInfo->dump) {
    *needs_writing = true;
    memcpy(&collInfo,
           &commInfo->completedCollInfo,
           sizeof(struct inspectorCompletedCollInfo));
    commInfo->dump = false;
  }
  inspectorUnlockRWLock(&commInfo->guard);

  if (*needs_writing) {
    JSON_CHK(jsonLockOutput(jfo));
    JSON_CHK(jsonStartObject(jfo));
    {
      JSON_CHK(jsonKey(jfo, "header"));
      inspectorCommInfoHeader(jfo, commInfo);

      JSON_CHK(jsonKey(jfo, "metadata"));
      inspectorCommInfoMetaHeader(jfo);

      JSON_CHK(jsonKey(jfo, "coll_perf"));
      INS_CHK(inspectorCompletedColl(jfo, &collInfo));
    }
    JSON_CHK(jsonFinishObject(jfo));
    JSON_CHK(jsonNewline(jfo));
    JSON_CHK(jsonUnlockOutput(jfo));
  }
  return inspectorSuccess;
}


/*
 * Description:
 *
 *   Dumps the state of all communicators in a commList to the JSON
 *   output.
 *
 * Thread Safety:
 *   Thread-safe - assumes no locks are taken and acquires all necessary
 *   locks to iterate through all communicator objects and dump their state.
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle (must not be NULL).
 *   struct inspectorCommInfoList* commList - list of communicators (must not be NULL).
 *
 * Output:
 *   State of all communicators is dumped to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inspectorResult_t inspectorCommInfoListDump(jsonFileOutput* jfo,
                                                   struct inspectorCommInfoList* commList) {
  bool flush = false;
  INS_CHK(inspectorLockRd(&commList->guard));
  inspectorResult_t res = inspectorSuccess;
  if (commList->ncomms > 0) {
    for (struct inspectorCommInfo* itr = commList->comms;
         itr != nullptr;
         itr = itr->next) {
      bool needs_writing;
      INS_CHK_GOTO(inspectorCommInfoDump(jfo, itr, &needs_writing), res, exit);
      if (needs_writing) {
        flush = true;
      }
    }
    if (flush) {
      JSON_CHK_GOTO(jsonLockOutput(jfo), res, exit);
      JSON_CHK_GOTO(jsonFlushOutput(jfo), res, exit);
      JSON_CHK_GOTO(jsonUnlockOutput(jfo), res, exit);
    }
  }
exit:
  INS_CHK(inspectorUnlockRWLock(&commList->guard));
  return res;
}

/*
 * Description:
 *   Finalizes and cleans up a commList, freeing all communicators.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   struct commList* commList - list of communicators.
 *
 * Output:
 *   All communicators are freed.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
inspectorResult_t inspectorCommInfoListFinalize(struct inspectorCommInfoList* commList) {
  struct inspectorCommInfo* nextComm = nullptr;
  INS_CHK(inspectorLockWr(&commList->guard));
  while (commList->comms != nullptr && commList->ncomms != 0) {
    TRACE_INSPECTOR("NCCL Inspector: comm %lu still in tracker",
                    commList->comms->commHash);
    nextComm = commList->comms->next;
    INS_CHK(inspectorLockDestroy(&commList->comms->guard));
    free(commList->comms);
    commList->comms = nextComm;
    commList->ncomms--;
  }
  INS_CHK(inspectorUnlockRWLock(&commList->guard));
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Ensures the given directory exists and is writable, creating it
 *   if necessary.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during initialization).
 *
 * Input:
 *   char* workdir - directory path.
 *
 * Output:
 *   Directory is created if needed.
 *
 * Return:
 *
 *   bool - true if directory exists and is writable, false otherwise.
 *
 */
static bool ensureDir(char* workdir) {
  struct stat st;

  // Check if directory exists
  if (stat(workdir, &st) == 0) {
    if (S_ISDIR(st.st_mode)) {
      // Directory exists, check if it's writable
      if (access(workdir, W_OK) == 0) {
        return true; // Directory exists and is writable
    } else {
      INFO_INSPECTOR(
        "NCCL Inspectoer: dump directory %s exists, but is not "
        "writable",
        workdir);
      return false;
    }
    } else {
      INFO_INSPECTOR(
        "NCCL Inspector: dump location %s exists, but is not a "
        "directory",
        workdir);
      return false;
    }
  } else {
    // Directory doesn't exist, try to create it
    const mode_t mode = 0777;
    if (mkdir(workdir, mode) == 0) {
      return true; // Directory created successfully
    } else {
      INFO_INSPECTOR(
        "NCCL Inspector: failed to create dump directory %s: %s", workdir,
        strerror(errno));
      return false;
    }
  }
}

/*
 * Description:
 *
 *   Generates the output dump directory path based on environment
 *   variables.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during initialization).
 *
 * Input:
 *   char** workdir - pointer to output directory string.
 *
 * Output:
 *   workdir is set to the generated directory path.
 *
 * Return:
 *   None.
 */
static void genDumpDir(char** workdir) {
  const char* dumpdir = getenv("NCCL_INSPECTOR_DUMP_DIR");
  if (dumpdir != NULL) {
    *workdir = strdup(dumpdir);
    // TODO check errors here
    return;
  }

  const char* jobid = getenv("SLURM_JOBID");
  bool badJobId = true;
  if (jobid != NULL) {
    errno = 0;
    const int intid = strtol(jobid, NULL, 10);
    if (errno == 0) {
      char tmp[2048];
      snprintf(tmp, 2048, "nccl-inspector-%d", intid);
      *workdir = strdup(tmp);
      badJobId = false;
    }
  }

  if (badJobId) {
    *workdir = strdup("nccl-inspector-unknown-jobid");
  }
}


inspectorDumpThread::inspectorDumpThread(const char* _outputRoot, uint64_t _sampleIntervalUsecs)
  : jfo(nullptr), outputRoot(strdup(_outputRoot)), sampleIntervalUsecs(_sampleIntervalUsecs) {
  if (inspectorLockInit(&guard) != inspectorSuccess) {
    INFO_INSPECTOR("NCCL Inspector inspectorDumpThread: couldn't init lock");
  }
}

inspectorDumpThread::~inspectorDumpThread() {
  // Close and cleanup Prometheus files, only in Prom mode
  if (enableNcclInspectorPromDump) {
    // Close any open Prometheus file handles
    for (size_t i = 0; i < deviceFlushEntries.size(); i++) {
      if (deviceFlushEntries[i].fileHandle) {
        fclose(deviceFlushEntries[i].fileHandle);
        deviceFlushEntries[i].fileHandle = NULL;
      }
    }

    // Cleanup (delete) prom files after closing them
    for (size_t i = 0; i < deviceFlushEntries.size(); i++) {
      if (deviceFlushEntries[i].filename[0] != '\0') {
        if (unlink(deviceFlushEntries[i].filename) == 0) {
          TRACE_INSPECTOR("NCCL Inspector: Cleaned up Prometheus file %s",
                          deviceFlushEntries[i].filename);
        } else {
          INFO_INSPECTOR("NCCL Inspector: Failed to cleanup Prometheus file %s: %s",
                         deviceFlushEntries[i].filename, strerror(errno));
        }
      }
    }
  }

  if (jfo != nullptr) {
    jsonFinalizeFileOutput(jfo);
    jfo = nullptr;
  }
  if (outputRoot != nullptr) {
    free(outputRoot);
    outputRoot = nullptr;
  }
  if (inspectorLockDestroy(&guard) != inspectorSuccess) {
    INFO_INSPECTOR("NCCL Inspector inspectorDumpThread: couldn't destroy lock");
  }
}

// Implementation of inspectorDumpThread methods
FILE* inspectorDumpThread::getOrCreateFileHandle(const char* deviceUuidStr,
                                                 const char* filename,
                                                 uint64_t currentTime) {
  int flushIndex = -1;
  bool needsFlush = false;

  // Find existing entry for this device UUID
  for (size_t i = 0; i < deviceFlushEntries.size(); i++) {
    if (strncmp(deviceFlushEntries[i].deviceUuidStr,
                deviceUuidStr,
                sizeof(deviceFlushEntries[i].deviceUuidStr) - 1) == 0) {
      flushIndex = static_cast<int>(i);

      // Check if we need to flush (clear) the file
      if (deviceFlushEntries[i].lastFlushTime == 0
          || ((currentTime - deviceFlushEntries[i].lastFlushTime)
              >= sampleIntervalUsecs)) {
        needsFlush = true;
      }
      break;
    }
  }

  // If not found, add new entry
  if (flushIndex == -1) {
    deviceFlushInfo newEntry;
    strncpy(newEntry.deviceUuidStr, deviceUuidStr, sizeof(newEntry.deviceUuidStr));
    newEntry.deviceUuidStr[sizeof(newEntry.deviceUuidStr) - 1] = '\0';
    strncpy(newEntry.filename, filename, sizeof(newEntry.filename) - 1);
    newEntry.filename[sizeof(newEntry.filename) - 1] = '\0';
    newEntry.lastFlushTime = 0;
    newEntry.fileHandle = NULL;
    newEntry.needsCreation = true;

    deviceFlushEntries.push_back(newEntry);
    flushIndex = static_cast<int>(deviceFlushEntries.size() - 1);
    needsFlush = true;
  }

  // Close existing handle if we need to flush (recreate file)
  if (needsFlush && deviceFlushEntries[flushIndex].fileHandle) {
    fclose(deviceFlushEntries[flushIndex].fileHandle);
    deviceFlushEntries[flushIndex].fileHandle = NULL;
  }

  // Open/create file if needed
  if (!deviceFlushEntries[flushIndex].fileHandle) {
    // Create file if flushing, otherwise append
    const char* mode = needsFlush ? "w" : "a";
    FILE* file = fopen(filename, mode);

    if (!file) {
      INFO_INSPECTOR("NCCL Inspector: Failed to open Prometheus file %s", filename);
      return NULL;
    }

    chmod(filename, 0777);

    deviceFlushEntries[flushIndex].fileHandle = file;

    if (needsFlush) {
      TRACE_INSPECTOR("NCCL Inspector: Created/flushed Prometheus file %s", filename);
      deviceFlushEntries[flushIndex].lastFlushTime = currentTime;
    }
  }

  return deviceFlushEntries[flushIndex].fileHandle;
}

void inspectorDumpThread::startThread() {
  inspectorLockWr(&guard);
  run = true;
  inspectorUnlockRWLock(&guard);
  if (pthread_create(&pthread, NULL, dumpMain, this) != 0) {
    INFO_INSPECTOR(
      "NCCL Inspector inspectorDumpThread: couldn't create dump thread!");
    return;
  }
  TRACE_INSPECTOR("NCCL Inspector inspectorDumpThread: created");
}

void inspectorDumpThread::stopThread() {
  INFO(NCCL_ENV, "NCCL Inspector Stopping Dump thread");
  inspectorLockWr(&guard);
  run = false;
  inspectorUnlockRWLock(&guard);
  struct timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = 1000000; // 1ms
  nanosleep(&ts, NULL);
  INFO_INSPECTOR( "NCCL Inspector inspectorDumpThread: stopped");
}

inspectorResult_t inspectorDumpThread::inspectorStateDump(const char* output_root) {
  if (!ncclInspectorInit) {
    return inspectorUninitializedError;
  }
  if (!enableNcclInspector) {
    INFO_INSPECTOR( "NCCL Inspector is not enabled, will not do ncclAllCommTallyDump");
    return inspectorDisabledError;
  }

  if (enableNcclInspectorPromDump) {
    return inspectorStateDumpProm(output_root);
  } else {
    return inspectorStateDumpJSON(output_root);
  }
}

inspectorResult_t inspectorDumpThread::inspectorStateDumpJSON(const char* output_root) {
  if (jfo == 0) {
    char hostname[256];
    gethostname(hostname, 255);
    char tmp[2048];
    snprintf(tmp, sizeof(tmp), "%s/%s-pid%d.log", output_root, hostname, getpid());
    jsonResult_t result = jsonInitFileOutput(&jfo, tmp);
    if (jsonSuccess != result) {
      INFO_INSPECTOR("Cannot open %s for writing: %s", tmp, jsonErrorString(result));
      return inspectorFileOpenError;
    }
    chmod(tmp, 0666);
  }

  if (jfo != nullptr) {
    inspectorCommInfoListDump(jfo, &g_state.liveComms);
    inspectorCommInfoListDump(jfo, &g_state.deletedComms);
  }

  if (g_state.deletedComms.ncomms > 0) {
    inspectorCommInfoListFinalize(&g_state.deletedComms);
  }
  return inspectorSuccess;
}

inspectorResult_t inspectorDumpThread::inspectorStateDumpProm(const char* output_root) {
  // Write communicators directly to files with per-device flushing handled inside
  inspectorResult_t dumpResult
    = inspectorPromCommInfoListDump(&g_state.liveComms,
                                    output_root,
                                    this);
  if (dumpResult != inspectorSuccess) {
    INFO_INSPECTOR("NCCL Inspector: Direct Prometheus dump failed: %s",
                   inspectorErrorString(dumpResult));
    return dumpResult;
  }

  // Finalize deleted communicators
  if (g_state.deletedComms.ncomms > 0) {
    inspectorCommInfoListFinalize(&g_state.deletedComms);
  }

  return inspectorSuccess;
}

void* inspectorDumpThread::dumpMain(void* arg) {
  inspectorDumpThread* dumper = (inspectorDumpThread*)arg;
  inspectorResult_t res = inspectorSuccess;
  struct timespec ts;
  ts.tv_sec = dumper->sampleIntervalUsecs / 1000000;
  ts.tv_nsec = dumper->sampleIntervalUsecs % 1000000;

  while (dumper->run) {
    inspectorLockWr(&dumper->guard);
    if (!dumper->run) {
      inspectorUnlockRWLock(&dumper->guard);
      break;
    }
    res = dumper->inspectorStateDump(dumper->outputRoot);
    if (res == inspectorFileOpenError || res == inspectorDisabledError) {
      inspectorUnlockRWLock(&dumper->guard);
      break;
    }
    inspectorUnlockRWLock(&dumper->guard);

    nanosleep(&ts, NULL);
  }

  return 0;
}

/*
 * Description:
 *
 *   Starts the internal dump thread with the specified interval.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during initialization).
 *
 * Input:
 *   uint64_t intervalUsecs - dump interval in microseconds.
 *
 * Output:
 *   Dump thread is started if successful.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static inspectorResult_t inspectorStartDumpThread(uint64_t intervalUsecs) {
  if (intervalUsecs == 0) {
    INFO_INSPECTOR( "NCCL Inspector: dump thread enabled but "
                    "dump interval is 0; not starting internal dump thread.");
    return inspectorSuccess;
  }

  char* dumpdir;
  genDumpDir(&dumpdir);

  if (dumpdir != nullptr) {
    if (!ensureDir(dumpdir)) {
      free(dumpdir);
      INFO_INSPECTOR( "NCCL Inspector: failed to generate a dump dir; not "
                      "starting internal dump thread.");
      return inspectorSuccess;
    }

    dumper = new inspectorDumpThread(dumpdir, intervalUsecs);
    INFO_INSPECTOR(
      "NCCL Inspector enabled with polling interval %lu us, "
      "output directory %s, format %s",
      intervalUsecs, dumpdir,
      enableNcclInspectorPromDump ? "Prometheus" : "JSON");
    dumper->startThread();

    free(dumpdir);
  } else {
    INFO_INSPECTOR( "NCCL Inspector: failed to generate a dump "
                    "dir; not starting internal dump thread.");
  }

  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Shows the NCCL Inspector plugin version and configuration
 *   environment variables in a structured format similar to NCCL's
 *   showVersion function.
 *
 * Thread Safety:
 *   Thread-safe (read-only environment variable access).
 *
 * Input:
 *   None.
 *
 * Output:
 *   Logs version and environment variables to debug output.
 *
 * Return:
 *   None.
 */
static void showInspectorVersion() {
  VERSION("NCCL Inspector Plugin - Version: %s", get_git_version_info());
}

/*
 * Description:
 *
 *   Shows all NCCL Inspector environment variables and their values
 *   in a structured format.
 *
 * Thread Safety:
 *   Thread-safe (read-only environment variable access).
 *
 * Input:
 *   None.
 *
 * Output:
 *   Logs environment variables to debug output.
 *
 * Return:
 *   None.
 */
static void showInspectorEnvVars() {
  struct {
    const char* name;
    const char* value;
    const char* defaultVal;
    const char* description;
  } envVars[] = {
    {"NCCL_INSPECTOR_ENABLE", getenv("NCCL_INSPECTOR_ENABLE"), "0", "Enable/disable inspector plugin"},
    {"NCCL_INSPECTOR_DUMP_THREAD_ENABLE", getenv("NCCL_INSPECTOR_DUMP_THREAD_ENABLE"), "1", "Enable/disable dump thread"},
    {"NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS", getenv("NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS"), "0", "Dump thread interval in microseconds"},
    {"NCCL_INSPECTOR_DUMP_DIR", getenv("NCCL_INSPECTOR_DUMP_DIR"), "(auto-generated)", "Output directory for inspector logs"},
    {"NCCL_INSPECTOR_DUMP_VERBOSE", getenv("NCCL_INSPECTOR_DUMP_VERBOSE"), "0", "Enable/disable verbose dumping (event_trace)"},
    {"NCCL_INSPECTOR_PROM_DUMP", getenv("NCCL_INSPECTOR_PROM_DUMP"), "0", "Enable/disable Prometheus format output dump"}
  };

  const int numEnvVars = sizeof(envVars) / sizeof(envVars[0]);

  VERSION("NCCL Inspector Environment Variables:");
  for (int i = 0; i < numEnvVars; i++) {
    VERSION("  %s = %s%s%s",
            envVars[i].name,
            envVars[i].value ? envVars[i].value : "(not set)",
            envVars[i].value ? "" : ", default=",
            envVars[i].value ? "" : envVars[i].defaultVal);
  }
}

/*
 * Description:
 *
 *   Initializes the global inspector state and starts the dump thread
 *   if enabled.
 *
 * Thread Safety:
 *
 *   Not thread-safe (should be called during initialization).
 *
 * Input:
 *   None.
 *
 * Output:
 *   Global state is initialized and dump thread may be started.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorGlobalInit(int rank) {
  TRACE_INSPECTOR("NCCL Inspector: inspectorGlobalInit");
  const char* str = getenv("NCCL_INSPECTOR_ENABLE");
  int enable = str ? atoi(str) : 0; // default disable
  enableNcclInspector = enable == 0 ? false : true;
  ncclInspectorInit = true;

  // Show version and environment configuration (similar to NCCL's showVersion)
  if (rank == 0) {
    showInspectorVersion();
    showInspectorEnvVars();
  }

  if (enableNcclInspector == false) {
    VERSION("NCCL Inspector Plugin DISABLED (NCCL_INSPECTOR_ENABLE=%s)",
            str ? str : "0");
    return inspectorDisabledError;
  }

  // Initialize CUDA wrapper for inspector
  inspectorResult_t cudaInitResult = inspectorCudaWrapInit();
  if (cudaInitResult != inspectorSuccess) {
    INFO_INSPECTOR("NCCL Inspector: Failed to initialize CUDA wrapper");
    return cudaInitResult;
  }

  INS_CHK(inspectorGlobalStateInit());

  str = getenv("NCCL_INSPECTOR_DUMP_THREAD_ENABLE");
  enable = str ? atoi(str) : 1; // default enable
  enableNcclInspectorDumpThread = enable == 0 ? false : true;

  str = getenv("NCCL_INSPECTOR_DUMP_VERBOSE");
  enable = str ? atoi(str) : 0; // default disable
  enableNcclInspectorDumpVerbose = enable == 0 ? false : true;

  // Check for Prometheus dump format
  str = getenv("NCCL_INSPECTOR_PROM_DUMP");
  enable = str ? atoi(str) : 0; // default disable
  enableNcclInspectorPromDump = enable == 0 ? false : true;

  // Read and validate dump interval once
  str = getenv("NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS");
  ncclInspectorDumpIntervalUsecs = str ? strtoull(str, 0, 0) : 0;

  // Apply Prometheus-specific interval validation if enabled
  if (enableNcclInspectorPromDump && enableNcclInspectorDumpThread) {
    ncclInspectorDumpIntervalUsecs
      = inspectorPromValidateInterval(ncclInspectorDumpIntervalUsecs);
  }

  if (enableNcclInspectorDumpThread) {
    INS_CHK(inspectorStartDumpThread(ncclInspectorDumpIntervalUsecs));
  } else {
    INFO_INSPECTOR(
      "NCCL Inspector: NCCL_INSPECTOR_DUMP_THREAD_ENABLE set to 0; not "
      "starting internal dump "
      "thread.");
  }
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Returns a string describing the given inspectorResult_t error
 *   code.
 *
 * Thread Safety:
 *   Thread-safe (read-only operation).
 *
 * Input:
 *   inspectorResult_t result - error code.
 *
 * Output:
 *   None.
 *
 * Return:
 *   const char* - error string.
 */
const char* inspectorErrorString(inspectorResult_t result) {
  switch (result) {
  case inspectorSuccess:
    return "Success";
  case inspectorUninitializedError:
    return "Inspector is not initialized";
  case inspectorMemoryError:
    return "Inspector encountered issue allocating memory";
  case inspectorFileOpenError:
    return "Inspector could not open file";
  case inspectorDisabledError:
    return "Inspector is disabled";
  case inspectorLockError:
    return "Inspector encountered error with lock";
  case inspectorPthreadError:
    return "Inspector encountered error with pthreads";
  case inspectorJsonError:
    return "Inspector encountered error while emitting JSON";
  case inspectorCudaError:
    return "Inspector encountered CUDA error";
  case inspectorBadHash:
    return "Inspector encountered bad communicator hash";
  case inspectorDeleteUnknownCommError:
    return "Inspector was asked to delete a communicator that it is not "
      "tracking";
  case inspectorAddDuplicateCommError:
    return "Inspector was asked to add a communicator it was already "
      "tracking";
  case inspectorNop:
    return "Inspector NOP";
  case inspectorNullTally:
    return "Inspector encountered a null OpTally";
  case inspectorGlobalInitError:
    return "Inspector encountered a repeated global init";
  case inspectorReturn:
    return "Inspector Unconditional Return";
  default:
    return "Unknown error";
  }
}

/*
 * Description:
 *   Converts a communicator hash to a string.
 *
 * Thread Safety:
 *   Thread-safe (writes to provided buffer).
 *
 * Input:
 *   uint64_t commHash - communicator hash.
 *   char hashStr[NCCL_COMM_HASH_LENGTH] - output buffer.
 *
 * Output:
 *   hashStr is set to the string representation of commHash.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorCommGetHashStr(uint64_t commHash,
                                          char hashStr[NCCL_COMM_HASH_LENGTH]) {
  snprintf(hashStr, NCCL_COMM_HASH_LENGTH, "0x%lx",
           commHash);
  return inspectorSuccess;
}

/*
 * Description:
 *   Compares two communicator configurations for equality.
 *
 * Thread Safety:
 *   Thread-safe (read-only comparison).
 *
 * Input:
 *   uint64_t lCommHash - left communicator hash.
 *   uint64_t rCommHash - right communicator hash.
 *   int lRank - left rank.
 *   int rRank - right rank.
 *
 * Output:
 *   None.
 *
 * Return:
 *   bool - true if communicators are equal (same hash and rank), false otherwise.
 */
static bool comm_eq(uint64_t lCommHash, uint64_t rCommHash,
                    int lRank, int rRank) {
  return lCommHash == rCommHash && lRank == rRank;
}

/*
 * Description:
 *   Initializes a communicator info structure with the provided parameters.
 *
 * Thread Safety:
 *   Not thread-safe - should be called during communicator initialization.
 *
 * Input:
 *   struct inspectorCommInfo* commInfo - communicator info structure to initialize (must not be NULL).
 *   const char* commName - communicator name (can be NULL).
 *   uint64_t commHash - communicator hash.
 *   int nnodes - number of nodes (must be > 0).
 *   int nranks - number of ranks (must be > 0).
 *   int rank - rank (must be >= 0 and < nranks).
 *
 * Output:
 *   commInfo is initialized with the provided parameters.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 * Preconditions:
 *   - commInfo must not be NULL
 *   - nnodes must be positive
 *   - nranks must be positive
 *   - rank must be non-negative and less than nranks
 */
static inspectorResult_t inspectorFillCommInfo(struct inspectorCommInfo* commInfo,
                                               const char* commName, uint64_t commHash,
                                               int nnodes, int nranks, int rank) {
  commInfo->commName = commName;
  commInfo->commHash = commHash;
  inspectorCommGetHashStr(commHash, commInfo->commHashStr);
  commInfo->rank = rank;
  commInfo->nranks = nranks;
  commInfo->nnodes = nnodes;
  commInfo->dump = false;

  // Capture current CUDA device ID and convert to UUID string
  int cudaDeviceId = -1;
  cudaError_t err = cudaGetDevice(&cudaDeviceId);
  if (err != cudaSuccess) {
    INFO_INSPECTOR("Inspector: Failed to get CUDA device ID: %s", cudaGetErrorString(err));
    return inspectorCudaError;
  }

  commInfo->cudaDeviceId = cudaDeviceId;

  // Get CUDA device handle for driver API
  CUdevice cuDevice;
  CUresult cuErr = INSPECTOR_CUPFN(cuDeviceGet)(&cuDevice, cudaDeviceId);
  if (cuErr != CUDA_SUCCESS) {
    INFO_INSPECTOR("Inspector: Failed to get CUDA device handle for device %d", cudaDeviceId);
    return inspectorCudaError;
  }

  // Get device UUID and convert to string
  CUuuid deviceUuid;
  cuErr = INSPECTOR_CUPFN(cuDeviceGetUuid)(&deviceUuid, cuDevice);
  if (cuErr != CUDA_SUCCESS) {
    INFO_INSPECTOR("Inspector: Failed to get device UUID for device %d", cudaDeviceId);
    return inspectorCudaError;
  }

  // Format UUID as string (standard UUID format)
  snprintf(commInfo->deviceUuidStr, sizeof(commInfo->deviceUuidStr),
           "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           (unsigned char)deviceUuid.bytes[0], (unsigned char)deviceUuid.bytes[1],
           (unsigned char)deviceUuid.bytes[2], (unsigned char)deviceUuid.bytes[3],
           (unsigned char)deviceUuid.bytes[4], (unsigned char)deviceUuid.bytes[5],
           (unsigned char)deviceUuid.bytes[6], (unsigned char)deviceUuid.bytes[7],
           (unsigned char)deviceUuid.bytes[8], (unsigned char)deviceUuid.bytes[9],
           (unsigned char)deviceUuid.bytes[10], (unsigned char)deviceUuid.bytes[11],
           (unsigned char)deviceUuid.bytes[12], (unsigned char)deviceUuid.bytes[13],
           (unsigned char)deviceUuid.bytes[14], (unsigned char)deviceUuid.bytes[15]);

  INS_CHK(inspectorLockInit(&commInfo->guard));
  commInfo->next = nullptr;

  // Cache static Prometheus labels if Prometheus mode is enabled
  if (enableNcclInspectorPromDump) {
    INS_CHK(inspectorPromCacheStaticLabels(commInfo));
  }

  return inspectorSuccess;
}

/*
 * Description:
 *   Adds a communicator to the global state.
 *
 * Thread Safety:
 *   Thread-safe (uses locks internally).
 *
 * Input:
 *   struct inspectorCommInfo **commInfo - pointer to output struct (must not be NULL).
 *   const char* commName - communicator name (can be NULL).
 *   uint64_t commHash - communicator hash.
 *   int nNodes - number of nodes (must be > 0).
 *   int nranks - number of ranks (must be > 0).
 *   int rank - rank (must be >= 0 and < nranks).
 *
 * Output:
 *   commInfo is set to the new communicator struct.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 * Preconditions:
 *   - commInfo must not be NULL
 *   - nNodes must be positive
 *   - nranks must be positive
 *   - rank must be non-negative and less than nranks
 */
inspectorResult_t inspectorAddComm(struct inspectorCommInfo **commInfo,
                                   const char* commName, uint64_t commHash,
                                   int nNodes, int nranks, int rank) {
  struct inspectorCommInfoList* liveCommInfoList = &g_state.liveComms;
  struct inspectorCommInfo* commInfoPtr = nullptr;

  inspectorResult_t res = inspectorSuccess;
  bool locked = false;
  INSPECTOR_LOCK_RD_FLAG(&liveCommInfoList->guard, locked,
                         "inspectorAddComm: commList::guard -rd");
  for (struct inspectorCommInfo* itr = liveCommInfoList->comms;
       itr != nullptr;
       itr = itr->next) {
    if (comm_eq(commHash, itr->commHash, rank, itr->rank)) {
      INFO_INSPECTOR("NCCL Inspector: comm 0x%lx already in tracker",
                     commHash);
      res = inspectorAddDuplicateCommError;
      goto exit;
    }
  }
  INSPECTOR_UNLOCK_RW_LOCK_FLAG(&liveCommInfoList->guard, locked,
                                "inspectorAddComm: commList::guard");
  commInfoPtr
    = (struct inspectorCommInfo*)calloc(1, sizeof(struct inspectorCommInfo));
  if (0 == commInfoPtr) {
    res = inspectorMemoryError;
    goto exit;
  }
  INS_CHK_GOTO(inspectorFillCommInfo(commInfoPtr,
                                     commName,
                                     commHash,
                                     nNodes,
                                     nranks,
                                     rank),
               res, fail);

  INSPECTOR_LOCK_WR_FLAG(&liveCommInfoList->guard, locked,
                         "inspectorAddComm: commList::guard -wr");
  ++liveCommInfoList->ncomms;
  commInfoPtr->next = liveCommInfoList->comms;
  liveCommInfoList->comms = commInfoPtr;

exit:
  INSPECTOR_UNLOCK_RW_LOCK_FLAG(&liveCommInfoList->guard, locked,
                                "inspectorAddComm: commList::guard");
  *commInfo = commInfoPtr;
  return res;
fail:
  if (commInfoPtr) {
    free(commInfoPtr);
    commInfoPtr = nullptr;
  }
  goto exit;
}

/*
 * Description:
 *
 *   Removes a communicator from the global state and moves it to the
 *   deleted list.
 *
 * Thread Safety:
 *   Thread-safe (uses locks internally).
 *
 * Input:
 *   struct inspectorCommInfo *commInfo - communicator to remove.
 *
 * Output:
 *   Communicator is removed from live list and added to deleted list.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorDelComm(struct inspectorCommInfo *commInfo) {
  struct inspectorCommInfoList* liveCommInfoList = &g_state.liveComms;
  struct inspectorCommInfoList* deletedCommInfoList = &g_state.deletedComms;
  struct inspectorCommInfo* commInfoPtr = nullptr;
  bool locked = false;

  TRACE_INSPECTOR("NCCL Inspector: DelComm removing 0x%lx",
                  commInfo->commHash);

  INSPECTOR_LOCK_WR_FLAG(&liveCommInfoList->guard, locked,
                         "inspectorDelComm: liveCommInfoList::guard -wr");
  struct inspectorCommInfo** prev_ptr = &liveCommInfoList->comms;
  for (struct inspectorCommInfo* itr = liveCommInfoList->comms;
       itr != nullptr;
       itr = itr->next) {
    if (comm_eq(commInfo->commHash, itr->commHash, commInfo->rank, itr->rank)) {
      *prev_ptr = itr->next;
      liveCommInfoList->ncomms--;

      commInfoPtr = itr;
      break;
    }
    prev_ptr = &itr->next;
  }
  INSPECTOR_UNLOCK_RW_LOCK_FLAG(&liveCommInfoList->guard, locked,
                                "inspectorDelComm: liveCommInfoList::guard -unlock");

  if (!commInfoPtr) {
    INFO_INSPECTOR("NCCL Inspector: DelComm can't remove 0x%lx, not present",
                   commInfo->commHash);
    return inspectorDeleteUnknownCommError;
  }

  inspectorLockWr(&commInfoPtr->guard);
  commInfoPtr->dump = false;
  inspectorUnlockRWLock(&commInfoPtr->guard);

  INSPECTOR_LOCK_WR_FLAG(&deletedCommInfoList->guard, locked,
                         "inspectorDelComm: deletedCommInfoList::guard -wr");
  commInfoPtr->next = deletedCommInfoList->comms;
  deletedCommInfoList->comms = commInfoPtr;
  deletedCommInfoList->ncomms++;
  INSPECTOR_UNLOCK_RW_LOCK_FLAG(&deletedCommInfoList->guard, locked,
                                "inspectorDelComm: deletedCommInfoList::guard -unlock");

  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Computes the algorithmic and bus bandwidth (in GB/s) for a given
 *   NCCL collective operation, based on the communication info and
 *   completed collective details. The calculation uses the message
 *   size, execution time, and the type of collective operation to
 *   determine the effective bandwidths. The 'factor' variable adjusts
 *   the bus bandwidth calculation according to the communication
 *   pattern of each collective, as described in the NCCL performance
 *   documentation:
 *   https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
 *
 * Thread Safety:
 *
 *   This function does not perform any locking and assumes the caller
 *   ensures thread safety if required.
 *
 * Input:
 *
 *   commInfo - Pointer to inspectorCommInfo structure containing
 *   communicator details.
 *
 *   completedColl- Pointer to inspectorCompletedCollInfo structure
 *   containing completed collective info.
 *
 *   collType - The type of NCCL collective operation (ncclFunc_t).
 *
 * Output:
 *   Updates the algoBwGbs and busBwGbs fields of the completedColl
 *   structure.
 *
 * Return:
 *   N.A. (void function)
 */
void inspectorComputeCollBw(struct inspectorCommInfo *commInfo,
                            struct inspectorCompletedCollInfo *completedColl,
                            ncclFunc_t collType) {
  double timeInSec = completedColl->execTimeUsecs / 1000000.0;
  double factor = 0.0;
  double trafficSize = 0.0;
  switch (collType) {
  case ncclFuncReduce:
  case ncclFuncBroadcast:
    trafficSize = (double)completedColl->msgSizeBytes;
    factor = 1;
    break;
  case ncclFuncAllReduce:
    trafficSize = (double)completedColl->msgSizeBytes;
    factor = ((double)(2 * (commInfo->nranks - 1))) / ((double)commInfo->nranks);
    break;
  case ncclFuncReduceScatter:
    trafficSize = (double)(completedColl->msgSizeBytes * commInfo->nranks);
    factor = ((double)(commInfo->nranks - 1)) / ((double)commInfo->nranks);
    break;
  case ncclFuncAllGather:
    trafficSize = (double)(completedColl->msgSizeBytes * commInfo->nranks);
    factor = ((double)(commInfo->nranks - 1)) / ((double)commInfo->nranks);
    break;
  case ncclFuncSendRecv:
  case ncclFuncSend:
  case ncclFuncRecv:
    trafficSize = (double)completedColl->msgSizeBytes;
    factor = 1;
    break;
  default:
    trafficSize = 0;
    factor = 0.0;
  }
  completedColl->algoBwGbs = timeInSec != 0 ? (trafficSize / 1.0E9 / timeInSec) : 0;
  completedColl->busBwGbs = completedColl->algoBwGbs * factor;
}

/*
 * Description:
 *
 *   Helper function to calculate kernel execution time using GPU
 *   clock values.  The GPU clock values are measured in nanoseconds
 *   from the globaltimer register.
 *
 * Thread Safety:
 *   Thread-safe (read-only operations on kernel info).
 *
 * Input:
 *   struct inspectorKernelChInfo *kernelCh - kernel channel info
 *   containing GPU clock values.
 *
 * Output:
 *   None.
 *
 * Return:
 *   uint64_t - execution time in microseconds, or 0 if invalid timing
 *   data.
 */
static uint64_t calculateKernelGpuExecTimeUsecs(struct inspectorKernelChInfo *kernelCh) {
  if (kernelCh->startGpuClk != 0 && kernelCh->stopGpuClk != 0) {
    if (kernelCh->stopGpuClk > kernelCh->startGpuClk) {
      uint64_t execTimeNanosecs = kernelCh->stopGpuClk - kernelCh->startGpuClk;
      return execTimeNanosecs / 1000;
    }
  }
  return 0;
}

/*
 * Description:
 *
 *   Calculates the maximum kernel execution time across all kernel
 *   channels in a collective operation, using GPU clock values when
 *   available and falling back to CPU timestamps when necessary.
 *
 * Thread Safety:
 *   Thread-safe (read-only operations on collective info).
 *
 * Input:
 *   struct inspectorCollInfo *collInfo - collective operation info
 *   containing kernel channels.
 *   inspectorTimingSource_t *timingSource - pointer to store the timing source used.
 *
 * Output:
 *   timingSource is set to indicate whether GPU, CPU, or collective timing was used.
 *
 * Return:
 *
 *   uint64_t - maximum execution time in microseconds across all
 *              kernels, or collective execution time if no kernel
 *              timing is available.
 *
 */
static uint64_t calculateMaxKernelExecTimeUsecs(struct inspectorCollInfo *collInfo,
                                                inspectorTimingSource_t *timingSource) {
  uint64_t maxKernelExecTimeUsecs = 0;
  inspectorTimingSource_t bestTimingSource = inspectorTimingSourceCollectiveCpu;

  for (uint32_t i = 0; i < collInfo->nChannels; i++) {
    struct inspectorKernelChInfo *kernelCh = &collInfo->kernelCh[i];
    uint64_t gpuExecTimeUsecs = calculateKernelGpuExecTimeUsecs(kernelCh);
    if (gpuExecTimeUsecs > 0) {
      if (gpuExecTimeUsecs > maxKernelExecTimeUsecs) {
        maxKernelExecTimeUsecs = gpuExecTimeUsecs;
        bestTimingSource = inspectorTimingSourceKernelGpu;
      }
    } else {
      if (kernelCh->tsCompletedUsec > kernelCh->tsStartUsec) {
        uint64_t cpuExecTimeUsecs = kernelCh->tsCompletedUsec - kernelCh->tsStartUsec;
        if (cpuExecTimeUsecs > maxKernelExecTimeUsecs) {
          maxKernelExecTimeUsecs = cpuExecTimeUsecs;
          bestTimingSource = inspectorTimingSourceKernelCpu;
        }
      }
    }
  }

  if (maxKernelExecTimeUsecs > 0) {
    *timingSource = bestTimingSource;
    return maxKernelExecTimeUsecs;
  } else {
    *timingSource = inspectorTimingSourceCollectiveCpu;
    return collInfo->tsCompletedUsec - collInfo->tsStartUsec;
  }
}

/*
 * Description:
 *
 *   Updates the performance information for a completed collective
 *   operation.
 *
 * Thread Safety:
 *   Thread-safe (uses locks internally).
 *
 * Input:
 *   struct inspectorCommInfo *commInfo - communicator info.
 *   struct inspectorCollInfo *collInfo - completed collective info.
 *
 * Output:
 *   commInfo is updated with completed collective info.
 *
 * Return:
 *   None.
 *
 */
void inspectorUpdateCollPerf(struct inspectorCompletedCollInfo *completedColl,
                             struct inspectorCollInfo *collInfo) {
  completedColl->func = ncclStringToFunc(collInfo->func);
  completedColl->sn = collInfo->sn;
  completedColl->msgSizeBytes = collInfo->msgSizeBytes;
  completedColl->execTimeUsecs =
    calculateMaxKernelExecTimeUsecs(collInfo, &completedColl->timingSource);
  completedColl->collEvtTrk = collInfo->collEvtTrk;
}

/*
 * Description:
 *
 *   Finalizes the global inspector state and stops the dump thread if
 *   running.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during teardown).
 *
 * Input:
 *   None.
 *
 * Output:
 *   Global state is finalized and dump thread is stopped.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
inspectorResult_t inspectorGlobalFinalize() {
  // Cleanup CUDA wrapper
  inspectorCudaWrapCleanup();
  if (dumper) {
    dumper->stopThread();
    delete dumper;
    dumper = nullptr;
  }
  return inspectorSuccess;
}
