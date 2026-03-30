/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "inspector.h"
#include "profiler.h"
#include "inspector_prom.h"
#include "inspector_json.h"
#include "inspector_cudawrap.h"
#include "inspector_ring.h"
#include "inspector_event_pool.h"

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
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#include "common.h"

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
// Per-communicator completed-collective ring buffer capacity
static uint32_t ncclInspectorDumpCollRingSize = 1024;
// Per-communicator completed-P2P ring buffer capacity
static uint32_t ncclInspectorDumpP2pRingSize = 1024;
// Minimum message size (bytes) to be tracked by inspector
size_t ncclInspectorDumpMinSizeBytes = 8192;
// Global dump interval in microseconds (-1 = disabled, 0 = continuous, >0 = periodic)
static int64_t ncclInspectorDumpIntervalUsecs = -1;
// Extra guard to prevent spurious messages for eager pollers that try to dump
// out results before we have initialized
static bool ncclInspectorInit = false;
// Global flag to control P2P tracking
bool enableNcclInspectorP2p = true;
// Global flag: require kernel-based timing; discard events without it
bool requireKernelTiming = true;
bool inspectorIsDumpVerboseEnabled() {
  return enableNcclInspectorDumpVerbose;
}

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
  case ncclFuncAll2All: return "All2All";
  case ncclFuncAllGatherV: return "AllGatherV";
  default: return "Invalid";
  }
}

ncclFunc_t ncclStringToFunc(const char* str) {
  if (strcmp(str, "AllGather") == 0) return ncclFuncAllGather;
  if (strcmp(str, "AllReduce") == 0) return ncclFuncAllReduce;
  if (strcmp(str, "Broadcast") == 0) return ncclFuncBroadcast;
  if (strcmp(str, "Recv") == 0) return ncclFuncRecv;
  if (strcmp(str, "Reduce") == 0) return ncclFuncReduce;
  if (strcmp(str, "ReduceScatter") == 0) return ncclFuncReduceScatter;
  if (strcmp(str, "SendRecv") == 0) return ncclFuncSendRecv;
  if (strcmp(str, "Send") == 0) return ncclFuncSend;
  if (strcmp(str, "All2All") == 0) return ncclFuncAll2All;
  if (strcmp(str, "AllGatherV") == 0) return ncclFuncAllGatherV;
  return ncclNumFuncs; // Invalid / unknown
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
    inspectorRingFinalize(&commList->comms->completedCollRing);
    inspectorRingFinalize(&commList->comms->completedP2pRing);
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
          "NCCL Inspector: dump directory %s exists, but is not "
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


inspectorDumpThread::inspectorDumpThread(const char* _outputRoot, int64_t _sampleIntervalUsecs)
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
      if (sampleIntervalUsecs > 0 &&
          (deviceFlushEntries[i].lastFlushTime == 0
           || ((currentTime - deviceFlushEntries[i].lastFlushTime)
               >= (uint64_t)sampleIntervalUsecs))) {
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
  threadStarted = true;
  TRACE_INSPECTOR("NCCL Inspector inspectorDumpThread: created");
}

void inspectorDumpThread::stopThread() {
  INFO(NCCL_ENV, "NCCL Inspector Stopping Dump thread");
  inspectorLockWr(&guard);
  run = false;
  inspectorUnlockRWLock(&guard);
  if (threadStarted) {
    pthread_join(pthread, nullptr);
    threadStarted = false;
  }
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
  // Write communicators directly to files with per-device flushing
  // handled inside
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

    // Sleep only if interval > 0; if interval == 0, dump continuously
    if (dumper->sampleIntervalUsecs > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(dumper->sampleIntervalUsecs));
    }
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
 *   int64_t intervalUsecs - dump interval in microseconds (-1 = disabled, 0 = continuous, >0 = periodic).
 *
 * Output:
 *   Dump thread is started if successful.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static inspectorResult_t inspectorStartDumpThread(int64_t intervalUsecs) {
  if (intervalUsecs < 0) {
    INFO_INSPECTOR( "NCCL Inspector: dump thread disabled "
                    "(interval is -1); not starting internal dump thread.");
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
    if (intervalUsecs == 0) {
      INFO_INSPECTOR(
        "NCCL Inspector enabled with continuous dumping, "
        "output directory %s, format %s",
        dumpdir,
        enableNcclInspectorPromDump ? "Prometheus" : "JSON");
    } else {
      INFO_INSPECTOR(
        "NCCL Inspector enabled with polling interval %ld us, "
        "output directory %s, format %s",
        intervalUsecs, dumpdir,
        enableNcclInspectorPromDump ? "Prometheus" : "JSON");
    }
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
    {"NCCL_INSPECTOR_ENABLE_P2P", getenv("NCCL_INSPECTOR_ENABLE_P2P"), "1", "Enable/disable P2P tracking"},
    {"NCCL_INSPECTOR_DUMP_THREAD_ENABLE", getenv("NCCL_INSPECTOR_DUMP_THREAD_ENABLE"), "1", "Enable/disable dump thread"},
    {"NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS", getenv("NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS"), "-1", "Dump interval in microseconds (-1 = disabled/dump only at teardown, 0 = continuous, >0 = periodic)"},
    {"NCCL_INSPECTOR_DUMP_DIR", getenv("NCCL_INSPECTOR_DUMP_DIR"), "(auto-generated)", "Output directory for inspector logs"},
    {"NCCL_INSPECTOR_DUMP_VERBOSE", getenv("NCCL_INSPECTOR_DUMP_VERBOSE"), "0", "Enable/disable verbose dumping (event_trace)"},
    {"NCCL_INSPECTOR_PROM_DUMP", getenv("NCCL_INSPECTOR_PROM_DUMP"), "0", "Enable/disable Prometheus format output dump"},
    {"NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES", getenv("NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES"), "8192", "Minimum message size (bytes) to be tracked by inspector"},
    {"NCCL_INSPECTOR_DUMP_COLL_RING_SIZE", getenv("NCCL_INSPECTOR_DUMP_COLL_RING_SIZE"), "1024", "Per-communicator completed-collective ring buffer capacity"},
    {"NCCL_INSPECTOR_DUMP_P2P_RING_SIZE", getenv("NCCL_INSPECTOR_DUMP_P2P_RING_SIZE"), "1024", "Per-communicator completed-P2P ring buffer capacity"},
    {"NCCL_INSPECTOR_COLL_POOL_SIZE", getenv("NCCL_INSPECTOR_COLL_POOL_SIZE"), "256", "Collective pool initial size/stride"},
    {"NCCL_INSPECTOR_P2P_POOL_SIZE", getenv("NCCL_INSPECTOR_P2P_POOL_SIZE"), "256", "P2P pool initial size/stride"},
    {"NCCL_INSPECTOR_COMM_POOL_SIZE", getenv("NCCL_INSPECTOR_COMM_POOL_SIZE"), "256", "Comm pool initial size/stride"},
    {"NCCL_INSPECTOR_POOL_GROW", getenv("NCCL_INSPECTOR_POOL_GROW"), "1", "Enable/disable dynamic growth of event pools"},
    {"NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING", getenv("NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING"), "1", "Require GPU-based kernel timing; discard events with CPU-measured timing"},
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
 *   Helper function to read pool size from environment variable with
 *   validation.
 *
 * Parameters:
 *   envVarName  - Name of the environment variable to read.
 *   description - Description of the pool (for logging).
 *   defaultSize - Default size if environment variable is not set.
 *   minSize     - Minimum allowed size.
 *
 * Return:
 *   uint32_t - The validated pool size (>= minSize).
 */
static uint32_t getPoolSizeFromEnv(const char* envVarName,
                                   const char* description,
                                   uint32_t defaultSize,
                                   uint32_t minSize) {
  const char* str = getenv(envVarName);
  uint64_t poolSize = str ? strtoull(str, 0, 0) : defaultSize;
  if (poolSize < minSize) {
    INFO_INSPECTOR("NCCL Inspector: %s %lu too small, using minimum of %u",
                   description, poolSize, minSize);
    poolSize = minSize;
  }
  if (poolSize > UINT32_MAX) {
    INFO_INSPECTOR("NCCL Inspector: %s %lu too large, using maximum of %u",
                   description, poolSize, UINT32_MAX);
    poolSize = UINT32_MAX;
  }
  return (uint32_t)poolSize;
}

/*
 * Description:
 *   Helper function to read ring size from environment variable with
 *   validation.
 *
 * Parameters:
 *   envVarName  - Name of the environment variable to read.
 *   defaultSize - Default size if environment variable is not set.
 *
 * Return:
 *   uint32_t - Validated ring size (>= 1).
 */
static uint32_t getRingSizeFromEnv(const char* envVarName,
                                   uint32_t defaultSize) {
  const char* str = getenv(envVarName);
  uint64_t ringSize = str ? strtoull(str, 0, 0) : defaultSize;
  if (ringSize == 0) {
    ringSize = 1;
  }
  if (ringSize > UINT32_MAX) {
    ringSize = UINT32_MAX;
  }
  return (uint32_t)ringSize;
}

/*
 * Description:
 *
 *   Initializes P2P tracking configuration from environment variables.
 *
 * Return:
 *   None.
 */
static void initP2pTrackingFromEnv() {
  const char* str = getenv("NCCL_INSPECTOR_ENABLE_P2P");
  int enable = str ? atoi(str) : 1;
  enableNcclInspectorP2p = enable == 0 ? false : true;
}

/*
 * Description:
 *
 *   Initializes kernel timing requirement from environment variables.
 *   When enabled (default), only events with GPU-based kernel timing
 *   (kernel_gpu) are recorded. Events with CPU-measured timing
 *   (kernel_cpu or collective_cpu) are discarded. Set
 *   NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=0 to allow all timing
 *   sources (previous behaviour).
 *
 * Return:
 *   None.
 */
static void initKernelTimingFromEnv() {
  const char* str = getenv("NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING");
  int val = str ? atoi(str) : 1;
  requireKernelTiming = val != 0;
}

/*
 * Description:
 *
 *   Initializes event pools using sizes from environment variables.
 *
 * Return:
 *   inspectorResult_t - Result from inspectorEventPoolInit.
 */
static inspectorResult_t inspectorEventPoolInitFromEnv() {
  uint32_t collPoolSize
    = getPoolSizeFromEnv("NCCL_INSPECTOR_COLL_POOL_SIZE",
                         "Collective pool size", 256, 10);
  uint32_t p2pPoolSize
    = getPoolSizeFromEnv("NCCL_INSPECTOR_P2P_POOL_SIZE",
                         "P2P pool size", 256, 10);
  uint32_t commPoolSize
    = getPoolSizeFromEnv("NCCL_INSPECTOR_COMM_POOL_SIZE",
                         "Comm pool size", 256, 10);

  return inspectorEventPoolInit(collPoolSize, p2pPoolSize, commPoolSize);
}

/*
 * Description:
 *
 *   Initializes dump thread and dump configuration from environment
 *   variables.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static inspectorResult_t initDumpThreadFromEnv() {
  const char* str = getenv("NCCL_INSPECTOR_DUMP_THREAD_ENABLE");
  int enable = str ? atoi(str) : 1;
  enableNcclInspectorDumpThread = enable == 0 ? false : true;

  str = getenv("NCCL_INSPECTOR_DUMP_VERBOSE");
  enable = str ? atoi(str) : 0;
  enableNcclInspectorDumpVerbose = enable == 0 ? false : true;

  str = getenv("NCCL_INSPECTOR_PROM_DUMP");
  enable = str ? atoi(str) : 0;
  enableNcclInspectorPromDump = enable == 0 ? false : true;

  str = getenv("NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS");
  if (str) {
    ncclInspectorDumpIntervalUsecs = strtoll(str, 0, 0);
  } else {
    ncclInspectorDumpIntervalUsecs = -1;
  }

  if (enableNcclInspectorPromDump && enableNcclInspectorDumpThread && ncclInspectorDumpIntervalUsecs >= 0) {
    ncclInspectorDumpIntervalUsecs
      = inspectorPromValidateInterval(ncclInspectorDumpIntervalUsecs);
  }

  str = getenv("NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES");
  ncclInspectorDumpMinSizeBytes = str ? strtoull(str, 0, 0) : 8192;

  ncclInspectorDumpCollRingSize
    = getRingSizeFromEnv("NCCL_INSPECTOR_DUMP_COLL_RING_SIZE", 1024);

  ncclInspectorDumpP2pRingSize
    = getRingSizeFromEnv("NCCL_INSPECTOR_DUMP_P2P_RING_SIZE", 1024);

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
  initP2pTrackingFromEnv();
  initKernelTimingFromEnv();
  INS_CHK(inspectorEventPoolInitFromEnv());
  INS_CHK(initDumpThreadFromEnv());
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
  commInfo->commNameStr[0] = '\0';
  if (commName && commName[0]) {
    snprintf(commInfo->commNameStr, sizeof(commInfo->commNameStr), "%s", commName);
  }
  commInfo->commName = commInfo->commNameStr;
  commInfo->commHash = commHash;
  inspectorCommGetHashStr(commHash, commInfo->commHashStr);
  commInfo->rank = rank;
  commInfo->nranks = nranks;
  commInfo->nnodes = nnodes;
  commInfo->dump_coll = false;
  commInfo->dump_p2p = false;
  commInfo->p2pSeqNum = 0;
  INS_CHK(inspectorRingInit(&commInfo->completedCollRing, ncclInspectorDumpCollRingSize,
                            sizeof(struct inspectorCompletedOpInfo)));
  INS_CHK(inspectorRingInit(&commInfo->completedP2pRing, ncclInspectorDumpP2pRingSize,
                            sizeof(struct inspectorCompletedOpInfo)));

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
  commInfoPtr->dump_coll = false;
  commInfoPtr->dump_p2p = false;
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
 *   Computes the algorithmic and bus bandwidth (in GB/s) for a completed
 *   collective or P2P operation.  For collectives the bus-bandwidth factor
 *   follows the standard NCCL formula described in:
 *   https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
 *   For P2P the factor is always 1 (point-to-point communication).
 *
 * Thread Safety:
 *
 *   This function does not perform any locking and assumes the caller
 *   ensures thread safety if required.
 *
 * Input:
 *
 *   commInfo - Pointer to the communicator info (for nranks, coll only).
 *
 *   op - Pointer to the inspectorCompletedOpInfo structure to update.
 *
 * Output:
 *   Updates the algoBwGbs and busBwGbs fields of op.
 *
 * Return:
 *   N.A. (void function)
 */
void inspectorComputeOpBw(struct inspectorCommInfo *commInfo,
                          struct inspectorCompletedOpInfo *op) {
  double timeInSec = op->execTimeUsecs / 1000000.0;
  double factor = 0.0;
  double trafficSize = 0.0;

  if (op->isP2p) {
    trafficSize = (double)op->msgSizeBytes;
    factor = 1.0;
  } else {
    switch (op->func) {
    case ncclFuncReduce:
    case ncclFuncBroadcast:
      trafficSize = (double)op->msgSizeBytes;
      factor = 1;
      break;
    case ncclFuncAllReduce:
      trafficSize = (double)op->msgSizeBytes;
      factor = ((double)(2 * (commInfo->nranks - 1))) / ((double)commInfo->nranks);
      break;
    case ncclFuncReduceScatter:
      trafficSize = (double)(op->msgSizeBytes * commInfo->nranks);
      factor = ((double)(commInfo->nranks - 1)) / ((double)commInfo->nranks);
      break;
    case ncclFuncAllGather:
    case ncclFuncAllGatherV:
      trafficSize = (double)(op->msgSizeBytes * commInfo->nranks);
      factor = ((double)(commInfo->nranks - 1)) / ((double)commInfo->nranks);
      break;
    case ncclFuncSendRecv:
    case ncclFuncSend:
    case ncclFuncRecv:
      trafficSize = (double)op->msgSizeBytes;
      factor = 1;
      break;
    default:
      trafficSize = 0;
      factor = 0.0;
    }
  }
  op->algoBwGbs = timeInSec != 0 ? (trafficSize / 1.0E9 / timeInSec) : 0;
  op->busBwGbs = op->algoBwGbs * factor;
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
void inspectorUpdateCollPerf(struct inspectorCompletedOpInfo *completedOp,
                             struct inspectorCollInfo *collInfo) {
  completedOp->isP2p = false;
  completedOp->func = ncclStringToFunc(collInfo->func);
  completedOp->sn = collInfo->sn;
  completedOp->msgSizeBytes = collInfo->msgSizeBytes;
  completedOp->execTimeUsecs =
    calculateMaxKernelExecTimeUsecs(collInfo, &completedOp->timingSource);
  completedOp->algo = collInfo->algo;
  completedOp->proto = collInfo->proto;
  completedOp->evtTrk = collInfo->collEvtTrk;
}

/*
 * Description:
 *
 *   Calculates the maximum kernel execution time across all kernel
 *   channels in a P2P operation, using GPU clock values when
 *   available and falling back to CPU timestamps when necessary.
 *
 * Thread Safety:
 *   Thread-safe (read-only operations on P2P info).
 *
 * Input:
 *   struct inspectorP2pInfo *p2pInfo - P2P operation info.
 *   inspectorTimingSource_t *timingSource - output parameter for timing source used.
 *
 * Output:
 *   timingSource is set to the timing method used.
 *
 * Return:
 *   uint64_t - maximum execution time in microseconds across all channels.
 */
static uint64_t calculateMaxKernelExecTimeUsecsP2p(struct inspectorP2pInfo *p2pInfo,
                                                   inspectorTimingSource_t *timingSource) {
  uint64_t maxExecTimeUsecs = 0;
  bool hasGpuTiming = false;

  for (uint32_t i = 0; i < p2pInfo->nChannels; i++) {
    struct inspectorKernelChInfo *kernelCh = &p2pInfo->kernelCh[i];
    uint64_t gpuExecTimeUsecs = calculateKernelGpuExecTimeUsecs(kernelCh);

    if (gpuExecTimeUsecs > 0) {
      hasGpuTiming = true;
      if (gpuExecTimeUsecs > maxExecTimeUsecs) {
        maxExecTimeUsecs = gpuExecTimeUsecs;
      }
    }
  }

  if (hasGpuTiming) {
    *timingSource = inspectorTimingSourceKernelGpu;
    return maxExecTimeUsecs;
  }

  // Fall back to CPU timestamps
  for (uint32_t i = 0; i < p2pInfo->nChannels; i++) {
    struct inspectorKernelChInfo *kernelCh = &p2pInfo->kernelCh[i];
    if (kernelCh->tsCompletedUsec > kernelCh->tsStartUsec) {
      uint64_t cpuExecTimeUsecs = kernelCh->tsCompletedUsec - kernelCh->tsStartUsec;
      if (cpuExecTimeUsecs > maxExecTimeUsecs) {
        maxExecTimeUsecs = cpuExecTimeUsecs;
      }
    }
  }

  if (maxExecTimeUsecs > 0) {
    *timingSource = inspectorTimingSourceKernelCpu;
    return maxExecTimeUsecs;
  }

  // Last resort: use P2P-level CPU timestamps
  if (p2pInfo->tsCompletedUsec > p2pInfo->tsStartUsec) {
    *timingSource = inspectorTimingSourceCollectiveCpu;
    return p2pInfo->tsCompletedUsec - p2pInfo->tsStartUsec;
  }

  *timingSource = inspectorTimingSourceCollectiveCpu;
  return 0;
}

void inspectorUpdateP2pPerf(struct inspectorCompletedOpInfo *completedOp,
                            struct inspectorP2pInfo *p2pInfo) {
  completedOp->isP2p = true;
  completedOp->func = ncclStringToFunc(p2pInfo->func);
  completedOp->sn = p2pInfo->sn;
  completedOp->msgSizeBytes = p2pInfo->msgSizeBytes;
  completedOp->peer = p2pInfo->peer;
  completedOp->execTimeUsecs =
    calculateMaxKernelExecTimeUsecsP2p(p2pInfo, &completedOp->timingSource);
  completedOp->evtTrk = p2pInfo->p2pEvtTrk;
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
  // Finalize event pools
  inspectorEventPoolFinalize();
  return inspectorSuccess;
}
