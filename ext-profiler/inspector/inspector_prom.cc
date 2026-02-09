#include "inspector_prom.h"
#include "inspector.h"
#include "inspector_cudawrap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include <cuda_runtime.h>

// External references from inspector.cc
extern struct inspectorState g_state;
extern inspectorResult_t inspectorCommInfoListFinalize(struct inspectorCommInfoList* commList);
extern const char* inspectorTimingSourceToString(inspectorTimingSource_t timingSource);

extern const char* ncclFuncToString(ncclFunc_t fn);

/*
 * Description:
 *
 *   Converts bytes to human-readable format (KB, MB, GB, etc.).
 *
 * Thread Safety:
 *
 *   Not thread-safe. Onus of thread safety is on the caller/owner of
 *   the buffer.
 *
 * Input:
 *   size_t bytes - number of bytes.
 *   char* output - output buffer for formatted string.
 *   size_t outputSize - size of output buffer.
 *
 * Output:
 *   Human-readable size string is written to buffer.
 *
 * Return:
 *   None.
 */
static void inspectorFormatHumanReadableSize(size_t bytes, char* output, size_t outputSize) {
  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  int unitIndex = 0;
  double size = (double)bytes;

  while (size >= 1024.0 && unitIndex < 4) {
    size /= 1024.0;
    unitIndex++;
  }

  if (unitIndex == 0) {
    // For bytes, show as integer
    snprintf(output, outputSize, "%zuB", bytes);
  } else {
    // For larger units, show with decimal precision
    snprintf(output, outputSize, "%.2f%s", size, units[unitIndex]);
  }
}

/*
 * Description:
 *
 *   Caches the static parts of Prometheus labels (hostname, job info, comm info)
 *   for a communicator. This should be called once when the communicator is created.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during comm initialization).
 *
 * Input:
 *   struct inspectorCommInfo* commInfo - communicator info.
 *
 * Output:
 *   commInfo->cachedStaticLabels is populated.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorPromCacheStaticLabels(struct inspectorCommInfo* commInfo) {
  char hostname[256];
  const char* jobId = getenv("SLURM_JOB_ID");
  const char* jobName = getenv("SLURM_JOB_NAME");

  gethostname(hostname, sizeof(hostname)-1);
  hostname[sizeof(hostname)-1] = '\0';

  char gpuDeviceStr[16];
  snprintf(gpuDeviceStr, sizeof(gpuDeviceStr), "GPU%d", commInfo->cudaDeviceId);

  int ret = snprintf(commInfo->cachedStaticLabels, sizeof(commInfo->cachedStaticLabels),
                     "comm_id=\"%s\",hostname=\"%s\",rank=\"%d\","
                     "slurm_job=\"%s\",slurm_job_id=\"%s\",nranks=\"%d\","
                     "n_nodes=\"%d\",gpu_device_id=\"%s\"",
                     commInfo->commHashStr,
                     hostname,
                     commInfo->rank,
                     jobName ? jobName : "unknown",
                     jobId ? jobId : "unknown",
                     commInfo->nranks,
                     commInfo->nnodes,
                     gpuDeviceStr);

  if (ret < 0 || (size_t)ret >= sizeof(commInfo->cachedStaticLabels)) {
    return inspectorMemoryError;
  }
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Formats labels for Prometheus metrics from communicator and collective info.
 *   Uses cached static labels and only adds dynamic parts (collective, timestamp, etc).
 *
 * Thread Safety:
 *
 *   Not thread-safe. Onus of thread safety is on the caller/owner of
 *   the buffer.
 *
 * Input:
 *   char* labels - output buffer for formatted labels.
 *   size_t labelSize - size of labels buffer.
 *   struct inspectorCommInfo* commInfo - communicator info (with cached static labels).
 *   struct inspectorCompletedCollInfo* collInfo - completed collective info.
 *
 * Output:
 *   Formatted labels string is written to buffer.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static inspectorResult_t inspectorPromGetLabels(char* labels,
                                                size_t labelSize,
                                                struct inspectorCommInfo* commInfo,
                                                struct inspectorCompletedCollInfo* collInfo) {
  char datetimeStr[32];
  INS_CHK(inspectorGetTimeUTC(datetimeStr, sizeof(datetimeStr)));

  char msgSizeStr[32];
  inspectorFormatHumanReadableSize(collInfo->msgSizeBytes, msgSizeStr, sizeof(msgSizeStr));


  int ret = snprintf(labels, labelSize,
                     "%s,collective=\"%s\",coll_sn=\"%lu\",timestamp=\"%s\",message_size=\"%s\"",
                     commInfo->cachedStaticLabels,
                     ncclFuncToString(collInfo->func),
                     collInfo->sn,
                     datetimeStr,
                     msgSizeStr);

  if (ret < 0 || (size_t)ret >= labelSize) {
    return inspectorMemoryError;
  }
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Generates GPU-specific Prometheus filename using pre-computed
 *   device UUID.  Each GPU gets its own file containing metrics from
 *   all communicators using that GPU.
 *
 * Thread Safety:
 *
 *   Not thread-safe. Onus of thread safety is on the caller/owner of
 *   the buffer.
 *
 * Input:
 *   const char* baseFilename - base output file path.
 *   const char* deviceUuidStr - pre-computed device UUID string.
 *   char* output - output buffer for filename.
 *   size_t outputSize - size of output buffer.
 *
 * Output:
 *   UUID-based filename is written to output buffer.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static inspectorResult_t inspectorPromGetFilename(const char* baseFilename,
                                                  const char* deviceUuidStr,
                                                  char* output,
                                                  size_t outputSize) {
  snprintf(output, outputSize,
           "%s/nccl_inspector_metrics_%s.prom",
           baseFilename, deviceUuidStr);

  return inspectorSuccess;
}


/*
 * Description:
 *
 *   Writes Prometheus metrics for a completed collective to a file.
 *
 * Thread Safety:
 *
 *   Not thread-safe. Onus of thread safety is on the caller/owner of
 *   the file handle.
 *
 * Input:
 *   const char* filename - output file path.
 *   struct inspectorCommInfo* commInfo - communicator info.
 *   struct inspectorCompletedCollInfo* collInfo - completed collective info.
 *
 * Output:
 *   Metrics are written to file.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static inspectorResult_t inspectorPromWriteCollInfo(FILE* file,
                                                    struct inspectorCommInfo* commInfo,
                                                    struct inspectorCompletedCollInfo* collInfo) {
  if (!file) {
    return inspectorFileOpenError;
  }

  char labels[512];
  memset(labels, 0, sizeof(labels));

  INS_CHK(inspectorPromGetLabels(labels,
                                 sizeof(labels),
                                 commInfo,
                                 collInfo));

  char buffer[2048];
  int written = snprintf(buffer, sizeof(buffer),
                         "nccl_algorithm_bandwidth_gbs{%s} %.6g\n"
                         "nccl_bus_bandwidth_gbs{%s} %.6g\n"
                         "nccl_collective_exec_time_microseconds{%s} %.6g\n",
                         labels, collInfo->algoBwGbs,
                         labels, collInfo->busBwGbs,
                         labels, (double)collInfo->execTimeUsecs);

  if (written < 0 || (size_t)written >= sizeof(buffer)) {
    return inspectorMemoryError;
  }

  if (fwrite(buffer, 1, written, file) != (size_t)written) {
    return inspectorFileOpenError;
  }

  fflush(file);
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Dumps the state of a single communicator to Prometheus format.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   struct inspectorCommInfo* commInfo - communicator info.
 *   const char* filename - output filename.
 *   bool* needs_writing - set to true if output was written.
 *
 * Output:
 *   Prometheus metrics are written to file if needed.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
static inspectorResult_t inspectorPromCommInfoDump(struct inspectorCommInfo* commInfo,
                                                   FILE* file,
                                                   bool* needs_writing) {
  *needs_writing = false;

  if (commInfo == nullptr || file == nullptr) {
    return inspectorSuccess;
  }

  struct inspectorCompletedCollInfo collInfo;
  memset(&collInfo, 0, sizeof(struct inspectorCompletedCollInfo));

  inspectorLockWr(&commInfo->guard);
  if (commInfo->dump) {
    *needs_writing = true;
    memcpy(&collInfo, &commInfo->completedCollInfo,
           sizeof(struct inspectorCompletedCollInfo));
    commInfo->dump = false; // Clear flag after reading
  }
  inspectorUnlockRWLock(&commInfo->guard);

  if (*needs_writing) {
    TRACE_INSPECTOR("NCCL Inspector: Writing metrics for comm %s directly",
                    commInfo->commHashStr);
    INS_CHK(inspectorPromWriteCollInfo(file,
                                       commInfo,
                                       &collInfo));
  }

  return inspectorSuccess;
}


/*
 * Description:
 *
 *   Dumps the state of all communicators in a commList to Prometheus format.
 *
 * Thread Safety:
 *   Thread-safe - acquires necessary locks to iterate through communicators.
 *
 * Input:
 *   struct inspectorCommInfoList* commList - list of communicators.
 *   const char* output_root - base output directory.
 *
 * Output:
 *   Prometheus metrics are written to UUID-named file.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorPromCommInfoListDump(struct inspectorCommInfoList* commList,
                                                const char* output_root,
                                                struct inspectorDumpThread* dumpThread) {
  INS_CHK(inspectorLockRd(&commList->guard));
  inspectorResult_t res = inspectorSuccess;

  if (commList->ncomms > 0) {
    uint32_t processed = 0;
    uint64_t currentTime = inspectorGetTime();

    for (struct inspectorCommInfo* itr = commList->comms;
         itr != nullptr;
         itr = itr->next) {
      bool needs_writing;

      // Get filename for this specific communicator's device
      char filename[1024];
      INS_CHK_GOTO(inspectorPromGetFilename(output_root,
                                            itr->deviceUuidStr,
                                            filename,
                                            sizeof(filename)),
                   res,
                   exit);

      FILE* file = dumpThread ? dumpThread->getOrCreateFileHandle(itr->deviceUuidStr,
                                                                  filename,
                                                                  currentTime) : NULL;
      if (!file) {
        INFO_INSPECTOR("NCCL Inspector: Failed to get file handle for device UUID %s, file %s",
                      itr->deviceUuidStr, filename);
        continue;
      }

      INS_CHK_GOTO(inspectorPromCommInfoDump(itr, file, &needs_writing),
                   res, exit);

      if (needs_writing) {
        processed++;
        TRACE_INSPECTOR(
          "NCCL Inspector: Processed comm %u for CUDA device (rank %d)",
          processed, itr->rank);
      }
    }
    TRACE_INSPECTOR(
      "NCCL Inspector: Completed dump across devices, flushed %u/%u communicators",
      processed, commList->ncomms);
  }

exit:
  INS_CHK(inspectorUnlockRWLock(&commList->guard));
  return res;
}

/*
 * Description:
 *
 *   Validates and adjusts the dump interval for Prometheus-specific requirements.
 *   Prometheus requires a minimum 30-second interval to match node exporter poll interval.
 *
 * Thread Safety:
 *   Thread-safe.
 *
 * Input:
 *   uint64_t interval - raw interval in microseconds from environment variable.
 *
 * Output:
 *   None.
 *
 * Return:
 *   uint64_t - validated interval in microseconds.
 */
uint64_t inspectorPromValidateInterval(uint64_t interval) {
  const uint64_t MIN_PROM_INTERVAL = 30000000;

  if (interval > 0 && interval < MIN_PROM_INTERVAL) {
    INFO_INSPECTOR(
      "NCCL Inspector: Prometheus dump requires minimum interval of %lu microseconds "
      "to match node exporter poll interval, but got %lu. Setting to minimum.",
      MIN_PROM_INTERVAL, interval);
    return MIN_PROM_INTERVAL;
  } else if (interval == 0) {
    INFO_INSPECTOR(
      "NCCL Inspector: Using default interval of %lu microseconds for Prometheus dump",
      MIN_PROM_INTERVAL);
    return MIN_PROM_INTERVAL;
  }

  return interval;
}
