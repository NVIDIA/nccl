/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alloc.h"
#include "checks.h"
#include "cudawrap.h"
#include "diagnostics_checks.h"
#include "diagnostics_checks_common.h"
#include "nvmlwrap.h"
#include "param.h"

// *************************************************************************
// GPU model and count consistency check.
// *************************************************************************
#define RAS_DIAG_GPU_MODEL_NAME_LEN NVML_DEVICE_NAME_BUFFER_SIZE
#define RAS_DIAG_GPU_MODEL_UNKNOWN "unknown"

struct rasDiagnosticsGpuModelData {
  uint8_t nGpus;
  char model[RAS_DIAG_GPU_MODEL_NAME_LEN];
};

static ncclResult_t rasDiagnosticsGpuModelFillLocalData(const struct rasDiagnosticsCommSnapshot* comm,
                                                        void* checkData) {
  struct rasDiagnosticsGpuModelData* gpuData = (struct rasDiagnosticsGpuModelData*)checkData;

  unsigned int nDev = 0;

  gpuData->nGpus = 0;
  gpuData->model[0] = '\0';

  if (ncclNvmlDeviceGetCount(&nDev) == ncclSuccess) gpuData->nGpus = (uint8_t)nDev;
  if (comm->nvmlDev >= 0 && comm->nvmlDev < ncclNvmlDeviceCount) {
    nvmlDevice_t device;
    if (ncclNvmlDeviceGetHandleByIndex((unsigned int)comm->nvmlDev, &device) == ncclSuccess) {
      if (ncclNvmlDeviceGetName(device, gpuData->model, sizeof(gpuData->model)) != ncclSuccess) {
        gpuData->model[0] = '\0';
      }
    }
  }

  if (gpuData->model[0] == '\0') snprintf(gpuData->model, sizeof(gpuData->model), "%s", RAS_DIAG_GPU_MODEL_UNKNOWN);
  gpuData->model[sizeof(gpuData->model) - 1] = '\0';
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsGpuModelCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsGpuModelData),
                                              rasDiagnosticsGpuModelFillLocalData, data));
  return ncclSuccess;
}

static const struct rasDiagnosticsGpuModelData* rasDiagnosticsGpuModelDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsGpuModelData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

ncclResult_t rasDiagnosticsGpuModelSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsGpuModelData));
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics GPU model check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics GPU model check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics GPU model check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    const struct rasDiagnosticsGpuModelData* startData = rasDiagnosticsGpuModelDataFromRecord(startRecord);
    int expectedRank = startRank->commRank;
    int commNRanks = startRank->commNRanks;
    int expectedNGpus = startData->nGpus;
    int countMismatchRanks[RAS_DIAG_RANK_SET_MAX];
    int modelMismatchRanks[RAS_DIAG_RANK_SET_MAX];
    int nCountMismatchStored = 0, nCountMismatch = 0;
    int nModelMismatchStored = 0, nModelMismatch = 0;
    const char* expectedModel = startData->model;
    int end = start + 1;

    while (end < nRecords) {
      const char* record = records + end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsGpuModelData* gpuData = rasDiagnosticsGpuModelDataFromRecord(record);

      if (rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      if (gpuData->nGpus != expectedNGpus) {
        if (nCountMismatchStored < RAS_DIAG_RANK_SET_MAX) countMismatchRanks[nCountMismatchStored++] = rank->commRank;
        nCountMismatch++;
      }
      if (strncmp(gpuData->model, expectedModel, RAS_DIAG_GPU_MODEL_NAME_LEN) != 0) {
        if (nModelMismatchStored < RAS_DIAG_RANK_SET_MAX) modelMismatchRanks[nModelMismatchStored++] = rank->commRank;
        nModelMismatch++;
      }
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "GPU inventory", startRank, end - start), ret, exit);
    } else if (nCountMismatch == 0 && nModelMismatch == 0) {
      bool countKnown = expectedNGpus > 0;
      bool modelKnown = strncmp(expectedModel, RAS_DIAG_GPU_MODEL_UNKNOWN, RAS_DIAG_GPU_MODEL_NAME_LEN) != 0;
      if (countKnown && modelKnown) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                                           "GPU inventory: %dx %s per node consistent across %d ranks in comm 0x%lx",
                                           expectedNGpus, expectedModel, commNRanks, startRank->commId.commHash),
                      ret, exit);
      } else if (countKnown) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "GPU inventory: %d GPUs per node consistent across %d ranks in comm 0x%lx, "
                                           "GPU model unavailable via NVML",
                                           expectedNGpus, commNRanks, startRank->commId.commHash),
                      ret, exit);
      } else if (modelKnown) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "GPU inventory: %s per node consistent across %d ranks in comm 0x%lx, "
                                           "GPU count unavailable via NVML",
                                           expectedModel, commNRanks, startRank->commId.commHash),
                      ret, exit);
      } else {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "GPU inventory: unavailable via NVML across %d ranks in comm 0x%lx",
                                           commNRanks, startRank->commId.commHash),
                      ret, exit);
      }
    } else {
      if (nCountMismatch > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), countMismatchRanks, nCountMismatchStored, nCountMismatch);
        NCCLCHECKGOTO(
          rasDiagnosticsReport(
            reporter, RAS_DIAG_TAG_INFO,
            "GPU inventory: count mismatch across %d ranks in comm 0x%lx, rank(s) %s differ from rank %d (%d)",
            commNRanks, startRank->commId.commHash, rankSet, expectedRank, expectedNGpus),
          ret, exit);
      }
      if (nModelMismatch > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), modelMismatchRanks, nModelMismatchStored, nModelMismatch);
        NCCLCHECKGOTO(
          rasDiagnosticsReport(
            reporter, RAS_DIAG_TAG_INFO,
            "GPU inventory: model mismatch across %d ranks in comm 0x%lx, rank(s) %s differ from rank %d (%s)",
            commNRanks, startRank->commId.commHash, rankSet, expectedRank, expectedModel),
          ret, exit);
      }
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}

// *************************************************************************
// Version consistency checks.
// *************************************************************************
static const void* rasDiagnosticsVersionDataFromRecord(const char* record) {
  return record + sizeof(struct rasDiagnosticsRankHeader);
}

// format converts a check-specific value to text and returns whether it was collected successfully.
static ncclResult_t rasDiagnosticsVersionSummarize(const struct rasDiagnosticsReporter* reporter, const char* data,
                                                   int nData, const char* checkLabel, size_t dataSize,
                                                   bool (*format)(const void* data, char* buf, size_t bufLen)) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(dataSize);
  int nRecords;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics %s check received invalid reporter", checkLabel);
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics %s check received null data with size %d", checkLabel, nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics %s check received malformed data size %d", checkLabel, nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    const void* expectedVersion = rasDiagnosticsVersionDataFromRecord(startRecord);
    int expectedRank = startRank->commRank;
    int commNRanks = startRank->commNRanks;
    int mismatchRanks[RAS_DIAG_RANK_SET_MAX];
    int nMismatchStored = 0, nMismatch = 0;
    int end = start + 1;

    while (end < nRecords) {
      const char* record = records + end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const void* version = rasDiagnosticsVersionDataFromRecord(record);

      if (rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      // Version PODs are fully initialized, so their byte representation can be compared directly.
      if (memcmp(version, expectedVersion, dataSize) != 0) {
        if (nMismatchStored < RAS_DIAG_RANK_SET_MAX) mismatchRanks[nMismatchStored++] = rank->commRank;
        nMismatch++;
      }
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, checkLabel, startRank, end - start), ret, exit);
    } else if (nMismatch == 0) {
      char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
      if (!format(expectedVersion, version, sizeof(version))) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO, "%s: %s across %d ranks in comm 0x%lx",
                                           checkLabel, version, commNRanks, startRank->commId.commHash),
                      ret, exit);
      } else {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK, "%s: %s consistent across %d ranks in comm 0x%lx",
                                           checkLabel, version, commNRanks, startRank->commId.commHash),
                      ret, exit);
      }
    } else {
      char rankSet[128];
      char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
      (void)format(expectedVersion, version, sizeof(version));
      rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), mismatchRanks, nMismatchStored, nMismatch);
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "%s: mismatch across %d ranks in comm 0x%lx, "
                                         "rank(s) %s differ from rank %d (%s)",
                                         checkLabel, commNRanks, startRank->commId.commHash, rankSet, expectedRank,
                                         version),
                    ret, exit);
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}

// CUDA driver version.
#define RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN UINT32_MAX

struct rasDiagnosticsCudaDriverVersionData {
  uint32_t version;
};

static bool rasDiagnosticsCudaDriverVersionFormat(const void* data, char* buf, size_t bufLen) {
  uint32_t version = ((const struct rasDiagnosticsCudaDriverVersionData*)data)->version;
  if (version == RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN) snprintf(buf, bufLen, "unavailable");
  else if (version == 0) snprintf(buf, bufLen, "no driver installed");
  else snprintf(buf, bufLen, "%u", version);
  return version != RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN && version != 0;
}

static ncclResult_t rasDiagnosticsCudaDriverVersionFillLocalData(const struct rasDiagnosticsCommSnapshot* comm,
                                                                 void* checkData) {
  struct rasDiagnosticsCudaDriverVersionData* versionData = (struct rasDiagnosticsCudaDriverVersionData*)checkData;
  int version = 0;

  (void)comm;
  versionData->version = RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN;
  if (ncclCudaDriverVersion(&version) == ncclSuccess) {
    versionData->version = (uint32_t)version;
  }
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsCudaDriverVersionCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                         struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsCudaDriverVersionData),
                                              rasDiagnosticsCudaDriverVersionFillLocalData, data));
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsCudaDriverVersionSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  (void)ctx;
  return rasDiagnosticsVersionSummarize(reporter, data, nData, "CUDA driver version",
                                        sizeof(struct rasDiagnosticsCudaDriverVersionData),
                                        rasDiagnosticsCudaDriverVersionFormat);
}

// NVIDIA graphics driver version.
struct rasDiagnosticsNvidiaDriverVersionData {
  char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
};

static bool rasDiagnosticsNvidiaDriverVersionFormat(const void* data, char* buf, size_t bufLen) {
  const struct rasDiagnosticsNvidiaDriverVersionData* versionData =
    (const struct rasDiagnosticsNvidiaDriverVersionData*)data;
  if (versionData->version[0] == '\0') snprintf(buf, bufLen, "unavailable via NVML");
  else snprintf(buf, bufLen, "%.*s", NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE, versionData->version);
  return versionData->version[0] != '\0';
}

static ncclResult_t rasDiagnosticsNvidiaDriverVersionFillLocalData(const struct rasDiagnosticsCommSnapshot* comm,
                                                                   void* checkData) {
  struct rasDiagnosticsNvidiaDriverVersionData* versionData = (struct rasDiagnosticsNvidiaDriverVersionData*)checkData;

  (void)comm;
  memset(versionData, 0, sizeof(*versionData));
  if (ncclNvmlSystemGetDriverVersion(versionData->version, sizeof(versionData->version)) != ncclSuccess)
    memset(versionData, 0, sizeof(*versionData));
  versionData->version[sizeof(versionData->version) - 1] = '\0';
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsNvidiaDriverVersionCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                           struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsNvidiaDriverVersionData),
                                              rasDiagnosticsNvidiaDriverVersionFillLocalData, data));
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsNvidiaDriverVersionSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  (void)ctx;
  return rasDiagnosticsVersionSummarize(reporter, data, nData, "NVIDIA graphics driver version",
                                        sizeof(struct rasDiagnosticsNvidiaDriverVersionData),
                                        rasDiagnosticsNvidiaDriverVersionFormat);
}

// *************************************************************************
// Volatile uncorrected ECC counter check.
// *************************************************************************
// Uncorrected volatile ECC errors are always flagged; corrected errors are only flagged once they reach
// NCCL_DIAGNOSTICS_ECC_THRESHOLD (0 disables corrected reporting).
NCCL_PARAM(DiagnosticsEccThreshold, "DIAGNOSTICS_ECC_THRESHOLD", 0);

struct rasDiagnosticsEccData {
  uint64_t correctedSram;
  uint64_t uncorrectedSram;
  uint64_t correctedDram;
  uint64_t uncorrectedDram;
  uint8_t available; // 1 only if NVML reported every ECC counter for this rank; 0 otherwise.
};

// Reads one volatile ECC counter. Returns true only if NVML answered.
static bool rasDiagnosticsEccReadCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType,
                                         nvmlMemoryLocation_t location, uint64_t* out) {
  unsigned long long value = 0;
  if (ncclNvmlDeviceGetMemoryErrorCounter(device, errorType, NVML_VOLATILE_ECC, location, &value) != ncclSuccess)
    return false;
  *out = value;
  return true;
}

static ncclResult_t rasDiagnosticsEccFillLocalData(const struct rasDiagnosticsCommSnapshot* comm, void* checkData) {
  struct rasDiagnosticsEccData* eccData = (struct rasDiagnosticsEccData*)checkData;

  eccData->correctedSram = 0;
  eccData->uncorrectedSram = 0;
  eccData->correctedDram = 0;
  eccData->uncorrectedDram = 0;
  eccData->available = 0;

  if (comm->nvmlDev >= 0 && comm->nvmlDev < ncclNvmlDeviceCount) {
    nvmlDevice_t device;
    if (ncclNvmlDeviceGetHandleByIndex((unsigned int)comm->nvmlDev, &device) == ncclSuccess) {
      // Available only if NVML answered every counter, so a failed read is never reported as a zero.
      bool ok = true;
      ok &= rasDiagnosticsEccReadCounter(device, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_MEMORY_LOCATION_SRAM,
                                         &eccData->correctedSram);
      ok &= rasDiagnosticsEccReadCounter(device, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_MEMORY_LOCATION_SRAM,
                                         &eccData->uncorrectedSram);
      ok &= rasDiagnosticsEccReadCounter(device, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_MEMORY_LOCATION_DRAM,
                                         &eccData->correctedDram);
      ok &= rasDiagnosticsEccReadCounter(device, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_MEMORY_LOCATION_DRAM,
                                         &eccData->uncorrectedDram);
      eccData->available = ok ? 1 : 0;
    }
  }
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsEccCollectLocal(const struct rasDiagnosticsContext* ctx,
                                           struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsEccData), rasDiagnosticsEccFillLocalData,
                                              data));
  return ncclSuccess;
}

static const struct rasDiagnosticsEccData* rasDiagnosticsEccDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsEccData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

ncclResult_t rasDiagnosticsEccSummarize(const struct rasDiagnosticsContext* ctx,
                                        const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsEccData));
  const unsigned long long threshold = (unsigned long long)ncclParamDiagnosticsEccThreshold();
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics ECC check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics ECC check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics ECC check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    int commNRanks = startRank->commNRanks;
    int uncorrectedRanks[RAS_DIAG_RANK_SET_MAX];
    int correctedRanks[RAS_DIAG_RANK_SET_MAX];
    int nUncorrectedStored = 0, nUncorrected = 0;
    int nCorrectedStored = 0, nCorrected = 0;
    int nAvailable = 0;
    unsigned long long worstUncorrected = 0, worstCorrected = 0;
    int end = start;

    while (end < nRecords) {
      const char* record = records + end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsEccData* eccData;
      unsigned long long uncorrected, corrected;

      if (end > start && rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      eccData = rasDiagnosticsEccDataFromRecord(record);
      if (eccData->available) nAvailable++;
      uncorrected = eccData->uncorrectedSram + eccData->uncorrectedDram;
      corrected = eccData->correctedSram + eccData->correctedDram;
      if (uncorrected > 0) {
        if (nUncorrectedStored < RAS_DIAG_RANK_SET_MAX) uncorrectedRanks[nUncorrectedStored++] = rank->commRank;
        nUncorrected++;
        if (uncorrected > worstUncorrected) worstUncorrected = uncorrected;
      }
      if (threshold > 0 && corrected >= threshold) {
        if (nCorrectedStored < RAS_DIAG_RANK_SET_MAX) correctedRanks[nCorrectedStored++] = rank->commRank;
        nCorrected++;
        if (corrected > worstCorrected) worstCorrected = corrected;
      }
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "ECC", startRank, end - start), ret, exit);
    } else if (nAvailable == 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ECC: unavailable via NVML across %d ranks in comm 0x%lx", commNRanks,
                                         startRank->commId.commHash),
                    ret, exit);
    } else if (nUncorrected == 0 && nCorrected == 0) {
      if (nAvailable == commNRanks) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                                           "ECC: no uncorrected volatile errors across %d ranks in comm 0x%lx",
                                           commNRanks, startRank->commId.commHash),
                      ret, exit);
      } else {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "ECC: no uncorrected volatile errors across %d of %d ranks in comm 0x%lx "
                                           "(ECC counters unavailable via NVML on %d ranks)",
                                           nAvailable, commNRanks, startRank->commId.commHash, commNRanks - nAvailable),
                      ret, exit);
      }
    } else {
      if (nUncorrected > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), uncorrectedRanks, nUncorrectedStored, nUncorrected);
        NCCLCHECKGOTO(
          rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                               "ECC: uncorrected volatile errors on rank(s) %s (worst=%llu) across %d ranks "
                               "in comm 0x%lx",
                               rankSet, worstUncorrected, commNRanks, startRank->commId.commHash),
          ret, exit);
      }
      if (nCorrected > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), correctedRanks, nCorrectedStored, nCorrected);
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "ECC: corrected volatile errors at or above threshold %llu on rank(s) %s "
                                           "(worst=%llu) across %d ranks in comm 0x%lx",
                                           threshold, rankSet, worstCorrected, commNRanks, startRank->commId.commHash),
                      ret, exit);
      }
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}

// *************************************************************************
// Per-NVLink presence, operational state and speed check.
// *************************************************************************
struct rasDiagnosticsNvLinkData {
  uint8_t nLinks;
  uint8_t nInactive;
  uint32_t minSpeedMBps;
  uint32_t maxSpeedMBps;
};

static ncclResult_t rasDiagnosticsNvLinkFillLocalData(const struct rasDiagnosticsCommSnapshot* comm, void* checkData) {
  struct rasDiagnosticsNvLinkData* nvlData = (struct rasDiagnosticsNvLinkData*)checkData;
  nvmlFieldValue_t linkCount = {};
  unsigned int nUnreadable = 0;
  nvmlDevice_t device;

  *nvlData = {};

  if (comm->nvmlDev >= 0 && comm->nvmlDev < ncclNvmlDeviceCount &&
      ncclNvmlDeviceGetHandleByIndex((unsigned int)comm->nvmlDev, &device) == ncclSuccess) {
    linkCount.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;
    if (ncclNvmlDeviceGetFieldValues(device, 1, &linkCount) == ncclSuccess && linkCount.nvmlReturn == NVML_SUCCESS)
      nvlData->nLinks = (uint8_t)linkCount.value.uiVal;

    for (unsigned int link = 0; link < nvlData->nLinks; link++) {
      nvmlFieldValue_t values[2] = {};
      unsigned int speedMBps = 0;

      values[0].fieldId = NVML_FI_DEV_NVLINK_GET_STATE;
      values[0].scopeId = link;
      values[1].fieldId = NVML_FI_DEV_NVLINK_GET_SPEED;
      values[1].scopeId = link;
      // A link counts as up only where NVML confirms it enabled at a known speed, as topology detection requires.
      if (ncclNvmlDeviceGetFieldValues(device, 2, values) == ncclSuccess && values[0].nvmlReturn == NVML_SUCCESS) {
        if ((nvmlEnableState_t)values[0].value.uiVal == NVML_FEATURE_ENABLED && values[1].nvmlReturn == NVML_SUCCESS)
          speedMBps = values[1].value.uiVal;
      } else {
        nUnreadable++;
      }

      if (speedMBps == 0) {
        nvlData->nInactive++;
        continue;
      }
      if (nvlData->minSpeedMBps == 0 || speedMBps < nvlData->minSpeedMBps) nvlData->minSpeedMBps = speedMBps;
      if (speedMBps > nvlData->maxSpeedMBps) nvlData->maxSpeedMBps = speedMBps;
    }
    // Links whose state NVML cannot report leave nothing to check, so treat the device as having no NVLink.
    if (nUnreadable == nvlData->nLinks) *nvlData = {};
  }
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsNvLinkCollectLocal(const struct rasDiagnosticsContext* ctx,
                                              struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsNvLinkData),
                                              rasDiagnosticsNvLinkFillLocalData, data));
  return ncclSuccess;
}

static const struct rasDiagnosticsNvLinkData* rasDiagnosticsNvLinkDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsNvLinkData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

ncclResult_t rasDiagnosticsNvLinkSummarize(const struct rasDiagnosticsContext* ctx,
                                           const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNvLinkData));
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics NVLink check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics NVLink check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics NVLink check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    // Sorted by rank, so the group's first record is its lowest rank; use it as the reference.
    const struct rasDiagnosticsNvLinkData* refData = rasDiagnosticsNvLinkDataFromRecord(startRecord);
    int commNRanks = startRank->commNRanks;
    int countMismatchRanks[RAS_DIAG_RANK_SET_MAX];
    int inactiveRanks[RAS_DIAG_RANK_SET_MAX];
    int speedMismatchRanks[RAS_DIAG_RANK_SET_MAX];
    int nCountMismatchStored = 0, nCountMismatch = 0;
    int nInactiveStored = 0, nInactive = 0;
    int nSpeedMismatchStored = 0, nSpeedMismatch = 0;
    int nWithLinks = 0;
    int end = start;

    while (end < nRecords) {
      const char* record = records + end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsNvLinkData* nvlData;

      if (end > start && rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      nvlData = rasDiagnosticsNvLinkDataFromRecord(record);
      if (nvlData->nLinks > 0) nWithLinks++;
      if (nvlData->nLinks != refData->nLinks) {
        if (nCountMismatchStored < RAS_DIAG_RANK_SET_MAX) countMismatchRanks[nCountMismatchStored++] = rank->commRank;
        nCountMismatch++;
      }
      if (nvlData->nInactive > 0) {
        if (nInactiveStored < RAS_DIAG_RANK_SET_MAX) inactiveRanks[nInactiveStored++] = rank->commRank;
        nInactive++;
      }
      // A rank with no link up reports no speed; the inactive-link report covers it instead.
      if (nvlData->minSpeedMBps != nvlData->maxSpeedMBps ||
          (nvlData->maxSpeedMBps > 0 && refData->maxSpeedMBps > 0 && nvlData->maxSpeedMBps != refData->maxSpeedMBps)) {
        if (nSpeedMismatchStored < RAS_DIAG_RANK_SET_MAX) speedMismatchRanks[nSpeedMismatchStored++] = rank->commRank;
        nSpeedMismatch++;
      }
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "NVLink", startRank, end - start), ret, exit);
    } else if (nWithLinks == 0) {
      // Nothing to check: either no device exposes NVLink (e.g. PCIe-only), or NVML cannot report it.
    } else if (nCountMismatch == 0 && nInactive == 0 && nSpeedMismatch == 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                                         "NVLink: %d links per GPU, all active at consistent speed across %d ranks "
                                         "in comm 0x%lx",
                                         refData->nLinks, commNRanks, startRank->commId.commHash),
                    ret, exit);
    } else {
      if (nCountMismatch > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), countMismatchRanks, nCountMismatchStored, nCountMismatch);
        NCCLCHECKGOTO(
          rasDiagnosticsReport(
            reporter, RAS_DIAG_TAG_INFO,
            "NVLink: link-count mismatch across %d ranks in comm 0x%lx, rank(s) %s differ from rank %d (%d)",
            commNRanks, startRank->commId.commHash, rankSet, startRank->commRank, refData->nLinks),
          ret, exit);
      }
      if (nInactive > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), inactiveRanks, nInactiveStored, nInactive);
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "NVLink: inactive link(s) on rank(s) %s across %d ranks in comm 0x%lx",
                                           rankSet, commNRanks, startRank->commId.commHash),
                      ret, exit);
      }
      if (nSpeedMismatch > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), speedMismatchRanks, nSpeedMismatchStored, nSpeedMismatch);
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "NVLink: inconsistent link speeds on rank(s) %s across %d ranks "
                                           "in comm 0x%lx",
                                           rankSet, commNRanks, startRank->commId.commHash),
                      ret, exit);
      }
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}
