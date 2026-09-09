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
#include "diagnostics_checks.h"
#include "diagnostics_checks_common.h"

// *************************************************************************
// NCCL_* environment variable consistency check.
// *************************************************************************
#define RAS_DIAG_ENV_BYTES 16384
#define RAS_DIAG_ENV_KEY_DISPLAY_BYTES 256
#define RAS_DIAG_ENV_VALUE_DISPLAY_BYTES 256

extern "C" char** environ;

struct rasDiagnosticsNcclEnvData {
  uint16_t bytesUsed;
  uint8_t truncated;
  char data[RAS_DIAG_ENV_BYTES];
};

struct rasDiagnosticsEnvKey {
  const char* key;
  uint16_t len;
};

static int rasDiagnosticsEnvEntryCompare(const void* p1, const void* p2) {
  const char* e1 = *(char* const*)p1;
  const char* e2 = *(char* const*)p2;
  return strcmp(e1, e2);
}

static int rasDiagnosticsEnvKeyCompare(const void* p1, const void* p2) {
  const struct rasDiagnosticsEnvKey* k1 = (const struct rasDiagnosticsEnvKey*)p1;
  const struct rasDiagnosticsEnvKey* k2 = (const struct rasDiagnosticsEnvKey*)p2;
  size_t minLen = k1->len < k2->len ? k1->len : k2->len;
  int cmp = strncmp(k1->key, k2->key, minLen);

  if (cmp != 0) return cmp;
  return (k1->len < k2->len ? -1 : (k1->len > k2->len ? 1 : 0));
}

static bool rasDiagnosticsEnvNextEntry(const struct rasDiagnosticsNcclEnvData* envData, int* offset, const char** entry,
                                       size_t* entryLen) {
  const char* start;
  const void* end;

  if (envData == nullptr || offset == nullptr || entry == nullptr || entryLen == nullptr) return false;
  if (envData->bytesUsed > RAS_DIAG_ENV_BYTES || *offset < 0 || *offset >= envData->bytesUsed) return false;

  start = envData->data + *offset;
  end = memchr(start, '\0', (size_t)envData->bytesUsed - (size_t)*offset);
  if (end == nullptr) return false;

  *entry = start;
  *entryLen = (const char*)end - start;
  *offset += (int)*entryLen + 1;
  return true;
}

static const char* rasDiagnosticsEnvFindValue(const struct rasDiagnosticsNcclEnvData* envData, const char* key,
                                              size_t keyLen) {
  int offset = 0;
  const char* entry;
  size_t entryLen;

  while (rasDiagnosticsEnvNextEntry(envData, &offset, &entry, &entryLen)) {
    const char* eq = (const char*)memchr(entry, '=', entryLen);
    if (eq == nullptr) continue;
    if ((size_t)(eq - entry) == keyLen && strncmp(entry, key, keyLen) == 0) return eq + 1;
  }
  return nullptr;
}

static bool rasDiagnosticsEnvValuesEqual(const char* v1, const char* v2) {
  // nullptr means the environment variable is not set on that rank.
  if (v1 == nullptr || v2 == nullptr) return v1 == v2;
  return strcmp(v1, v2) == 0;
}

static const char* rasDiagnosticsEnvKeyDisplay(const struct rasDiagnosticsEnvKey* key, char* buf, size_t bufLen) {
  size_t limit;
  size_t i;

  if (bufLen == 0) return "";
  limit = bufLen > 4 ? bufLen - 4 : bufLen - 1;
  for (i = 0; i < limit && i < key->len; i++) {
    unsigned char c = (unsigned char)key->key[i];
    buf[i] = (c < 32 || c == 127) ? '?' : key->key[i];
  }
  if (i < key->len && bufLen > 4) {
    memcpy(buf + i, "...", 4);
  } else {
    buf[i] = '\0';
  }
  return buf;
}

static const char* rasDiagnosticsEnvValueDisplay(const char* value, char* buf, size_t bufLen) {
  size_t limit;
  size_t i;

  if (bufLen == 0) return "";
  if (value == nullptr) {
    snprintf(buf, bufLen, "(unset)");
    return buf;
  }

  limit = bufLen > 4 ? bufLen - 4 : bufLen - 1;
  for (i = 0; i < limit && value[i] != '\0'; i++) {
    unsigned char c = (unsigned char)value[i];
    buf[i] = (c < 32 || c == 127) ? '?' : value[i];
  }
  if (value[i] != '\0' && bufLen > 4) {
    memcpy(buf + i, "...", 4);
  } else {
    buf[i] = '\0';
  }
  return buf;
}

static ncclResult_t rasDiagnosticsNcclEnvReportMismatch(const struct rasDiagnosticsReporter* reporter,
                                                        const struct rasDiagnosticsRankHeader* startRank,
                                                        const char* keyDisplay) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "NCCL environment: mismatch across %d ranks in comm 0x%lx for %s", startRank->commNRanks,
                              startRank->commId.commHash, keyDisplay);
}

static ncclResult_t rasDiagnosticsNcclEnvReportValue(const struct rasDiagnosticsReporter* reporter,
                                                     const char* keyDisplay, const char* valueDisplay,
                                                     const char* rankSet) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO, "NCCL environment: %s=%s on rank(s) %s", keyDisplay,
                              valueDisplay, rankSet);
}

static ncclResult_t rasDiagnosticsNcclEnvReportTruncated(const struct rasDiagnosticsReporter* reporter,
                                                         int nTruncated) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "NCCL environment: %d rank(s) had >%d bytes of NCCL_* env vars; comparison may be "
                              "partial",
                              nTruncated, RAS_DIAG_ENV_BYTES);
}

static ncclResult_t rasDiagnosticsNcclEnvReportConsistent(const struct rasDiagnosticsReporter* reporter, int commNRanks,
                                                          const struct rasCommId* commId) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                              "NCCL environment: NCCL_* env vars consistent across %d ranks in comm 0x%lx", commNRanks,
                              commId->commHash);
}

static ncclResult_t rasDiagnosticsNcclEnvReportMismatchSummary(const struct rasDiagnosticsReporter* reporter,
                                                               int nMismatch, const struct rasCommId* commId) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "NCCL environment: %d NCCL_* env var(s) differ across ranks in comm 0x%lx", nMismatch,
                              commId->commHash);
}

static const struct rasDiagnosticsNcclEnvData* rasDiagnosticsNcclEnvDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsNcclEnvData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

static bool rasDiagnosticsNcclEnvDataValid(const struct rasDiagnosticsNcclEnvData* envData) {
  if (envData->bytesUsed > RAS_DIAG_ENV_BYTES) return false;
  return envData->bytesUsed == 0 || envData->data[envData->bytesUsed - 1] == '\0';
}

static int rasDiagnosticsEnvCountCandidateKeys(const char* records, size_t recordStride, int start, int end) {
  int nKeys = 0;

  for (int i = start; i < end; i++) {
    const struct rasDiagnosticsNcclEnvData* envData =
      rasDiagnosticsNcclEnvDataFromRecord(records + (size_t)i * recordStride);
    int offset = 0;
    const char* entry;
    size_t entryLen;

    while (rasDiagnosticsEnvNextEntry(envData, &offset, &entry, &entryLen)) {
      if (memchr(entry, '=', entryLen) != nullptr) nKeys++;
    }
  }
  return nKeys;
}

static void rasDiagnosticsEnvFillCandidateKeys(const char* records, size_t recordStride, int start, int end,
                                               struct rasDiagnosticsEnvKey* keys) {
  int nKeys = 0;

  for (int i = start; i < end; i++) {
    const struct rasDiagnosticsNcclEnvData* envData =
      rasDiagnosticsNcclEnvDataFromRecord(records + (size_t)i * recordStride);
    int offset = 0;
    const char* entry;
    size_t entryLen;

    while (rasDiagnosticsEnvNextEntry(envData, &offset, &entry, &entryLen)) {
      const char* eq = (const char*)memchr(entry, '=', entryLen);
      if (eq == nullptr) continue;
      keys[nKeys].key = entry;
      keys[nKeys].len = (uint16_t)(eq - entry);
      nKeys++;
    }
  }
}

static ncclResult_t rasDiagnosticsNcclEnvFillLocalData(const struct rasDiagnosticsCommSnapshot* comm, void* checkData) {
  struct rasDiagnosticsNcclEnvData* envData = (struct rasDiagnosticsNcclEnvData*)checkData;
  char** entries = nullptr;
  int nEntries = 0;

  (void)comm;

  envData->bytesUsed = 0;
  envData->truncated = 0;

  for (char** ep = environ; ep != nullptr && *ep != nullptr; ep++) {
    if (strncmp(*ep, "NCCL_", 5) == 0) nEntries++;
  }
  if (nEntries == 0) return ncclSuccess;

  NCCLCHECK(ncclCalloc(&entries, nEntries));
  char** entry = entries;
  for (char** ep = environ; ep != nullptr && *ep != nullptr; ep++) {
    if (strncmp(*ep, "NCCL_", 5) == 0) *(entry++) = *ep;
  }
  qsort(entries, nEntries, sizeof(*entries), rasDiagnosticsEnvEntryCompare);

  for (int i = 0; i < nEntries; i++) {
    size_t len = strlen(entries[i]) + 1;
    if (len > (size_t)RAS_DIAG_ENV_BYTES - envData->bytesUsed) {
      envData->truncated = 1;
      break;
    }
    memcpy(envData->data + envData->bytesUsed, entries[i], len);
    envData->bytesUsed += (uint16_t)len;
  }

  free(entries);
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsNcclEnvCollectLocal(const struct rasDiagnosticsContext* ctx,
                                               struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsNcclEnvData),
                                              rasDiagnosticsNcclEnvFillLocalData, data));
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsNcclEnvReportKey(const struct rasDiagnosticsReporter* reporter, const char* records,
                                                   size_t recordStride, int start, int end,
                                                   const struct rasDiagnosticsRankHeader* startRank,
                                                   const struct rasDiagnosticsEnvKey* key, int* nMismatch) {
  ncclResult_t ret = ncclSuccess;
  bool anyDiff = false;
  int nGroupRanks = end - start;
  const char** values = nullptr;
  int* covered = nullptr;
  int* groupRanks = nullptr;
  char keyDisplay[RAS_DIAG_ENV_KEY_DISPLAY_BYTES];

  NCCLCHECKGOTO(ncclCalloc(&values, nGroupRanks), ret, exit);
  for (int i = 0; i < nGroupRanks; i++) {
    values[i] = rasDiagnosticsEnvFindValue(
      rasDiagnosticsNcclEnvDataFromRecord(records + (size_t)(start + i) * recordStride), key->key, key->len);
    if (i > 0 && !rasDiagnosticsEnvValuesEqual(values[0], values[i])) {
      anyDiff = true;
    }
  }
  if (!anyDiff) goto exit;

  (*nMismatch)++;
  NCCLCHECKGOTO(ncclCalloc(&covered, nGroupRanks), ret, exit);
  NCCLCHECKGOTO(ncclCalloc(&groupRanks, nGroupRanks), ret, exit);
  rasDiagnosticsEnvKeyDisplay(key, keyDisplay, sizeof(keyDisplay));

  NCCLCHECKGOTO(rasDiagnosticsNcclEnvReportMismatch(reporter, startRank, keyDisplay), ret, exit);

  for (int i = 0; i < nGroupRanks; i++) {
    const char* value;
    int nGroup = 0;
    char rankSet[128];
    char valueDisplay[RAS_DIAG_ENV_VALUE_DISPLAY_BYTES];

    if (covered[i]) continue;
    value = values[i];
    for (int j = i; j < nGroupRanks; j++) {
      const struct rasDiagnosticsRankHeader* rank =
        rasDiagnosticsRankHeaderFromRecord(records + (size_t)(start + j) * recordStride);
      const char* otherValue = values[j];

      if (covered[j] || !rasDiagnosticsEnvValuesEqual(value, otherValue)) continue;
      covered[j] = 1;
      groupRanks[nGroup++] = rank->commRank;
    }

    rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), groupRanks, nGroup, nGroup);
    rasDiagnosticsEnvValueDisplay(value, valueDisplay, sizeof(valueDisplay));
    NCCLCHECKGOTO(rasDiagnosticsNcclEnvReportValue(reporter, keyDisplay, valueDisplay, rankSet), ret, exit);
  }

exit:
  free(values);
  free(covered);
  free(groupRanks);
  return ret;
}

ncclResult_t rasDiagnosticsNcclEnvSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNcclEnvData));
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics NCCL environment check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics NCCL environment check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics NCCL environment check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + (size_t)start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    int commNRanks = startRank->commNRanks;
    int end = start;
    int nTruncated = 0;
    int nMismatch = 0;
    int nCandidateKeys;
    ncclUniquePtr<struct rasDiagnosticsEnvKey> keys;

    while (end < nRecords) {
      const char* record = records + (size_t)end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsNcclEnvData* envData;

      if (end > start && rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      envData = rasDiagnosticsNcclEnvDataFromRecord(record);
      if (!rasDiagnosticsNcclEnvDataValid(envData)) {
        WARN("RAS diagnostics NCCL environment check received malformed env payload");
        ret = ncclInternalError;
        goto exit;
      }
      if (envData->truncated) nTruncated++;
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "NCCL environment", startRank, end - start), ret, exit);
      start = end;
      continue;
    }

    nCandidateKeys = rasDiagnosticsEnvCountCandidateKeys(records, recordStride, start, end);
    if (nCandidateKeys > 0) {
      NCCLCHECKGOTO(ncclCalloc(keys, nCandidateKeys), ret, exit);
      rasDiagnosticsEnvFillCandidateKeys(records, recordStride, start, end, keys.get());
      qsort(keys.get(), nCandidateKeys, sizeof(*keys.get()), rasDiagnosticsEnvKeyCompare);

      for (int keyIdx = 0; keyIdx < nCandidateKeys; keyIdx++) {
        const struct rasDiagnosticsEnvKey* key = keys.get() + keyIdx;

        if (keyIdx > 0 && rasDiagnosticsEnvKeyCompare(key - 1, key) == 0) continue;
        ret = rasDiagnosticsNcclEnvReportKey(reporter, records, recordStride, start, end, startRank, key, &nMismatch);
        if (ret != ncclSuccess) goto exit;
      }
    }

    if (nTruncated > 0) {
      NCCLCHECKGOTO(rasDiagnosticsNcclEnvReportTruncated(reporter, nTruncated), ret, exit);
    }
    if (nMismatch == 0 && nTruncated == 0) {
      NCCLCHECKGOTO(rasDiagnosticsNcclEnvReportConsistent(reporter, commNRanks, &startRank->commId), ret, exit);
    } else if (nMismatch > 0) {
      NCCLCHECKGOTO(rasDiagnosticsNcclEnvReportMismatchSummary(reporter, nMismatch, &startRank->commId), ret, exit);
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}
