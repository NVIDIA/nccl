/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <mutex>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "compiler.h"
#include "diagnostics_checks_common.h"
#include "ras_internal.h"
#include "transport.h"

void rasDiagnosticsCommSnapshotInit(struct rasDiagnosticsCommSnapshot* snapshot, const struct ncclComm* comm) {
  snapshot->rank.commId.commHash = comm->commHash;
  snapshot->rank.commId.hostHash = comm->peerInfo[0].hostHash;
  snapshot->rank.commId.pidHash = comm->peerInfo[0].pidHash;
  snapshot->rank.commRank = comm->rank;
  snapshot->rank.commNRanks = comm->nRanks;
  snapshot->cudaDev = comm->cudaDev;
  snapshot->nvmlDev = comm->nvmlDev;
  snapshot->busId = comm->busId;
  snapshot->localRank = comm->localRank;
  snapshot->localRanks = comm->localRanks;
}

int rasDiagnosticsCommIdCompare(const struct rasCommId* id1, const struct rasCommId* id2) {
  if (id1->commHash != id2->commHash) return (id1->commHash < id2->commHash ? -1 : 1);
  if (id1->hostHash != id2->hostHash) return (id1->hostHash < id2->hostHash ? -1 : 1);
  return (id1->pidHash < id2->pidHash ? -1 : (id1->pidHash > id2->pidHash ? 1 : 0));
}

bool rasDiagnosticsCommMatchesContext(const struct rasDiagnosticsContext* ctx, const struct ncclComm* comm) {
  struct rasCommId commId;

  if (!ctx->hasCommFilter) return true;
  commId.commHash = comm->commHash;
  commId.hostHash = comm->peerInfo[0].hostHash;
  commId.pidHash = comm->peerInfo[0].pidHash;
  return rasDiagnosticsCommIdCompare(&commId, &ctx->commFilter) == 0;
}

size_t rasDiagnosticsLocalRecordStride(size_t checkDataSize) {
  const size_t align = alignof(struct rasDiagnosticsRankHeader);
  size_t recordSize = sizeof(struct rasDiagnosticsRankHeader) + checkDataSize;

  return ((recordSize + align - 1) / align) * align;
}

static ncclResult_t rasDiagnosticsFillLocalRecord(char* record, const struct rasDiagnosticsCommSnapshot* snapshot,
                                                  rasDiagnosticsFillLocalDataFn fillCheckData) {
  memcpy(record, &snapshot->rank, sizeof(snapshot->rank));

  NCCLCHECK(fillCheckData(snapshot, record + sizeof(struct rasDiagnosticsRankHeader)));
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsCollectLocalRecords(const struct rasDiagnosticsContext* ctx, size_t checkDataSize,
                                               rasDiagnosticsFillLocalDataFn fillCheckData,
                                               struct rasDiagnosticsLocalData* data) {
  ncclUniquePtr<struct rasDiagnosticsCommSnapshot> snapshots;
  ncclUniquePtr<char> records;
  size_t recordStride;
  int nRecords = 0;
  size_t nBytes;

  if (data == nullptr) {
    WARN("RAS diagnostics check local data output is null");
    return ncclInternalError;
  }
  memset(data, 0, sizeof(*data));

  if (ctx == nullptr) {
    WARN("RAS diagnostics check local collection requested with null context");
    return ncclInternalError;
  }
  if (fillCheckData == nullptr || checkDataSize > (size_t)INT_MAX - sizeof(struct rasDiagnosticsRankHeader)) {
    WARN("RAS diagnostics check local data size %zu is invalid", checkDataSize);
    return ncclInternalError;
  }

  recordStride = rasDiagnosticsLocalRecordStride(checkDataSize);
  if (recordStride > (size_t)INT_MAX) {
    WARN("RAS diagnostics check local record stride %zu is invalid", recordStride);
    return ncclInternalError;
  }

  {
    std::lock_guard<std::mutex> lock(ncclCommsMutex);

    for (int i = 0; i < nNcclComms; i++) {
      struct ncclComm* comm = ncclComms[i];
      if (comm == nullptr) continue;
      if (!COMPILER_ATOMIC_LOAD(&comm->peerInfoValid, std::memory_order_acquire)) continue;
      if (!rasDiagnosticsCommMatchesContext(ctx, comm)) continue;
      nRecords++;
    }

    if (nRecords == 0) return ncclSuccess;
    if ((size_t)nRecords > (size_t)INT_MAX / recordStride) {
      WARN("RAS diagnostics check local data too large");
      return ncclInternalError;
    }

    NCCLCHECK(ncclCalloc(snapshots, nRecords));
    for (int i = 0, recordIdx = 0; i < nNcclComms && recordIdx < nRecords; i++) {
      struct ncclComm* comm = ncclComms[i];
      if (comm == nullptr) continue;
      if (!COMPILER_ATOMIC_LOAD(&comm->peerInfoValid, std::memory_order_acquire)) continue;
      if (!rasDiagnosticsCommMatchesContext(ctx, comm)) continue;
      rasDiagnosticsCommSnapshotInit(snapshots.get() + recordIdx, comm);
      recordIdx++;
    }
  }

  nBytes = (size_t)nRecords * recordStride;
  NCCLCHECK(ncclCalloc(records, nBytes));

  // Check-specific probes consume snapshots after releasing ncclCommsMutex.
  for (int recordIdx = 0; recordIdx < nRecords; recordIdx++) {
    NCCLCHECK(rasDiagnosticsFillLocalRecord(records.get() + recordIdx * recordStride, snapshots.get() + recordIdx,
                                            fillCheckData));
  }

  data->records = records.release();
  data->recordsBytes = (int)nBytes;
  data->recordStride = (int)recordStride;
  data->nRecords = nRecords;
  return ncclSuccess;
}

const struct rasDiagnosticsRankHeader* rasDiagnosticsRankHeaderFromRecord(const char* record) {
  return (const struct rasDiagnosticsRankHeader*)record;
}

// Orders records by comm, then rank; depends only on the rank header, so all checks share it.
int rasDiagnosticsRankHeaderCompare(const void* p1, const void* p2) {
  const struct rasDiagnosticsRankHeader* r1 = (const struct rasDiagnosticsRankHeader*)p1;
  const struct rasDiagnosticsRankHeader* r2 = (const struct rasDiagnosticsRankHeader*)p2;
  int cmp = rasDiagnosticsCommIdCompare(&r1->commId, &r2->commId);

  if (cmp != 0) return cmp;
  return (r1->commRank < r2->commRank ? -1 : (r1->commRank > r2->commRank ? 1 : 0));
}

void rasDiagnosticsFormatRankSet(char* buf, size_t bufLen, const int* ranks, int nStored, int nTotal) {
  int show = nStored < RAS_DIAG_RANK_SET_MAX ? nStored : RAS_DIAG_RANK_SET_MAX;
  int pos = snprintf(buf, bufLen, "{");

  for (int i = 0; i < show && pos > 0 && (size_t)pos < bufLen; i++) {
    pos += snprintf(buf + pos, bufLen - pos, "%s%d", i == 0 ? "" : ",", ranks[i]);
  }
  if (pos > 0 && (size_t)pos < bufLen) {
    if (nTotal > show) snprintf(buf + pos, bufLen - pos, ",...} (N=%d)", nTotal);
    else snprintf(buf + pos, bufLen - pos, "}");
  }
}

ncclResult_t rasDiagnosticsReport(const struct rasDiagnosticsReporter* reporter, const char* tag, const char* fmt,
                                  ...) {
  char line[1024];
  int pos = snprintf(line, sizeof(line), "%s", tag);
  if (pos < 0 || (size_t)pos >= sizeof(line)) {
    WARN("RAS diagnostics report formatting failed");
    return ncclInternalError;
  }

  va_list args;
  va_start(args, fmt);
  vsnprintf(line + pos, sizeof(line) - (size_t)pos, fmt, args);
  va_end(args);
  return reporter->emit(reporter->target, line);
}

ncclResult_t rasDiagnosticsReportIncomplete(const struct rasDiagnosticsReporter* reporter, const char* checkName,
                                            const struct rasDiagnosticsRankHeader* rank, int gatheredRanks) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "%s: diagnostics incomplete, gathered %d/%d ranks in comm 0x%lx/0x%lx/0x%lx "
                              "(RAS overlay may not be ready)",
                              checkName, gatheredRanks, rank->commNRanks, rank->commId.commHash, rank->commId.hostHash,
                              rank->commId.pidHash);
}
