/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mutex>

#include "alloc.h"
#include "checks.h"
#include "diagnostics_checks.h"
#include "diagnostics_checks_common.h"
#include "graph.h"
#include "graph/topo.h"

// *************************************************************************
// Path type detection check.
// *************************************************************************
// Reports, per communicator, the union of PATH_* types its ranks use to reach their peers, annotating any type that is
// present on only a subset of ranks (partial coverage) with a "(present/total)" ratio. Uneven cluster shapes (e.g. some
// ranks C2C-connected, others not) surface as e.g. "self+C2C(6/7)+SYS+NET across 7 ranks".
//
// Unlike the NVML/environ checks, the per-rank probe reads comm->topo, so it must run while ncclCommsMutex is held
// (commFree calls ncclRasCommFini -- which nulls the ncclComms[] slot under the mutex -- before ncclTopoFree, so a
// non-null slot under the lock guarantees topo is still alive). It therefore uses a dedicated collector rather than
// the after-unlock rasDiagnosticsCollectLocalRecords() helper the scalar checks share, following the same pattern as
// the PCI check in diagnostics_pci.cc.

// Number of distinct PATH_* types (PATH_LOC..PATH_DIS), for sizing per-path-type arrays.
#define RAS_DIAG_PATH_TYPES (PATH_DIS + 1)

struct rasDiagnosticsPathsData {
  uint32_t pathMask; // Bit (1u<<PATH_*) set for every path type this rank uses.
  uint8_t localGpuMissing; // This rank's own GPU was absent from its topology, so pathMask is not meaningful.
};

// Computes this rank's path type usage from the (already-built) topology. `pathMask` has bit (1u<<PATH_*) set for every
// type observed (PATH_LOC for self). Peers absent from the local topology are attributed to PATH_NET. If this rank's
// own GPU is absent, no path type can be determined, so `localGpuMissing` is set and `pathMask` is left empty. Pure
// in-memory read of `topo`; the caller must hold ncclCommsMutex so that `topo` stays alive for the call.
static void rasDiagnosticsPathsUsage(struct ncclTopoSystem* topo, int rank, int nRanks,
                                     struct rasDiagnosticsPathsData* paths) {
  int myGpuIdx = -1;

  memset(paths, 0, sizeof(*paths));

  if (topo != nullptr) ncclTopoRankToIndex(topo, rank, &myGpuIdx, /*showWarn=*/false);
  if (myGpuIdx < 0) {
    // Without our own GPU we cannot classify any peer; report that rather than mislabelling every path as PATH_NET.
    paths->localGpuMissing = 1;
    return;
  }

  for (int peer = 0; peer < nRanks; peer++) {
    int t;
    if (peer == rank) {
      t = PATH_LOC;
    } else {
      int peerGpuIdx = -1;
      ncclTopoRankToIndex(topo, peer, &peerGpuIdx, /*showWarn=*/false);
      // A peer absent from the local topology is reached over the network.
      if (peerGpuIdx < 0) t = PATH_NET;
      else if (topo->nodes[GPU].nodes[myGpuIdx].paths[GPU] == nullptr) t = PATH_DIS;
      else t = topo->nodes[GPU].nodes[myGpuIdx].paths[GPU][peerGpuIdx].type;
    }
    if (t < 0 || t >= RAS_DIAG_PATH_TYPES) t = PATH_DIS;
    paths->pathMask |= 1u << t;
  }
}

ncclResult_t rasDiagnosticsPathsCollectLocal(const struct rasDiagnosticsContext* ctx,
                                             struct rasDiagnosticsLocalData* data) {
  ncclUniquePtr<char> records;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsPathsData));
  int nRecords = 0;

  if (data == nullptr) {
    WARN("RAS diagnostics paths check local data output is null");
    return ncclInternalError;
  }
  memset(data, 0, sizeof(*data));

  if (ctx == nullptr) {
    WARN("RAS diagnostics paths check local collection requested with null context");
    return ncclInternalError;
  }
  if (recordStride > (size_t)INT_MAX) {
    WARN("RAS diagnostics paths check local record stride %zu is invalid", recordStride);
    return ncclInternalError;
  }

  // Filtered requests originate from the initialization-time trigger, where the ranks have had time to build their
  // topology since the last synchronization point; unfiltered ones can arrive at any moment, so require readiness.
  const bool requireCommInitialized = !ctx->hasCommFilter;

  // topo is read below, so the whole probe runs under ncclCommsMutex (see comment above).
  {
    std::lock_guard<std::mutex> lock(ncclCommsMutex);

    for (int i = 0; i < nNcclComms; i++) {
      struct ncclComm* comm = ncclComms[i];
      if (comm == nullptr) continue;
      if (!COMPILER_ATOMIC_LOAD(&comm->peerInfoValid, std::memory_order_acquire)) continue;
      if (requireCommInitialized && COMPILER_ATOMIC_LOAD(&comm->initState, std::memory_order_acquire) != ncclSuccess)
        continue;
      if (!rasDiagnosticsCommMatchesContext(ctx, comm)) continue;
      nRecords++;
    }

    if (nRecords == 0) return ncclSuccess;
    if ((size_t)nRecords > (size_t)INT_MAX / recordStride) {
      WARN("RAS diagnostics paths check local data too large");
      return ncclInternalError;
    }

    NCCLCHECK(ncclCalloc(records, (size_t)nRecords * recordStride));
    for (int i = 0, recordIdx = 0; i < nNcclComms && recordIdx < nRecords; i++) {
      struct ncclComm* comm = ncclComms[i];
      if (comm == nullptr) continue;
      if (!COMPILER_ATOMIC_LOAD(&comm->peerInfoValid, std::memory_order_acquire)) continue;
      if (requireCommInitialized && COMPILER_ATOMIC_LOAD(&comm->initState, std::memory_order_acquire) != ncclSuccess)
        continue;
      if (!rasDiagnosticsCommMatchesContext(ctx, comm)) continue;

      char* record = records.get() + (size_t)recordIdx * recordStride;
      struct rasDiagnosticsCommSnapshot snapshot;
      struct rasDiagnosticsPathsData* paths;

      rasDiagnosticsCommSnapshotInit(&snapshot, comm);
      memcpy(record, &snapshot.rank, sizeof(snapshot.rank));

      paths = (struct rasDiagnosticsPathsData*)(record + sizeof(struct rasDiagnosticsRankHeader));
      rasDiagnosticsPathsUsage(comm->topo, comm->rank, comm->nRanks, paths);
      recordIdx++;
    }
  }

  data->records = records.release();
  data->recordsBytes = (int)((size_t)nRecords * recordStride);
  data->recordStride = (int)recordStride;
  data->nRecords = nRecords;
  return ncclSuccess;
}

static const struct rasDiagnosticsPathsData* rasDiagnosticsPathsDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsPathsData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

// Appends one path type to the summary buffer as "NAME" (all ranks) or "NAME(present/total)" (partial). Returns
// the running offset; on overflow it stops advancing so the caller-visible string stays NUL-terminated and bounded.
static int rasDiagnosticsPathsAppendType(char* buf, size_t bufLen, int pos, bool first, int type, int nPresent,
                                         int commNRanks) {
  const char* name = (type == PATH_LOC) ? "self" : topoPathTypeStr[type];

  if (pos < 0 || (size_t)pos >= bufLen) return pos;
  if (nPresent == commNRanks) {
    pos += snprintf(buf + pos, bufLen - pos, "%s%s", first ? "" : "+", name);
  } else {
    pos += snprintf(buf + pos, bufLen - pos, "%s%s(%d/%d)", first ? "" : "+", name, nPresent, commNRanks);
  }
  return pos;
}

ncclResult_t rasDiagnosticsPathsSummarize(const struct rasDiagnosticsContext* ctx,
                                          const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsPathsData));
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics paths check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics paths check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics paths check received malformed data size %d", nData);
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
    int ranksWithType[RAS_DIAG_PATH_TYPES] = {};
    int missingRanks[RAS_DIAG_RANK_SET_MAX];
    int nMissing = 0, nMissingStored = 0;
    int end = start;

    while (end < nRecords) {
      const char* record = records + (size_t)end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsPathsData* paths;

      if (end > start && rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      paths = rasDiagnosticsPathsDataFromRecord(record);
      if (paths->localGpuMissing) {
        if (nMissingStored < RAS_DIAG_RANK_SET_MAX) missingRanks[nMissingStored++] = rank->commRank;
        nMissing++;
      } else {
        for (int t = 0; t < RAS_DIAG_PATH_TYPES; t++) {
          if (paths->pathMask & (1u << t)) ranksWithType[t]++;
        }
      }
      end++;
    }

    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "Paths", startRank, end - start), ret, exit);
    } else {
      char buf[512];
      int pos = 0;
      bool first = true;
      // Partial coverage and disconnected paths are informational, not failures: uneven topologies (e.g. mixed
      // interop clusters) are legal, and a rank whose own GPU is missing simply could not be classified.
      bool info = (nMissing > 0) || (ranksWithType[PATH_DIS] > 0);

      if (nMissing > 0) {
        char rankSet[256];

        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), missingRanks, nMissingStored, nMissing);
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "Paths: could not locate the rank's own GPU in the topology, so its paths "
                                           "are unknown -- %s in comm 0x%lx",
                                           rankSet, startRank->commId.commHash),
                      ret, exit);
      }

      buf[0] = '\0';
      for (int t = 0; t < RAS_DIAG_PATH_TYPES; t++) {
        if (ranksWithType[t] == 0) continue;
        if (ranksWithType[t] != commNRanks) info = true;
        pos = rasDiagnosticsPathsAppendType(buf, sizeof(buf), pos, first, t, ranksWithType[t], commNRanks);
        first = false;
      }

      if (!first) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, info ? RAS_DIAG_TAG_INFO : RAS_DIAG_TAG_OK,
                                           "Paths: %s across %d ranks in comm 0x%lx", buf, commNRanks,
                                           startRank->commId.commHash),
                      ret, exit);
      }
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}
