/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "alloc.h"
#include "checks.h"
#include "diagnostics_checks.h"
#include "diagnostics_checks_common.h"
#include "graph/topo.h"
#include "nccl_net.h"
#include "os.h"

// *************************************************************************
// NET device consistency check.
// *************************************************************************

// The size_t cast also rejects negative codes.
#define RAS_DIAG_NET_STR(names, code) ((size_t)(code) < sizeof(names) / sizeof((names)[0]) ? (names)[code] : "unknown")

// Indexed by ncclNetDeviceType.
static const char* const kRasDiagNetDeviceTypes[] = {"Host",       "Unpack",  "GIN Proxy",
                                                     "GIN GDA-KI", "GIN GPI", "GIN EFA GDA"};
// Indexed by RAS_DIAG_NET_LINK_*.
static const char* const kRasDiagNetLinkLayers[] = {"unknown", "InfiniBand", "Ethernet"};
// sysfs reports e.g. "4: ACTIVE"; the record stores the number.
static const char* const kRasDiagNetPortStates[] = {"unknown", "DOWN", "INIT", "ARMED", "ACTIVE", "ACTIVE_DEFER"};
static const char* const kRasDiagNetPhysStates[] = {
  "unknown", "SLEEP", "POLLING", "DISABLED", "TRAINING", "LinkUp", "LINK_ERROR_RECOVERY", "PHY_TEST"
};
#define RAS_DIAG_NET_PORT_ACTIVE 4
#define RAS_DIAG_NET_PHYS_LINKUP 5

static void rasDiagnosticsNetFormatProps(char* buffer, size_t size, const struct rasDiagnosticsNetDeviceData* netData) {
  snprintf(buffer, size, "net device type=%s, speed=%d Mbps, link layer=%s%s, state=%s/%s",
           RAS_DIAG_NET_STR(kRasDiagNetDeviceTypes, netData->netDeviceType), netData->speed,
           RAS_DIAG_NET_STR(kRasDiagNetLinkLayers, netData->linkLayer), netData->dataDirect ? ", Data Direct" : "",
           RAS_DIAG_NET_STR(kRasDiagNetPortStates, netData->portState),
           RAS_DIAG_NET_STR(kRasDiagNetPhysStates, netData->physState));
}

// Only single-port devices resolve (to port 1): the plugin's port number does not index sysfs.
static bool rasDiagnosticsNetHasOnePort(const char* base) {
  char path[PATH_MAX];

  // access(): the sysfs reader logs every attribute it fails to open.
  snprintf(path, sizeof(path), "%s/ports/1/state", base);
  if (access(path, R_OK) != 0) return false;
  snprintf(path, sizeof(path), "%s/ports/2/state", base);
  return access(path, R_OK) != 0;
}

// Best effort: unreadable attributes stay 0.
void rasDiagnosticsNetReadPortAttrs(struct rasDiagnosticsNetDeviceData* netData, const char* sysfsRoot) {
  // Bounded by the record's name length, so appending a port cannot truncate.
  char base[64 + RAS_DIAG_NET_DEVICE_NAME_LEN];
  char path[PATH_MAX];
  char value[64];

  // Reject names that could escape the sysfs directory.
  if (netData->name[0] == '\0' || strchr(netData->name, '/') != nullptr) return;
  snprintf(base, sizeof(base), "%s/%s", sysfsRoot, netData->name);
  if (!rasDiagnosticsNetHasOnePort(base)) {
    // "<device>_dma" (Data Direct) has no sysfs entry; resolve it through "<device>".
    const char* suffix = strrchr(netData->name, '_');
    if (suffix == nullptr || suffix == netData->name || strcmp(suffix, "_dma") != 0) return;
    snprintf(base, sizeof(base), "%s/%.*s", sysfsRoot, (int)(suffix - netData->name), netData->name);
    if (!rasDiagnosticsNetHasOnePort(base)) return;
    netData->name[suffix - netData->name] = '\0';
    netData->dataDirect = 1;
  }
  netData->port = 1;
  snprintf(path, sizeof(path), "%s/ports/1", base);

  (void)ncclOsTopoGetStrFromSys(path, "link_layer", value, sizeof(value));
  if (strcmp(value, "InfiniBand") == 0) netData->linkLayer = RAS_DIAG_NET_LINK_INFINIBAND;
  else if (strcmp(value, "Ethernet") == 0) netData->linkLayer = RAS_DIAG_NET_LINK_ETHERNET;

  (void)ncclOsTopoGetStrFromSys(path, "state", value, sizeof(value));
  netData->portState = (int8_t)strtol(value, nullptr, 10);
  (void)ncclOsTopoGetStrFromSys(path, "phys_state", value, sizeof(value));
  netData->physState = (int8_t)strtol(value, nullptr, 10);

  // Rate reads e.g. "400 Gb/sec (4X NDR)" or "2.5 Gb/sec (1X SDR)"; parsed digit-wise, strtod being locale-dependent.
  char* end = nullptr;
  (void)ncclOsTopoGetStrFromSys(path, "rate", value, sizeof(value));
  const long rate = strtol(value, &end, 10);
  const int tenths = (*end == '.' && end[1] >= '0' && end[1] <= '9') ? end[1] - '0' : 0;
  if (rate > 0) netData->speed = (int)rate * 1000 + tenths * 100;
}

static bool rasDiagnosticsNetDeviceSeen(const char* records, size_t recordStride, int count, const char* name,
                                        int port) {
  for (int i = 0; i < count; i++) {
    const struct rasDiagnosticsNetDeviceData* record =
      (const struct rasDiagnosticsNetDeviceData*)(records + (size_t)i * recordStride);

    if (record->port == port && strcmp(record->name, name) == 0) return true;
  }
  return false;
}

// Expands the NET devices the plugin exposes into wire records, merged devices becoming their members.
int rasDiagnosticsNetDevicePrepare(const struct ncclTopoNetPropertiesSnapshot* netProps, int nNetProps,
                                   uint64_t hostHash, char* out, size_t outStride, int capacity) {
  int count = 0;
  for (int dev = 0; dev < nNetProps; dev++) {
    const struct ncclTopoNetPropertiesSnapshot* props = netProps + dev;
    if (!props->propertiesValid || props->name[0] == '\0') continue;
    if (props->vProps.ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
      WARN("RAS diagnostics received invalid member count %d for merged NET device %s", props->vProps.ndevs,
           props->name);
      continue;
    }
    int nMembers = props->vProps.ndevs > 1 ? props->vProps.ndevs : 1;
    for (int m = 0; m < nMembers; m++) {
      const struct ncclTopoNetPropertiesSnapshot* member = props;
      if (props->vProps.ndevs > 1) {
        const int physicalDev = props->vProps.devs[m];
        if (physicalDev < 0 || physicalDev >= nNetProps) {
          WARN("RAS diagnostics received invalid member index %d for merged NET device %s", physicalDev, props->name);
          continue;
        }
        member = netProps + physicalDev;
        if (!member->propertiesValid || member->name[0] == '\0') continue;
      }
      if (rasDiagnosticsNetDeviceSeen(out, outStride, count, member->name, member->port) || count >= capacity) continue;
      struct rasDiagnosticsNetDeviceData* record = (struct rasDiagnosticsNetDeviceData*)(out + count * outStride);
      memset(record, 0, sizeof(*record));
      record->hostHash = hostHash;
      record->netDeviceType = member->netDeviceType;
      record->speed = member->speed;
      record->port = member->port;
      snprintf(record->name, sizeof(record->name), "%.*s", (int)sizeof(record->name) - 1, member->name);
      count++;
    }
  }
  return count;
}

static ncclResult_t rasDiagnosticsNetDeviceFillLocalRecords(
  const struct rasDiagnosticsCommSnapshot* comm, char* records, size_t recordStride, int maxRecords, int* nRecords) {
  char* checkData = records + sizeof(struct rasDiagnosticsRankHeader);

  *nRecords = 0;
  if (comm->netIbVerbs)
    *nRecords = rasDiagnosticsNetDevicePrepare(comm->netProps, comm->nNetProps, comm->hostHash, checkData, recordStride,
                                               maxRecords);
  if (*nRecords > 0) {
    int nUnique = 0;
    for (int i = 0; i < *nRecords; i++) {
      struct rasDiagnosticsNetDeviceData* netData =
        (struct rasDiagnosticsNetDeviceData*)(checkData + (size_t)i * recordStride);

      rasDiagnosticsNetReadPortAttrs(netData);
      // The read can rename "<device>_dma" onto the same port as "<device>", so deduplicate again.
      if (rasDiagnosticsNetDeviceSeen(checkData, recordStride, nUnique, netData->name, netData->port)) continue;
      // The caller writes the rank headers once the count is final.
      if (nUnique != i) memcpy(checkData + (size_t)nUnique * recordStride, netData, sizeof(*netData));
      nUnique++;
    }
    *nRecords = nUnique;
    return ncclSuccess;
  }
  // Emit a marker so the rank is not mistaken for one that never answered.
  if (maxRecords < 1) {
    WARN("RAS diagnostics NET device record capacity exceeded");
    return ncclInternalError;
  }
  struct rasDiagnosticsNetDeviceData* netData = (struct rasDiagnosticsNetDeviceData*)checkData;
  memset(netData, 0, sizeof(*netData));
  netData->hostHash = comm->hostHash;
  netData->ibVerbsWithoutDevice = comm->netIbVerbs;
  *nRecords = 1;
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsNetDeviceCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                 struct rasDiagnosticsLocalData* data) {
  ncclResult_t ret = ncclSuccess;
  ncclUniquePtr<char> records;
  struct rasDiagnosticsCommSnapshot* comms = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNetDeviceData));
  size_t maxRecords = 0;
  int nComms = 0, nRecords = 0;

  if (ctx == nullptr || data == nullptr) return ncclInternalError;
  memset(data, 0, sizeof(*data));
  NCCLCHECK(rasDiagnosticsCollectCommSnapshots(ctx, /*collectNetDevices=*/true, &comms, &nComms));
  if (nComms == 0) return ncclSuccess;
  // Every comm contributes at least one record.
  for (int i = 0; i < nComms; i++)
    maxRecords += comms[i].netIbVerbs && comms[i].nNetProps > 0 ? (size_t)comms[i].nNetProps : 1;
  if (maxRecords > (size_t)INT_MAX / recordStride) {
    WARN("RAS diagnostics NET device data is too large");
    ret = ncclInternalError;
    goto exit;
  }
  NCCLCHECKGOTO(ncclCalloc(records, maxRecords * recordStride), ret, exit);

  for (int i = 0; i < nComms; i++) {
    char* output = records.get() + (size_t)nRecords * recordStride;
    int filledRecords;

    NCCLCHECKGOTO(rasDiagnosticsNetDeviceFillLocalRecords(comms + i, output, recordStride, (int)(maxRecords - nRecords),
                                                          &filledRecords),
                  ret, exit);
    for (int j = 0; j < filledRecords; j++)
      memcpy(output + (size_t)j * recordStride, &comms[i].rank, sizeof(comms[i].rank));
    nRecords += filledRecords;
  }

  data->records = records.release();
  data->recordsBytes = (int)((size_t)nRecords * recordStride);
  data->recordStride = (int)recordStride;
  data->nRecords = nRecords;
exit:
  rasDiagnosticsFreeCommSnapshots(comms, nComms);
  return ret;
}

static const struct rasDiagnosticsNetDeviceData* rasDiagnosticsNetDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsNetDeviceData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

// Sort key for the pass by node: communicator, node, device name, port, rank.
static int rasDiagnosticsNetNodeCompare(const void* p1, const void* p2) {
  const struct rasDiagnosticsRankHeader* r1 = (const struct rasDiagnosticsRankHeader*)p1;
  const struct rasDiagnosticsRankHeader* r2 = (const struct rasDiagnosticsRankHeader*)p2;
  const struct rasDiagnosticsNetDeviceData* d1 = rasDiagnosticsNetDataFromRecord((const char*)p1);
  const struct rasDiagnosticsNetDeviceData* d2 = rasDiagnosticsNetDataFromRecord((const char*)p2);
  int cmp = rasDiagnosticsCommIdCompare(&r1->commId, &r2->commId);
  if (cmp != 0) return cmp;
  if (d1->hostHash != d2->hostHash) return d1->hostHash < d2->hostHash ? -1 : 1;
  cmp = strcmp(d1->name, d2->name);
  if (cmp != 0) return cmp;
  if (d1->port != d2->port) return d1->port < d2->port ? -1 : 1;
  return r1->commRank < r2->commRank ? -1 : (r1->commRank > r2->commRank ? 1 : 0);
}

// Sort key for the pass by device, within one communicator: device name, port, node, rank. Markers
// (empty name) sort first.
static int rasDiagnosticsNetDeviceCompare(const void* p1, const void* p2) {
  const struct rasDiagnosticsRankHeader* r1 = (const struct rasDiagnosticsRankHeader*)p1;
  const struct rasDiagnosticsRankHeader* r2 = (const struct rasDiagnosticsRankHeader*)p2;
  const struct rasDiagnosticsNetDeviceData* d1 = rasDiagnosticsNetDataFromRecord((const char*)p1);
  const struct rasDiagnosticsNetDeviceData* d2 = rasDiagnosticsNetDataFromRecord((const char*)p2);
  int cmp = strcmp(d1->name, d2->name);
  if (cmp != 0) return cmp;
  if (d1->port != d2->port) return d1->port < d2->port ? -1 : 1;
  if (d1->hostHash != d2->hostHash) return d1->hostHash < d2->hostHash ? -1 : 1;
  return r1->commRank < r2->commRank ? -1 : (r1->commRank > r2->commRank ? 1 : 0);
}

static bool rasDiagnosticsNetDataEqual(const struct rasDiagnosticsNetDeviceData* d1,
                                       const struct rasDiagnosticsNetDeviceData* d2) {
  return d1->netDeviceType == d2->netDeviceType && d1->speed == d2->speed && d1->linkLayer == d2->linkLayer &&
         d1->portState == d2->portState && d1->physState == d2->physState && d1->dataDirect == d2->dataDirect;
}

ncclResult_t rasDiagnosticsNetDeviceSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNetDeviceData));
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics Net device check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics Net device check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics Net device check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsNetNodeCompare);

  for (int commStart = 0; commStart < nRecords;) {
    const struct rasDiagnosticsRankHeader* startRank =
      rasDiagnosticsRankHeaderFromRecord(records + (size_t)commStart * recordStride);
    ncclUniquePtr<uint8_t> seenRanks;
    char* commRecords = records + (size_t)commStart * recordStride;
    int commEnd = commStart + 1;
    int nCommRecords;
    int deviceStart = 0;
    int nCommRanks = 0;
    int nCommNodes = 0; // nodes reporting at least one device
    int nExpectedNodes = 0; // plus nodes whose NICs went uninspected
    int nUninspectedRanks = 0;
    int nSkippedRanks = 0;
    int nNodesDiffer = 0;

    while (commEnd < nRecords) {
      const struct rasDiagnosticsRankHeader* rank =
        rasDiagnosticsRankHeaderFromRecord(records + (size_t)commEnd * recordStride);
      if (rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      commEnd++;
    }
    nCommRecords = commEnd - commStart;

    if (startRank->commNRanks > 0) NCCLCHECKGOTO(ncclCalloc(seenRanks, startRank->commNRanks), ret, exit);
    for (int i = 0; i < nCommRecords; i++) {
      const struct rasDiagnosticsRankHeader* rank =
        rasDiagnosticsRankHeaderFromRecord(commRecords + (size_t)i * recordStride);
      if (rank->commRank >= 0 && rank->commRank < startRank->commNRanks && !seenRanks.get()[rank->commRank]) {
        seenRanks.get()[rank->commRank] = 1;
        nCommRanks++;
      }
    }
    if (nCommRanks != startRank->commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "Net device", startRank, nCommRanks), ret, exit);
      commStart = commEnd;
      continue;
    }

    // Pass by node: classify markers, count nodes, compare the NICs within each node.
    for (int nodeStart = 0; nodeStart < nCommRecords;) {
      const struct rasDiagnosticsNetDeviceData* reference = nullptr;
      const uint64_t hostHash =
        rasDiagnosticsNetDataFromRecord(commRecords + (size_t)nodeStart * recordStride)->hostHash;
      bool uninspected = false;
      bool differs = false;
      int nodeEnd = nodeStart;

      for (; nodeEnd < nCommRecords; nodeEnd++) {
        const struct rasDiagnosticsNetDeviceData* netData =
          rasDiagnosticsNetDataFromRecord(commRecords + (size_t)nodeEnd * recordStride);
        if (netData->hostHash != hostHash) break;
        if (netData->name[0] == '\0') {
          // Skipped (unrecognized plugin) or uninspected (recognized plugin, no device).
          if (netData->ibVerbsWithoutDevice) {
            nUninspectedRanks++;
            uninspected = true;
          } else {
            nSkippedRanks++;
          }
        } else if (reference == nullptr) {
          reference = netData;
        } else if (!rasDiagnosticsNetDataEqual(reference, netData)) {
          differs = true;
        }
      }
      if (reference != nullptr) {
        nCommNodes++;
        if (differs) nNodesDiffer++;
      }
      if (reference != nullptr || uninspected) nExpectedNodes++;
      nodeStart = nodeEnd;
    }

    if (nSkippedRanks > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "Net device: skipped across %d ranks in comm 0x%lx "
                                         "(network plugin is not a recognized verbs-based InfiniBand/RoCE plugin)",
                                         nSkippedRanks, startRank->commId.commHash),
                    ret, exit);
    }
    if (nUninspectedRanks > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "Net device: not inspected across %d ranks in comm 0x%lx "
                                         "(the plugin exposed no device, or no device could be queried)",
                                         nUninspectedRanks, startRank->commId.commHash),
                    ret, exit);
    }

    // Pass by device: compare each device across the nodes reporting it.
    qsort(commRecords, nCommRecords, recordStride, rasDiagnosticsNetDeviceCompare);
    while (deviceStart < nCommRecords &&
           rasDiagnosticsNetDataFromRecord(commRecords + (size_t)deviceStart * recordStride)->name[0] == '\0')
      deviceStart++;

    for (int start = deviceStart; start < nCommRecords;) {
      const struct rasDiagnosticsNetDeviceData* groupData =
        rasDiagnosticsNetDataFromRecord(commRecords + (size_t)start * recordStride);
      int referenceIdx = start; // lowest rank reporting the device
      int mismatchRanks[RAS_DIAG_RANK_SET_MAX];
      int nMismatchStored = 0;
      int nMismatch = 0;
      int nReportingNodes = 0;
      int end = start;

      // Within a device's run, each node's reports are contiguous.
      for (; end < nCommRecords; end++) {
        const char* record = commRecords + (size_t)end * recordStride;
        const struct rasDiagnosticsNetDeviceData* netData = rasDiagnosticsNetDataFromRecord(record);
        if (strcmp(groupData->name, netData->name) != 0 || groupData->port != netData->port) break;
        if (end == start || netData->hostHash != rasDiagnosticsNetDataFromRecord(record - recordStride)->hostHash)
          nReportingNodes++;
        if (rasDiagnosticsRankHeaderFromRecord(record)->commRank <
            rasDiagnosticsRankHeaderFromRecord(commRecords + (size_t)referenceIdx * recordStride)->commRank)
          referenceIdx = end;
      }

      const char* referenceRecord = commRecords + (size_t)referenceIdx * recordStride;
      const struct rasDiagnosticsNetDeviceData* reference = rasDiagnosticsNetDataFromRecord(referenceRecord);
      const int referenceRank = rasDiagnosticsRankHeaderFromRecord(referenceRecord)->commRank;
      for (int i = start; i < end; i++) {
        const char* record = commRecords + (size_t)i * recordStride;

        if (i == referenceIdx) continue;
        if (!rasDiagnosticsNetDataEqual(reference, rasDiagnosticsNetDataFromRecord(record))) {
          if (nMismatchStored < RAS_DIAG_RANK_SET_MAX)
            mismatchRanks[nMismatchStored++] = rasDiagnosticsRankHeaderFromRecord(record)->commRank;
          nMismatch++;
        }
      }

      // [OK] requires all attributes read and the port ACTIVE/LinkUp.
      const char* portNote = "";
      if (reference->linkLayer == RAS_DIAG_NET_LINK_UNKNOWN || reference->portState == 0 || reference->physState == 0)
        portNote = " (some port attributes unavailable)";
      else if (reference->portState != RAS_DIAG_NET_PORT_ACTIVE || reference->physState != RAS_DIAG_NET_PHYS_LINKUP)
        portNote = " (port is not active)";
      char props[256];

      rasDiagnosticsNetFormatProps(props, sizeof(props), reference);
      if (nMismatch > 0) {
        char rankSet[128];
        rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), mismatchRanks, nMismatchStored, nMismatch);
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "Net device %s:%d: rank(s) %s differ from rank %d (%s) across %d of %d "
                                           "nodes in comm 0x%lx",
                                           reference->name, reference->port, rankSet, referenceRank, props,
                                           nReportingNodes, nExpectedNodes, startRank->commId.commHash),
                      ret, exit);
      } else if (nReportingNodes < nExpectedNodes) {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                           "Net device %s:%d: reported on only %d of %d nodes in comm 0x%lx "
                                           "(device may be missing on some nodes)%s",
                                           reference->name, reference->port, nReportingNodes, nExpectedNodes,
                                           startRank->commId.commHash, portNote),
                      ret, exit);
      } else {
        NCCLCHECKGOTO(rasDiagnosticsReport(reporter, portNote[0] == '\0' ? RAS_DIAG_TAG_OK : RAS_DIAG_TAG_INFO,
                                           "Net device %s:%d: %s across %d nodes in comm 0x%lx%s", reference->name,
                                           reference->port, props, nReportingNodes, startRank->commId.commHash,
                                           portNote),
                      ret, exit);
      }
      start = end;
    }

    if (nNodesDiffer > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "Net device: NIC configuration (device type, speed, link layer, port "
                                         "state or Data Direct use) differs within %d of %d nodes in comm 0x%lx",
                                         nNodesDiffer, nCommNodes, startRank->commId.commHash),
                    ret, exit);
    } else if (nCommNodes > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                                         "Net device: NIC configuration consistent within each node across %d "
                                         "nodes in comm 0x%lx",
                                         nCommNodes, startRank->commId.commHash),
                    ret, exit);
    }

    commStart = commEnd;
  }

exit:
  free(records);
  return ret;
}
