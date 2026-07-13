/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <mutex>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "compiler.h"
#include "diagnostics_checks.h"
#include "diagnostics_checks_common.h"
#include "graph.h"
#include "graph/topo.h"
#include "ras_internal.h"
#include "transport.h"
#include "utils.h"
#include "../diagnostics.h"

#define RAS_DIAG_PCI_MAX_NET_DEVS 16
#define RAS_DIAG_PCI_MAX_GPU_NIC_PAIRS 16
#define RAS_DIAG_PCI_BDF_BYTES sizeof("00000000:00:00.0")
#define RAS_DIAG_RDMA_TOPO_OUTPUT_BYTES 4096
#define RAS_DIAG_RDMA_TOPO_SAMPLE_BYTES 256
#define RAS_DIAG_RDMA_TOPO_CHILD_TIMEOUT_SEC 2
#define RAS_DIAG_LSPCI_OUTPUT_BYTES (1024 * 1024)
#define RAS_DIAG_LSPCI_CHILD_TIMEOUT_SEC 2
// ncclDiagChildRun uses timeout(1), whose wrapper failures start at 124; signal-derived statuses start at 128.
#define RAS_DIAG_RDMA_TOPO_CHILD_FAILURE_STATUS_MIN 124

enum rasDiagnosticsRdmaTopoSource : uint8_t {
  RAS_DIAG_RDMA_TOPO_SOURCE_TOOL = 0,
  RAS_DIAG_RDMA_TOPO_SOURCE_FALLBACK = 1,
};

enum rasDiagnosticsRdmaTopoState : uint8_t {
  RAS_DIAG_RDMA_TOPO_NOT_TRIED = 0,
  RAS_DIAG_RDMA_TOPO_RAN = 1,
  // rdma_topo is present, but reports this host has no ConnectX DMA Direct functions.
  RAS_DIAG_RDMA_TOPO_NOT_APPLICABLE = 2,
  // rdma_topo could not inspect privileged PCI VPD data, so fallback checks are used.
  RAS_DIAG_RDMA_TOPO_NEEDS_ROOT = 3,
  RAS_DIAG_RDMA_TOPO_INCONCLUSIVE = 4,
  RAS_DIAG_RDMA_TOPO_UNAVAILABLE = 5,
};

enum rasDiagnosticsIommuMode : uint8_t {
  RAS_DIAG_IOMMU_MODE_OFF = 0,
  RAS_DIAG_IOMMU_MODE_PASSTHROUGH = 1,
  RAS_DIAG_IOMMU_MODE_ON = 2,
  RAS_DIAG_IOMMU_MODE_UNAVAILABLE = 3,
};

enum rasDiagnosticsIommuGroupState : uint8_t {
  RAS_DIAG_IOMMU_GROUP_NONE = 0,
  RAS_DIAG_IOMMU_GROUP_AVAILABLE = 1,
  RAS_DIAG_IOMMU_GROUP_UNAVAILABLE = 2,
};

enum rasDiagnosticsAtsState : uint8_t {
  RAS_DIAG_ATS_UNAVAILABLE = 0,
  RAS_DIAG_ATS_NEEDS_ROOT = 1,
  RAS_DIAG_ATS_OFF = 2,
  RAS_DIAG_ATS_ON = 3,
};

enum rasDiagnosticsPciNetState : uint8_t {
  RAS_DIAG_PCI_NET_UNAVAILABLE = 0,
  RAS_DIAG_PCI_NET_NO_NICS = 1,
  RAS_DIAG_PCI_NET_AVAILABLE = 2,
  RAS_DIAG_PCI_NET_TOPOLOGY_NOT_READY = 3,
};

enum rasDiagnosticsPciFallbackState : uint8_t {
  RAS_DIAG_PCI_FALLBACK_SKIPPED = 0,
  RAS_DIAG_PCI_FALLBACK_COLLECTED = 1,
  RAS_DIAG_PCI_FALLBACK_NO_NICS = 2,
  RAS_DIAG_PCI_FALLBACK_UNAVAILABLE = 3,
  RAS_DIAG_PCI_FALLBACK_TOPOLOGY_NOT_READY = 4,
};

// Gathered per-rank payload.
struct rasDiagnosticsRdmaTopoToolData {
  int16_t exitCode;
  uint8_t reportedIssue;
  char sample[RAS_DIAG_RDMA_TOPO_SAMPLE_BYTES];
};

struct rasDiagnosticsRdmaTopoData {
  rasDiagnosticsRdmaTopoSource source;
  rasDiagnosticsRdmaTopoState rdmaTopoState;
  struct rasDiagnosticsRdmaTopoToolData rdmaTopo;
};

struct rasDiagnosticsIommuGroup {
  rasDiagnosticsIommuGroupState state;
  int32_t id;
};

struct rasDiagnosticsIommuDeviceData {
  char bdf[RAS_DIAG_PCI_BDF_BYTES];
  rasDiagnosticsIommuMode mode;
  struct rasDiagnosticsIommuGroup group;
};

struct rasDiagnosticsIommuPairData {
  struct rasDiagnosticsIommuDeviceData gpu;
  struct rasDiagnosticsIommuDeviceData nic;
};

struct rasDiagnosticsIommuData {
  rasDiagnosticsPciFallbackState fallbackState;
  uint8_t netSelectionIncomplete;
  uint8_t nPairs;
  uint8_t nPairsTruncated;
  struct rasDiagnosticsIommuPairData pairs[RAS_DIAG_PCI_MAX_GPU_NIC_PAIRS];
};

struct rasDiagnosticsAtsNicData {
  char bdf[RAS_DIAG_PCI_BDF_BYTES];
  rasDiagnosticsAtsState state;
};

struct rasDiagnosticsAtsData {
  rasDiagnosticsPciFallbackState fallbackState;
  uint8_t netSelectionIncomplete;
  uint8_t nNics;
  uint8_t netSelectionTruncated;
  uint8_t lspciUnavailable;
  uint8_t lspciOutputTruncated;
  struct rasDiagnosticsAtsNicData nics[RAS_DIAG_PCI_MAX_NET_DEVS];
};

// Local collection state.
// Virtual network devices expand to one entry per selected proxy-GPU/physical-NIC pair.
struct rasDiagnosticsPciPairSnapshot {
  int64_t gpuBusId;
  char nicBdf[RAS_DIAG_PCI_BDF_BYTES];
};

struct rasDiagnosticsPciCommSnapshot {
  struct rasDiagnosticsRankHeader rank;
  rasDiagnosticsPciNetState netState;
  uint8_t netSelectionIncomplete;
  uint8_t nPairs;
  uint8_t nPairsTruncated;
  struct rasDiagnosticsPciPairSnapshot pairs[RAS_DIAG_PCI_MAX_GPU_NIC_PAIRS];
};

struct rasDiagnosticsRdmaTopoCache {
  bool outputTruncated;
  bool reportedIssue;
  rasDiagnosticsRdmaTopoState state;
  int16_t exitCode;
  char sample[RAS_DIAG_RDMA_TOPO_SAMPLE_BYTES];
};

static bool rasDiagnosticsPciValidBdf(const char* bdf) {
  size_t domainLen;
  size_t len;

  if (bdf == nullptr) return false;
  len = strnlen(bdf, RAS_DIAG_PCI_BDF_BYTES);
  if (len < 12 || len >= RAS_DIAG_PCI_BDF_BYTES) return false;

  domainLen = len - 8;
  if (domainLen < 4 || domainLen > 8) return false;
  for (size_t i = 0; i < len; i++) {
    if (i == domainLen || i == domainLen + 3) {
      if (bdf[i] != ':') return false;
    } else if (i == domainLen + 6) {
      if (bdf[i] != '.') return false;
    } else if (!isxdigit((unsigned char)bdf[i])) {
      return false;
    }
  }
  return true;
}

static bool rasDiagnosticsPciBdfFromPath(const char* pciPath, char* bdf, size_t bdfLen) {
  char tmp[RAS_DIAG_PCI_BDF_BYTES];
  const char* start;
  const char* end;
  size_t len;

  if (pciPath == nullptr || bdf == nullptr || bdfLen < RAS_DIAG_PCI_BDF_BYTES || pciPath[0] == '\0') return false;

  end = pciPath + strlen(pciPath);
  while (end > pciPath && end[-1] == '/') end--;
  start = end;
  while (start > pciPath && start[-1] != '/') start--;
  len = end - start;
  if (len >= sizeof(tmp)) return false;

  memcpy(tmp, start, len);
  tmp[len] = '\0';
  if (!rasDiagnosticsPciValidBdf(tmp)) return false;

  memcpy(bdf, tmp, len + 1);
  return true;
}

static bool rasDiagnosticsPciDevicePath(const char* bdf, char* path, size_t pathLen) {
  int n;

  if (!rasDiagnosticsPciValidBdf(bdf) || path == nullptr || pathLen == 0) return false;
  n = snprintf(path, pathLen, "/sys/bus/pci/devices/%s", bdf);
  return n >= 0 && (size_t)n < pathLen;
}

static struct rasDiagnosticsIommuGroup rasDiagnosticsPciReadIommuGroup(const char* bdf) {
  struct rasDiagnosticsIommuGroup group = {RAS_DIAG_IOMMU_GROUP_UNAVAILABLE, -1};
  char devicePath[PATH_MAX];
  char target[PATH_MAX];
  const char* start;
  char* end = nullptr;
  int fd;
  long parsed;
  ssize_t n;

  if (!rasDiagnosticsPciDevicePath(bdf, devicePath, sizeof(devicePath))) return group;
  fd = open(devicePath, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return group;

  errno = 0;
  n = readlinkat(fd, "iommu_group", target, sizeof(target) - 1);
  if (n < 0) {
    if (errno == ENOENT) group.state = RAS_DIAG_IOMMU_GROUP_NONE;
    close(fd);
    return group;
  }
  close(fd);
  if ((size_t)n == sizeof(target) - 1) return group;
  target[n] = '\0';
  // The iommu_group symlink ends with the numeric group ID.
  start = target + strlen(target);
  while (start > target && start[-1] != '/') start--;

  errno = 0;
  parsed = strtol(start, &end, 10);
  if (errno != 0 || end == start || *end != '\0' || parsed < 0 || parsed > INT32_MAX) return group;
  group.state = RAS_DIAG_IOMMU_GROUP_AVAILABLE;
  group.id = (int32_t)parsed;
  return group;
}

static rasDiagnosticsIommuMode rasDiagnosticsIommuReadMode(const struct rasDiagnosticsIommuGroup* group) {
  char path[PATH_MAX];
  char type[32];
  char* end;
  int fd;
  int n;
  ssize_t bytesRead;
  ssize_t extra = 0;

  if (group == nullptr || group->state == RAS_DIAG_IOMMU_GROUP_UNAVAILABLE) {
    return RAS_DIAG_IOMMU_MODE_UNAVAILABLE;
  }
  if (group->state == RAS_DIAG_IOMMU_GROUP_NONE) return RAS_DIAG_IOMMU_MODE_OFF;
  n = snprintf(path, sizeof(path), "/sys/kernel/iommu_groups/%d/type", group->id);
  if (n < 0 || (size_t)n >= sizeof(path)) return RAS_DIAG_IOMMU_MODE_UNAVAILABLE;

  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return RAS_DIAG_IOMMU_MODE_UNAVAILABLE;
  do {
    bytesRead = read(fd, type, sizeof(type) - 1);
  } while (bytesRead < 0 && errno == EINTR);
  if (bytesRead > 0 && type[bytesRead - 1] != '\n') {
    char unused;
    do {
      extra = read(fd, &unused, sizeof(unused));
    } while (extra < 0 && errno == EINTR);
  }
  if (close(fd) != 0 || bytesRead <= 0 || extra != 0) return RAS_DIAG_IOMMU_MODE_UNAVAILABLE;
  type[bytesRead] = '\0';

  end = type + strlen(type);
  while (end > type && isspace((unsigned char)end[-1])) end--;
  *end = '\0';

  // The group type is the effective mapping mode for this PCI device.
  if (strcmp(type, "identity") == 0) return RAS_DIAG_IOMMU_MODE_PASSTHROUGH;
  if (strcmp(type, "DMA") == 0 || strcmp(type, "DMA-FQ") == 0) return RAS_DIAG_IOMMU_MODE_ON;
  return RAS_DIAG_IOMMU_MODE_UNAVAILABLE;
}

static bool rasDiagnosticsPciSpanContainsCaseInsensitive(const char* haystack, size_t haystackLen, const char* needle) {
  size_t needleLen;

  if (haystack == nullptr || needle == nullptr) return false;
  needleLen = strlen(needle);
  if (needleLen == 0) return true;
  if (needleLen > haystackLen) return false;

  for (size_t p = 0; p + needleLen <= haystackLen; p++) {
    if (strncasecmp(haystack + p, needle, needleLen) == 0) return true;
  }
  return false;
}

static void rasDiagnosticsRdmaTopoSample(const char* output, char* sample, size_t sampleLen) {
  const char* lastLine = nullptr;
  size_t lastLen = 0;

  if (sample == nullptr || sampleLen == 0) return;
  sample[0] = '\0';
  if (output == nullptr || output[0] == '\0') return;

  // Prefer the first error-like line; otherwise retain the last nonempty line.
  for (const char* line = output; *line != '\0';) {
    const char* end = strchr(line, '\n');
    size_t len = end == nullptr ? strlen(line) : (size_t)(end - line);

    while (len > 0 && (line[len - 1] == '\r' || line[len - 1] == ' ' || line[len - 1] == '\t')) len--;
    if (len > 0) {
      lastLine = line;
      lastLen = len;
      if (rasDiagnosticsPciSpanContainsCaseInsensitive(line, len, "fail") ||
          rasDiagnosticsPciSpanContainsCaseInsensitive(line, len, "error") ||
          rasDiagnosticsPciSpanContainsCaseInsensitive(line, len, "incorrect")) {
        snprintf(sample, sampleLen, "%.*s", (int)len, line);
        return;
      }
    }
    if (end == nullptr) break;
    line = end + 1;
  }

  if (lastLine != nullptr) snprintf(sample, sampleLen, "%.*s", (int)lastLen, lastLine);
}

static bool rasDiagnosticsRdmaTopoHasLinePrefix(const char* output, const char* prefix) {
  size_t prefixLen;

  if (output == nullptr || prefix == nullptr || prefix[0] == '\0') return false;
  prefixLen = strlen(prefix);
  for (const char* line = output; *line != '\0';) {
    if (strncmp(line, prefix, prefixLen) == 0) return true;
    line = strchr(line, '\n');
    if (line == nullptr) break;
    line++;
  }
  return false;
}

static void rasDiagnosticsRdmaTopoRun(struct rasDiagnosticsRdmaTopoCache* cache) {
  char output[RAS_DIAG_RDMA_TOPO_OUTPUT_BYTES];
  bool reportedFailure;
  bool reportedSuccess;
  bool reportedToolError;
  int exitCode;

  memset(cache, 0, sizeof(*cache));
  cache->state = RAS_DIAG_RDMA_TOPO_UNAVAILABLE;
  cache->exitCode = -1;

  output[0] = '\0';
  exitCode = ncclDiagChildRun("rdma_topo check", RAS_DIAG_RDMA_TOPO_CHILD_TIMEOUT_SEC, output, sizeof(output),
                              &cache->outputTruncated);
  cache->exitCode = (int16_t)exitCode;
  if (exitCode < 0 || exitCode >= RAS_DIAG_RDMA_TOPO_CHILD_FAILURE_STATUS_MIN) {
    cache->state = RAS_DIAG_RDMA_TOPO_UNAVAILABLE;
    rasDiagnosticsRdmaTopoSample(output, cache->sample, sizeof(cache->sample));
    if (cache->sample[0] == '\0') {
      if (exitCode == RAS_DIAG_RDMA_TOPO_CHILD_FAILURE_STATUS_MIN) {
        snprintf(cache->sample, sizeof(cache->sample), "rdma_topo timed out");
      } else {
        snprintf(cache->sample, sizeof(cache->sample), "rdma_topo execution failed (status %d)", exitCode);
      }
    }
    return;
  }

  rasDiagnosticsRdmaTopoSample(output, cache->sample, sizeof(cache->sample));
  if (cache->sample[0] == '\0') {
    if (exitCode == 0) snprintf(cache->sample, sizeof(cache->sample), "rdma_topo produced no output");
    else snprintf(cache->sample, sizeof(cache->sample), "rdma_topo exited with status %d without output", exitCode);
    return;
  }

  // rdma_topo emits structured audit results as OK/FAIL lines and operational errors as E: lines.
  reportedFailure = rasDiagnosticsRdmaTopoHasLinePrefix(output, "FAIL\t");
  reportedSuccess = rasDiagnosticsRdmaTopoHasLinePrefix(output, "OK\t");
  reportedToolError = rasDiagnosticsRdmaTopoHasLinePrefix(output, "E:");

  if (strstr(output, "No ConnectX DMA Direct functions detected") != nullptr ||
      strstr(output, "No supported topology detected") != nullptr) {
    cache->state = RAS_DIAG_RDMA_TOPO_NOT_APPLICABLE;
  } else if (strstr(output, "Need access to the PCI VPD") != nullptr ||
             rasDiagnosticsPciSpanContainsCaseInsensitive(output, strlen(output), "are you root")) {
    cache->state = RAS_DIAG_RDMA_TOPO_NEEDS_ROOT;
  } else if (reportedFailure) {
    cache->state = RAS_DIAG_RDMA_TOPO_RAN;
    cache->reportedIssue = true;
  } else if (cache->outputTruncated) {
    cache->state = RAS_DIAG_RDMA_TOPO_INCONCLUSIVE;
    snprintf(cache->sample, sizeof(cache->sample), "rdma_topo output was truncated");
  } else if (reportedToolError) {
    cache->state = RAS_DIAG_RDMA_TOPO_UNAVAILABLE;
  } else if (exitCode == 0 && reportedSuccess) {
    cache->state = RAS_DIAG_RDMA_TOPO_RAN;
  }
}

static std::once_flag rasDiagnosticsRdmaTopoOnce;
static struct rasDiagnosticsRdmaTopoCache rasDiagnosticsRdmaTopoCachedResult;

static const struct rasDiagnosticsRdmaTopoCache* rasDiagnosticsRdmaTopoResult() {
  // The PCI topology is process-wide, so all communicator records share one immutable tool result.
  std::call_once(rasDiagnosticsRdmaTopoOnce, []() { rasDiagnosticsRdmaTopoRun(&rasDiagnosticsRdmaTopoCachedResult); });
  return &rasDiagnosticsRdmaTopoCachedResult;
}

static const char* rasDiagnosticsRdmaTopoStateName(rasDiagnosticsRdmaTopoState state) {
  switch (state) {
  case RAS_DIAG_RDMA_TOPO_RAN:
    return "ran";
  case RAS_DIAG_RDMA_TOPO_NOT_APPLICABLE:
    return "not applicable";
  case RAS_DIAG_RDMA_TOPO_NEEDS_ROOT:
    return "needs root";
  case RAS_DIAG_RDMA_TOPO_INCONCLUSIVE:
    return "inconclusive";
  case RAS_DIAG_RDMA_TOPO_UNAVAILABLE:
    return "unavailable";
  default:
    return "not tried";
  }
}

static const char* rasDiagnosticsIommuModeName(rasDiagnosticsIommuMode mode) {
  switch (mode) {
  case RAS_DIAG_IOMMU_MODE_OFF:
    return "disabled";
  case RAS_DIAG_IOMMU_MODE_PASSTHROUGH:
    return "passthrough";
  case RAS_DIAG_IOMMU_MODE_ON:
    return "enabled (DMA remapping)";
  default:
    return "unavailable";
  }
}

static void rasDiagnosticsPciAddPair(int64_t gpuBusId, const char* name, const char* pciPath,
                                     struct rasDiagnosticsPciCommSnapshot* snapshot) {
  char bdf[RAS_DIAG_PCI_BDF_BYTES] = {};
  char devicePath[PATH_MAX];
  char resolvedPath[PATH_MAX];

  bool bdfFound = false;
  if (name != nullptr && name[0] != '\0' && strchr(name, '/') == nullptr) {
    int n = snprintf(devicePath, sizeof(devicePath), "/sys/class/infiniband/%s/device", name);

    if (n >= 0 && (size_t)n < sizeof(devicePath) && realpath(devicePath, resolvedPath) != nullptr) {
      bdfFound = rasDiagnosticsPciBdfFromPath(resolvedPath, bdf, sizeof(bdf));
    }
  }
  if (!bdfFound) bdfFound = rasDiagnosticsPciBdfFromPath(pciPath, bdf, sizeof(bdf));
  if (!bdfFound) {
    snapshot->netSelectionIncomplete = 1;
    return;
  }
  for (int i = 0; i < snapshot->nPairs; i++) {
    const struct rasDiagnosticsPciPairSnapshot* pair = snapshot->pairs + i;

    if (pair->gpuBusId == gpuBusId && strcmp(pair->nicBdf, bdf) == 0) return;
  }
  if (snapshot->nPairs == RAS_DIAG_PCI_MAX_GPU_NIC_PAIRS) {
    snapshot->nPairsTruncated = 1;
    return;
  }
  snapshot->pairs[snapshot->nPairs].gpuBusId = gpuBusId;
  memcpy(snapshot->pairs[snapshot->nPairs++].nicBdf, bdf, sizeof(bdf));
}

static void rasDiagnosticsPciAddNetDevice(const struct ncclTopoNetPropertiesSnapshot* properties, int nProperties,
                                          int netDev, int64_t gpuBusId,
                                          struct rasDiagnosticsPciCommSnapshot* snapshot) {
  if (netDev < 0 || netDev >= nProperties || !properties[netDev].propertiesValid) {
    snapshot->netSelectionIncomplete = 1;
    return;
  }
  const struct ncclTopoNetPropertiesSnapshot* device = properties + netDev;

  if (device->vProps.ndevs < 0 || device->vProps.ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
    snapshot->netSelectionIncomplete = 1;
    return;
  }
  if (device->vProps.ndevs == 0) {
    rasDiagnosticsPciAddPair(gpuBusId, device->name, device->pciPath, snapshot);
    return;
  }
  for (int member = 0; member < device->vProps.ndevs; member++) {
    int physicalDev = device->vProps.devs[member];

    if (physicalDev < 0 || physicalDev >= nProperties || !properties[physicalDev].propertiesValid) {
      snapshot->netSelectionIncomplete = 1;
      continue;
    }
    rasDiagnosticsPciAddPair(gpuBusId, properties[physicalDev].name, properties[physicalDev].pciPath, snapshot);
  }
}

// Must be called while ncclCommsMutex prevents communicator teardown and network plugin unload.
static void rasDiagnosticsPciSnapshotNet(struct ncclComm* comm, bool requireCommInitialized,
                                         struct rasDiagnosticsPciCommSnapshot* snapshot) {
  static const int graphAlgorithms[] = {NCCL_ALGO_TREE, NCCL_ALGO_RING, NCCL_ALGO_COLLNET_DIRECT,
                                        NCCL_ALGO_COLLNET_CHAIN, NCCL_ALGO_NVLS};
  struct ncclTopoNetPropertiesSnapshot* rawProperties = nullptr;
  ncclResult_t ret;
  int nProperties = 0;
  bool selectedNetDevice = false;

  snapshot->netState = RAS_DIAG_PCI_NET_UNAVAILABLE;
  ret = ncclTopoGetNetPropertiesSnapshot(comm, requireCommInitialized, &rawProperties, &nProperties);
  if (ret == ncclInProgress) {
    snapshot->netState = RAS_DIAG_PCI_NET_TOPOLOGY_NOT_READY;
    return;
  }
  if (ret != ncclSuccess) {
    snapshot->netSelectionIncomplete = 1;
    return;
  }
  ncclUniquePtr<struct ncclTopoNetPropertiesSnapshot> properties(rawProperties);
  if (nProperties == 0) {
    snapshot->netState = RAS_DIAG_PCI_NET_NO_NICS;
    return;
  }

  // Use the same graph routes as the NET transport instead of pairing the GPU with every plugin device.
  for (size_t algorithm = 0; algorithm < sizeof(graphAlgorithms) / sizeof(graphAlgorithms[0]); algorithm++) {
    int algorithmId = graphAlgorithms[algorithm];
    struct ncclTopoGraph* graph = comm->graphs + algorithmId;

    if ((algorithmId == NCCL_ALGO_COLLNET_DIRECT || algorithmId == NCCL_ALGO_COLLNET_CHAIN) &&
        !comm->config.collnetEnable) {
      continue;
    }
    if (algorithmId == NCCL_ALGO_NVLS && !comm->nvlsSupport) continue;
    if (graph->nChannels <= 0) continue;
    if (graph->nChannels > MAXCHANNELS) {
      snapshot->netSelectionIncomplete = 1;
      continue;
    }
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
      int nLocalGpus = comm->topo->nodes[GPU].count;
      bool isHead = false;

      if (nLocalGpus <= 0 || nLocalGpus > NCCL_TOPO_MAX_NODES) {
        snapshot->netSelectionIncomplete = 1;
        continue;
      }
      for (int channel = 0; channel < graph->nChannels; channel++) {
        if (graph->intra[channel * nLocalGpus] == comm->rank) isHead = true;
      }
      if (!isHead) continue;
    }

    for (int channel = 0; channel < graph->nChannels; channel++) {
      int netDev = -1;
      int proxyRank = -1;

      if (ncclTopoGetNetDev(comm, comm->rank, graph, channel, -1, nullptr, &netDev, &proxyRank) != ncclSuccess ||
          proxyRank < 0 || proxyRank >= comm->nRanks ||
          comm->peerInfo[proxyRank].hostHash != comm->peerInfo[comm->rank].hostHash) {
        snapshot->netSelectionIncomplete = 1;
        continue;
      }

      selectedNetDevice = true;
      rasDiagnosticsPciAddNetDevice(properties.get(), nProperties, netDev, comm->peerInfo[proxyRank].busId, snapshot);
    }
  }

  if (snapshot->nPairs > 0) snapshot->netState = RAS_DIAG_PCI_NET_AVAILABLE;
  else if (!selectedNetDevice && !snapshot->netSelectionIncomplete) snapshot->netState = RAS_DIAG_PCI_NET_NO_NICS;
}

static ncclResult_t rasDiagnosticsPciCollectSnapshots(const struct rasDiagnosticsContext* ctx, bool collectNetDevices,
                                                      ncclUniquePtr<struct rasDiagnosticsPciCommSnapshot>& snapshots,
                                                      int* nSnapshots) {
  int count = 0;
  int snapshotIdx = 0;

  if (ctx == nullptr || nSnapshots == nullptr) {
    WARN("RAS PCI diagnostics snapshot collection received invalid arguments");
    return ncclInternalError;
  }
  *nSnapshots = 0;
  // Filtered requests originate from the initialization-time trigger.
  const bool requireCommInitialized = !ctx->hasCommFilter;

  std::lock_guard<std::mutex> lock(ncclCommsMutex);
  for (int i = 0; i < nNcclComms; i++) {
    struct ncclComm* comm = ncclComms[i];

    if (comm == nullptr) continue;
    if (!COMPILER_ATOMIC_LOAD(&comm->peerInfoValid, std::memory_order_acquire)) continue;
    if (rasDiagnosticsCommMatchesContext(ctx, comm)) count++;
  }
  if (count == 0) return ncclSuccess;

  NCCLCHECK(ncclCalloc(snapshots, count));
  for (int i = 0; i < nNcclComms && snapshotIdx < count; i++) {
    struct ncclComm* comm = ncclComms[i];
    struct rasDiagnosticsCommSnapshot base;

    if (comm == nullptr) continue;
    if (!COMPILER_ATOMIC_LOAD(&comm->peerInfoValid, std::memory_order_acquire)) continue;
    if (!rasDiagnosticsCommMatchesContext(ctx, comm)) continue;

    rasDiagnosticsCommSnapshotInit(&base, comm);
    snapshots.get()[snapshotIdx].rank = base.rank;
    if (collectNetDevices) rasDiagnosticsPciSnapshotNet(comm, requireCommInitialized, snapshots.get() + snapshotIdx);
    snapshotIdx++;
  }
  *nSnapshots = snapshotIdx;
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsRdmaTopoFillLocalData(const struct rasDiagnosticsCommSnapshot* comm,
                                                        void* checkData) {
  const struct rasDiagnosticsRdmaTopoCache* rdmaTopo = rasDiagnosticsRdmaTopoResult();
  struct rasDiagnosticsRdmaTopoData* topoData = (struct rasDiagnosticsRdmaTopoData*)checkData;

  (void)comm;
  memset(topoData, 0, sizeof(*topoData));
  topoData->source =
    rdmaTopo->state == RAS_DIAG_RDMA_TOPO_RAN ? RAS_DIAG_RDMA_TOPO_SOURCE_TOOL : RAS_DIAG_RDMA_TOPO_SOURCE_FALLBACK;
  topoData->rdmaTopoState = rdmaTopo->state;
  topoData->rdmaTopo.exitCode = rdmaTopo->exitCode;
  topoData->rdmaTopo.reportedIssue = rdmaTopo->reportedIssue;
  snprintf(topoData->rdmaTopo.sample, sizeof(topoData->rdmaTopo.sample), "%s", rdmaTopo->sample);
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsRdmaTopoCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                struct rasDiagnosticsLocalData* data) {
  return rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsRdmaTopoData),
                                           rasDiagnosticsRdmaTopoFillLocalData, data);
}

static const struct rasDiagnosticsRdmaTopoData* rasDiagnosticsRdmaTopoDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsRdmaTopoData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

// Summarizer-only aggregation state.
struct rasDiagnosticsPciRankSet {
  int ranks[RAS_DIAG_RANK_SET_MAX];
  int nStored;
  int nTotal;
};

struct rasDiagnosticsPciNicIssue {
  int count;
  int nAffectedRanks;
  int lastRank;
  int firstRank;
  char firstBdf[RAS_DIAG_PCI_BDF_BYTES];
};

static void rasDiagnosticsPciRankSetAdd(struct rasDiagnosticsPciRankSet* rankSet, int rank) {
  if (rankSet->nStored < RAS_DIAG_RANK_SET_MAX) rankSet->ranks[rankSet->nStored++] = rank;
  rankSet->nTotal++;
}

static void rasDiagnosticsPciRecordNicIssue(struct rasDiagnosticsPciNicIssue* issue, int rank, const char* bdf) {
  if (issue->count == 0) {
    issue->firstRank = rank;
    snprintf(issue->firstBdf, sizeof(issue->firstBdf), "%s", bdf[0] == '\0' ? "unknown" : bdf);
  }
  if (issue->count == 0 || issue->lastRank != rank) issue->nAffectedRanks++;
  issue->lastRank = rank;
  issue->count++;
}

static bool rasDiagnosticsPciStringTerminated(const char* text, size_t len) {
  return text != nullptr && memchr(text, '\0', len) != nullptr;
}

static bool rasDiagnosticsPciBdfPayloadValid(const char* bdf) {
  if (!rasDiagnosticsPciStringTerminated(bdf, RAS_DIAG_PCI_BDF_BYTES)) return false;
  return rasDiagnosticsPciValidBdf(bdf);
}

static bool rasDiagnosticsRdmaTopoPayloadValid(const struct rasDiagnosticsRdmaTopoData* topoData) {
  if (topoData == nullptr) return false;
  if (topoData->source > RAS_DIAG_RDMA_TOPO_SOURCE_FALLBACK) return false;
  if (topoData->rdmaTopoState > RAS_DIAG_RDMA_TOPO_UNAVAILABLE) return false;
  if (topoData->rdmaTopo.reportedIssue > 1) return false;

  switch (topoData->source) {
  case RAS_DIAG_RDMA_TOPO_SOURCE_TOOL:
    return topoData->rdmaTopoState == RAS_DIAG_RDMA_TOPO_RAN && topoData->rdmaTopo.exitCode >= 0 &&
           topoData->rdmaTopo.exitCode < RAS_DIAG_RDMA_TOPO_CHILD_FAILURE_STATUS_MIN &&
           (topoData->rdmaTopo.reportedIssue || topoData->rdmaTopo.exitCode == 0) &&
           topoData->rdmaTopo.sample[0] != '\0' &&
           rasDiagnosticsPciStringTerminated(topoData->rdmaTopo.sample, sizeof(topoData->rdmaTopo.sample));
  case RAS_DIAG_RDMA_TOPO_SOURCE_FALLBACK:
    if (topoData->rdmaTopoState != RAS_DIAG_RDMA_TOPO_NOT_APPLICABLE &&
        topoData->rdmaTopoState != RAS_DIAG_RDMA_TOPO_NEEDS_ROOT &&
        topoData->rdmaTopoState != RAS_DIAG_RDMA_TOPO_INCONCLUSIVE &&
        topoData->rdmaTopoState != RAS_DIAG_RDMA_TOPO_UNAVAILABLE) {
      return false;
    }
    return !topoData->rdmaTopo.reportedIssue && topoData->rdmaTopo.sample[0] != '\0' &&
           rasDiagnosticsPciStringTerminated(topoData->rdmaTopo.sample, sizeof(topoData->rdmaTopo.sample));
  default:
    return false;
  }
}

static ncclResult_t rasDiagnosticsPciReportTopologyNotReady(const struct rasDiagnosticsReporter* reporter,
                                                            const char* check,
                                                            const struct rasDiagnosticsRankHeader* startRank,
                                                            int nRanks) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "%s: incomplete for comm 0x%lx because communicator topology was not ready on "
                              "%d/%d ranks",
                              check, startRank->commId.commHash, nRanks, startRank->commNRanks);
}

// Report rdma_topo results for ranks in one communicator, ignoring records that used another source.
static ncclResult_t rasDiagnosticsRdmaTopoReportResults(const struct rasDiagnosticsReporter* reporter,
                                                        const char* records, size_t recordStride, int start, int end,
                                                        const struct rasDiagnosticsRankHeader* startRank,
                                                        const char* tag) {
  struct rasDiagnosticsPciRankSet issueRanks = {};
  int nRdmaTopo = 0;
  int firstIssueRank = -1;
  const struct rasDiagnosticsRdmaTopoToolData* firstIssue = nullptr;

  for (int i = start; i < end; i++) {
    const char* record = records + (size_t)i * recordStride;
    const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
    const struct rasDiagnosticsRdmaTopoData* topoData = rasDiagnosticsRdmaTopoDataFromRecord(record);

    if (topoData->source != RAS_DIAG_RDMA_TOPO_SOURCE_TOOL) continue;
    nRdmaTopo++;
    if (topoData->rdmaTopo.reportedIssue) {
      rasDiagnosticsPciRankSetAdd(&issueRanks, rank->commRank);
      if (firstIssue == nullptr) {
        firstIssue = &topoData->rdmaTopo;
        firstIssueRank = rank->commRank;
      }
    }
  }

  if (nRdmaTopo == 0) return ncclSuccess;
  if (issueRanks.nTotal == 0) {
    return rasDiagnosticsReport(reporter, tag, "rdma_topo check: passed on %d/%d ranks in comm 0x%lx", nRdmaTopo,
                                startRank->commNRanks, startRank->commId.commHash);
  }

  char rankSet[128];
  rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), issueRanks.ranks, issueRanks.nStored, issueRanks.nTotal);
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "rdma_topo check: failed on rank(s) %s across %d/%d ranks in comm 0x%lx; "
                              "first rank %d exit %d: %s",
                              rankSet, nRdmaTopo, startRank->commNRanks, startRank->commId.commHash, firstIssueRank,
                              (int)firstIssue->exitCode, firstIssue->sample);
}

// Report why rdma_topo could not provide a conclusive result for ranks that use the fallbacks.
static ncclResult_t rasDiagnosticsRdmaTopoReportFallbackReasons(const struct rasDiagnosticsReporter* reporter,
                                                                const char* records, size_t recordStride, int start,
                                                                int end,
                                                                const struct rasDiagnosticsRankHeader* startRank) {
  const struct rasDiagnosticsRdmaTopoData* first = nullptr;
  int firstRank = -1;
  int nFallback = 0;
  bool mixedResults = false;

  for (int i = start; i < end; i++) {
    const char* record = records + (size_t)i * recordStride;
    const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
    const struct rasDiagnosticsRdmaTopoData* topoData = rasDiagnosticsRdmaTopoDataFromRecord(record);

    if (topoData->source != RAS_DIAG_RDMA_TOPO_SOURCE_FALLBACK) continue;
    if (first == nullptr) {
      first = topoData;
      firstRank = rank->commRank;
    } else if (topoData->rdmaTopoState != first->rdmaTopoState ||
               strcmp(topoData->rdmaTopo.sample, first->rdmaTopo.sample) != 0) {
      mixedResults = true;
    }
    nFallback++;
  }
  if (nFallback == 0) return ncclSuccess;

  if (mixedResults) {
    return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                "rdma_topo check: not usable on %d/%d ranks in comm 0x%lx "
                                "(mixed results); first rank %d: %s",
                                nFallback, startRank->commNRanks, startRank->commId.commHash, firstRank,
                                first->rdmaTopo.sample);
  }
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "rdma_topo check: not usable on %d/%d ranks in comm 0x%lx (%s): %s", nFallback,
                              startRank->commNRanks, startRank->commId.commHash,
                              rasDiagnosticsRdmaTopoStateName(first->rdmaTopoState), first->rdmaTopo.sample);
}

// Validate and group gathered records by communicator, then report each communicator's rdma_topo results.
ncclResult_t rasDiagnosticsRdmaTopoSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsRdmaTopoData));
  int nRecords;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics rdma_topo check received invalid reporter");
    return ncclInternalError;
  }
  if (ctx == nullptr || (ctx->hasCommFilter && ctx->commNRanks <= 0)) {
    WARN("RAS diagnostics rdma_topo check received invalid context");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics rdma_topo check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics rdma_topo check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  // Keep each communicator's records contiguous and ordered by rank.
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + (size_t)start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    int nRdmaTopo = 0, nFallback = 0;
    int previousCommRank = -1;
    int end = start;

    if (ctx->hasCommFilter && (rasDiagnosticsCommIdCompare(&startRank->commId, &ctx->commFilter) != 0 ||
                               startRank->commNRanks != ctx->commNRanks)) {
      WARN("RAS diagnostics rdma_topo check received records outside the requested communicator");
      ret = ncclInternalError;
      goto exit;
    }

    while (end < nRecords) {
      const char* record = records + (size_t)end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsRdmaTopoData* topoData;

      if (end > start && rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      if (rank->commNRanks <= 0 || rank->commRank < 0 || rank->commRank >= rank->commNRanks ||
          rank->commNRanks != startRank->commNRanks || rank->commRank == previousCommRank) {
        WARN("RAS diagnostics rdma_topo check received malformed rank header");
        ret = ncclInternalError;
        goto exit;
      }
      previousCommRank = rank->commRank;
      topoData = rasDiagnosticsRdmaTopoDataFromRecord(record);
      if (!rasDiagnosticsRdmaTopoPayloadValid(topoData)) {
        WARN("RAS diagnostics rdma_topo check received malformed payload");
        ret = ncclInternalError;
        goto exit;
      }
      if (topoData->source == RAS_DIAG_RDMA_TOPO_SOURCE_TOOL) nRdmaTopo++;
      else nFallback++;
      end++;
    }

    if (end - start != startRank->commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "rdma_topo check", startRank, end - start), ret, exit);
    } else if (nRdmaTopo == startRank->commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsRdmaTopoReportResults(reporter, records, recordStride, start, end, startRank,
                                                        RAS_DIAG_TAG_OK),
                    ret, exit);
    } else if (nFallback == startRank->commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsRdmaTopoReportFallbackReasons(reporter, records, recordStride, start, end, startRank),
                    ret, exit);
    } else {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "rdma_topo check: mixed results across %d ranks in comm 0x%lx "
                                         "(conclusive=%d not-usable=%d)",
                                         startRank->commNRanks, startRank->commId.commHash, nRdmaTopo, nFallback),
                    ret, exit);
      if (nRdmaTopo > 0) {
        NCCLCHECKGOTO(rasDiagnosticsRdmaTopoReportResults(reporter, records, recordStride, start, end, startRank,
                                                          RAS_DIAG_TAG_INFO),
                      ret, exit);
      }
      if (nFallback > 0) {
        NCCLCHECKGOTO(rasDiagnosticsRdmaTopoReportFallbackReasons(reporter, records, recordStride, start, end,
                                                                  startRank),
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
// IOMMU mode check.
// *************************************************************************

static rasDiagnosticsPciFallbackState rasDiagnosticsPciFallbackStateForSnapshot(
  const struct rasDiagnosticsPciCommSnapshot* snapshot, const struct rasDiagnosticsRdmaTopoCache* rdmaTopo) {
  if (rdmaTopo == nullptr) return RAS_DIAG_PCI_FALLBACK_UNAVAILABLE;
  // A conclusive rdma_topo pass or failure supersedes the heuristic fallbacks.
  if (rdmaTopo->state == RAS_DIAG_RDMA_TOPO_RAN) return RAS_DIAG_PCI_FALLBACK_SKIPPED;
  if (snapshot->netState == RAS_DIAG_PCI_NET_TOPOLOGY_NOT_READY) {
    return RAS_DIAG_PCI_FALLBACK_TOPOLOGY_NOT_READY;
  }
  if (snapshot->netState == RAS_DIAG_PCI_NET_NO_NICS) return RAS_DIAG_PCI_FALLBACK_NO_NICS;
  return snapshot->netState == RAS_DIAG_PCI_NET_AVAILABLE ? RAS_DIAG_PCI_FALLBACK_COLLECTED :
                                                            RAS_DIAG_PCI_FALLBACK_UNAVAILABLE;
}

static bool rasDiagnosticsPciFallbackNeeded(const struct rasDiagnosticsPciCommSnapshot* snapshots, int nSnapshots,
                                            const struct rasDiagnosticsRdmaTopoCache* rdmaTopo) {
  for (int i = 0; i < nSnapshots; i++) {
    if (rasDiagnosticsPciFallbackStateForSnapshot(snapshots + i, rdmaTopo) == RAS_DIAG_PCI_FALLBACK_COLLECTED) {
      return true;
    }
  }
  return false;
}

static void rasDiagnosticsIommuFillDevice(const char* bdf, struct rasDiagnosticsIommuDeviceData* device) {
  memset(device, 0, sizeof(*device));
  snprintf(device->bdf, sizeof(device->bdf), "%s", bdf);
  device->group = rasDiagnosticsPciReadIommuGroup(bdf);
  device->mode = rasDiagnosticsIommuReadMode(&device->group);
}

static ncclResult_t rasDiagnosticsIommuFillRecord(const struct rasDiagnosticsPciCommSnapshot* snapshot,
                                                  const struct rasDiagnosticsRdmaTopoCache* rdmaTopo,
                                                  struct rasDiagnosticsIommuData* iommuData) {
  memset(iommuData, 0, sizeof(*iommuData));
  iommuData->fallbackState = rasDiagnosticsPciFallbackStateForSnapshot(snapshot, rdmaTopo);
  if (iommuData->fallbackState != RAS_DIAG_PCI_FALLBACK_COLLECTED) return ncclSuccess;

  iommuData->netSelectionIncomplete = snapshot->netSelectionIncomplete;
  iommuData->nPairs = snapshot->nPairs;
  iommuData->nPairsTruncated = snapshot->nPairsTruncated;
  for (int pairIdx = 0; pairIdx < snapshot->nPairs; pairIdx++) {
    const struct rasDiagnosticsPciPairSnapshot* snapshotPair = snapshot->pairs + pairIdx;
    struct rasDiagnosticsIommuPairData* pair = iommuData->pairs + pairIdx;
    char gpuBdf[RAS_DIAG_PCI_BDF_BYTES];

    NCCLCHECK(int64ToBusId(snapshotPair->gpuBusId, gpuBdf));
    rasDiagnosticsIommuFillDevice(gpuBdf, &pair->gpu);
    rasDiagnosticsIommuFillDevice(snapshotPair->nicBdf, &pair->nic);
  }
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsIommuCollectLocal(const struct rasDiagnosticsContext* ctx,
                                             struct rasDiagnosticsLocalData* data) {
  ncclUniquePtr<struct rasDiagnosticsPciCommSnapshot> snapshots;
  ncclUniquePtr<char> records;
  const struct rasDiagnosticsRdmaTopoCache* rdmaTopo = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsIommuData));
  int nRecords = 0;
  size_t nBytes;

  if (data == nullptr) {
    WARN("RAS diagnostics IOMMU mode check local data output is null");
    return ncclInternalError;
  }
  memset(data, 0, sizeof(*data));
  if (ctx == nullptr || recordStride > (size_t)INT_MAX) {
    WARN("RAS diagnostics IOMMU mode check local collection received invalid arguments");
    return ncclInternalError;
  }

  rdmaTopo = rasDiagnosticsRdmaTopoResult();
  NCCLCHECK(rasDiagnosticsPciCollectSnapshots(ctx, rdmaTopo->state != RAS_DIAG_RDMA_TOPO_RAN, snapshots, &nRecords));
  if (nRecords == 0) return ncclSuccess;
  if ((size_t)nRecords > (size_t)INT_MAX / recordStride) {
    WARN("RAS diagnostics IOMMU mode check local data too large");
    return ncclInternalError;
  }
  nBytes = (size_t)nRecords * recordStride;
  NCCLCHECK(ncclCalloc(records, nBytes));
  for (int i = 0; i < nRecords; i++) {
    char* record = records.get() + (size_t)i * recordStride;

    memcpy(record, &snapshots.get()[i].rank, sizeof(snapshots.get()[i].rank));
    NCCLCHECK(rasDiagnosticsIommuFillRecord(
      snapshots.get() + i, rdmaTopo,
      (struct rasDiagnosticsIommuData*)(record + sizeof(struct rasDiagnosticsRankHeader))));
  }

  data->records = records.release();
  data->recordsBytes = (int)nBytes;
  data->recordStride = (int)recordStride;
  data->nRecords = nRecords;
  return ncclSuccess;
}

static const struct rasDiagnosticsIommuData* rasDiagnosticsIommuDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsIommuData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

static bool rasDiagnosticsIommuGroupValid(const struct rasDiagnosticsIommuGroup* group) {
  if (group->state > RAS_DIAG_IOMMU_GROUP_UNAVAILABLE) return false;
  return group->state == RAS_DIAG_IOMMU_GROUP_AVAILABLE ? group->id >= 0 : group->id == -1;
}

static bool rasDiagnosticsIommuDeviceValid(const struct rasDiagnosticsIommuDeviceData* device) {
  if (!rasDiagnosticsPciBdfPayloadValid(device->bdf) || device->mode > RAS_DIAG_IOMMU_MODE_UNAVAILABLE ||
      !rasDiagnosticsIommuGroupValid(&device->group)) {
    return false;
  }
  if (device->group.state == RAS_DIAG_IOMMU_GROUP_NONE) return device->mode == RAS_DIAG_IOMMU_MODE_OFF;
  if (device->group.state == RAS_DIAG_IOMMU_GROUP_UNAVAILABLE) {
    return device->mode == RAS_DIAG_IOMMU_MODE_UNAVAILABLE;
  }
  return device->mode != RAS_DIAG_IOMMU_MODE_OFF;
}

static bool rasDiagnosticsIommuPayloadValid(const struct rasDiagnosticsIommuData* iommuData) {
  if (iommuData == nullptr || iommuData->fallbackState > RAS_DIAG_PCI_FALLBACK_TOPOLOGY_NOT_READY ||
      iommuData->netSelectionIncomplete > 1 || iommuData->nPairs > RAS_DIAG_PCI_MAX_GPU_NIC_PAIRS ||
      iommuData->nPairsTruncated > 1) {
    return false;
  }
  if (iommuData->fallbackState != RAS_DIAG_PCI_FALLBACK_COLLECTED) {
    return iommuData->nPairs == 0 && !iommuData->nPairsTruncated && !iommuData->netSelectionIncomplete;
  }
  if (iommuData->nPairs == 0 || (iommuData->nPairsTruncated && iommuData->nPairs != RAS_DIAG_PCI_MAX_GPU_NIC_PAIRS)) {
    return false;
  }
  for (int i = 0; i < iommuData->nPairs; i++) {
    const struct rasDiagnosticsIommuPairData* pair = iommuData->pairs + i;

    if (!rasDiagnosticsIommuDeviceValid(&pair->gpu) || !rasDiagnosticsIommuDeviceValid(&pair->nic)) return false;
    for (int j = 0; j < i; j++) {
      const struct rasDiagnosticsIommuPairData* previous = iommuData->pairs + j;

      if (strcmp(pair->gpu.bdf, previous->gpu.bdf) == 0 && strcmp(pair->nic.bdf, previous->nic.bdf) == 0) {
        return false;
      }
    }
  }
  return true;
}

struct rasDiagnosticsIommuModeOccurrence {
  int count;
  int firstRank;
  char firstBdf[RAS_DIAG_PCI_BDF_BYTES];
};

struct rasDiagnosticsIommuGroupIssue {
  int count;
  int firstRank;
  char firstNicBdf[RAS_DIAG_PCI_BDF_BYTES];
  int32_t firstGpuGroup;
  int32_t firstNicGroup;
};

struct rasDiagnosticsIommuSummary {
  int nFallback;
  int nUnavailable;
  int nDevices;
  int nPairs;
  int nGroupSkipped;
  int nGroupMatches;
  struct rasDiagnosticsIommuModeOccurrence modes[RAS_DIAG_IOMMU_MODE_UNAVAILABLE + 1];
  struct rasDiagnosticsIommuGroupIssue groupMismatch;
  struct rasDiagnosticsIommuGroupIssue groupUnavailable;
  struct rasDiagnosticsPciRankSet truncated;
  struct rasDiagnosticsPciRankSet incomplete;
};

static void rasDiagnosticsIommuRecordMode(struct rasDiagnosticsIommuSummary* summary, int rank,
                                          const struct rasDiagnosticsIommuDeviceData* device) {
  struct rasDiagnosticsIommuModeOccurrence* occurrence = summary->modes + device->mode;

  if (occurrence->count == 0) {
    occurrence->firstRank = rank;
    snprintf(occurrence->firstBdf, sizeof(occurrence->firstBdf), "%s", device->bdf);
  }
  occurrence->count++;
  summary->nDevices++;
}

static void rasDiagnosticsIommuRecordGroupIssue(struct rasDiagnosticsIommuGroupIssue* issue, int rank,
                                                const char* nicBdf, int32_t gpuGroup, int32_t nicGroup) {
  if (issue->count == 0) {
    issue->firstRank = rank;
    issue->firstGpuGroup = gpuGroup;
    issue->firstNicGroup = nicGroup;
    snprintf(issue->firstNicBdf, sizeof(issue->firstNicBdf), "%s", nicBdf);
  }
  issue->count++;
}

static void rasDiagnosticsIommuInspectRecord(const struct rasDiagnosticsRankHeader* rank,
                                             const struct rasDiagnosticsIommuData* iommuData,
                                             struct rasDiagnosticsIommuSummary* summary) {
  for (int pairIdx = 0; pairIdx < iommuData->nPairs; pairIdx++) {
    const struct rasDiagnosticsIommuPairData* pair = iommuData->pairs + pairIdx;
    const struct rasDiagnosticsIommuGroup* gpuGroup = &pair->gpu.group;
    const struct rasDiagnosticsIommuGroup* nicGroup = &pair->nic.group;
    bool gpuSeen = false;
    bool nicSeen = false;

    for (int previousIdx = 0; previousIdx < pairIdx; previousIdx++) {
      const struct rasDiagnosticsIommuPairData* previous = iommuData->pairs + previousIdx;

      if (strcmp(pair->gpu.bdf, previous->gpu.bdf) == 0) gpuSeen = true;
      if (strcmp(pair->nic.bdf, previous->nic.bdf) == 0) nicSeen = true;
    }
    if (!gpuSeen) rasDiagnosticsIommuRecordMode(summary, rank->commRank, &pair->gpu);
    if (!nicSeen) rasDiagnosticsIommuRecordMode(summary, rank->commRank, &pair->nic);
    summary->nPairs++;
    // Without an active IOMMU group there is no group-isolation relationship to compare.
    if (gpuGroup->state == RAS_DIAG_IOMMU_GROUP_NONE || nicGroup->state == RAS_DIAG_IOMMU_GROUP_NONE) {
      summary->nGroupSkipped++;
      continue;
    }
    if (gpuGroup->state == RAS_DIAG_IOMMU_GROUP_UNAVAILABLE || nicGroup->state == RAS_DIAG_IOMMU_GROUP_UNAVAILABLE) {
      rasDiagnosticsIommuRecordGroupIssue(&summary->groupUnavailable, rank->commRank, pair->nic.bdf, gpuGroup->id,
                                          nicGroup->id);
    } else if (gpuGroup->id != nicGroup->id) {
      rasDiagnosticsIommuRecordGroupIssue(&summary->groupMismatch, rank->commRank, pair->nic.bdf, gpuGroup->id,
                                          nicGroup->id);
    } else {
      summary->nGroupMatches++;
    }
  }
  if (iommuData->nPairsTruncated) rasDiagnosticsPciRankSetAdd(&summary->truncated, rank->commRank);
  if (iommuData->netSelectionIncomplete) rasDiagnosticsPciRankSetAdd(&summary->incomplete, rank->commRank);
}

static ncclResult_t rasDiagnosticsIommuReportMode(const struct rasDiagnosticsReporter* reporter,
                                                  const struct rasDiagnosticsRankHeader* startRank,
                                                  const struct rasDiagnosticsIommuSummary* summary) {
  int knownMode = -1;
  int nKnownModes = 0;
  bool complete = summary->nFallback == startRank->commNRanks && summary->truncated.nTotal == 0 &&
                  summary->incomplete.nTotal == 0 && summary->modes[RAS_DIAG_IOMMU_MODE_UNAVAILABLE].count == 0;

  for (int mode = 0; mode < RAS_DIAG_IOMMU_MODE_UNAVAILABLE; mode++) {
    if (summary->modes[mode].count == 0) continue;
    knownMode = mode;
    nKnownModes++;
  }

  if (nKnownModes == 1 && summary->modes[RAS_DIAG_IOMMU_MODE_UNAVAILABLE].count == 0) {
    const char* tag = complete && knownMode != RAS_DIAG_IOMMU_MODE_ON ? RAS_DIAG_TAG_OK : RAS_DIAG_TAG_INFO;
    return rasDiagnosticsReport(reporter, tag,
                                "IOMMU mode: %s for %d relevant device(s) across %d/%d ranks in comm 0x%lx",
                                rasDiagnosticsIommuModeName((rasDiagnosticsIommuMode)knownMode), summary->nDevices,
                                summary->nFallback, startRank->commNRanks, startRank->commId.commHash);
  }

  const char* result = nKnownModes == 0 ? "unavailable" : nKnownModes == 1 ? "incomplete" : "mismatch";
  NCCLCHECK(rasDiagnosticsReport(
    reporter, RAS_DIAG_TAG_INFO, "IOMMU mode: %s across %d relevant device(s) on %d/%d ranks in comm 0x%lx", result,
    summary->nDevices, summary->nFallback, startRank->commNRanks, startRank->commId.commHash));
  for (int mode = 0; mode <= RAS_DIAG_IOMMU_MODE_UNAVAILABLE; mode++) {
    const struct rasDiagnosticsIommuModeOccurrence* occurrence = summary->modes + mode;

    if (occurrence->count == 0) continue;
    NCCLCHECK(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                   "IOMMU mode: %s on %d device(s); first rank %d device %s in comm 0x%lx",
                                   rasDiagnosticsIommuModeName((rasDiagnosticsIommuMode)mode), occurrence->count,
                                   occurrence->firstRank, occurrence->firstBdf, startRank->commId.commHash));
  }
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsIommuReportGroups(const struct rasDiagnosticsReporter* reporter,
                                                    const struct rasDiagnosticsRankHeader* startRank,
                                                    const struct rasDiagnosticsIommuSummary* summary) {
  int nCompared = summary->nGroupMatches + summary->groupMismatch.count;
  bool complete = summary->nFallback == startRank->commNRanks && summary->nGroupSkipped == 0 &&
                  summary->groupUnavailable.count == 0 && summary->truncated.nTotal == 0 &&
                  summary->incomplete.nTotal == 0;

  if (nCompared == 0 && summary->groupUnavailable.count == 0) return ncclSuccess;
  if (summary->groupMismatch.count > 0) {
    NCCLCHECK(rasDiagnosticsReport(
      reporter, RAS_DIAG_TAG_INFO,
      "IOMMU groups: GPU and NIC groups differ on %d pair(s) in comm 0x%lx; first rank %d GPU group %d "
      "NIC %s group %d",
      summary->groupMismatch.count, startRank->commId.commHash, summary->groupMismatch.firstRank,
      summary->groupMismatch.firstGpuGroup, summary->groupMismatch.firstNicBdf, summary->groupMismatch.firstNicGroup));
  } else if (summary->groupUnavailable.count > 0) {
    NCCLCHECK(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                   "IOMMU groups: comparison unavailable on %d GPU/NIC pair(s) in comm 0x%lx",
                                   summary->groupUnavailable.count, startRank->commId.commHash));
  } else {
    NCCLCHECK(rasDiagnosticsReport(reporter, complete ? RAS_DIAG_TAG_OK : RAS_DIAG_TAG_INFO,
                                   "IOMMU groups: GPU and NIC groups match on %d pair(s) across %d/%d ranks "
                                   "in comm 0x%lx",
                                   summary->nGroupMatches, summary->nFallback, startRank->commNRanks,
                                   startRank->commId.commHash));
  }

  if (summary->groupUnavailable.count > 0) {
    NCCLCHECK(rasDiagnosticsReport(
      reporter, RAS_DIAG_TAG_INFO,
      "IOMMU groups: unavailable on %d pair(s) in comm 0x%lx; first rank %d GPU group %d NIC %s group %d",
      summary->groupUnavailable.count, startRank->commId.commHash, summary->groupUnavailable.firstRank,
      summary->groupUnavailable.firstGpuGroup, summary->groupUnavailable.firstNicBdf,
      summary->groupUnavailable.firstNicGroup));
  }
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsIommuSummarize(const struct rasDiagnosticsContext* ctx,
                                          const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsIommuData));
  int nRecords;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics IOMMU mode check received invalid reporter");
    return ncclInternalError;
  }
  if (ctx == nullptr || (ctx->hasCommFilter && ctx->commNRanks <= 0)) {
    WARN("RAS diagnostics IOMMU mode check received invalid context");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr || nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics IOMMU mode check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + (size_t)start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    struct rasDiagnosticsIommuSummary summary = {};
    int nTopologyNotReady = 0;
    int previousCommRank = -1;
    int end = start;

    if (ctx->hasCommFilter && (rasDiagnosticsCommIdCompare(&startRank->commId, &ctx->commFilter) != 0 ||
                               startRank->commNRanks != ctx->commNRanks)) {
      WARN("RAS diagnostics IOMMU mode check received records outside the requested communicator");
      ret = ncclInternalError;
      goto exit;
    }

    while (end < nRecords) {
      const char* record = records + (size_t)end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsIommuData* iommuData;

      if (end > start && rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      if (rank->commNRanks <= 0 || rank->commRank < 0 || rank->commRank >= rank->commNRanks ||
          rank->commNRanks != startRank->commNRanks || rank->commRank == previousCommRank) {
        WARN("RAS diagnostics IOMMU mode check received malformed rank header");
        ret = ncclInternalError;
        goto exit;
      }
      previousCommRank = rank->commRank;
      iommuData = rasDiagnosticsIommuDataFromRecord(record);
      if (!rasDiagnosticsIommuPayloadValid(iommuData)) {
        WARN("RAS diagnostics IOMMU mode check received malformed payload");
        ret = ncclInternalError;
        goto exit;
      }

      if (iommuData->fallbackState == RAS_DIAG_PCI_FALLBACK_COLLECTED) {
        summary.nFallback++;
        rasDiagnosticsIommuInspectRecord(rank, iommuData, &summary);
      } else if (iommuData->fallbackState == RAS_DIAG_PCI_FALLBACK_UNAVAILABLE) {
        summary.nUnavailable++;
      } else if (iommuData->fallbackState == RAS_DIAG_PCI_FALLBACK_TOPOLOGY_NOT_READY) {
        nTopologyNotReady++;
      }
      end++;
    }

    if (end - start != startRank->commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "IOMMU mode", startRank, end - start), ret, exit);
    } else if (summary.nFallback > 0) {
      NCCLCHECKGOTO(rasDiagnosticsIommuReportMode(reporter, startRank, &summary), ret, exit);
      NCCLCHECKGOTO(rasDiagnosticsIommuReportGroups(reporter, startRank, &summary), ret, exit);
    } else if (summary.nUnavailable > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "IOMMU mode: unavailable on %d/%d ranks in comm 0x%lx", summary.nUnavailable,
                                         startRank->commNRanks, startRank->commId.commHash),
                    ret, exit);
    }
    if (nTopologyNotReady > 0) {
      NCCLCHECKGOTO(rasDiagnosticsPciReportTopologyNotReady(reporter, "IOMMU mode", startRank, nTopologyNotReady), ret,
                    exit);
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}

// *************************************************************************
// ATS state check.
// *************************************************************************

static rasDiagnosticsAtsState rasDiagnosticsAtsStateFromLspci(const char* output, const char* bdf) {
  bool inDevice = false;
  size_t bdfLen = strlen(bdf);

  if (output == nullptr || output[0] == '\0' || !rasDiagnosticsPciValidBdf(bdf)) return RAS_DIAG_ATS_UNAVAILABLE;
  for (const char* line = output; *line != '\0';) {
    const char* end = strchr(line, '\n');
    size_t len = end == nullptr ? strlen(line) : (size_t)(end - line);

    if (len > 0 && !isspace((unsigned char)line[0])) {
      // lspci starts each device block with its domain:bus:device.function identifier.
      if (inDevice) break;
      inDevice = len > bdfLen && strncasecmp(line, bdf, bdfLen) == 0 && isspace((unsigned char)line[bdfLen]);
    }
    if (inDevice && rasDiagnosticsPciSpanContainsCaseInsensitive(line, len, "access denied")) {
      return RAS_DIAG_ATS_NEEDS_ROOT;
    }
    if (inDevice && rasDiagnosticsPciSpanContainsCaseInsensitive(line, len, "ATSCtl:") &&
        rasDiagnosticsPciSpanContainsCaseInsensitive(line, len, "Enable+")) {
      return RAS_DIAG_ATS_ON;
    }
    if (inDevice && rasDiagnosticsPciSpanContainsCaseInsensitive(line, len, "ATSCtl:") &&
        rasDiagnosticsPciSpanContainsCaseInsensitive(line, len, "Enable-")) {
      return RAS_DIAG_ATS_OFF;
    }
    if (end == nullptr) break;
    line = end + 1;
  }
  return RAS_DIAG_ATS_UNAVAILABLE;
}

static ncclResult_t rasDiagnosticsAtsFillRecord(const struct rasDiagnosticsPciCommSnapshot* snapshot,
                                                const struct rasDiagnosticsRdmaTopoCache* rdmaTopo,
                                                const char* lspciOutput, bool lspciUnavailable,
                                                bool lspciOutputTruncated, struct rasDiagnosticsAtsData* atsData) {
  memset(atsData, 0, sizeof(*atsData));
  atsData->fallbackState = rasDiagnosticsPciFallbackStateForSnapshot(snapshot, rdmaTopo);
  if (atsData->fallbackState != RAS_DIAG_PCI_FALLBACK_COLLECTED) return ncclSuccess;

  atsData->netSelectionIncomplete = snapshot->netSelectionIncomplete;
  atsData->netSelectionTruncated = snapshot->nPairsTruncated;
  atsData->lspciUnavailable = lspciUnavailable;
  atsData->lspciOutputTruncated = lspciOutputTruncated;
  for (int pairIdx = 0; pairIdx < snapshot->nPairs; pairIdx++) {
    const char* bdf = snapshot->pairs[pairIdx].nicBdf;
    bool duplicate = false;

    for (int dev = 0; dev < atsData->nNics; dev++) {
      if (strcmp(atsData->nics[dev].bdf, bdf) == 0) duplicate = true;
    }
    if (duplicate) continue;

    struct rasDiagnosticsAtsNicData* nic = atsData->nics + atsData->nNics++;

    memcpy(nic->bdf, bdf, sizeof(nic->bdf));
    nic->state = lspciUnavailable ? RAS_DIAG_ATS_UNAVAILABLE : rasDiagnosticsAtsStateFromLspci(lspciOutput, nic->bdf);
  }
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsAtsCollectLocal(const struct rasDiagnosticsContext* ctx,
                                           struct rasDiagnosticsLocalData* data) {
  ncclUniquePtr<struct rasDiagnosticsPciCommSnapshot> snapshots;
  ncclUniquePtr<char> records;
  ncclUniquePtr<char> lspciOutput;
  const struct rasDiagnosticsRdmaTopoCache* rdmaTopo = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsAtsData));
  int nRecords = 0;
  bool lspciOutputTruncated = false;
  bool lspciUnavailable = false;
  size_t nBytes;

  if (data == nullptr) {
    WARN("RAS diagnostics ATS state check local data output is null");
    return ncclInternalError;
  }
  memset(data, 0, sizeof(*data));
  if (ctx == nullptr || recordStride > (size_t)INT_MAX) {
    WARN("RAS diagnostics ATS state check local collection received invalid arguments");
    return ncclInternalError;
  }

  rdmaTopo = rasDiagnosticsRdmaTopoResult();
  NCCLCHECK(rasDiagnosticsPciCollectSnapshots(ctx, rdmaTopo->state != RAS_DIAG_RDMA_TOPO_RAN, snapshots, &nRecords));
  if (nRecords == 0) return ncclSuccess;
  if ((size_t)nRecords > (size_t)INT_MAX / recordStride) {
    WARN("RAS diagnostics ATS state check local data too large");
    return ncclInternalError;
  }
  if (rasDiagnosticsPciFallbackNeeded(snapshots.get(), nRecords, rdmaTopo)) {
    NCCLCHECK(ncclCalloc(lspciOutput, RAS_DIAG_LSPCI_OUTPUT_BYTES));
    int exitCode = ncclDiagChildRun("lspci -D -vvv", RAS_DIAG_LSPCI_CHILD_TIMEOUT_SEC, lspciOutput.get(),
                                    RAS_DIAG_LSPCI_OUTPUT_BYTES, &lspciOutputTruncated);
    lspciUnavailable = exitCode != 0 || lspciOutput.get()[0] == '\0';
  }

  nBytes = (size_t)nRecords * recordStride;
  NCCLCHECK(ncclCalloc(records, nBytes));
  for (int i = 0; i < nRecords; i++) {
    char* record = records.get() + (size_t)i * recordStride;
    memcpy(record, &snapshots.get()[i].rank, sizeof(snapshots.get()[i].rank));
    NCCLCHECK(rasDiagnosticsAtsFillRecord(
      snapshots.get() + i, rdmaTopo, lspciOutput.get(), lspciUnavailable, lspciOutputTruncated,
      (struct rasDiagnosticsAtsData*)(record + sizeof(struct rasDiagnosticsRankHeader))));
  }

  data->records = records.release();
  data->recordsBytes = (int)nBytes;
  data->recordStride = (int)recordStride;
  data->nRecords = nRecords;
  return ncclSuccess;
}

static const struct rasDiagnosticsAtsData* rasDiagnosticsAtsDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsAtsData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

static bool rasDiagnosticsAtsPayloadValid(const struct rasDiagnosticsAtsData* atsData) {
  if (atsData == nullptr || atsData->fallbackState > RAS_DIAG_PCI_FALLBACK_TOPOLOGY_NOT_READY ||
      atsData->netSelectionIncomplete > 1 || atsData->nNics > RAS_DIAG_PCI_MAX_NET_DEVS ||
      atsData->netSelectionTruncated > 1 || atsData->lspciUnavailable > 1 || atsData->lspciOutputTruncated > 1) {
    return false;
  }
  if (atsData->fallbackState != RAS_DIAG_PCI_FALLBACK_COLLECTED) {
    return !atsData->netSelectionIncomplete && atsData->nNics == 0 && !atsData->netSelectionTruncated &&
           !atsData->lspciUnavailable && !atsData->lspciOutputTruncated;
  }
  if (atsData->nNics == 0) return false;

  for (int i = 0; i < atsData->nNics; i++) {
    const struct rasDiagnosticsAtsNicData* nic = atsData->nics + i;

    if (!rasDiagnosticsPciBdfPayloadValid(nic->bdf) || nic->state > RAS_DIAG_ATS_ON) return false;
    if (atsData->lspciUnavailable && nic->state != RAS_DIAG_ATS_UNAVAILABLE) return false;
    for (int j = 0; j < i; j++) {
      if (strcmp(nic->bdf, atsData->nics[j].bdf) == 0) return false;
    }
  }
  return true;
}

ncclResult_t rasDiagnosticsAtsSummarize(const struct rasDiagnosticsContext* ctx,
                                        const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsAtsData));
  int nRecords;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics ATS state check received invalid reporter");
    return ncclInternalError;
  }
  if (ctx == nullptr || (ctx->hasCommFilter && ctx->commNRanks <= 0)) {
    WARN("RAS diagnostics ATS state check received invalid context");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr) {
    WARN("RAS diagnostics ATS state check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics ATS state check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + (size_t)start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    struct rasDiagnosticsPciNicIssue off = {};
    struct rasDiagnosticsPciNicIssue needsRoot = {};
    struct rasDiagnosticsPciNicIssue unavailable = {};
    struct rasDiagnosticsPciRankSet truncated = {};
    struct rasDiagnosticsPciRankSet incomplete = {};
    struct rasDiagnosticsPciRankSet lspciUnavailableRanks = {};
    struct rasDiagnosticsPciRankSet lspciTruncatedRanks = {};
    int nFallback = 0;
    int nUnavailable = 0;
    int nTopologyNotReady = 0;
    int nAtsOn = 0;
    int previousCommRank = -1;
    int end = start;

    if (ctx->hasCommFilter && (rasDiagnosticsCommIdCompare(&startRank->commId, &ctx->commFilter) != 0 ||
                               startRank->commNRanks != ctx->commNRanks)) {
      WARN("RAS diagnostics ATS state check received records outside the requested communicator");
      ret = ncclInternalError;
      goto exit;
    }

    while (end < nRecords) {
      const char* record = records + (size_t)end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsAtsData* atsData;

      if (end > start && rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      if (rank->commNRanks <= 0 || rank->commRank < 0 || rank->commRank >= rank->commNRanks ||
          rank->commNRanks != startRank->commNRanks || rank->commRank == previousCommRank) {
        WARN("RAS diagnostics ATS state check received malformed rank header");
        ret = ncclInternalError;
        goto exit;
      }
      previousCommRank = rank->commRank;
      atsData = rasDiagnosticsAtsDataFromRecord(record);
      if (!rasDiagnosticsAtsPayloadValid(atsData)) {
        WARN("RAS diagnostics ATS state check received malformed payload");
        ret = ncclInternalError;
        goto exit;
      }

      if (atsData->fallbackState == RAS_DIAG_PCI_FALLBACK_COLLECTED) {
        nFallback++;
        for (int nicIdx = 0; nicIdx < atsData->nNics; nicIdx++) {
          const struct rasDiagnosticsAtsNicData* nic = atsData->nics + nicIdx;

          if (nic->state == RAS_DIAG_ATS_ON) nAtsOn++;
          else if (nic->state == RAS_DIAG_ATS_OFF) {
            rasDiagnosticsPciRecordNicIssue(&off, rank->commRank, nic->bdf);
          } else if (nic->state == RAS_DIAG_ATS_NEEDS_ROOT) {
            rasDiagnosticsPciRecordNicIssue(&needsRoot, rank->commRank, nic->bdf);
          } else {
            rasDiagnosticsPciRecordNicIssue(&unavailable, rank->commRank, nic->bdf);
          }
        }
        if (atsData->netSelectionTruncated) rasDiagnosticsPciRankSetAdd(&truncated, rank->commRank);
        if (atsData->netSelectionIncomplete) rasDiagnosticsPciRankSetAdd(&incomplete, rank->commRank);
        if (atsData->lspciUnavailable) rasDiagnosticsPciRankSetAdd(&lspciUnavailableRanks, rank->commRank);
        if (atsData->lspciOutputTruncated) rasDiagnosticsPciRankSetAdd(&lspciTruncatedRanks, rank->commRank);
      } else if (atsData->fallbackState == RAS_DIAG_PCI_FALLBACK_UNAVAILABLE) {
        nUnavailable++;
      } else if (atsData->fallbackState == RAS_DIAG_PCI_FALLBACK_TOPOLOGY_NOT_READY) {
        nTopologyNotReady++;
      }
      end++;
    }

    if (end - start != startRank->commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "ATS state", startRank, end - start), ret, exit);
    } else if (nFallback == 0 && nUnavailable > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ATS state: network device selection incomplete on %d/%d ranks in comm 0x%lx",
                                         nUnavailable, startRank->commNRanks, startRank->commId.commHash),
                    ret, exit);
    } else if (nFallback == startRank->commNRanks && off.count == 0 && needsRoot.count == 0 && unavailable.count == 0 &&
               lspciUnavailableRanks.nTotal == 0 && lspciTruncatedRanks.nTotal == 0 && truncated.nTotal == 0 &&
               incomplete.nTotal == 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                                         "ATS state: enabled on %d rank/NIC pair(s) across %d ranks in comm 0x%lx",
                                         nAtsOn, startRank->commNRanks, startRank->commId.commHash),
                    ret, exit);
    } else if (off.count > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ATS state: disabled on %d rank/NIC pair(s) across %d/%d ranks in comm 0x%lx",
                                         off.count, off.nAffectedRanks, startRank->commNRanks,
                                         startRank->commId.commHash),
                    ret, exit);
    } else if (needsRoot.count > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(
                      reporter, RAS_DIAG_TAG_INFO,
                      "ATS state: PCIe capabilities require root access on %d rank/NIC pair(s) across "
                      "%d/%d ranks in comm 0x%lx",
                      needsRoot.count, needsRoot.nAffectedRanks, startRank->commNRanks, startRank->commId.commHash),
                    ret, exit);
    } else if (unavailable.count > 0 || lspciUnavailableRanks.nTotal > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ATS state: unavailable across %d/%d ranks in comm 0x%lx",
                                         unavailable.nAffectedRanks, startRank->commNRanks, startRank->commId.commHash),
                    ret, exit);
    } else if (nFallback > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ATS state: enabled on %d rank/NIC pair(s) across %d/%d ranks in comm 0x%lx",
                                         nAtsOn, nFallback, startRank->commNRanks, startRank->commId.commHash),
                    ret, exit);
    } else {
      // A conclusive rdma_topo result or the absence of relevant NICs makes the fallback unnecessary.
      if (nTopologyNotReady == 0) {
        start = end;
        continue;
      }
    }

    if (nTopologyNotReady > 0) {
      NCCLCHECKGOTO(rasDiagnosticsPciReportTopologyNotReady(reporter, "ATS state", startRank, nTopologyNotReady), ret,
                    exit);
    }

    if (off.count > 0) {
      NCCLCHECKGOTO(
        rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                             "ATS state: ATSCtl disabled on %d rank/NIC pair(s) in comm 0x%lx; first rank %d NIC %s",
                             off.count, startRank->commId.commHash, off.firstRank, off.firstBdf),
        ret, exit);
    }
    if (needsRoot.count > 0) {
      NCCLCHECKGOTO(
        rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                             "ATS state: PCIe capabilities inaccessible on %d rank/NIC pair(s) in comm 0x%lx; "
                             "first rank %d NIC %s",
                             needsRoot.count, startRank->commId.commHash, needsRoot.firstRank, needsRoot.firstBdf),
        ret, exit);
    }
    if (unavailable.count > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ATS state: ATSCtl unavailable on %d rank/NIC pair(s) in comm 0x%lx; "
                                         "first rank %d NIC %s",
                                         unavailable.count, startRank->commId.commHash, unavailable.firstRank,
                                         unavailable.firstBdf),
                    ret, exit);
    }
    if (lspciUnavailableRanks.nTotal > 0) {
      char rankSet[128];
      rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), lspciUnavailableRanks.ranks, lspciUnavailableRanks.nStored,
                                  lspciUnavailableRanks.nTotal);
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ATS state: lspci unavailable on rank(s) %s in comm 0x%lx", rankSet,
                                         startRank->commId.commHash),
                    ret, exit);
    }
    if (lspciTruncatedRanks.nTotal > 0) {
      char rankSet[128];
      rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), lspciTruncatedRanks.ranks, lspciTruncatedRanks.nStored,
                                  lspciTruncatedRanks.nTotal);
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ATS state: lspci output truncated on rank(s) %s in comm 0x%lx", rankSet,
                                         startRank->commId.commHash),
                    ret, exit);
    }
    if (nUnavailable > 0) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ATS state: network device selection incomplete on %d rank(s) in comm 0x%lx",
                                         nUnavailable, startRank->commId.commHash),
                    ret, exit);
    }
    if (truncated.nTotal > 0) {
      char rankSet[128];
      rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), truncated.ranks, truncated.nStored, truncated.nTotal);
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ATS state: network device selection truncated on rank(s) %s in comm 0x%lx",
                                         rankSet, startRank->commId.commHash),
                    ret, exit);
    }
    if (incomplete.nTotal > 0) {
      char rankSet[128];
      rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), incomplete.ranks, incomplete.nStored, incomplete.nTotal);
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "ATS state: network device selection incomplete on rank(s) %s "
                                         "in comm 0x%lx",
                                         rankSet, startRank->commId.commHash),
                    ret, exit);
    }

    start = end;
  }

exit:
  free(records);
  return ret;
}
