/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ib_write_bw.h"

#include "os.h"

#if defined(NCCL_OS_LINUX)

#include "alloc.h"
#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "diagnostics.h"
#include "diagnostics_log.h"
#include "graph.h"
#include "param.h"
#include "transport.h"
#include "utils.h"

#include <algorithm>
#include <vector>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern int64_t ncclParamIbQpsPerConn();

NCCL_PARAM(DiagIbBwTimeout, "DIAGNOSTICS_IB_BW_TIMEOUT", 5);

#define IB_BW_NAME_SIZE 64
#define IB_BW_HOSTNAME_SIZE 1024
#define IB_BW_TOOL_OUTPUT_BYTES 32768
#define IB_BW_COMMAND_BYTES 512
#define IB_BW_PAYLOAD_BYTES 65536
#define IB_BW_ITERS 1000
// Per-tool watchdog; healthy runs finish well under a second, so it only catches wedged tools.
#define IB_BW_TIMEOUT_SEC 5
#define IB_BW_PAIR_SYNC_TAG 0x1b000001
#define IB_BW_SERVER_READY_TAG 0x1b000002
#define IB_BW_PORT_BASE 10000
#define IB_BW_PORT_SPAN 20000
#define IB_BW_DMA_SUFFIX "_dma"
// Per-rank outlier lines printed per mode before the remainder is reported as one count.
#define IB_BW_OUTLIER_REPORT_MAX 8

namespace {

struct LocalInfo {
  int deviceCount; // distinct devices this rank's channels use; only `device` is measured
  int port; // physical port of `device`, from the plugin's properties
  bool cuda;
  bool dmabuf;
  bool setupFailed;
  char hostname[IB_BW_HOSTNAME_SIZE];
  char device[IB_BW_NAME_SIZE];
};

struct RankBandwidth {
  double direct;
  double cross;
};

// One device of a node and the ranks sharing it, in local-rank order.
struct DeviceGroup {
  const char* name;
  std::vector<int> ranks;
};

// Distinct first-device names in first-appearance order (the pairing geometry), each with its ranks.
static std::vector<DeviceGroup> nodeInventory(const ncclComm* comm, const LocalInfo* rankInfo, int node) {
  std::vector<DeviceGroup> inventory;
  for (int localRank = 0; localRank < comm->nodeRanks[node].localRanks; localRank++) {
    int rank = comm->nodeRanks[node].localRankToRank[localRank];
    const char* name = rankInfo[rank].device;
    size_t group = 0;
    while (group < inventory.size() && strcmp(inventory[group].name, name) != 0) group++;
    if (group == inventory.size()) inventory.push_back({name, {rank}});
    else inventory[group].ranks.push_back(rank);
  }
  return inventory;
}

static int findGroup(const std::vector<DeviceGroup>& inventory, const char* name) {
  for (size_t i = 0; i < inventory.size(); i++) {
    if (strcmp(inventory[i].name, name) == 0) return (int)i;
  }
  return -1;
}

// Position of `rank` among the ranks sharing its device: its pairing occurrence.
static int rankPosition(const DeviceGroup& group, int rank) {
  for (size_t i = 0; i < group.ranks.size(); i++) {
    if (group.ranks[i] == rank) return (int)i;
  }
  return -1;
}

// Hosts form a ring; parity-p nodes serve their +1 neighbor in phase p (odd rings skip the wrap).
// Server devices anchor on the same-named client device, rotated by one when cross; ranks sharing
// a device pair k-th with k-th. False when this rank has no check this phase.
static bool findPair(const ncclComm* comm, const LocalInfo* rankInfo, int phase, bool cross, int& serverRank,
                     int& clientRank) {
  int nNodes = comm->nNodes;
  int serverNode, clientNode;
  if (comm->node % 2 == phase) {
    serverNode = comm->node;
    clientNode = (comm->node + 1) % nNodes;
  } else {
    serverNode = (comm->node + nNodes - 1) % nNodes;
    clientNode = comm->node;
  }
  if (nNodes % 2 != 0 && serverNode == nNodes - 1) return false; // odd ring: wrapping pair skipped
  std::vector<DeviceGroup> client = nodeInventory(comm, rankInfo, clientNode);
  int rotation = cross ? 1 : 0;
  if (rotation >= (int)client.size()) return false; // cross-device needs a second client device
  const char* myDevice = rankInfo[comm->rank].device;
  if (comm->node == serverNode) {
    std::vector<DeviceGroup> server = nodeInventory(comm, rankInfo, serverNode);
    int occurrence = rankPosition(server[findGroup(server, myDevice)], comm->rank);
    int anchor = findGroup(client, myDevice);
    if (anchor < 0) return false; // device not in common
    const DeviceGroup& target = client[(anchor + rotation) % client.size()];
    if (occurrence >= (int)target.ranks.size()) return false; // fewer ranks share the client device
    serverRank = comm->rank;
    clientRank = target.ranks[occurrence];
  } else {
    int mine = findGroup(client, myDevice); // my own node's inventory: always present
    int occurrence = rankPosition(client[mine], comm->rank);
    int anchor = (mine + (int)client.size() - rotation) % (int)client.size();
    std::vector<DeviceGroup> server = nodeInventory(comm, rankInfo, serverNode);
    int source = findGroup(server, client[anchor].name);
    if (source < 0) return false; // anchor device missing on the server node
    if (occurrence >= (int)server[source].ranks.size()) return false; // fewer ranks share the server device
    serverRank = server[source].ranks[occurrence];
    clientRank = comm->rank;
  }
  return true;
}

// Keyed on the communicator and the server GPU's host-global index (nvmlDev): distinct per server
// host within a comm, and comms initializing concurrently on the same host land on distinct ports
// with high probability. The band sits above privileged ports, below the ephemeral range (32768+).
static int benchmarkPort(const ncclComm* comm, int gpuIndex) {
  return IB_BW_PORT_BASE + (unsigned int)(comm->commHash + gpuIndex) % IB_BW_PORT_SPAN;
}

// Failure description for a tool exit code; nullptr on success.
static const char* toolFailure(int exitCode) {
  if (exitCode == 0) return nullptr;
  if (exitCode == 124) return "tool run timed out";
  if (exitCode == 127) return "tool missing";
  return "tool run failed";
}

// Matches a known plugin name exactly or followed by a delimiter, rejecting prefix-sharing names.
static bool pluginSupportsIb(const char* name) {
  static const char* const kIbPluginNames[] = {"IB", "IBext", "SPCX", "NCCL RDMA Plugin"};
  if (name == nullptr) return false;
  for (const char* plugin : kIbPluginNames) {
    const size_t len = strlen(plugin);
    if (strncmp(name, plugin, len) == 0 && (name[len] == '\0' || name[len] == '_' || name[len] == ' ')) return true;
  }
  return false;
}

static bool ibDeviceExists(const char* name) {
  char path[PATH_MAX];
  int length = snprintf(path, sizeof(path), "/sys/class/infiniband/%s", name);
  return length > 0 && length < static_cast<int>(sizeof(path)) && access(path, F_OK) == 0;
}

// Resolves one segment of a (possibly merged) device name, cutting a "_dma" suffix that sysfs
// does not know. False when the name is malformed or not present in sysfs.
static bool resolveDeviceName(const char* start, size_t length, char out[IB_BW_NAME_SIZE]) {
  if (length == 0 || length >= IB_BW_NAME_SIZE) return false;
  memcpy(out, start, length);
  out[length] = '\0';
  if (ibDeviceExists(out)) return true;
  const size_t suffixLength = sizeof(IB_BW_DMA_SUFFIX) - 1;
  if (length <= suffixLength || strcmp(out + length - suffixLength, IB_BW_DMA_SUFFIX) != 0) return false;
  out[length - suffixLength] = '\0';
  return ibDeviceExists(out);
}

// Physical devices in a (possibly merged) device name: its '+'-separated segments.
static int segmentCount(const char* name) {
  if (name == nullptr) return 0;
  int count = 1;
  for (const char* c = name; *c != '\0'; c++) count += *c == '+';
  return count;
}

static int probeCapabilities(LocalInfo* local) {
  char output[IB_BW_TOOL_OUTPUT_BYTES];
  int exitCode = ncclDiagChildRun("ib_write_bw --help", IB_BW_TIMEOUT_SEC, output, IB_BW_TOOL_OUTPUT_BYTES);
  local->cuda = strstr(output, "--use_cuda") != nullptr;
  local->dmabuf = local->cuda && strstr(output, "--use_cuda_dmabuf") != nullptr;
  return exitCode;
}

static void discoverLocal(ncclComm* comm, LocalInfo* local) {
  memset(local, 0, sizeof(*local));

  if (gethostname(local->hostname, sizeof(local->hostname) - 1) != 0) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: cannot determine local hostname in comm 0x%lx", (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }

  int probeExit = probeCapabilities(local); // also sets local->cuda / local->dmabuf
  if (probeExit == 127) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: required external tool missing in comm 0x%lx", (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }
  if (probeExit == 124) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: capability probe timed out in comm 0x%lx", (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }
  local->cuda &= comm->cudaDev >= 0;
  local->dmabuf &= local->cuda && comm->dmaBufSupport;

  if (comm->ncclNet == nullptr || !pluginSupportsIb(comm->ncclNet->name)) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: failed: selected network plugin does not support IB in comm 0x%lx",
               (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }
  int count = 0;
  if (comm->ncclNet->devices(&count) != ncclSuccess) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: cannot enumerate network devices in comm 0x%lx",
               (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }
  if (count <= 0 || comm->topo == nullptr) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: failed: no usable IB device in comm 0x%lx", (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }
  if (comm->nChannels <= 0) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: invalid channel count=%d in comm 0x%lx", comm->nChannels,
               (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }

  // Only channel 0's device is validated strictly: its name is what reaches ib_write_bw.
  int device = -1;
  if (ncclTopoGetLocalNet(comm->topo, comm->rank, /*channelId=*/0, nullptr, &device) != ncclSuccess) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: cannot select a local network device in comm 0x%lx",
               (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }
  if (device < 0 || device >= count) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: failed: no usable IB device in comm 0x%lx", (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }
  ncclNetProperties_t properties = {};
  if (comm->ncclNet->getProperties(device, &properties) != ncclSuccess) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: cannot query the selected network device in comm 0x%lx",
               (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }
  const char* segmentEnd = properties.name == nullptr ? nullptr : strchr(properties.name, '+');
  size_t segmentLength = properties.name == nullptr ?
                           0 :
                           (segmentEnd ? static_cast<size_t>(segmentEnd - properties.name) : strlen(properties.name));
  if (!resolveDeviceName(properties.name, segmentLength, local->device)) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: failed: cannot resolve the selected IB device in comm 0x%lx",
               (unsigned long)comm->commHash);
    local->setupFailed = true;
    return;
  }

  // Merged members beyond the first are not measured; the count feeds the untested-devices line.
  local->deviceCount = segmentCount(properties.name);
  local->port = properties.port;
}

// Builds the ib_write_bw command line; the client form appends the server hostname (the connect
// target). Returns false on truncation.
static bool buildCommand(char* out, int outSize, bool server, const LocalInfo& local, const LocalInfo& serverInfo,
                         bool cudaEnabled, bool dmabufEnabled, int cudaDev, int port, int qps) {
  char cuda[80] = "";
  if (cudaEnabled) {
    snprintf(cuda, sizeof(cuda), " --use_cuda=%d%s", cudaDev, dmabufEnabled ? " --use_cuda_dmabuf" : "");
  }
  int n = snprintf(out, outSize, "ib_write_bw -d %s -i %d -s %d --report_gbits -q %d -p %d -n %d%s", local.device,
                   local.port, IB_BW_PAYLOAD_BYTES, qps, port, IB_BW_ITERS, cuda);
  if (!server && n > 0 && n < outSize) {
    n += snprintf(out + n, outSize - n, " %s", serverInfo.hostname);
  }
  return n > 0 && n < outSize;
}

// Finds the "#bytes #iterations peak average" result row and extracts the average bandwidth.
static bool parseBandwidth(const char* output, double& average) {
  for (const char* line = output; line != nullptr && *line != '\0';) {
    double peak;
    if (sscanf(line, "%*u %*u %lf %lf", &peak, &average) == 2 && isfinite(peak) && isfinite(average)) return true;
    line = strchr(line, '\n');
    if (line != nullptr) line++;
  }
  return false;
}

static int computeBandwidthStats(const RankBandwidth* bandwidth, int nRanks, bool cross, double* sorted,
                                 double& minimum, double& median, double& maximum) {
  int measuredRanks = 0;
  for (int rank = 0; rank < nRanks; rank++) {
    double value = cross ? bandwidth[rank].cross : bandwidth[rank].direct;
    if (value >= 0) sorted[measuredRanks++] = value;
  }
  if (measuredRanks == 0) return 0;
  std::sort(sorted, sorted + measuredRanks);
  minimum = sorted[0];
  maximum = sorted[measuredRanks - 1];
  int middle = measuredRanks / 2;
  median = measuredRanks % 2 != 0 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  return measuredRanks;
}

// One-byte allgather vote: true only when `localOk` was true on every rank. Completing the
// allgather is also the lockstep point, so a rank that cannot proceed never strands a pair peer.
// `transportOk` reports the transport; on failure the vote is false.
static bool allRanksOk(ncclComm* comm, bool* votes, bool localOk, bool& transportOk) {
  votes[comm->rank] = localOk;
  transportOk = bootstrapAllGather(comm->bootstrap, votes, sizeof(*votes)) == ncclSuccess;
  if (!transportOk) return false;
  bool ok = true;
  for (int rank = 0; rank < comm->nRanks; rank++) ok = ok && votes[rank];
  return ok;
}

// Echoes the client's measurement to the server, so both endpoints hold the client-measured value.
static bool syncPair(ncclComm* comm, bool isServer, int peer, double& bandwidth) {
  bool ok;
  if (!isServer) {
    ok = bootstrapSend(comm->bootstrap, peer, IB_BW_PAIR_SYNC_TAG, &bandwidth, sizeof(bandwidth)) == ncclSuccess;
    ok = ok && bootstrapRecv(comm->bootstrap, peer, IB_BW_PAIR_SYNC_TAG, &bandwidth, sizeof(bandwidth)) == ncclSuccess;
  } else {
    ok = bootstrapRecv(comm->bootstrap, peer, IB_BW_PAIR_SYNC_TAG, &bandwidth, sizeof(bandwidth)) == ncclSuccess;
    ok = ok && bootstrapSend(comm->bootstrap, peer, IB_BW_PAIR_SYNC_TAG, &bandwidth, sizeof(bandwidth)) == ncclSuccess;
  }
  return ok;
}

// A measurement is an outlier when it deviates by more than 30% from the median.
static bool bandwidthOutlier(double value, double median) {
  return value < 0.7 * median || value > 1.3 * median;
}

static void reportBandwidthStats(const ncclComm* comm, const LocalInfo* rankInfo, const RankBandwidth* bandwidth,
                                 double* sorted, bool cross, bool complete) {
  if (comm->rank != 0) return;
  const char* mode = cross ? "cross-nic" : "same-nic";
  double minimum = 0, median = 0, maximum = 0;
  int measuredRanks = computeBandwidthStats(bandwidth, comm->nRanks, cross, sorted, minimum, median, maximum);
  if (measuredRanks == 0) return;
  bool hasOutliers = bandwidthOutlier(minimum, median) || bandwidthOutlier(maximum, median);
  DIAG_PRINT("NCCL DIAG [%s] net bw: %.1f/%.1f/%.1f Gbit/s min/median/max "
             "%s bw (across %d ranks) in comm 0x%lx",
             hasOutliers || !complete ? "INFO" : "OK", minimum, median, maximum, mode, measuredRanks,
             (unsigned long)comm->commHash);
  int outliers = 0;
  for (int rank = 0; rank < comm->nRanks; rank++) {
    double value = cross ? bandwidth[rank].cross : bandwidth[rank].direct;
    if (value < 0 || !bandwidthOutlier(value, median)) continue;
    if (++outliers > IB_BW_OUTLIER_REPORT_MAX) continue;
    DIAG_PRINT("NCCL DIAG [INFO] net bw: %s rank %d (%s): %.1f Gbit/s, >30%% off median %.1f Gbit/s in comm 0x%lx",
               mode, rank, rankInfo[rank].hostname, value, median, (unsigned long)comm->commHash);
  }
  if (outliers > IB_BW_OUTLIER_REPORT_MAX)
    DIAG_PRINT("NCCL DIAG [INFO] net bw: %s: %d more ranks >30%% off median in comm 0x%lx", mode,
               outliers - IB_BW_OUTLIER_REPORT_MAX, (unsigned long)comm->commHash);
}

// The server sends its client one status byte: 1 when the tool announces its listen socket, 0 if
// it ends without doing so. A failed send surfaces as the client's recv failure, so it is not
// tracked here.
struct ServerReadyCtx {
  ncclComm* comm;
  int clientRank;
  bool sent;
};

static void serverListenLine(const char* line, void* opaque) {
  ServerReadyCtx* ctx = static_cast<ServerReadyCtx*>(opaque);
  if (ctx->sent || strstr(line, "Waiting for client") == nullptr) return;
  bool ready = true;
  ctx->sent = true;
  (void)bootstrapSend(ctx->comm->bootstrap, ctx->clientRank, IB_BW_SERVER_READY_TAG, &ready, sizeof(ready));
}

// Returns the client-measured bandwidth in Gbit/s (identical on both endpoints, see syncPair),
// or -1 when the pair produced no measurement.
static double runPair(ncclComm* comm, const LocalInfo* rankInfo, int serverRank, int clientRank, bool cross) {
  bool isServer = comm->rank == serverRank;
  int peer = isServer ? clientRank : serverRank;
  const LocalInfo& serverInfo = rankInfo[serverRank];
  const LocalInfo& clientInfo = rankInfo[clientRank];

  bool cudaEnabled = serverInfo.cuda && clientInfo.cuda;
  bool dmabufEnabled = cudaEnabled && serverInfo.dmabuf && clientInfo.dmabuf;

  const LocalInfo& local = rankInfo[comm->rank];
  int port = benchmarkPort(comm, comm->peerInfo[serverRank].nvmlDev);
  if (ncclParamIbQpsPerConn() < 1) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: invalid qps parameter=%ld in comm 0x%lx", ncclParamIbQpsPerConn(),
               (unsigned long)comm->commHash);
    return -1;
  }
  char command[IB_BW_COMMAND_BYTES] = "";
  const char* detail = nullptr;
  double sample = -1;
  if (!buildCommand(command, sizeof(command), isServer, local, serverInfo, cudaEnabled, dmabufEnabled, comm->cudaDev,
                    port, (int)ncclParamIbQpsPerConn())) {
    detail = "cannot build command";
  }

  if (isServer) {
    ServerReadyCtx ready = {comm, clientRank, false};
    if (detail == nullptr)
      detail = toolFailure(ncclDiagChildRunStream(command, IB_BW_TIMEOUT_SEC, nullptr, 0, serverListenLine, &ready));
    if (!ready.sent) { // command build failed, or the tool ended without announcing its socket
      bool serverUp = false;
      (void)bootstrapSend(comm->bootstrap, clientRank, IB_BW_SERVER_READY_TAG, &serverUp, sizeof(serverUp));
    }
    if (detail != nullptr) {
      DIAG_PRINT("NCCL DIAG [INFO] net bw: mode=%s server_rank=%d server_host=%s server_device=%s client_rank=%d "
                 "client_host=%s client_device=%s %s in comm 0x%lx",
                 cross ? "cross" : "same", serverRank, serverInfo.hostname, serverInfo.device, clientRank,
                 clientInfo.hostname, clientInfo.device, detail, (unsigned long)comm->commHash);
    }
  } else {
    char output[IB_BW_TOOL_OUTPUT_BYTES];
    // Receive the status byte even after a local failure, or it corrupts a later phase.
    bool serverReady = false;
    if (bootstrapRecv(comm->bootstrap, serverRank, IB_BW_SERVER_READY_TAG, &serverReady, sizeof(serverReady)) !=
        ncclSuccess) {
      if (detail == nullptr) detail = "cannot receive the server status";
    } else if (!serverReady && detail == nullptr) {
      detail = "server did not start";
    }
    if (detail == nullptr)
      detail = toolFailure(ncclDiagChildRun(command, IB_BW_TIMEOUT_SEC, output, IB_BW_TOOL_OUTPUT_BYTES));
    double average = 0;
    if (detail == nullptr && !parseBandwidth(output, average)) {
      detail = "no bandwidth data";
    }
    if (detail == nullptr) {
      // Successful measurements only feed the per-rank aggregates reported on rank 0.
      sample = average;
    } else {
      DIAG_PRINT("NCCL DIAG [INFO] net bw: mode=%s server_rank=%d server_host=%s server_device=%s client_rank=%d "
                 "client_host=%s client_device=%s "
                 "memory=%s %s in comm 0x%lx",
                 cross ? "cross" : "same", serverRank, serverInfo.hostname, serverInfo.device, clientRank,
                 clientInfo.hostname, clientInfo.device,
                 cudaEnabled ? (dmabufEnabled ? "cuda+dmabuf" : "cuda") : "host", detail,
                 (unsigned long)comm->commHash);
    }
  }
  if (!syncPair(comm, isServer, peer, sample)) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: cannot exchange the measurement with rank %d in comm "
               "0x%lx",
               peer, (unsigned long)comm->commHash);
    return -1;
  }
  return sample;
}

// Each phase opens with a vote that keeps phases in lockstep and stops all ranks together when the
// allocated time runs out. Pair failures report locally; the return value is transport health only.
static bool runSchedule(ncclComm* comm, const LocalInfo* rankInfo, bool* votes, bool useCrossNic, RankBandwidth& result,
                        bool& allPairsRan) {
  double directSum = 0, crossSum = 0;
  int directCount = 0, crossCount = 0;
  bool complete = true;
  int64_t timeout = ncclParamDiagIbBwTimeout();
  if (timeout < 1) timeout = IB_BW_TIMEOUT_SEC;
  uint64_t deadlineNs = clockNano() + (uint64_t)timeout * 1000000000ULL;
  result = {-1, -1};
  for (int phase = 0; phase < 2; phase++) {
    bool transportOk = true;
    bool beforeDeadline = allRanksOk(comm, votes, clockNano() < deadlineNs, transportOk);
    if (!transportOk) return false;
    if (!beforeDeadline) {
      if (comm->rank == 0) {
        DIAG_PRINT("NCCL DIAG [INFO] net bw: test unable to complete in allocated time; ran %d of %d test "
                   "phases in comm 0x%lx",
                   phase, 2, (unsigned long)comm->commHash);
      }
      complete = false;
      break;
    }
    bool cross = useCrossNic && phase == 0; // the cross rotation runs in phase 0, device-aligned in 1
    int serverRank, clientRank;
    if (!findPair(comm, rankInfo, phase, cross, serverRank, clientRank)) continue;
    double sample = runPair(comm, rankInfo, serverRank, clientRank, cross);
    if (sample >= 0) {
      (cross ? crossSum : directSum) += sample;
      (cross ? crossCount : directCount)++;
    } else {
      complete = false;
    }
  }
  if (directCount > 0) result.direct = directSum / directCount;
  if (crossCount > 0) result.cross = crossSum / crossCount;
  // Closing vote: every rank learns whether all scheduled pairs produced a measurement.
  bool transportOk;
  allPairsRan = allRanksOk(comm, votes, complete, transportOk);
  return transportOk;
}

static bool commUsesCrossNic(const ncclComm* comm) {
  for (int algo = 0; algo < NCCL_NUM_ALGORITHMS; algo++) {
    if (comm->graphs[algo].nChannels > 0 && comm->graphs[algo].crossNic != 0) return true;
  }
  return false;
}

} // namespace

void ncclDiagRunIbWriteBw(ncclComm* comm) {
  if (comm == nullptr || comm->peerInfo == nullptr || comm->nodeRanks == nullptr || comm->nRanks <= 0) return;
  LocalInfo* rankInfo = nullptr;
  bool* votes = nullptr;
  RankBandwidth* allBandwidth = nullptr;
  double* sortedBandwidth = nullptr;
  bool useCrossNic = false;
  bool allPairsRan = true;

  // Pre-collective allocations: a rank that cannot allocate its gather buffers cannot join any
  // exchange either way (same exposure as the p2p check's setupResults buffer).
  if (ncclCalloc(&rankInfo, comm->nRanks) != ncclSuccess || ncclCalloc(&votes, comm->nRanks) != ncclSuccess ||
      ncclCalloc(&allBandwidth, comm->nRanks) != ncclSuccess ||
      ncclCalloc(&sortedBandwidth, comm->nRanks) != ncclSuccess) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: cannot allocate diagnostic state in comm 0x%lx",
               (unsigned long)comm->commHash);
    goto cleanup;
  }
  discoverLocal(comm, &rankInfo[comm->rank]);
  if (bootstrapAllGather(comm->bootstrap, rankInfo, sizeof(LocalInfo)) != ncclSuccess) {
    DIAG_PRINT("NCCL DIAG [INFO] net bw: cannot share local configuration in comm 0x%lx",
               (unsigned long)comm->commHash);
    goto cleanup;
  }
  // Gathered data is identical everywhere, so all ranks take this exit together.
  for (int rank = 0; rank < comm->nRanks; rank++) {
    if (!rankInfo[rank].setupFailed) continue;
    if (comm->rank == 0)
      DIAG_PRINT("NCCL DIAG [INFO] net bw: setup failed on rank %d in comm 0x%lx", rank, (unsigned long)comm->commHash);
    goto cleanup;
  }

  if (comm->rank == 0) {
    // A rank with multiple devices only measures its first one; make the coverage gap visible.
    for (int rank = 0; rank < comm->nRanks; rank++) {
      if (rankInfo[rank].deviceCount <= 1) continue;
      DIAG_PRINT("NCCL DIAG [INFO] net bw: multiple net devices per rank detected, only the first device is "
                 "tested in comm 0x%lx",
                 (unsigned long)comm->commHash);
      break;
    }
    // Ranks sharing a device measure it concurrently, splitting its bandwidth.
    bool shared = false;
    for (int node = 0; node < comm->nNodes && !shared; node++) {
      for (const DeviceGroup& group : nodeInventory(comm, rankInfo, node)) {
        if (group.ranks.size() > 1) {
          shared = true;
          break;
        }
      }
    }
    if (shared)
      DIAG_PRINT("NCCL DIAG [INFO] net bw: net devices are shared across ranks, concurrent use can lower the "
                 "bandwidth in comm 0x%lx",
                 (unsigned long)comm->commHash);
  }

  // Pairings are a pure function of the gathered rankInfo, so all ranks stay in lockstep.
  useCrossNic = commUsesCrossNic(comm);
  if (!runSchedule(comm, rankInfo, votes, useCrossNic, allBandwidth[comm->rank], allPairsRan)) goto bootstrapError;

  // The summary allgather is also the closing synchronization point.
  if (bootstrapAllGather(comm->bootstrap, allBandwidth, sizeof(RankBandwidth)) != ncclSuccess) goto bootstrapError;
  reportBandwidthStats(comm, rankInfo, allBandwidth, sortedBandwidth, /*cross=*/false, allPairsRan);
  if (useCrossNic) reportBandwidthStats(comm, rankInfo, allBandwidth, sortedBandwidth, /*cross=*/true, allPairsRan);

cleanup:
  free(sortedBandwidth);
  free(allBandwidth);
  free(votes);
  free(rankInfo);
  // Observational check: findings were already reported via DIAG_PRINT; nothing is returned.
  return;
bootstrapError:
  DIAG_PRINT("NCCL DIAG [INFO] net bw: bootstrap failure, aborting check in comm 0x%lx", (unsigned long)comm->commHash);
  goto cleanup;
}

#endif // NCCL_OS_LINUX
