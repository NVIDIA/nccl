/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NET_OBSERV_H_
#define NCCL_NET_OBSERV_H_

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <thread>
#include <regex>
#include <cstdint>
#include <array>

#include <grpcpp/server_builder.h>

// For gethostname and gethostbyname
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>

#include "ruijie-json.pb.h"
#include "ruijie-json.grpc.pb.h"

namespace net_observ {

extern const char* const COLOR_RED;
extern const char* const COLOR_YELLOW;
extern const char* const COLOR_GREEN;
extern const char* const COLOR_CYAN;
extern const char* const COLOR_MAGENTA;
extern const char* const COLOR_BOLD;
extern const char* const COLOR_DIM;
extern const char* const COLOR_RESET;

enum class AlertLevel {
  PREDICTIVE,
  CONFIRMED
};

extern const char* const RETRANS_COUNTER_NAMES[];
extern const int RETRANS_COUNTER_NAMES_COUNT;

extern const std::regex NCCL_HCA_PATTERN;
extern const std::regex NCCL_PEER_IP_PATTERN;

struct NicCounterSnapshot {
  double timestamp;
  std::map<std::string, int64_t> counters;
};

struct NicDeviceInfo {
  std::string ibDevName;
  int portNum;
  std::string sysfsPath;
};

struct PortMappingInfo {
  std::string node;
  std::vector<int> ranks;
  std::string rdmaNic;
};

struct Incident {
  int incidentId;
  std::string incidentTag;
  std::string portName;
  std::string switchName;
  std::string deviceModel;
  int64_t dropCount;
  std::string timestamp;
  std::vector<int> affectedRanks;
  std::vector<std::string> faultPaths;
  std::string rdmaNic;
  std::map<std::string, int64_t> nicDelta;
  bool hasRetryGrowth;
  double createTime;
  bool confirmed;
  double confirmTime;
  std::string ncclError;
};

struct LldpNeighborInfo {
  std::string switchPort;
  std::string nodeIp;
  std::string remotePortDesc;
  std::string remoteChassisId;
  std::string remoteSysName;
  std::string remoteSysDesc;
  int64_t timeMark;
};

struct LldpTopologyEntry {
  std::string switchPort;
  std::string nodeIp;
  std::string rdmaNic;
  std::string description;
};

class TopologyConfig {
 public:
  std::string switchIp;
  std::string switchName;
  int grpcPort;
  std::map<std::string, PortMappingInfo> portMapping;

  std::vector<std::string> getPortsForNode(const std::string& nodeIp) const;
  std::vector<int> getRanksForPort(const std::string& portName) const;
  std::string getNodeForPort(const std::string& portName) const;
  std::string getRdmaNicForPort(const std::string& portName) const;
  std::vector<std::string> getAllPorts() const;
  std::string getPortByNodeAndNic(const std::string& nodeIp, const std::string& rdmaNic) const;

  static TopologyConfig loadFromFile(const std::string& configPath);
};

class NicCounterReader {
 public:
  static constexpr const char* IB_SYSFS_BASE = "/sys/class/infiniband";

  NicCounterReader();

  std::vector<std::string> discoverNics();
  void captureBaseline(const std::string& nicKey = "");
  std::map<std::string, int64_t> readDelta(const std::string& nicKey);
  std::pair<bool, std::map<std::string, int64_t>> checkNicRetrans(const std::string& rdmaNic);

 private:
  std::map<std::string, NicDeviceInfo> deviceCache_;
  std::map<std::string, NicCounterSnapshot> baselines_;

  std::map<std::string, int64_t> readCounters(const std::string& nicKey);
};

class SwitchEventServicer : public ruijie_json::Json::Service {
 public:
  grpc::Status JsonSend(grpc::ServerContext* context,
                        const ruijie_json::JsonRequest* request,
                        ruijie_json::JsonReply* response) override;
  grpc::Status JsonStreamSend(grpc::ServerContext* context,
                              grpc::ServerReaderWriter<ruijie_json::JsonReply,
                              ruijie_json::JsonRequest>* stream) override;

  std::vector<ruijie_json::JsonRequest> getPendingEvents();

 private:
  std::mutex mtx_;
  std::vector<ruijie_json::JsonRequest> events_;
};

class NetworkObserver {
 public:
  NetworkObserver(TopologyConfig* topology, int rank, double pollInterval = 2.0);
  ~NetworkObserver();

  int start();
  void stop();
  void confirmFromException(const std::string& errorText);

  // Direct IB error handling from transport layer
  void handleIbError(const std::string& rdmaNic, const std::string& peerIp, int wcStatus);

  // Accessors for C interface functions
  NicCounterReader& getNicReader() { return nicReader_; }
  std::unordered_map<std::string, Incident>& getActiveIncidents() { return activeIncidents_; }
  std::mutex& getIncidentsMutex() { return incidentsMutex_; }

 private:
  TopologyConfig* topology_;
  int rank_;
  double pollInterval_;

  volatile bool stopRequested_;
  std::unique_ptr<std::thread> monitorThread_;
  std::unique_ptr<grpc::Server> grpcServer_;
  SwitchEventServicer eventServicer_;

  int incidentId_;
  NicCounterReader nicReader_;
  std::unordered_map<std::string, Incident> activeIncidents_;
  std::mutex incidentsMutex_;

  void monitorLoop();
  void handleEvent(const ruijie_json::JsonRequest& event);
  void handleSwitchDropEvent(const ruijie_json::JsonRequest& event);
  void handleLldpEvent(const ruijie_json::JsonRequest& event);
  std::string formatTimestamp(const std::string& timestampRaw);
  std::vector<std::string> buildFaultPath(const std::vector<std::string>& affectedPorts,
                                           const std::string& deviceModel,
                                           const std::string& peerPort = "");
  std::string formatCounterDelta(const std::map<std::string, int64_t>& delta);
  void emitSwitchDropAlert(const Incident& incident);
  void emitPredictiveAlert(const Incident& incident);
  void emitConfirmedAlert(const Incident& incident);

  std::vector<LldpNeighborInfo> parseLldpJson(const std::string& jsonString);
  std::string inferRdmaNicFromPortDesc(const std::string& portDesc, const std::string& nodeIp = "");
  void updateTopologyFromLldp(const std::vector<LldpNeighborInfo>& lldpInfos,
                              const std::string& deviceModel);
  void printLldpTopology(const std::vector<LldpTopologyEntry>& entries);

  // SSH remote query helpers
  std::string getLocalIpAddress();
  std::string executeCommand(const std::string& cmd);
  std::string queryRemoteRdmaNic(const std::string& remoteIp, const std::string& netdev);
};

std::string extractJsonString(const std::string& json, const std::string& key);
int64_t extractJsonInt(const std::string& json, const std::string& key);

// ---- Global singleton management (integrated into NCCL init) ----
// Reads NCCL_NET_OBSERV_ENABLE, NCCL_NET_OBSERV_CONFIG, NCCL_NET_OBSERV_POLL_INTERVAL
// env vars and manages the NetworkObserver lifecycle automatically.
// Returns 0 on success, non-zero on failure (ncclSuccess/ncclSystemError convention).
int ncclNetObservInit(void);
void ncclNetObservFinalize(void);

// ---- Rank topology update (called after ncclCommInitRank) ----
// Updates the topology mapping with rank distribution information from NCCL comm.
// This should be called after ncclCommInitRank to get accurate rank-to-node mapping.
// nodeRanks: array of rank lists for each node, indexed by node ID
void ncclNetObservUpdateRankTopology(const std::vector<std::vector<int>>& nodeRanks);

// ---- IB error handling (called from net_ib/p2p.cc) ----
// Directly handles IB completion queue errors from the IB transport layer.
// This provides faster error detection compared to parsing NCCL exceptions.
// rdmaNic: RDMA NIC device name (e.g., "mlx5_0")
// peerIp: Peer node IP address (optional, can be empty)
// wcStatus: IB work completion status code
void ncclNetObservHandleIbError(const char* rdmaNic, const char* peerIp, int wcStatus);

}  // namespace net_observ

#endif  // NCCL_NET_OBSERV_H_
