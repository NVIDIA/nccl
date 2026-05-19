/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "net_observ/net_observ.h"

#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <sys/stat.h>
#include <dirent.h>
#include <array>
#include <memory>

#include "checks.h"
#include "param.h"

namespace net_observ {

const char* const COLOR_RED = "\033[91m";
const char* const COLOR_YELLOW = "\033[93m";
const char* const COLOR_GREEN = "\033[92m";
const char* const COLOR_CYAN = "\033[96m";
const char* const COLOR_MAGENTA = "\033[95m";
const char* const COLOR_BOLD = "\033[1m";
const char* const COLOR_DIM = "\033[2m";
const char* const COLOR_RESET = "\033[0m";

const char* const RETRANS_COUNTER_NAMES[] = {
  "roce_adp_retrans",
  "roce_adp_retrans_to",
  "roce_slow_restart",
  "roce_slow_restart_cnps",
  "roce_slow_restart_cnp_acks",
};
const int RETRANS_COUNTER_NAMES_COUNT = sizeof(RETRANS_COUNTER_NAMES) / sizeof(RETRANS_COUNTER_NAMES[0]);

const std::regex NCCL_HCA_PATTERN("hca\\s+(mlx5_\\d+)");
const std::regex NCCL_PEER_IP_PATTERN("remoteGids?::ffff:([\\d.]+)");

// ---- JSON field extraction helpers ----

/**
 * @brief 从JSON字符串中提取指定键的字符串值
 * @param json 输入的JSON字符串
 * @param key 要提取的字段名
 * @return 字段对应的字符串值，未找到返回空字符串
 * 
 * 支持格式："key":"value" 或 "key" : "value"
 */
std::string extractJsonString(const std::string& json, const std::string& key) {
  std::string search = "\"" + key + "\":\"";
  auto pos = json.find(search);
  if (pos == std::string::npos) {
    search = "\"" + key + "\" : \"";
    pos = json.find(search);
    if (pos == std::string::npos) return "";
  }
  pos += search.length();
  auto end = json.find("\"", pos);
  if (end == std::string::npos) return "";
  return json.substr(pos, end - pos);
}

/**
 * @brief 从JSON字符串中提取指定键的整数值
 * @param json 输入的JSON字符串
 * @param key 要提取的字段名
 * @return 字段对应的整数值，未找到返回0
 * 
 * 支持格式："key":123 或 "key" : 123
 */
int64_t extractJsonInt(const std::string& json, const std::string& key) {
  std::string search = "\"" + key + "\":";
  auto pos = json.find(search);
  if (pos == std::string::npos) {
    search = "\"" + key + "\" : ";
    pos = json.find(search);
    if (pos == std::string::npos) return 0;
  }
  pos += search.length();
  while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
  std::string num;
  while (pos < json.length() && json[pos] >= '0' && json[pos] <= '9') {
    num += json[pos];
    pos++;
  }
  if (num.empty()) return 0;
  return std::stoll(num);
}

/**
 * @brief 去除字符串首尾的空白字符
 * @param s 输入字符串
 * @return 去除空白后的字符串
 * 
 * 去除的字符包括：空格、制表符、换行符、回车符
 */
static std::string trim(const std::string& s) {
  size_t start = 0;
  while (start < s.length() && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')) start++;
  size_t end = s.length();
  while (end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r')) end--;
  return s.substr(start, end - start);
}

/**
 * @brief 将JSON转义字符还原为原始字符
 * @param s 包含转义序列的JSON字符串
 * @return 还原后的字符串
 *
 * 处理的转义序列：\" → "，\\ → \，\/ → /
 */
static std::string unescapeJson(const std::string& s) {
  std::string result;
  for (size_t i = 0; i < s.length(); i++) {
    if (s[i] == '\\' && i + 1 < s.length()) {
      switch (s[i+1]) {
        case '"': result += '"'; i++; break;
        case '\\': result += '\\'; i++; break;
        case '/': result += '/'; i++; break;
        default: result += s[i]; break;
      }
    } else {
      result += s[i];
    }
  }
  return result;
}

// ---- TopologyConfig ----

/**
 * @brief 获取指定节点IP关联的所有交换机端口
 * @param nodeIp 节点IP地址
 * @return 端口名称列表
 */
std::vector<std::string> TopologyConfig::getPortsForNode(const std::string& nodeIp) const {
  std::vector<std::string> result;
  for (const auto& entry : portMapping) {
    if (entry.second.node == nodeIp) result.push_back(entry.first);
  }
  return result;
}

/**
 * @brief 获取指定端口关联的NCCL rank列表
 * @param portName 交换机端口名称
 * @return rank列表，端口不存在返回空列表
 */
std::vector<int> TopologyConfig::getRanksForPort(const std::string& portName) const {
  auto it = portMapping.find(portName);
  if (it != portMapping.end()) return it->second.ranks;
  return {};
}

/**
 * @brief 获取指定端口关联的节点IP
 * @param portName 交换机端口名称
 * @return 节点IP地址，端口不存在返回"unknown"
 */
std::string TopologyConfig::getNodeForPort(const std::string& portName) const {
  auto it = portMapping.find(portName);
  if (it != portMapping.end()) return it->second.node;
  return "unknown";
}

/**
 * @brief 获取指定端口关联的RDMA网卡名称
 * @param portName 交换机端口名称
 * @return RDMA网卡名称（如mlx5_0），端口不存在返回"unknown"
 */
std::string TopologyConfig::getRdmaNicForPort(const std::string& portName) const {
  auto it = portMapping.find(portName);
  if (it != portMapping.end()) return it->second.rdmaNic;
  return "unknown";
}

/**
 * @brief 获取所有已配置的交换机端口名称
 * @return 端口名称列表
 */
std::vector<std::string> TopologyConfig::getAllPorts() const {
  std::vector<std::string> result;
  for (const auto& entry : portMapping) {
    result.push_back(entry.first);
  }
  return result;
}

/**
 * @brief 根据节点IP和RDMA网卡名称查找对应的交换机端口
 * @param nodeIp 节点IP地址
 * @param rdmaNic RDMA网卡名称
 * @return 交换机端口名称，未找到返回空字符串
 */
std::string TopologyConfig::getPortByNodeAndNic(const std::string& nodeIp, const std::string& rdmaNic) const {
  for (const auto& entry : portMapping) {
    if (entry.second.node == nodeIp && entry.second.rdmaNic == rdmaNic) return entry.first;
  }
  return "";
}

/**
 * @brief 从JSON配置文件加载拓扑配置
 * @param configPath 配置文件路径
 * @return 加载的TopologyConfig对象，失败返回空配置
 *
 * 解析字段：
 * - switch_ip: 交换机IP地址
 * - switch_name: 交换机名称
 * - grpc_port: gRPC服务端口
 * - port_mapping: 端口映射对象，包含node、rdma_nic、ranks等
 */
TopologyConfig TopologyConfig::loadFromFile(const std::string& configPath) {
  std::ifstream file(configPath);
  if (!file.is_open()) {
    WARN("NET_OBSERV: Failed to open topology config file: %s", configPath.c_str());
    return TopologyConfig{};
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();

  TopologyConfig cfg;
  cfg.switchIp = extractJsonString(content, "switch_ip");
  cfg.switchName = extractJsonString(content, "switch_name");
  cfg.grpcPort = static_cast<int>(extractJsonInt(content, "grpc_port"));
  if (cfg.grpcPort == 0) cfg.grpcPort = 50051;

  // Parse port_mapping object
  auto pmPos = content.find("\"port_mapping\"");
  if (pmPos == std::string::npos) {
    INFO(NCCL_NET, "NET_OBSERV: No port_mapping found in config");
    return cfg;
  }

  auto objStart = content.find("{", pmPos);
  if (objStart == std::string::npos) return cfg;

  // Track brace depth to find the matching closing brace
  int braceDepth = 0;
  size_t i = objStart;
  for (; i < content.length(); i++) {
    if (content[i] == '{') braceDepth++;
    else if (content[i] == '}') {
      braceDepth--;
      if (braceDepth == 0) break;
    }
  }
  std::string pmContent = content.substr(objStart + 1, i - objStart - 1);

  // Parse each port entry: "port_name": { ... }
  size_t pos = 0;
  while (pos < pmContent.length()) {
    // Find the next quoted key (port name)
    auto quoteStart = pmContent.find("\"", pos);
    if (quoteStart == std::string::npos) break;
    auto quoteEnd = pmContent.find("\"", quoteStart + 1);
    if (quoteEnd == std::string::npos) break;
    std::string portName = pmContent.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
    pos = quoteEnd + 1;

    // Find the value object start
    auto valStart = pmContent.find("{", pos);
    if (valStart == std::string::npos) break;

    // Find matching closing brace
    int depth = 1;
    size_t valEnd = valStart + 1;
    for (; valEnd < pmContent.length(); valEnd++) {
      if (pmContent[valEnd] == '{') depth++;
      else if (pmContent[valEnd] == '}') {
        depth--;
        if (depth == 0) break;
      }
    }
    std::string portObj = pmContent.substr(valStart + 1, valEnd - valStart - 1);

    PortMappingInfo info;
    info.node = extractJsonString(portObj, "node");
    info.rdmaNic = extractJsonString(portObj, "rdma_nic");

    // Parse ranks array
    auto ranksStart = portObj.find("\"ranks\"");
    if (ranksStart != std::string::npos) {
      auto arrStart = portObj.find("[", ranksStart);
      if (arrStart != std::string::npos) {
        auto arrEnd = portObj.find("]", arrStart);
        if (arrEnd != std::string::npos) {
          std::string arrStr = portObj.substr(arrStart + 1, arrEnd - arrStart - 1);
          std::stringstream ss(arrStr);
          std::string token;
          while (std::getline(ss, token, ',')) {
            token = trim(token);
            if (!token.empty()) {
              info.ranks.push_back(std::stoi(token));
            }
          }
        }
      }
    }

    cfg.portMapping[portName] = info;
    pos = valEnd + 1;
  }

  INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: Loaded topology config: switch=%s, ports=%zu",
       cfg.switchName.c_str(), cfg.portMapping.size());
  return cfg;
}

// ---- NicCounterReader ----

/**
 * @brief NicCounterReader构造函数
 */
NicCounterReader::NicCounterReader() {}

/**
 * @brief 发现系统中所有InfiniBand/RDMA网卡设备
 * @return 发现的网卡标识符列表（格式：devName:portNum）
 *
 * 扫描/sys/class/infiniband目录，枚举所有IB设备和端口，
 * 填充deviceCache_并返回网卡键列表
 */
std::vector<std::string> NicCounterReader::discoverNics() {
  std::vector<std::string> nics;
  struct stat st;
  if (stat(IB_SYSFS_BASE, &st) != 0 || !S_ISDIR(st.st_mode)) {
    INFO(NCCL_NET, "NET_OBSERV: sysfs path %s not found, IB counter monitoring disabled", IB_SYSFS_BASE);
    return nics;
  }

  DIR* dir = opendir(IB_SYSFS_BASE);
  if (!dir) {
    WARN("NET_OBSERV: Failed to open %s: %s", IB_SYSFS_BASE, strerror(errno));
    return nics;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_name[0] == '.') continue;
    std::string devName(entry->d_name);
    std::string devPath = std::string(IB_SYSFS_BASE) + "/" + devName;
    if (stat(devPath.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;

    std::string portsDir = devPath + "/ports";
    if (stat(portsDir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;

    DIR* ports = opendir(portsDir.c_str());
    if (!ports) continue;
    struct dirent* portEntry;
    while ((portEntry = readdir(ports)) != nullptr) {
      if (portEntry->d_name[0] == '.') continue;
      errno = 0;
      char* endptr = nullptr;
      long portNum = strtol(portEntry->d_name, &endptr, 10);
      if (endptr == portEntry->d_name || *endptr != '\0' || errno != 0) continue;

      std::string hwCountersDir = portsDir + "/" + portEntry->d_name + "/hw_counters";
      NicDeviceInfo info;
      info.ibDevName = devName;
      info.portNum = static_cast<int>(portNum);
      info.sysfsPath = hwCountersDir;

      std::string nicKey = devName + ":" + portEntry->d_name;
      deviceCache_[nicKey] = info;
      nics.push_back(nicKey);

      std::string retransPath = hwCountersDir + "/roce_adp_retrans";
      struct stat rs;
      bool available = (stat(retransPath.c_str(), &rs) == 0);
      INFO(NCCL_NET, "NET_OBSERV: Discovered IB device: %s, roce_adp_retrans=%s",
           nicKey.c_str(), available ? "available" : "unavailable");
    }
    closedir(ports);
  }
  closedir(dir);

  return nics;
}

/**
 * @brief 捕获指定网卡或所有网卡的计数器基线值
 * @param nicKey 网卡标识符（如"mlx5_0:1"），为空则捕获所有网卡
 *
 * 读取当前计数器值并保存为基线，后续通过readDelta计算增量
 */
void NicCounterReader::captureBaseline(const std::string& nicKey) {
  if (!nicKey.empty()) {
    auto counters = readCounters(nicKey);
    if (!counters.empty()) {
      baselines_[nicKey] = NicCounterSnapshot{
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(),
        counters
      };
      INFO(NCCL_NET, "NET_OBSERV: Baseline captured for %s", nicKey.c_str());
    }
  } else {
    for (const auto& entry : deviceCache_) {
      captureBaseline(entry.first);
    }
  }
}

/**
 * @brief 读取指定网卡的硬件计数器值
 * @param nicKey 网卡标识符（如"mlx5_0:1"）
 * @return 计数器名称到值的映射
 *
 * 从/sys/class/infiniband/<dev>/ports/<port>/hw_counters/读取：
 * - roce_adp_retrans
 * - roce_adp_retrans_to
 * - roce_slow_restart
 * - roce_slow_restart_cnps
 * - roce_slow_restart_cnp_acks
 */
std::map<std::string, int64_t> NicCounterReader::readCounters(const std::string& nicKey) {
  auto it = deviceCache_.find(nicKey);
  if (it == deviceCache_.end()) return {};

  std::map<std::string, int64_t> counters;
  const std::string& hwDir = it->second.sysfsPath;

  struct stat st;
  if (stat(hwDir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return {};

  for (int i = 0; i < RETRANS_COUNTER_NAMES_COUNT; i++) {
    std::string fpath = hwDir + "/" + RETRANS_COUNTER_NAMES[i];
    struct stat fs;
    if (stat(fpath.c_str(), &fs) != 0) continue;

    std::ifstream f(fpath);
    if (!f.is_open()) continue;
    std::string valStr;
    std::getline(f, valStr);
    f.close();
    try {
      counters[RETRANS_COUNTER_NAMES[i]] = std::stoll(trim(valStr));
    } catch (...) {}
  }

  return counters.empty() ? std::map<std::string, int64_t>{} : counters;
}

/**
 * @brief 计算指定网卡计数器自基线以来的增量
 * @param nicKey 网卡标识符
 * @return 计数器增量映射（只包含正值）
 *
 * 读取当前值并减去基线值，只返回有增长的计数器
 */
std::map<std::string, int64_t> NicCounterReader::readDelta(const std::string& nicKey) {
  auto baseIt = baselines_.find(nicKey);
  if (baseIt == baselines_.end()) return {};

  auto current = readCounters(nicKey);
  if (current.empty()) return {};

  std::map<std::string, int64_t> delta;
  for (const auto& curEntry : current) {
    auto baseValIt = baseIt->second.counters.find(curEntry.first);
    int64_t baseVal = (baseValIt != baseIt->second.counters.end()) ? baseValIt->second : 0;
    int64_t d = curEntry.second - baseVal;
    if (d > 0) delta[curEntry.first] = d;
  }
  return delta;
}

/**
 * @brief 检查指定RDMA网卡是否有重传计数器增长
 * @param rdmaNic RDMA网卡名称（如"mlx5_0"）
 * @return pair<是否有重传增长, 聚合的计数器增量>
 *
 * 遍历该网卡的所有端口，聚合各端口的计数器增量，
 * 并判断是否包含重传类计数器的增长
 */
std::pair<bool, std::map<std::string, int64_t>> NicCounterReader::checkNicRetrans(const std::string& rdmaNic) {
  std::map<std::string, int64_t> aggregated;
  bool hasRetryGrowth = false;

  for (const auto& entry : deviceCache_) {
    if (entry.first.find(rdmaNic) == std::string::npos) continue;
    auto delta = readDelta(entry.first);
    if (delta.empty()) continue;
    for (const auto& deltaEntry : delta) {
      bool isRetrans = false;
      for (int i = 0; i < RETRANS_COUNTER_NAMES_COUNT; i++) {
        if (deltaEntry.first == RETRANS_COUNTER_NAMES[i]) {
          isRetrans = true;
          break;
        }
      }
      if (isRetrans) hasRetryGrowth = true;
      aggregated[deltaEntry.first] += deltaEntry.second;
    }
  }

  return {hasRetryGrowth, aggregated};
}

// ---- SwitchEventServicer ----

/**
 * @brief 处理交换机发送的单次JSON事件（Unary RPC）
 * @param context gRPC上下文
 * @param request JSON事件请求
 * @param response 回复消息
 * @return gRPC状态码
 *
 * 将收到的请求加入events_队列，返回ret=0表示成功接收
 */
grpc::Status SwitchEventServicer::JsonSend(
    grpc::ServerContext* context,
    const ruijie_json::JsonRequest* request,
    ruijie_json::JsonReply* response) {
  (void)context;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    events_.push_back(*request);
  }
  response->set_ret(0);
  return grpc::Status::OK;
}

/**
 * @brief 处理交换机发送的流式JSON事件（Streaming RPC）
 * @param context gRPC上下文
 * @param stream 双向流对象
 * @return gRPC状态码
 *
 * 持续读取流中的事件请求，每个事件都加入队列并回复ACK(ret=0)
 */
grpc::Status SwitchEventServicer::JsonStreamSend(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<ruijie_json::JsonReply, ruijie_json::JsonRequest>* stream) {
  ruijie_json::JsonRequest request;
  while (stream->Read(&request)) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      events_.push_back(request);
    }
    ruijie_json::JsonReply reply;
    reply.set_ret(0);
    stream->Write(reply);
  }
  return grpc::Status::OK;
}

/**
 * @brief 获取所有待处理的事件请求
 * @return 待处理事件列表，返回后队列被清空
 *
 * 线程安全地取出events_中的所有事件，使用move避免拷贝
 */
std::vector<ruijie_json::JsonRequest> SwitchEventServicer::getPendingEvents() {
  std::lock_guard<std::mutex> lock(mtx_);
  std::vector<ruijie_json::JsonRequest> events = std::move(events_);
  events_.clear();
  return events;
}

// ---- NetworkObserver ----

/**
 * @brief NetworkObserver构造函数
 * @param topology 拓扑配置指针
 * @param rank 当前NCCL rank
 * @param pollInterval 事件轮询间隔（秒）
 */
NetworkObserver::NetworkObserver(TopologyConfig* topology, int rank, double pollInterval)
  : topology_(topology)
  , rank_(rank)
  , pollInterval_(pollInterval)
  , stopRequested_(false)
  , incidentId_(0) {
}

/**
 * @brief NetworkObserver析构函数
 * 自动调用stop()停止监控线程和gRPC服务器
 */
NetworkObserver::~NetworkObserver() {
  stop();
}

/**
 * @brief 启动网络观测器
 * @return 0表示成功，-1表示失败
 *
 * 启动流程：
 * 1. 启动gRPC服务器监听事件
 * 2. 发现IB网卡并捕获计数器基线
 * 3. 启动监控线程轮询事件
 */
int NetworkObserver::start() {
  grpc::ServerBuilder builder;
  builder.AddListeningPort("[::]:" + std::to_string(topology_->grpcPort),
                           grpc::InsecureServerCredentials());
  builder.RegisterService(&eventServicer_);
  grpcServer_ = builder.BuildAndStart();
  if (!grpcServer_) {
    WARN("NET_OBSERV: Failed to start gRPC server on port %d", topology_->grpcPort);
    return -1;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: gRPC server started on port %d, waiting for switch connection...",
       topology_->grpcPort);

  auto discoveredNics = nicReader_.discoverNics();
  if (!discoveredNics.empty()) {
    nicReader_.captureBaseline();
    INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: NIC counter baselines captured for %zu device(s)",
         discoveredNics.size());
  } else {
    INFO(NCCL_NET, "NET_OBSERV: No IB NICs discovered, NIC-side counter monitoring disabled");
  }

  stopRequested_ = false;
  monitorThread_ = std::make_unique<std::thread>(&NetworkObserver::monitorLoop, this);

  INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: Observer started, switch_poll=%.1fs", pollInterval_);
  return 0;
}

/**
 * @brief 停止网络观测器
 *
 * 停止流程：
 * 1. 设置停止标志
 * 2. 等待监控线程结束
 * 3. 关闭gRPC服务器
 */
void NetworkObserver::stop() {
  stopRequested_ = true;
  if (monitorThread_ && monitorThread_->joinable()) {
    monitorThread_->join();
  }
  if (grpcServer_) {
    grpcServer_->Shutdown();
  }
  INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: Observer stopped");
}

/**
 * @brief 监控线程主循环
 *
 * 周期性执行：
 * 1. 获取待处理事件
 * 2. 调用handleEvent处理每个事件
 * 3. 休眠pollInterval_秒
 */
void NetworkObserver::monitorLoop() {
  while (!stopRequested_) {
    try {
      auto events = eventServicer_.getPendingEvents();
      for (const auto& event : events) {
        handleEvent(event);
      }
    } catch (const std::exception& e) {
      INFO(NCCL_NET, "NET_OBSERV: Monitor error: %s", e.what());
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(pollInterval_));
  }
}

/**
 * @brief 根据NCCL异常确认告警
 * @param errorText NCCL错误文本
 *
 * 解析异常信息中的RDMA网卡和对端IP，
 * 匹配activeIncidents_中的未确认告警并升级为CONFIRMED状态
 */
void NetworkObserver::confirmFromException(const std::string& errorText) {
  std::string rdmaNic;
  std::string peerIp;
  bool isNcclIbError = false;

  std::smatch match;

  // Check for NCCL IB error patterns
  const std::regex ibErrorPatterns[] = {
    std::regex("IBV_WC_RETRY_EXC_ERR"),
    std::regex("Got CQE with error"),
    std::regex("Got completion.*status=IBV_WC_RETRY_EXC_ERR"),
  };

  for (const auto& pattern : ibErrorPatterns) {
    if (std::regex_search(errorText, match, pattern)) {
      isNcclIbError = true;
      break;
    }
  }

  if (!isNcclIbError) {
    std::string ncclKeywords[] = {"NCCL", "nccl", "Distributed", "dist_backend"};
    bool hasNcclKeyword = false;
    for (const auto& kw : ncclKeywords) {
      if (errorText.find(kw) != std::string::npos) {
        hasNcclKeyword = true;
        break;
      }
    }
    if (!hasNcclKeyword) return;
  }

  if (std::regex_search(errorText, match, NCCL_HCA_PATTERN)) {
    rdmaNic = match[1].str();
  }

  if (std::regex_search(errorText, match, NCCL_PEER_IP_PATTERN)) {
    peerIp = match[1].str();
  }

  INFO(NCCL_NET, "NET_OBSERV: Caught NCCL exception, rdma_nic=%s, peer_ip=%s, is_ib_error=%d",
       rdmaNic.c_str(), peerIp.c_str(), isNcclIbError);

  std::unordered_map<std::string, Incident> incidentsSnapshot;
  {
    std::lock_guard<std::mutex> lock(incidentsMutex_);
    incidentsSnapshot = activeIncidents_;
  }

  std::vector<Incident*> matched;
  for (auto& entry : incidentsSnapshot) {
    if (entry.second.confirmed) continue;
    if (!rdmaNic.empty() && entry.second.rdmaNic == rdmaNic) {
      matched.push_back(&entry.second);
    } else if (rdmaNic.empty()) {
      matched.push_back(&entry.second);
    }
  }

  if (matched.empty() && !incidentsSnapshot.empty()) {
    for (auto& entry : incidentsSnapshot) {
      if (!entry.second.confirmed) matched.push_back(&entry.second);
    }
  }

  for (auto* incident : matched) {
    std::pair<bool, std::map<std::string, int64_t>> ret = nicReader_.checkNicRetrans(incident->rdmaNic);
    if (!ret.second.empty()) {
      incident->nicDelta = ret.second;
    }

    if (!peerIp.empty()) {
      std::string peerPort = topology_->getPortByNodeAndNic(peerIp, incident->rdmaNic);
      if (!peerPort.empty()) {
        incident->faultPaths = buildFaultPath({incident->portName}, incident->deviceModel, peerPort);
      } else {
        for (const auto& pmEntry : topology_->portMapping) {
          if (pmEntry.second.node == peerIp) {
            incident->faultPaths = buildFaultPath({incident->portName}, incident->deviceModel, pmEntry.first);
            break;
          }
        }
      }
    }

    incident->confirmed = true;
    incident->confirmTime = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    incident->ncclError = errorText.substr(0, 200);
    emitConfirmedAlert(*incident);
  }
}

// 事件类型常量定义
constexpr uint32_t GRPC_JSON_EVENT_SAMPLE_INGRESS_PORT_PKT_DROP = 0x00020000;
constexpr uint32_t GRPC_JSON_EVENT_SAMPLE_LLDP_INFO = 0x10080000;

/**
 * @brief 统一事件处理入口
 * @param event gRPC事件请求
 *
 * 根据event.json_event()的值分发到不同的处理函数：
 * - 0x00020000 (INGRESS_PORT_PKT_DROP): 交换机端口丢包实时事件
 * - 0x10080000 (LLDP_INFO): LLDP邻居信息周期性上报事件
 */
void NetworkObserver::handleEvent(const ruijie_json::JsonRequest& event) {
  uint32_t eventId = event.json_event();

  if (eventId == GRPC_JSON_EVENT_SAMPLE_LLDP_INFO) {
    handleLldpEvent(event);
  } else if (eventId == GRPC_JSON_EVENT_SAMPLE_INGRESS_PORT_PKT_DROP) {
    handleSwitchDropEvent(event);
  } else {
    INFO(NCCL_NET, "NET_OBSERV: Unknown event type 0x%08x, skipping", eventId);
  }
}

/**
 * @brief 处理交换机端口丢包事件（实时事件）
 * @param event gRPC事件请求
 *
 * 处理流程：
 * 1. 提取端口名、丢包数、时间戳
 * 2. 查询拓扑获取关联的rank和RDMA网卡
 * 3. 检查NIC重传计数器
 * 4. 根据重传状态生成WATCH/PREDICTIVE/CONFIRMED告警
 */
void NetworkObserver::handleSwitchDropEvent(const ruijie_json::JsonRequest& event) {
  std::string deviceModel = "unknown";
  if (event.has_device_info()) {
    deviceModel = event.device_info().device_model();
  }

  std::string jsonString = event.json_string();
  std::string portName;
  int64_t dropCount = 0;
  std::string timestampRaw;

  if (!jsonString.empty()) {
    portName = extractJsonString(jsonString, "port_name");
    dropCount = extractJsonInt(jsonString, "dropped");
    timestampRaw = extractJsonString(jsonString, "timestamp");
  }

  if (portName.empty()) {
    for (const auto& entry : topology_->portMapping) {
      portName = entry.first;
      break;
    }
  }
  if (portName.empty()) {
    INFO(NCCL_NET, "NET_OBSERV: Event has no port_name and topology has no ports, skipping");
    return;
  }

  int incId = ++incidentId_;

  auto affectedRanks = topology_->getRanksForPort(portName);
  std::string rdmaNic = topology_->getRdmaNicForPort(portName);
  auto pathLines = buildFaultPath({portName}, deviceModel);

  std::pair<bool, std::map<std::string, int64_t>> ret = nicReader_.checkNicRetrans(rdmaNic);

  INFO(NCCL_NET, "NET_OBSERV: Switch drop event (0x%08x) on %s, rdma_nic=%s, has_retry=%d",
       GRPC_JSON_EVENT_SAMPLE_INGRESS_PORT_PKT_DROP, portName.c_str(), rdmaNic.c_str(), ret.first);

  char tagBuf[64];
  snprintf(tagBuf, sizeof(tagBuf), "NET-%d", 10000 + incId);

  std::string formattedTs = formatTimestamp(timestampRaw);

  Incident incident;
  incident.incidentId = incId;
  incident.incidentTag = tagBuf;
  incident.portName = portName;
  incident.switchName = topology_->switchName;
  incident.deviceModel = deviceModel;
  incident.dropCount = dropCount;
  incident.timestamp = formattedTs;
  incident.affectedRanks = affectedRanks;
  incident.faultPaths = pathLines;
  incident.rdmaNic = rdmaNic;
  incident.nicDelta = ret.second;
  incident.hasRetryGrowth = ret.first;
  incident.createTime = std::chrono::duration<double>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  incident.confirmed = false;
  incident.confirmTime = 0;
  incident.ncclError = "";

  if (ret.first) {
    auto retransToIt = ret.second.find("roce_adp_retrans_to");
    bool hasRetransTo = (retransToIt != ret.second.end() && retransToIt->second > 0);
    if (hasRetransTo) {
      incident.confirmed = true;
      incident.confirmTime = std::chrono::duration<double>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      incident.ncclError = "roce_adp_retrans_to > 0: RDMA adaptive retransmission timeout reached";
      emitSwitchDropAlert(incident);
      emitConfirmedAlert(incident);
    } else {
      emitSwitchDropAlert(incident);
      emitPredictiveAlert(incident);
    }
  } else {
    emitSwitchDropAlert(incident);
  }

  {
    std::lock_guard<std::mutex> lock(incidentsMutex_);
    activeIncidents_[portName] = incident;
  }
}

/**
 * @brief 输出交换机丢包告警（WATCH级别）
 * @param incident 告警事件对象
 *
 * 黄色告警，表示检测到交换机丢包，正在监控NIC计数器
 */
void NetworkObserver::emitSwitchDropAlert(const Incident& incident) {
  std::string rankStr;
  for (size_t i = 0; i < incident.affectedRanks.size(); i++) {
    if (i > 0) rankStr += ", ";
    rankStr += "Rank-" + std::to_string(incident.affectedRanks[i]);
  }
  std::string nicRetryStr = incident.nicDelta.empty()
    ? "Not yet detected (monitoring...)"
    : formatCounterDelta(incident.nicDelta);

  INFO(NCCL_NET, "%s%s%s", COLOR_YELLOW, COLOR_BOLD,
       "========================================================================");
  INFO(NCCL_NET, "%s%s[NETWORK_ADVISOR][WATCH] Switch Packet Drop Detected%s",
       COLOR_YELLOW, COLOR_BOLD, COLOR_RESET);
  INFO(NCCL_NET, "%s%s========================================================================",
       COLOR_YELLOW, COLOR_RESET);
  INFO(NCCL_NET, "  %sIncident ID:%s    %s", COLOR_CYAN, COLOR_RESET, incident.incidentTag.c_str());
  INFO(NCCL_NET, "  %sTimestamp:%s     %s", COLOR_CYAN, COLOR_RESET, incident.timestamp.c_str());
  INFO(NCCL_NET, "  %sAffected Ranks:%s %s", COLOR_CYAN, COLOR_RESET, rankStr.c_str());
  INFO(NCCL_NET, "  %sAffected Port:%s  %s", COLOR_CYAN, COLOR_RESET, incident.portName.c_str());
  INFO(NCCL_NET, "  %sDrop Count:%s     %ld", COLOR_CYAN, COLOR_RESET, (long)incident.dropCount);
  INFO(NCCL_NET, "  %sRDMA NIC:%s       %s", COLOR_CYAN, COLOR_RESET, incident.rdmaNic.c_str());
  INFO(NCCL_NET, "  %sNIC Retry:%s      %s", COLOR_CYAN, COLOR_RESET, nicRetryStr.c_str());
  for (const auto& pathLine : incident.faultPaths) {
    INFO(NCCL_NET, "  %sFault Path:%s     %s", COLOR_CYAN, COLOR_RESET, pathLine.c_str());
  }
  INFO(NCCL_NET, "  %sStatus:%s        %sWATCHING%s - Switch drop detected, checking NIC counters",
       COLOR_CYAN, COLOR_RESET, COLOR_YELLOW, COLOR_RESET);
  INFO(NCCL_NET, "%s%s========================================================================",
       COLOR_YELLOW, COLOR_RESET);
}

/**
 * @brief 输出预测性告警（PREDICTIVE级别）
 * @param incident 告警事件对象
 *
 * 洋红色告警，表示交换机丢包导致RDMA重传但尚未超时，
 * 预测可能发生NCCL超时，建议 preemptive 检查
 */
void NetworkObserver::emitPredictiveAlert(const Incident& incident) {
  std::string rankStr;
  for (size_t i = 0; i < incident.affectedRanks.size(); i++) {
    if (i > 0) rankStr += ", ";
    rankStr += "Rank-" + std::to_string(incident.affectedRanks[i]);
  }

  INFO(NCCL_NET, "%s%s%s", COLOR_MAGENTA, COLOR_BOLD,
       "========================================================================");
  INFO(NCCL_NET, "%s%s[NETWORK_ADVISOR][PREDICTIVE] RDMA Retransmission Detected - Fault Path Predicted%s",
       COLOR_MAGENTA, COLOR_BOLD, COLOR_RESET);
  INFO(NCCL_NET, "%s%s========================================================================",
       COLOR_MAGENTA, COLOR_RESET);
  INFO(NCCL_NET, "  %sIncident ID:%s    %s", COLOR_CYAN, COLOR_RESET, incident.incidentTag.c_str());
  INFO(NCCL_NET, "  %sTimestamp:%s     %s", COLOR_CYAN, COLOR_RESET, incident.timestamp.c_str());
  INFO(NCCL_NET, "  %sAffected Ranks:%s %s", COLOR_CYAN, COLOR_RESET, rankStr.c_str());
  INFO(NCCL_NET, "  %sAffected Port:%s  %s", COLOR_CYAN, COLOR_RESET, incident.portName.c_str());
  INFO(NCCL_NET, "  %sSwitch Drop:%s    %ld packets", COLOR_CYAN, COLOR_RESET, (long)incident.dropCount);
  INFO(NCCL_NET, "  %sRDMA NIC:%s       %s", COLOR_CYAN, COLOR_RESET, incident.rdmaNic.c_str());
  INFO(NCCL_NET, "  %sNIC Counter Delta:%s", COLOR_CYAN, COLOR_RESET);
  for (const auto& entry : incident.nicDelta) {
    std::string marker = (entry.first == "roce_adp_retrans") ? " <<<" : "";
    INFO(NCCL_NET, "    %s%s:%s +%ld%s", COLOR_DIM, entry.first.c_str(), COLOR_RESET, (long)entry.second, marker.c_str());
  }
  for (const auto& pathLine : incident.faultPaths) {
    INFO(NCCL_NET, "  %sPredicted Fault Path:%s %s", COLOR_CYAN, COLOR_RESET, pathLine.c_str());
  }
  INFO(NCCL_NET, "  %sRoot Cause:%s     [Congestion/Link Issue] Switch drop + NIC retry growth",
       COLOR_CYAN, COLOR_RESET);
  INFO(NCCL_NET, "  %sImpact:%s         RDMA retransmission in progress, training may degrade",
       COLOR_CYAN, COLOR_RESET);
  INFO(NCCL_NET, "  %sStatus:%s        %sPREDICTIVE%s - RDMA is retrying, timeout not yet reached",
       COLOR_CYAN, COLOR_RESET, COLOR_MAGENTA, COLOR_RESET);
  INFO(NCCL_NET, "  %sNext Step:%s      Waiting for NCCL IBV_WC_RETRY_EXC_ERR exception",
       COLOR_CYAN, COLOR_RESET);
  INFO(NCCL_NET, "  %sSuggestion:%s     Pre-emptive: check cable/optical module at %s",
       COLOR_CYAN, COLOR_RESET, incident.portName.c_str());
  INFO(NCCL_NET, "%s%s========================================================================",
       COLOR_MAGENTA, COLOR_RESET);
}

/**
 * @brief 输出已确认告警（CONFIRMED级别）
 * @param incident 告警事件对象
 *
 * 红色告警，表示RDMA重传超时已发生，NCCL训练可能已卡死，
 * 需要立即处理
 */
void NetworkObserver::emitConfirmedAlert(const Incident& incident) {
  std::string rankStr;
  for (size_t i = 0; i < incident.affectedRanks.size(); i++) {
    if (i > 0) rankStr += ", ";
    rankStr += "Rank-" + std::to_string(incident.affectedRanks[i]);
  }

  auto now = std::chrono::system_clock::now();
  auto nowTimeT = std::chrono::system_clock::to_time_t(now);
  char confirmTs[64];
  struct tm utcTm;
  gmtime_r(&nowTimeT, &utcTm);
  strftime(confirmTs, sizeof(confirmTs), "%Y-%m-%d %H:%M:%S UTC", &utcTm);

  std::string elapsedStr = "N/A";
  if (incident.confirmTime > 0) {
    double elapsed = incident.confirmTime - incident.createTime;
    char elapsedBuf[32];
    snprintf(elapsedBuf, sizeof(elapsedBuf), "%.1fs", elapsed);
    elapsedStr = elapsedBuf;
  }

  INFO(NCCL_NET, "%s%s%s", COLOR_RED, COLOR_BOLD,
       "========================================================================");
  INFO(NCCL_NET, "%s%s[NETWORK_ADVISOR][CONFIRMED] RDMA Timeout - Fault Path Confirmed!%s",
       COLOR_RED, COLOR_BOLD, COLOR_RESET);
  INFO(NCCL_NET, "%s%s========================================================================",
       COLOR_RED, COLOR_RESET);
  INFO(NCCL_NET, "  %sIncident ID:%s    %s", COLOR_CYAN, COLOR_RESET, incident.incidentTag.c_str());
  INFO(NCCL_NET, "  %sOriginal Time:%s  %s", COLOR_CYAN, COLOR_RESET, incident.timestamp.c_str());
  INFO(NCCL_NET, "  %sConfirmed At:%s   %s", COLOR_CYAN, COLOR_RESET, confirmTs);
  INFO(NCCL_NET, "  %sTime to Confirm:%s %s", COLOR_CYAN, COLOR_RESET, elapsedStr.c_str());
  INFO(NCCL_NET, "  %sAffected Ranks:%s %s", COLOR_CYAN, COLOR_RESET, rankStr.c_str());
  INFO(NCCL_NET, "  %sAffected Port:%s  %s", COLOR_CYAN, COLOR_RESET, incident.portName.c_str());
  INFO(NCCL_NET, "  %sSwitch Drop:%s    %ld packets", COLOR_CYAN, COLOR_RESET, (long)incident.dropCount);
  INFO(NCCL_NET, "  %sRDMA NIC:%s       %s", COLOR_CYAN, COLOR_RESET, incident.rdmaNic.c_str());
  INFO(NCCL_NET, "  %sNIC Counter Delta (cumulative):%s", COLOR_CYAN, COLOR_RESET);
  for (const auto& entry : incident.nicDelta) {
    std::string marker;
    if (entry.first == "roce_adp_retrans" || entry.first == "roce_adp_retrans_to") marker = " <<<";
    INFO(NCCL_NET, "    %s%s:%s +%ld%s", COLOR_DIM, entry.first.c_str(), COLOR_RESET, (long)entry.second, marker.c_str());
  }
  for (const auto& pathLine : incident.faultPaths) {
    INFO(NCCL_NET, "  %sConfirmed Fault Path:%s %s", COLOR_CYAN, COLOR_RESET, pathLine.c_str());
  }
  if (!incident.ncclError.empty()) {
    INFO(NCCL_NET, "  %sNCCL Error:%s    %s", COLOR_CYAN, COLOR_RESET, incident.ncclError.c_str());
  }
  INFO(NCCL_NET, "  %sRoot Cause:%s     [Confirmed] Switch drop caused RDMA timeout on %s",
       COLOR_CYAN, COLOR_RESET, incident.rdmaNic.c_str());
  INFO(NCCL_NET, "  %sImpact:%s         Training stalled, NCCL timeout likely",
       COLOR_CYAN, COLOR_RESET);
  INFO(NCCL_NET, "  %sStatus:%s        %s%sCONFIRMED%s - RDMA timeout reached",
       COLOR_CYAN, COLOR_RESET, COLOR_RED, COLOR_BOLD, COLOR_RESET);
  INFO(NCCL_NET, "  %sSuggestion:%s     Immediate action required: check %s and %s",
       COLOR_CYAN, COLOR_RESET, incident.portName.c_str(), incident.rdmaNic.c_str());
  INFO(NCCL_NET, "%s%s========================================================================",
       COLOR_RED, COLOR_RESET);
}

/**
 * @brief 格式化计数器增量为可读字符串
 * @param delta 计数器增量映射
 * @return 格式化后的字符串（如"adp_retrans=+15, slow_restart=+3"）
 *
 * 将计数器名称缩短（去掉roce_前缀），格式为"name=+value"
 */
std::string NetworkObserver::formatCounterDelta(const std::map<std::string, int64_t>& delta) {
  if (delta.empty()) return "no change";
  std::string result;
  for (const auto& entry : delta) {
    if (!result.empty()) result += ", ";
    std::string shortName = entry.first;
    auto underscorePos = entry.first.find('_');
    if (underscorePos != std::string::npos) {
      shortName = entry.first.substr(underscorePos + 1);
    }
    result += shortName + "=+" + std::to_string(entry.second);
  }
  return result;
}

/**
 * @brief 构建故障路径描述字符串
 * @param affectedPorts 受影响的交换机端口列表
 * @param deviceModel 交换机设备型号
 * @param peerPort 对端端口（可选，来自异常上下文）
 * @return 故障路径描述列表
 *
 * 构建端到端路径：Rank-X -> mlx5_X -> Switch(port) -> Switch(peerPort) -> mlx5_X -> Rank-Y
 */
std::vector<std::string> NetworkObserver::buildFaultPath(
    const std::vector<std::string>& affectedPorts,
    const std::string& deviceModel,
    const std::string& peerPort) {
  if (affectedPorts.empty()) {
    return {deviceModel + " (" + ")"};
  }

  std::vector<std::string> paths;
  for (const auto& faultPort : affectedPorts) {
    std::string srcNic = topology_->getRdmaNicForPort(faultPort);
    std::string srcNode = topology_->getNodeForPort(faultPort);
    auto srcRanks = topology_->getRanksForPort(faultPort);
    if (srcRanks.empty()) continue;

    std::vector<std::string> otherPorts;
    if (!peerPort.empty()) {
      otherPorts.push_back(peerPort);
    } else {
      for (const auto& pmEntry : topology_->portMapping) {
        if (pmEntry.second.node != srcNode && pmEntry.second.rdmaNic == srcNic) {
          otherPorts.push_back(pmEntry.first);
        }
      }
    }

    for (const auto& otherPort : otherPorts) {
      if (otherPort == faultPort) continue;
      auto dstRanks = topology_->getRanksForPort(otherPort);
      if (dstRanks.empty()) continue;
      std::string srcRankStr = "Rank-" + std::to_string(srcRanks[0]) + "~" + std::to_string(srcRanks.back());
      std::string dstRankStr = "Rank-" + std::to_string(dstRanks[0]) + "~" + std::to_string(dstRanks.back());
      std::string path = srcRankStr + " -> " + srcNic + " -> "
          + deviceModel + "(" + faultPort + ") -> " + deviceModel + "(" + otherPort + ") -> "
          + srcNic + " -> " + dstRankStr;
      paths.push_back(path);
    }
  }
  return paths;
}

/**
 * @brief 格式化时间戳为可读字符串
 * @param timestampRaw 原始时间戳字符串（支持秒/毫秒/微秒）
 * @return 格式化后的UTC时间字符串（如"2026-05-18 16:30:00 UTC"）
 *
 * 自动检测时间戳精度（16位=微秒，13位=毫秒，10位=秒）并转换为UTC格式
 */
std::string NetworkObserver::formatTimestamp(const std::string& timestampRaw) {
  if (timestampRaw.empty()) {
    auto now = std::chrono::system_clock::now();
    auto nowTimeT = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    struct tm utcTm;
    gmtime_r(&nowTimeT, &utcTm);
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S UTC", &utcTm);
    return buf;
  }

  try {
    int64_t ts = std::stoll(timestampRaw);
    if (ts > 1000000000000000LL) {
      ts /= 1000000;
    } else if (ts > 1000000000000LL) {
      ts /= 1000;
    }
    auto timeT = static_cast<time_t>(ts);
    char buf[64];
    struct tm utcTm;
    gmtime_r(&timeT, &utcTm);
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S UTC", &utcTm);
    return buf;
  } catch (...) {}

  auto now = std::chrono::system_clock::now();
  auto nowTimeT = std::chrono::system_clock::to_time_t(now);
  char buf[64];
  struct tm utcTm;
  gmtime_r(&nowTimeT, &utcTm);
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S UTC", &utcTm);
  return buf;
}

// ---- LLDP Event Handling ----

/**
 * @brief 从JSON字符串中提取指定数组的第index个元素
 * @param json 输入的JSON字符串
 * @param arrayName 数组字段名（如"data"）
 * @param index 要提取的元素索引（从0开始）
 * @return 提取到的JSON对象字符串，失败返回空字符串
 * 
 * 示例：json = {"data":[{"a":1},{"b":2}]}, arrayName="data", index=1
 *       返回：{"b":2}
 */
static std::string extractJsonArrayItem(const std::string& json, const std::string& arrayName, size_t index) {
  std::string search = "\"" + arrayName + "\":";
  auto pos = json.find(search);
  if (pos == std::string::npos) {
    search = "\"" + arrayName + "\" :";
    pos = json.find(search);
    if (pos == std::string::npos) return "";
  }
  pos += search.length();
  while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
  if (pos >= json.length() || json[pos] != '[') return "";
  pos++;

  size_t braceCount = 0;
  size_t itemStart = pos;
  size_t itemIdx = 0;

  for (size_t i = pos; i < json.length(); i++) {
    if (json[i] == '{') braceCount++;
    else if (json[i] == '}') {
      braceCount--;
      if (braceCount == 0) {
        if (itemIdx == index) {
          return json.substr(itemStart, i - itemStart + 1);
        }
        itemIdx++;
        itemStart = i + 1;
        while (itemStart < json.length() && (json[itemStart] == ',' || json[itemStart] == ' ' || json[itemStart] == '\t' || json[itemStart] == '\n')) itemStart++;
        if (itemStart < json.length() && json[itemStart] == '{') {
          i = itemStart - 1;
        }
      }
    }
  }
  return "";
}

/**
 * @brief 统计JSON数组中的元素个数
 * @param json 输入的JSON字符串
 * @param arrayName 数组字段名
 * @return 数组元素个数，失败返回0
 * 
 * 通过匹配大括号{}来计数，支持嵌套对象
 */
static size_t countJsonArrayItems(const std::string& json, const std::string& arrayName) {
  std::string search = "\"" + arrayName + "\":";
  auto pos = json.find(search);
  if (pos == std::string::npos) {
    search = "\"" + arrayName + "\" :";
    pos = json.find(search);
    if (pos == std::string::npos) return 0;
  }
  pos += search.length();
  while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
  if (pos >= json.length() || json[pos] != '[') return 0;
  pos++;

  size_t braceCount = 0;
  size_t count = 0;

  for (size_t i = pos; i < json.length(); i++) {
    if (json[i] == '{') {
      if (braceCount == 0) count++;
      braceCount++;
    } else if (json[i] == '}') {
      braceCount--;
    } else if (json[i] == ']' && braceCount == 0) {
      break;
    }
  }
  return count;
}

/**
 * @brief 解析LLDP事件JSON字符串，提取邻居信息列表
 * @param jsonString LLDP事件JSON字符串
 * @return LLDP邻居信息列表
 * 
 * 解析字段包括：
 * - lldp_rem_man_addr: 远端管理IP（服务器IP）
 * - lldp_loc_port_desc: 本地端口描述（交换机端口）
 * - lldp_rem_port_desc: 远端端口描述（服务器网卡名）
 * - lldp_rem_chassis_id: 远端机箱ID（MAC地址）
 * - lldp_rem_sys_name: 远端系统名称（主机名）
 * - lldp_rem_sys_desc: 远端系统描述（OS信息）
 * - lldp_rem_time_mark: 时间戳
 */
std::vector<LldpNeighborInfo> NetworkObserver::parseLldpJson(const std::string& jsonString) {
  std::vector<LldpNeighborInfo> results;

  size_t dataCount = countJsonArrayItems(jsonString, "data");
  if (dataCount == 0) {
    INFO(NCCL_NET, "NET_OBSERV: LLDP JSON has no 'data' array items");
    return results;
  }

  for (size_t i = 0; i < dataCount; i++) {
    std::string item = extractJsonArrayItem(jsonString, "data", i);
    if (item.empty()) continue;

    LldpNeighborInfo info;
    info.nodeIp = extractJsonString(item, "lldp_rem_man_addr");
    info.switchPort = extractJsonString(item, "lldp_loc_port_desc");
    info.remotePortDesc = extractJsonString(item, "lldp_rem_port_desc");
    info.remoteChassisId = extractJsonString(item, "lldp_rem_chassis_id");
    info.remoteSysName = extractJsonString(item, "lldp_rem_sys_name");
    info.remoteSysDesc = extractJsonString(item, "lldp_rem_sys_desc");
    info.timeMark = extractJsonInt(item, "lldp_rem_time_mark");

    if (!info.switchPort.empty() && !info.nodeIp.empty()) {
      results.push_back(info);
    }
  }

  return results;
}

/**
 * @brief 获取本机IP地址
 * @return 本机IP地址字符串，获取失败返回空字符串
 *
 * 通过读取网络接口获取本机IP，优先返回非回环地址
 */
std::string NetworkObserver::getLocalIpAddress() {
  // 首先尝试从环境变量获取
  const char* envIp = ncclGetEnv("NET_OBSERV_LOCAL_IP");
  if (envIp && strlen(envIp) > 0) {
    return std::string(envIp);
  }

  // 通过hostname获取
  std::array<char, 128> hostname;
  if (gethostname(hostname.data(), hostname.size()) == 0) {
    struct hostent* host = gethostbyname(hostname.data());
    if (host) {
      struct in_addr** addr_list = reinterpret_cast<struct in_addr**>(host->h_addr_list);
      for (int i = 0; addr_list[i] != nullptr; i++) {
        std::string ip = inet_ntoa(*addr_list[i]);
        // 跳过回环地址
        if (ip != "127.0.0.1") {
          return ip;
        }
      }
    }
  }

  // 回退：尝试从网络接口获取
  std::array<char, 256> buffer;
  FILE* fp = popen("hostname -I 2>/dev/null | awk '{print $1}'", "r");
  if (fp) {
    if (fgets(buffer.data(), buffer.size(), fp) != nullptr) {
      std::string ip = trim(std::string(buffer.data()));
      pclose(fp);
      if (!ip.empty()) {
        return ip;
      }
    } else {
      pclose(fp);
    }
  }

  INFO(NCCL_NET, "NET_OBSERV: Failed to get local IP address");
  return "";
}

/**
 * @brief 执行shell命令并获取输出
 * @param cmd 要执行的命令
 * @return 命令的标准输出，执行失败返回空字符串
 */
std::string NetworkObserver::executeCommand(const std::string& cmd) {
  std::array<char, 256> buffer;
  std::string result;

  FILE* fp = popen(cmd.c_str(), "r");
  if (!fp) {
    return "";
  }

  while (fgets(buffer.data(), buffer.size(), fp) != nullptr) {
    result += buffer.data();
  }

  pclose(fp);
  return trim(result);
}

/**
 * @brief 通过SSH远程查询节点的RDMA NIC名称
 * @param remoteIp 远程节点IP地址
 * @param netdev 网卡设备名（如"ens43f0np0"）
 * @return RDMA NIC名称（如"mlx5_0"），查询失败返回空字符串
 *
 * 通过SSH连接到远程节点，读取/sys/class/net/<netdev>/device/infiniband获取mlx5设备名
 * 需要预先配置SSH免密登录
 */
std::string NetworkObserver::queryRemoteRdmaNic(const std::string& remoteIp, const std::string& netdev) {
  // 构建SSH命令
  std::string sshCmd = "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no " + remoteIp +
                       " 'ls /sys/class/net/" + netdev + "/device/infiniband/ 2>/dev/null | grep mlx5_'";

  std::string result = executeCommand(sshCmd);

  if (!result.empty() && result.find("mlx5_") != std::string::npos) {
    // 提取mlx5_X部分
    size_t pos = result.find("mlx5_");
    size_t end = pos + 5;
    while (end < result.length() && result[end] >= '0' && result[end] <= '9') end++;
    return result.substr(pos, end - pos);
  }

  INFO(NCCL_NET, "NET_OBSERV: Failed to query remote RDMA NIC from %s for %s", remoteIp.c_str(), netdev.c_str());
  return "";
}

/**
 * @brief 从服务器网卡描述获取RDMA NIC名称（支持本地和远程查询）
 * @param portDesc 服务器侧网卡描述（如"ens43f0np0"、"mlx5_0"）
 * @param nodeIp 节点IP地址，用于判断是本地还是远程查询
 * @return 获取到的RDMA NIC名称（如"mlx5_0"），无法获取返回空字符串
 *
 * 获取规则：
 * 1. 如果描述中包含"mlx5_"，直接提取mlx5_X部分
 * 2. 如果是本节点，通过本地sysfs获取mlx5_X
 * 3. 如果是远程节点，通过SSH查询获取mlx5_X
 * 4. 其他情况返回空字符串
 */
std::string NetworkObserver::inferRdmaNicFromPortDesc(const std::string& portDesc, const std::string& nodeIp) {
  // 如果描述中直接包含mlx5_，直接提取
  if (portDesc.find("mlx5_") != std::string::npos) {
    size_t pos = portDesc.find("mlx5_");
    size_t end = pos + 5;
    while (end < portDesc.length() && portDesc[end] >= '0' && portDesc[end] <= '9') end++;
    return portDesc.substr(pos, end - pos);
  }

  // 如果是网卡名（ens/enp/eth开头）
  if (portDesc.find("ens") != std::string::npos ||
      portDesc.find("enp") != std::string::npos ||
      portDesc.find("eth") != std::string::npos) {
    
    // 判断是本节点还是远程节点
    std::string localIp = getLocalIpAddress();
    bool isLocalNode = (nodeIp == localIp);
    
    if (isLocalNode) {
      // 本节点：通过本地sysfs获取mlx5设备
      std::string ibPath = "/sys/class/net/" + portDesc + "/device/infiniband";
      DIR* dir = opendir(ibPath.c_str());
      if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
          // 跳过 . 和 ..
          if (entry->d_name[0] == '.') continue;
          // 查找 mlx5_X 格式的目录名
          std::string name(entry->d_name);
          if (name.find("mlx5_") != std::string::npos) {
            closedir(dir);
            return name;
          }
        }
        closedir(dir);
      }
      // 如果sysfs读取失败，回退到默认的mlx5_0
      INFO(NCCL_NET, "NET_OBSERV: Failed to read IB device from sysfs for %s, fallback to mlx5_0",
           portDesc.c_str());
      return "mlx5_0";
    } else {
      // 远程节点：通过SSH查询获取mlx5设备
      INFO(NCCL_NET, "NET_OBSERV: Querying remote node %s for RDMA NIC of %s", nodeIp.c_str(), portDesc.c_str());
      return queryRemoteRdmaNic(nodeIp, portDesc);
    }
  }

  return "";
}

/**
 * @brief 根据LLDP邻居信息更新网络拓扑映射
 * @param lldpInfos LLDP邻居信息列表
 * @param deviceModel 交换机设备型号
 * 
 * 构建交换机端口到服务器IP和RDMA NIC的映射关系，
 * 并更新到topology_->portMapping中
 */
void NetworkObserver::updateTopologyFromLldp(const std::vector<LldpNeighborInfo>& lldpInfos,
                                              const std::string& deviceModel) {
  std::vector<LldpTopologyEntry> entries;

  // Update switch name/model from LLDP context
  if (!deviceModel.empty()) {
    topology_->switchName = deviceModel;
  }

  for (const auto& info : lldpInfos) {
    LldpTopologyEntry entry;
    entry.switchPort = info.switchPort;
    entry.nodeIp = info.nodeIp;
    // 传递nodeIp以支持本地/远程查询判断
    entry.rdmaNic = inferRdmaNicFromPortDesc(info.remotePortDesc, info.nodeIp);
    entry.description = info.remoteSysName + " (" + info.remotePortDesc + ")";

    if (!entry.rdmaNic.empty()) {
      entries.push_back(entry);

      PortMappingInfo pmInfo;
      pmInfo.node = entry.nodeIp;
      pmInfo.rdmaNic = entry.rdmaNic;
      topology_->portMapping[entry.switchPort] = pmInfo;
    }
  }

  if (!entries.empty()) {
    INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: Updated topology from LLDP with %zu entries", entries.size());
    printLldpTopology(entries);
  }
}

/**
 * @brief 打印LLDP发现的拓扑信息表格
 * @param entries LLDP拓扑条目列表
 * 
 * 输出格式化的表格，展示交换机端口、服务器IP和RDMA NIC的映射关系
 */
void NetworkObserver::printLldpTopology(const std::vector<LldpTopologyEntry>& entries) {
  INFO(NCCL_NET, "NET_OBSERV: +----------------------------------------+");
  INFO(NCCL_NET, "NET_OBSERV: | %-38s |", "LLDP Discovered Topology");
  INFO(NCCL_NET, "NET_OBSERV: +------------------+---------------------+");
  INFO(NCCL_NET, "NET_OBSERV: | %-16s | %-17s |", "Switch Port", "Node IP");
  INFO(NCCL_NET, "NET_OBSERV: +------------------+---------------------+");

  for (const auto& entry : entries) {
    INFO(NCCL_NET, "NET_OBSERV: | %-16s | %-17s |",
         entry.switchPort.c_str(), entry.nodeIp.c_str());
    INFO(NCCL_NET, "NET_OBSERV: | %-16s | %-17s |",
         "", ("NIC: " + entry.rdmaNic).c_str());
  }

  INFO(NCCL_NET, "NET_OBSERV: +------------------+---------------------+");
}

/**
 * @brief 处理LLDP周期性订阅事件的主入口
 * @param event gRPC请求中的JSON事件
 *
 * 处理流程：
 * 1. 提取设备型号信息
 * 2. 解析JSON字符串中的LLDP邻居数据
 * 3. 更新网络拓扑映射
 *
 * 事件类型：json_event = 0x10080000 (GRPC_JSON_EVENT_SAMPLE_LLDP_INFO)
 */
void NetworkObserver::handleLldpEvent(const ruijie_json::JsonRequest& event) {
  std::string deviceModel = "unknown";
  if (event.has_device_info()) {
    deviceModel = event.device_info().device_model();
  }

  std::string jsonString = event.json_string();
  if (jsonString.empty()) {
    INFO(NCCL_NET, "NET_OBSERV: LLDP event has empty json_string, skipping");
    return;
  }

  auto lldpInfos = parseLldpJson(jsonString);
  if (lldpInfos.empty()) {
    INFO(NCCL_NET, "NET_OBSERV: No valid LLDP neighbors parsed from event");
    return;
  }

  INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: Received LLDP event (0x%08x) from %s, %zu neighbors",
       GRPC_JSON_EVENT_SAMPLE_LLDP_INFO, deviceModel.c_str(), lldpInfos.size());

  updateTopologyFromLldp(lldpInfos, deviceModel);
}

// ---- Global singleton management ----

static NetworkObserver* g_netObserver = nullptr;
static TopologyConfig* g_netObservTopology = nullptr;
static std::mutex g_netObservMutex;

/**
 * @brief 读取环境变量中的整数
 * @param envName 环境变量名（不含NCCL_前缀）
 * @param defaultValue 默认值
 * @return 解析后的整数值，解析失败返回默认值
 */
static int readEnvInt(const char* envName, int defaultValue) {
  const char* val = ncclGetEnv(envName);
  if (!val) return defaultValue;
  char* end;
  long result = strtol(val, &end, 10);
  if (end == val || *end != '\0') return defaultValue;
  return static_cast<int>(result);
}

int ncclNetObservInit(void) {
  const char* enable = ncclGetEnv("NCCL_NET_OBSERV_ENABLE");
  if (!enable || strcmp(enable, "1") != 0) {
    return 0;  // Disabled, treated as success
  }

  std::lock_guard<std::mutex> lock(g_netObservMutex);
  if (g_netObserver) {
    return 0;  // Already initialized
  }

  int mode = readEnvInt("NCCL_NET_OBSERV_MODE", 0);

  double pollInterval = 1.0;
  const char* envPoll = ncclGetEnv("NCCL_NET_OBSERV_POLL_INTERVAL");
  if (envPoll) {
    char* end;
    double val = strtod(envPoll, &end);
    if (end != envPoll && val > 0) {
      pollInterval = val;
    }
  }

  if (mode == 0) {
    // Mode 0: LLDP mode - learn topology from switch events
    int grpcPort = readEnvInt("NCCL_NET_OBSERV_GRPC_PORT", 50051);

    g_netObservTopology = new TopologyConfig();
    g_netObservTopology->grpcPort = grpcPort;

    g_netObserver = new NetworkObserver(g_netObservTopology, /*rank*/0, pollInterval);
    int ret = g_netObserver->start();
    if (ret != 0) {
      INFO(NCCL_NET, "NET_OBSERV: Failed to start NetworkObserver");
      delete g_netObserver;
      g_netObserver = nullptr;
      delete g_netObservTopology;
      g_netObservTopology = nullptr;
      return ret;
    }

    INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: NetworkObserver started (LLDP mode, port=%d)", grpcPort);
  } else {
    // Mode 1: Config file mode
    const char* configPath = ncclGetEnv("NCCL_NET_OBSERV_CONFIG");
    if (!configPath) {
      INFO(NCCL_NET, "NET_OBSERV: NCCL_NET_OBSERV_ENABLE=1 but NCCL_NET_OBSERV_CONFIG not set, disabling");
      return 0;
    }

    struct stat st;
    if (stat(configPath, &st) != 0) {
      INFO(NCCL_NET, "NET_OBSERV: Topology config %s not found, disabling", configPath);
      return 0;
    }

    g_netObservTopology = new TopologyConfig(TopologyConfig::loadFromFile(configPath));

    g_netObserver = new NetworkObserver(g_netObservTopology, /*rank*/0, pollInterval);
    int ret = g_netObserver->start();
    if (ret != 0) {
      INFO(NCCL_NET, "NET_OBSERV: Failed to start NetworkObserver");
      delete g_netObserver;
      g_netObserver = nullptr;
      delete g_netObservTopology;
      g_netObservTopology = nullptr;
      return ret;
    }

    INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: NetworkObserver started (config=%s)", configPath);
  }

  return 0;
}

void ncclNetObservFinalize(void) {
  std::lock_guard<std::mutex> lock(g_netObservMutex);
  if (g_netObserver) {
    g_netObserver->stop();
    delete g_netObserver;
    g_netObserver = nullptr;
  }
  if (g_netObservTopology) {
    delete g_netObservTopology;
    g_netObservTopology = nullptr;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: NetworkObserver finalized");
}

/**
 * @brief 更新rank拓扑映射（在ncclCommInitRank后调用）
 * @param nodeRanks 每个节点包含的rank列表，按node ID索引
 *
 * 该函数在NCCL通信域建立后被调用，用于更新拓扑中的rank分布信息。
 * 通过node ID（从LLDP事件中获取的节点顺序）匹配，而不是通过IP地址。
 */
void ncclNetObservUpdateRankTopology(const std::vector<std::vector<int>>& nodeRanks) {
  std::lock_guard<std::mutex> lock(g_netObservMutex);
  if (!g_netObservTopology) {
    INFO(NCCL_NET, "NET_OBSERV: Cannot update rank topology, NetworkObserver not initialized");
    return;
  }

  // 收集topology中所有唯一的节点IP，并按出现顺序分配node ID
  std::vector<std::string> uniqueNodes;
  for (const auto& entry : g_netObservTopology->portMapping) {
    const std::string& nodeIp = entry.second.node;
    // 检查是否已存在
    bool found = false;
    for (const auto& existing : uniqueNodes) {
      if (existing == nodeIp) {
        found = true;
        break;
      }
    }
    if (!found) {
      uniqueNodes.push_back(nodeIp);
    }
  }

  // 按node ID顺序更新每个节点的ranks
  for (size_t nodeId = 0; nodeId < nodeRanks.size() && nodeId < uniqueNodes.size(); nodeId++) {
    const std::string& nodeIp = uniqueNodes[nodeId];
    const std::vector<int>& ranks = nodeRanks[nodeId];

    // 更新该节点IP对应的所有端口映射
    for (auto& entry : g_netObservTopology->portMapping) {
      if (entry.second.node == nodeIp) {
        entry.second.ranks = ranks;
        INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: Updated port %s with %zu ranks for node %s",
             entry.first.c_str(), ranks.size(), nodeIp.c_str());
      }
    }
  }

  // 打印更新后的拓扑信息
  INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV: Rank topology updated for %zu nodes", nodeRanks.size());
  for (size_t i = 0; i < nodeRanks.size() && i < uniqueNodes.size(); i++) {
    std::string ranksStr;
    if (nodeRanks[i].size() <= 4) {
      for (size_t j = 0; j < nodeRanks[i].size(); j++) {
        if (j > 0) ranksStr += ",";
        ranksStr += std::to_string(nodeRanks[i][j]);
      }
    } else {
      ranksStr = std::to_string(nodeRanks[i].front()) + "-" + std::to_string(nodeRanks[i].back());
    }
    INFO(NCCL_INIT|NCCL_NET, "NET_OBSERV:   Node %s: ranks [%s]", uniqueNodes[i].c_str(), ranksStr.c_str());
  }
}

/**
 * @brief NetworkObserver成员方法：处理IB传输层错误
 * @param rdmaNic RDMA网卡设备名
 * @param peerIp 对端节点IP地址（可选）
 * @param wcStatus IB工作完成状态码
 *
 * 该函数在IB传输层检测到CQE错误时被调用，无需等待NCCL异常。
 * 相比confirmFromException的文本解析方式，这种方式更直接、更及时。
 */
void NetworkObserver::handleIbError(const std::string& rdmaNic, const std::string& peerIp, int wcStatus) {
  INFO(NCCL_NET, "NET_OBSERV: Direct IB error from %s, peer=%s, wc_status=%d",
       rdmaNic.c_str(), peerIp.c_str(), wcStatus);

  std::lock_guard<std::mutex> lock(incidentsMutex_);

  // 查找匹配的未确认告警
  std::vector<Incident*> matched;
  for (auto& entry : activeIncidents_) {
    if (entry.second.confirmed) continue;
    if (entry.second.rdmaNic == rdmaNic) {
      matched.push_back(&entry.second);
    }
  }

  // 如果没有精确匹配，匹配所有未确认的告警
  if (matched.empty()) {
    for (auto& entry : activeIncidents_) {
      if (!entry.second.confirmed) {
        matched.push_back(&entry.second);
      }
    }
  }

  // 更新并确认告警
  for (auto* incident : matched) {
    // 获取最新的NIC计数器
    std::pair<bool, std::map<std::string, int64_t>> ret =
        nicReader_.checkNicRetrans(incident->rdmaNic);
    if (!ret.second.empty()) {
      incident->nicDelta = ret.second;
    }

    // 如果提供了peerIp，更新故障路径
    if (!peerIp.empty()) {
      std::string peerPort = topology_->getPortByNodeAndNic(peerIp, incident->rdmaNic);
      if (!peerPort.empty()) {
        incident->faultPaths = buildFaultPath(
            {incident->portName}, incident->deviceModel, peerPort);
      } else {
        for (const auto& pmEntry : topology_->portMapping) {
          if (pmEntry.second.node == peerIp) {
            incident->faultPaths = buildFaultPath(
                {incident->portName}, incident->deviceModel, pmEntry.first);
            break;
          }
        }
      }
    }

    // 确认告警
    incident->confirmed = true;
    incident->confirmTime = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // 构建错误信息
    char errorBuf[256];
    snprintf(errorBuf, sizeof(errorBuf),
             "IB CQE error: dev=%s, wc_status=%d, peer=%s",
             rdmaNic.c_str(), wcStatus, peerIp.c_str());
    incident->ncclError = errorBuf;

    // 发送确认告警
    emitConfirmedAlert(*incident);

    INFO(NCCL_NET, "NET_OBSERV: Confirmed incident %d from direct IB error",
         incident->incidentId);
  }
}

/**
 * @brief C接口：处理IB传输层错误（从p2p.cc直接调用）
 * @param rdmaNic RDMA网卡设备名
 * @param peerIp 对端节点IP地址（可选）
 * @param wcStatus IB工作完成状态码
 */
void ncclNetObservHandleIbError(const char* rdmaNic, const char* peerIp, int wcStatus) {
  if (!rdmaNic || !g_netObserver) {
    return;
  }

  std::string nicName(rdmaNic);
  std::string peerIpStr(peerIp ? peerIp : "");

  std::lock_guard<std::mutex> lock(g_netObservMutex);
  if (!g_netObservTopology) {
    return;
  }

  g_netObserver->handleIbError(nicName, peerIpStr, wcStatus);
}

}  // namespace net_observ
