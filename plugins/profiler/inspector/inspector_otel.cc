/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "inspector_otel.h"
#include "inspector_ring.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>

#define INSPECTOR_OTEL_DEFAULT_ENDPOINT "http://localhost:4318/v1/metrics"
#define INSPECTOR_OTEL_DEFAULT_TIMEOUT_MS 2000
#define INSPECTOR_OTEL_MIN_TIMEOUT_MS 1
#define INSPECTOR_OTEL_BODY_BYTES_PER_POINT 512
#define INSPECTOR_OTEL_DEFAULT_FORMAT_MINOR 1
#define INSPECTOR_OTEL_VERBOSE_FORMAT_MINOR 2

extern const char* ncclFuncToString(ncclFunc_t fn);

static bool gOtelEnabled = false;
static bool gOtelVerboseEnabled = false;
static std::string gOtelEndpoint;
static int gOtelTimeoutMs = INSPECTOR_OTEL_DEFAULT_TIMEOUT_MS;
static std::vector<inspectorOtelAttribute> gOtelHeaders;
static std::vector<inspectorOtelAttribute> gOtelResourceAttributes;
static uint64_t gOtelExportFailures = 0;
static uint64_t gOtelDroppedPoints = 0;
static std::string gOtelNodeName = "unknown";

struct inspectorOtelUrl {
  std::string host;
  std::string port;
  std::string path;
  std::string hostHeader;
};

struct inspectorOtelBucketAgg {
  uint64_t count = 0;
  uint64_t latestTimestampUsec = 0;
  double execTimeSum = 0.0;
  double algoLogSum = 0.0;
  uint64_t algoCount = 0;
  double busLogSum = 0.0;
  uint64_t busCount = 0;
};

struct inspectorOtelCollBucketKey {
  int nranks;
  int nnodes;
  ncclFunc_t func;
  size_t msgSizeRangeBytes;
  std::string commName;
  std::string algoProto;

  bool operator<(const inspectorOtelCollBucketKey& other) const {
    if (nranks != other.nranks) return nranks < other.nranks;
    if (nnodes != other.nnodes) return nnodes < other.nnodes;
    if (func != other.func) return func < other.func;
    if (msgSizeRangeBytes != other.msgSizeRangeBytes) {
      return msgSizeRangeBytes < other.msgSizeRangeBytes;
    }
    if (commName != other.commName) return commName < other.commName;
    return algoProto < other.algoProto;
  }
};

struct inspectorOtelP2pBucketKey {
  int nranks;
  int nnodes;
  ncclFunc_t func;
  size_t msgSizeRangeBytes;
  std::string commName;

  bool operator<(const inspectorOtelP2pBucketKey& other) const {
    if (nranks != other.nranks) return nranks < other.nranks;
    if (nnodes != other.nnodes) return nnodes < other.nnodes;
    if (func != other.func) return func < other.func;
    if (msgSizeRangeBytes != other.msgSizeRangeBytes) {
      return msgSizeRangeBytes < other.msgSizeRangeBytes;
    }
    return commName < other.commName;
  }
};

using inspectorOtelCollBucketMap = std::map<inspectorOtelCollBucketKey,
                                            inspectorOtelBucketAgg>;
using inspectorOtelP2pBucketMap = std::map<inspectorOtelP2pBucketKey,
                                           inspectorOtelBucketAgg>;

struct inspectorOtelDevice {
  std::string deviceUuidStr;
  std::string gpuName;
  inspectorOtelCollBucketMap collBuckets;
  inspectorOtelP2pBucketMap p2pBuckets;
  bool hasData = false;
};

static const char* inspectorOtelFirstEnv(const char* name0,
                                         const char* name1,
                                         const char* name2) {
  const char* value = getenv(name0);
  if (value && value[0]) return value;
  value = name1 ? getenv(name1) : nullptr;
  if (value && value[0]) return value;
  value = name2 ? getenv(name2) : nullptr;
  if (value && value[0]) return value;
  return nullptr;
}

static std::string inspectorOtelTrim(const std::string& value) {
  size_t first = 0;
  while (first < value.size() && (value[first] == ' ' || value[first] == '\t')) {
    first++;
  }
  size_t last = value.size();
  while (last > first && (value[last - 1] == ' ' || value[last - 1] == '\t')) {
    last--;
  }
  return value.substr(first, last - first);
}

static bool inspectorOtelHasAttribute(
    const std::vector<inspectorOtelAttribute>& attrs,
    const char* key) {
  for (size_t i = 0; i < attrs.size(); i++) {
    if (attrs[i].key == key) return true;
  }
  return false;
}

static void inspectorOtelAddDefaultResourceAttribute(const char* key,
                                                     const std::string& value) {
  if (!inspectorOtelHasAttribute(gOtelResourceAttributes, key)) {
    gOtelResourceAttributes.push_back(inspectorOtelAttribute{key, value});
  }
}

static void inspectorOtelSetResourceAttribute(const char* key,
                                              const std::string& value) {
  for (size_t i = 0; i < gOtelResourceAttributes.size(); i++) {
    if (gOtelResourceAttributes[i].key == key) {
      gOtelResourceAttributes[i].value = value;
      return;
    }
  }
  gOtelResourceAttributes.push_back(inspectorOtelAttribute{key, value});
}

static void inspectorOtelParseKeyValueList(
    const char* input,
    std::vector<inspectorOtelAttribute>* output) {
  if (input == nullptr || output == nullptr) return;

  const char* cursor = input;
  while (*cursor != '\0') {
    const char* comma = strchr(cursor, ',');
    std::string item = comma ? std::string(cursor, comma - cursor)
                             : std::string(cursor);
    size_t equals = item.find('=');
    if (equals != std::string::npos) {
      std::string key = inspectorOtelTrim(item.substr(0, equals));
      std::string value = inspectorOtelTrim(item.substr(equals + 1));
      if (!key.empty()) {
        output->push_back(inspectorOtelAttribute{key, value});
      }
    }
    if (!comma) break;
    cursor = comma + 1;
  }
}

static void inspectorOtelAggUpdate(inspectorOtelBucketAgg& agg,
                                   uint64_t timestampUsec,
                                   double algoBwGbs,
                                   double busBwGbs,
                                   uint64_t execTimeUsecs) {
  agg.count++;
  if (timestampUsec > agg.latestTimestampUsec) {
    agg.latestTimestampUsec = timestampUsec;
  }
  agg.execTimeSum += static_cast<double>(execTimeUsecs);
  if (algoBwGbs > 0.0) {
    agg.algoLogSum += std::log(algoBwGbs);
    agg.algoCount++;
  }
  if (busBwGbs > 0.0) {
    agg.busLogSum += std::log(busBwGbs);
    agg.busCount++;
  }
}

static void inspectorOtelFormatMessageSizeRange(size_t bytes,
                                                char* output,
                                                size_t outputSize) {
  if (bytes == 0) {
    snprintf(output, outputSize, "0B");
    return;
  }

  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  size_t unitIndex = 0;
  double byteValue = static_cast<double>(bytes);

  while (byteValue >= 1024.0 && unitIndex < 4) {
    byteValue /= 1024.0;
    unitIndex++;
  }

  double lower = std::floor(byteValue);
  if (lower < 1.0) {
    lower = 1.0;
  }
  double upper = lower + 1.0;

  snprintf(output, outputSize, "%.0f-%.0f%s", lower, upper, units[unitIndex]);
}

static size_t inspectorOtelMessageSizeRangeLowerBound(size_t bytes) {
  if (bytes == 0) {
    return 0;
  }

  size_t unitIndex = 0;
  double value = static_cast<double>(bytes);
  while (value >= 1024.0 && unitIndex < 4) {
    value /= 1024.0;
    unitIndex++;
  }

  double lower = std::floor(value);
  if (lower < 1.0) {
    lower = 1.0;
  }

  size_t unitBytes = static_cast<size_t>(std::pow(1024.0, unitIndex));
  return static_cast<size_t>(lower) * unitBytes;
}

static const char* inspectorOtelCommName(const struct inspectorCommInfo* commInfo) {
  return (commInfo != nullptr && commInfo->commName && commInfo->commName[0])
      ? commInfo->commName : "unknown";
}

static const char* inspectorOtelCommId(const struct inspectorCommInfo* commInfo) {
  return (commInfo != nullptr && commInfo->commHashStr[0])
      ? commInfo->commHashStr : "unknown";
}

static void inspectorOtelAddAttr(std::vector<inspectorOtelAttribute>& attrs,
                                 const char* key,
                                 const std::string& value) {
  attrs.push_back(inspectorOtelAttribute{key, value});
}

static std::string inspectorOtelFormatVersion(int minor) {
  char version[16];
  snprintf(version, sizeof(version),
           "v%d.%d",
           NCCL_PROFILER_INTERFACE_VERSION,
           minor);
  return version;
}

static void inspectorOtelAddCommonAttr(std::vector<inspectorOtelAttribute>& attrs,
                                       const inspectorOtelDevice& device,
                                       const char* commName,
                                       const char* commId,
                                       int nranks,
                                       int nnodes,
                                       const char* msgSizeAttr,
                                       size_t msgSizeBytes,
                                       bool includeCommId,
                                       int formatMinor) {
  inspectorOtelAddAttr(attrs, "version",
                       inspectorOtelFormatVersion(formatMinor));
  inspectorOtelAddAttr(attrs, "node", gOtelNodeName);
  inspectorOtelAddAttr(attrs, "gpu",
                       device.gpuName.empty() ? "unknown" : device.gpuName);
  inspectorOtelAddAttr(attrs, "comm_name",
                       (commName && commName[0]) ? commName : "unknown");
  if (includeCommId) {
    inspectorOtelAddAttr(attrs, "comm_id",
                         (commId && commId[0]) ? commId : "unknown");
  }
  inspectorOtelAddAttr(attrs, "n_nodes", std::to_string(nnodes));
  inspectorOtelAddAttr(attrs, "nranks", std::to_string(nranks));
  inspectorOtelAddAttr(attrs,
                       msgSizeAttr ? msgSizeAttr : "msg_size_bytes",
                       std::to_string(msgSizeBytes));
}

static void inspectorOtelAddBucketAttr(std::vector<inspectorOtelAttribute>& attrs,
                                       const inspectorOtelDevice& device,
                                       const char* commName,
                                       int nranks,
                                       int nnodes,
                                       size_t msgSizeRangeBytes) {
  char msgSizeStr[32];
  inspectorOtelFormatMessageSizeRange(msgSizeRangeBytes,
                                      msgSizeStr,
                                      sizeof(msgSizeStr));

  inspectorOtelAddAttr(attrs, "gpu",
                       device.gpuName.empty() ? "unknown" : device.gpuName);
  inspectorOtelAddAttr(attrs, "comm_name",
                       (commName && commName[0]) ? commName : "unknown");
  inspectorOtelAddAttr(attrs, "version",
                       inspectorOtelFormatVersion(INSPECTOR_OTEL_DEFAULT_FORMAT_MINOR));
  inspectorOtelAddAttr(attrs, "node", gOtelNodeName);
  inspectorOtelAddAttr(attrs, "n_nodes", std::to_string(nnodes));
  inspectorOtelAddAttr(attrs, "nranks", std::to_string(nranks));
  inspectorOtelAddAttr(attrs, "message_size", msgSizeStr);
}

static void inspectorOtelAddMetricPoint(
    std::vector<inspectorOtelMetricPoint>& points,
    const char* name,
    const char* description,
    const char* unit,
    const std::vector<inspectorOtelAttribute>& attrs,
    double value,
    uint64_t timestampUsec) {
  if (!std::isfinite(value)) {
    return;
  }

  inspectorOtelMetricPoint point;
  point.name = name;
  point.description = description;
  point.unit = unit;
  point.attributes = attrs;
  point.value = value;
  point.timestampUsec = timestampUsec;
  points.push_back(point);
}

static inspectorResult_t inspectorOtelAddCompletedColl(
    std::vector<inspectorOtelMetricPoint>& points,
    const inspectorOtelDevice& device,
    const struct inspectorCommInfo* commInfo,
    const inspectorCompletedOpInfo& collInfo,
    const std::string& algoProto) {
  std::vector<inspectorOtelAttribute> attrs;
  inspectorOtelAddCommonAttr(attrs,
                             device,
                             inspectorOtelCommName(commInfo),
                             inspectorOtelCommId(commInfo),
                             commInfo->nranks,
                             commInfo->nnodes,
                             "coll_msg_size_bytes",
                             collInfo.msgSizeBytes,
                             true,
                             INSPECTOR_OTEL_VERBOSE_FORMAT_MINOR);
  inspectorOtelAddAttr(attrs, "collective", ncclFuncToString(collInfo.func));
  inspectorOtelAddAttr(attrs, "coll_sn", std::to_string(collInfo.sn));
  inspectorOtelAddAttr(attrs, "algo_proto", algoProto);

  inspectorOtelAddMetricPoint(points,
                              "nccl_bus_bandwidth_gbs",
                              "NCCL bus bandwidth for collective operations",
                              "GB/s",
                              attrs,
                              collInfo.busBwGbs,
                              collInfo.timestampUsec);
  inspectorOtelAddMetricPoint(points,
                              "nccl_collective_exec_time_microseconds",
                              "NCCL collective execution time",
                              "us",
                              attrs,
                              static_cast<double>(collInfo.execTimeUsecs),
                              collInfo.timestampUsec);
  inspectorOtelAddMetricPoint(points,
                              "nccl_collective_algobw_gbs",
                              "NCCL algorithm bandwidth for collective operations",
                              "GB/s",
                              attrs,
                              collInfo.algoBwGbs,
                              collInfo.timestampUsec);
  return inspectorSuccess;
}

static inspectorResult_t inspectorOtelAddCompletedP2p(
    std::vector<inspectorOtelMetricPoint>& points,
    const inspectorOtelDevice& device,
    const struct inspectorCommInfo* commInfo,
    const inspectorCompletedOpInfo& p2pInfo) {
  std::vector<inspectorOtelAttribute> attrs;
  inspectorOtelAddCommonAttr(attrs,
                             device,
                             inspectorOtelCommName(commInfo),
                             inspectorOtelCommId(commInfo),
                             commInfo->nranks,
                             commInfo->nnodes,
                             "p2p_msg_size_bytes",
                             p2pInfo.msgSizeBytes,
                             true,
                             INSPECTOR_OTEL_VERBOSE_FORMAT_MINOR);
  inspectorOtelAddAttr(attrs, "p2p_operation", ncclFuncToString(p2pInfo.func));
  inspectorOtelAddAttr(attrs, "p2p_sn", std::to_string(p2pInfo.sn));
  inspectorOtelAddAttr(attrs, "p2p_peer", std::to_string(p2pInfo.peer));

  inspectorOtelAddMetricPoint(points,
                              "nccl_p2p_bus_bandwidth_gbs",
                              "NCCL bus bandwidth for P2P operations",
                              "GB/s",
                              attrs,
                              p2pInfo.busBwGbs,
                              p2pInfo.timestampUsec);
  inspectorOtelAddMetricPoint(points,
                              "nccl_p2p_exec_time_microseconds",
                              "NCCL P2P execution time",
                              "us",
                              attrs,
                              static_cast<double>(p2pInfo.execTimeUsecs),
                              p2pInfo.timestampUsec);
  inspectorOtelAddMetricPoint(points,
                              "nccl_p2p_algobw_gbs",
                              "NCCL algorithm bandwidth for P2P operations",
                              "GB/s",
                              attrs,
                              p2pInfo.algoBwGbs,
                              p2pInfo.timestampUsec);
  return inspectorSuccess;
}

static inspectorResult_t inspectorOtelAddCollBucket(
    std::vector<inspectorOtelMetricPoint>& points,
    const inspectorOtelDevice& device,
    const inspectorOtelCollBucketKey& key,
    const inspectorOtelBucketAgg& agg) {
  std::vector<inspectorOtelAttribute> attrs;
  inspectorOtelAddBucketAttr(attrs,
                             device,
                             key.commName.c_str(),
                             key.nranks,
                             key.nnodes,
                             key.msgSizeRangeBytes);
  inspectorOtelAddAttr(attrs, "collective", ncclFuncToString(key.func));
  inspectorOtelAddAttr(attrs, "algo_proto", key.algoProto);

  double execMean = agg.count ? (agg.execTimeSum / agg.count) : 0.0;
  double busMean = agg.busCount ? std::exp(agg.busLogSum / agg.busCount) : 0.0;

  inspectorOtelAddMetricPoint(points,
                              "nccl_bus_bandwidth_gbs",
                              "NCCL bus bandwidth for collective operations",
                              "GB/s",
                              attrs,
                              busMean,
                              agg.latestTimestampUsec);
  inspectorOtelAddMetricPoint(points,
                              "nccl_collective_exec_time_microseconds",
                              "NCCL collective execution time",
                              "us",
                              attrs,
                              execMean,
                              agg.latestTimestampUsec);
  return inspectorSuccess;
}

static inspectorResult_t inspectorOtelAddP2pBucket(
    std::vector<inspectorOtelMetricPoint>& points,
    const inspectorOtelDevice& device,
    const inspectorOtelP2pBucketKey& key,
    const inspectorOtelBucketAgg& agg) {
  std::vector<inspectorOtelAttribute> attrs;
  inspectorOtelAddBucketAttr(attrs,
                             device,
                             key.commName.c_str(),
                             key.nranks,
                             key.nnodes,
                             key.msgSizeRangeBytes);
  inspectorOtelAddAttr(attrs, "p2p_operation", ncclFuncToString(key.func));

  double execMean = agg.count ? (agg.execTimeSum / agg.count) : 0.0;
  double busMean = agg.busCount ? std::exp(agg.busLogSum / agg.busCount) : 0.0;

  inspectorOtelAddMetricPoint(points,
                              "nccl_p2p_bus_bandwidth_gbs",
                              "NCCL bus bandwidth for P2P operations",
                              "GB/s",
                              attrs,
                              busMean,
                              agg.latestTimestampUsec);
  inspectorOtelAddMetricPoint(points,
                              "nccl_p2p_exec_time_microseconds",
                              "NCCL P2P execution time",
                              "us",
                              attrs,
                              execMean,
                              agg.latestTimestampUsec);
  return inspectorSuccess;
}

static inspectorResult_t inspectorOtelBuildAggregatedPoints(
    const std::map<std::string, inspectorOtelDevice>& devices,
    std::vector<inspectorOtelMetricPoint>& points) {
  for (const auto& entry : devices) {
    const inspectorOtelDevice& device = entry.second;
    if (!device.hasData) {
      continue;
    }
    for (const auto& collEntry : device.collBuckets) {
      INS_CHK(inspectorOtelAddCollBucket(points,
                                         device,
                                         collEntry.first,
                                         collEntry.second));
    }
    for (const auto& p2pEntry : device.p2pBuckets) {
      INS_CHK(inspectorOtelAddP2pBucket(points,
                                        device,
                                        p2pEntry.first,
                                        p2pEntry.second));
    }
  }
  return inspectorSuccess;
}

static bool inspectorOtelUrlHasPath(const std::string& endpoint) {
  size_t scheme = endpoint.find("://");
  size_t hostStart = scheme == std::string::npos ? 0 : scheme + 3;
  size_t pathStart = endpoint.find('/', hostStart);
  return pathStart != std::string::npos && pathStart + 1 < endpoint.size();
}

static std::string inspectorOtelEnsureMetricsPath(const std::string& endpoint) {
  if (inspectorOtelUrlHasPath(endpoint)) {
    return endpoint;
  }
  if (!endpoint.empty() && endpoint[endpoint.size() - 1] == '/') {
    return endpoint + "v1/metrics";
  }
  return endpoint + "/v1/metrics";
}

static bool inspectorOtelParseHttpUrl(const std::string& endpoint,
                                      inspectorOtelUrl* url) {
  const char* prefix = "http://";
  const size_t prefixLen = strlen(prefix);
  if (endpoint.compare(0, prefixLen, prefix) != 0) {
    INFO_INSPECTOR("NCCL Inspector OTEL: only http:// endpoints are supported: %s",
                   endpoint.c_str());
    return false;
  }

  size_t cursor = prefixLen;
  size_t hostEnd = endpoint.find_first_of("/?", cursor);
  std::string authority = endpoint.substr(
      cursor,
      hostEnd == std::string::npos ? std::string::npos : hostEnd - cursor);
  if (authority.empty()) return false;

  std::string host;
  std::string port = "80";
  if (authority[0] == '[') {
    size_t bracket = authority.find(']');
    if (bracket == std::string::npos) return false;
    host = authority.substr(1, bracket - 1);
    if (bracket + 1 < authority.size()) {
      if (authority[bracket + 1] != ':') return false;
      port = authority.substr(bracket + 2);
    }
  } else {
    size_t colon = authority.rfind(':');
    if (colon != std::string::npos) {
      host = authority.substr(0, colon);
      port = authority.substr(colon + 1);
    } else {
      host = authority;
    }
  }

  if (host.empty() || port.empty()) return false;

  std::string path = "/";
  if (hostEnd != std::string::npos) {
    path = endpoint.substr(hostEnd);
    if (!path.empty() && path[0] == '?') {
      path = "/" + path;
    }
  }

  url->host = host;
  url->port = port;
  url->path = path;
  if (host.find(':') != std::string::npos) {
    url->hostHeader = "[" + host + "]:" + port;
  } else {
    url->hostHeader = host + ":" + port;
  }
  return true;
}

static void inspectorOtelAppendJsonString(std::string* out,
                                          const std::string& value) {
  out->push_back('"');
  for (size_t i = 0; i < value.size(); i++) {
    unsigned char c = static_cast<unsigned char>(value[i]);
    switch (c) {
    case '"': *out += "\\\""; break;
    case '\\': *out += "\\\\"; break;
    case '\b': *out += "\\b"; break;
    case '\f': *out += "\\f"; break;
    case '\n': *out += "\\n"; break;
    case '\r': *out += "\\r"; break;
    case '\t': *out += "\\t"; break;
    default:
      if (c < 0x20) {
        char encoded[7];
        snprintf(encoded, sizeof(encoded), "\\u%04x", c);
        *out += encoded;
      } else {
        out->push_back(static_cast<char>(c));
      }
    }
  }
  out->push_back('"');
}

static void inspectorOtelAppendAttributes(
    std::string* out,
    const std::vector<inspectorOtelAttribute>& attrs) {
  *out += "\"attributes\":[";
  for (size_t i = 0; i < attrs.size(); i++) {
    if (i != 0) *out += ",";
    *out += "{\"key\":";
    inspectorOtelAppendJsonString(out, attrs[i].key);
    *out += ",\"value\":{\"stringValue\":";
    inspectorOtelAppendJsonString(out, attrs[i].value);
    *out += "}}";
  }
  *out += "]";
}

static std::string inspectorOtelDoubleToString(double value) {
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

static void inspectorOtelAppendMetricPoint(std::string* out,
                                           const inspectorOtelMetricPoint& point,
                                           uint64_t timestampNano) {
  uint64_t pointTimestampNano = point.timestampUsec
      ? point.timestampUsec * 1000 : timestampNano;
  *out += "{";
  inspectorOtelAppendAttributes(out, point.attributes);
  *out += ",\"timeUnixNano\":\"";
  *out += std::to_string(pointTimestampNano);
  *out += "\",\"asDouble\":";
  *out += inspectorOtelDoubleToString(point.value);
  *out += "}";
}

static void inspectorOtelAppendBody(
    std::string* out,
    const std::vector<inspectorOtelMetricPoint>& points,
    uint64_t timestampNano) {
  std::map<std::string, std::vector<const inspectorOtelMetricPoint*> > byMetric;
  for (size_t i = 0; i < points.size(); i++) {
    if (std::isfinite(points[i].value)) {
      byMetric[points[i].name].push_back(&points[i]);
    }
  }

  *out += "{\"resourceMetrics\":[{\"resource\":{";
  inspectorOtelAppendAttributes(out, gOtelResourceAttributes);
  *out += "},\"scopeMetrics\":[{\"scope\":{\"name\":\"nccl-inspector\"},\"metrics\":[";

  bool firstMetric = true;
  for (std::map<std::string, std::vector<const inspectorOtelMetricPoint*> >::const_iterator it
       = byMetric.begin(); it != byMetric.end(); ++it) {
    if (it->second.empty()) continue;
    if (!firstMetric) *out += ",";
    firstMetric = false;

    const inspectorOtelMetricPoint* firstPoint = it->second[0];
    *out += "{\"name\":";
    inspectorOtelAppendJsonString(out, firstPoint->name);
    *out += ",\"description\":";
    inspectorOtelAppendJsonString(out, firstPoint->description);
    *out += ",\"unit\":";
    inspectorOtelAppendJsonString(out, firstPoint->unit);
    *out += ",\"gauge\":{\"dataPoints\":[";
    for (size_t i = 0; i < it->second.size(); i++) {
      if (i != 0) *out += ",";
      inspectorOtelAppendMetricPoint(out, *it->second[i], timestampNano);
    }
    *out += "]}}";
  }

  *out += "]}]}]}";
}

static void inspectorOtelSetSocketTimeouts(int fd, int timeoutMs) {
  struct timeval timeout;
  timeout.tv_sec = timeoutMs / 1000;
  timeout.tv_usec = (timeoutMs % 1000) * 1000;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

static bool inspectorOtelConnectWithTimeout(int fd,
                                            const struct sockaddr* addr,
                                            socklen_t addrLen,
                                            int timeoutMs) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) return false;
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) return false;

  int ret = connect(fd, addr, addrLen);
  if (ret == 0) {
    fcntl(fd, F_SETFL, flags);
    return true;
  }
  if (errno != EINPROGRESS) {
    fcntl(fd, F_SETFL, flags);
    return false;
  }

  fd_set writeFds;
  FD_ZERO(&writeFds);
  FD_SET(fd, &writeFds);
  struct timeval timeout;
  timeout.tv_sec = timeoutMs / 1000;
  timeout.tv_usec = (timeoutMs % 1000) * 1000;
  ret = select(fd + 1, nullptr, &writeFds, nullptr, &timeout);
  if (ret <= 0) {
    fcntl(fd, F_SETFL, flags);
    return false;
  }

  int err = 0;
  socklen_t errLen = sizeof(err);
  if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &errLen) != 0 || err != 0) {
    fcntl(fd, F_SETFL, flags);
    return false;
  }

  fcntl(fd, F_SETFL, flags);
  return true;
}

static int inspectorOtelOpenSocket(const inspectorOtelUrl& url) {
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo* result = nullptr;
  int ret = getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &result);
  if (ret != 0) return -1;

  int fd = -1;
  for (struct addrinfo* itr = result; itr != nullptr; itr = itr->ai_next) {
    fd = socket(itr->ai_family, itr->ai_socktype, itr->ai_protocol);
    if (fd < 0) continue;
    inspectorOtelSetSocketTimeouts(fd, gOtelTimeoutMs);
    if (inspectorOtelConnectWithTimeout(fd,
                                        itr->ai_addr,
                                        itr->ai_addrlen,
                                        gOtelTimeoutMs)) {
      break;
    }
    close(fd);
    fd = -1;
  }

  freeaddrinfo(result);
  return fd;
}

static bool inspectorOtelSendAll(int fd, const std::string& request) {
  const char* data = request.data();
  size_t remaining = request.size();
  while (remaining > 0) {
#ifdef MSG_NOSIGNAL
    ssize_t sent = send(fd, data, remaining, MSG_NOSIGNAL);
#else
    ssize_t sent = send(fd, data, remaining, 0);
#endif
    if (sent < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (sent == 0) return false;
    data += sent;
    remaining -= sent;
  }
  return true;
}

static bool inspectorOtelParseStatus(int fd, int* statusCode) {
  char response[512];
  ssize_t got = recv(fd, response, sizeof(response) - 1, 0);
  if (got <= 0) return false;
  response[got] = '\0';

  char* space = strchr(response, ' ');
  if (space == nullptr) return false;
  *statusCode = atoi(space + 1);
  return *statusCode >= 200 && *statusCode < 300;
}

static bool inspectorOtelHeaderIsSafe(const inspectorOtelAttribute& header) {
  return header.key.find('\r') == std::string::npos
      && header.key.find('\n') == std::string::npos
      && header.value.find('\r') == std::string::npos
      && header.value.find('\n') == std::string::npos
      && strcasecmp(header.key.c_str(), "content-length") != 0
      && strcasecmp(header.key.c_str(), "content-type") != 0
      && strcasecmp(header.key.c_str(), "host") != 0;
}

static bool inspectorOtelPostJson(const std::string& body, int* statusCode) {
  inspectorOtelUrl url;
  if (!inspectorOtelParseHttpUrl(gOtelEndpoint, &url)) return false;

  std::string request;
  request.reserve(body.size() + 1024);
  request += "POST ";
  request += url.path;
  request += " HTTP/1.1\r\nHost: ";
  request += url.hostHeader;
  request += "\r\nContent-Type: application/json\r\nAccept: application/json\r\n";
  request += "Connection: close\r\nContent-Length: ";
  request += std::to_string(body.size());
  request += "\r\n";
  for (size_t i = 0; i < gOtelHeaders.size(); i++) {
    if (!inspectorOtelHeaderIsSafe(gOtelHeaders[i])) continue;
    request += gOtelHeaders[i].key;
    request += ": ";
    request += gOtelHeaders[i].value;
    request += "\r\n";
  }
  request += "\r\n";
  request += body;

  int fd = inspectorOtelOpenSocket(url);
  if (fd < 0) return false;

  bool ok = inspectorOtelSendAll(fd, request)
      && inspectorOtelParseStatus(fd, statusCode);
  close(fd);
  return ok;
}

static void inspectorOtelLogExportFailure(int statusCode, size_t droppedPoints) {
  gOtelExportFailures++;
  gOtelDroppedPoints += droppedPoints;
  if (gOtelExportFailures <= 3 || (gOtelExportFailures % 10) == 0) {
    if (statusCode > 0) {
      INFO_INSPECTOR(
        "NCCL Inspector OTEL: export to %s failed with HTTP %d "
        "(dropped %zu points this export, %lu total dropped, %lu failures)",
        gOtelEndpoint.c_str(), statusCode, droppedPoints,
        (unsigned long)gOtelDroppedPoints, (unsigned long)gOtelExportFailures);
    } else {
      INFO_INSPECTOR(
        "NCCL Inspector OTEL: export to %s failed "
        "(dropped %zu points this export, %lu total dropped, %lu failures)",
        gOtelEndpoint.c_str(), droppedPoints,
        (unsigned long)gOtelDroppedPoints, (unsigned long)gOtelExportFailures);
    }
  }
}

static inline bool inspectorOtelInitEndpoint() {
  const char* endpoint = inspectorOtelFirstEnv(
      "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
      "OTEL_EXPORTER_OTLP_ENDPOINT",
      nullptr);
  gOtelEndpoint = endpoint && endpoint[0]
      ? inspectorOtelEnsureMetricsPath(endpoint)
      : INSPECTOR_OTEL_DEFAULT_ENDPOINT;

  // Validate once at init so unsupported schemes never drain rings for OTLP.
  inspectorOtelUrl url;
  if (!inspectorOtelParseHttpUrl(gOtelEndpoint, &url)) {
    VERSION(
      "NCCL Inspector OTEL DISABLED: only http:// endpoints are supported "
      "(got %s). Set OTEL_EXPORTER_OTLP_METRICS_ENDPOINT to an http:// URL.",
      gOtelEndpoint.c_str());
    INFO_INSPECTOR(
      "NCCL Inspector OTEL DISABLED: only http:// endpoints are supported "
      "(got %s)",
      gOtelEndpoint.c_str());
    return false;
  }
  return true;
}

static inline void inspectorOtelInitTimeout() {
  gOtelTimeoutMs = INSPECTOR_OTEL_DEFAULT_TIMEOUT_MS;

  const char* timeout = inspectorOtelFirstEnv(
      "OTEL_EXPORTER_OTLP_METRICS_TIMEOUT",
      "OTEL_EXPORTER_OTLP_TIMEOUT",
      nullptr);
  if (timeout && timeout[0]) {
    int parsed = atoi(timeout);
    if (parsed >= INSPECTOR_OTEL_MIN_TIMEOUT_MS) {
      gOtelTimeoutMs = parsed;
    } else {
      INFO_INSPECTOR(
        "NCCL Inspector OTEL: ignoring timeout %s; expected positive milliseconds",
        timeout);
    }
  }
}

static inline void inspectorOtelInitHeaders() {
  const char* headers = inspectorOtelFirstEnv(
      "OTEL_EXPORTER_OTLP_METRICS_HEADERS",
      "OTEL_EXPORTER_OTLP_HEADERS",
      nullptr);
  inspectorOtelParseKeyValueList(headers, &gOtelHeaders);
}

static inline void inspectorOtelInitResourceAttributes() {
  const char* resourceAttrs = getenv("OTEL_RESOURCE_ATTRIBUTES");
  inspectorOtelParseKeyValueList(resourceAttrs, &gOtelResourceAttributes);

  const char* serviceName = getenv("OTEL_SERVICE_NAME");
  if (serviceName && serviceName[0]) {
    inspectorOtelSetResourceAttribute("service.name", serviceName);
  } else {
    inspectorOtelAddDefaultResourceAttribute("service.name", "nccl-inspector");
  }

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname) - 1) == 0) {
    hostname[sizeof(hostname) - 1] = '\0';
    gOtelNodeName = hostname;
  }
  const char* slurmJobId = getenv("SLURM_JOB_ID");
  if (slurmJobId && slurmJobId[0]) {
    inspectorOtelAddDefaultResourceAttribute("slurm.job.id", slurmJobId);
  }
}

void inspectorOtelInitFromEnv() {
  gOtelEnabled = false;
  gOtelVerboseEnabled = false;
  gOtelHeaders.clear();
  gOtelResourceAttributes.clear();
  gOtelEndpoint.clear();
  gOtelExportFailures = 0;
  gOtelDroppedPoints = 0;
  gOtelNodeName = "unknown";

  const char* enabled = getenv("NCCL_INSPECTOR_OTEL_EXPORT");
  gOtelEnabled = enabled ? atoi(enabled) != 0 : false;
  if (!gOtelEnabled) return;
  const char* verbose = getenv("NCCL_INSPECTOR_OTEL_VERBOSE");
  gOtelVerboseEnabled = verbose ? atoi(verbose) != 0 : false;

  if (!inspectorOtelInitEndpoint()) {
    gOtelEnabled = false;
    gOtelVerboseEnabled = false;
    return;
  }
  inspectorOtelInitTimeout();
  inspectorOtelInitHeaders();
  inspectorOtelInitResourceAttributes();

  INFO_INSPECTOR("NCCL Inspector OTEL: exporting %s metrics to %s",
                 gOtelVerboseEnabled ? "verbose per-operation" : "aggregated",
                 gOtelEndpoint.c_str());
}

bool inspectorOtelIsEnabled() {
  return gOtelEnabled;
}

bool inspectorOtelVerboseEnabled() {
  return gOtelEnabled && gOtelVerboseEnabled;
}

const char* inspectorOtelEndpoint() {
  return gOtelEndpoint.c_str();
}

inspectorResult_t inspectorOtelExportMetrics(
    const std::vector<inspectorOtelMetricPoint>& points,
    uint64_t timestampUsec) {
  if (!gOtelEnabled || points.empty()) {
    return inspectorSuccess;
  }

  std::string body;
  body.reserve(points.size() * INSPECTOR_OTEL_BODY_BYTES_PER_POINT);
  inspectorOtelAppendBody(&body, points, timestampUsec * 1000);

  int statusCode = 0;
  if (!inspectorOtelPostJson(body, &statusCode)) {
    inspectorOtelLogExportFailure(statusCode, points.size());
  }
  return inspectorSuccess;
}

static inspectorResult_t inspectorOtelCommInfoDumpColl(
    struct inspectorCommInfo* commInfo,
    inspectorOtelDevice& device,
    std::vector<inspectorOtelMetricPoint>& points,
    bool verbose,
    bool* needsWriting) {
  if (commInfo == nullptr) {
    return inspectorSuccess;
  }

  thread_local std::vector<inspectorCompletedOpInfo> drainedColl;
  drainedColl.clear();

  inspectorLockWr(&commInfo->guard);
  if (commInfo->dump_coll) {
    if (commInfo->completedCollRing.size > 0
        && drainedColl.capacity() < commInfo->completedCollRing.size) {
      drainedColl.reserve(commInfo->completedCollRing.size);
    }
    INS_CHK(inspectorRingDrain<inspectorCompletedOpInfo>(&commInfo->completedCollRing,
                                                        drainedColl));
    commInfo->dump_coll = inspectorRingNonEmpty(&commInfo->completedCollRing);
  }
  inspectorUnlockRWLock(&commInfo->guard);

  if (!drainedColl.empty()) {
    *needsWriting = true;
    for (size_t i = 0; i < drainedColl.size(); i++) {
      const inspectorCompletedOpInfo& collInfo = drainedColl[i];
      const char* algo = (collInfo.algo && collInfo.algo[0]) ? collInfo.algo : "unknown";
      const char* proto = (collInfo.proto && collInfo.proto[0]) ? collInfo.proto : "unknown";
      std::string algoProto = std::string(algo) + "_" + proto;

      if (verbose) {
        INS_CHK(inspectorOtelAddCompletedColl(points,
                                             device,
                                             commInfo,
                                             collInfo,
                                             algoProto));
      } else {
        inspectorOtelCollBucketKey key {
          commInfo->nranks,
          commInfo->nnodes,
          collInfo.func,
          inspectorOtelMessageSizeRangeLowerBound(collInfo.msgSizeBytes),
          inspectorOtelCommName(commInfo),
          algoProto
        };
        inspectorOtelAggUpdate(device.collBuckets[key],
                               collInfo.timestampUsec,
                               collInfo.algoBwGbs,
                               collInfo.busBwGbs,
                               collInfo.execTimeUsecs);
      }
    }
  }

  return inspectorSuccess;
}

static inspectorResult_t inspectorOtelCommInfoDumpP2p(
    struct inspectorCommInfo* commInfo,
    inspectorOtelDevice& device,
    std::vector<inspectorOtelMetricPoint>& points,
    bool verbose,
    bool* needsWriting) {
  if (commInfo == nullptr) {
    return inspectorSuccess;
  }

  thread_local std::vector<inspectorCompletedOpInfo> drainedP2p;
  drainedP2p.clear();

  inspectorLockWr(&commInfo->guard);
  if (commInfo->dump_p2p) {
    if (commInfo->completedP2pRing.size > 0
        && drainedP2p.capacity() < commInfo->completedP2pRing.size) {
      drainedP2p.reserve(commInfo->completedP2pRing.size);
    }
    INS_CHK(inspectorRingDrain<inspectorCompletedOpInfo>(&commInfo->completedP2pRing,
                                                        drainedP2p));
    commInfo->dump_p2p = inspectorRingNonEmpty(&commInfo->completedP2pRing);
  }
  inspectorUnlockRWLock(&commInfo->guard);

  if (!drainedP2p.empty()) {
    *needsWriting = true;
    for (size_t i = 0; i < drainedP2p.size(); i++) {
      const inspectorCompletedOpInfo& p2pInfo = drainedP2p[i];
      if (verbose) {
        INS_CHK(inspectorOtelAddCompletedP2p(points,
                                            device,
                                            commInfo,
                                            p2pInfo));
      } else {
        inspectorOtelP2pBucketKey key {
          commInfo->nranks,
          commInfo->nnodes,
          p2pInfo.func,
          inspectorOtelMessageSizeRangeLowerBound(p2pInfo.msgSizeBytes),
          inspectorOtelCommName(commInfo)
        };
        inspectorOtelAggUpdate(device.p2pBuckets[key],
                               p2pInfo.timestampUsec,
                               p2pInfo.algoBwGbs,
                               p2pInfo.busBwGbs,
                               p2pInfo.execTimeUsecs);
      }
    }
  }

  return inspectorSuccess;
}

static inspectorResult_t inspectorOtelCommInfoDump(
    struct inspectorCommInfo* commInfo,
    inspectorOtelDevice& device,
    std::vector<inspectorOtelMetricPoint>& points,
    bool verbose,
    bool* needsWriting) {
  *needsWriting = false;

  INS_CHK(inspectorOtelCommInfoDumpColl(commInfo,
                                        device,
                                        points,
                                        verbose,
                                        needsWriting));
  INS_CHK(inspectorOtelCommInfoDumpP2p(commInfo,
                                       device,
                                       points,
                                       verbose,
                                       needsWriting));

  return inspectorSuccess;
}

static inspectorResult_t inspectorOtelFillDeviceBuckets(
    struct inspectorCommInfoList* commList,
    std::map<std::string, inspectorOtelDevice>& devices,
    std::vector<inspectorOtelMetricPoint>& points,
    bool verbose,
    uint32_t* processedOut) {
  if (processedOut == nullptr) {
    return inspectorMemoryError;
  }
  *processedOut = 0;

  for (struct inspectorCommInfo* itr = commList->comms;
       itr != nullptr;
       itr = itr->next) {
    bool needsWriting;

    std::string deviceKey(itr->deviceUuidStr);
    inspectorOtelDevice& device = devices[deviceKey];
    if (device.deviceUuidStr.empty()) {
      device.deviceUuidStr = deviceKey;
    }
    if (device.gpuName.empty()) {
      char gpuName[16];
      snprintf(gpuName, sizeof(gpuName), "GPU%d", itr->cudaDeviceId);
      device.gpuName = gpuName;
    }

    INS_CHK(inspectorOtelCommInfoDump(itr,
                                      device,
                                      points,
                                      verbose,
                                      &needsWriting));

    if (needsWriting) {
      device.hasData = true;
      (*processedOut)++;
      TRACE_INSPECTOR(
        "NCCL Inspector OTEL: processed comm %u for CUDA device (rank %d)",
        *processedOut, itr->rank);
    }
  }

  return inspectorSuccess;
}

inspectorResult_t inspectorOtelCommInfoListDump(struct inspectorCommInfoList* commList) {
  INS_CHK(inspectorLockRd(&commList->guard));
  inspectorResult_t res = inspectorSuccess;
  std::vector<inspectorOtelMetricPoint> points;
  uint64_t currentTime = 0;
  uint32_t processed = 0;
  bool verbose = false;
  bool shouldExport = false;

  // Drain/build metric points under the comm-list lock, then release before
  // HTTP export so DNS/connect/send/recv never run while holding the lock.
  if (commList->ncomms > 0) {
    currentTime = inspectorGetTime();
    std::map<std::string, inspectorOtelDevice> devices;
    verbose = inspectorOtelVerboseEnabled();

    INS_CHK_GOTO(inspectorOtelFillDeviceBuckets(commList,
                                                devices,
                                                points,
                                                verbose,
                                                &processed),
                 res, unlock);
    if (!verbose) {
      INS_CHK_GOTO(inspectorOtelBuildAggregatedPoints(devices, points),
                   res, unlock);
    }
    shouldExport = true;
  }

unlock:
  INS_CHK(inspectorUnlockRWLock(&commList->guard));
  if (res != inspectorSuccess || !shouldExport) {
    return res;
  }

  INS_CHK(inspectorOtelExportMetrics(points, currentTime));
  if (processed > 0) {
    TRACE_INSPECTOR(
      "NCCL Inspector OTEL: exported %zu %s metric points",
      points.size(),
      verbose ? "verbose" : "aggregated");
  }
  return inspectorSuccess;
}
