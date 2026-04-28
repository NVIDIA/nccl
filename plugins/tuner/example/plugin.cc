/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "tuner.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#define __hidden __attribute__ ((visibility("hidden")))

// CSV field indices for configuration parsing
// Format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
enum ConfigField {
  kCollType = 0, kMinBytes, kMaxBytes, kAlgorithm, kProtocol,
  kChannels, kNNodes, kNRanks, kPipeOps, kRegBuff,
};
static constexpr int kFieldsRequired = 8;
static constexpr int kFieldsMax = 10;

struct TuningConfig {
  ncclFunc_t collType;
  size_t minBytes;
  size_t maxBytes;
  int algorithm;
  int protocol;
  int nChannels;
  int nNodes;
  int nRanks;
  int numPipeOps;
  int regBuff;
};

struct TunerContext {
  std::vector<TuningConfig> configs;
  size_t nRanks;
  size_t nNodes;
  ncclDebugLogger_t logFunction;
  ncclNvlDomainInfo_v5_t nvlDomainInfo;
};

// --- Lookup tables: replace if-else chains for name/value mapping ---

struct NamedCollType { const char* name; ncclFunc_t value; };
static const NamedCollType collTypeTable[] = {
  {"broadcast",     ncclFuncBroadcast},
  {"reduce",        ncclFuncReduce},
  {"allgather",     ncclFuncAllGather},
  {"reducescatter", ncclFuncReduceScatter},
  {"allreduce",     ncclFuncAllReduce},
};

struct NamedAlgorithm { const char* name; int value; };
static const NamedAlgorithm algorithmTable[] = {
  {"tree",           NCCL_ALGO_TREE},
  {"ring",           NCCL_ALGO_RING},
  {"collnet_direct", NCCL_ALGO_COLLNET_DIRECT},
  {"collnet_chain",  NCCL_ALGO_COLLNET_CHAIN},
  {"nvls",           NCCL_ALGO_NVLS},
  {"nvls_tree",      NCCL_ALGO_NVLS_TREE},
  {"pat",            NCCL_ALGO_PAT},
};

struct NamedProtocol { const char* name; int value; };
static const NamedProtocol protocolTable[] = {
  {"ll",     NCCL_PROTO_LL},
  {"ll128",  NCCL_PROTO_LL128},
  {"simple", NCCL_PROTO_SIMPLE},
};

static ncclFunc_t parseCollType(const std::string& str) {
  for (const auto& e : collTypeTable)
    if (str == e.name) return e.value;
  return ncclFuncAllReduce;
}

static const char* collTypeToString(ncclFunc_t collType) {
  for (const auto& e : collTypeTable)
    if (collType == e.value) return e.name;
  return "unknown";
}

static int parseAlgorithm(const std::string& str) {
  for (const auto& e : algorithmTable)
    if (str == e.name) return e.value;
  return NCCL_ALGO_RING;
}

static const char* algorithmToString(int algorithm) {
  for (const auto& e : algorithmTable)
    if (algorithm == e.value) return e.name;
  return "unknown";
}

static int parseProtocol(const std::string& str) {
  for (const auto& e : protocolTable)
    if (str == e.name) return e.value;
  return NCCL_PROTO_SIMPLE;
}

static const char* protocolToString(int protocol) {
  for (const auto& e : protocolTable)
    if (protocol == e.value) return e.name;
  return "unknown";
}

// --- String helpers ---

static std::string trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t");
  return s.substr(start, end - start + 1);
}

static std::vector<std::string> splitCSV(const std::string& line) {
  std::vector<std::string> fields;
  std::istringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(trim(field));
  }
  return fields;
}

// Format an optional int field: -1 returns the label, otherwise decimal
static const char* fmtOptional(int value, const char* label, char* buf, size_t bufLen) {
  if (value == -1) return label;
  snprintf(buf, bufLen, "%d", value);
  return buf;
}

// Load configuration from file (single-pass — std::vector grows as needed)
static ncclResult_t loadConfig(TunerContext* ctx, const char* filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Config file %s not found, using defaults", filename);
    }
    return ncclSuccess;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::vector<std::string> fields = splitCSV(line);
    int nFields = static_cast<int>(fields.size());
    if (nFields < kFieldsRequired || nFields > kFieldsMax) continue;

    TuningConfig config{};
    config.collType   = parseCollType(fields[kCollType]);
    config.minBytes   = strtoull(fields[kMinBytes].c_str(), nullptr, 10);
    config.maxBytes   = strtoull(fields[kMaxBytes].c_str(), nullptr, 10);
    config.algorithm  = parseAlgorithm(fields[kAlgorithm]);
    config.protocol   = parseProtocol(fields[kProtocol]);
    config.nChannels  = std::atoi(fields[kChannels].c_str());
    config.nNodes     = std::atoi(fields[kNNodes].c_str());
    config.nRanks     = std::atoi(fields[kNRanks].c_str());
    config.numPipeOps = (nFields > kPipeOps)  ? std::atoi(fields[kPipeOps].c_str()) : -1;
    config.regBuff    = (nFields > kRegBuff)   ? std::atoi(fields[kRegBuff].c_str()) : -1;

    ctx->configs.push_back(config);

    if (ctx->logFunction) {
      char poBuf[16], rbBuf[16];
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=%s regBuff=%s",
                       fields[kCollType].c_str(), config.minBytes, config.maxBytes,
                       fields[kAlgorithm].c_str(), fields[kProtocol].c_str(),
                       config.nChannels, config.nNodes, config.nRanks,
                       fmtOptional(config.numPipeOps, "any", poBuf, sizeof(poBuf)),
                       fmtOptional(config.regBuff, "any", rbBuf, sizeof(rbBuf)));
    }
  }

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: Loaded %d tuning configurations from %s",
                     static_cast<int>(ctx->configs.size()), filename);
  }
  return ncclSuccess;
}

__hidden ncclResult_t pluginInit(void** context, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction,
                                 ncclNvlDomainInfo_v5_t* nvlDomainInfo, ncclTunerConstants_v5_t* constants) {

  if (constants != nullptr) {
    // NCCL constants tuning
    // Note: Example numbers are for reference only.
    //       Actual numbers may vary depending on the hardware and network topology.
    //       These numbers are not guaranteed to be optimal for all cases.
    constants->perChMaxTreeBws[NCCL_BLACKWELL_COMPCAP_IDX][NCCL_TUNING_SCALE_4NODES] = 15.0;
    constants->perChMaxRingLL128Bws[NCCL_BLACKWELL_COMPCAP_IDX][NCCL_TUNING_SCALE_4NODES] = 20.0;
    constants->hwLatencies[NCCL_HW_NET][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = 24.0;
  }

  auto* ctx = new (std::nothrow) TunerContext{};
  if (!ctx) return ncclSystemError;

  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  ctx->logFunction = logFunction;
  ctx->nvlDomainInfo = nvlDomainInfo ? *nvlDomainInfo : ncclNvlDomainInfo_v5_t{};

  if (logFunction) {
    logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "TUNER/ExamplePlugin: Initializing tuner for %zu nodes, %zu ranks, %d NVL domains",
                nNodes, nRanks, ctx->nvlDomainInfo.nNvlDomains);
  }

  const char* configFile = std::getenv("NCCL_TUNER_CONFIG_FILE");
  if (!configFile) configFile = "nccl_tuner.conf";

  ncclResult_t result = loadConfig(ctx, configFile);
  if (result != ncclSuccess) {
    delete ctx;
    return result;
  }

  *context = ctx;
  return ncclSuccess;
}

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels) {
  auto* ctx = static_cast<TunerContext*>(context);
  if (!ctx) return ncclInternalError;
  auto* table = reinterpret_cast<float (*)[NCCL_NUM_PROTOCOLS]>(collCostTable);

  *nChannels = 1;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: pluginGetCollInfo called - collType=%s, nBytes=%zu, numPipeOps=%d, regBuff=%d, numConfigs=%d",
                     collTypeToString(collType), nBytes, numPipeOps, regBuff, static_cast<int>(ctx->configs.size()));
  }

  for (int i = 0; i < static_cast<int>(ctx->configs.size()); i++) {
    const auto& config = ctx->configs[i];

    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Checking config %d - collType=%s, minBytes=%zu, maxBytes=%zu, algo=%s, proto=%s, nNodes=%d, nRanks=%d, numPipeOps=%d, regBuff=%d",
                       i, collTypeToString(config.collType), config.minBytes, config.maxBytes, algorithmToString(config.algorithm), protocolToString(config.protocol),
                       config.nNodes, config.nRanks, config.numPipeOps, config.regBuff);
    }

    if (config.collType == collType &&
        nBytes >= config.minBytes &&
        nBytes <= config.maxBytes &&
        (config.nNodes == -1 || config.nNodes == static_cast<int>(ctx->nNodes)) &&
        (config.nRanks == -1 || config.nRanks == static_cast<int>(ctx->nRanks)) &&
        (config.numPipeOps == -1 || config.numPipeOps == numPipeOps) &&
        (config.regBuff == -1 || config.regBuff == regBuff)) {

      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/ExamplePlugin: Config matches. Applying algo=%s, proto=%s, channels=%d",
                         algorithmToString(config.algorithm), protocolToString(config.protocol), config.nChannels);
      }

      if (config.algorithm < numAlgo && config.protocol < numProto) {
        if (table[config.algorithm][config.protocol] != NCCL_ALGO_PROTO_IGNORE) {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Setting cost table[%s][%s] (%p) = 0.0 (was %.1f)",
                             algorithmToString(config.algorithm), protocolToString(config.protocol),
                             static_cast<void*>(&table[config.algorithm][config.protocol]), table[config.algorithm][config.protocol]);
          }
          table[config.algorithm][config.protocol] = 0.0;

          if (config.nChannels != -1) {
            *nChannels = config.nChannels;
          }

          if (ctx->logFunction) {
            char chBuf[16];
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=%s (nodes=%d, ranks=%d)",
                             collTypeToString(config.collType), nBytes, numPipeOps, regBuff, algorithmToString(config.algorithm), protocolToString(config.protocol),
                             fmtOptional(config.nChannels, "default", chBuf, sizeof(chBuf)),
                             config.nNodes, config.nRanks);
          }
          return ncclSuccess;
        } else {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Algorithm/protocol combination [%s][%s] is marked as IGNORE",
                             algorithmToString(config.algorithm), protocolToString(config.protocol));
          }
        }
      } else {
        if (ctx->logFunction) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Algorithm/protocol out of bounds - algo=%s (max %d), proto=%s (max %d)",
                           algorithmToString(config.algorithm), numAlgo, protocolToString(config.protocol), numProto);
        }
      }
    } else {
      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/ExamplePlugin: Config does not match - collType match=%d, size match=%d, nodes match=%d, ranks match=%d, pipeOps match=%d, regBuff match=%d",
                         config.collType == collType,
                         (nBytes >= config.minBytes && nBytes <= config.maxBytes),
                         (config.nNodes == -1 || config.nNodes == static_cast<int>(ctx->nNodes)),
                         (config.nRanks == -1 || config.nRanks == static_cast<int>(ctx->nRanks)),
                         (config.numPipeOps == -1 || config.numPipeOps == numPipeOps),
                         (config.regBuff == -1 || config.regBuff == regBuff));
      }
    }
  }

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: No matching config found");
  }

  return ncclSuccess;
}

__hidden ncclResult_t pluginGetChunkSize(void* context, ncclFunc_t collType, size_t nBytes,
                                         int algo, int proto, int nChannels, size_t* chunkSize) {
  auto* ctx = static_cast<TunerContext*>(context);
  if (!ctx) return ncclInternalError;

  size_t originalChunkSize = *chunkSize;
  size_t minChunkSize = 0;

  if (algo == NCCL_ALGO_NVLS_TREE && proto == NCCL_PROTO_SIMPLE) {
    minChunkSize = 32768;
  }

  if (minChunkSize > 0 && *chunkSize < minChunkSize) {
    *chunkSize = minChunkSize;
  }

  if (ctx->logFunction && *chunkSize != originalChunkSize) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: getChunkSize - collType=%s, nBytes=%zu, algo=%s, proto=%s, nChannels=%d: "
                     "chunk size %zu -> %zu",
                     collTypeToString(collType), nBytes, algorithmToString(algo), protocolToString(proto),
                     nChannels, originalChunkSize, *chunkSize);
  }

  return ncclSuccess;
}

__hidden ncclResult_t pluginFinalize(void* context) {
  delete static_cast<TunerContext*>(context);
  return ncclSuccess;
}


#define PLUGIN_NAME "Example"

// Exported symbols need C linkage for dlsym lookup
extern "C" {

const ncclTuner_v6_t ncclTunerPlugin_v6 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .finalize = pluginFinalize,
  .getChunkSize = pluginGetChunkSize
};

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .finalize = pluginFinalize
};

} // extern "C"
