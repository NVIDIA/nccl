/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "tuner.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define __hidden __attribute__ ((visibility("hidden")))
#define MAX_LINE_LENGTH 256

// CSV field indices for configuration parsing
// Format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
#define CONFIG_FIELD_COLLTYPE     0
#define CONFIG_FIELD_MINBYTES     1
#define CONFIG_FIELD_MAXBYTES     2
#define CONFIG_FIELD_ALGORITHM    3
#define CONFIG_FIELD_PROTOCOL     4
#define CONFIG_FIELD_CHANNELS     5
#define CONFIG_FIELD_NNODES       6
#define CONFIG_FIELD_NRANKS       7
#define CONFIG_FIELD_PIPEOPS      8  // Optional field
#define CONFIG_FIELD_REGBUFF      9  // Optional field

// Field count constants
#define CONFIG_FIELDS_REQUIRED    8   // Minimum required fields (up to nRanks)
#define CONFIG_FIELDS_WITH_PIPEOPS 9  // Fields including numPipeOps
#define CONFIG_FIELDS_WITH_REGBUFF 10 // Fields including both numPipeOps and regBuff
#define CONFIG_FIELDS_MAX         10  // Maximum number of fields supported

typedef struct {
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
} TuningConfig;

typedef struct {
  TuningConfig* configs;  // Changed from static array to dynamic pointer
  int numConfigs;
  int maxConfigs;         // Added to track allocated size
  size_t nRanks;
  size_t nNodes;
  ncclDebugLogger_t logFunction;
  ncclNvlDomainInfo_v5_t nvlDomainInfo;
} TunerContext;

// Parse collective type from string
static ncclFunc_t parseCollType(const char* str) {
  if (strcmp(str, "broadcast") == 0) return ncclFuncBroadcast;
  if (strcmp(str, "reduce") == 0) return ncclFuncReduce;
  if (strcmp(str, "allgather") == 0) return ncclFuncAllGather;
  if (strcmp(str, "reducescatter") == 0) return ncclFuncReduceScatter;
  if (strcmp(str, "allreduce") == 0) return ncclFuncAllReduce;
  return ncclFuncAllReduce; // default
}

// Convert collective type to string
static const char* collTypeToString(ncclFunc_t collType) {
  switch (collType) {
    case ncclFuncBroadcast: return "broadcast";
    case ncclFuncReduce: return "reduce";
    case ncclFuncAllGather: return "allgather";
    case ncclFuncReduceScatter: return "reducescatter";
    case ncclFuncAllReduce: return "allreduce";
    default: return "unknown";
  }
}

// Parse algorithm from string
static int parseAlgorithm(const char* str) {
  if (strcmp(str, "tree") == 0) return NCCL_ALGO_TREE;
  if (strcmp(str, "ring") == 0) return NCCL_ALGO_RING;
  if (strcmp(str, "collnet_direct") == 0) return NCCL_ALGO_COLLNET_DIRECT;
  if (strcmp(str, "collnet_chain") == 0) return NCCL_ALGO_COLLNET_CHAIN;
  if (strcmp(str, "nvls") == 0) return NCCL_ALGO_NVLS;
  if (strcmp(str, "nvls_tree") == 0) return NCCL_ALGO_NVLS_TREE;
  if (strcmp(str, "pat") == 0) return NCCL_ALGO_PAT;
  return NCCL_ALGO_RING; // default
}

// Convert algorithm to string
static const char* algorithmToString(int algorithm) {
  switch (algorithm) {
    case NCCL_ALGO_TREE: return "tree";
    case NCCL_ALGO_RING: return "ring";
    case NCCL_ALGO_COLLNET_DIRECT: return "collnet_direct";
    case NCCL_ALGO_COLLNET_CHAIN: return "collnet_chain";
    case NCCL_ALGO_NVLS: return "nvls";
    case NCCL_ALGO_NVLS_TREE: return "nvls_tree";
    case NCCL_ALGO_PAT: return "pat";
    default: return "unknown";
  }
}

// Parse protocol from string
static int parseProtocol(const char* str) {
  if (strcmp(str, "ll") == 0) return NCCL_PROTO_LL;
  if (strcmp(str, "ll128") == 0) return NCCL_PROTO_LL128;
  if (strcmp(str, "simple") == 0) return NCCL_PROTO_SIMPLE;
  return NCCL_PROTO_SIMPLE; // default
}

// Convert protocol to string
static const char* protocolToString(int protocol) {
  switch (protocol) {
    case NCCL_PROTO_LL: return "ll";
    case NCCL_PROTO_LL128: return "ll128";
    case NCCL_PROTO_SIMPLE: return "simple";
    default: return "unknown";
  }
}

// Helper function to count valid configuration lines in file
static int countConfigLines(const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    return 0;
  }

  char line[MAX_LINE_LENGTH];
  int count = 0;

  while (fgets(line, sizeof(line), file)) {
    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\n') continue;

    // Remove trailing newline
    line[strcspn(line, "\n")] = 0;

    // Check if line has content
    if (strlen(line) > 0) {
      count++;
    }
  }

  fclose(file);
  return count;
}

// Load configuration from file
static ncclResult_t loadConfig(TunerContext* ctx, const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Config file %s not found, using defaults", filename);
    }
    return ncclSuccess; // Not finding config file is not an error
  }

  // First pass: count valid configuration lines
  int configCount = countConfigLines(filename);
  if (configCount == 0) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: No valid configurations found in %s", filename);
    }
    fclose(file);
    return ncclSuccess;
  }

  // Allocate memory for configurations based on actual count
  ctx->configs = (TuningConfig*)malloc(configCount * sizeof(TuningConfig));
  if (!ctx->configs) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Failed to allocate memory for %d configurations", configCount);
    }
    fclose(file);
    return ncclSystemError;
  }

  ctx->maxConfigs = configCount;
  ctx->numConfigs = 0;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: Allocated memory for %d configurations", configCount);
  }

  // Reset file pointer to beginning
  fseek(file, 0, SEEK_SET);

  char line[MAX_LINE_LENGTH];

  while (fgets(line, sizeof(line), file) && ctx->numConfigs < ctx->maxConfigs) {
    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\n') continue;

    // Remove trailing newline
    line[strcspn(line, "\n")] = 0;

    // Parse CSV format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
    char* token;
    char* tokens[CONFIG_FIELDS_MAX];
    int tokenCount = 0;

    // Make a copy of the line for tokenizing
    char lineCopy[MAX_LINE_LENGTH];
    strncpy(lineCopy, line, sizeof(lineCopy));
    lineCopy[sizeof(lineCopy) - 1] = '\0';

    // Tokenize by comma
    token = strtok(lineCopy, ",");
    while (token != NULL && tokenCount < CONFIG_FIELDS_MAX) {
      // Trim whitespace
      while (*token == ' ' || *token == '\t') token++;
      char* end = token + strlen(token) - 1;
      while (end > token && (*end == ' ' || *end == '\t')) {
        *end = '\0';
        end--;
      }
      tokens[tokenCount++] = token;
      token = strtok(NULL, ",");
    }

    // Validate field count: support required fields (8), with pipeOps (9), or with regBuff (10)
    if (tokenCount >= CONFIG_FIELDS_REQUIRED && tokenCount <= CONFIG_FIELDS_MAX) {
      TuningConfig* config = &ctx->configs[ctx->numConfigs];
      config->collType = parseCollType(tokens[CONFIG_FIELD_COLLTYPE]);
      config->minBytes = (size_t)strtoull(tokens[CONFIG_FIELD_MINBYTES], NULL, 10);
      config->maxBytes = (size_t)strtoull(tokens[CONFIG_FIELD_MAXBYTES], NULL, 10);
      config->algorithm = parseAlgorithm(tokens[CONFIG_FIELD_ALGORITHM]);
      config->protocol = parseProtocol(tokens[CONFIG_FIELD_PROTOCOL]);
      config->nChannels = atoi(tokens[CONFIG_FIELD_CHANNELS]);
      config->nNodes = atoi(tokens[CONFIG_FIELD_NNODES]);
      config->nRanks = atoi(tokens[CONFIG_FIELD_NRANKS]);

      // numPipeOps is optional (9th field, index 8)
      if (tokenCount >= CONFIG_FIELDS_WITH_PIPEOPS) {
        config->numPipeOps = atoi(tokens[CONFIG_FIELD_PIPEOPS]);
      } else {
        config->numPipeOps = -1; // -1 means match any numPipeOps
      }

      // regBuff is optional (10th field, index 9)
      if (tokenCount >= CONFIG_FIELDS_WITH_REGBUFF) {
        config->regBuff = atoi(tokens[CONFIG_FIELD_REGBUFF]);
      } else {
        config->regBuff = -1; // -1 means match any regBuff value
      }

      ctx->numConfigs++;

      if (ctx->logFunction) {
        if (config->numPipeOps == -1 && config->regBuff == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=any regBuff=any",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks);
        } else if (config->regBuff == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=%d regBuff=any",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->numPipeOps);
        } else if (config->numPipeOps == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=any regBuff=%d",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->regBuff);
        } else {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=%d regBuff=%d",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->numPipeOps, config->regBuff);
        }
      }
    }
  }

  fclose(file);
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: Loaded %d tuning configurations from %s", ctx->numConfigs, filename);
  }
  return ncclSuccess;
}

__hidden ncclResult_t pluginInit(void** context, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction,
                                 ncclNvlDomainInfo_v5_t* nvlDomainInfo, ncclTunerConstants_v5_t* constants) {

  if (NULL != constants) {
    // NCCL constants tuning
    // Tune NCCL's internal tuning model to improve base algo/proto selection.
    // Note: Example numbers are for reference only.
    //       Actual numbers may vary depending on the hardware and network topology.
    //       These numbers are not guaranteed to be optimal for all cases.
    // Limit the tree bandwidth to 15GB/s
    constants->perChMaxTreeBws[NCCL_BLACKWELL_COMPCAP_IDX][NCCL_TUNING_SCALE_4NODES] = 15.0;

    // Limit the ring bandwidth to 20GB/s
    constants->perChMaxRingLL128Bws[NCCL_BLACKWELL_COMPCAP_IDX][NCCL_TUNING_SCALE_4NODES] = 20.0;

    // Set NVLSTree base network latency to 24us
    constants->hwLatencies[NCCL_HW_NET][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = 24.0;
  }

  TunerContext* ctx = (TunerContext*)malloc(sizeof(TunerContext));
  if (!ctx) return ncclSystemError;

  ctx->configs = NULL;     // Initialize to NULL
  ctx->numConfigs = 0;
  ctx->maxConfigs = 0;     // Initialize to 0
  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  ctx->logFunction = logFunction;
  if (nvlDomainInfo) {
    ctx->nvlDomainInfo = *nvlDomainInfo;
  } else {
    memset(&ctx->nvlDomainInfo, 0, sizeof(ncclNvlDomainInfo_v5_t));
  }

  if (logFunction) {
    logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "TUNER/ExamplePlugin: Initializing tuner for %zu nodes, %zu ranks, %d NVL domains",
                nNodes, nRanks, ctx->nvlDomainInfo.nNvlDomains);
  }

  // Try to load config file from environment variable or default location
  const char* configFile = getenv("NCCL_TUNER_CONFIG_FILE");
  if (!configFile) {
    configFile = "nccl_tuner.conf"; // default config file name
  }

  ncclResult_t result = loadConfig(ctx, configFile);
  if (result != ncclSuccess) {
    if (ctx->configs) {
      free(ctx->configs);  // Clean up allocated memory on error
    }
    free(ctx);
    return result;
  }

  *context = ctx;
  return ncclSuccess;
}

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels) {
  TunerContext* ctx = (TunerContext*)context;
  if (!ctx) return ncclInternalError;

  // Default channels
  *nChannels = 1;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: pluginGetCollInfo called - collType=%s, nBytes=%zu, numPipeOps=%d, regBuff=%d, numConfigs=%d",
                     collTypeToString(collType), nBytes, numPipeOps, regBuff, ctx->numConfigs);
  }

  // Look for matching configuration
  for (int i = 0; i < ctx->numConfigs; i++) {
    TuningConfig* config = &ctx->configs[i];

    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Checking config %d - collType=%s, minBytes=%zu, maxBytes=%zu, algo=%s, proto=%s, nNodes=%d, nRanks=%d, numPipeOps=%d, regBuff=%d",
                       i, collTypeToString(config->collType), config->minBytes, config->maxBytes, algorithmToString(config->algorithm), protocolToString(config->protocol),
                       config->nNodes, config->nRanks, config->numPipeOps, config->regBuff);
    }

    // Check if this config matches the current collective, size range, topology, pipeline ops, and regBuff
    if (config->collType == collType &&
        nBytes >= config->minBytes &&
        nBytes <= config->maxBytes &&
        (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes) &&
        (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks) &&
        (config->numPipeOps == -1 || config->numPipeOps == numPipeOps) &&
        (config->regBuff == -1 || config->regBuff == regBuff)) {

      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/ExamplePlugin: Config matches. Applying algo=%s, proto=%s, channels=%d",
                         algorithmToString(config->algorithm), protocolToString(config->protocol), config->nChannels);
      }

      // Check bounds
      if (config->algorithm < numAlgo && config->protocol < numProto) {
        if (collCostTable[config->algorithm][config->protocol] != NCCL_ALGO_PROTO_IGNORE) {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Setting cost table[%s][%s] (%p) = 0.0 (was %.1f)",
                             algorithmToString(config->algorithm), protocolToString(config->protocol),
                             &collCostTable[config->algorithm][config->protocol], collCostTable[config->algorithm][config->protocol]);
          }
          collCostTable[config->algorithm][config->protocol] = 0.0; // Set low cost to prefer this configuration

          // Only override channels if not set to -1 (keep default)
          if (config->nChannels != -1) {
            *nChannels = config->nChannels;
          }

          if (ctx->logFunction) {
            if (config->nChannels == -1) {
              ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                               "TUNER/ExamplePlugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=default (nodes=%d, ranks=%d)",
                               collTypeToString(config->collType), nBytes, numPipeOps, regBuff, algorithmToString(config->algorithm), protocolToString(config->protocol),
                               config->nNodes, config->nRanks);
            } else {
              ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                               "TUNER/ExamplePlugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=%d (nodes=%d, ranks=%d)",
                               collTypeToString(config->collType), nBytes, numPipeOps, regBuff, algorithmToString(config->algorithm), protocolToString(config->protocol),
                               config->nChannels, config->nNodes, config->nRanks);
            }
          }
          return ncclSuccess;
        } else {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Algorithm/protocol combination [%s][%s] is marked as IGNORE",
                             algorithmToString(config->algorithm), protocolToString(config->protocol));
          }
        }
      } else {
        if (ctx->logFunction) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Algorithm/protocol out of bounds - algo=%s (max %d), proto=%s (max %d)",
                           algorithmToString(config->algorithm), numAlgo, protocolToString(config->protocol), numProto);
        }
      }
    } else {
      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/ExamplePlugin: Config does not match - collType match=%d, size match=%d, nodes match=%d, ranks match=%d, pipeOps match=%d, regBuff match=%d",
                         config->collType == collType,
                         (nBytes >= config->minBytes && nBytes <= config->maxBytes),
                         (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes),
                         (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks),
                         (config->numPipeOps == -1 || config->numPipeOps == numPipeOps),
                         (config->regBuff == -1 || config->regBuff == regBuff));
      }
    }
  }

  // If no specific config found, apply default behavior
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: No matching config found");
  }

  return ncclSuccess;
}

__hidden ncclResult_t pluginFinalize(void* context) {
  if (context) {
    TunerContext* ctx = (TunerContext*)context;
    if (ctx->configs) {
      free(ctx->configs);  // Free dynamically allocated configs array
    }
    free(context);
  }
  return ncclSuccess;
}


#define PLUGIN_NAME "Example"

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .finalize = pluginFinalize
};
