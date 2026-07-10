/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Model Transfer Benchmark (C++)
 *
 * Benchmarks disaggregated model resharding using real model config and
 * system config JSON files.  This replaces synthetic tensor templates
 * with actual per-parameter shapes, dtypes, and placement rules so that
 * the resulting timings faithfully predict end-to-end model transfer
 * cost.
 *
 * Features:
 *   - Reads HF-style model config JSON  (param -> shape + dtype)
 *   - Reads system config JSON           (train/gen PP/TP/EP/DP)
 *   - Implements placement_rules.py in C++ (column/row-parallel,
 *     expert-parallel, vocab-parallel, replicate)
 *   - Groups per-expert 2D tensors into combined 3D expert tensors
 *   - Deduplicates across layers (keeps one representative per pattern)
 *   - PP-aware: one NCCL communicator per (train_stage, gen_stage) pair
 *   - User-window API: one ncclWindow_t per PP comm, pre-registered
 *   - Per-pattern and aggregate bandwidth / latency reporting
 *
 * Grid order (innermost to outermost): TP -> CP -> EP -> DP -> PP
 *
 * Usage:
 *   mpirun -np <worldSize> reshard_model_bench \
 *       --model-config benchmarks/configs/model_configs/dsv3-toy.model.json \
 *       --system-config benchmarks/configs/system_configs/dsv3-256gpus-gb200.json \
 *       [--iterations 10] [--warmup 2] [--verbose] [--validate]
 ************************************************************************/

#include "bench_common.h"
#include "bench_common_kernels.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "nccl_m2n.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ============================================================================
// Dtype Handling
// ============================================================================

static const std::map<std::string, size_t> DTYPE_ELEMENT_SIZE = {
  {"F32", 4}, {"F16", 2}, {"BF16", 2}, {"I64", 8},     {"I32", 4},
  {"I16", 2}, {"I8", 1},  {"U8", 1},   {"F8_E4M3", 1}, {"F8_E5M2", 1},
};

static size_t getElementSize(const std::string& dtype) {
  auto it = DTYPE_ELEMENT_SIZE.find(dtype);
  if (it != DTYPE_ELEMENT_SIZE.end()) return it->second;
  return 2; // default BF16
}

static const std::map<std::string, ncclDataType_t> DTYPE_TO_NCCL = {
  {"F32", ncclFloat32},  {"F16", ncclFloat16}, {"BF16", ncclBfloat16},
  {"I64", ncclInt64},    {"I32", ncclInt32},   {"I16", ncclFloat16}, // no ncclInt16; use same-size type
  {"I8", ncclInt8},      {"U8", ncclUint8},    {"F8_E4M3", ncclInt8}, // 1-byte type
  {"F8_E5M2", ncclInt8}, // 1-byte type
};

static ncclDataType_t getNcclDtype(const std::string& dtype) {
  auto it = DTYPE_TO_NCCL.find(dtype);
  if (it != DTYPE_TO_NCCL.end()) return it->second;
  return ncclBfloat16; // default
}

// ============================================================================
// Placement Rules  (port of src/placement_rules.py)
// ============================================================================

static const std::vector<std::string> COLUMN_PARALLEL_SUFFIXES = {
  "q_proj.weight",    "k_proj.weight",   "v_proj.weight",   "gate_proj.weight",
  "up_proj.weight",   "q_a_proj.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight",
  "kv_b_proj.weight",
};

static const std::vector<std::string> ROW_PARALLEL_SUFFIXES = {
  "o_proj.weight",
  "down_proj.weight",
};

static const std::vector<std::string> VOCAB_PARALLEL_NAMES = {
  "embed_tokens.weight",
  "lm_head.weight",
};

static bool endsWith(const std::string& s, const std::string& suffix) {
  if (suffix.size() > s.size()) return false;
  return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool contains(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

static bool isExpertParam(const std::string& name) {
  return contains(name, ".experts.");
}

// Returns -1 if replicated.
static int getTPShardDim(const std::string& name) {
  if (isExpertParam(name)) return -1;
  if (endsWith(name, "mlp.gate.weight") || contains(name, "e_score_correction_bias")) return -1;

  for (auto& s : COLUMN_PARALLEL_SUFFIXES)
    if (endsWith(name, s)) return 0;
  for (auto& s : ROW_PARALLEL_SUFFIXES)
    if (endsWith(name, s)) return 1;
  for (auto& s : VOCAB_PARALLEL_NAMES)
    if (contains(name, s)) return 0;

  return -1;
}

// For expert params, TP shard dim without the .experts. skip.
static int getExpertTPShardDim(const std::string& name) {
  for (auto& s : COLUMN_PARALLEL_SUFFIXES)
    if (endsWith(name, s)) return 0;
  for (auto& s : ROW_PARALLEL_SUFFIXES)
    if (endsWith(name, s)) return 1;
  return -1;
}

// ============================================================================
// Configuration Structures
// ============================================================================

struct ParallelConfig {
  int tp = 1;
  int cp = 1;
  int ep = 1;
  int dp = 1;
  int pp = 1;
  int numGpus = 0;

  int ranksPerStage() const { return tp * cp * ep * dp; }
  int totalRanks() const { return ranksPerStage() * pp; }
};

struct ParamInfo {
  std::string name;
  std::vector<size_t> shape;
  std::string dtype;
  size_t elementSize;
};

// Placement: array of per-mesh-dim shard dims.  -1 = Replicate.
struct Placement {
  std::vector<int> shardDims;
};

// ============================================================================
// Mesh Dimension Map
//
// Maps dimension names ("tp","ep","dp") to mesh axis indices.
// Excludes PP (which is handled by separate NCCL comms, not mesh axes).
// Active dims are reversed from the mapping string ordering so that the
// outermost in the name is axis 0 in the mesh (matching Python).
// ============================================================================

struct DimMap {
  std::map<std::string, int> nameToAxis;
  std::vector<int> meshShape;
  int numAxes() const { return (int)meshShape.size(); }
};

static DimMap buildDimMap(const ParallelConfig& cfg) {
  // Default mapping order: tp-cp-ep-dp-pp
  std::vector<std::string> dimNames = {"tp", "cp", "ep", "dp"};
  std::map<std::string, int> sizes = {
    {"tp", cfg.tp},
    {"cp", cfg.cp},
    {"ep", cfg.ep},
    {"dp", cfg.dp},
  };

  std::vector<std::pair<std::string, int>> active;
  for (auto& n : dimNames)
    if (sizes[n] > 1) active.push_back({n, sizes[n]});

  DimMap dm;
  // Reverse so outermost is axis 0 (matches Python's reversed())
  std::reverse(active.begin(), active.end());
  for (int i = 0; i < (int)active.size(); i++) {
    dm.nameToAxis[active[i].first] = i;
    dm.meshShape.push_back(active[i].second);
  }
  return dm;
}

// ============================================================================
// ETP=1 Two-Mesh Builder (Megatron Parallel Folding)
//
// Builds ETP=1 train meshes for Megatron-style parallel folding:
//   - Non-expert mesh: [DP_ne, TP(, CP)] where DP_ne = ranksPerStage / (TP*CP)
//   - Expert mesh:     [EDP,  EP]         where EDP  = ranksPerStage / EP
// Both meshes cover the same physical ranks with different logical layouts.
// ============================================================================

struct DimMapPair {
  DimMap nonExpert;
  DimMap expert;
};

static DimMapPair buildDimMapsETP1(const ParallelConfig& cfg) {
  // Derive per-stage rank count from num_gpus (not TP*CP*EP*DP which
  // would be wrong under Parallel Folding where EP overlaps TP*CP*DP).
  int perStage = cfg.numGpus / cfg.pp;

  ParallelConfig neCfg;
  neCfg.tp = cfg.tp;
  neCfg.cp = cfg.cp;
  neCfg.ep = 1;
  neCfg.dp = perStage / (cfg.tp * cfg.cp);
  neCfg.pp = 1;
  neCfg.numGpus = perStage;

  ParallelConfig exCfg;
  exCfg.tp = 1;
  exCfg.cp = 1;
  exCfg.ep = cfg.ep;
  exCfg.dp = perStage / cfg.ep;
  exCfg.pp = 1;
  exCfg.numGpus = perStage;

  return {buildDimMap(neCfg), buildDimMap(exCfg)};
}

// ============================================================================
// Placement Computation  (port of perf_test._get_placements)
// ============================================================================

static Placement getPlacements(const std::string& paramName, const DimMap& dm, int ndim) {
  int numAxes = dm.numAxes();
  if (numAxes == 0) numAxes = 1;
  Placement pl;
  pl.shardDims.assign(numAxes, -1);

  if (ndim < 2) return pl;

  if (isExpertParam(paramName)) {
    auto epIt = dm.nameToAxis.find("ep");
    if (epIt != dm.nameToAxis.end()) {
      // ETP=1: expert mesh has EP — shard experts only along EP dim.
      // TP acts as DP for experts (not in this mesh's dim_map).
      pl.shardDims[epIt->second] = 0;
    } else {
      // No EP axis (e.g. gen side): TP-shard the weight dim (+1 for
      // the leading expert dimension in the combined 3D tensor).
      int tpDim = getExpertTPShardDim(paramName);
      if (tpDim >= 0) {
        auto tpIt = dm.nameToAxis.find("tp");
        if (tpIt != dm.nameToAxis.end()) pl.shardDims[tpIt->second] = tpDim + 1;
      }
    }
  } else {
    int tpDim = getTPShardDim(paramName);
    if (tpDim >= 0) {
      auto tpIt = dm.nameToAxis.find("tp");
      if (tpIt != dm.nameToAxis.end()) pl.shardDims[tpIt->second] = tpDim;
    }
  }
  return pl;
}

// ============================================================================
// Local Shape Computation
// ============================================================================

// Returns false if shape is not evenly divisible.
static bool computeLocalShape(const std::vector<size_t>& globalShape, const DimMap& dm, const Placement& pl,
                              std::vector<size_t>& localShape) {
  int ndim = (int)globalShape.size();
  std::vector<size_t> factors(ndim, 1);

  for (int meshDim = 0; meshDim < (int)pl.shardDims.size(); meshDim++) {
    int sd = pl.shardDims[meshDim];
    if (sd >= 0 && sd < ndim) {
      int meshSize = (meshDim < (int)dm.meshShape.size()) ? dm.meshShape[meshDim] : 1;
      factors[sd] *= meshSize;
    }
  }

  localShape.resize(ndim);
  for (int d = 0; d < ndim; d++) {
    if (globalShape[d] % factors[d] != 0) return false;
    localShape[d] = globalShape[d] / factors[d];
  }
  return true;
}

// ============================================================================
// JSON Parsing
// ============================================================================

static ParallelConfig parseParallelConfig(const json& j) {
  ParallelConfig c;
  c.numGpus = j.at("num_gpus").get<int>();
  c.tp = j.value("tp_size", 1);
  c.cp = j.value("cp_size", 1);
  c.ep = j.value("ep_size", 1);
  c.dp = j.value("dp_size", 1);
  c.pp = j.value("pp_size", 1);
  return c;
}

static std::map<std::string, ParamInfo> loadModelConfig(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    printf("ERROR: Cannot open model config: %s\n", path.c_str());
    abort();
  }
  json data = json::parse(f);
  std::map<std::string, ParamInfo> params;
  for (auto& [name, obj] : data.items()) {
    ParamInfo pi;
    pi.name = name;
    for (auto& d : obj["shape"]) pi.shape.push_back(d.get<size_t>());
    pi.dtype = obj.value("dtype", "BF16");
    pi.elementSize = getElementSize(pi.dtype);
    params[name] = pi;
  }
  return params;
}

// ============================================================================
// Expert Param Grouping  (port of _group_expert_params)
// ============================================================================

static std::map<std::string, ParamInfo> groupExpertParams(const std::map<std::string, ParamInfo>& modelConfig) {
  std::map<std::string, ParamInfo> result;
  std::map<std::string, std::vector<const ParamInfo*>> expertGroups;

  std::regex expertRe(R"((.+\.experts)\.\d+\.(.+))");
  for (auto& [name, info] : modelConfig) {
    if (isExpertParam(name)) {
      std::smatch m;
      if (std::regex_match(name, m, expertRe)) {
        std::string groupKey = m[1].str() + "." + m[2].str();
        expertGroups[groupKey].push_back(&info);
      } else {
        result[name] = info;
      }
    } else {
      result[name] = info;
    }
  }

  for (auto& [groupKey, experts] : expertGroups) {
    size_t numExperts = experts.size();
    const ParamInfo* rep = experts[0];
    ParamInfo combined;
    combined.name = groupKey;
    combined.shape.push_back(numExperts);
    for (auto d : rep->shape) combined.shape.push_back(d);
    combined.dtype = rep->dtype;
    combined.elementSize = rep->elementSize;
    result[groupKey] = combined;
  }
  return result;
}

// ============================================================================
// Layer Counting + PP Stage Mapping  (port of model_refit_test helpers)
// ============================================================================

static int countLayers(const std::map<std::string, ParamInfo>& params) {
  int maxIdx = -1;
  std::regex layerRe(R"(layers\.(\d+))");
  for (auto& [name, _] : params) {
    std::smatch m;
    if (std::regex_search(name, m, layerRe)) {
      int idx = std::stoi(m[1].str());
      if (idx > maxIdx) maxIdx = idx;
    }
  }
  return maxIdx >= 0 ? maxIdx + 1 : 0;
}

static int getPPStage(int layerIdx, int numLayers, int ppSize) {
  return layerIdx * ppSize / numLayers;
}

static std::pair<int, int> getParamPPStages(const std::string& name, int numLayers, int trainPP, int genPP) {
  std::regex layerRe(R"(layers\.(\d+))");
  std::smatch m;
  if (std::regex_search(name, m, layerRe)) {
    int layerIdx = std::stoi(m[1].str());
    return {getPPStage(layerIdx, numLayers, trainPP), getPPStage(layerIdx, numLayers, genPP)};
  }
  if (contains(name, "embed_tokens")) return {0, 0};
  return {trainPP - 1, genPP - 1};
}

// ============================================================================
// Deduplication  (port of _deduplicate_params_pp)
// ============================================================================

static std::map<std::string, ParamInfo> deduplicateParamsPP(
  const std::map<std::string, ParamInfo>& params, int numLayers, int trainPP, int genPP, int maxPerGroup = 1) {
  std::map<std::string, std::vector<std::pair<std::string, ParamInfo>>> seen;
  std::regex layerRe(R"(layers\.(\d+))");

  for (auto& [name, info] : params) {
    std::string pattern;
    std::smatch m;
    if (std::regex_search(name, m, layerRe)) {
      int layerIdx = std::stoi(m[1].str());
      int tStage = getPPStage(layerIdx, numLayers, trainPP);
      int gStage = getPPStage(layerIdx, numLayers, genPP);
      std::string replacement = "layers.T" + std::to_string(tStage) + "G" + std::to_string(gStage);
      pattern = std::regex_replace(name, std::regex(R"(layers\.\d+)"), replacement);
    } else {
      pattern = name;
    }
    if ((int)seen[pattern].size() < maxPerGroup) seen[pattern].push_back({name, info});
  }

  std::map<std::string, ParamInfo> result;
  for (auto& [_, group] : seen)
    for (auto& [name, info] : group) result[name] = info;
  return result;
}

// ============================================================================
// ncclMesh_t Construction from Placements
//
// The ncclReshardWithWindow API takes a 2D mesh: dims[2], startRank, placement[2].
//   dims[0] = replicated count (product of non-sharding mesh axes)
//   dims[1] = shard count     (mesh axis that shards the tensor)
//   placement[0] = REPLICATE
//   placement[1] = SHARD(tensor_dim) or REPLICATE
// ============================================================================

struct MeshSpec {
  int repCount;
  int shardCount;
  int shardTensorDim; // -1 if fully replicated
};

static MeshSpec buildMeshSpec(const Placement& pl, const DimMap& dm) {
  int rep = 1;
  int shard = 1;
  int shardTensorDim = -1;

  for (int ax = 0; ax < (int)pl.shardDims.size() && ax < (int)dm.meshShape.size(); ax++) {
    if (pl.shardDims[ax] >= 0) {
      shard *= dm.meshShape[ax];
      shardTensorDim = pl.shardDims[ax];
    } else {
      rep *= dm.meshShape[ax];
    }
  }

  if (shard == 1) {
    // Fully replicated: the whole stage is rep
    int totalRanks = 1;
    for (auto s : dm.meshShape) totalRanks *= s;
    rep = totalRanks;
  }

  return {rep, shard, shardTensorDim};
}

// ============================================================================
// Transfer Descriptor
// ============================================================================

struct TransferDesc {
  std::string paramName;
  ParamInfo param;
  int trainStage;
  int genStage;
  MeshSpec srcMesh;
  MeshSpec dstMesh;
  Placement srcPlacement;
  Placement dstPlacement;
  std::vector<size_t> srcLocalShape;
  std::vector<size_t> dstLocalShape;
  size_t srcLocalBytes;
  size_t dstLocalBytes;
  size_t maxLocalBytes;
  int ndims;
};

// ============================================================================
// Main
// ============================================================================

static void printUsage(const char* prog) {
  printf("Usage: %s --model-config <json> --system-config <json> [options]\n", prog);
  printf("\nModel transfer benchmark using real model/system config files.\n");
  printf("\nRequired:\n");
  printf("  --model-config <file>   Model config JSON (HF per-param "
         "shapes/dtypes)\n");
  printf("  --system-config <file>  System config JSON (train/gen "
         "parallelism)\n");
  printf("\nOptions:\n");
  printf("  --iterations <N>        Timed iterations (default: 10)\n");
  printf("  --warmup <N>            Warmup iterations (default: 2)\n");
  printf("  --gpus-per-node <N>     GPUs per node (default: 8)\n");
  printf("  --algorithm <auto|ring|direct>  Reshard algorithm (default: "
         "auto)\n");
  printf("  --lb-mode <uniform|node>        Load balance mode (default: "
         "uniform)\n");
  printf("  --no-dedup              Disable param deduplication\n");
  printf("  --validate              Validate correctness after warmup\n");
  printf("  --validate-iterations <N>  Validation iterations (default: 3)\n");
  printf("  --verbose               Enable debug output\n");
  printf("  --help                  Show this help\n");
}

int main(int argc, char* argv[]) {
  MPICHECK(MPI_Init(&argc, &argv));

  int mpiRank, mpiSize;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));

  // Defaults
  std::string modelConfigPath;
  std::string systemConfigPath;
  int iterations = 10;
  int warmup = 2;
  int gpusPerNode = 8;
  bool deduplicate = true;
  bool validate = false;
  int validateIterations = 3;
  bool verbose = false;
  const char* algorithm = "AUTO";
  const char* lbMode = "UNIFORM";

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--model-config") == 0) {
      modelConfigPath = argv[++i];
    } else if (strcmp(argv[i], "--system-config") == 0) {
      systemConfigPath = argv[++i];
    } else if (strcmp(argv[i], "--iterations") == 0) {
      iterations = benchParseInt(argv[++i], "--iterations");
    } else if (strcmp(argv[i], "--warmup") == 0) {
      warmup = benchParseInt(argv[++i], "--warmup");
    } else if (strcmp(argv[i], "--gpus-per-node") == 0) {
      gpusPerNode = benchParseInt(argv[++i], "--gpus-per-node");
    } else if (strcmp(argv[i], "--no-dedup") == 0) {
      deduplicate = false;
    } else if (strcmp(argv[i], "--validate") == 0) {
      validate = true;
    } else if (strcmp(argv[i], "--validate-iterations") == 0) {
      validateIterations = benchParseInt(argv[++i], "--validate-iterations");
    } else if (strcmp(argv[i], "--verbose") == 0) {
      verbose = true;
    } else if (strcmp(argv[i], "--algorithm") == 0) {
      ++i;
      if (strcmp(argv[i], "direct") == 0) algorithm = "DIRECT";
      else if (strcmp(argv[i], "ring") == 0) algorithm = "RING";
      else algorithm = "AUTO";
    } else if (strcmp(argv[i], "--lb-mode") == 0) {
      ++i;
      if (strcmp(argv[i], "node") == 0) lbMode = "NODE_AWARE";
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      if (mpiRank == 0) printUsage(argv[0]);
      MPI_Finalize();
      return 0;
    }
  }

  if (modelConfigPath.empty() || systemConfigPath.empty()) {
    if (mpiRank == 0) {
      printf("ERROR: --model-config and --system-config are required\n");
      printUsage(argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  // ========================================================================
  // Parse configs
  // ========================================================================
  std::ifstream sysFile(systemConfigPath);
  if (!sysFile.is_open()) {
    if (mpiRank == 0) printf("ERROR: Cannot open system config: %s\n", systemConfigPath.c_str());
    MPI_Finalize();
    return 1;
  }
  json sysJson = json::parse(sysFile);
  ParallelConfig trainCfg = parseParallelConfig(sysJson["train"]);
  ParallelConfig genCfg = parseParallelConfig(sysJson["generation"]);

  int trainNumGpus = trainCfg.numGpus;
  int genNumGpus = genCfg.numGpus;
  int totalGpus = trainNumGpus + genNumGpus;

  if (mpiSize != totalGpus) {
    if (mpiRank == 0)
      printf("ERROR: worldSize=%d != train(%d) + gen(%d) = %d\n", mpiSize, trainNumGpus, genNumGpus, totalGpus);
    MPI_Finalize();
    return 1;
  }

  auto rawModelConfig = loadModelConfig(modelConfigPath);
  auto modelConfig = groupExpertParams(rawModelConfig);

  int numLayers = countLayers(modelConfig);
  int trainPP = trainCfg.pp;
  int genPP = genCfg.pp;
  int trainStageSize = trainNumGpus / trainPP;
  int genStageSize = genNumGpus / genPP;

  if (deduplicate) modelConfig = deduplicateParamsPP(modelConfig, numLayers, trainPP, genPP, 1);

  // ETP=1 two-mesh model: non-expert and expert params use different
  // logical mesh layouts over the same physical ranks.
  DimMapPair trainDimMaps = buildDimMapsETP1(trainCfg);

  // Gen side typically has no EP (vLLM doesn't use EP), so a single mesh
  // suffices.  Use buildDimMap for both non-expert and expert on gen.
  DimMap genDimMap = buildDimMap(genCfg);

  // ========================================================================
  // Determine rank role
  // ========================================================================
  bool isTrainer = (mpiRank < trainNumGpus);
  int myGlobalOffset = isTrainer ? 0 : trainNumGpus;
  int myLocalIdx = mpiRank - myGlobalOffset;
  int myStageSize = isTrainer ? trainStageSize : genStageSize;
  int myStage = myLocalIdx / myStageSize;

  // ========================================================================
  // Build transfer descriptors + enumerate PP comm pairs
  // ========================================================================
  std::set<std::pair<int, int>> ppCommPairs;
  std::vector<TransferDesc> allTransfers;

  int skipped = 0;
  for (auto& [paramName, param] : modelConfig) {
    auto [tStage, gStage] = getParamPPStages(paramName, numLayers, trainPP, genPP);
    ppCommPairs.insert({tStage, gStage});

    int ndim = (int)param.shape.size();

    if (ndim > 3) {
      skipped++;
      continue;
    }

    // ETP=1: select non-expert or expert DimMap for the train side.
    // Gen side uses a single mesh (no EP).
    const DimMap& trainDM = isExpertParam(paramName) ? trainDimMaps.expert : trainDimMaps.nonExpert;

    Placement srcPl = getPlacements(paramName, trainDM, ndim);
    Placement dstPl = getPlacements(paramName, genDimMap, ndim);

    std::vector<size_t> srcLocal, dstLocal;
    if (!computeLocalShape(param.shape, trainDM, srcPl, srcLocal) ||
        !computeLocalShape(param.shape, genDimMap, dstPl, dstLocal)) {
      skipped++;
      continue;
    }

    MeshSpec srcMesh = buildMeshSpec(srcPl, trainDM);
    MeshSpec dstMesh = buildMeshSpec(dstPl, genDimMap);

    size_t srcBytes = param.elementSize;
    for (auto d : srcLocal) srcBytes *= d;
    size_t dstBytes = param.elementSize;
    for (auto d : dstLocal) dstBytes *= d;

    // Skip fully-replicated tiny params (layernorms, biases) that are
    // replicated on both sides -- they still transfer but are small.
    // We include them for accuracy.

    TransferDesc td;
    td.paramName = paramName;
    td.param = param;
    td.trainStage = tStage;
    td.genStage = gStage;
    td.srcMesh = srcMesh;
    td.dstMesh = dstMesh;
    td.srcPlacement = srcPl;
    td.dstPlacement = dstPl;
    td.srcLocalShape = srcLocal;
    td.dstLocalShape = dstLocal;
    td.srcLocalBytes = srcBytes;
    td.dstLocalBytes = dstBytes;
    td.maxLocalBytes = std::max(srcBytes, dstBytes);
    td.ndims = ndim;
    allTransfers.push_back(td);
  }

  // Ensure embed/lm_head comms exist
  ppCommPairs.insert({0, 0});
  ppCommPairs.insert({trainPP - 1, genPP - 1});

  // ========================================================================
  // Configure reshard library via env vars
  // ========================================================================
  if (verbose) benchSetEnv("NCCL_RESHARD_LOG_LEVEL", "DEBUG");
  benchSetEnv("NCCL_RESHARD_ALGORITHM", algorithm);
  benchSetEnv("NCCL_RESHARD_LB_MODE", lbMode);
  NCCLCHECK(ncclM2nInit(NULL));

  // ========================================================================
  // Print configuration
  // ========================================================================
  if (mpiRank == 0) {
    printf("=== Model Transfer Benchmark ===\n");
    printf("Model config : %s\n", modelConfigPath.c_str());
    printf("System config: %s\n", systemConfigPath.c_str());
    printf("Trainer: TP=%d, CP=%d, EP=%d, DP=%d, PP=%d -> %d GPUs "
           "(%d/stage)\n",
           trainCfg.tp, trainCfg.cp, trainCfg.ep, trainCfg.dp, trainCfg.pp, trainNumGpus, trainStageSize);
    printf("  ETP=1 non-expert mesh: [");
    for (int i = 0; i < (int)trainDimMaps.nonExpert.meshShape.size(); i++)
      printf("%s%d", i ? "," : "", trainDimMaps.nonExpert.meshShape[i]);
    printf("]\n");
    printf("  ETP=1 expert mesh:     [");
    for (int i = 0; i < (int)trainDimMaps.expert.meshShape.size(); i++)
      printf("%s%d", i ? "," : "", trainDimMaps.expert.meshShape[i]);
    printf("]\n");
    printf("Generator: TP=%d, CP=%d, EP=%d, DP=%d, PP=%d -> %d GPUs "
           "(%d/stage)\n",
           genCfg.tp, genCfg.cp, genCfg.ep, genCfg.dp, genCfg.pp, genNumGpus, genStageSize);
    printf("Layers: %d, Params (after grouping+dedup): %zu (%d skipped)\n", numLayers, allTransfers.size(), skipped);
    printf("PP comm pairs: %zu\n", ppCommPairs.size());
    printf("Algorithm: %s, LB Mode: %s\n", algorithm, lbMode);
    printf("Iterations: %d (warmup: %d), Validate: %s (iters: %d), Dedup: "
           "%s\n",
           iterations, warmup, validate ? "yes" : "no", validateIterations, deduplicate ? "yes" : "no");
    fflush(stdout);
  }

  // ========================================================================
  // Setup CUDA device
  // ========================================================================
  int localRank = mpiRank % gpusPerNode;
  int numDevices;
  CUDACHECK(cudaGetDeviceCount(&numDevices));
  CUDACHECK(cudaSetDevice(localRank % numDevices));

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // ========================================================================
  // Create NCCL Communicators — one per (train_stage, gen_stage) pair
  // ========================================================================
  struct PPCommEntry {
    ncclComm_t comm = nullptr;
    int localRank = -1;
    int commSize = 0;
    bool participates = false;
    cudaStream_t stream = nullptr;
  };

  std::map<std::pair<int, int>, PPCommEntry> ppComms;

  for (auto& [tStage, gStage] : ppCommPairs) {
    PPCommEntry entry;

    bool participates = false;
    int key = 0;
    if (isTrainer) {
      participates = (myStage == tStage);
      key = participates ? (myLocalIdx % trainStageSize) : 0;
    } else {
      participates = (myStage == gStage);
      key = participates ? (trainStageSize + myLocalIdx % genStageSize) : 0;
    }
    entry.participates = participates;
    entry.commSize = trainStageSize + genStageSize;

    // All ranks must call ncclGetUniqueId / MPI_Bcast in same order.

    ncclUniqueId uniqueId;
    // Root = lowest participating rank.  Train stage 0 rank is always
    // lower than gen ranks, so if this rank is trainer stage tStage
    // with local 0, it's the root.
    int trainStageStart = tStage * trainStageSize;
    int root = trainStageStart; // global rank of train stage's first rank

    if (mpiRank == root) NCCLCHECK(ncclGetUniqueId(&uniqueId));
    MPICHECK(MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, root, MPI_COMM_WORLD));

    if (participates) {
      entry.localRank = key;
      NCCLCHECK(ncclCommInitRank(&entry.comm, entry.commSize, uniqueId, key));
      CUDACHECK(cudaStreamCreate(&entry.stream));
    }

    ppComms[{tStage, gStage}] = entry;
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  if (mpiRank == 0 && verbose) {
    printf("[DEBUG] %zu PP communicators created\n", ppComms.size());
    fflush(stdout);
  }

  // ========================================================================
  // Allocate per-param buffers + register windows
  //
  // Each transfer descriptor gets its own buffer and window so that
  // multiple params in the same pattern group can be in flight without
  // overwriting each other.  Streams remain one-per-PP-comm (NCCL ops
  // on the same communicator serialize anyway).
  // ========================================================================
  struct TransferBufferEntry {
    void* buffer = nullptr;
    ncclWindow_t window = nullptr;
    size_t allocSize = 0;
  };
  std::vector<TransferBufferEntry> transferBuffers(allTransfers.size());

  for (size_t i = 0; i < allTransfers.size(); i++) {
    auto& td = allTransfers[i];
    auto key = std::make_pair(td.trainStage, td.genStage);
    auto& commEntry = ppComms[key];
    if (!commEntry.participates) continue;

    size_t bufSize = td.maxLocalBytes;
    if (bufSize == 0) bufSize = 4096;

    TransferBufferEntry& tbe = transferBuffers[i];
    tbe.allocSize = bufSize;
    NCCLCHECK(ncclMemAlloc(&tbe.buffer, bufSize));
    CUDACHECK(cudaMemset(tbe.buffer, 0, bufSize));
    NCCLCHECK(ncclCommWindowRegister(commEntry.comm, tbe.buffer, bufSize, &tbe.window, NCCL_WIN_COLL_SYMMETRIC));

    if (verbose) {
      printf("[Rank %d] buffer registered: param=%s comm=(%d,%d) size=%zu\n", mpiRank, td.paramName.c_str(), key.first,
             key.second, bufSize);
      fflush(stdout);
    }
  }

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // ========================================================================
  // Group transfers by base pattern for per-pattern reporting
  // ========================================================================
  struct PatternGroup {
    std::string baseName;
    std::vector<size_t> transferIndices;
  };
  std::map<std::string, PatternGroup> patternGroups;
  std::regex layerDigitRe(R"(layers\.\d+)");
  for (size_t i = 0; i < allTransfers.size(); i++) {
    std::string pattern = std::regex_replace(allTransfers[i].paramName, layerDigitRe, "layers.X");
    auto& pg = patternGroups[pattern];
    pg.baseName = pattern;
    pg.transferIndices.push_back(i);
  }

  // ========================================================================
  // Run a single transfer
  // ========================================================================
  auto runOneTransfer = [&](const TransferDesc& td, const TransferBufferEntry& tbe) {
    auto key = std::make_pair(td.trainStage, td.genStage);
    auto commIt = ppComms.find(key);
    if (commIt == ppComms.end() || !commIt->second.participates) return;

    ncclComm_t comm = commIt->second.comm;
    cudaStream_t stream = commIt->second.stream;
    void* buffer = tbe.buffer;
    ncclWindow_t win = tbe.window;

    ncclMesh_t srcMesh;
    srcMesh.dims[0] = td.srcMesh.repCount;
    srcMesh.dims[1] = td.srcMesh.shardCount;
    srcMesh.startRank = 0;
    srcMesh.placement[0] = NCCL_RESHARD_REPLICATE;
    srcMesh.placement[1] =
      (td.srcMesh.shardTensorDim >= 0) ? NCCL_RESHARD_SHARD(td.srcMesh.shardTensorDim) : NCCL_RESHARD_REPLICATE;

    ncclMesh_t dstMesh;
    dstMesh.dims[0] = td.dstMesh.repCount;
    dstMesh.dims[1] = td.dstMesh.shardCount;
    dstMesh.startRank = trainStageSize;
    dstMesh.placement[0] = NCCL_RESHARD_REPLICATE;
    dstMesh.placement[1] =
      (td.dstMesh.shardTensorDim >= 0) ? NCCL_RESHARD_SHARD(td.dstMesh.shardTensorDim) : NCCL_RESHARD_REPLICATE;

    bool rankIsTrainInComm = (commIt->second.localRank < trainStageSize);

    ncclDistTensor_t srcTensor = {};
    srcTensor.dataPtr = rankIsTrainInComm ? buffer : nullptr;
    srcTensor.ndims = td.ndims;
    srcTensor.dtype = getNcclDtype(td.param.dtype);
    srcTensor.mesh = &srcMesh;
    if (rankIsTrainInComm)
      for (int d = 0; d < td.ndims; d++) srcTensor.localShape[d] = td.srcLocalShape[d];

    ncclDistTensor_t dstTensor = {};
    dstTensor.dataPtr = rankIsTrainInComm ? nullptr : buffer;
    dstTensor.ndims = td.ndims;
    dstTensor.dtype = getNcclDtype(td.param.dtype);
    dstTensor.mesh = &dstMesh;
    if (!rankIsTrainInComm)
      for (int d = 0; d < td.ndims; d++) dstTensor.localShape[d] = td.dstLocalShape[d];

    NCCLCHECK(ncclReshardWithWindow(comm, win, &srcTensor, &dstTensor, stream));
  };

  auto syncAllStreams = [&]() {
    for (auto& [key, entry] : ppComms)
      if (entry.participates) CUDACHECK(cudaStreamSynchronize(entry.stream));
  };

  // ========================================================================
  // Warmup
  // ========================================================================
  if (mpiRank == 0) {
    printf("\nRunning %d warmup iterations...\n", warmup);
    fflush(stdout);
  }

  for (int w = 0; w < warmup; w++) {
    for (size_t i = 0; i < allTransfers.size(); i++) runOneTransfer(allTransfers[i], transferBuffers[i]);
    syncAllStreams();
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  if (mpiRank == 0) {
    printf("Warmup complete.\n");
    fflush(stdout);
  }

  // ========================================================================
  // Timed Iterations — per-pattern reporting with optional validation
  // ========================================================================
  if (mpiRank == 0) {
    printf("%-50s %5s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n", "Parameter Pattern", "Count", "T_MB_min",
           "T_MB_max", "T_MB_avg", "G_MB_min", "G_MB_max", "G_MB_avg", "Lat_max", "BW_min", "BW_max");
    printf("%-50s %5s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n", std::string(50, '-').c_str(), "-----",
           "----------", "----------", "----------", "----------", "----------", "----------", "----------",
           "----------", "----------");
    fflush(stdout);
  }

  double cumulativeLatencyMs = 0.0;
  size_t cumulativeBytes = 0;
  int patternCount = 0;
  bool globalValidationPassed = true;

  for (auto& patternGroup : patternGroups) {
    const auto& patternName = patternGroup.first;
    auto& pg = patternGroup.second;
    // Compute local bytes for this group, split by role
    size_t groupLocalBytes = 0;
    size_t trainerLocalBytes = 0;
    size_t genLocalBytes = 0;
    for (size_t idx : pg.transferIndices) {
      auto& td = allTransfers[idx];
      auto key = std::make_pair(td.trainStage, td.genStage);
      auto commIt = ppComms.find(key);
      if (commIt == ppComms.end() || !commIt->second.participates) continue;
      bool rankIsTrainInComm = (commIt->second.localRank < trainStageSize);
      size_t bytes = rankIsTrainInComm ? td.srcLocalBytes : td.dstLocalBytes;
      groupLocalBytes += bytes;
      if (rankIsTrainInComm) trainerLocalBytes += bytes;
      else genLocalBytes += bytes;
    }

    auto runGroup = [&]() {
      for (size_t idx : pg.transferIndices) runOneTransfer(allTransfers[idx], transferBuffers[idx]);
      syncAllStreams();
    };

    // Warmup per pattern
    for (int w = 0; w < warmup; w++) {
      runGroup();
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // ----------------------------------------------------------------
    // Per-pattern validation (if --validate)
    //
    // Runs validateIterations passes.  Each pass encodes (iteration,
    // bufferId) into the byte pattern so that every buffer in every
    // iteration has a unique expected value.  Dest buffers are only
    // zeroed on the first pass; subsequent passes must fully overwrite
    // stale data from the previous iteration.
    // ----------------------------------------------------------------
    if (validate) {
      bool patternValid = true;

      for (int vi = 0; vi < validateIterations; vi++) {
        // 1. Init source data (trainer) / zero dest buffer (gen,
        // first iter only).
        for (size_t idx : pg.transferIndices) {
          auto& td = allTransfers[idx];
          auto& tbe = transferBuffers[idx];
          auto key = std::make_pair(td.trainStage, td.genStage);
          auto commIt = ppComms.find(key);
          if (commIt == ppComms.end() || !commIt->second.participates) continue;

          cudaStream_t stream = commIt->second.stream;
          bool rankIsTrainInComm = (commIt->second.localRank < trainStageSize);

          if (rankIsTrainInComm) {
            int rankInStage = myLocalIdx % trainStageSize;
            int shardDim = td.srcMesh.shardTensorDim;
            int shardIdx = (shardDim >= 0) ? (rankInStage % td.srcMesh.shardCount) : 0;
            int shardCount = (shardDim >= 0) ? td.srcMesh.shardCount : 1;

            size_t localDims[3] = {
              td.srcLocalShape[0],
              td.srcLocalShape.size() >= 2 ? td.srcLocalShape[1] : 1,
              td.srcLocalShape.size() >= 3 ? td.srcLocalShape[2] : 1,
            };
            localDims[td.ndims - 1] *= td.param.elementSize;

            benchInitSourceData((char*)tbe.buffer, localDims, td.ndims, shardDim, shardIdx, shardCount, stream, vi,
                                (int)idx);
          } else if (vi == 0) {
            CUDACHECK(cudaMemsetAsync(tbe.buffer, 0, tbe.allocSize, stream));
          }
        }
        syncAllStreams();
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // 2. Run transfers (same code path as timed loop)
        for (size_t idx : pg.transferIndices) runOneTransfer(allTransfers[idx], transferBuffers[idx]);
        syncAllStreams();
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // 3. Validate on gen ranks
        if (!isTrainer) {
          for (size_t idx : pg.transferIndices) {
            auto& td = allTransfers[idx];
            auto& tbe = transferBuffers[idx];
            auto key = std::make_pair(td.trainStage, td.genStage);
            auto commIt = ppComms.find(key);
            if (commIt == ppComms.end() || !commIt->second.participates) continue;

            int rankInStage = myLocalIdx % genStageSize;
            int shardDim = td.dstMesh.shardTensorDim;
            int shardIdx = (shardDim >= 0) ? (rankInStage % td.dstMesh.shardCount) : 0;
            int shardCount = (shardDim >= 0) ? td.dstMesh.shardCount : 1;

            size_t localDims[3] = {
              td.dstLocalShape[0],
              td.dstLocalShape.size() >= 2 ? td.dstLocalShape[1] : 1,
              td.dstLocalShape.size() >= 3 ? td.dstLocalShape[2] : 1,
            };
            localDims[td.ndims - 1] *= td.param.elementSize;

            bool ok = benchValidateDestData((const char*)tbe.buffer, localDims, td.ndims, shardDim, shardIdx,
                                            shardCount, mpiRank, commIt->second.stream, vi, (int)idx);
            if (!ok) {
              printf("[Rank %d] VALIDATION FAILED (iter %d): %s\n", mpiRank, vi, td.paramName.c_str());
              patternValid = false;
            }
          }
        }

        int localOk = patternValid ? 1 : 0;
        int groupOk = 0;
        MPICHECK(MPI_Allreduce(&localOk, &groupOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (!groupOk) {
          globalValidationPassed = false;
          if (mpiRank == 0) {
            printf("  [VALIDATE] %s (iter %d): FAILED\n", patternName.c_str(), vi);
            fflush(stdout);
          }
          break;
        }
      }

      // Re-zero gen buffers for clean timed iterations
      if (!isTrainer) {
        for (size_t idx : pg.transferIndices) {
          auto& tbe = transferBuffers[idx];
          CUDACHECK(cudaMemset(tbe.buffer, 0, tbe.allocSize));
        }
      }
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // Timed iterations
    double totalLatMs = 0.0;
    for (int it = 0; it < iterations; it++) {
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      CUDACHECK(cudaDeviceSynchronize());

      auto t0 = std::chrono::high_resolution_clock::now();
      runGroup();
      auto t1 = std::chrono::high_resolution_clock::now();

      double latMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
      totalLatMs += latMs;
    }

    double avgLatMs = totalLatMs / iterations;
    cumulativeLatencyMs += avgLatMs;
    cumulativeBytes += groupLocalBytes;

    // Per-rank BW for this pattern
    double myBw = (groupLocalBytes > 0 && avgLatMs > 0) ?
                    ((double)groupLocalBytes / (avgLatMs / 1000.0)) / (1024.0 * 1024.0 * 1024.0) :
                    0.0;

    // Gather latency min/max/avg across ranks
    double latMin = 0.0, latMax = 0.0, latSum = 0.0;
    MPICHECK(MPI_Reduce(&avgLatMs, &latMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&avgLatMs, &latMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&avgLatMs, &latSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

    // Gather BW min/max/avg across ranks
    double bwMin = 0.0, bwMax = 0.0, bwSum = 0.0;
    MPICHECK(MPI_Reduce(&myBw, &bwMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&myBw, &bwMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&myBw, &bwSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

    // Gather per-role MB stats. Use INFINITY sentinel for the
    // non-participating role.
    double tMB = (double)trainerLocalBytes / (1024.0 * 1024.0);
    double tMBforMin = isTrainer ? tMB : INFINITY;
    double gMB = (double)genLocalBytes / (1024.0 * 1024.0);
    double gMBforMin = isTrainer ? INFINITY : gMB;

    double tMBmin = 0, tMBmax = 0, tMBsum = 0;
    MPICHECK(MPI_Reduce(&tMBforMin, &tMBmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&tMB, &tMBmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&tMB, &tMBsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

    double gMBmin = 0, gMBmax = 0, gMBsum = 0;
    MPICHECK(MPI_Reduce(&gMBforMin, &gMBmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&gMB, &gMBmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&gMB, &gMBsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

    if (mpiRank == 0) {
      printf("%-50s %5zu %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.3f "
             "%10.2f %10.2f\n",
             patternName.c_str(), pg.transferIndices.size(), tMBmin, tMBmax, tMBsum / trainNumGpus, gMBmin, gMBmax,
             gMBsum / genNumGpus, latMax, bwMin, bwMax);
      fflush(stdout);
    }
    patternCount++;
  }

  // Aggregate per-rank validation status across the world so the summary
  // and exit code reflect *any* rank's failure, not just rank 0's view.
  if (validate) {
    int localOk = globalValidationPassed ? 1 : 0;
    int worldOk = 0;
    MPICHECK(MPI_Allreduce(&localOk, &worldOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    globalValidationPassed = (worldOk != 0);

    if (mpiRank == 0) {
      printf("\n*** VALIDATION %s ***\n\n", globalValidationPassed ? "PASSED" : "FAILED");
      fflush(stdout);
    }
  }

  // ========================================================================
  // Aggregate Summary
  // ========================================================================
  double cumLatMin = 0.0, cumLatMax = 0.0, cumLatSum = 0.0;
  double cumBytesGlobal = 0.0;
  double dCumBytes = (double)cumulativeBytes;
  MPICHECK(MPI_Reduce(&cumulativeLatencyMs, &cumLatMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Reduce(&cumulativeLatencyMs, &cumLatMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Reduce(&cumulativeLatencyMs, &cumLatSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Reduce(&dCumBytes, &cumBytesGlobal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

  // Per-rank BW for min/max/avg
  double myBwGBs = (cumulativeBytes > 0 && cumulativeLatencyMs > 0) ?
                     ((double)cumulativeBytes / (cumulativeLatencyMs / 1000.0)) / (1024.0 * 1024.0 * 1024.0) :
                     0.0;

  double bwMin, bwMax, bwSum;
  MPICHECK(MPI_Reduce(&myBwGBs, &bwMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Reduce(&myBwGBs, &bwMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Reduce(&myBwGBs, &bwSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

  // Trainer-only stats
  double trainerBw = isTrainer ? myBwGBs : 0.0;
  double trainerBwForMin = isTrainer ? myBwGBs : INFINITY;
  double tbMin, tbMax, tbSum;
  MPICHECK(MPI_Reduce(&trainerBwForMin, &tbMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Reduce(&trainerBw, &tbMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Reduce(&trainerBw, &tbSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

  // Generator-only stats
  double genBw = isTrainer ? 0.0 : myBwGBs;
  double genBwForMin = isTrainer ? INFINITY : myBwGBs;
  double gbMin, gbMax, gbSum;
  MPICHECK(MPI_Reduce(&genBwForMin, &gbMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Reduce(&genBw, &gbMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Reduce(&genBw, &gbSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

  if (mpiRank == 0) {
    printf("\n==================================================\n");
    printf("       MODEL TRANSFER BENCHMARK RESULTS\n");
    printf("==================================================\n");
    printf("Model  : %s\n", modelConfigPath.c_str());
    printf("System : %s\n", systemConfigPath.c_str());
    printf("Trainer: TP=%d, CP=%d, EP=%d, DP=%d, PP=%d\n", trainCfg.tp, trainCfg.cp, trainCfg.ep, trainCfg.dp,
           trainCfg.pp);
    printf("Generator: TP=%d, CP=%d, EP=%d, DP=%d, PP=%d\n", genCfg.tp, genCfg.cp, genCfg.ep, genCfg.dp, genCfg.pp);
    printf("Params benchmarked: %d patterns, %zu transfers\n", patternCount, allTransfers.size());
    printf("Iterations: %d (warmup: %d)\n", iterations, warmup);

    printf("\n--- Cumulative (sum across all patterns) ---\n");
    printf("Total data (all ranks): %.2f MB\n", cumBytesGlobal / (1024.0 * 1024.0));
    printf("Latency  Min=%.3f  Max=%.3f  Avg=%.3f ms\n", cumLatMin, cumLatMax, cumLatSum / mpiSize);
    double aggBw = (cumBytesGlobal / (cumLatMax / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    printf("Aggregate bandwidth (using max latency): %.2f GB/s\n", aggBw);

    printf("\n--- Per-rank bandwidth (all %d ranks) ---\n", mpiSize);
    printf("  Min=%.2f  Max=%.2f  Avg=%.2f GB/s\n", bwMin, bwMax, bwSum / mpiSize);

    printf("\n--- Trainers only (%d ranks) ---\n", trainNumGpus);
    printf("  Min=%.2f  Max=%.2f  Avg=%.2f GB/s\n", tbMin, tbMax, tbSum / trainNumGpus);

    printf("\n--- Generators only (%d ranks) ---\n", genNumGpus);
    printf("  Min=%.2f  Max=%.2f  Avg=%.2f GB/s\n", gbMin, gbMax, gbSum / genNumGpus);

    printf("==================================================\n");
    fflush(stdout);
  }

  // ========================================================================
  // Cleanup
  // ========================================================================
  ncclM2nFinalize();

  for (size_t i = 0; i < allTransfers.size(); i++) {
    auto& tbe = transferBuffers[i];
    if (!tbe.buffer) continue;
    auto key = std::make_pair(allTransfers[i].trainStage, allTransfers[i].genStage);
    auto commIt = ppComms.find(key);
    if (commIt != ppComms.end() && commIt->second.comm) ncclCommWindowDeregister(commIt->second.comm, tbe.window);
    NCCLCHECK(ncclMemFree(tbe.buffer));
  }

  for (auto& [key, entry] : ppComms) {
    if (entry.stream) CUDACHECK(cudaStreamDestroy(entry.stream));
    if (entry.comm) ncclCommDestroy(entry.comm);
  }

  MPICHECK(MPI_Finalize());

  // Propagate validation result to exit code. Without this, a corrupted
  // reshard prints VALIDATION FAILED but the process exits zero — the
  // caller (CI, driver script) has no way to spot the failure.
  int validationRc = (validate && !globalValidationPassed) ? 1 : 0;
  if (mpiRank == 0) {
    if (validationRc == 0) printf("\nBenchmark completed successfully!\n");
    else printf("\nBenchmark completed with VALIDATION FAILURES.\n");
  }

  return validationRc;
}
