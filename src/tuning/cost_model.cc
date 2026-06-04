/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "tuning.h"
#include "cost_model.h"
#include "comm.h"

// Parse a map of prefixes to a list of elements. The first prefix is
// optional and, if not present, the list of elements will be applied
// to all prefixes. Only the first list of elements can lack a
// prefix. Prefixes (if present) are followed by a colon. Lists of
// elements are comma delimited. Mappings of prefix to the lists of
// elements are semi-colon delimited.
//
// For example:
//
//     NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"
// Enable ring and collnetdirect for all functions, then select tree
// and collnetdirect for allreduce and ring for broadcast.
//
//     NCCL_PROTO="LL,Simple;allreduce:^LL"
// Enable LL and Simple for all functions, but everything except LL
// for allreduce.
//
//     NCCL_PROTO="^LL128;allreduce:LL128"
// Enable everything but LL128, but only LL128 for allreduce.
static ncclResult_t parseList(const char* str, const char* const prefixElems[], int nprefixes,
                              const char* const elems[], int nelems, int* list, int* forced) {
  ncclResult_t ret = ncclSuccess;
  char* fullStr = strdup(str);
  char* tmpFullStr;
  char* fullToken = strtok_r(fullStr, ";", &tmpFullStr);
  char* subToken = nullptr;
  char* tokStr = nullptr;
  while (fullToken) {
    subToken = strdup(fullToken);
    char* tmpSubStr;
    char* prefix = strtok_r(subToken, ":", &tmpSubStr);
    char* elemList = strtok_r(NULL, ":", &tmpSubStr);
    if (elemList == NULL) {
      if (fullToken != fullStr) {
        // It makes no sense for any entry other than the first to not have a prefix,
        // because then all the prefixes before the prefix-less entry would be
        // overwritten.
        WARN("All entries except the first must have a prefix: \"%s\"", str);
        ret = ncclInvalidUsage;
        goto fail;
      }
      elemList = prefix;
      prefix = NULL;
    }

    int unset, set;
    if (elemList[0] == '^') {
      unset = 1;
      set = 0;
      elemList++;
    } else {
      unset = 0;
      set = 1;
    }

    bool foundPrefix = false;
    for (int p = 0; p < nprefixes; p++) {
      if (prefix && strcasecmp(prefix, prefixElems[p]) != 0) continue;
      foundPrefix = true;
      for (int e = 0; e < nelems; e++) list[p * nelems + e] = unset;

      tokStr = strdup(elemList);
      char* tmpStr;
      char* elem = strtok_r(tokStr, ",", &tmpStr);
      while (elem) {
        int e;
        for (e = 0; e < nelems; e++) {
          if (strcasecmp(elem, elems[e]) == 0) {
            list[p * nelems + e] = set;
            forced[p] = 1;
            break;
          }
        }
        if (e == nelems) {
          WARN("Unrecognized element token \"%s\" when parsing \"%s\"", elem, str);
          ret = ncclInvalidUsage;
          goto fail;
        }
        elem = strtok_r(NULL, ",", &tmpStr);
      }
      free(tokStr);
      tokStr = nullptr;
    }
    if (!foundPrefix) {
      WARN("Unrecognized prefix token \"%s\" when parsing \"%s\"", prefix, str);
      ret = ncclInvalidUsage;
      goto fail;
    }
    free(subToken);
    subToken = nullptr;

    fullToken = strtok_r(NULL, ";", &tmpFullStr);
  }

exit:
  free(tokStr);
  free(subToken);
  free(fullStr);
  return ret;
fail:
  goto exit;
}

NCCL_PARAM(Ll128C2c, "LL128_C2C", 1);

static int isLL128Enabled(int minCompCap, int maxCompCap, int interType, int intraType, int nRanks, int func,
                          int algo) {
  int ret = 1;
  if (ncclParamLl128C2c() && minCompCap >= 90) {
    // Enable LL128 by default only on Hopper/Blackwell for all connections up to P2C and PXN.
    ret &= (interType <= PATH_PXN);
  } else {
    // Enable LL128 only up to PXB. Don't enable LL128 over PxN because PxN can encapsulate PxB or P2C links.
    ret &= (interType <= PATH_PXB);
    if (!ncclParamLl128C2c() && minCompCap >= 90)
      INFO(
        NCCL_GRAPH | NCCL_TUNING,
        "Disabling LL128 over all PxN connections (PXB and C2C). This ensures that no C2C link will be used by LL128.");
  }
  ret &= (intraType <= PATH_NVB);
  // Enable LL128 for interoperability between GPUs with different compcap (Hopper and above)
  ret &= (minCompCap == maxCompCap || minCompCap >= 90);
  ret &= !(minCompCap < 70 || (minCompCap == 90 && CUDART_VERSION == 11080 && func == ncclFuncAllReduce &&
                               algo == NCCL_ALGO_RING && nRanks == 2));
  return ret;
}

// Default tuner constants (positional initializers for C++17 compatibility)
static const ncclTunerConstants_t ncclTunerConstantsDefaults = {
    // baseLatencies
  {
    {6.8, 14.0, 8.4},  // Tree
    {6.6, 14.0, 8.4},  // Ring
    {0, 0, 0},         // Collnet Direct
    {0, 0, 0},         // Collnet Chain
    {0, 0, 0},         // NVLS
    {0, 0, 0},         // NVLS Tree
    {8.0, 8.0, 8.0}    // PAT
  },
    // hwLatencies
  {
    /* NVLINK */
    {
      {0.6, 1.25, 4.0}, // Tree (LL/LL128/Simple)
      {0.6, 1.9, 3.4},  // Ring (LL/LL128/Simple)
      {0, 0, 3.7},      // Collnet Direct (LL/LL128/Simple)
      {0, 0, 2.8},      // CollNetChain (LL/LL128/Simple)
      {0, 0, 25},       // NVLS (LL/LL128/Simple)
      {0, 0, 25},       // NVLSTree (LL/LL128/Simple)
      {0, 0, 4.0}       // PAT (LL/LL128/Simple)
    },
    /* PCI */
    {
      {1.0, 1.9, 4.0}, // Tree (LL/LL128/Simple)
      {1.0, 2.5, 5.7}, // Ring (LL/LL128/Simple)
      {0, 0, 3.7},     // Collnet Direct (LL/LL128/Simple)
      {0, 0, 2.8},     // CollNetChain (LL/LL128/Simple)
      {0, 0, 0},       // NVLS (LL/LL128/Simple)
      {0, 0, 0},       // NVLSTree (LL/LL128/Simple)
      {0, 0, 4.0}      // PAT (LL/LL128/Simple)
    },
    /* NET */
    {
      {5.0, 8.5, 14},   // Tree (LL/LL128/Simple)
      {2.7, 4.0, 14.0}, // Ring (LL/LL128/Simple)
      {0, 0, 31},       // Collnet Direct (LL/LL128/Simple)
      {0, 0, 30},       // CollNetChain (LL/LL128/Simple)
      {0, 0, 18},       // NVLS (LL/LL128/Simple)
      {0, 0, 20.9},     // NVLSTree (LL/LL128/Simple)
      {0, 0, 14}        // PAT (LL/LL128/Simple)
    },
  },
    // llMaxBws
  {
    {39.0, 39.0, 20.4}, /* Volta-N1/Intel-N2/Intel-N4) */
    {87.7, 22.5 /*avg of ring & tree*/, 19.0}, /* Ampere-N1/AMD-N2/AMD-N4) */
    {141.0, 45.0 /*avg of ring & tree*/, 35.0}, /* Hopper-N1/AMD-N2/AMD-N4) */
    {2 * 141.0, 2 * 45.0 /*avg of ring & tree*/, 2 * 35.0}, /* Blackwell-N1/AMD-N2/AMD-N4) */
  },
    // perChMaxRingLL128Bws
  {
    {20.0, 20.0, 20.0}, /* Volta (N1/N2/N4) */
    {20.0, 20.0, 20.0}, /* Ampere (N1/N2/N4) */
    {36.7, 36.7, 36.7}, /* Hopper (N1/N2/N4) */
    {40.0, 40.0, 40.0}, /* Blackwell (N1/N2/N4) */
  },
    // perChMaxTreeLL128Bws
  {
    {20.0, 20.0, 20.0}, /* Volta (N1/N2/N4) */
    {20.0, 20.0, 20.0}, /* Ampere (N1/N2/N4) */
    {36.7, 36.7, 29.0}, /* Hopper (N1/N2/N4) */
    {55.6, 31.67, 20.0}, /* Blackwell (N1/N2/N4) */
  },
    // perChMaxTreeBws
  {
    {26.5, 18.5, 10.0}, /* Volta (N1/N2/N4) */
    {24.0, 23.6, 17.8}, /* Ampere (N1/N2/N4) */
    {38.7, 41.4, 36.0}, /* Hopper (N1/N2/N4) */
    {70.0, 42.8, 24.0}, /* Blackwell (N1/N2/N4) */
  },
    // perChMaxNVLSTreeBws
  {
    {26.5, 18.5, 10.0}, /* Volta (N1/N2/N4) */
    {24.0, 23.6, 17.8}, /* Ampere (N1/N2/N4) */
    {0.0, 57.7, 45.5}, /* Hopper (N1/N2/N4) */
    {0.0, 96.0, 80.0} /* Blackwell (N1/N2/N4) */
  }
};

float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0, 1.0},
  {1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0, 1.0},
  {.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9, .9}
};

static struct ncclTuningModelEntry_t modelMap[] = {
    /*
Initialize default, static models here
{mod_init, mod_sim, mod_final, enabled}
Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce
*/
  {ncclTuningTreeModelInit, ncclTuningTreeModelSim, nullptr, {0, 0, 0, 0, 1}},       // Tree/LL
  {ncclTuningTreeModelInit, ncclTuningTreeModelSim, nullptr, {0, 0, 0, 0, 1}},       // Tree/LL128
  {ncclTuningTreeModelInit, ncclTuningTreeModelSim, nullptr, {0, 0, 0, 0, 1}},       // Tree/Simple
  {ncclTuningRingModelInit, ncclTuningRingModelSim, nullptr, {1, 1, 1, 1, 1}},       // Ring/LL
  {ncclTuningRingModelInit, ncclTuningRingModelSim, nullptr, {1, 1, 1, 1, 1}},       // Ring/LL128
  {ncclTuningRingModelInit, ncclTuningRingModelSim, nullptr, {1, 1, 1, 1, 1}},       // Ring/Simple
  {nullptr, nullptr, nullptr, {0}}, // CollNetDirect/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}}, // CollNetDirect/LL128, disabled as there is no implementation
  {ncclTuningCollnetModelInit, ncclTuningCollnetModelSim, nullptr, {0, 0, 1, 1, 1}}, // CollNetDirect/Simple
  {nullptr, nullptr, nullptr, {0}}, // CollNetChain/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}}, // CollNetChain/LL128, disabled as there is no implementation
  {ncclTuningCollnetModelInit, ncclTuningCollnetModelSim, nullptr, {0, 0, 0, 0, 1}}, // CollNetChain/Simple
  {nullptr, nullptr, nullptr, {0}}, // NVLS/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}}, // NVLS/LL128, disabled as there is no implementation
  {ncclTuningNvlsModelInit, ncclTuningNvlsModelSim, nullptr, {0, 0, 1, 1, 1}}, // NVLS/Simple
  {nullptr, nullptr, nullptr, {0}}, // NVLSTree/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}}, // NVLSTree/LL128, disabled as there is no implementation
  {ncclTuningNvlsModelInit, ncclTuningNvlsModelSim, nullptr, {0, 0, 1, 1, 1}}, // NVLSTree/Simple
  {nullptr, nullptr, nullptr, {0}}, // PAT/LL
  {nullptr, nullptr, nullptr, {0}}, // PAT/LL128
  {ncclTuningPatModelInit, ncclTuningPatModelSim, nullptr, {0, 0, 1, 1, 0}}, // PAT/Simple
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 0, 1}}, // AllReduce_AGxLL_R
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 0, 1}}, // AllReduce_AGxLLMC_R
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 0, 1}}, // AllReduce_RSxTmaLD_AGxTmaST
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 0, 1}}, // AllReduce_RSxLD_AGxST
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 0, 1}}, // AllReduce_RSxLDMC_AGxSTMC
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 1, 0, 0}}, // AllGather_LL
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 1, 0, 0}}, // AllGather_LLMC
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 1, 0, 0}}, // AllGather_TmaST
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 1, 0, 0}}, // AllGather_ST
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 1, 0, 0}}, // AllGather_TmaSTMC
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 1, 0, 0}}, // AllGather_STMC
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 1, 0, 0}}, // AllGather_RailRing_LsaSTMC
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 1, 0}}, // ReduceScatter_LL
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 1, 0}}, // ReduceScatter_TmaLD
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 1, 0}}, // ReduceScatter_LD
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 1, 0}}, // ReduceScatter_LDMC
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 1, 0}}, // ReduceScatter_RailA2A_LsaLD
  {nullptr, ncclTuningSymkModelSim, nullptr, {0, 0, 0, 1, 0}}, // ReduceScatter_RailA2A_LsaLDMC
};

/*
  Get a model entry from the model map.
  This will return the model entry for the given id.
  If the id is out of bounds, it will return an invalid argument error.
*/
static ncclResult_t getModelEntry(int id, struct ncclTuningModelEntry_t** entry) {
  if (id < 0 || id >= NCCL_TUNING_COUNT) {
    return ncclInvalidArgument;
  }
  *entry = &modelMap[id];
  return ncclSuccess;
}

/*
  Initialize the cost model for a communicator.
  This will initialize the cost model for all models that are enabled.
  If a model fails to initialize, it will be disabled and the error will be logged.
*/
ncclResult_t ncclTuningCostModelInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  comm->tuningContext.tuningConstants = ncclTunerConstantsDefaults;
  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain cases.
  int protoEnable[NCCL_NUM_FUNCTIONS * NCCL_NUM_PROTOCOLS];
  int algoEnable[NCCL_NUM_FUNCTIONS * NCCL_NUM_ALGORITHMS];
  int symKernelIdEnable[NCCL_NUM_FUNCTIONS * ncclSymkKernelId_Count];
  for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      protoEnable[f * NCCL_NUM_PROTOCOLS + p] = p == NCCL_PROTO_LL128 ? 2 : 1;
    }
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      algoEnable[f * NCCL_NUM_ALGORITHMS + a] = 1;
    }
    for (int k = 0; k < ncclSymkKernelId_Count; k++) {
      symKernelIdEnable[f * ncclSymkKernelId_Count + k] = 1;
    }
  }
  const char* protoStr = ncclGetEnv("NCCL_PROTO");
  const char* algoStr = ncclGetEnv("NCCL_ALGO");
  const char* symKernelIdStr = ncclGetEnv("NCCL_SYM_KERNEL");
  if ((algoStr && strlen(algoStr) > 0) || (symKernelIdStr && strlen(symKernelIdStr) > 0)) {
    std::fill_n(algoEnable, NCCL_NUM_FUNCTIONS * NCCL_NUM_ALGORITHMS, 0);
    std::fill_n(symKernelIdEnable, NCCL_NUM_FUNCTIONS * ncclSymkKernelId_Count, 0);
  }
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(parseList(protoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable,
                        comm->tuningContext.forced));
  }
  if (algoStr) {
    INFO(NCCL_ENV, "NCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(parseList(algoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoEnable,
                        comm->tuningContext.forced));
  }
  if (symKernelIdStr) {
    INFO(NCCL_ENV, "NCCL_SYM_KERNEL set by environment to %s", symKernelIdStr);
    NCCLCHECK(parseList(symKernelIdStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclSymKernelStr, ncclSymkKernelId_Count,
                        symKernelIdEnable, comm->tuningContext.forced));
  }
  for (int i = 0; i < NCCL_TUNING_COUNT; i++) {
    struct ncclTuningModelEntry_t* model = nullptr;
    NCCLCHECKGOTO(getModelEntry(i, &model), ret, fail);
    if (model == nullptr) {
      ret = ncclInternalError;
      goto fail;
    }
    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
      comm->tuningContext.enabled[i][f] = model->enabled[f];
    }
    if (model->init != nullptr) {
      NCCLCHECKGOTO(model->init(comm, i, comm->tuningContext.enabled[i]), ret, fail);
    }
    int algo, proto, symKernelId;
    NCCLCHECK(ncclTuningExpandId(i, &algo, &proto, &symKernelId));
    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
      if (comm->tuningContext.forced[f] == 0 || comm->tuningContext.enabled[i][f] == 0) continue;
      comm->tuningContext.enabled[i][f] = 0;
      TRACE(NCCL_TUNING, "a/p/s %s/%s/%s enabled %d/%d/%d", ncclAlgoStr[algo], ncclProtoStr[proto],
            ncclSymkKernelIdToString(symKernelId),
            algo != NCCL_ALGO_UNDEF ? algoEnable[f * NCCL_NUM_ALGORITHMS + algo] : -1,
            proto != NCCL_PROTO_UNDEF ? protoEnable[f * NCCL_NUM_PROTOCOLS + proto] : -1,
            symKernelId != ncclSymkKernelId_Count ? symKernelIdEnable[f * ncclSymkKernelId_Count + symKernelId] : -1);
      if (((algo != NCCL_ALGO_UNDEF && algoEnable[f * NCCL_NUM_ALGORITHMS + algo] != 0) &&
           (proto != NCCL_PROTO_UNDEF && protoEnable[f * NCCL_NUM_PROTOCOLS + proto] != 0)) ||
          (symKernelId != ncclSymkKernelId_Count && symKernelIdEnable[f * ncclSymkKernelId_Count + symKernelId] != 0)) {
        comm->tuningContext.enabled[i][f] = 1;
      }
      if (proto == NCCL_PROTO_LL128 && !isLL128Enabled(comm->minCompCap, comm->maxCompCap, comm->graphs[algo].typeInter,
                                                       comm->graphs[algo].typeIntra, comm->nRanks, f, algo)) {
        comm->tuningContext.enabled[i][f] = 0;
      }
    }
  }

  if (comm->rank == 0 && (algoStr || protoStr || symKernelIdStr)) {
    constexpr int strLength = 4096;
    char funcAlgoProtoSymKernelIdTuningStr[strLength];
    int offset = 0;
    offset +=
      snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), "\n     Function | ");
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      offset +=
        snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), "%8s  ", ncclProtoStr[p]);
    }
    offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), " | ");
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      offset +=
        snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), "%13s  ", ncclAlgoStr[a]);
    }
    offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), " | ");
    for (int k = 0; k < ncclSymkKernelId_Count; k++) {
      offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), "%13s  ",
                         ncclSymkKernelIdToString(k));
    }
    offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), "\n");

    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
      offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), "%13s | ",
                         ncclFuncStr[f]);
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
        offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), "%8d  ",
                           protoEnable[f * NCCL_NUM_PROTOCOLS + p]);
      }
      offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), " | ");
      for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
        offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), "%13d  ",
                           algoEnable[f * NCCL_NUM_ALGORITHMS + a]);
      }
      offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), " | ");
      for (int k = 0; k < ncclSymkKernelId_Count; k++) {
        offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), "%13d  ",
                           symKernelIdEnable[f * ncclSymkKernelId_Count + k]);
      }
      offset += snprintf(funcAlgoProtoSymKernelIdTuningStr + offset, std::max(0, strLength - offset), "\n");
    }

    INFO(NCCL_ENV, "Enabled NCCL Func/Proto/Algo Matrix:%s", funcAlgoProtoSymKernelIdTuningStr);
  }

exit:
  return ret;
fail:
  goto exit;
}

/*
  Finalize the cost model for a communicator.
  This will finalize the cost model for all models regardless of
  if the model is enabled.  This is to allow all models to
  reallocate any memory allocations even if something disabled them later.
  If a model fails to finalize, the error will be logged.
*/
ncclResult_t ncclTuningCostModelFinalize(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  for (int i = 0; i < NCCL_TUNING_COUNT; i++) {
    struct ncclTuningModelEntry_t* model = nullptr;
    NCCLCHECKGOTO(getModelEntry(i, &model), ret, fail);
    if (model == nullptr) {
      ret = ncclInternalError;
      goto fail;
    }
    if (model->finalize != nullptr) {
      NCCLCHECKGOTO(model->finalize(comm, i), ret, fail);
    }
  }
exit:
  return ncclSuccess;
fail:
  goto exit;
}

/*
  Get a model estimate for a given input.
  This will get a model estimate for the given input.
  If the input is invalid, it will return an invalid argument error.
  If the model is not enabled, it will return a success with a time of 0.
*/
ncclResult_t ncclTuningCostModelSimModel(int id, struct ncclTuningInput_t* const input,
                                         struct ncclTuningResult_t* const result) {
  struct ncclTuningModelEntry_t* model = nullptr;
  ncclResult_t ret = ncclSuccess;
  result->forced = input->comm->tuningContext.forced[input->func];
  NCCLCHECKGOTO(getModelEntry(id, &model), ret, not_valid);
  if (model == nullptr) {
    ret = ncclInternalError;
    goto not_valid;
  }
  if (input->comm->tuningContext.enabled[id][input->func] == 0) {
    goto not_valid;
  }
  if (model->model != nullptr) {
    NCCLCHECKGOTO(model->model(input, result), ret, not_valid);
    if (result->timeUs <= 0.0) {
      goto not_valid;
    }
  } else {
    goto not_valid;
  }
exit:
  return ret;
not_valid:
  result->timeUs = NCCL_TUNING_IGNORE;
  result->valid = 0;
  goto exit;
}
