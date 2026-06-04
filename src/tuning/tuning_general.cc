 /*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cost_model.h"
#include "nccl_tuner.h"
#include "comm.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);
NCCL_PARAM(Ll128Nthreads, "LL128_NTHREADS", -2);

int ncclTuningGetNsteps(int coll, int nRanks) {
  if (coll == ncclFuncAllReduce) {
    return 2 * (nRanks - 1);
  } else if (coll == ncclFuncReduceScatter || coll == ncclFuncAllGather) {
    return nRanks - 1;
  } else {
    return nRanks;
  }
}

int ncclTuningGetCompCapIndex(struct ncclComm* comm) {
  int minCompCap = comm->minCompCap;
  if (minCompCap >= 100) {
    return NCCL_BLACKWELL_COMPCAP_IDX;
  } else if (minCompCap >= 90) {
    return NCCL_HOPPER_COMPCAP_IDX;
  } else if (minCompCap >= 80) {
    return NCCL_AMPERE_COMPCAP_IDX;
  } else {
    return NCCL_VOLTA_COMPCAP_IDX;
  }
}

void ncclTuningGetConstantsIndexes(struct ncclComm* comm, int* index1, int* index2) {
  *index2 = comm->nNodes <= 2 ? comm->nNodes - 1 : 2;
  // LL: for single node, we look at GPU type; for multi-node, we look at CPU type
  *index1 = comm->nNodes == 1 ? ncclTuningGetCompCapIndex(comm) :
            (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD || comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) ? 1 :
                                                                                                             0;
}

void ncclTuningGetHwIndexes(struct ncclComm* comm, int a, int* intraHw, int* interHw) {
  int intra = comm->graphs[a].typeIntra == PATH_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  if (intraHw != nullptr) *intraHw = intra;
  if (interHw != nullptr) *interHw = comm->nNodes == 1 ? intra : NCCL_HW_NET;
}

float ncclTuningGetTime(struct ncclTuningInput_t* const inputs, int a, float* lat, float* bw) {
  int latCount = a == NCCL_ALGO_RING ? inputs->numPipeOps : DIVUP(inputs->numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
  return *lat * latCount + inputs->nBytes / (1000 * (*bw));
}

static int ncclTuningGetNthreads(const char* name, int env, int min, int max, int def) {
  int nt = env;
  if (nt > 0) {
    if (nt % WARP_SIZE != 0) {
      INFO(NCCL_GRAPH | NCCL_ENV, "Invalid %s %d (must be a multiple of %d)", name, nt, WARP_SIZE);
      nt = max;
    } else if (nt > max) {
      INFO(NCCL_GRAPH | NCCL_ENV, "Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      INFO(NCCL_GRAPH | NCCL_ENV, "Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
    }
  } else {
    nt = def;
  }
  return nt;
}

ncclResult_t ncclTuningSetThreadThresholds(struct ncclComm* comm) {
  int simpleDefaultThreads = NCCL_SIMPLE_MAX_NTHREADS;
  comm->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = ncclTuningGetNthreads(
    "NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = ncclTuningGetNthreads(
    "NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->tuningContext.maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] =
    comm->tuningContext.maxThreads[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] =
      comm->tuningContext.maxThreads[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] =
        comm->tuningContext.maxThreads[NCCL_ALGO_NVLS_TREE][NCCL_PROTO_SIMPLE] = NCCL_MAX_NTHREADS;
  comm->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] =
    comm->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] =
      ncclTuningGetNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_LL_MAX_NTHREADS,
                            NCCL_LL_MAX_NTHREADS);
  comm->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] =
    comm->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] =
      ncclTuningGetNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS / 4,
                            NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);

  // Set per-thread amount of work before we increase nThreads and nChannels
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    comm->tuningContext.threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
    comm->tuningContext.threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
    comm->tuningContext.threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  }
  comm->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] *= comm->nRanks;
  comm->tuningContext.threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 512;
  comm->tuningContext.threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] = 512;

  // Override defaults with user env
  const char* str = ncclGetEnv("NCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "NCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[2][NCCL_NUM_PROTOCOLS] = {{-2, -2, -2}, {-2, -2, -2}};
    sscanf(str, "%ld %ld %ld %ld %ld %ld", t[0], t[0] + 1, t[0] + 2, t[1], t[1] + 1, t[1] + 2);
    for (int a = 0; a < 2; a++) {
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->tuningContext.threadThresholds[a][p] = t[a][p];
      }
    }
  }

  INFO(NCCL_INIT, "threadThresholds %ld/%ld/%ld | %ld/%ld/%ld | %ld | %ld",
       comm->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL],
       comm->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128],
       comm->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE],
       comm->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL],
       comm->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128],
       comm->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE],
       comm->tuningContext.threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE],
       comm->tuningContext.threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE]);
  return ncclSuccess;
}

ncclResult_t ncclTuningGetChannels(struct ncclTuningInput_t* const input, struct ncclTuningResult_t* const result) {
  int nc = input->comm->nChannels;
  int nt = input->comm->tuningContext.maxThreads[result->algo][result->proto];
  int threadThreshold = input->comm->tuningContext.threadThresholds[result->algo][result->proto];
  if (result->algo == NCCL_ALGO_COLLNET_DIRECT) {
    // CollNet channel tuning
    int ncSwitch = 16;
    bool flag = true;
    while (ncSwitch >= 1 && flag) {
      while ((flag = input->nBytes < nc * nt * input->comm->channels[0].collnetDirect.nHeads * threadThreshold) &&
             nc > ncSwitch) {
        if (nc == ncSwitch + ncSwitch / 2) threadThreshold /= 2;
        nc--;
      }
      ncSwitch /= 2;
    }
  } else if (result->algo == NCCL_ALGO_NVLS || result->algo == NCCL_ALGO_NVLS_TREE) {
    // NVLS should not need more than 16 channels to get peak BW.
    if (input->comm->nNodes > 1 && result->algo == NCCL_ALGO_NVLS) {
      nc = std::min(input->comm->nvlsChannels, input->comm->nChannels);
    } else {
      nc = input->comm->nvlsChannels;
    }
  } else {
    // Ring/Tree channel tuning
    while (input->nBytes < nc * nt * threadThreshold) {
      if (nc >= 2) nc--;
      else break;
    }
  }

  if (result->algo != NCCL_ALGO_NVLS && result->algo != NCCL_ALGO_NVLS_TREE &&
      result->algo != NCCL_ALGO_COLLNET_DIRECT) {
    while (input->nBytes < nc * nt * threadThreshold) {
      if (nt % 128 == 0) nt /= 2;
      else break;
    }
  }
  if (result->proto == NCCL_PROTO_SIMPLE) {
    if (result->algo == NCCL_ALGO_RING) nt += WARP_SIZE; // Extra warp for sync
    // More threads or sync warps needed due to split thread model
    if (result->algo == NCCL_ALGO_TREE) nt += 4 * WARP_SIZE;
  }
  nt = nt / WARP_SIZE < 3 ? 3 * WARP_SIZE : nt;
  if (result->algo == NCCL_ALGO_TREE) nt = NCCL_MAX_NTHREADS; // Tree now uses all threads always.
  if (result->algo == NCCL_ALGO_PAT) nt = NCCL_MAX_NTHREADS;
  if (result->maxChannels > 0 && result->maxChannels < nc) {
    nc = result->maxChannels;
  } else {
    result->maxChannels = nc;
  }
  result->nChannels = nc;
  result->nWarps = nt / WARP_SIZE;
  TRACE(NCCL_TUNING, "nChannels: %d, nWarps: %d", result->nChannels, result->nWarps);
  return ncclSuccess;
}

/*
  Translate a tuning id into the kernel identifiers.
*/
ncclResult_t ncclTuningExpandId(int tuningId, int* algo, int* proto, int* symKernelId) {
  if (tuningId < 0 || tuningId >= NCCL_TUNING_COUNT) {
    return ncclInvalidUsage;
  }
  if (algo != nullptr) *algo = NCCL_ALGO_UNDEF;
  if (proto != nullptr) *proto = NCCL_PROTO_UNDEF;
  if (symKernelId != nullptr) *symKernelId = ncclSymkKernelId_Count;
  if (tuningId < NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS) {
    if (algo == nullptr || proto == nullptr) return ncclInvalidUsage;
    *algo = tuningId / NCCL_NUM_PROTOCOLS;
    *proto = tuningId % NCCL_NUM_PROTOCOLS;
  } else if (tuningId >= NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS && tuningId < NCCL_TUNING_COUNT) {
    if (symKernelId == nullptr) return ncclInvalidUsage;
    *symKernelId = tuningId - NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS;
  } else {
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}