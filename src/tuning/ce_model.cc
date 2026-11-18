/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "ce_coll.h"
#include "ce_allgather_schedule.h"
#include "tuning.h"
#include "cost_model.h"
#include "cudawrap.h"
#include <cstring>

// Fallback: latency floor plus transfer time for architectures/APIs without a detailed fit.
// Blackwell uses the B200 floors until its detailed model is enabled.
struct ncclCeAgFloorParams {
  float base;
  float perRank;
};

enum {
  CE_ARCH_SM90 = 0,
  CE_ARCH_SM100 = 1,
  CE_ARCH_COUNT
};
enum {
  CE_NOGRAPH = 0,
  CE_GRAPH = 1
};

static const struct ncclCeAgFloorParams CE_AG_FLOOR[CE_ARCH_COUNT][2][2] = {
  // [uc/mc][no-graph, graph]. Captured batch UC uses the no-graph calibration.
  {{{17.50f, 1.957f}, {9.98f, 6.160f}},    // sm_90 UC
   {{14.90f, 0.726f}, {14.45f, 2.004f}}},   // sm_90 MC
  {{{32.04f, 0.130f}, {22.28f, 6.664f}},    // sm_100 UC
   {{29.14f, 0.137f}, {24.51f, 2.755f}}},   // sm_100 MC
};

static constexpr float CE_NVLINK_BW[CE_ARCH_COUNT] = {392.0f, 779.0f};  // sm_90, sm_100

static inline int ncclCeArchBucket(struct ncclComm* comm) {
  return (comm->minCompCap >= 100) ? CE_ARCH_SM100 : CE_ARCH_SM90;
}

static constexpr float CE_D2H_BW_PER_ENGINE[CE_ARCH_COUNT] = {53.0f, 53.0f};  // [arch] sm_90, sm_100
static constexpr int CE_D2H_NUM_ENGINES_X86[CE_ARCH_COUNT] = {2, 2};   // [arch] sm_90, sm_100
static constexpr int CE_BLACKWELL_FIX_DRIVER = 13040;

static float ncclCeD2hBw(struct ncclComm* comm) {
  int arch = ncclCeArchBucket(comm);
  return CE_D2H_BW_PER_ENGINE[arch] * (float)CE_D2H_NUM_ENGINES_X86[arch];
}

static float ncclCeAllGatherFallbackTimeUs(struct ncclComm* comm, bool multicast, int nRanks, size_t perRankBytes,
                                           bool captured, bool inPlace, int driverVersion) {
  int arch = ncclCeArchBucket(comm);
  bool useMemcpyBatch = ncclCeMemcpyBatchSupported(driverVersion, captured, false);
  int capturedIdx = captured && (multicast || !useMemcpyBatch) ? CE_GRAPH : CE_NOGRAPH;
  const struct ncclCeAgFloorParams f = CE_AG_FLOOR[arch][multicast ? 1 : 0][capturedIdx];
  float floorUs = f.base + f.perRank * (float)nRanks;

  // D2H effects (MC bw cap + OOP-unicast slice) only apply on non-coherent CPUs; skip on coherent.
  bool coherent = (comm->cpuArch == NCCL_TOPO_CPU_ARCH_ARM);

  // Pre-fix driver caps multicast D2H bw at min(d2hBw, nvlinkBw/N)*N.
  float nvlinkBw = CE_NVLINK_BW[arch];
  float bw = nvlinkBw;
  if (multicast && !coherent) {
    if (driverVersion < CE_BLACKWELL_FIX_DRIVER) {
      float d2hBw = ncclCeD2hBw(comm);
      float perLink = nvlinkBw / (float)nRanks;
      float lo = (d2hBw < perLink) ? d2hBw : perLink;
      bw = lo * (float)nRanks;
    }
  }

  double dataBytes = (double)perRankBytes * (multicast ? nRanks : (nRanks - 1));
  float bwUs = (bw > 0.0f) ? (float)(dataBytes / bw / 1e3) : 0.0f;
  float total = floorUs + bwUs;

  // Preserve the non-graph D2H bound and apply it to captured batch UC as well.
  if (!multicast && !inPlace && (!captured || useMemcpyBatch) && !coherent) {
    float d2hBw = ncclCeD2hBw(comm);
    if (d2hBw > 0.0f) {
      float altUs = floorUs + (float)((double)perRankBytes / d2hBw / 1e3);
      if (altUs > total) total = altUs;
    }
  }

  if (!multicast && !useMemcpyBatch) {
    int copies = nRanks - int(inPlace);
    ncclCeBatchCopyPlan copyPlan(copies, perRankBytes);
    if (copyPlan.intraBatchSync) {
      int blocks = (copies + copyPlan.syncFreq - 1) / copyPlan.syncFreq;
      // Individual copies synchronize every eight copies, including the even-count workaround.
      total += ((blocks - 1) + (blocks % 2 == 0)) * floorUs;
    }
  }

  return total;
}

// Latency terms are microseconds; bandwidths are decimal GB/s.
struct ncclCeAlgoConstants {
  double setupUs;
  double copyUs;
  double jumpUs;
  double dispatchUs;
  double bandwidthGBps;
  double selfBandwidthGBps;
  double chunkUs;
  size_t jumpBytesThreshold;
};

struct ncclCeAllGatherSyncConstants {
  double unicastSyncUs;
  double multicastSyncUs;
  double sequenceWriteUs;
  double flagResetUs;
};

struct ncclCeAllGatherConstants {
  ncclCeAlgoConstants unicast;
  ncclCeAlgoConstants multicast;
  ncclCeAllGatherSyncConstants sync;
};

enum ncclCeAllGatherArch {
  ncclCeAllGatherArch_VR200,
  ncclCeAllGatherArch_Count
};

// Rows follow ncclCeAllGatherArch; payload costs are shared by graph and non-graph execution.
static constexpr ncclCeAllGatherConstants ceAllGatherConstants[ncclCeAllGatherArch_Count] = {
  {// UC: setup, copy, jump, dispatch (us), remote/self bandwidth (GB/s), extra chunk (us), jump (bytes).
   {15.4324, 0.1385, 3.3173, 0.0, 643.9979, 5305.0398, 0.0649, 20 * 1024},
   // MC uses individual copies; self bandwidth and UC scheduling terms are unused.
   {21.5914, 0.0, 0.0, 7.9352, 1262.6263, 0.0, 0.0, 0},
   // UC peer, MC peer, sequence write, flag reset (us).
   {2.7454, 0.2825, 3.5250, 0.5729}
  }
};

static const ncclCeAllGatherConstants* ncclCeAllGatherConstantsForComm(const struct ncclComm* comm) {
  // The fit assumes one NVLD, homogeneous GPUs, and full GPUs without localized memory.
  if (comm->minCompCap != comm->maxCompCap || comm->devrState.lsaTeamSize != comm->nRanks || comm->hasMloPart ||
      comm->hasMultiRankNvml)
    return nullptr;
  if (RUBIN_AND_LATER(comm->minCompCap) && comm->cpuArch == NCCL_TOPO_CPU_ARCH_ARM)
    return &ceAllGatherConstants[ncclCeAllGatherArch_VR200];
  return nullptr;
}

bool ncclCeAllGatherModelAvailable(const struct ncclComm* comm) {
  if (ncclCeAllGatherConstantsForComm(comm) != nullptr) return true;
  // Blackwell compares its fallback model with SM kernels until its detailed fit is enabled.
  return comm->minCompCap >= 100 && !RUBIN_AND_LATER(comm->minCompCap) && comm->devrState.lsaTeamSize == comm->nRanks;
}

// Predict collective completion time in microseconds, including both barriers.
static double ncclCeAllGatherTimeUs(struct ncclComm* comm, size_t perRankBytes, bool captured, bool inPlace, int method,
                                    const ncclCeAllGatherConstants& constants) {
  const ncclCeAllGatherSyncConstants& sync = constants.sync;
  const double peerSyncUs = comm->symkState.hasLsaMultimem ? sync.multicastSyncUs : sync.unicastSyncUs;
  // Graphs reset every rank's flag; non-graph execution writes one sequence value per barrier.
  const double barrierUs =
    (comm->nRanks - 1) * peerSyncUs + (captured ? comm->nRanks * sync.flagResetUs : sync.sequenceWriteUs);
  if (method == ncclCeMethodId_AllGather_MC) {
    const ncclCeAlgoConstants& algo = constants.multicast;
    size_t copies = ncclCeMcCopyPlan::maxCopies(comm->nRanks, perRankBytes);
    // Multicast delivers to self even in-place. GB/s is 1,000 bytes per microsecond.
    return algo.setupUs + 2 * barrierUs + (copies - 1) * algo.dispatchUs +
           double(comm->nRanks) * perRankBytes / (1000 * algo.bandwidthGBps);
  }

  const ncclCeAlgoConstants& algo = constants.unicast;
  const size_t copies = comm->nRanks - int(inPlace);
  const ncclCeBatchCopyPlan copyPlan(copies, perRankBytes);
  const size_t copyOps = copies * copyPlan.rounds;
  // Extra batch chunks have an empirical scheduling cost, without additional global barriers.
  const double chunkUs =
    double(copyOps - copies) * (copies > copyPlan.syncFreq ? copies - copyPlan.syncFreq : 0) * algo.chunkUs;
  const double jumpUs = perRankBytes > algo.jumpBytesThreshold ? copies * algo.jumpUs : 0;
  return algo.setupUs + 2 * barrierUs + chunkUs + copyOps * algo.copyUs + jumpUs +
         double(comm->nRanks - 1) * perRankBytes / (1000 * algo.bandwidthGBps) +
         (inPlace ? 0 : perRankBytes / (1000 * algo.selfBandwidthGBps));
}

ncclResult_t ncclTuningCeModelSim(struct ncclTuningInput_t* const input, struct ncclTuningResult_t* const result,
                                  struct ncclTuningModelState* /*internal*/) {
  int method = result->id - NCCL_TUNING_CE_METHOD_ID_OFFSET;
  bool multicast = method == ncclCeMethodId_AllGather_MC;
  if ((method != ncclCeMethodId_AllGather_UC && method != ncclCeMethodId_AllGather_MC) ||
      (multicast && !input->comm->symkState.hasLsaMultimem)) {
    result->valid = 0;
    return ncclSuccess;
  }
  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
  size_t bytes = input->count * ncclTypeSize(input->datatype);
  const ncclCeAllGatherConstants* constants = ncclCeAllGatherConstantsForComm(input->comm);
  // Use a detailed fit when available; UC additionally needs the fitted batch API.
  if (constants && (multicast || ncclCeMemcpyBatchSupported(driverVersion, input->captured, false))) {
    result->timeUs = ncclCeAllGatherTimeUs(input->comm, bytes, input->captured, input->inPlace, method, *constants);
  } else {
    // Keep functional UC available on older CUDA versions using the fallback estimate.
    result->timeUs = ncclCeAllGatherFallbackTimeUs(input->comm, multicast, input->comm->devrState.lsaTeamSize, bytes,
                                                   input->captured, input->inPlace, driverVersion);
  }
  result->selectionTimeUs = result->timeUs;
  return ncclSuccess;
}
