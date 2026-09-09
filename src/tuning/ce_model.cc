/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "ce_coll.h"
#include "tuning.h"
#include "cost_model.h"
#include "cudawrap.h"
#include <cstring>
#include <cstdint>

// Latency floor (us); sm_100 row is pre-fix (post-fix handled by the bw cap below).
// GB200/coherent reuses these sm_100 (B200) floors: conservative, may under-pick MC at high ranks.
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
  // [uc/mc][captured: no-graph, graph]
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

static float ncclCeAgLatencyUs(struct ncclComm* comm, bool multicast, int nRanks, size_t perRankBytes, int captured,
                               int inPlace) {
  int arch = ncclCeArchBucket(comm);
  int capturedIdx = captured ? CE_GRAPH : CE_NOGRAPH;
  const struct ncclCeAgFloorParams f = CE_AG_FLOOR[arch][multicast ? 1 : 0][capturedIdx];
  float floorUs = f.base + f.perRank * (float)nRanks;

  // D2H effects (MC bw cap + OOP-unicast slice) only apply on non-coherent CPUs; skip on coherent.
  bool coherent = (comm->cpuArch == NCCL_TOPO_CPU_ARCH_ARM);

  // Pre-fix driver caps multicast D2H bw at min(d2hBw, nvlinkBw/N)*N.
  float nvlinkBw = CE_NVLINK_BW[arch];
  float bw = nvlinkBw;
  if (multicast && !coherent) {
    int driver = 0;
    (void)ncclCudaDriverVersion(&driver);
    if (driver < CE_BLACKWELL_FIX_DRIVER) {
      float d2hBw = ncclCeD2hBw(comm);
      float perLink = nvlinkBw / (float)nRanks;
      float lo = (d2hBw < perLink) ? d2hBw : perLink;
      bw = lo * (float)nRanks;
    }
  }

  double dataBytes = (double)perRankBytes * (multicast ? nRanks : (nRanks - 1));
  float bwUs = (bw > 0.0f) ? (float)(dataBytes / bw / 1e3) : 0.0f;
  float total = floorUs + bwUs;

  // Out-of-place unicast (no graph) also stages one slice over D2H; lower-bound by that.
  if (!multicast && !inPlace && !captured && !coherent) {
    float d2hBw = ncclCeD2hBw(comm);
    if (d2hBw > 0.0f) {
      float altUs = floorUs + (float)((double)perRankBytes / d2hBw / 1e3);
      if (altUs > total) total = altUs;
    }
  }

  // Graph-captured unicast also pays the intra-batch sync latency; add it.
  if (!multicast && captured) {
    int numOps = inPlace ? (nRanks - 1) : nRanks;
    int freq = (int)comm->ceColl.intraBatchSyncFreq;
    if (numOps > freq && (uint64_t)perRankBytes * (uint64_t)numOps >= comm->ceColl.intraBatchSyncMsgThreshold) {
      int blocks = (numOps + freq - 1) / freq;
      int numSyncs = (blocks - 1) + ((blocks % 2 == 0) ? 1 : 0); // per-freq syncs + even-count workaround
      total += (float)numSyncs * floorUs;
    }
  }
  return total;
}

ncclResult_t ncclTuningCeModelSim(struct ncclTuningInput_t* const input, struct ncclTuningResult_t* const result) {
  int method = result->id - NCCL_TUNING_CE_METHOD_ID_OFFSET;
  bool multicast = (method == ncclCeMethodId_AllGather_MC);
  int nRanks = input->comm->devrState.lsaSize;
  result->timeUs = ncclCeAgLatencyUs(input->comm, multicast, nRanks, input->nBytes, input->captured, input->inPlace);
  return ncclSuccess;
}
