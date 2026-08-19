/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "lsa_a2a.h"

#include "comm.h"
#include "core.h"

#include <algorithm>
#include <cmath>

static constexpr double gbpsToBytesPerUs = 1.e9 / 1.e6;
static constexpr size_t allGatherA2AMinAggregateBytes = 256 * 1024;

// Multimem AllGather CTA bandwidth is the measured store injection rate before
// multicast fanout; peak bandwidth is aggregate traffic.
const struct ncclSymkLsaA2AParameters
  ncclSymkLsaA2AParameterTable[NCCL_SYMK_LSA_A2A_ARCH_COUNT][ncclSymkKernelId_Count] = {
    // SM100
    {
      {}, // AllReduce_AGxLL_R
      {}, // AllReduce_AGxLLMC_R
      {}, // AllReduce_RSxTmaLD_AGxTmaST
      {}, // AllReduce_RSxLD_AGxST
      {}, // AllReduce_RSxLDMC_AGxSTMC
      {}, // AllGather_LL
      {}, // AllGather_LLMC
      {true, 10.061860371492349, 0.067985251913265293, 64.15772392585157, 671.60525655802655, false}, // AllGather_TmaST
      {true, 10.590215893817202, 0.056361727150537638, 64.48109855086463, 650.43780647742426, false}, // AllGather_ST
      {true, 8.2723873901367178, 0.062372589111328119, 51.55, 715.14513870274175, true}, // AllGather_TmaSTMC
      {true, 8.3313720703125007, 0.056103515625000003, 50.83, 715.14513870274175, true}, // AllGather_STMC
      {}, // AllGather_RailRing_LsaSTMC
      {}, // ReduceScatter_LL
      {}, // ReduceScatter_TmaLD
      {}, // ReduceScatter_LD
      {}, // ReduceScatter_LDMC
      {}, // ReduceScatter_RailA2A_LsaLD
      {}, // ReduceScatter_RailA2A_LsaLDMC
    },
    // SM103
    {
      {}, // AllReduce_AGxLL_R
      {}, // AllReduce_AGxLLMC_R
      {}, // AllReduce_RSxTmaLD_AGxTmaST
      {}, // AllReduce_RSxLD_AGxST
      {}, // AllReduce_RSxLDMC_AGxSTMC
      {}, // AllGather_LL
      {}, // AllGather_LLMC
      {true, 10.061860371492349, 0.067985251913265293, 64.15772392585157, 671.60525655802655, false}, // AllGather_TmaST
      {true, 10.590215893817202, 0.056361727150537638, 64.48109855086463, 650.43780647742426, false}, // AllGather_ST
      {true, 8.2723873901367178, 0.062372589111328119, 51.55, 715.14513870274175, true}, // AllGather_TmaSTMC
      {true, 8.3313720703125007, 0.056103515625000003, 50.83, 715.14513870274175, true}, // AllGather_STMC
      {}, // AllGather_RailRing_LsaSTMC
      {}, // ReduceScatter_LL
      {}, // ReduceScatter_TmaLD
      {}, // ReduceScatter_LD
      {}, // ReduceScatter_LDMC
      {}, // ReduceScatter_RailA2A_LsaLD
      {}, // ReduceScatter_RailA2A_LsaLDMC
    },
};

static inline int ncclSymkLsaA2AArchBucket(struct ncclComm* comm) {
  if (comm->minCompCap == 100) return NCCL_SYMK_LSA_A2A_ARCH_SM100;
  if (comm->minCompCap == 103) return NCCL_SYMK_LSA_A2A_ARCH_SM103;
  return -1;
}

// AllGather ST and STMC use forEachWork<char>. For the current single-work
// model, the scheduler assigns ceil(cells / requestedCtas) cells per CTA and
// leaves the tail of the requested grid idle.
static int activeCtasForEachWork(size_t logicalBytes, int requestedCtas) {
  size_t cells = divUp(logicalBytes, size_t(NCCL_SYM_KERNEL_CELL_SIZE));
  size_t cellsPerCta = divUp(cells, size_t(requestedCtas));
  return (int)divUp(cells, cellsPerCta);
}

static const struct ncclSymkLsaA2AParameters* parametersFor(const struct ncclTuningInput_t* input,
                                                            enum ncclSymkKernelId kernelId) {
  int kernel = static_cast<int>(kernelId);
  int arch = input == nullptr || input->comm == nullptr ? -1 : ncclSymkLsaA2AArchBucket(input->comm);
  if (input == nullptr || input->comm == nullptr || kernel < 0 || kernel >= ncclSymkKernelId_Count || arch < 0 ||
      !ncclSymkLsaA2AParameterTable[arch][kernel].valid) {
    return nullptr;
  }

  int nRanks = input->comm->nRanks;
  size_t logicalBytes = input->count * ncclTypeSize(input->datatype);
  return nRanks >= 2 && input->nWorks == 1 && logicalBytes >= divUp(allGatherA2AMinAggregateBytes, size_t(nRanks)) ?
           &ncclSymkLsaA2AParameterTable[arch][kernel] :
           nullptr;
}

bool ncclSymkLsaA2AModel(const struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, int requestedCtas,
                         float* timeUs, int* activeCtas, const struct ncclSymkLsaA2AParameters* parameters) {
  const struct ncclSymkLsaA2AParameters* tableParameters = parametersFor(input, kernelId);
  if (tableParameters == nullptr || requestedCtas < 1) return false;
  if (parameters == nullptr) parameters = tableParameters;
  if (!parameters->valid) return false;

  size_t logicalBytes = input->count * ncclTypeSize(input->datatype);
  bool isMulticast = kernelId == ncclSymkKernelId_AllGather_TmaSTMC || kernelId == ncclSymkKernelId_AllGather_STMC;
  int nRanks = input->comm->nRanks;

  int modeledCtas = activeCtasForEachWork(logicalBytes, requestedCtas);
  double latencyUs = parameters->baseLatUs + (nRanks - 1) * parameters->rankLatUs;
  double rankFactor = parameters->peakRankEfficiency ? (double)(nRanks - 1) / nRanks : 1.0;
  double ctaCopies = isMulticast ? 1.0 : nRanks - (input->inPlace != 0);
  double ctaTimeUs = ctaCopies * logicalBytes / (modeledCtas * parameters->ctaBwGBps * gbpsToBytesPerUs);
  double peakTimeUs = (nRanks - 1) * logicalBytes / (parameters->peakBwGBps * rankFactor * gbpsToBytesPerUs);
  double estimateUs = latencyUs + std::max(ctaTimeUs, peakTimeUs);
  if (!std::isfinite(estimateUs) || !(estimateUs > 0.0)) return false;

  *timeUs = static_cast<float>(estimateUs);
  *activeCtas = modeledCtas;
  return true;
}
