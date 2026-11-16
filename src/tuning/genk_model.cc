/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cost_model.h"
#include "checks.h"
#include "comm.h"
#include "sym_kernels.h"

#include <algorithm>

static constexpr int genkMinChunkPayloadBytes[][NCCL_NUM_PROTOCOLS] = {
  {4 << 10, 8 << 10, 16 << 10}, // Tree {LL, LL128, Simple}
  {4 << 10, 16 << 10, 32 << 10}   // Ring {LL, LL128, Simple}
};

static constexpr int genkMinBytesPerThread[NCCL_NUM_PROTOCOLS] = {8, 15, 16};

static ncclResult_t ncclTuningGenkGetChannels(struct ncclTuningInput_t* const input,
                                              struct ncclTuningResult_t* const result) {
  int nc = input->comm->nChannels;
  int nt = ncclSymkMaxThreads;
  result->minChunkPayloadBytes = genkMinChunkPayloadBytes[result->algo][result->proto];

  // Ring AllReduce divides each CTA's work into one chunk per rank. Other
  // general-kernel collectives divide the payload only between CTAs.
  size_t const chunkMultiplicity =
    result->algo == NCCL_ALGO_RING && input->func == ncclFuncAllReduce ? input->comm->nRanks : 1;
  size_t const channelMinBytes = (size_t)result->minChunkPayloadBytes * chunkMultiplicity;

  // Do not launch channels whose first chunk would be empty.
  while (nc >= 2 && input->nBytes <= (size_t)(nc - 1) * channelMinBytes) nc--;

  if (result->maxChannels > 0 && result->maxChannels < nc) {
    nc = result->maxChannels;
  } else {
    result->maxChannels = nc;
  }
  nc = std::max(nc, input->minCTAs);
  nc = std::min(nc, input->maxCTAs);
  // A per-call minimum cannot select more channels than were allocated.
  nc = std::min(nc, std::min(input->comm->nChannels, input->comm->config.maxCTAs));

  // Once channel parallelism is exhausted, reduce workers for small chunks.
  size_t const bytesPerThread = (size_t)genkMinBytesPerThread[result->proto] * chunkMultiplicity;
  while (nt > 2 * WARP_SIZE && input->nBytes <= (size_t)(nt - WARP_SIZE) * bytesPerThread * nc) nt -= WARP_SIZE;

  result->nChannels = nc;
  result->nWarps = nt / WARP_SIZE;
  TRACE(NCCL_TUNING, "nChannels: %d, nWarps: %d", result->nChannels, result->nWarps);
  return ncclSuccess;
}

static ncclResult_t ncclGenkGetAlgorithmProtocol(enum ncclSymkKernelId kernelId, int* algorithm, int* protocol) {
  if ((ncclGenkKernelMask() >> kernelId & 1) == 0) return ncclInternalError;

  bool const isTree = kernelId == ncclSymkKernelId_AllReduce_Tree_Simple ||
                      kernelId == ncclSymkKernelId_AllReduce_Tree_LL ||
                      kernelId == ncclSymkKernelId_AllReduce_Tree_LL128;
  *algorithm = isTree ? NCCL_ALGO_TREE : NCCL_ALGO_RING;

  switch (kernelId) {
  case ncclSymkKernelId_AllReduce_Ring_Simple:
  case ncclSymkKernelId_AllReduce_Tree_Simple:
  case ncclSymkKernelId_AllGather_Ring_Simple:
  case ncclSymkKernelId_ReduceScatter_Ring_Simple:
  case ncclSymkKernelId_Broadcast_Ring_Simple:
  case ncclSymkKernelId_Reduce_Ring_Simple:
    *protocol = NCCL_PROTO_SIMPLE;
    break;
  case ncclSymkKernelId_AllReduce_Ring_LL:
  case ncclSymkKernelId_AllReduce_Tree_LL:
  case ncclSymkKernelId_AllGather_Ring_LL:
  case ncclSymkKernelId_ReduceScatter_Ring_LL:
  case ncclSymkKernelId_Broadcast_Ring_LL:
  case ncclSymkKernelId_Reduce_Ring_LL:
    *protocol = NCCL_PROTO_LL;
    break;
  case ncclSymkKernelId_AllReduce_Ring_LL128:
  case ncclSymkKernelId_AllReduce_Tree_LL128:
  case ncclSymkKernelId_AllGather_Ring_LL128:
  case ncclSymkKernelId_ReduceScatter_Ring_LL128:
  case ncclSymkKernelId_Broadcast_Ring_LL128:
  case ncclSymkKernelId_Reduce_Ring_LL128:
    *protocol = NCCL_PROTO_LL128;
    break;
  default:
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTuningGenkModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning) {
  int algorithm;
  int protocol;
  NCCLCHECK(ncclGenkGetAlgorithmProtocol((enum ncclSymkKernelId)tuning->symKernelId, &algorithm, &protocol));

  struct ncclTuningResult_t generalTuning = *tuning;
  generalTuning.algo = algorithm;
  generalTuning.proto = protocol;
  generalTuning.symKernelId = ncclSymkKernelId_Count;
  if (algorithm == NCCL_ALGO_TREE) {
    NCCLCHECK(ncclTuningTreeModelSim(inputs, &generalTuning, nullptr));
  } else {
    NCCLCHECK(ncclTuningRingModelSim(inputs, &generalTuning, nullptr));
  }
  if (!generalTuning.valid) {
    tuning->valid = 0;
    tuning->timeUs = generalTuning.timeUs;
    return ncclSuccess;
  }

  NCCLCHECK(ncclTuningGenkGetChannels(inputs, &generalTuning));
  tuning->timeUs = generalTuning.timeUs;
  tuning->nChannels = generalTuning.nChannels;
  tuning->maxChannels = generalTuning.maxChannels;
  tuning->minChunkPayloadBytes = generalTuning.minChunkPayloadBytes;
  int const nParts = algorithm == NCCL_ALGO_TREE ? ncclGenkTreeParts : 1;
  int const signalWarpsPerPart =
    protocol == NCCL_PROTO_LL128 ? NCCL_FLOW_LL128_SIGNAL_WARPS_PER_PART : NCCL_FLOW_SIGNAL_WARPS_PER_PART;
  int maxWorkerWarps = algorithm == NCCL_ALGO_TREE ? ncclGenkTreeMaxWorkerWarps : ncclGenkRingMaxWorkerWarps;
  // Fit LL128's workers and signaling warps within its 640-thread limit.
  if (protocol == NCCL_PROTO_LL128) {
    maxWorkerWarps = NCCL_LL128_MAX_NTHREADS / WARP_SIZE - nParts * signalWarpsPerPart;
  }
  int nWorkerWarps = generalTuning.nWarps;
  if (algorithm == NCCL_ALGO_TREE) {
    nWorkerWarps = std::max(nWorkerWarps, 2 * nParts);
    nWorkerWarps = std::min(nWorkerWarps, maxWorkerWarps);
  } else {
    nWorkerWarps = std::min(nWorkerWarps, maxWorkerWarps);
  }
  tuning->nWarps = nWorkerWarps + nParts * signalWarpsPerPart;
  return ncclSuccess;
}
