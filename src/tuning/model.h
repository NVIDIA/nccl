/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TUNING_MODEL_H_
#define NCCL_TUNING_MODEL_H_

#include "cost_model.h"
#include "comm.h"
#include "nccl_tuner.h"

#include <algorithm>
#include <float.h>
#include <cmath>

inline double ncclSoftMin(double x, double ceiling, double softness) {
  // looks like a smooth version of: min(x, ceiling)
  return ceiling - softness * std::log1p((std::exp(ceiling / softness) - 1) * std::exp(-x / softness));
}

inline double ncclSoftPlus(double x, double softness) {
  // looks like a smooth version of: max(0, x)
  double z = x / softness;
  return 100.0 <= z ? x : softness * std::log1p(std::exp(z));
}

// NVLS efficiency factor.
static const float nvlsEfficiency[NCCL_NUM_COMPCAPS] = {
  0.0f, // Volta
  0.0f, // Ampere
  0.85f, // Hopper
  0.74f, // Blackwell
  0.87f, // Rubin
};

inline float ncclTuningProtoBWFactor(int proto) {
  return (proto == NCCL_PROTO_LL) ? 0.5f : (proto == NCCL_PROTO_LL128) ? 120.0f / 128.0f : 1.0f;
}

// bwPerCTA is the expected per CTA bandwidth of the algo/proto for the model.
// This value is clamped to the tuning constant if one if defined and is positive and non-zero.
inline float ncclTuningProtoBW(struct ncclComm* comm, int algo, int proto, float bwPerCTA) {
  int index, compCapIndex;
  ncclTuningGetConstantsIndexes(comm, nullptr, &index);
  compCapIndex = ncclTuningGetCompCapIndex(comm);
  double (*constants)[NCCL_NUM_TUNING_SCALES] = nullptr;
  if (proto == NCCL_PROTO_LL && (algo == NCCL_ALGO_RING || algo == NCCL_ALGO_TREE)) {
    constants = comm->tuningContext.tuningConstants.llMaxBws;
  } else if (proto == NCCL_PROTO_LL128) {
    switch (algo) {
    case NCCL_ALGO_RING:
      constants = comm->tuningContext.tuningConstants.perChMaxRingLL128Bws;
      break;
    case NCCL_ALGO_TREE:
      constants = comm->tuningContext.tuningConstants.perChMaxTreeLL128Bws;
      break;
    default:
      constants = nullptr;
      break;
    }
  } else if (proto == NCCL_PROTO_SIMPLE) {
    switch (algo) {
    case NCCL_ALGO_TREE:
      constants = comm->tuningContext.tuningConstants.perChMaxTreeBws;
      break;
    case NCCL_ALGO_NVLS_TREE:
      constants = comm->tuningContext.tuningConstants.perChMaxNVLSTreeBws;
      break;
    default:
      constants = nullptr;
      break;
    }
  }
  float clamp = FLT_MAX;
  if (constants != nullptr && constants[compCapIndex][index] > 0) clamp = constants[compCapIndex][index];
  bwPerCTA = bwPerCTA * ncclTuningProtoBWFactor(proto);
  TRACE(NCCL_TUNING, "a/p: %s/%s, bwPerCTA: %f, clamp: %f", ncclAlgoToString(algo), ncclProtoToString(proto), bwPerCTA,
        clamp);
  return std::min(bwPerCTA, clamp);
}

#endif // NCCL_TUNING_MODEL_H_
