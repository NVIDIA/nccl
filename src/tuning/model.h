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

#endif // NCCL_TUNING_MODEL_H_
