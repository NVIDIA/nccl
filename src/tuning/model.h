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

#endif // NCCL_TUNING_MODEL_H_
