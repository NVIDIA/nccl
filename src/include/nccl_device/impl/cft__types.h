/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CFT__TYPES_H_
#define _NCCL_DEVICE_CFT__TYPES_H_

#include "../cft.h"
#include "nccl_device/core.h"

#if __cplusplus
struct ncclCftSmem {
  alignas(16) uint64_t bar;
};

template <typename Coop>
struct ncclCft_internal {
  Coop coop;
  ncclCftSmem& cftSmem;
  uint32_t txCount;
  uint32_t phaseParity;
};

template <typename T>
struct ncclCftOpSum {};
template <typename T>
struct ncclCftOpAnd {};
template <typename T>
struct ncclCftOpXor {};
template <typename T>
struct ncclCftOpOr {};
template <typename T>
struct ncclCftOpMin {};
template <typename T>
struct ncclCftOpMax {};
#endif

#endif // _NCCL_DEVICE_CFT__TYPES_H_
