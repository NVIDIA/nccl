/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#ifndef NCCL_CONFIG_ALGORITHM_REGISTRY_H_
#define NCCL_CONFIG_ALGORITHM_REGISTRY_H_

#include "nccl.h"
#include "nccl_common.h" // ncclFunc_t
#include <stdint.h>

// One row per selectable algorithm. name is the selection key, matched as a case-insensitive
// substring. mask == 1ull << (cost-model tuning id): general in [0,21), symmetric in [21,39).
struct ncclAlgRegEntry {
  const char* name;
  uint64_t mask;
  uint32_t collMask;   // OR of (1u << ncclFunc_t)
  int16_t algo;        // NCCL_ALGO_* for general rows, else -1
  int16_t proto;       // NCCL_PROTO_* for general rows, else -1
  int16_t symkKernelId; // ncclSymkKernelId for symmetric rows, else -1
};

uint64_t ncclAlgAllBits();                           // OR of all selectable rows (parser '^' base)
uint64_t ncclAlgTagMask(const char* tag);           // case-insensitive name-prefix match; 0 == unknown
uint64_t ncclAlgValidForFuncMask(ncclFunc_t func);
const char* ncclAlgNameForGeneral(int algo, int proto); // introspection (NULL if none)
const char* ncclAlgNameForSymk(int symkKernelId);

#endif
