/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PROFILER_NET_H_
#define PROFILER_NET_H_

#define NCCL_PROFILER_NET_VER_BITS  (16)
#define NCCL_PROFILER_NET_VER_MASK  (~0U >> NCCL_PROFILER_NET_VER_BITS)
#define NCCL_PROFILER_NET_TYPE_MASK (~0U << NCCL_PROFILER_NET_VER_BITS)

typedef enum {
  NCCL_PROFILER_NET_TYPE_IB   = (1U << NCCL_PROFILER_NET_VER_BITS),
  NCCL_PROFILER_NET_TYPE_SOCK = (2U << NCCL_PROFILER_NET_VER_BITS),
} ncclProfilerNetType;

#include "net_ib_v1.h"
#include "net_socket_v1.h"

#endif
