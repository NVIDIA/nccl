/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PROGRESS_COUNTER_MONITOR_H_
#define NCCL_PROGRESS_COUNTER_MONITOR_H_

#include "nccl_common.h"

struct ncclComm;

// RAS represents local CUDA devices with a 64-bit mask.
static constexpr int kRasMaxCudaDevices = 64;

ncclResult_t ncclProgressCounterMonitorInit(struct ncclComm* comm);
ncclResult_t ncclProgressCounterMonitorDestroy(struct ncclComm* comm);

#endif
