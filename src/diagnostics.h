/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DIAGNOSTICS_H_
#define NCCL_DIAGNOSTICS_H_

#include "nccl.h"

ncclResult_t ncclRunDiagnostics(ncclComm* comm);

#if defined(NCCL_OS_LINUX)

// Run one external tool to completion for a diagnostics check. `command` is a tool
// invocation (e.g. "ib_write_bw -d mlx5_0 ..."); it will run with a timeout and get killed afterwards.
// If provided, `outputTruncated` reports whether output exceeded the caller's buffer.
int ncclDiagChildRun(const char* command, int timeoutSec, char* output, int outputSize,
                     bool* outputTruncated = nullptr);

#endif // NCCL_OS_LINUX

#endif // NCCL_DIAGNOSTICS_H_
