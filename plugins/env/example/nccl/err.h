/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef ERR_H_
#define ERR_H_

// NCCL error codes
#define ncclSuccess 0
#define ncclSystemError 1
#define ncclInternalError 2
#define ncclInvalidUsage 3
#define ncclInvalidArgument 4
#define ncclUnhandledCudaError 5

typedef int ncclResult_t;

#endif
