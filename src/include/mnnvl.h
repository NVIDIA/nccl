/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MNNVL_H_
#define NCCL_MNNVL_H_

#include "nccl.h"
#include "comm.h"

ncclResult_t ncclMnnvlCheck(struct ncclComm* comm);

#endif
