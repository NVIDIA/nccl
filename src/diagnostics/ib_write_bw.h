/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DIAGNOSTICS_IB_WRITE_BW_H_
#define NCCL_DIAGNOSTICS_IB_WRITE_BW_H_

#include "nccl.h"

struct ncclComm;

// Runs the active ib_write_bw diagnostic. The check is purely observational: all findings and
// failures are reported via DIAG_PRINT and nothing is returned, so it can never fail communicator
// initialization.
void ncclDiagRunIbWriteBw(struct ncclComm* comm);

#endif // NCCL_DIAGNOSTICS_IB_WRITE_BW_H_
