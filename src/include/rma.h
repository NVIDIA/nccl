/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_RMA_H_
#define NCCL_INT_RMA_H_

#include "nccl_rma.h"

ncclResult_t ncclRmaInit(struct ncclComm* comm);
ncclResult_t ncclRmaInitFromParent(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclRmaGetDevCount(int ginPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclRmaFinalize(struct ncclComm* comm);

extern ncclRma_t ncclRmaIbProxy;

#endif
