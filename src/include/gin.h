/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_GIN_H_
#define NCCL_INT_GIN_H_

#include "nccl_gin.h"

ncclResult_t ncclGinInit(struct ncclComm* comm);
ncclResult_t ncclGinInitFromParent(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclGinGetDevCount(int ginPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclGinFinalize(struct ncclComm* comm);

extern ncclGin_t ncclGinIb;
extern ncclGin_t ncclGinIbGdaki;
extern ncclGin_t ncclGinIbProxy;

#endif
