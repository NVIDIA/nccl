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

extern ncclGin_t ncclGinIbGdaki;

ncclResult_t ncclGinIbGdakiCreateContextGroup(void** collComms, int nComms, ncclGinConfig_t* config,
                                              const uint8_t* remoteConnByPeer, void** ginCtxs,
                                              ncclNetDeviceHandle_t** devHandles);
ncclResult_t ncclGinIbGdakiRegMrSymGroup(void** collComms, int nComms, const uint8_t* remoteConnByPeer, void* data,
                                         size_t size, int type, uint64_t mrFlags,
                                         void** mhandles, void** ginHandles);

#endif
