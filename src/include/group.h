/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GROUP_H_
#define NCCL_GROUP_H_

#include "nccl.h"
#include "core.h"

bool ncclAsyncMode();
ncclResult_t ncclAsyncErrCheck(ncclResult_t ret);

typedef ncclResult_t(*ncclInitFunc_t)(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, bool, bool);

ncclResult_t ncclAsyncInit(ncclInitFunc_t func, int cudaDev, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, bool initSharp, bool initRing);

typedef ncclResult_t(*ncclCollFunc_t)(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclAsyncColl(ncclComm_t comm);
#endif
