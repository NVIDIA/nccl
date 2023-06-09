/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_LIFECYCLE_H_
#define MSCCL_LIFECYCLE_H_

#include "enqueue.h"

#include "msccl/msccl_struct.h"

bool mscclEnabled();

void mscclSetIsCallerFlag();
void mscclClearIsCallerFlag();
bool mscclIsCaller();

bool mscclAvailable();

int getEnvInt(const char* env, int64_t deftVal);

ncclResult_t mscclInit(ncclComm_t comm);

ncclResult_t mscclInitKernelsForDevice(int cudaArch, size_t* maxStackSize);

ncclResult_t mscclGroupStart();

ncclResult_t mscclEnqueueCheck(
    const void* sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void* recvbuff, const size_t recvcounts[], const size_t rdispls[],
    size_t count, ncclDataType_t datatype, int root, int peer, ncclRedOp_t op,
    mscclFunc_t mscclFunc, ncclComm_t comm, cudaStream_t stream);

ncclResult_t mscclGroupEnd();

ncclResult_t mscclTeardown();

#endif
