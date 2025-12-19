/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_H_
#define NCCL_H_

/* Minimal nccl.h for Windows compilation testing */

#include "platform/platform.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/* Version macros */
#define NCCL_MAJOR 2
#define NCCL_MINOR 25
#define NCCL_PATCH 0
#define NCCL_SUFFIX ""
#define NCCL_VERSION_CODE 22500
#define NCCL_VERSION(X, Y, Z) (((X) * 10000) + ((Y) * 100) + (Z))

    /* Basic types */
    typedef struct ncclComm *ncclComm_t;

/* CUDA types - stubbed for Windows testing without CUDA */
#ifndef __CUDACC__
    typedef void *cudaStream_t;
    typedef int cudaError_t;
#define cudaSuccess 0
#endif

    /* Error codes */
    typedef enum
    {
        ncclSuccess = 0,
        ncclUnhandledCudaError = 1,
        ncclSystemError = 2,
        ncclInternalError = 3,
        ncclInvalidArgument = 4,
        ncclInvalidUsage = 5,
        ncclRemoteError = 6,
        ncclInProgress = 7,
        ncclNumResults = 8
    } ncclResult_t;

    /* Data types */
    typedef enum
    {
        ncclInt8 = 0,
        ncclChar = 0,
        ncclUint8 = 1,
        ncclInt32 = 2,
        ncclInt = 2,
        ncclUint32 = 3,
        ncclInt64 = 4,
        ncclUint64 = 5,
        ncclFloat16 = 6,
        ncclHalf = 6,
        ncclFloat32 = 7,
        ncclFloat = 7,
        ncclFloat64 = 8,
        ncclDouble = 8,
        ncclBfloat16 = 9,
        ncclNumTypes = 10
    } ncclDataType_t;

    /* Reduction ops */
    typedef enum
    {
        ncclSum = 0,
        ncclProd = 1,
        ncclMax = 2,
        ncclMin = 3,
        ncclAvg = 4,
        ncclNumOps = 5
    } ncclRedOp_t;

/* Unique ID for communicator setup */
#define NCCL_UNIQUE_ID_BYTES 128
    typedef struct
    {
        char internal[NCCL_UNIQUE_ID_BYTES];
    } ncclUniqueId;

    /* Configuration */
    typedef struct ncclConfig_v21700
    {
        size_t size;
        unsigned int magic;
        unsigned int version;
        int blocking;
        int cgaClusterSize;
        int minCTAs;
        int maxCTAs;
        int netName[8];
        int splitShare;
    } ncclConfig_t;

#define NCCL_CONFIG_INITIALIZER {                     \
    sizeof(ncclConfig_t),                             \
    0xcafebeef,                                       \
    NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH), \
    1,   /* blocking */                               \
    0,   /* cgaClusterSize */                         \
    1,   /* minCTAs */                                \
    0,   /* maxCTAs */                                \
    {0}, /* netName */                                \
    0    /* splitShare */                             \
}

    /* Function declarations (stubs for Windows testing) */
    const char *ncclGetErrorString(ncclResult_t result);
    ncclResult_t ncclGetVersion(int *version);
    ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId);
    ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank);
    ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist);
    ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t *config);
    ncclResult_t ncclCommFinalize(ncclComm_t comm);
    ncclResult_t ncclCommDestroy(ncclComm_t comm);
    ncclResult_t ncclCommAbort(ncclComm_t comm);
    ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config);
    const char *ncclGetLastError(ncclComm_t comm);
    ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);
    ncclResult_t ncclCommCount(const ncclComm_t comm, int *count);
    ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device);
    ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank);
    ncclResult_t ncclMemAlloc(void **ptr, size_t size);
    ncclResult_t ncclMemFree(void *ptr);

    /* Collective operations */
    ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream);
    ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                               ncclDataType_t datatype, int root, ncclComm_t comm,
                               cudaStream_t stream);
    ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op, int root,
                            ncclComm_t comm, cudaStream_t stream);
    ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                               ncclDataType_t datatype, ncclComm_t comm,
                               cudaStream_t stream);
    ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                                   ncclDataType_t datatype, ncclRedOp_t op,
                                   ncclComm_t comm, cudaStream_t stream);

    /* Point-to-point */
    ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype,
                          int peer, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype,
                          int peer, ncclComm_t comm, cudaStream_t stream);

    /* Group semantics */
    ncclResult_t ncclGroupStart();
    ncclResult_t ncclGroupEnd();

#ifdef __cplusplus
}
#endif

#endif /* NCCL_H_ */
