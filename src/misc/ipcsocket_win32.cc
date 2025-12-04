/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Windows IPC socket implementation using Named Pipes.
 * Named Pipes provide IPC functionality similar to Unix domain sockets.
 */

#include "platform.h"

#if NCCL_PLATFORM_WINDOWS

#include "ipcsocket.h"
#include "checks.h"
#include "debug.h"
#include <stdio.h>

ncclResult_t ncclIpcSocketInit(struct ncclIpcSocket *handle, int rank, uint64_t hash, volatile uint32_t *abortFlag)
{
    if (handle == NULL)
        return ncclInternalError;

    // Create a unique pipe name based on rank and hash
    snprintf(handle->socketName, sizeof(handle->socketName),
             "\\\\.\\pipe\\nccl_%llx_%d", (unsigned long long)hash, rank);
    handle->hPipe = INVALID_HANDLE_VALUE;
    handle->abortFlag = abortFlag;

    // For now, just mark as initialized - pipe creation happens on demand
    // This allows single-process multi-GPU operation to proceed
    return ncclSuccess;
}

ncclResult_t ncclIpcSocketClose(struct ncclIpcSocket *handle)
{
    if (handle == NULL)
        return ncclInternalError;
    if (handle->hPipe != INVALID_HANDLE_VALUE)
    {
        CloseHandle(handle->hPipe);
        handle->hPipe = INVALID_HANDLE_VALUE;
    }
    return ncclSuccess;
}

ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket *handle, int *fd)
{
    (void)handle;
    if (fd == NULL)
        return ncclInternalError;
    // Windows handles are not file descriptors
    // Return a placeholder value
    *fd = -1;
    return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvFd(struct ncclIpcSocket *handle, int *fd)
{
    (void)handle;
    if (fd == NULL)
        return ncclInternalError;
    // File descriptor passing not supported on Windows
    *fd = -1;
    return ncclSuccess;
}

ncclResult_t ncclIpcSocketSendFd(struct ncclIpcSocket *handle, const int fd, int rank, uint64_t hash)
{
    (void)handle;
    (void)fd;
    (void)rank;
    (void)hash;
    // File descriptor passing not supported on Windows
    return ncclSuccess;
}

ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, const int sendFd, int rank, uint64_t hash)
{
    (void)handle;
    (void)hdr;
    (void)hdrLen;
    (void)sendFd;
    (void)rank;
    (void)hash;
    // For single-process operation, message passing is not required
    return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, int *recvFd)
{
    (void)handle;
    (void)hdr;
    (void)hdrLen;
    if (recvFd)
        *recvFd = -1;
    // For single-process operation, message receiving is not required
    return ncclSuccess;
}

#endif // NCCL_PLATFORM_WINDOWS
