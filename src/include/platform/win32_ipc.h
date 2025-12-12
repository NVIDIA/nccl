/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_WIN32_IPC_H_
#define NCCL_WIN32_IPC_H_

#ifdef _WIN32

#include "win32_defs.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/*
 * Windows implementation of IPC socket functionality
 * Replaces Unix Domain Sockets with Named Pipes
 *
 * Unix Domain Sockets (AF_UNIX) are used in NCCL for inter-process
 * communication within a single node. On Windows, we use Named Pipes
 * which provide similar functionality.
 */

#define NCCL_IPC_SOCKNAME_LEN 256
#define NCCL_IPC_PIPE_PREFIX "\\\\.\\pipe\\nccl_"

/* IPC handle structure */
struct ncclIpcSocketWin32
{
    HANDLE hPipe;
    char pipeName[NCCL_IPC_SOCKNAME_LEN];
    volatile uint32_t *abortFlag;
    int isServer;
    OVERLAPPED overlapped;
    int asyncMode;
};

typedef struct ncclIpcSocketWin32 ncclIpcSocket;

/*
 * Generate a unique pipe name based on rank and hash
 */
static inline void ncclIpcMakePipeName(char *name, size_t nameLen, int rank, uint64_t hash)
{
    snprintf(name, nameLen, "%s%d_%llx", NCCL_IPC_PIPE_PREFIX, rank, (unsigned long long)hash);
}

/*
 * Initialize an IPC socket (create named pipe server)
 */
static inline int ncclIpcSocketInit(ncclIpcSocket *handle, int rank, uint64_t hash,
                                    volatile uint32_t *abortFlag)
{
    if (handle == NULL)
        return -1;

    memset(handle, 0, sizeof(*handle));
    handle->hPipe = INVALID_HANDLE_VALUE;
    handle->abortFlag = abortFlag;
    handle->isServer = 1;

    /* Create unique pipe name */
    ncclIpcMakePipeName(handle->pipeName, sizeof(handle->pipeName), rank, hash);

    /* Create named pipe (server mode) */
    handle->hPipe = CreateNamedPipeA(
        handle->pipeName,
        PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
        PIPE_UNLIMITED_INSTANCES,
        4096, /* Output buffer size */
        4096, /* Input buffer size */
        0,    /* Default timeout */
        NULL  /* Default security */
    );

    if (handle->hPipe == INVALID_HANDLE_VALUE)
    {
        return -1;
    }

    /* Create event for overlapped I/O */
    handle->overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (handle->overlapped.hEvent == NULL)
    {
        CloseHandle(handle->hPipe);
        handle->hPipe = INVALID_HANDLE_VALUE;
        return -1;
    }

    return 0;
}

/*
 * Connect to an IPC socket (client mode)
 */
static inline int ncclIpcSocketConnect(ncclIpcSocket *handle, int rank, uint64_t hash,
                                       volatile uint32_t *abortFlag)
{
    char pipeName[NCCL_IPC_SOCKNAME_LEN];
    int retries = 0;
    const int maxRetries = 100;
    const int retryDelayMs = 10;

    if (handle == NULL)
        return -1;

    memset(handle, 0, sizeof(*handle));
    handle->hPipe = INVALID_HANDLE_VALUE;
    handle->abortFlag = abortFlag;
    handle->isServer = 0;

    ncclIpcMakePipeName(pipeName, sizeof(pipeName), rank, hash);
    strncpy(handle->pipeName, pipeName, sizeof(handle->pipeName) - 1);

    /* Try to connect with retries */
    while (retries < maxRetries)
    {
        if (abortFlag && *abortFlag)
        {
            return -1;
        }

        /* Wait for pipe to be available */
        if (WaitNamedPipeA(pipeName, retryDelayMs))
        {
            handle->hPipe = CreateFileA(
                pipeName,
                GENERIC_READ | GENERIC_WRITE,
                0,
                NULL,
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,
                NULL);

            if (handle->hPipe != INVALID_HANDLE_VALUE)
            {
                /* Set message mode */
                DWORD mode = PIPE_READMODE_MESSAGE;
                SetNamedPipeHandleState(handle->hPipe, &mode, NULL, NULL);

                /* Create event for overlapped I/O */
                handle->overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
                if (handle->overlapped.hEvent == NULL)
                {
                    CloseHandle(handle->hPipe);
                    handle->hPipe = INVALID_HANDLE_VALUE;
                    return -1;
                }

                return 0;
            }
        }

        retries++;
        Sleep(retryDelayMs);
    }

    return -1;
}

/*
 * Accept a connection on an IPC socket
 */
static inline int ncclIpcSocketAccept(ncclIpcSocket *handle)
{
    BOOL connected;

    if (handle == NULL || handle->hPipe == INVALID_HANDLE_VALUE)
        return -1;

    /* Start overlapped connect operation */
    connected = ConnectNamedPipe(handle->hPipe, &handle->overlapped);

    if (!connected)
    {
        DWORD err = GetLastError();

        if (err == ERROR_IO_PENDING)
        {
            /* Wait for connection with abort check */
            while (1)
            {
                DWORD waitResult = WaitForSingleObject(handle->overlapped.hEvent, 100);

                if (waitResult == WAIT_OBJECT_0)
                {
                    DWORD bytesTransferred;
                    if (GetOverlappedResult(handle->hPipe, &handle->overlapped, &bytesTransferred, FALSE))
                    {
                        return 0;
                    }
                    return -1;
                }

                if (handle->abortFlag && *handle->abortFlag)
                {
                    CancelIo(handle->hPipe);
                    return -1;
                }
            }
        }
        else if (err == ERROR_PIPE_CONNECTED)
        {
            /* Already connected */
            return 0;
        }
        else
        {
            return -1;
        }
    }

    return 0;
}

/*
 * Close an IPC socket
 */
static inline int ncclIpcSocketClose(ncclIpcSocket *handle)
{
    if (handle == NULL)
        return 0;

    if (handle->overlapped.hEvent != NULL)
    {
        CloseHandle(handle->overlapped.hEvent);
        handle->overlapped.hEvent = NULL;
    }

    if (handle->hPipe != INVALID_HANDLE_VALUE)
    {
        if (handle->isServer)
        {
            DisconnectNamedPipe(handle->hPipe);
        }
        CloseHandle(handle->hPipe);
        handle->hPipe = INVALID_HANDLE_VALUE;
    }

    return 0;
}

/*
 * Get pipe handle
 */
static inline int ncclIpcSocketGetFd(ncclIpcSocket *handle, HANDLE *hPipe)
{
    if (handle == NULL)
        return -1;
    if (hPipe)
        *hPipe = handle->hPipe;
    return 0;
}

/*
 * Send data through IPC socket
 */
static inline int ncclIpcSocketSend(ncclIpcSocket *handle, const void *data, size_t size)
{
    DWORD bytesWritten;
    OVERLAPPED ov = {0};
    DWORD err;

    if (handle == NULL || handle->hPipe == INVALID_HANDLE_VALUE)
        return -1;

    ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (ov.hEvent == NULL)
        return -1;

    if (!WriteFile(handle->hPipe, data, (DWORD)size, &bytesWritten, &ov))
    {
        err = GetLastError();
        if (err == ERROR_IO_PENDING)
        {
            /* Wait with abort checking */
            while (1)
            {
                DWORD waitResult = WaitForSingleObject(ov.hEvent, 100);

                if (waitResult == WAIT_OBJECT_0)
                {
                    if (GetOverlappedResult(handle->hPipe, &ov, &bytesWritten, FALSE))
                    {
                        CloseHandle(ov.hEvent);
                        return (bytesWritten == size) ? 0 : -1;
                    }
                    CloseHandle(ov.hEvent);
                    return -1;
                }

                if (handle->abortFlag && *handle->abortFlag)
                {
                    CancelIo(handle->hPipe);
                    CloseHandle(ov.hEvent);
                    return -1;
                }
            }
        }
        CloseHandle(ov.hEvent);
        return -1;
    }

    CloseHandle(ov.hEvent);
    return (bytesWritten == size) ? 0 : -1;
}

/*
 * Receive data from IPC socket
 */
static inline int ncclIpcSocketRecv(ncclIpcSocket *handle, void *data, size_t size)
{
    DWORD bytesRead;
    OVERLAPPED ov = {0};
    DWORD err;

    if (handle == NULL || handle->hPipe == INVALID_HANDLE_VALUE)
        return -1;

    ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (ov.hEvent == NULL)
        return -1;

    if (!ReadFile(handle->hPipe, data, (DWORD)size, &bytesRead, &ov))
    {
        err = GetLastError();
        if (err == ERROR_IO_PENDING)
        {
            /* Wait with abort checking */
            while (1)
            {
                DWORD waitResult = WaitForSingleObject(ov.hEvent, 100);

                if (waitResult == WAIT_OBJECT_0)
                {
                    if (GetOverlappedResult(handle->hPipe, &ov, &bytesRead, FALSE))
                    {
                        CloseHandle(ov.hEvent);
                        return (int)bytesRead;
                    }
                    CloseHandle(ov.hEvent);
                    return -1;
                }

                if (handle->abortFlag && *handle->abortFlag)
                {
                    CancelIo(handle->hPipe);
                    CloseHandle(ov.hEvent);
                    return -1;
                }
            }
        }
        CloseHandle(ov.hEvent);
        return -1;
    }

    CloseHandle(ov.hEvent);
    return (int)bytesRead;
}

/*
 * Send a message with a file descriptor (handle)
 * On Windows, we use DuplicateHandle to share handles between processes
 */
static inline int ncclIpcSocketSendFd(ncclIpcSocket *handle, HANDLE targetProcess,
                                      HANDLE srcHandle, int rank, uint64_t hash)
{
    struct
    {
        HANDLE duplicatedHandle;
        int rank;
        uint64_t hash;
    } msg;

    HANDLE dupHandle;

    /* Duplicate handle for target process */
    if (!DuplicateHandle(
            GetCurrentProcess(),
            srcHandle,
            targetProcess,
            &dupHandle,
            0,
            FALSE,
            DUPLICATE_SAME_ACCESS))
    {
        return -1;
    }

    msg.duplicatedHandle = dupHandle;
    msg.rank = rank;
    msg.hash = hash;

    return ncclIpcSocketSend(handle, &msg, sizeof(msg));
}

/*
 * Receive a message with a file descriptor (handle)
 */
static inline int ncclIpcSocketRecvFd(ncclIpcSocket *handle, HANDLE *recvHandle)
{
    struct
    {
        HANDLE duplicatedHandle;
        int rank;
        uint64_t hash;
    } msg;

    int ret = ncclIpcSocketRecv(handle, &msg, sizeof(msg));
    if (ret < 0)
        return ret;

    if (recvHandle)
        *recvHandle = msg.duplicatedHandle;

    return 0;
}

/*
 * Send a message with header and optionally a handle
 */
static inline int ncclIpcSocketSendMsg(ncclIpcSocket *handle, void *hdr, int hdrLen,
                                       HANDLE sendHandle, int rank, uint64_t hash)
{
    struct
    {
        int hdrLen;
        int hasHandle;
        HANDLE dupHandle;
    } header;
    int ret;

    (void)rank;
    (void)hash;

    header.hdrLen = hdrLen;
    header.hasHandle = (sendHandle != NULL && sendHandle != INVALID_HANDLE_VALUE);
    header.dupHandle = INVALID_HANDLE_VALUE;

    if (header.hasHandle)
    {
        /* For same-process communication, just pass the handle */
        header.dupHandle = sendHandle;
    }

    /* Send header */
    ret = ncclIpcSocketSend(handle, &header, sizeof(header));
    if (ret < 0)
        return ret;

    /* Send user data */
    if (hdr != NULL && hdrLen > 0)
    {
        ret = ncclIpcSocketSend(handle, hdr, hdrLen);
        if (ret < 0)
            return ret;
    }

    return 0;
}

/*
 * Receive a message with header and optionally a handle
 */
static inline int ncclIpcSocketRecvMsg(ncclIpcSocket *handle, void *hdr, int hdrLen,
                                       HANDLE *recvHandle)
{
    struct
    {
        int hdrLen;
        int hasHandle;
        HANDLE dupHandle;
    } header;
    int ret;

    /* Receive header */
    ret = ncclIpcSocketRecv(handle, &header, sizeof(header));
    if (ret < 0)
        return ret;

    /* Receive user data */
    if (hdr != NULL && hdrLen > 0 && header.hdrLen > 0)
    {
        int toRead = (header.hdrLen < hdrLen) ? header.hdrLen : hdrLen;
        ret = ncclIpcSocketRecv(handle, hdr, toRead);
        if (ret < 0)
            return ret;
    }

    if (recvHandle != NULL)
    {
        *recvHandle = header.hasHandle ? header.dupHandle : INVALID_HANDLE_VALUE;
    }

    return 0;
}

#endif /* _WIN32 */

#endif /* NCCL_WIN32_IPC_H_ */
