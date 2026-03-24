/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* IPC via Unix domain sockets and fd passing is not supported on Windows.
 * Provide stubs so the API compiles; callers get ncclInternalError when using IPC. */

#include "ipcsocket.h"
#include "utils.h"
#include "os.h"
#include <stdlib.h>
#include <string.h>

// Windows implementation using Named Pipes
#include <windows.h>
#include <process.h>

#define NCCL_IPC_PIPENAME_STR "\\\\.\\pipe\\nccl-socket-%d-%llx"
#define PIPE_BUFFER_SIZE 4096
#define PIPE_TIMEOUT_MS 5000

// Structure to send handle and optional header data
struct ncclIpcMsg {
  int hdrLen;
  HANDLE sourceProcess;  // Source process handle for DuplicateHandle
  HANDLE handleToDup;    // Handle to duplicate
  char hdrData[PIPE_BUFFER_SIZE - sizeof(int) - sizeof(HANDLE) * 2];
};

ncclResult_t ncclIpcSocketInit(ncclIpcSocket *handle, int rank, uint64_t hash, volatile uint32_t* abortFlag) {
  if (handle == NULL) {
    return ncclInternalError;
  }

  handle->fd = NCCL_INVALID_SOCKET;
  handle->socketName[0] = '\0';
  handle->abortFlag = abortFlag;

  // Create unique name for the pipe
  char pipeName[NCCL_IPC_SOCKNAME_LEN];
  int len = snprintf(pipeName, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_PIPENAME_STR, rank, hash);
  if (len >= NCCL_IPC_SOCKNAME_LEN) {
    WARN("IPC: Pipe name too long");
    return ncclInternalError;
  }

  TRACE(NCCL_INIT|NCCL_P2P, "IPC: Creating named pipe %s", pipeName);

  // Create named pipe for receiving
  HANDLE hPipe = CreateNamedPipeA(
    pipeName,
    PIPE_ACCESS_DUPLEX,       // Read/write access
    PIPE_TYPE_MESSAGE |       // Message type pipe
    PIPE_READMODE_MESSAGE |   // Message-read mode
    PIPE_WAIT,                // Blocking mode
    PIPE_UNLIMITED_INSTANCES, // Max instances
    PIPE_BUFFER_SIZE,         // Output buffer size
    PIPE_BUFFER_SIZE,         // Input buffer size
    PIPE_TIMEOUT_MS,          // Client timeout
    NULL                      // Default security
  );

  if (hPipe == INVALID_HANDLE_VALUE) {
    WARN("IPC: Failed to create named pipe %s: error %d", pipeName, GetLastError());
    return ncclSystemError;
  }

  // Store pipe handle as fd (cast HANDLE to intptr_t to int)
  handle->fd = (int)hPipe;
  strncpy(handle->socketName, pipeName, NCCL_IPC_SOCKNAME_LEN);
  handle->socketName[NCCL_IPC_SOCKNAME_LEN - 1] = '\0';

  TRACE(NCCL_INIT|NCCL_P2P, "IPC: Named pipe created successfully: %s (handle: %p)", pipeName, hPipe);
  return ncclSuccess;
}

ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* handle, int* fd) {
  if (handle == NULL) {
    WARN("ncclIpcSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (fd) *fd = (int)handle->fd;
  return ncclSuccess;
}

ncclResult_t ncclIpcSocketClose(ncclIpcSocket *handle) {
  if (handle == NULL) {
    return ncclInternalError;
  }
  if (handle->fd != NCCL_INVALID_SOCKET) {
    HANDLE hPipe = (HANDLE)(intptr_t)handle->fd;
    DisconnectNamedPipe(hPipe);
    CloseHandle(hPipe);
    handle->fd = NCCL_INVALID_SOCKET;
  }
  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, int *recvFd) {
  if (handle == NULL || handle->fd == NCCL_INVALID_SOCKET) {
    WARN("IPC: Invalid socket handle");
    return ncclInvalidArgument;
  }

  HANDLE hPipe = (HANDLE)(intptr_t)handle->fd;
  struct ncclIpcMsg msg;
  DWORD bytesRead = 0;
  BOOL success = FALSE;

  TRACE(NCCL_INIT|NCCL_P2P, "IPC: Waiting for connection on pipe %s", handle->socketName);

  // Wait for client to connect
  BOOL connected = ConnectNamedPipe(hPipe, NULL);
  if (!connected && GetLastError() != ERROR_PIPE_CONNECTED) {
    WARN("IPC: Failed to connect named pipe: error %d", GetLastError());
    return ncclSystemError;
  }

  TRACE(NCCL_INIT|NCCL_P2P, "IPC: Client connected, reading message");

  // Read message with retry logic for abort flag
  while (TRUE) {
    success = ReadFile(hPipe, &msg, sizeof(msg), &bytesRead, NULL);

    if (success && bytesRead > 0) {
      break;  // Successfully read data
    }

    DWORD error = GetLastError();
    if (error == ERROR_NO_DATA || error == ERROR_BROKEN_PIPE) {
      // Check abort flag
      if (handle->abortFlag && COMPILER_ATOMIC_LOAD(handle->abortFlag, std::memory_order_acquire)) {
        WARN("IPC: Operation aborted");
        return ncclInternalError;
      }
      continue;
    }

    WARN("IPC: ReadFile failed: error %d", error);
    return ncclSystemError;
  }

  TRACE(NCCL_INIT|NCCL_P2P, "IPC: Read %d bytes from pipe", bytesRead);

  // Copy header data if requested
  if (hdr != NULL && msg.hdrLen > 0) {
    memcpy(hdr, msg.hdrData, (msg.hdrLen < hdrLen) ? msg.hdrLen : hdrLen);
  }

  // Duplicate handle if requested
  if (recvFd != NULL && msg.handleToDup != NULL) {
    HANDLE duplicatedHandle;
    BOOL dupSuccess = DuplicateHandle(
      msg.sourceProcess,      // Source process
      msg.handleToDup,        // Source handle
      GetCurrentProcess(),    // Target process
      &duplicatedHandle,      // Target handle
      0,                      // Desired access (same as source)
      FALSE,                  // Not inheritable
      DUPLICATE_SAME_ACCESS   // Same access as source
    );

    if (!dupSuccess) {
      WARN("IPC: DuplicateHandle failed: error %d", GetLastError());
      return ncclSystemError;
    }

    *recvFd = (int)duplicatedHandle;
    TRACE(NCCL_INIT|NCCL_P2P, "IPC: Duplicated handle %p -> %p", msg.handleToDup, duplicatedHandle);
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvFd(ncclIpcSocket *handle, int *fd) {
  return ncclIpcSocketRecvMsg(handle, NULL, 0, fd);
}

ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, const int sendFd, int rank, uint64_t hash) {
  // Construct target pipe name
  char pipeName[NCCL_IPC_SOCKNAME_LEN];
  int len = snprintf(pipeName, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_PIPENAME_STR, rank, hash);
  if (len >= NCCL_IPC_SOCKNAME_LEN) {
    WARN("IPC: Pipe name too long");
    return ncclInternalError;
  }

  TRACE(NCCL_INIT|NCCL_P2P, "IPC: Connecting to pipe %s", pipeName);

  // Connect to the named pipe with retry logic (retries indefinitely until success or abort)
  HANDLE hPipe = INVALID_HANDLE_VALUE;

  while (TRUE) {
    hPipe = CreateFileA(
      pipeName,
      GENERIC_READ | GENERIC_WRITE,
      0,              // No sharing
      NULL,           // Default security
      OPEN_EXISTING,  // Opens existing pipe
      0,              // Default attributes
      NULL            // No template
    );

    if (hPipe != INVALID_HANDLE_VALUE) {
      break;  // Successfully connected
    }

    DWORD error = GetLastError();
    if (error != ERROR_PIPE_BUSY && error != ERROR_FILE_NOT_FOUND) {
      WARN("IPC: Failed to connect to pipe %s: error %d", pipeName, error);
      return ncclSystemError;
    }

    // Check abort flag
    if (handle && handle->abortFlag && COMPILER_ATOMIC_LOAD(handle->abortFlag, std::memory_order_acquire)) {
      WARN("IPC: Operation aborted");
      return ncclInternalError;
    }
  }

  TRACE(NCCL_INIT|NCCL_P2P, "IPC: Connected to pipe %s", pipeName);

  // Set pipe to message mode
  DWORD mode = PIPE_READMODE_MESSAGE;
  if (!SetNamedPipeHandleState(hPipe, &mode, NULL, NULL)) {
    WARN("IPC: SetNamedPipeHandleState failed: error %d", GetLastError());
    CloseHandle(hPipe);
    return ncclSystemError;
  }

  // Prepare message
  struct ncclIpcMsg msg;
  memset(&msg, 0, sizeof(msg));
  msg.hdrLen = hdrLen;
  msg.sourceProcess = GetCurrentProcess();
  msg.handleToDup = (sendFd != -1) ? (HANDLE)sendFd : NULL;

  if (hdr != NULL && hdrLen > 0) {
    int copyLen = (hdrLen < (int)sizeof(msg.hdrData)) ? hdrLen : (int)sizeof(msg.hdrData);
    memcpy(msg.hdrData, hdr, copyLen);
  }

  TRACE(NCCL_INIT|NCCL_P2P, "IPC: Sending message (hdrLen=%d, handle=%p)", hdrLen, msg.handleToDup);

  // Send message
  DWORD bytesWritten = 0;
  BOOL success = WriteFile(hPipe, &msg, sizeof(msg), &bytesWritten, NULL);

  if (!success || bytesWritten != sizeof(msg)) {
    WARN("IPC: WriteFile failed: error %d, wrote %d of %zu bytes", GetLastError(), bytesWritten, sizeof(msg));
    CloseHandle(hPipe);
    return ncclSystemError;
  }

  // Flush to ensure message is sent
  FlushFileBuffers(hPipe);

  TRACE(NCCL_INIT|NCCL_P2P, "IPC: Message sent successfully (%d bytes)", bytesWritten);

  CloseHandle(hPipe);
  return ncclSuccess;
}

ncclResult_t ncclIpcSocketSendFd(ncclIpcSocket *handle, const int sendFd, int rank, uint64_t hash) {
  return ncclIpcSocketSendMsg(handle, NULL, 0, sendFd, rank, hash);
}
