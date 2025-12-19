/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "ipcsocket.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>

#if NCCL_PLATFORM_LINUX

#include <errno.h>

// Enable Linux abstract socket naming
#define USE_ABSTRACT_SOCKET

#define NCCL_IPC_SOCKNAME_STR "/tmp/nccl-socket-%d-%lx"

/*
 * Create a Unix Domain Socket
 */
ncclResult_t ncclIpcSocketInit(ncclIpcSocket *handle, int rank, uint64_t hash, volatile uint32_t *abortFlag)
{
  int fd = -1;
  struct sockaddr_un cliaddr;
  char temp[NCCL_IPC_SOCKNAME_LEN] = "";

  if (handle == NULL)
  {
    return ncclInternalError;
  }

  handle->fd = -1;
  handle->socketName[0] = '\0';
  if ((fd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0)
  {
    WARN("UDS: Socket creation error : %s (%d)", strerror(errno), errno);
    return ncclSystemError;
  }

  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  // Create unique name for the socket.
  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1))
  {
    WARN("UDS: Cannot bind provided name to socket. Name too large");
    close(fd);
    return ncclInternalError;
  }
#ifndef USE_ABSTRACT_SOCKET
  unlink(temp);
#endif

  TRACE(NCCL_INIT, "UDS: Creating socket %s", temp);

  strncpy(cliaddr.sun_path, temp, len);
#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
#endif
  if (bind(fd, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0)
  {
    WARN("UDS: Binding to socket %s failed : %s (%d)", temp, strerror(errno), errno);
    close(fd);
    return ncclSystemError;
  }

  handle->fd = fd;
  strcpy(handle->socketName, temp);

  handle->abortFlag = abortFlag;
  // Mark socket as non-blocking
  if (handle->abortFlag)
  {
    int flags;
    SYSCHECK(flags = fcntl(fd, F_GETFL), "fcntl");
    SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket *handle, int *fd)
{
  if (handle == NULL)
  {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (fd)
    *fd = handle->fd;
  return ncclSuccess;
}

ncclResult_t ncclIpcSocketClose(ncclIpcSocket *handle)
{
  if (handle == NULL)
  {
    return ncclInternalError;
  }
  if (handle->fd <= 0)
  {
    return ncclSuccess;
  }
#ifndef USE_ABSTRACT_SOCKET
  if (handle->socketName[0] != '\0')
  {
    unlink(handle->socketName);
  }
#endif
  close(handle->fd);

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, int *recvFd)
{
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];

  // Union to guarantee alignment requirements for control array
  union
  {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1];
  int ret;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  if (hdr == NULL)
  {
    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  }
  else
  {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0)
  {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
    {
      WARN("UDS: Receiving data over socket failed : %d", errno);
      return ncclSystemError;
    }
    if (handle->abortFlag && __atomic_load_n(handle->abortFlag, __ATOMIC_ACQUIRE))
      return ncclInternalError;
  }

  if (recvFd != NULL)
  {
    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int))))
    {
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS))
      {
        WARN("UDS: Receiving data over socket failed");
        return ncclSystemError;
      }

      memmove(recvFd, CMSG_DATA(cmptr), sizeof(*recvFd));
    }
    else
    {
      WARN("UDS: Receiving data over socket %s failed", handle->socketName);
      return ncclSystemError;
    }
    TRACE(NCCL_INIT | NCCL_P2P, "UDS: Got recvFd %d from socket %s", *recvFd, handle->socketName);
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvFd(ncclIpcSocket *handle, int *recvFd)
{
  return ncclIpcSocketRecvMsg(handle, NULL, 0, recvFd);
}

ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, const int sendFd, int rank, uint64_t hash)
{
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];
  char temp[NCCL_IPC_SOCKNAME_LEN];

  union
  {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1] = {'\0'};
  struct sockaddr_un cliaddr;

  // Construct client address to send this shareable handle to
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1))
  {
    WARN("UDS: Cannot connect to provided name for socket. Name too large");
    return ncclInternalError;
  }
  (void)strncpy(cliaddr.sun_path, temp, len);

#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
#endif

  TRACE(NCCL_INIT, "UDS: Sending hdr %p len %d fd %d to UDS socket %s", hdr, hdrLen, sendFd, temp);

  if (sendFd != -1)
  {
    memset(&control_un, '\0', sizeof(control_un));
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memmove(CMSG_DATA(cmptr), &sendFd, sizeof(sendFd));
  }

  msg.msg_name = (void *)&cliaddr;
  msg.msg_namelen = sizeof(struct sockaddr_un);

  if (hdr == NULL)
  {
    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  }
  else
  {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;

  ssize_t sendResult;
  while ((sendResult = sendmsg(handle->fd, &msg, 0)) < 0)
  {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
    {
      WARN("UDS: Sending data over socket %s failed : %s (%d)", temp, strerror(errno), errno);
      return ncclSystemError;
    }
    if (handle->abortFlag && __atomic_load_n(handle->abortFlag, __ATOMIC_ACQUIRE))
      return ncclInternalError;
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketSendFd(ncclIpcSocket *handle, const int sendFd, int rank, uint64_t hash)
{
  return ncclIpcSocketSendMsg(handle, NULL, 0, sendFd, rank, hash);
}

#elif NCCL_PLATFORM_WINDOWS

// Windows implementation using Named Pipes
// Note: Windows does not support passing file descriptors between processes like Unix SCM_RIGHTS.
// This implementation uses named pipes for IPC communication, but handle passing must be done
// through other mechanisms (e.g., DuplicateHandle with explicit process handle).

#define NCCL_IPC_PIPENAME_STR "\\\\.\\pipe\\nccl-pipe-%d-%llx"

ncclResult_t ncclIpcSocketInit(ncclIpcSocket *handle, int rank, uint64_t hash, volatile uint32_t *abortFlag)
{
  if (handle == NULL)
  {
    return ncclInternalError;
  }

  handle->hPipe = INVALID_HANDLE_VALUE;
  handle->socketName[0] = '\0';
  handle->abortFlag = abortFlag;

  // Create unique name for the pipe
  snprintf(handle->socketName, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_PIPENAME_STR, rank, (unsigned long long)hash);

  TRACE(NCCL_INIT, "Named Pipe: Creating pipe %s", handle->socketName);

  // Create named pipe as server
  handle->hPipe = CreateNamedPipeA(
      handle->socketName,
      PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
      PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
      PIPE_UNLIMITED_INSTANCES,
      4096, // Output buffer size
      4096, // Input buffer size
      0,    // Default timeout
      NULL  // Default security
  );

  if (handle->hPipe == INVALID_HANDLE_VALUE)
  {
    WARN("Named Pipe: Creation error: %lu", GetLastError());
    return ncclSystemError;
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket *handle, int *fd)
{
  if (handle == NULL)
  {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  // On Windows, we can't return a file descriptor for a pipe handle
  // Return -1 to indicate this is not a valid socket
  if (fd)
    *fd = -1;
  return ncclSuccess;
}

ncclResult_t ncclIpcSocketClose(ncclIpcSocket *handle)
{
  if (handle == NULL)
  {
    return ncclInternalError;
  }
  if (handle->hPipe == INVALID_HANDLE_VALUE)
  {
    return ncclSuccess;
  }

  DisconnectNamedPipe(handle->hPipe);
  CloseHandle(handle->hPipe);
  handle->hPipe = INVALID_HANDLE_VALUE;

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, int *recvFd)
{
  char dummy_buffer[1];
  DWORD bytesRead = 0;
  BOOL success;

  if (hdr == NULL)
  {
    success = ReadFile(handle->hPipe, dummy_buffer, sizeof(dummy_buffer), &bytesRead, NULL);
  }
  else
  {
    success = ReadFile(handle->hPipe, hdr, hdrLen, &bytesRead, NULL);
  }

  if (!success)
  {
    DWORD err = GetLastError();
    if (err != ERROR_IO_PENDING && err != ERROR_MORE_DATA)
    {
      WARN("Named Pipe: Receiving data failed: %lu", err);
      return ncclSystemError;
    }
  }

  // Note: Windows named pipes cannot pass handles like Unix SCM_RIGHTS
  // The recvFd functionality is not directly supported
  if (recvFd != NULL)
  {
    *recvFd = -1; // Indicate no fd received
    WARN("Named Pipe: File descriptor passing not supported on Windows");
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvFd(ncclIpcSocket *handle, int *recvFd)
{
  return ncclIpcSocketRecvMsg(handle, NULL, 0, recvFd);
}

ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, const int sendFd, int rank, uint64_t hash)
{
  char pipeName[NCCL_IPC_SOCKNAME_LEN];
  char dummy_buffer[1] = {'\0'};
  DWORD bytesWritten = 0;
  HANDLE hClientPipe;

  // Construct pipe name to connect to
  snprintf(pipeName, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_PIPENAME_STR, rank, (unsigned long long)hash);

  TRACE(NCCL_INIT, "Named Pipe: Sending hdr %p len %d fd %d to pipe %s", hdr, hdrLen, sendFd, pipeName);

  // Connect to the named pipe
  hClientPipe = CreateFileA(
      pipeName,
      GENERIC_READ | GENERIC_WRITE,
      0,
      NULL,
      OPEN_EXISTING,
      0,
      NULL);

  if (hClientPipe == INVALID_HANDLE_VALUE)
  {
    WARN("Named Pipe: Connection to %s failed: %lu", pipeName, GetLastError());
    return ncclSystemError;
  }

  // Send the message
  BOOL success;
  if (hdr == NULL)
  {
    success = WriteFile(hClientPipe, dummy_buffer, sizeof(dummy_buffer), &bytesWritten, NULL);
  }
  else
  {
    success = WriteFile(hClientPipe, hdr, hdrLen, &bytesWritten, NULL);
  }

  CloseHandle(hClientPipe);

  if (!success)
  {
    WARN("Named Pipe: Sending data failed: %lu", GetLastError());
    return ncclSystemError;
  }

  // Note: sendFd is ignored on Windows as named pipes cannot pass handles
  if (sendFd != -1)
  {
    TRACE(NCCL_INIT, "Named Pipe: File descriptor passing not supported on Windows (fd=%d ignored)", sendFd);
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketSendFd(ncclIpcSocket *handle, const int sendFd, int rank, uint64_t hash)
{
  return ncclIpcSocketSendMsg(handle, NULL, 0, sendFd, rank, hash);
}

#endif // NCCL_PLATFORM_WINDOWS
