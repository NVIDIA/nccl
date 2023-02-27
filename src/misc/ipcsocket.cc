/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "ipcsocket.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// Enable Linux abstract socket naming
#define USE_ABSTRACT_SOCKET

#define NCCL_IPC_SOCKNAME_STR "/tmp/nccl-socket-%d-%lx"

/*
 * Create a Unix Domain Socket
 */
ncclResult_t ncclIpcSocketInit(ncclIpcSocket *handle, int rank, uint64_t hash, volatile uint32_t* abortFlag) {
  int fd = -1;
  struct sockaddr_un cliaddr;
  char temp[NCCL_IPC_SOCKNAME_LEN] = "";

  if (handle == NULL) {
    return ncclInternalError;
  }

  handle->fd = -1;
  handle->socketName[0] = '\0';
  if ((fd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
    WARN("UDS: Socket creation error : %d", errno);
    return ncclSystemError;
  }

  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  // Create unique name for the socket.
  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    WARN("UDS: Cannot bind provided name to socket. Name too large");
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
  if (bind(fd, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
    WARN("UDS: Binding to socket %s failed : %d", temp, errno);
    close(fd);
    return ncclSystemError;
  }

  handle->fd = fd;
  strcpy(handle->socketName, temp);

  handle->abortFlag = abortFlag;
  // Mark socket as non-blocking
  if (handle->abortFlag) {
    int flags;
    EQCHECK(flags = fcntl(fd, F_GETFL), -1);
    SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketClose(ncclIpcSocket *handle) {
  if (handle == NULL) {
    return ncclInternalError;
  }
  if (handle->fd <= 0) {
    return ncclSuccess;
  }
#ifndef USE_ABSTRACT_SOCKET
  if (handle->socketName[0] != '\0') {
    unlink(handle->socketName);
  }
#endif
  close(handle->fd);

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvFd(ncclIpcSocket *handle, int *recvFd) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];

  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1];
  int ret;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  iov[0].iov_base = (void *)dummy_buffer;
  iov[0].iov_len = sizeof(dummy_buffer);

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      WARN("UDS: Receiving data over socket failed : %d", errno);
      return ncclSystemError;
    }
    if (handle->abortFlag && *handle->abortFlag) return ncclInternalError;
  }

  if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
    if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
      WARN("UDS: Receiving data over socket failed");
      return ncclSystemError;
    }

    memmove(recvFd, CMSG_DATA(cmptr), sizeof(*recvFd));
  } else {
    WARN("UDS: Receiving data over socket %s failed", handle->socketName);
    return ncclSystemError;
  }

  TRACE(NCCL_INIT|NCCL_P2P, "UDS: Got recvFd %d from socket %s", *recvFd, handle->socketName);

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketSendFd(ncclIpcSocket *handle, const int sendFd, int rank, uint64_t hash) {
  struct msghdr msg;
  struct iovec iov[1];
  char temp[NCCL_IPC_SOCKNAME_LEN];

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  struct sockaddr_un cliaddr;

  // Construct client address to send this shareable handle to
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    WARN("UDS: Cannot connect to provided name for socket. Name too large");
    return ncclInternalError;
  }
  (void) strncpy(cliaddr.sun_path, temp, len);

  TRACE(NCCL_INIT, "UDS: Sending fd %d to UDS socket %s", sendFd, temp);

#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
#endif

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  cmptr = CMSG_FIRSTHDR(&msg);
  cmptr->cmsg_len = CMSG_LEN(sizeof(int));
  cmptr->cmsg_level = SOL_SOCKET;
  cmptr->cmsg_type = SCM_RIGHTS;

  memmove(CMSG_DATA(cmptr), &sendFd, sizeof(sendFd));

  msg.msg_name = (void *)&cliaddr;
  msg.msg_namelen = sizeof(struct sockaddr_un);

  iov[0].iov_base = (void *)"";
  iov[0].iov_len = 1;
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;

  ssize_t sendResult;
  while ((sendResult = sendmsg(handle->fd, &msg, 0)) <= 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      WARN("UDS: Sending data over socket %s failed : %d", temp, errno);
      return ncclSystemError;
    }
    if (handle->abortFlag && *handle->abortFlag) return ncclInternalError;
  }

  return ncclSuccess;
}
