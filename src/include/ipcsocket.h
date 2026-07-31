/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_IPCSOCKET_H
#define NCCL_IPCSOCKET_H

#include "nccl.h"
#include <stdio.h>
#include "os.h"
#include <errno.h>
#include <memory.h>
#include <inttypes.h>

#define NCCL_IPC_SOCKNAME_LEN 64

#if defined(NCCL_OS_WINDOWS)
typedef intptr_t ncclIpcFd;
#else
typedef int ncclIpcFd;
#endif

#define NCCL_INVALID_IPC_FD ((ncclIpcFd) - 1)

struct ncclIpcSocket {
  ncclIpcFd fd;
  char socketName[NCCL_IPC_SOCKNAME_LEN];
  volatile uint32_t* abortFlag;
};

ncclResult_t ncclIpcSocketInit(struct ncclIpcSocket* handle, int rank, uint64_t hash, volatile uint32_t* abortFlag);
ncclResult_t ncclIpcSocketClose(struct ncclIpcSocket* handle);
ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* handle, ncclIpcFd* fd);

ncclResult_t ncclIpcSocketRecvFd(struct ncclIpcSocket* handle, ncclIpcFd* fd);
ncclResult_t ncclIpcSocketSendFd(struct ncclIpcSocket* handle, ncclIpcFd fd, int rank, uint64_t hash);

ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket* handle, void* hdr, int hdrLen, ncclIpcFd sendFd, int rank,
                                  uint64_t hash);
ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket* handle, void* hdr, int hdrLen, ncclIpcFd* recvFd);
int ncclIpcFdClose(ncclIpcFd fd);

#endif /* NCCL_IPCSOCKET_H */
