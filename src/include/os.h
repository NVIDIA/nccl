/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_OS_H_
#define NCCL_OS_H_

#include "nccl.h"

#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <mutex>

#ifdef NCCL_OS_WINDOWS
/* Prevent windows.h from defining min/max macros that conflict with std::min/max */
#ifndef NOMINMAX
#define NOMINMAX
#endif
/* ssize_t: defined by some MSVC CRT versions but not all */
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef __int64 ssize_t;
#endif
/* winsock2.h must be included before windows.h to avoid winsock.h conflicts */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <windows.h>
/* POSIX shutdown() direction flags: map to Winsock SD_* equivalents */
#ifndef SHUT_RD
#define SHUT_RD   SD_RECEIVE
#endif
#ifndef SHUT_WR
#define SHUT_WR   SD_SEND
#endif
#ifndef SHUT_RDWR
#define SHUT_RDWR SD_BOTH
#endif
/* POSIX close(fd) for file descriptors (not sockets — those use closesocket) */
#include <io.h>
#ifndef NCCL_CLOSE_DEFINED
#define NCCL_CLOSE_DEFINED
static inline int close(int fd) { return _close(fd); }
#endif
/* usleep: POSIX microsecond sleep; map to Windows Sleep(ms) */
#include <synchapi.h>
static inline void usleep(unsigned int usec) { Sleep((usec + 999U) / 1000U); }
/* localtime_r: thread-safe localtime */
#include <time.h>
static inline struct tm* localtime_r(const time_t* t, struct tm* buf) {
  localtime_s(buf, t); return buf;
}
/* Receive flags not available on Windows — silently ignored */
#ifndef MSG_DONTWAIT
#define MSG_DONTWAIT 0
#endif
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#endif

#ifdef NCCL_OS_LINUX
#define NCCL_INVALID_SOCKET -1
#elif defined(NCCL_OS_WINDOWS)
#define NCCL_INVALID_SOCKET INVALID_SOCKET
#endif

#ifdef NCCL_OS_LINUX
typedef int ncclSocketDescriptor;
#elif defined(NCCL_OS_WINDOWS)
typedef SOCKET ncclSocketDescriptor;
#endif

uint64_t ncclOsGetPid();
uint64_t ncclOsGetTid();
size_t ncclOsGetPageSize();
ncclResult_t ncclOsInitialize();

/* Aligned memory allocation */
void* ncclOsAlignedAlloc(size_t alignment, size_t size);
void ncclOsAlignedFree(void* ptr);

std::tm* ncclOsLocaltime(const time_t* timer, std::tm* buf);

void ncclOsSetEnv(const char* name, const char* value);

/* Socket functions */
bool ncclOsSocketIsValid(struct ncclSocket* sock);
bool ncclOsSocketDescriptorIsValid(ncclSocketDescriptor sock);
ncclResult_t ncclOsFindInterfaces(const char* prefixList, char* names, union ncclSocketAddress *addrs, int sock_family,
  int maxIfNameSize, int maxIfs, int* found);
void ncclOsPollSocket(ncclSocketDescriptor sock, int op);
ncclResult_t ncclOsSocketPollConnect(struct ncclSocket* sock);
ncclResult_t ncclOsSocketStartConnect(struct ncclSocket* sock);
ncclResult_t ncclOsSocketSetFlags(struct ncclSocket* sock);
ncclResult_t ncclOsSocketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int block, int* closed);
ncclResult_t ncclOsSocketResetFd(struct ncclSocket* sock);
void ncclOsSocketResetAccept(struct ncclSocket* sock);
ncclResult_t ncclOsSocketTryAccept(struct ncclSocket* sock);

void ncclOsSetMutexCondShared(std::mutex &mutex, std::condition_variable &cond);

/* Affinity functions */
#ifdef NCCL_OS_LINUX
typedef cpu_set_t ncclAffinity;
#elif defined(NCCL_OS_WINDOWS)
typedef DWORD_PTR ncclAffinity;
#endif
void ncclOsCpuZero(ncclAffinity& affinity);
int ncclOsCpuCount(const ncclAffinity affinity);
void ncclOsCpuSet(ncclAffinity& affinity, int cpu);
bool ncclOsCpuIsSet(const ncclAffinity affinity, int cpu);
ncclAffinity ncclOsCpuAnd(const ncclAffinity& a, const ncclAffinity& b);
ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity);
ncclResult_t ncclOsSetAffinity(const ncclAffinity affinity);
int ncclOsGetCpu();

#ifdef NCCL_OS_WINDOWS
/* POSIX getline implementation for Windows */
#include <stdio.h>
#include <stdlib.h>
static inline ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
  if (!lineptr || !n || !stream) return -1;
  if (!*lineptr || *n == 0) {
    *n = 256;
    *lineptr = (char*)malloc(*n);
    if (!*lineptr) return -1;
  }
  size_t pos = 0;
  int c;
  while ((c = fgetc(stream)) != EOF) {
    if (pos + 2 > *n) {
      size_t newsz = *n * 2;
      char *tmp = (char*)realloc(*lineptr, newsz);
      if (!tmp) return -1;
      *lineptr = tmp;
      *n = newsz;
    }
    (*lineptr)[pos++] = (char)c;
    if (c == '\n') break;
  }
  if (pos == 0) return -1;
  (*lineptr)[pos] = '\0';
  return (ssize_t)pos;
}
#endif // NCCL_OS_WINDOWS

#endif
