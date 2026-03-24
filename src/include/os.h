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

#if defined(NCCL_OS_WINDOWS)
#include "os/windows.h"
#elif defined(NCCL_OS_LINUX)
#include "os/linux.h"
#endif

// POSIX PATH_MAX equivalent on Windows
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif

uint64_t ncclOsGetPid();
uint64_t ncclOsGetTid();
size_t ncclOsGetPageSize();
ncclResult_t ncclOsInitialize();

ncclResult_t ncclOsSetFilesLimit();

/* Aligned memory allocation */
void* ncclOsAlignedAlloc(size_t alignment, size_t size);
void ncclOsAlignedFree(void* ptr);

std::tm* ncclOsLocaltime(const time_t* timer, std::tm* buf);

void ncclOsSetEnv(const char* name, const char* value);

/* Dynamic library loading */
typedef void* ncclOsLibraryHandle;
#define NCCL_OS_DL_LAZY 0
#define NCCL_OS_DL_NOW  1
ncclOsLibraryHandle ncclOsDlopen(const char* filename);
ncclOsLibraryHandle ncclOsDlopen(const char* path, int mode);
void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol);
void ncclOsDlclose(ncclOsLibraryHandle handle);
const char* ncclOsDlerror();

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

void ncclOsCpuZero(ncclAffinity& affinity);
int ncclOsCpuCount(const ncclAffinity& affinity);
void ncclOsCpuSet(ncclAffinity& affinity, int cpu);
bool ncclOsCpuIsSet(const ncclAffinity& affinity, int cpu);
ncclAffinity ncclOsCpuAnd(const ncclAffinity& a, const ncclAffinity& b);
ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity);
ncclResult_t ncclOsSetAffinity(const ncclAffinity& affinity);
int ncclOsGetCpu();
/* Path resolution */
char* ncclOsRealpath(const char* path, char* resolved_path);

/* Shared memory functions - platform-specific implementations in os/linux.cc and os/windows.cc */
#include <stddef.h>
#include <stdbool.h>

// Platform-specific shared memory descriptor (similar to ncclSocketDescriptor)
#ifdef NCCL_OS_LINUX
typedef int ncclShmDescriptor;
#elif defined(NCCL_OS_WINDOWS)
typedef HANDLE ncclShmDescriptor;
#else
typedef int ncclShmDescriptor;  /* stub when no OS defined */
#endif

// Shared memory handle structure
struct ncclShmHandleInternal {
  ncclShmDescriptor shmDesc;
  char* shmPath;
  char* shmPtr;
  void* devShmPtr;
  size_t shmSize;
  size_t realShmSize;
  int* refcount;
};

/* Initialize the shared memory handle structure */
void ncclOsShmHandleInit(ncclShmDescriptor shmDesc, char* shmPath, size_t shmSize, size_t realShmSize,
                         char* hptr, void* dptr, bool create,
                         struct ncclShmHandleInternal* handle);

/* Create or open shared memory */
ncclResult_t ncclOsShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize,
                           void** shmPtr, void** devShmPtr, int refcount,
                           struct ncclShmHandleInternal** handle);

/* Close and cleanup shared memory */
ncclResult_t ncclOsShmClose(struct ncclShmHandleInternal* handle);

/* Unlink shared memory (remove from system) */
ncclResult_t ncclOsShmUnlink(struct ncclShmHandleInternal* handle);


/* NVML */
ncclResult_t ncclOsNvmlOpen(ncclOsLibraryHandle* handle);

#endif
