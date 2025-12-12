/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "shmutils.h"
#include "comm.h"
#include "checks.h"
#include "platform.h"

#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_shm.h"
#include "platform/win32_misc.h"
#else
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <utils.h>

struct shmHandleInternal
{
#if NCCL_PLATFORM_WINDOWS
  HANDLE hMapFile;
  HANDLE hFile;
#else
  int fd;
#endif
  char *shmPath;
  char *shmPtr;
  void *devShmPtr;
  size_t shmSize;
  size_t realShmSize;
  int *refcount;
};

#if NCCL_PLATFORM_WINDOWS
static void shmHandleInit(HANDLE hFile, HANDLE hMapFile, char *shmPath, size_t shmSize, size_t realShmSize, char *hptr, void *dptr, bool create, struct shmHandleInternal *handle)
{
  handle->hFile = hFile;
  handle->hMapFile = hMapFile;
  handle->shmPtr = hptr;
  handle->devShmPtr = dptr;
  handle->shmSize = shmSize;
  handle->realShmSize = realShmSize;
  handle->refcount = (hptr != NULL) ? (int *)(hptr + shmSize) : NULL;
  if (create)
  {
    int slen = (int)strlen(shmPath);
    handle->shmPath = (char *)malloc(slen + 1);
    memcpy(handle->shmPath, shmPath, slen + 1);
    if (hptr)
      memset(hptr, 0, shmSize);
  }
  else
  {
    handle->shmPath = NULL;
  }
  return;
}
#else
static void shmHandleInit(int fd, char *shmPath, size_t shmSize, size_t realShmSize, char *hptr, void *dptr, bool create, struct shmHandleInternal *handle)
{
  handle->fd = fd;
  handle->shmPtr = hptr;
  handle->devShmPtr = dptr;
  handle->shmSize = shmSize;
  handle->realShmSize = realShmSize;
  handle->refcount = (hptr != NULL) ? (int *)(hptr + shmSize) : NULL;
  if (create)
  {
    int slen = strlen(shmPath);
    handle->shmPath = (char *)malloc(slen + 1);
    memcpy(handle->shmPath, shmPath, slen + 1);
    if (hptr)
      memset(hptr, 0, shmSize);
  }
  else
  {
    handle->shmPath = NULL;
  }
  return;
}
#endif

ncclResult_t ncclShmOpen(char *shmPath, size_t shmPathSize, size_t shmSize, void **shmPtr, void **devShmPtr, int refcount, ncclShmHandle_t *handle)
{
#if NCCL_PLATFORM_WINDOWS
  HANDLE hFile = INVALID_HANDLE_VALUE;
  HANDLE hMapFile = NULL;
  char *hptr = NULL;
  void *dptr = NULL;
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal *tmphandle;
  bool create = refcount > 0 ? true : false;
  const size_t refSize = sizeof(int);
  const size_t realShmSize = shmSize + refSize;
  DWORD highSize = (DWORD)((realShmSize >> 32) & 0xFFFFFFFF);
  DWORD lowSize = (DWORD)(realShmSize & 0xFFFFFFFF);
  char fullName[MAX_PATH];
  const char *shmSuffix;
  const char *prefix = "/dev/shm/nccl-";

  *handle = *shmPtr = NULL;
  EQCHECKGOTO(tmphandle = (struct shmHandleInternal *)calloc(1, sizeof(struct shmHandleInternal)), NULL, ret, fail);

  // Strip "/dev/shm/nccl-" prefix if present to get just the suffix
  shmSuffix = shmPath;
  if (strncmp(shmPath, prefix, strlen(prefix)) == 0)
  {
    shmSuffix = shmPath + strlen(prefix);
  }

  if (create)
  {
    if (shmPath[0] == '\0')
    {
      /* Generate unique name - will fill XXXXXX template or generate new name */
      ncclShmMkstemp(shmPath, shmPathSize);
      /* Re-check for prefix after generation */
      shmSuffix = shmPath;
      if (strncmp(shmPath, prefix, strlen(prefix)) == 0)
      {
        shmSuffix = shmPath + strlen(prefix);
      }
    }

    /* Use Windows named shared memory */
    snprintf(fullName, sizeof(fullName), "Local\\nccl-%s", shmSuffix);

    hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
        highSize, lowSize, fullName);

    if (hMapFile == NULL)
    {
      WARN("Error: failed to create shared memory mapping %s, error %lu", fullName, GetLastError());
      ret = ncclSystemError;
      goto fail;
    }
    INFO(NCCL_ALLOC, "Allocated %lld bytes of shared memory in %s", (long long)realShmSize, fullName);
  }
  else
  {
    snprintf(fullName, sizeof(fullName), "Local\\nccl-%s", shmSuffix);
    hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, fullName);
    if (hMapFile == NULL)
    {
      WARN("Error: failed to open shared memory mapping %s, error %lu", fullName, GetLastError());
      ret = ncclSystemError;
      goto fail;
    }
  }

  hptr = (char *)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, realShmSize);
  if (hptr == NULL)
  {
    WARN("Error: Could not map %s size %zu, error %lu", fullName, realShmSize, GetLastError());
    ret = ncclSystemError;
    goto fail;
  }

  if (create)
  {
    *(int *)(hptr + shmSize) = refcount;
  }
  else
  {
    int remref = ncclAtomicRefCountDecrement((int *)(hptr + shmSize));
    if (remref == 0)
    {
      /* Last peer - mapping will be cleaned up when all handles close */
    }
  }

  if (devShmPtr)
  {
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail);
    CUDACHECKGOTO(cudaHostRegister((void *)hptr, realShmSize, cudaHostRegisterPortable | cudaHostRegisterMapped), ret, fail);
    CUDACHECKGOTO(cudaHostGetDevicePointer(&dptr, (void *)hptr, 0), ret, fail);
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail);
  }

  shmHandleInit(hFile, hMapFile, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
exit:
  *shmPtr = hptr;
  if (devShmPtr)
    *devShmPtr = dptr;
  *handle = (ncclShmHandle_t)tmphandle;
  return ret;
fail:
  WARN("Error while %s shared memory segment %s (size %lld), error: %lu", create ? "creating" : "attaching to",
       shmPath, (long long)shmSize, GetLastError());
  if (tmphandle)
  {
    shmHandleInit(hFile, hMapFile, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
    (void)ncclShmClose((ncclShmHandle_t)tmphandle);
    tmphandle = NULL;
  }
  hptr = NULL;
  dptr = NULL;
  goto exit;

#else  /* POSIX */
  int fd = -1;
  char *hptr = NULL;
  void *dptr = NULL;
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal *tmphandle;
  bool create = refcount > 0 ? true : false;
  const size_t refSize = sizeof(int); /* extra sizeof(int) bytes for reference count */
  const size_t realShmSize = shmSize + refSize;

  *handle = *shmPtr = NULL; /* assume shmPtr and handle always set correctly by users. */
  EQCHECKGOTO(tmphandle = (struct shmHandleInternal *)calloc(1, sizeof(struct shmHandleInternal)), NULL, ret, fail);
  if (create)
  {
    /* refcount > 0 means the caller tries to allocate a shared memory. This shared memory segment will have
     * refcount references; when the peer attaches, it should pass -1 to reduce one reference count. When it
     * goes down to 0, unlink should be called in order to delete shared memory file. */
    if (shmPath[0] == '\0')
    {
      snprintf(shmPath, shmPathSize, "/dev/shm/nccl-XXXXXX");
    retry_mkstemp:
      fd = mkstemp(shmPath);
      if (fd < 0)
      {
        if (errno == EINTR)
        {
          INFO(NCCL_ALL, "mkstemp: Failed to create %s, error: %s (%d) - retrying", shmPath, strerror(errno), errno);
          goto retry_mkstemp;
        }
        WARN("Error: failed to create shared memory file %s, error %s (%d)", shmPath, strerror(errno), errno);
        ret = ncclSystemError;
        goto fail;
      }
    }
    else
    {
      SYSCHECKGOTO(fd = open(shmPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail);
    }

  retry_fallocate:
    if (fallocate(fd, 0, 0, realShmSize) != 0)
    {
      if (errno == EINTR)
      {
        INFO(NCCL_ALL, "fallocate: Failed to extend %s to %ld bytes, error: %s (%d) - retrying", shmPath, realShmSize, strerror(errno), errno);
        goto retry_fallocate;
      }
      WARN("Error: failed to extend %s to %ld bytes, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
      ret = ncclSystemError;
      goto fail;
    }
    INFO(NCCL_ALLOC, "Allocated %ld bytes of shared memory in %s", realShmSize, shmPath);
  }
  else
  {
    SYSCHECKGOTO(fd = open(shmPath, O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail);
  }

  hptr = (char *)mmap(NULL, realShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (hptr == MAP_FAILED)
  {
    WARN("Error: Could not map %s size %zu, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
    ret = ncclSystemError;
    hptr = NULL;
    goto fail;
  }

  if (create)
  {
    *(int *)(hptr + shmSize) = refcount;
  }
  else
  {
    int remref = ncclAtomicRefCountDecrement((int *)(hptr + shmSize));
    if (remref == 0)
    {
      /* the last peer has completed attachment, it should unlink the shm mem file. */
      if (unlink(shmPath) != 0)
      {
        INFO(NCCL_ALLOC, "unlink shared memory %s failed, error: %s (%d)", shmPath, strerror(errno), errno);
      }
    }
  }

  if (devShmPtr)
  {
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail);
    CUDACHECKGOTO(cudaHostRegister((void *)hptr, realShmSize, cudaHostRegisterPortable | cudaHostRegisterMapped), ret, fail);
    CUDACHECKGOTO(cudaHostGetDevicePointer(&dptr, (void *)hptr, 0), ret, fail);
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail);
  }

  shmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
exit:
  *shmPtr = hptr;
  if (devShmPtr)
    *devShmPtr = dptr;
  *handle = (ncclShmHandle_t)tmphandle;
  return ret;
fail:
  WARN("Error while %s shared memory segment %s (size %ld), error: %s (%d)", create ? "creating" : "attaching to",
       shmPath, shmSize, strerror(errno), errno);
  if (tmphandle)
  {
    shmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
    (void)ncclShmClose((ncclShmHandle_t)tmphandle);
    tmphandle = NULL;
  }
  hptr = NULL;
  dptr = NULL;
  goto exit;
#endif /* NCCL_PLATFORM_WINDOWS */
}

ncclResult_t ncclShmClose(ncclShmHandle_t handle)
{
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal *tmphandle = (struct shmHandleInternal *)handle;
  if (tmphandle)
  {
#if NCCL_PLATFORM_WINDOWS
    if (tmphandle->shmPtr)
    {
      if (tmphandle->devShmPtr)
        CUDACHECK(cudaHostUnregister(tmphandle->shmPtr));
      if (!UnmapViewOfFile(tmphandle->shmPtr))
      {
        WARN("UnmapViewOfFile of shared memory %p failed, error: %lu", tmphandle->shmPtr, GetLastError());
        ret = ncclSystemError;
      }
    }
    if (tmphandle->hMapFile != NULL)
    {
      CloseHandle(tmphandle->hMapFile);
    }
    if (tmphandle->hFile != INVALID_HANDLE_VALUE)
    {
      CloseHandle(tmphandle->hFile);
    }
    free(tmphandle->shmPath);
#else
    if (tmphandle->fd >= 0)
    {
      close(tmphandle->fd);
      if (tmphandle->shmPath != NULL && tmphandle->refcount != NULL && *tmphandle->refcount > 0)
      {
        if (unlink(tmphandle->shmPath) != 0)
        {
          WARN("unlink shared memory %s failed, error: %s (%d)", tmphandle->shmPath, strerror(errno), errno);
          ret = ncclSystemError;
        }
      }
      free(tmphandle->shmPath);
    }

    if (tmphandle->shmPtr)
    {
      if (tmphandle->devShmPtr)
        CUDACHECK(cudaHostUnregister(tmphandle->shmPtr));
      if (munmap(tmphandle->shmPtr, tmphandle->realShmSize) != 0)
      {
        WARN("munmap of shared memory %p size %ld failed, error: %s (%d)", tmphandle->shmPtr, tmphandle->realShmSize, strerror(errno), errno);
        ret = ncclSystemError;
      }
    }
#endif
    free(tmphandle);
  }
  return ret;
}

ncclResult_t ncclShmUnlink(ncclShmHandle_t handle)
{
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal *tmphandle = (struct shmHandleInternal *)handle;
  if (tmphandle)
  {
#if NCCL_PLATFORM_WINDOWS
    /* On Windows, named mappings are automatically cleaned up when all handles close */
    if (tmphandle->shmPath != NULL)
    {
      free(tmphandle->shmPath);
      tmphandle->shmPath = NULL;
    }
#else
    if (tmphandle->shmPath != NULL && tmphandle->refcount != NULL && *tmphandle->refcount > 0)
    {
      if (unlink(tmphandle->shmPath) != 0)
      {
        WARN("unlink shared memory %s failed, error: %s (%d)", tmphandle->shmPath, strerror(errno), errno);
        ret = ncclSystemError;
      }
      free(tmphandle->shmPath);
      tmphandle->shmPath = NULL;
    }
#endif
  }
  return ret;
}

ncclResult_t ncclShmemAllgather(struct ncclComm *comm, struct ncclShmemCollBuff *shmem, void *sendbuff, void *recvbuff, size_t typeSize)
{
  ncclResult_t ret = ncclSuccess;
  int nextRound = shmem->round + 1;
  int curIndex = shmem->round % 2;
  bool done;
  int index = 0;
  size_t maxTypeSize = shmem->maxTypeSize;

  if (comm == NULL || shmem == NULL || sendbuff == NULL || recvbuff == NULL || maxTypeSize < typeSize)
  {
    ret = ncclInvalidArgument;
    goto exit;
  }

  memcpy((char *)shmem->ptr[curIndex] + comm->localRank * maxTypeSize, sendbuff, typeSize);
  /* reset the previous round and notify I arrive this round */
  __atomic_store_n((int *)((char *)shmem->cnt[curIndex] + CACHE_LINE_SIZE * comm->localRank), nextRound, __ATOMIC_RELEASE);

  do
  {
    done = true;
    for (int i = index; i < comm->localRanks; ++i)
    {
      if (i != comm->localRank && __atomic_load_n((int *)((char *)shmem->cnt[curIndex] + CACHE_LINE_SIZE * i), __ATOMIC_ACQUIRE) < nextRound)
      {
        done = false;
        index = i;
        break;
      }
    }
  } while (!done);

  for (int i = 0; i < comm->localRanks; ++i)
  {
    memcpy((uint8_t *)recvbuff + i * typeSize, (uint8_t *)shmem->ptr[curIndex] + i * maxTypeSize, typeSize);
  }
  shmem->round = nextRound;

exit:
  return ret;
}
