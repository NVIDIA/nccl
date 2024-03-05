/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "shm.h"
#include "comm.h"
#include "checks.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <utils.h>

struct shmHandleInternal {
  int fd;
  char* shmPath;
  char* shmPtr;
  void* devShmPtr;
  size_t shmSize;
  size_t realShmSize;
  int* refcount;
};

static void shmHandleInit(int fd, char* shmPath, size_t shmSize, size_t realShmSize, char* hptr, void* dptr, bool create, struct shmHandleInternal* handle) {
  handle->fd = fd;
  handle->shmPtr = hptr;
  handle->devShmPtr = dptr;
  handle->shmSize = shmSize;
  handle->realShmSize = realShmSize;
  handle->refcount = (hptr != NULL) ? (int*)(hptr + shmSize) : NULL;
  if (create) {
    int slen = strlen(shmPath);
    handle->shmPath = (char*)malloc(slen + 1);
    memcpy(handle->shmPath, shmPath, slen + 1);
    if (hptr) memset(hptr, 0, shmSize);
  } else {
    handle->shmPath = NULL;
  }
  return;
}

ncclResult_t ncclShmOpen(char* shmPath, size_t shmSize, void** shmPtr, void** devShmPtr, int refcount, ncclShmHandle_t* handle) {
  int fd = -1;
  char* hptr = NULL;
  void* dptr = NULL;
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle;
  bool create = refcount > 0 ? true : false;
  const size_t refSize = sizeof(int); /* extra sizeof(int) bytes for reference count */
  const size_t realShmSize = shmSize + refSize;

  *handle = *shmPtr = NULL; /* assume shmPtr and handle always set correctly by users. */
  EQCHECKGOTO(tmphandle = (struct shmHandleInternal*)calloc(1, sizeof(struct shmHandleInternal)), NULL, ret, fail);
  if (create) {
    /* refcount > 0 means the caller tries to allocate a shared memory. This shared memory segment will have
     * refcount references; when the peer attaches, it should pass -1 to reduce one reference count. When it
     * goes down to 0, unlink should be called in order to delete shared memory file. */
    if (shmPath[0] == '\0') {
      sprintf(shmPath, "/dev/shm/nccl-XXXXXX");
      fd = mkstemp(shmPath);
    } else {
      SYSCHECKGOTO(fd = open(shmPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), ret, fail);
    }

    if (fallocate(fd, 0, 0, realShmSize) != 0) {
      WARN("Error: failed to extend %s to %ld bytes", shmPath, realShmSize);
      ret = ncclSystemError;
      goto fail;
    }
    INFO(NCCL_ALLOC, "Allocated %ld bytes of shared memory in %s", realShmSize, shmPath);
  } else {
    SYSCHECKGOTO(fd = open(shmPath, O_RDWR, S_IRUSR | S_IWUSR), ret, fail);
  }

  hptr = (char*)mmap(NULL, realShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (hptr == MAP_FAILED) {
    WARN("Could not map %s size %zi, error: %s", shmPath, realShmSize, strerror(errno));
    ret = ncclSystemError;
    hptr = NULL;
    goto fail;
  }

  if (create) {
    *(int*)(hptr + shmSize) = refcount;
  } else {
    int remref = ncclAtomicRefCountDecrement((int*)(hptr + shmSize));
    if (remref == 0) {
      /* the last peer has completed attachment, it should unlink the shm mem file. */
      if (unlink(shmPath) != 0) {
        WARN("unlink shared memory %s failed, error: %s", shmPath, strerror(errno));
      }
    }
  }

  if (devShmPtr) {
    CUDACHECKGOTO(cudaHostRegister((void*)hptr, realShmSize, cudaHostRegisterMapped), ret, fail);
    CUDACHECKGOTO(cudaHostGetDevicePointer(&dptr, (void*)hptr, 0), ret, fail);
  }

  shmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
exit:
  *shmPtr = hptr;
  if (devShmPtr) *devShmPtr = dptr;
  *handle = (ncclShmHandle_t)tmphandle;
  return ret;
fail:
  WARN("Error while %s shared memory segment %s (size %ld)", create ? "creating" : "attaching to", shmPath, shmSize);
  if (tmphandle) {
    shmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
    ncclShmClose((ncclShmHandle_t)tmphandle);
    tmphandle = NULL;
  }
  hptr = NULL;
  dptr = NULL;
  goto exit;
}

ncclResult_t ncclShmClose(ncclShmHandle_t handle) {
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle = (struct shmHandleInternal*)handle;
  if (tmphandle) {
    if (tmphandle->fd >= 0) {
      close(tmphandle->fd);
      if (tmphandle->shmPath != NULL && tmphandle->refcount != NULL && *tmphandle->refcount > 0) {
        if (unlink(tmphandle->shmPath) != 0) {
          WARN("unlink shared memory %s failed, error: %s", tmphandle->shmPath, strerror(errno));
          ret = ncclSystemError;
        }
      }
      free(tmphandle->shmPath);
    }

    if (tmphandle->shmPtr) {
      if (tmphandle->devShmPtr) CUDACHECK(cudaHostUnregister(tmphandle->shmPtr));
      if (munmap(tmphandle->shmPtr, tmphandle->realShmSize) != 0) {
        WARN("munmap of shared memory %p size %ld failed, error: %s", tmphandle->shmPtr, tmphandle->realShmSize, strerror(errno));
        ret = ncclSystemError;
      }
    }
    free(tmphandle);
  }
  return ret;
}

ncclResult_t ncclShmUnlink(ncclShmHandle_t handle) {
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle = (struct shmHandleInternal*)handle;
  if (tmphandle) {
    if (tmphandle->shmPath != NULL) {
      if (unlink(tmphandle->shmPath) != 0) {
        WARN("unlink shared memory %s failed, error: %s", tmphandle->shmPath, strerror(errno));
        ret = ncclSystemError;
      }
      free(tmphandle->shmPath);
      tmphandle->shmPath = NULL;
    }
  }
  return ret;
}

ncclResult_t ncclShmemAllgather(struct ncclComm *comm, struct ncclShmemCollBuff *shmem, void *sendbuff, void *recvbuff, size_t typeSize) {
  ncclResult_t ret = ncclSuccess;
  int curRound = shmem->round;
  size_t mycnt;

  if (comm == NULL || shmem == NULL || sendbuff == NULL || recvbuff == NULL || shmem->maxTypeSize < typeSize) {
    ret = ncclInvalidArgument;
    goto exit;
  }

  memcpy((char*)shmem->ptr[curRound] + comm->localRank * typeSize, sendbuff, typeSize);
  /* sync among local ranks */
  mycnt = __atomic_add_fetch(shmem->cnt[curRound], 1, __ATOMIC_ACQ_REL);
  if (mycnt == comm->localRanks) {
    *shmem->cnt[curRound ^ 1] = 0; /* prepare next round */
    __atomic_store_n(shmem->cnt[curRound], comm->localRanks + 1, __ATOMIC_RELEASE); /* release everyone */
  } else {
    uint64_t t0 = clockNano();
    while(__atomic_load_n(shmem->cnt[curRound], __ATOMIC_ACQUIRE) != comm->localRanks + 1) {
      if (clockNano() - t0 >= 5 * 1000) sched_yield();
      if (__atomic_load_n(comm->abortFlag, __ATOMIC_RELAXED) == 1) {
        ret = ncclInternalError;
        goto exit;
      }
    }
  }

  memcpy(recvbuff, (const void*)shmem->ptr[curRound], comm->localRanks * typeSize);
  shmem->round ^= 1;

exit:
  return ret;
}
