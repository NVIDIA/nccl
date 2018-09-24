/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SHM_H_
#define NCCL_SHM_H_

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

static ncclResult_t shmOpen(const char* shmname, const int shmsize, void** shmPtr, void** devShmPtr, int create) {
  *shmPtr = NULL;
  int fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    WARN("shm_open failed to open %s : %s", shmname, strerror(errno));
    return ncclSystemError;
  }

  if (create) {
    int res = posix_fallocate(fd, 0, shmsize);
    if (res != 0) {
      WARN("Unable to allocate shared memory (%d bytes) : %s", shmsize, strerror(res));
      shm_unlink(shmname);
      close(fd);
      return ncclSystemError;
    }
  }

  void *ptr = mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    WARN("failure in mmap of %s (size %d) : %s", shmname, shmsize, strerror(errno));
    shm_unlink(shmname);
    return ncclSystemError;
  }
  if (create) {
    memset(ptr, 0, shmsize);
  }

  cudaError_t e;
  if ((e=cudaHostRegister(ptr, shmsize, cudaHostRegisterMapped)) != cudaSuccess) {
    WARN("failed to register host buffer %p : %s", ptr, cudaGetErrorString(e));
    if (create) shm_unlink(shmname);
    munmap(ptr, shmsize);
    return ncclUnhandledCudaError;
  }

  if ((e=cudaHostGetDevicePointer(devShmPtr, ptr, 0)) != cudaSuccess) {
    WARN("failed to get device pointer for local shmem %p : %s", ptr, cudaGetErrorString(e));
    if (create) shm_unlink(shmname);
    munmap(ptr, shmsize);
    return ncclUnhandledCudaError;
  }
  *shmPtr = ptr;
  return ncclSuccess;
}

static ncclResult_t shmUnlink(const char* shmname) {
  if (shmname != NULL) SYSCHECK(shm_unlink(shmname), "shm_unlink");
  return ncclSuccess;
}

static ncclResult_t shmClose(void* shmPtr, void* devShmPtr, const int shmsize) {
  CUDACHECK(cudaHostUnregister(shmPtr));
  if (munmap(shmPtr, shmsize) != 0) {
    WARN("munmap of shared memory failed");
    return ncclSystemError;
  }
  return ncclSuccess;
}

#endif
