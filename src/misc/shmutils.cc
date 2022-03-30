/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "shm.h"
#include "checks.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Change functions behavior to match other SYS functions
static int shm_allocate(int fd, const int shmSize) {
  int err = posix_fallocate(fd, 0, shmSize);
  if (err) { errno = err; return -1; }
  return 0;
}
static int shm_map(int fd, const int shmSize, void** ptr) {
  *ptr = mmap(NULL, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return (*ptr == MAP_FAILED) ? -1 : 0;
}

static ncclResult_t ncclShmSetup(char* shmPath, const int shmSize, int* fd, void** ptr, int create) {
  if (create) {
    if (shmPath[0] == '\0') {
      sprintf(shmPath, "/dev/shm/nccl-XXXXXX");
      *fd = mkstemp(shmPath);
    } else {
      SYSCHECKVAL(open(shmPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "open", *fd);
    }
    if (ftruncate(*fd, shmSize) != 0) {
      WARN("Error: failed to extend %s to %d bytes", shmPath, shmSize);
      return ncclSystemError;
    }
  } else {
    SYSCHECKVAL(open(shmPath, O_RDWR, S_IRUSR | S_IWUSR), "open", *fd);
  }
  *ptr = (char*)mmap(NULL, shmSize, PROT_READ|PROT_WRITE, MAP_SHARED, *fd, 0);
  if (*ptr == NULL) {
    WARN("Could not map %s\n", shmPath);
    return ncclSystemError;
  }
  close(*fd);
  *fd = -1;
  if (create) memset(*ptr, 0, shmSize);
  return ncclSuccess;
}

ncclResult_t ncclShmOpen(char* shmPath, const int shmSize, void** shmPtr, void** devShmPtr, int create) {
  int fd = -1;
  void* ptr = MAP_FAILED;
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(ncclShmSetup(shmPath, shmSize, &fd, &ptr, create), res, sysError);
  if (devShmPtr) {
    CUDACHECKGOTO(cudaHostRegister(ptr, shmSize, cudaHostRegisterMapped), res, cudaError);
    CUDACHECKGOTO(cudaHostGetDevicePointer(devShmPtr, ptr, 0), res, cudaError);
  }

  *shmPtr = ptr;
  return ncclSuccess;
sysError:
  WARN("Error while %s shared memory segment %s (size %d)", create ? "creating" : "attaching to", shmPath, shmSize);
cudaError:
  if (fd != -1) close(fd);
  if (create) shm_unlink(shmPath);
  if (ptr != MAP_FAILED) munmap(ptr, shmSize);
  *shmPtr = NULL;
  return res;
}

ncclResult_t ncclShmUnlink(const char* shmPath) {
  if (shmPath != NULL) SYSCHECK(unlink(shmPath), "unlink");
  return ncclSuccess;
}

ncclResult_t ncclShmClose(void* shmPtr, void* devShmPtr, const int shmSize) {
  if (devShmPtr) CUDACHECK(cudaHostUnregister(shmPtr));
  if (munmap(shmPtr, shmSize) != 0) {
    WARN("munmap of shared memory failed");
    return ncclSystemError;
  }
  return ncclSuccess;
}
