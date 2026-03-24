/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "shmutils.h"
#include "comm.h"
#include "checks.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utils.h>

#ifdef NCCL_OS_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#endif

struct shmHandleInternal {
#ifdef NCCL_OS_WINDOWS
  HANDLE hFile;
  HANDLE hMapping;
  bool cudaAllocated; // allocated via cudaHostAlloc, not MapViewOfFile
#else
  int fd;
#endif
  char* shmPath;
  char* shmPtr;
  void* devShmPtr;
  size_t shmSize;
  size_t realShmSize;
  int* refcount;
};

ncclResult_t ncclShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize, void** shmPtr, void** devShmPtr, int refcount, ncclShmHandle_t* handle) {
  char* hptr = NULL;
  void* dptr = NULL;
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle = NULL;
  bool create = refcount > 0;
  const size_t refSize = sizeof(int);
  const size_t realShmSize = shmSize + refSize;

  *handle = *shmPtr = NULL;
  tmphandle = (struct shmHandleInternal*)calloc(1, sizeof(struct shmHandleInternal));
  if (tmphandle == NULL) { ret = ncclSystemError; goto ret_early; }

#ifdef NCCL_OS_WINDOWS
  {
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMapping = NULL;
    tmphandle->hFile = INVALID_HANDLE_VALUE;
    tmphandle->hMapping = NULL;

    if (create) {
      if (shmPath[0] == '\0' && devShmPtr) {
        // Same-process anonymous allocation: use cudaHostAlloc(Portable) instead
        // of MapViewOfFile + cudaHostRegister.  Under WDDM, MapViewOfFile-backed
        // pages registered with cudaHostRegister do not produce DMA-coherent
        // device mappings — GPU writes through one device pointer are not visible
        // to reads through a device pointer derived by a different GPU context for
        // the same physical pages.  cudaHostAlloc bypasses this by letting CUDA
        // manage the pinning from the outset.
        //
        // The shmPath is left empty intentionally: for same-process connections
        // the importer (shmSendConnect / shmRecvConnect) reuses info->buf.hptr
        // directly, so no named section token is needed.
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        void* hptr_void = NULL;
        CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail_win);
        CUDACHECKGOTO(cudaHostAlloc(&hptr_void, realShmSize, cudaHostAllocPortable), ret, fail_win);
        hptr = (char*)hptr_void;
        // Mark cudaAllocated now so fail_win uses cudaFreeHost instead of UnmapViewOfFile
        tmphandle->cudaAllocated = true;
        CUDACHECKGOTO(cudaHostGetDevicePointer(&dptr, hptr_void, 0), ret, fail_win);
        CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail_win);

        *(int*)(hptr + shmSize) = refcount;
        memset(hptr, 0, shmSize);

        tmphandle->hFile        = INVALID_HANDLE_VALUE;
        tmphandle->hMapping     = NULL;
        tmphandle->shmPtr       = hptr;
        tmphandle->devShmPtr    = dptr;
        tmphandle->shmSize      = shmSize;
        tmphandle->realShmSize  = realShmSize;
        tmphandle->refcount     = (int*)(hptr + shmSize);
        tmphandle->shmPath      = NULL;
        INFO(NCCL_ALLOC, "cudaHostAlloc %zu bytes for SHM buffer hptr=%p dptr=%p", realShmSize, hptr, dptr);
        goto exit_win;
      } else if (shmPath[0] == '\0') {
        // No devShmPtr requested: fall through to named section path.
        // (This case is not expected in practice but handled for completeness.)
        //
        // Use a named page-file-backed section.  shm.cc extracts the 6-char
        // token from shmPath in two different ways depending on whether
        // NCCL_OS_WINDOWS is defined when shm.cc is compiled:
        //   - NCCL_OS_WINDOWS: memcpy(shmSuffix, shmPath,      7)  (offset  0)
        //   - Linux path:      memcpy(shmSuffix, shmPath + 14,  7)  (offset 14)
        // Write the token at BOTH offsets so either code path retrieves it.
        // The caller zero-initialises shmPath[SHM_PATH_MAX], so bytes 7-13
        // are already '\0' — the string at offset 0 terminates correctly.
        static volatile LONG shmCounter = 0;
        LONG cnt = InterlockedIncrement(&shmCounter);
        DWORD pid = GetCurrentProcessId();
        char token[8];
        snprintf(token, sizeof(token), "%06lx",
                 (unsigned long)(((LONGLONG)pid * 4096 + cnt) & 0xFFFFFF));
        snprintf(shmPath, shmPathSize, "%s", token);          /* offset  0 */
        if ((size_t)shmPathSize >= 14 + 7) {  /* 7 = 6 hex chars + null */
          memcpy(shmPath + 14, token, 7);                     /* offset 14 */
        }
        char mapName[32];
        snprintf(mapName, sizeof(mapName), "Local\\nccl-%s", token);
        DWORD hi = (DWORD)(realShmSize >> 32), lo = (DWORD)(realShmSize & 0xFFFFFFFF);
        hMapping = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                      hi, lo, mapName);
        if (hMapping == NULL) {
          WARN("Error: failed to create named shm section %s, error %lu", mapName, GetLastError());
          ret = ncclSystemError;
          goto fail_win;
        }
        INFO(NCCL_ALLOC, "Allocated %zu bytes of named shared memory section %s", realShmSize, mapName);
      } else {
        // File-backed shared memory at the caller-supplied path.
        hFile = CreateFileA(shmPath, GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
          WARN("Error: failed to create shared memory file %s, error %lu", shmPath, GetLastError());
          ret = ncclSystemError;
          goto fail_win;
        }
        /* Extend the file to realShmSize */
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG)realShmSize;
        if (!SetFilePointerEx(hFile, li, NULL, FILE_BEGIN) || !SetEndOfFile(hFile)) {
          WARN("Error: failed to extend %s to %zu bytes, error %lu", shmPath, realShmSize, GetLastError());
          ret = ncclSystemError;
          goto fail_win;
        }
        INFO(NCCL_ALLOC, "Allocated %zu bytes of file-backed shared memory in %s", realShmSize, shmPath);
        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
        if (hMapping == NULL) {
          WARN("Error: could not create mapping for %s, error %lu", shmPath, GetLastError());
          ret = ncclSystemError;
          goto fail_win;
        }
      }
    } else {
      // shmPath arrives as either:
      //   A) bare token "abc123"            (NCCL_OS_WINDOWS path in shm.cc)
      //   B) "/dev/shm/nccl-abc123"        (Linux path in shm.cc)
      // Both identify the same named page-file-backed section.
      static const char devShmPrefix[] = "/dev/shm/nccl-";
      const char *token = NULL;
      if (strncmp(shmPath, devShmPrefix, sizeof(devShmPrefix) - 1) == 0) {
        token = shmPath + sizeof(devShmPrefix) - 1;   /* case B */
      } else if (strchr(shmPath, '\\') == NULL && strchr(shmPath, '/') == NULL
                 && strchr(shmPath, ':') == NULL) {
        token = shmPath;                               /* case A */
      }
      bool namedSection = (token != NULL && token[0] != '\0');
      if (namedSection) {
        char mapName[32];
        snprintf(mapName, sizeof(mapName), "Local\\nccl-%s", token);
        hMapping = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, mapName);
        if (hMapping == NULL) {
          WARN("Error: failed to open named shm section %s, error %lu", mapName, GetLastError());
          ret = ncclSystemError;
          goto fail_win;
        }
      } else {
        hFile = CreateFileA(shmPath, GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
          WARN("Error: failed to open shared memory file %s, error %lu", shmPath, GetLastError());
          ret = ncclSystemError;
          goto fail_win;
        }
        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
        if (hMapping == NULL) {
          WARN("Error: could not create mapping for %s, error %lu", shmPath, GetLastError());
          ret = ncclSystemError;
          goto fail_win;
        }
      }
    }

    hptr = (char*)MapViewOfFile(hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (hptr == NULL) {
      WARN("Error: could not map %s size %zu, error %lu", shmPath, realShmSize, GetLastError());
      ret = ncclSystemError;
      goto fail_win;
    }

    if (create) {
      *(int*)(hptr + shmSize) = refcount;
    } else {
      int remref = ncclAtomicRefCountDecrement((int*)(hptr + shmSize));
      if (remref == 0) {
        // Mirror Linux behaviour: just "unlink" (remove the backing artefact so
        // the segment is cleaned up after everyone closes their handles), but
        // keep the view mapped so the caller can still use it.
        // For page-file-backed named sections there is no backing file to delete
        // (the section auto-destroys once the last CloseHandle is called).
        if (hFile != INVALID_HANDLE_VALUE) {
          // File-backed section: delete the file path so it is auto-cleaned up.
          if (!DeleteFileA(shmPath)) {
            INFO(NCCL_ALLOC, "DeleteFile %s failed, error %lu", shmPath, GetLastError());
          }
        }
        // Do NOT unmap or close the section handle here — the caller still
        // needs the mapped pointer.  ncclShmClose handles the actual cleanup.
      }
    }

    if (devShmPtr) {
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail_win);
      CUDACHECKGOTO(cudaHostRegister((void*)hptr, realShmSize, cudaHostRegisterPortable | cudaHostRegisterMapped), ret, fail_win);
      CUDACHECKGOTO(cudaHostGetDevicePointer(&dptr, (void*)hptr, 0), ret, fail_win);
      CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail_win);
    }

    tmphandle->hFile    = hFile;
    tmphandle->hMapping = hMapping;
    tmphandle->shmPtr   = hptr;
    tmphandle->devShmPtr = dptr;
    tmphandle->shmSize   = shmSize;
    tmphandle->realShmSize = realShmSize;
    tmphandle->refcount = (hptr != NULL) ? (int*)(hptr + shmSize) : NULL;
    if (create) {
      int slen = (int)strlen(shmPath);
      tmphandle->shmPath = (char*)malloc(slen + 1);
      memcpy(tmphandle->shmPath, shmPath, slen + 1);
      if (hptr) memset(hptr, 0, shmSize);
    } else {
      tmphandle->shmPath = NULL;
    }
    goto exit_win;

  fail:  /* alias for fail_win so EQCHECKGOTO/CHECKGOTO macros that use "fail" work on Windows */
  fail_win:
    WARN("Error while %s shared memory segment %s (size %zu)", create ? "creating" : "attaching to", shmPath, shmSize);
    if (hptr) {
      if (tmphandle && tmphandle->cudaAllocated) { (void)cudaFreeHost(hptr); }
      else { UnmapViewOfFile(hptr); }
      hptr = NULL;
    }
    if (hMapping){ CloseHandle(hMapping); hMapping = NULL; }
    if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
    if (tmphandle) { free(tmphandle->shmPath); free(tmphandle); tmphandle = NULL; }
    dptr = NULL;

  exit_win:
    *shmPtr = hptr;
    if (devShmPtr) *devShmPtr = dptr;
    *handle = (ncclShmHandle_t)tmphandle;
    return ret;
  }

#else  /* POSIX */
  {
    int fd = -1;

    if (create) {
      if (shmPath[0] == '\0') {
        snprintf(shmPath, shmPathSize, "/dev/shm/nccl-XXXXXX");
      retry_mkstemp:
        fd = mkstemp(shmPath);
        if (fd < 0) {
          if (errno == EINTR) {
            INFO(NCCL_ALL, "mkstemp: Failed to create %s, error: %s (%d) - retrying", shmPath, strerror(errno), errno);
            goto retry_mkstemp;
          }
          WARN("Error: failed to create shared memory file %s, error %s (%d)", shmPath, strerror(errno), errno);
          ret = ncclSystemError;
          goto fail_posix;
        }
      } else {
        SYSCHECKGOTO(fd = open(shmPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail_posix);
      }

    retry_fallocate:
      if (fallocate(fd, 0, 0, realShmSize) != 0) {
        if (errno == EINTR) {
          INFO(NCCL_ALL, "fallocate: Failed to extend %s to %ld bytes, error: %s (%d) - retrying", shmPath, realShmSize, strerror(errno), errno);
          goto retry_fallocate;
        }
        WARN("Error: failed to extend %s to %ld bytes, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
        ret = ncclSystemError;
        goto fail_posix;
      }
      INFO(NCCL_ALLOC, "Allocated %ld bytes of shared memory in %s", realShmSize, shmPath);
    } else {
      SYSCHECKGOTO(fd = open(shmPath, O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail_posix);
    }

    hptr = (char*)mmap(NULL, realShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (hptr == MAP_FAILED) {
      WARN("Error: Could not map %s size %zu, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
      ret = ncclSystemError;
      hptr = NULL;
      goto fail_posix;
    }

    if (create) {
      *(int*)(hptr + shmSize) = refcount;
    } else {
      int remref = ncclAtomicRefCountDecrement((int*)(hptr + shmSize));
      if (remref == 0) {
        if (unlink(shmPath) != 0) {
          INFO(NCCL_ALLOC, "unlink shared memory %s failed, error: %s (%d)", shmPath, strerror(errno), errno);
        }
      }
    }

    if (devShmPtr) {
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail_posix);
      CUDACHECKGOTO(cudaHostRegister((void*)hptr, realShmSize, cudaHostRegisterPortable | cudaHostRegisterMapped), ret, fail_posix);
      CUDACHECKGOTO(cudaHostGetDevicePointer(&dptr, (void*)hptr, 0), ret, fail_posix);
      CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail_posix);
    }

    tmphandle->fd = fd;
    tmphandle->shmPtr   = hptr;
    tmphandle->devShmPtr = dptr;
    tmphandle->shmSize   = shmSize;
    tmphandle->realShmSize = realShmSize;
    tmphandle->refcount = (hptr != NULL) ? (int*)(hptr + shmSize) : NULL;
    if (create) {
      int slen = (int)strlen(shmPath);
      tmphandle->shmPath = (char*)malloc(slen + 1);
      memcpy(tmphandle->shmPath, shmPath, slen + 1);
      if (hptr) memset(hptr, 0, shmSize);
    } else {
      tmphandle->shmPath = NULL;
    }
    goto exit_posix;

  fail_posix:
    WARN("Error while %s shared memory segment %s (size %ld), error: %s (%d)", create ? "creating" : "attaching to",
         shmPath, shmSize, strerror(errno), errno);
    if (tmphandle) {
      tmphandle->fd = fd;
      tmphandle->shmPtr = hptr;
      tmphandle->devShmPtr = dptr;
      tmphandle->shmSize = shmSize;
      tmphandle->realShmSize = realShmSize;
      tmphandle->refcount = (hptr != NULL) ? (int*)(hptr + shmSize) : NULL;
      tmphandle->shmPath = NULL;
      (void)ncclShmClose((ncclShmHandle_t)tmphandle);
      tmphandle = NULL;
    }
    hptr = NULL;
    dptr = NULL;

  exit_posix:
    *shmPtr = hptr;
    if (devShmPtr) *devShmPtr = dptr;
    *handle = (ncclShmHandle_t)tmphandle;
    return ret;
  }
#endif /* NCCL_OS_WINDOWS */
ret_early:
  return ret;
}

ncclResult_t ncclShmClose(ncclShmHandle_t handle) {
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle = (struct shmHandleInternal*)handle;
  if (tmphandle) {
#ifdef NCCL_OS_WINDOWS
    if (tmphandle->cudaAllocated) {
      // Allocated via cudaHostAlloc — just free with cudaFreeHost.
      if (tmphandle->shmPtr) CUDACHECK(cudaFreeHost(tmphandle->shmPtr));
      free(tmphandle);
      return ret;
    }
    // Only delete a backing file if there actually is one (named sections have hFile == INVALID_HANDLE_VALUE).
    if (tmphandle->hFile != INVALID_HANDLE_VALUE && tmphandle->shmPath != NULL
        && tmphandle->refcount != NULL && *tmphandle->refcount > 0) {
      if (!DeleteFileA(tmphandle->shmPath)) {
        WARN("DeleteFile %s failed, error %lu", tmphandle->shmPath, GetLastError());
        ret = ncclSystemError;
      }
    }
    free(tmphandle->shmPath);

    if (tmphandle->shmPtr) {
      if (tmphandle->devShmPtr) CUDACHECK(cudaHostUnregister(tmphandle->shmPtr));
      if (!UnmapViewOfFile(tmphandle->shmPtr)) {
        WARN("UnmapViewOfFile %p failed, error %lu", tmphandle->shmPtr, GetLastError());
        ret = ncclSystemError;
      }
    }
    if (tmphandle->hMapping && tmphandle->hMapping != INVALID_HANDLE_VALUE)
      CloseHandle(tmphandle->hMapping);
    if (tmphandle->hFile && tmphandle->hFile != INVALID_HANDLE_VALUE)
      CloseHandle(tmphandle->hFile);
#else
    if (tmphandle->fd >= 0) {
      close(tmphandle->fd);
      if (tmphandle->shmPath != NULL && tmphandle->refcount != NULL && *tmphandle->refcount > 0) {
        if (unlink(tmphandle->shmPath) != 0) {
          WARN("unlink shared memory %s failed, error: %s (%d)", tmphandle->shmPath, strerror(errno), errno);
          ret = ncclSystemError;
        }
      }
      free(tmphandle->shmPath);
    }

    if (tmphandle->shmPtr) {
      if (tmphandle->devShmPtr) CUDACHECK(cudaHostUnregister(tmphandle->shmPtr));
      if (munmap(tmphandle->shmPtr, tmphandle->realShmSize) != 0) {
        WARN("munmap of shared memory %p size %ld failed, error: %s (%d)", tmphandle->shmPtr, tmphandle->realShmSize, strerror(errno), errno);
        ret = ncclSystemError;
      }
    }
#endif /* NCCL_OS_WINDOWS */
    free(tmphandle);
  }
  return ret;
}

ncclResult_t ncclShmUnlink(ncclShmHandle_t handle) {
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle = (struct shmHandleInternal*)handle;
  if (tmphandle) {
    if (tmphandle->shmPath != NULL && tmphandle->refcount != NULL && *tmphandle->refcount > 0) {
#ifdef NCCL_OS_WINDOWS
      // Only file-backed sections have a real filesystem path to delete.
      // Named page-file sections (hFile == INVALID_HANDLE_VALUE) have no file
      // system entry and auto-destroy when the last handle is closed.
      if (tmphandle->hFile != INVALID_HANDLE_VALUE) {
        if (!DeleteFileA(tmphandle->shmPath)) {
          WARN("DeleteFile %s failed, error %lu", tmphandle->shmPath, GetLastError());
          ret = ncclSystemError;
        }
      }
#else
      if (unlink(tmphandle->shmPath) != 0) {
        WARN("unlink shared memory %s failed, error: %s (%d)", tmphandle->shmPath, strerror(errno), errno);
        ret = ncclSystemError;
      }
#endif
      free(tmphandle->shmPath);
      tmphandle->shmPath = NULL;
    }
  }
  return ret;
}

ncclResult_t ncclShmemAllgather(struct ncclComm *comm, struct ncclShmemCollBuff *shmem, void *sendbuff, void *recvbuff, size_t typeSize) {
  ncclResult_t ret = ncclSuccess;
  int nextRound = shmem->round + 1;
  int curIndex = shmem->round % 2;
  bool done;
  int index = 0;
  size_t maxTypeSize = shmem->maxTypeSize;

  if (comm == NULL || shmem == NULL || sendbuff == NULL || recvbuff == NULL || maxTypeSize < typeSize) {
    ret = ncclInvalidArgument;
    goto exit;
  }

  memcpy((char*)shmem->ptr[curIndex] + comm->localRank * maxTypeSize, sendbuff, typeSize);
  /* reset the previous round and notify I arrive this round */
  COMPILER_ATOMIC_STORE((int*)((char*)shmem->cnt[curIndex] + CACHE_LINE_SIZE * comm->localRank), nextRound, std::memory_order_release);

  do {
    done = true;
    for (int i = index; i < comm->localRanks; ++i) {
      if (i != comm->localRank && COMPILER_ATOMIC_LOAD((int*)((char*)shmem->cnt[curIndex] + CACHE_LINE_SIZE * i), std::memory_order_acquire) < nextRound) {
        done = false;
        index = i;
        break;
      }
    }
  } while (!done);

  for (int i = 0; i < comm->localRanks; ++i) {
    memcpy((uint8_t*)recvbuff + i * typeSize, (uint8_t*)shmem->ptr[curIndex] + i * maxTypeSize, typeSize);
  }
  shmem->round = nextRound;

exit:
  return ret;
}
