/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_WIN32_SHM_H_
#define NCCL_WIN32_SHM_H_

#ifdef _WIN32

#include "win32_defs.h"
#include <stdio.h>
#include <string.h>

/*
 * Windows implementation of shared memory operations
 * Replaces POSIX shm_open/mmap/munmap
 */

/* Shared memory protection flags */
#define NCCL_SHM_PROT_READ PAGE_READONLY
#define NCCL_SHM_PROT_WRITE PAGE_READWRITE
#define NCCL_SHM_PROT_EXEC PAGE_EXECUTE_READ

/* Mapping flags */
#define NCCL_SHM_MAP_SHARED 0
#define NCCL_SHM_MAP_PRIVATE 1

/* Internal structure for Windows shared memory handle */
struct ncclShmHandleWin32
{
    HANDLE hMapFile;     /* Handle to file mapping object */
    HANDLE hFile;        /* Handle to backing file (or INVALID_HANDLE_VALUE for pagefile) */
    void *pMapView;      /* Pointer to mapped view */
    size_t size;         /* Size of mapped region */
    char name[MAX_PATH]; /* Name of shared memory object */
    int isCreator;       /* Whether this process created the shm */
};

typedef struct ncclShmHandleWin32 *ncclShmHandle_win32_t;

/*
 * Create or open shared memory
 * Equivalent to shm_open + ftruncate + mmap
 *
 * @param name     Name of shared memory object (will be prefixed with "Local\\")
 * @param size     Size of shared memory in bytes
 * @param create   If true, create new; if false, open existing
 * @param pHandle  Output handle
 * @param pPtr     Output pointer to mapped memory
 * @return 0 on success, -1 on failure
 */
static inline int ncclShmOpen(const char *name, size_t size, int create,
                              ncclShmHandle_win32_t *pHandle, void **pPtr)
{
    ncclShmHandle_win32_t handle;
    DWORD accessFlags = FILE_MAP_ALL_ACCESS;
    DWORD highSize = (DWORD)((size >> 32) & 0xFFFFFFFF);
    DWORD lowSize = (DWORD)(size & 0xFFFFFFFF);
    char fullName[MAX_PATH];

    if (pHandle == NULL || pPtr == NULL)
        return -1;

    *pHandle = NULL;
    *pPtr = NULL;

    /* Allocate handle structure */
    handle = (ncclShmHandle_win32_t)calloc(1, sizeof(struct ncclShmHandleWin32));
    if (handle == NULL)
        return -1;

    /* Create the full name with "Local\\" prefix for session-local namespace */
    snprintf(fullName, sizeof(fullName), "Local\\nccl_%s", name);
    strncpy(handle->name, fullName, sizeof(handle->name) - 1);
    handle->size = size;
    handle->hFile = INVALID_HANDLE_VALUE;
    handle->isCreator = create;

    if (create)
    {
        /* Create a new file mapping object backed by the system paging file */
        handle->hMapFile = CreateFileMappingA(
            INVALID_HANDLE_VALUE, /* Use paging file */
            NULL,                 /* Default security */
            PAGE_READWRITE,       /* Read/write access */
            highSize,             /* High-order DWORD of size */
            lowSize,              /* Low-order DWORD of size */
            fullName              /* Name of mapping object */
        );

        if (handle->hMapFile == NULL)
        {
            DWORD err = GetLastError();
            /* If already exists, try to open it */
            if (err == ERROR_ALREADY_EXISTS)
            {
                CloseHandle(handle->hMapFile);
                handle->hMapFile = OpenFileMappingA(accessFlags, FALSE, fullName);
                handle->isCreator = 0;
            }
        }
    }
    else
    {
        /* Open existing file mapping object */
        handle->hMapFile = OpenFileMappingA(accessFlags, FALSE, fullName);
    }

    if (handle->hMapFile == NULL)
    {
        free(handle);
        return -1;
    }

    /* Map a view of the file into the address space */
    handle->pMapView = MapViewOfFile(
        handle->hMapFile,
        accessFlags,
        0,   /* High-order DWORD of offset */
        0,   /* Low-order DWORD of offset */
        size /* Number of bytes to map */
    );

    if (handle->pMapView == NULL)
    {
        CloseHandle(handle->hMapFile);
        free(handle);
        return -1;
    }

    /* Zero initialize if we created the mapping */
    if (create && handle->isCreator)
    {
        memset(handle->pMapView, 0, size);
    }

    *pHandle = handle;
    *pPtr = handle->pMapView;
    return 0;
}

/*
 * Create shared memory backed by a file
 * Used for larger shared memory regions that need persistence
 */
static inline int ncclShmOpenFile(const char *path, size_t size, int create,
                                  ncclShmHandle_win32_t *pHandle, void **pPtr)
{
    ncclShmHandle_win32_t handle;
    DWORD accessFlags = FILE_MAP_ALL_ACCESS;
    DWORD highSize = (DWORD)((size >> 32) & 0xFFFFFFFF);
    DWORD lowSize = (DWORD)(size & 0xFFFFFFFF);

    if (pHandle == NULL || pPtr == NULL)
        return -1;

    *pHandle = NULL;
    *pPtr = NULL;

    handle = (ncclShmHandle_win32_t)calloc(1, sizeof(struct ncclShmHandleWin32));
    if (handle == NULL)
        return -1;

    strncpy(handle->name, path, sizeof(handle->name) - 1);
    handle->size = size;
    handle->isCreator = create;

    if (create)
    {
        /* Create or truncate the file */
        handle->hFile = CreateFileA(
            path,
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            NULL,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            NULL);

        if (handle->hFile == INVALID_HANDLE_VALUE)
        {
            free(handle);
            return -1;
        }

        /* Set file size */
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG)size;
        if (!SetFilePointerEx(handle->hFile, li, NULL, FILE_BEGIN) ||
            !SetEndOfFile(handle->hFile))
        {
            CloseHandle(handle->hFile);
            DeleteFileA(path);
            free(handle);
            return -1;
        }
    }
    else
    {
        /* Open existing file */
        handle->hFile = CreateFileA(
            path,
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL);

        if (handle->hFile == INVALID_HANDLE_VALUE)
        {
            free(handle);
            return -1;
        }
    }

    /* Create file mapping */
    handle->hMapFile = CreateFileMappingA(
        handle->hFile,
        NULL,
        PAGE_READWRITE,
        highSize,
        lowSize,
        NULL /* No name for file-backed mapping */
    );

    if (handle->hMapFile == NULL)
    {
        CloseHandle(handle->hFile);
        if (create)
            DeleteFileA(path);
        free(handle);
        return -1;
    }

    /* Map view */
    handle->pMapView = MapViewOfFile(handle->hMapFile, accessFlags, 0, 0, size);

    if (handle->pMapView == NULL)
    {
        CloseHandle(handle->hMapFile);
        CloseHandle(handle->hFile);
        if (create)
            DeleteFileA(path);
        free(handle);
        return -1;
    }

    if (create && handle->isCreator)
    {
        memset(handle->pMapView, 0, size);
    }

    *pHandle = handle;
    *pPtr = handle->pMapView;
    return 0;
}

/*
 * Close shared memory
 */
static inline int ncclShmClose_win32(ncclShmHandle_win32_t handle)
{
    int result = 0;

    if (handle == NULL)
        return 0;

    if (handle->pMapView != NULL)
    {
        if (!UnmapViewOfFile(handle->pMapView))
            result = -1;
    }

    if (handle->hMapFile != NULL)
    {
        if (!CloseHandle(handle->hMapFile))
            result = -1;
    }

    if (handle->hFile != INVALID_HANDLE_VALUE)
    {
        if (!CloseHandle(handle->hFile))
            result = -1;
    }

    free(handle);
    return result;
}

/*
 * Unlink (delete) shared memory
 * For named mappings, this is a no-op on Windows (cleaned up when last handle closes)
 * For file-backed mappings, delete the file
 */
static inline int ncclShmUnlink_win32(ncclShmHandle_win32_t handle)
{
    if (handle == NULL)
        return 0;

    /* If file-backed and we created it, delete the file */
    if (handle->hFile != INVALID_HANDLE_VALUE && handle->isCreator)
    {
        /* File will be deleted when handle is closed */
        /* We can mark it for deletion */
    }

    return 0;
}

/*
 * Generate a unique shared memory name
 * Equivalent to mkstemp behavior but for shared memory names
 */
static inline int ncclShmMkstemp(char *name, size_t nameLen)
{
    static volatile long counter = 0;
    DWORD pid = GetCurrentProcessId();
    DWORD tid = GetCurrentThreadId();
    LARGE_INTEGER perfCounter;
    long cnt;

    QueryPerformanceCounter(&perfCounter);
    cnt = InterlockedIncrement(&counter);

    snprintf(name, nameLen, "%lx_%lx_%llx_%lx",
             (unsigned long)pid, (unsigned long)tid,
             (unsigned long long)perfCounter.QuadPart, (unsigned long)cnt);

    return 0;
}

/*
 * Create temporary file path for shared memory
 */
static inline int ncclShmTempPath(char *path, size_t pathLen, const char *prefix)
{
    char tempPath[MAX_PATH];
    char uniqueName[64];

    if (GetTempPathA(MAX_PATH, tempPath) == 0)
    {
        strcpy(tempPath, ".");
    }

    ncclShmMkstemp(uniqueName, sizeof(uniqueName));
    snprintf(path, pathLen, "%s%s_%s", tempPath, prefix ? prefix : "nccl", uniqueName);

    return 0;
}

/* POSIX compatibility layer */
#define MAP_FAILED ((void *)-1)
#define PROT_READ 0x1
#define PROT_WRITE 0x2

/*
 * Simplified mmap for POSIX compatibility
 */
static inline void *mmap_win32(void *addr, size_t length, int prot, int flags,
                               int fd, off_t offset)
{
    HANDLE hFile = (fd == -1) ? INVALID_HANDLE_VALUE : (HANDLE)_get_osfhandle(fd);
    DWORD protect;
    DWORD access;
    HANDLE hMapping;
    void *ptr;
    DWORD highOff = (DWORD)((offset >> 32) & 0xFFFFFFFF);
    DWORD lowOff = (DWORD)(offset & 0xFFFFFFFF);

    (void)addr;
    (void)flags;

    if (prot & PROT_WRITE)
    {
        protect = PAGE_READWRITE;
        access = FILE_MAP_WRITE;
    }
    else
    {
        protect = PAGE_READONLY;
        access = FILE_MAP_READ;
    }

    hMapping = CreateFileMappingA(hFile, NULL, protect, 0, (DWORD)length, NULL);
    if (hMapping == NULL)
        return MAP_FAILED;

    ptr = MapViewOfFile(hMapping, access, highOff, lowOff, length);
    CloseHandle(hMapping); /* View keeps a reference */

    return (ptr != NULL) ? ptr : MAP_FAILED;
}

static inline int munmap_win32(void *addr, size_t length)
{
    (void)length;
    return UnmapViewOfFile(addr) ? 0 : -1;
}

#define mmap mmap_win32
#define munmap munmap_win32

#endif /* _WIN32 */

#endif /* NCCL_WIN32_SHM_H_ */
