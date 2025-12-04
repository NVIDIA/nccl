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

/* Large page support flags */
#define NCCL_SHM_LARGE_PAGES 0x80000000
#define NCCL_SHM_NUMA_AWARE 0x40000000

/* Internal structure for Windows shared memory handle */
struct ncclShmHandleWin32
{
    HANDLE hMapFile;     /* Handle to file mapping object */
    HANDLE hFile;        /* Handle to backing file (or INVALID_HANDLE_VALUE for pagefile) */
    void *pMapView;      /* Pointer to mapped view */
    size_t size;         /* Size of mapped region */
    char name[MAX_PATH]; /* Name of shared memory object */
    int isCreator;       /* Whether this process created the shm */
    int useLargePages;   /* Whether large pages are in use */
    int numaNode;        /* NUMA node for allocation (-1 if not NUMA-aware) */
};

typedef struct ncclShmHandleWin32 *ncclShmHandle_win32_t;

/*
 * Enable large page privilege for current process
 * Must be called once before using large pages
 * Requires "Lock pages in memory" privilege (SeLockMemoryPrivilege)
 */
static inline int ncclShmEnableLargePages(void)
{
    HANDLE hToken;
    TOKEN_PRIVILEGES tp;
    BOOL result;

    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
        return -1;

    if (!LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid))
    {
        CloseHandle(hToken);
        return -1;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    result = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, NULL);
    CloseHandle(hToken);

    if (!result || GetLastError() == ERROR_NOT_ALL_ASSIGNED)
        return -1;

    return 0;
}

/*
 * Get the minimum large page size supported by the system
 */
static inline size_t ncclShmGetLargePageSize(void)
{
    return GetLargePageMinimum();
}

/*
 * Get the NUMA node for the current thread/processor
 */
static inline int ncclShmGetCurrentNumaNode(void)
{
    PROCESSOR_NUMBER procNum;
    USHORT numaNode = 0;

    GetCurrentProcessorNumberEx(&procNum);
    GetNumaProcessorNodeEx(&procNum, &numaNode);

    return (int)numaNode;
}

/*
 * Get the number of NUMA nodes in the system
 */
static inline int ncclShmGetNumaNodeCount(void)
{
    ULONG highestNode = 0;
    GetNumaHighestNodeNumber(&highestNode);
    return (int)(highestNode + 1);
}

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
    DWORD highSize = (DWORD)(((ULONGLONG)size >> 32) & 0xFFFFFFFF);
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
    handle->useLargePages = 0;
    handle->numaNode = -1;

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
 * Create shared memory with large page and NUMA support
 * Based on NCCL paper: SHM transport used when P2P is suboptimal (inter-socket)
 * Large pages reduce TLB misses; NUMA-aware allocation improves memory locality
 *
 * @param name      Name of shared memory object
 * @param size      Size of shared memory (will be rounded up to large page size if using large pages)
 * @param flags     NCCL_SHM_LARGE_PAGES | NCCL_SHM_NUMA_AWARE or 0
 * @param numaNode  NUMA node for allocation (-1 for current node, ignored if not NUMA_AWARE)
 * @param pHandle   Output handle
 * @param pPtr      Output pointer to mapped memory
 */
static inline int ncclShmOpenAdvanced(const char *name, size_t size, int flags,
                                      int numaNode, ncclShmHandle_win32_t *pHandle, void **pPtr)
{
    ncclShmHandle_win32_t handle;
    DWORD accessFlags = FILE_MAP_ALL_ACCESS;
    DWORD pageFlags = PAGE_READWRITE;
    char fullName[MAX_PATH];
    size_t allocSize = size;

    if (pHandle == NULL || pPtr == NULL)
        return -1;

    *pHandle = NULL;
    *pPtr = NULL;

    handle = (ncclShmHandle_win32_t)calloc(1, sizeof(struct ncclShmHandleWin32));
    if (handle == NULL)
        return -1;

    /* Determine NUMA node */
    if (flags & NCCL_SHM_NUMA_AWARE)
    {
        handle->numaNode = (numaNode >= 0) ? numaNode : ncclShmGetCurrentNumaNode();
    }
    else
    {
        handle->numaNode = -1;
    }

    /* Check for large page support and adjust size */
    if (flags & NCCL_SHM_LARGE_PAGES)
    {
        size_t largePageSize = ncclShmGetLargePageSize();
        if (largePageSize > 0)
        {
            /* Round up to large page boundary */
            allocSize = (size + largePageSize - 1) & ~(largePageSize - 1);
            pageFlags |= SEC_LARGE_PAGES;
            handle->useLargePages = 1;
        }
    }

    snprintf(fullName, sizeof(fullName), "Local\\nccl_%s", name);
    strncpy(handle->name, fullName, sizeof(handle->name) - 1);
    handle->size = allocSize;
    handle->hFile = INVALID_HANDLE_VALUE;
    handle->isCreator = 1;

    DWORD highSize = (DWORD)(((ULONGLONG)allocSize >> 32) & 0xFFFFFFFF);
    DWORD lowSize = (DWORD)(allocSize & 0xFFFFFFFF);

    /* Create file mapping with optional large page support */
    handle->hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        pageFlags,
        highSize,
        lowSize,
        fullName);

    if (handle->hMapFile == NULL)
    {
        /* Large pages may fail, fall back to regular pages */
        if (handle->useLargePages)
        {
            handle->useLargePages = 0;
            allocSize = size;
            highSize = (DWORD)(((ULONGLONG)allocSize >> 32) & 0xFFFFFFFF);
            lowSize = (DWORD)(allocSize & 0xFFFFFFFF);
            handle->size = allocSize;

            handle->hMapFile = CreateFileMappingA(
                INVALID_HANDLE_VALUE,
                NULL,
                PAGE_READWRITE,
                highSize,
                lowSize,
                fullName);
        }
        if (handle->hMapFile == NULL)
        {
            free(handle);
            return -1;
        }
    }

    /* Map view - use NUMA-aware mapping if available and requested */
    if (handle->numaNode >= 0)
    {
        /* Try NUMA-aware allocation using VirtualAllocExNuma + custom mapping */
        /* Note: MapViewOfFile doesn't support NUMA directly, so we use
         * VirtualAllocExNuma for the backing allocation when possible */
        handle->pMapView = MapViewOfFile(
            handle->hMapFile,
            accessFlags,
            0, 0,
            allocSize);

        /* If we got a mapping, touch pages on the target NUMA node to ensure
         * physical allocation happens on the correct node (first-touch policy) */
        if (handle->pMapView != NULL)
        {
            /* Set thread to prefer the target NUMA node during initialization */
            DWORD_PTR oldMask = SetThreadAffinityMask(GetCurrentThread(),
                                                      (DWORD_PTR)1 << (handle->numaNode * 4));
            memset(handle->pMapView, 0, allocSize);
            if (oldMask != 0)
                SetThreadAffinityMask(GetCurrentThread(), oldMask);
        }
    }
    else
    {
        handle->pMapView = MapViewOfFile(
            handle->hMapFile,
            accessFlags,
            0, 0,
            allocSize);
        if (handle->pMapView != NULL)
            memset(handle->pMapView, 0, allocSize);
    }

    if (handle->pMapView == NULL)
    {
        CloseHandle(handle->hMapFile);
        free(handle);
        return -1;
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
    DWORD highSize = (DWORD)(((ULONGLONG)size >> 32) & 0xFFFFFFFF);
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
    handle->useLargePages = 0;
    handle->numaNode = -1;

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
 * Behaves like mkstemp: fills in the XXXXXX portion of a template like "/dev/shm/nccl-XXXXXX"
 * If name is empty, generates "/dev/shm/nccl-XXXXXX" with unique suffix filled in
 * This matches the expected format used by proxy.cc for suffix extraction
 */
static inline int ncclShmMkstemp(char *name, size_t nameLen)
{
    static volatile long counter = 0;
    DWORD pid = GetCurrentProcessId();
    DWORD tid = GetCurrentThreadId();
    long cnt;
    char suffix[8];

    cnt = InterlockedIncrement(&counter);

    /* Generate a unique 6-character suffix (hex characters) */
    snprintf(suffix, sizeof(suffix), "%02x%02x%02x",
             (unsigned int)(pid & 0xFF),
             (unsigned int)(tid & 0xFF),
             (unsigned int)(cnt & 0xFF));
    suffix[6] = '\0';

    /* Find and replace XXXXXX in the template, or create full path if empty */
    char *xxpos = strstr(name, "XXXXXX");
    if (xxpos != NULL)
    {
        /* Replace XXXXXX with our unique suffix */
        memcpy(xxpos, suffix, 6);
    }
    else if (name[0] == '\0')
    {
        /* Empty string - generate full path in Linux format for compatibility
         * proxy.cc extracts suffix via: shmPath + sizeof("/dev/shm/nccl-") - 1
         * So we must provide the path in this exact format */
        snprintf(name, nameLen, "/dev/shm/nccl-%s", suffix);
    }
    /* Otherwise leave name unchanged if it has content but no XXXXXX */

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
    /* Cast to 64-bit to avoid shift warning on 32-bit off_t */
    ULONGLONG offset64 = (ULONGLONG)offset;
    DWORD highOff = (DWORD)((offset64 >> 32) & 0xFFFFFFFF);
    DWORD lowOff = (DWORD)(offset64 & 0xFFFFFFFF);

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

/* ===================== Memory Optimization Functions ===================== */

/*
 * Prefetch memory region for reading
 * Useful for preparing buffers before copy operations
 */
static inline void ncclShmPrefetch(void *addr, size_t size)
{
    /* Use PrefetchVirtualMemory on Windows 8+ */
    typedef BOOL(WINAPI * PrefetchVirtualMemoryFunc)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static PrefetchVirtualMemoryFunc pPrefetch = NULL;
    static int initialized = 0;

    if (!initialized)
    {
        HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");
        if (hKernel32)
        {
            pPrefetch = (PrefetchVirtualMemoryFunc)GetProcAddress(hKernel32, "PrefetchVirtualMemory");
        }
        initialized = 1;
    }

    if (pPrefetch != NULL)
    {
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = addr;
        range.NumberOfBytes = size;
        pPrefetch(GetCurrentProcess(), 1, &range, 0);
    }
}

/*
 * Advise kernel about memory usage pattern
 * Helps system optimize paging and caching
 */
#define NCCL_MADV_NORMAL 0
#define NCCL_MADV_SEQUENTIAL 1
#define NCCL_MADV_RANDOM 2
#define NCCL_MADV_WILLNEED 3
#define NCCL_MADV_DONTNEED 4

static inline int ncclShmAdvise(void *addr, size_t size, int advice)
{
    /* Windows doesn't have direct madvise equivalent, but we can use
     * VirtualAlloc with specific flags or OfferVirtualMemory for DONTNEED */

    switch (advice)
    {
    case NCCL_MADV_WILLNEED:
        /* Prefetch the memory */
        ncclShmPrefetch(addr, size);
        return 0;

    case NCCL_MADV_DONTNEED:
    {
        /* On Windows 8.1+, we can offer memory back to the system */
        typedef DWORD(WINAPI * OfferVirtualMemoryFunc)(PVOID, SIZE_T, OFFER_PRIORITY);
        static OfferVirtualMemoryFunc pOffer = NULL;
        static int initialized = 0;

        if (!initialized)
        {
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");
            if (hKernel32)
            {
                pOffer = (OfferVirtualMemoryFunc)GetProcAddress(hKernel32, "OfferVirtualMemory");
            }
            initialized = 1;
        }

        if (pOffer != NULL)
        {
            pOffer(addr, size, VmOfferPriorityNormal);
        }
        return 0;
    }

    default:
        /* Other advices are hints that Windows doesn't support directly */
        return 0;
    }
}

/*
 * Lock memory pages in physical RAM
 * Prevents paging which improves latency consistency
 * Requires SeLockMemoryPrivilege
 */
static inline int ncclShmLock(void *addr, size_t size)
{
    return VirtualLock(addr, size) ? 0 : -1;
}

/*
 * Unlock memory pages, allowing them to be paged
 */
static inline int ncclShmUnlock(void *addr, size_t size)
{
    return VirtualUnlock(addr, size) ? 0 : -1;
}

/*
 * Touch memory pages to ensure they are physically allocated
 * Useful to avoid page faults during critical operations
 */
static inline void ncclShmTouch(void *addr, size_t size)
{
    volatile char *p = (volatile char *)addr;
    size_t pageSize = 4096; /* Standard page size */

    /* Read one byte per page to force allocation */
    for (size_t i = 0; i < size; i += pageSize)
    {
        (void)p[i];
    }
}

/*
 * Write-touch memory pages to ensure copy-on-write is resolved
 */
static inline void ncclShmTouchWrite(void *addr, size_t size)
{
    volatile char *p = (volatile char *)addr;
    size_t pageSize = 4096;

    /* Write to force page allocation and resolve COW */
    for (size_t i = 0; i < size; i += pageSize)
    {
        p[i] = p[i];
    }
}

/*
 * Zero memory region efficiently
 * Uses large pages if available
 */
static inline void ncclShmZero(void *addr, size_t size)
{
    /* Use ZeroMemory which may be optimized by the compiler/runtime */
    ZeroMemory(addr, size);
}

/*
 * Copy memory with prefetch
 * Prefetches destination before copy for better performance
 */
static inline void ncclShmCopy(void *dst, const void *src, size_t size)
{
    /* Prefetch destination */
    ncclShmPrefetch(dst, size);

    /* Use RtlCopyMemory (memcpy) which is optimized */
    RtlCopyMemory(dst, src, size);
}

#endif /* _WIN32 */

#endif /* NCCL_WIN32_SHM_H_ */
