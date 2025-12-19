/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PLATFORM_LINUX_SHM_H_
#define NCCL_PLATFORM_LINUX_SHM_H_

/*
 * Linux Shared Memory Optimizations for NCCL
 *
 * Provides NUMA-aware and huge page shared memory allocation
 * for optimal SHM transport performance.
 */

#ifndef NCCL_PLATFORM_LINUX
#error "This header is for Linux only"
#endif

#define _GNU_SOURCE
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <numa.h>
#include <numaif.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>

/* Huge page sizes */
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

/* SHM optimization flags */
#define NCCL_SHM_NUMA_AWARE 0x01
#define NCCL_SHM_HUGE_PAGES 0x02
#define NCCL_SHM_HUGE_1GB 0x04
#define NCCL_SHM_PREFAULT 0x08

/* SHM handle structure */
typedef struct ncclShmHandle_linux
{
    int fd;         /* File descriptor */
    void *ptr;      /* Mapped pointer */
    size_t size;    /* Size of mapping */
    char name[256]; /* SHM object name */
    int flags;      /* Optimization flags used */
    int numaNode;   /* NUMA node (-1 if not NUMA-aware) */
} ncclShmHandle_linux_t;

/* ========================================================================== */
/*                    NUMA Information Functions                              */
/* ========================================================================== */

/*
 * ncclShmGetCurrentNumaNode - Get NUMA node of current thread
 *
 * Returns: NUMA node number, or 0 if NUMA not available
 */
static inline int ncclShmGetCurrentNumaNode(void)
{
    if (numa_available() < 0)
        return 0;
    return numa_node_of_cpu(sched_getcpu());
}

/*
 * ncclShmGetNumaNodeCount - Get number of NUMA nodes
 *
 * Returns: Number of NUMA nodes, or 1 if NUMA not available
 */
static inline int ncclShmGetNumaNodeCount(void)
{
    if (numa_available() < 0)
        return 1;
    return numa_max_node() + 1;
}

/*
 * ncclShmGetHugePageSize - Get huge page size
 *
 * Returns: Huge page size in bytes (typically 2MB)
 */
static inline size_t ncclShmGetHugePageSize(void)
{
    /* Read from /proc/meminfo */
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f)
        return 2 * 1024 * 1024; /* Default 2MB */

    char line[256];
    size_t size = 2 * 1024 * 1024;

    while (fgets(line, sizeof(line), f))
    {
        if (strncmp(line, "Hugepagesize:", 13) == 0)
        {
            unsigned long kb;
            if (sscanf(line + 13, "%lu", &kb) == 1)
                size = kb * 1024;
            break;
        }
    }

    fclose(f);
    return size;
}

/*
 * ncclShmGetAvailableHugePages - Get number of available huge pages
 *
 * Returns: Number of free huge pages
 */
static inline size_t ncclShmGetAvailableHugePages(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f)
        return 0;

    char line[256];
    size_t free_pages = 0;

    while (fgets(line, sizeof(line), f))
    {
        if (strncmp(line, "HugePages_Free:", 15) == 0)
        {
            unsigned long pages;
            if (sscanf(line + 15, "%lu", &pages) == 1)
                free_pages = pages;
            break;
        }
    }

    fclose(f);
    return free_pages;
}

/* ========================================================================== */
/*                    Shared Memory Functions                                 */
/* ========================================================================== */

/*
 * ncclShmOpen - Open/create shared memory (basic)
 *
 * Parameters:
 *   name   - Name for the shared memory object
 *   size   - Size in bytes
 *   create - 1 to create, 0 to open existing
 *   handle - Output handle
 *   ptr    - Output mapped pointer
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclShmOpen(const char *name, size_t size, int create,
                              ncclShmHandle_linux_t *handle, void **ptr)
{
    if (!handle || !ptr || !name)
        return -1;

    memset(handle, 0, sizeof(*handle));
    handle->numaNode = -1;

    /* Format name with leading slash for POSIX shm */
    snprintf(handle->name, sizeof(handle->name), "/%s", name);
    handle->size = size;

    /* Open or create shared memory */
    int flags = create ? (O_CREAT | O_RDWR) : O_RDWR;
    int fd = shm_open(handle->name, flags, 0600);
    if (fd < 0)
        return -1;

    handle->fd = fd;

    /* Set size if creating */
    if (create)
    {
        if (ftruncate(fd, size) != 0)
        {
            close(fd);
            shm_unlink(handle->name);
            return -1;
        }
    }

    /* Map the memory */
    void *mapped = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED)
    {
        close(fd);
        if (create)
            shm_unlink(handle->name);
        return -1;
    }

    handle->ptr = mapped;
    *ptr = mapped;

    return 0;
}

/*
 * ncclShmOpenAdvanced - Open/create shared memory with optimizations
 *
 * Parameters:
 *   name     - Name for the shared memory object
 *   size     - Size in bytes
 *   flags    - Optimization flags (NCCL_SHM_*)
 *   numaNode - NUMA node for allocation (-1 for current)
 *   handle   - Output handle
 *   ptr      - Output mapped pointer
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclShmOpenAdvanced(const char *name, size_t size, int flags,
                                      int numaNode, ncclShmHandle_linux_t *handle, void **ptr)
{
    if (!handle || !ptr || !name)
        return -1;

    memset(handle, 0, sizeof(*handle));
    handle->flags = flags;
    handle->numaNode = numaNode;

    /* Format name */
    snprintf(handle->name, sizeof(handle->name), "/%s", name);

    /* Determine actual NUMA node */
    int targetNode = numaNode;
    if ((flags & NCCL_SHM_NUMA_AWARE) && targetNode < 0)
        targetNode = ncclShmGetCurrentNumaNode();

    /* Adjust size for huge pages */
    size_t alignedSize = size;
    if (flags & NCCL_SHM_HUGE_PAGES)
    {
        size_t pageSize = ncclShmGetHugePageSize();
        alignedSize = (size + pageSize - 1) & ~(pageSize - 1);
    }
    handle->size = alignedSize;

    /* Try huge page allocation first if requested */
    void *mapped = MAP_FAILED;

    if (flags & NCCL_SHM_HUGE_PAGES)
    {
        int mapFlags = MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB;
        if (flags & NCCL_SHM_HUGE_1GB)
            mapFlags |= MAP_HUGE_1GB;
        else
            mapFlags |= MAP_HUGE_2MB;

        if (flags & NCCL_SHM_PREFAULT)
            mapFlags |= MAP_POPULATE;

        mapped = mmap(NULL, alignedSize, PROT_READ | PROT_WRITE, mapFlags, -1, 0);
    }

    /* Fall back to regular shared memory if huge pages failed */
    if (mapped == MAP_FAILED)
    {
        int fd = shm_open(handle->name, O_CREAT | O_RDWR, 0600);
        if (fd < 0)
            return -1;

        handle->fd = fd;

        if (ftruncate(fd, alignedSize) != 0)
        {
            close(fd);
            shm_unlink(handle->name);
            return -1;
        }

        int mapFlags = MAP_SHARED;
        if (flags & NCCL_SHM_PREFAULT)
            mapFlags |= MAP_POPULATE;

        mapped = mmap(NULL, alignedSize, PROT_READ | PROT_WRITE, mapFlags, fd, 0);
        if (mapped == MAP_FAILED)
        {
            close(fd);
            shm_unlink(handle->name);
            return -1;
        }
    }
    else
    {
        handle->fd = -1; /* Anonymous mapping */
    }

    handle->ptr = mapped;

    /* Apply NUMA policy if requested */
    if ((flags & NCCL_SHM_NUMA_AWARE) && numa_available() >= 0 && targetNode >= 0)
    {
        unsigned long nodemask = 1UL << targetNode;
        mbind(mapped, alignedSize, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, MPOL_MF_MOVE);
        handle->numaNode = targetNode;
    }

    /* Prefault pages if requested */
    if (flags & NCCL_SHM_PREFAULT)
    {
        memset(mapped, 0, alignedSize);
    }

    *ptr = mapped;
    return 0;
}

/*
 * ncclShmClose - Close and unmap shared memory
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclShmClose(ncclShmHandle_linux_t *handle)
{
    if (!handle)
        return -1;

    int result = 0;

    if (handle->ptr && handle->ptr != MAP_FAILED)
    {
        if (munmap(handle->ptr, handle->size) != 0)
            result = -1;
    }

    if (handle->fd >= 0)
    {
        close(handle->fd);
        shm_unlink(handle->name);
    }

    memset(handle, 0, sizeof(*handle));
    handle->fd = -1;

    return result;
}

/*
 * ncclShmPrefetch - Prefetch shared memory to CPU cache
 *
 * Uses madvise to hint the kernel about access patterns.
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclShmPrefetch(void *ptr, size_t size)
{
    return madvise(ptr, size, MADV_WILLNEED);
}

/*
 * ncclShmSetSequential - Hint sequential access pattern
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclShmSetSequential(void *ptr, size_t size)
{
    return madvise(ptr, size, MADV_SEQUENTIAL);
}

/*
 * ncclShmSetRandom - Hint random access pattern
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclShmSetRandom(void *ptr, size_t size)
{
    return madvise(ptr, size, MADV_RANDOM);
}

#endif /* NCCL_PLATFORM_LINUX_SHM_H_ */
