/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PLATFORM_LINUX_THREAD_H_
#define NCCL_PLATFORM_LINUX_THREAD_H_

/*
 * Linux Thread Optimizations for NCCL
 *
 * Provides thread priority, affinity, and NUMA-aware thread management.
 */

#ifndef NCCL_PLATFORM_LINUX
#error "This header is for Linux only"
#endif

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#include <errno.h>

/* Real-time scheduling policies */
#ifndef SCHED_FIFO
#define SCHED_FIFO 1
#endif

#ifndef SCHED_RR
#define SCHED_RR 2
#endif

/* ========================================================================== */
/*                    Thread Priority Functions                               */
/* ========================================================================== */

/*
 * ncclThreadSetPriority - Set thread priority
 *
 * Priority levels:
 * - 7: Highest (real-time FIFO)
 * - 6: High (real-time RR)
 * - 5: Above normal
 * - 4: Normal
 * - 3: Below normal
 * - 1-2: Low
 * - 0: Idle
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclThreadSetPriority(pthread_t thread, int priority)
{
    struct sched_param param;
    int policy;

    if (priority >= 7)
    {
        /* Real-time FIFO - highest priority */
        policy = SCHED_FIFO;
        param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    }
    else if (priority == 6)
    {
        /* Real-time RR */
        policy = SCHED_RR;
        param.sched_priority = sched_get_priority_max(SCHED_RR) / 2;
    }
    else
    {
        /* Normal scheduling with nice value */
        policy = SCHED_OTHER;
        param.sched_priority = 0;

        /* Set nice value for normal threads */
        int nice_val;
        switch (priority)
        {
        case 5:
            nice_val = -10;
            break;
        case 4:
            nice_val = 0;
            break;
        case 3:
            nice_val = 5;
            break;
        case 2:
            nice_val = 10;
            break;
        case 1:
            nice_val = 15;
            break;
        default:
            nice_val = 19;
            break;
        }
        setpriority(PRIO_PROCESS, 0, nice_val);
    }

    return pthread_setschedparam(thread, policy, &param);
}

/*
 * ncclThreadSetCurrentPriority - Set current thread's priority
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclThreadSetCurrentPriority(int priority)
{
    return ncclThreadSetPriority(pthread_self(), priority);
}

/*
 * ncclThreadGetPriority - Get thread priority
 *
 * Returns: Priority level (0-7), or -1 on failure
 */
static inline int ncclThreadGetPriority(pthread_t thread)
{
    struct sched_param param;
    int policy;

    if (pthread_getschedparam(thread, &policy, &param) != 0)
        return -1;

    if (policy == SCHED_FIFO)
        return 7;
    if (policy == SCHED_RR)
        return 6;

    /* Map nice value to priority */
    int nice_val = getpriority(PRIO_PROCESS, 0);
    if (nice_val <= -10)
        return 5;
    if (nice_val <= 0)
        return 4;
    if (nice_val <= 5)
        return 3;
    if (nice_val <= 10)
        return 2;
    if (nice_val <= 15)
        return 1;
    return 0;
}

/* ========================================================================== */
/*                    Thread Affinity Functions                               */
/* ========================================================================== */

/*
 * ncclThreadSetAffinity - Set thread CPU affinity
 *
 * Parameters:
 *   thread - Thread to modify
 *   cpuset - CPU set to bind to
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclThreadSetAffinity(pthread_t thread, const cpu_set_t *cpuset)
{
    return pthread_setaffinity_np(thread, sizeof(cpu_set_t), cpuset);
}

/*
 * ncclThreadSetAffinityMask - Set thread affinity from bitmask
 *
 * Parameters:
 *   thread - Thread to modify
 *   mask   - Bitmask of CPUs (bit N = CPU N)
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclThreadSetAffinityMask(pthread_t thread, unsigned long mask)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (int i = 0; i < (int)(sizeof(mask) * 8); i++)
    {
        if (mask & (1UL << i))
            CPU_SET(i, &cpuset);
    }

    return pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

/*
 * ncclThreadSetAffinityCpu - Bind thread to a single CPU
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclThreadSetAffinityCpu(pthread_t thread, int cpu)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    return pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

/*
 * ncclThreadGetAffinity - Get thread CPU affinity
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclThreadGetAffinity(pthread_t thread, cpu_set_t *cpuset)
{
    return pthread_getaffinity_np(thread, sizeof(cpu_set_t), cpuset);
}

/* ========================================================================== */
/*                    NUMA-Aware Thread Functions                             */
/* ========================================================================== */

/*
 * ncclThreadBindToNumaNode - Bind thread to CPUs on a NUMA node
 *
 * Parameters:
 *   thread   - Thread to modify
 *   numaNode - NUMA node to bind to
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclThreadBindToNumaNode(pthread_t thread, int numaNode)
{
    /* Read NUMA node CPU mask from sysfs */
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpumap", numaNode);

    FILE *f = fopen(path, "r");
    if (!f)
        return -1;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    /* Parse comma-separated hex values */
    char buf[256];
    if (fgets(buf, sizeof(buf), f))
    {
        /* Parse the cpumap format: comma-separated 32-bit hex values */
        char *token = strtok(buf, ",\n");
        int word = 0;
        while (token)
        {
            unsigned long val = strtoul(token, NULL, 16);
            for (int i = 0; i < 32; i++)
            {
                if (val & (1UL << i))
                    CPU_SET(word * 32 + i, &cpuset);
            }
            token = strtok(NULL, ",\n");
            word++;
        }
    }

    fclose(f);
    return pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

/*
 * ncclThreadGetNumaNode - Get NUMA node of current thread
 *
 * Returns: NUMA node number, or -1 on failure
 */
static inline int ncclThreadGetNumaNode(void)
{
    int cpu = sched_getcpu();
    if (cpu < 0)
        return -1;

    /* Read CPU's NUMA node from sysfs */
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/node0", cpu);

    /* Check which node directory exists */
    for (int node = 0; node < 256; node++)
    {
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/node%d", cpu, node);
        if (access(path, F_OK) == 0)
            return node;
    }

    return 0; /* Default to node 0 */
}

/* ========================================================================== */
/*                    Spinlock Implementation                                 */
/* ========================================================================== */

typedef volatile int ncclSpinlock_t;

/*
 * ncclSpinlockInit - Initialize spinlock
 */
static inline void ncclSpinlockInit(ncclSpinlock_t *lock)
{
    *lock = 0;
}

/*
 * ncclSpinlockLock - Acquire spinlock
 */
static inline void ncclSpinlockLock(ncclSpinlock_t *lock)
{
    while (__sync_lock_test_and_set(lock, 1))
    {
        while (*lock)
            __builtin_ia32_pause();
    }
}

/*
 * ncclSpinlockTryLock - Try to acquire spinlock
 *
 * Returns: 1 if acquired, 0 if not
 */
static inline int ncclSpinlockTryLock(ncclSpinlock_t *lock)
{
    return __sync_lock_test_and_set(lock, 1) == 0;
}

/*
 * ncclSpinlockUnlock - Release spinlock
 */
static inline void ncclSpinlockUnlock(ncclSpinlock_t *lock)
{
    __sync_lock_release(lock);
}

#endif /* NCCL_PLATFORM_LINUX_THREAD_H_ */
