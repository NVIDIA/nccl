/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_WIN32_THREAD_H_
#define NCCL_WIN32_THREAD_H_

#ifdef _WIN32

#include "win32_defs.h"
#include <process.h>
#include <stdlib.h>
#include <string.h>

/*
 * Windows implementation of pthread-like threading primitives
 * Provides a thin wrapper around Windows threading APIs
 */

/* Thread handle type */
typedef HANDLE pthread_t;

/* Thread attributes (mostly ignored on Windows) */
typedef struct
{
    DWORD dwStackSize;
    int detached;
} pthread_attr_t;

/* Mutex types */
typedef CRITICAL_SECTION pthread_mutex_t;

typedef struct
{
    int type;
} pthread_mutexattr_t;

#define PTHREAD_MUTEX_INITIALIZER {0}

/* Condition variable */
typedef CONDITION_VARIABLE pthread_cond_t;

typedef struct
{
    int dummy;
} pthread_condattr_t;

#define PTHREAD_COND_INITIALIZER {0}

/* Read-write lock */
typedef SRWLOCK pthread_rwlock_t;

typedef struct
{
    int dummy;
} pthread_rwlockattr_t;

#define PTHREAD_RWLOCK_INITIALIZER SRWLOCK_INIT

/* Once control */
typedef INIT_ONCE pthread_once_t;
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

/* Thread function wrapper for Windows _beginthreadex */
typedef struct
{
    void *(*start_routine)(void *);
    void *arg;
} ncclThreadWrapper_t;

static unsigned __stdcall ncclThreadWrapperFunc(void *arg)
{
    ncclThreadWrapper_t *wrapper = (ncclThreadWrapper_t *)arg;
    void *(*start_routine)(void *) = wrapper->start_routine;
    void *thread_arg = wrapper->arg;
    free(wrapper);
    start_routine(thread_arg);
    return 0;
}

/* Thread creation */
static inline int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                                 void *(*start_routine)(void *), void *arg)
{
    ncclThreadWrapper_t *wrapper = (ncclThreadWrapper_t *)malloc(sizeof(ncclThreadWrapper_t));
    if (wrapper == NULL)
        return ENOMEM;

    wrapper->start_routine = start_routine;
    wrapper->arg = arg;

    DWORD stackSize = (attr != NULL) ? attr->dwStackSize : 0;

    *thread = (HANDLE)_beginthreadex(NULL, stackSize, ncclThreadWrapperFunc, wrapper, 0, NULL);

    if (*thread == NULL)
    {
        free(wrapper);
        return EAGAIN;
    }

    return 0;
}

/* Thread join */
static inline int pthread_join(pthread_t thread, void **retval)
{
    (void)retval;

    if (WaitForSingleObject(thread, INFINITE) != WAIT_OBJECT_0)
    {
        return EINVAL;
    }

    CloseHandle(thread);
    return 0;
}

/* Thread detach */
static inline int pthread_detach(pthread_t thread)
{
    CloseHandle(thread);
    return 0;
}

/* Get current thread */
static inline pthread_t pthread_self(void)
{
    return GetCurrentThread();
}

/* Thread attributes */
static inline int pthread_attr_init(pthread_attr_t *attr)
{
    if (attr == NULL)
        return EINVAL;
    attr->dwStackSize = 0;
    attr->detached = 0;
    return 0;
}

static inline int pthread_attr_destroy(pthread_attr_t *attr)
{
    (void)attr;
    return 0;
}

static inline int pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize)
{
    if (attr == NULL)
        return EINVAL;
    attr->dwStackSize = (DWORD)stacksize;
    return 0;
}

static inline int pthread_attr_getstacksize(const pthread_attr_t *attr, size_t *stacksize)
{
    if (attr == NULL || stacksize == NULL)
        return EINVAL;
    /* Return default stack size if not set */
    *stacksize = (attr->dwStackSize > 0) ? attr->dwStackSize : (1024 * 1024); /* 1MB default */
    return 0;
}

/* Mutex operations */
static inline int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr)
{
    (void)attr;
    InitializeCriticalSection(mutex);
    return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t *mutex)
{
    DeleteCriticalSection(mutex);
    return 0;
}

static inline int pthread_mutex_lock(pthread_mutex_t *mutex)
{
    EnterCriticalSection(mutex);
    return 0;
}

static inline int pthread_mutex_trylock(pthread_mutex_t *mutex)
{
    return TryEnterCriticalSection(mutex) ? 0 : EBUSY;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *mutex)
{
    LeaveCriticalSection(mutex);
    return 0;
}

/* Mutex attributes */
static inline int pthread_mutexattr_init(pthread_mutexattr_t *attr)
{
    if (attr == NULL)
        return EINVAL;
    attr->type = 0;
    return 0;
}

static inline int pthread_mutexattr_destroy(pthread_mutexattr_t *attr)
{
    (void)attr;
    return 0;
}

/* Condition variable operations */
static inline int pthread_cond_init(pthread_cond_t *cond, const pthread_condattr_t *attr)
{
    (void)attr;
    InitializeConditionVariable(cond);
    return 0;
}

static inline int pthread_cond_destroy(pthread_cond_t *cond)
{
    (void)cond;
    /* Windows condition variables don't need explicit destruction */
    return 0;
}

static inline int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex)
{
    if (!SleepConditionVariableCS(cond, mutex, INFINITE))
    {
        return EINVAL;
    }
    return 0;
}

static inline int pthread_cond_timedwait(pthread_cond_t *cond, pthread_mutex_t *mutex,
                                         const struct timespec *abstime)
{
    DWORD timeout;

    if (abstime == NULL)
    {
        timeout = INFINITE;
    }
    else
    {
        /* Calculate timeout in milliseconds from absolute time */
        FILETIME ft;
        ULARGE_INTEGER now;
        ULARGE_INTEGER target;

        GetSystemTimeAsFileTime(&ft);
        now.LowPart = ft.dwLowDateTime;
        now.HighPart = ft.dwHighDateTime;

        /* Convert abstime to Windows FILETIME (100ns intervals since Jan 1, 1601) */
        target.QuadPart = (ULONGLONG)abstime->tv_sec * 10000000ULL +
                          (ULONGLONG)abstime->tv_nsec / 100ULL +
                          116444736000000000ULL; /* Unix epoch offset */

        if (target.QuadPart <= now.QuadPart)
        {
            timeout = 0;
        }
        else
        {
            timeout = (DWORD)((target.QuadPart - now.QuadPart) / 10000ULL); /* Convert to ms */
        }
    }

    if (!SleepConditionVariableCS(cond, mutex, timeout))
    {
        if (GetLastError() == ERROR_TIMEOUT)
        {
            return ETIMEDOUT;
        }
        return EINVAL;
    }
    return 0;
}

static inline int pthread_cond_signal(pthread_cond_t *cond)
{
    WakeConditionVariable(cond);
    return 0;
}

static inline int pthread_cond_broadcast(pthread_cond_t *cond)
{
    WakeAllConditionVariable(cond);
    return 0;
}

/* Read-write lock operations */
static inline int pthread_rwlock_init(pthread_rwlock_t *rwlock, const pthread_rwlockattr_t *attr)
{
    (void)attr;
    InitializeSRWLock(rwlock);
    return 0;
}

static inline int pthread_rwlock_destroy(pthread_rwlock_t *rwlock)
{
    (void)rwlock;
    /* SRW locks don't need explicit destruction */
    return 0;
}

static inline int pthread_rwlock_rdlock(pthread_rwlock_t *rwlock)
{
    AcquireSRWLockShared(rwlock);
    return 0;
}

static inline int pthread_rwlock_wrlock(pthread_rwlock_t *rwlock)
{
    AcquireSRWLockExclusive(rwlock);
    return 0;
}

static inline int pthread_rwlock_tryrdlock(pthread_rwlock_t *rwlock)
{
    return TryAcquireSRWLockShared(rwlock) ? 0 : EBUSY;
}

static inline int pthread_rwlock_trywrlock(pthread_rwlock_t *rwlock)
{
    return TryAcquireSRWLockExclusive(rwlock) ? 0 : EBUSY;
}

static inline int pthread_rwlock_unlock(pthread_rwlock_t *rwlock)
{
    /* SRW locks require knowing which type of lock to release */
    /* This is a limitation - caller must use correct unlock function */
    /* Assume write lock for compatibility */
    ReleaseSRWLockExclusive(rwlock);
    return 0;
}

static inline int pthread_rwlock_rd_unlock(pthread_rwlock_t *rwlock)
{
    ReleaseSRWLockShared(rwlock);
    return 0;
}

static inline int pthread_rwlock_wr_unlock(pthread_rwlock_t *rwlock)
{
    ReleaseSRWLockExclusive(rwlock);
    return 0;
}

/* Once control */
static BOOL CALLBACK ncclOnceCallback(PINIT_ONCE once, PVOID param, PVOID *context)
{
    (void)once;
    (void)context;
    void (*init_routine)(void) = (void (*)(void))param;
    init_routine();
    return TRUE;
}

static inline int pthread_once(pthread_once_t *once_control, void (*init_routine)(void))
{
    if (!InitOnceExecuteOnce(once_control, ncclOnceCallback, (PVOID)init_routine, NULL))
    {
        return EINVAL;
    }
    return 0;
}

/* Thread-specific data (TLS) */
typedef DWORD pthread_key_t;

static inline int pthread_key_create(pthread_key_t *key, void (*destructor)(void *))
{
    (void)destructor; /* Windows TLS doesn't support destructors directly */
    *key = TlsAlloc();
    if (*key == TLS_OUT_OF_INDEXES)
    {
        return EAGAIN;
    }
    return 0;
}

static inline int pthread_key_delete(pthread_key_t key)
{
    return TlsFree(key) ? 0 : EINVAL;
}

static inline int pthread_setspecific(pthread_key_t key, const void *value)
{
    return TlsSetValue(key, (LPVOID)value) ? 0 : EINVAL;
}

static inline void *pthread_getspecific(pthread_key_t key)
{
    return TlsGetValue(key);
}

/* Thread naming (Windows 10+ / Server 2016+) */
typedef HRESULT(WINAPI *SetThreadDescriptionFunc)(HANDLE, PCWSTR);

static inline int pthread_setname_np(pthread_t thread, const char *name)
{
    static SetThreadDescriptionFunc pSetThreadDescription = NULL;
    static int initialized = 0;

    if (!initialized)
    {
        HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");
        if (hKernel32)
        {
            pSetThreadDescription = (SetThreadDescriptionFunc)GetProcAddress(hKernel32, "SetThreadDescription");
        }
        initialized = 1;
    }

    if (pSetThreadDescription != NULL)
    {
        wchar_t wname[256];
        MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, 256);
        pSetThreadDescription(thread, wname);
    }

    return 0;
}

/*
 * CPU affinity is defined in win32_misc.h with a more comprehensive implementation
 * that supports up to 256 CPUs. The cpu_set_t type and CPU_SET/CPU_ZERO/etc macros
 * along with sched_setaffinity/sched_getaffinity are provided there.
 */

/* ===================== Thread Priority Optimization ===================== */

/*
 * Set thread priority for performance-critical paths
 * NCCL communication threads benefit from elevated priority
 */
static inline int ncclSetThreadPriority(pthread_t thread, int priority)
{
    int winPriority;

    /* Map NCCL priority levels to Windows priority classes */
    switch (priority)
    {
    case 3: /* Critical - communication hot path */
        winPriority = THREAD_PRIORITY_TIME_CRITICAL;
        break;
    case 2: /* High - proxy threads */
        winPriority = THREAD_PRIORITY_HIGHEST;
        break;
    case 1: /* Above normal - worker threads */
        winPriority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
    case 0: /* Normal */
    default:
        winPriority = THREAD_PRIORITY_NORMAL;
        break;
    case -1: /* Below normal - background tasks */
        winPriority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
    }

    return SetThreadPriority(thread, winPriority) ? 0 : -1;
}

/*
 * Get current thread priority
 */
static inline int ncclGetThreadPriority(pthread_t thread)
{
    int winPriority = GetThreadPriority(thread);

    switch (winPriority)
    {
    case THREAD_PRIORITY_TIME_CRITICAL:
        return 3;
    case THREAD_PRIORITY_HIGHEST:
        return 2;
    case THREAD_PRIORITY_ABOVE_NORMAL:
        return 1;
    case THREAD_PRIORITY_NORMAL:
        return 0;
    case THREAD_PRIORITY_BELOW_NORMAL:
    case THREAD_PRIORITY_LOWEST:
    case THREAD_PRIORITY_IDLE:
        return -1;
    default:
        return 0;
    }
}

/*
 * Boost priority for the current thread temporarily
 * Useful during latency-sensitive operations
 */
static inline int ncclThreadPriorityBoost(int enable)
{
    return SetThreadPriorityBoost(GetCurrentThread(), !enable) ? 0 : -1;
}

/*
 * Pin thread to specific processor group (for >64 CPU systems)
 * Windows supports up to 64 processors per group
 */
static inline int ncclSetThreadIdealProcessor(pthread_t thread, int processor)
{
    DWORD prev = SetThreadIdealProcessor(thread, (DWORD)processor);
    return (prev == (DWORD)-1) ? -1 : 0;
}

/*
 * Set thread affinity to a specific NUMA node
 * Ensures thread runs on processors local to memory
 */
static inline int ncclSetThreadNumaNode(pthread_t thread, int numaNode)
{
    GROUP_AFFINITY affinity = {0};

    /* Get the processor mask for this NUMA node */
    ULONGLONG nodeMask = 0;
    if (!GetNumaNodeProcessorMask((UCHAR)numaNode, &nodeMask))
        return -1;

    affinity.Mask = (KAFFINITY)nodeMask;
    affinity.Group = 0;

    return SetThreadGroupAffinity(thread, &affinity, NULL) ? 0 : -1;
}

/* ===================== High Resolution Timer ===================== */

/*
 * Request high resolution timer for accurate timing
 * Windows default timer resolution is ~15.6ms; this can reduce to ~1ms
 * Important for latency-sensitive communication
 */
typedef NTSTATUS(NTAPI *NtSetTimerResolutionFunc)(ULONG, BOOLEAN, PULONG);
typedef NTSTATUS(NTAPI *NtQueryTimerResolutionFunc)(PULONG, PULONG, PULONG);

static inline int ncclEnableHighResolutionTimer(ULONG *actualResolution)
{
    static NtSetTimerResolutionFunc pNtSetTimerResolution = NULL;
    static int initialized = 0;

    if (!initialized)
    {
        HMODULE hNtdll = GetModuleHandleW(L"ntdll.dll");
        if (hNtdll)
        {
            pNtSetTimerResolution = (NtSetTimerResolutionFunc)GetProcAddress(hNtdll, "NtSetTimerResolution");
        }
        initialized = 1;
    }

    if (pNtSetTimerResolution == NULL)
        return -1;

    /* Request 1ms (10000 * 100ns) timer resolution */
    ULONG actual = 0;
    NTSTATUS status = pNtSetTimerResolution(10000, TRUE, &actual);
    if (actualResolution)
        *actualResolution = actual;

    return (status >= 0) ? 0 : -1;
}

static inline int ncclDisableHighResolutionTimer(void)
{
    static NtSetTimerResolutionFunc pNtSetTimerResolution = NULL;
    static int initialized = 0;

    if (!initialized)
    {
        HMODULE hNtdll = GetModuleHandleW(L"ntdll.dll");
        if (hNtdll)
        {
            pNtSetTimerResolution = (NtSetTimerResolutionFunc)GetProcAddress(hNtdll, "NtSetTimerResolution");
        }
        initialized = 1;
    }

    if (pNtSetTimerResolution == NULL)
        return -1;

    ULONG actual = 0;
    NTSTATUS status = pNtSetTimerResolution(0, FALSE, &actual);
    return (status >= 0) ? 0 : -1;
}

static inline int ncclGetTimerResolution(ULONG *minResolution, ULONG *maxResolution,
                                         ULONG *currentResolution)
{
    static NtQueryTimerResolutionFunc pNtQueryTimerResolution = NULL;
    static int initialized = 0;

    if (!initialized)
    {
        HMODULE hNtdll = GetModuleHandleW(L"ntdll.dll");
        if (hNtdll)
        {
            pNtQueryTimerResolution = (NtQueryTimerResolutionFunc)GetProcAddress(hNtdll, "NtQueryTimerResolution");
        }
        initialized = 1;
    }

    if (pNtQueryTimerResolution == NULL)
        return -1;

    NTSTATUS status = pNtQueryTimerResolution(minResolution, maxResolution, currentResolution);
    return (status >= 0) ? 0 : -1;
}

/* ===================== Spinlock Implementation ===================== */

/*
 * POSIX-compatible spinlock for Windows
 * Uses InterlockedExchange for lock-free spinning
 */
typedef volatile LONG pthread_spinlock_t;

#define PTHREAD_PROCESS_PRIVATE 0
#define PTHREAD_PROCESS_SHARED 1

static inline int pthread_spin_init(pthread_spinlock_t *lock, int pshared)
{
    (void)pshared;
    if (lock == NULL)
        return EINVAL;
    *lock = 0;
    return 0;
}

static inline int pthread_spin_destroy(pthread_spinlock_t *lock)
{
    (void)lock;
    return 0;
}

static inline int pthread_spin_lock(pthread_spinlock_t *lock)
{
    int spins = 0;

    while (InterlockedExchange(lock, 1) != 0)
    {
        /* Adaptive spinning */
        if (spins < 16)
        {
            ncclCpuPause();
        }
        else if (spins < 1000)
        {
            ncclCpuYield();
        }
        else
        {
            /* Very long wait - yield time slice */
            Sleep(0);
        }
        spins++;
    }
    return 0;
}

static inline int pthread_spin_trylock(pthread_spinlock_t *lock)
{
    return (InterlockedExchange(lock, 1) == 0) ? 0 : EBUSY;
}

static inline int pthread_spin_unlock(pthread_spinlock_t *lock)
{
    ncclStoreFence();
    *lock = 0;
    return 0;
}

/* ===================== High-Precision Sleep ===================== */

/*
 * Nanosecond-precision sleep using waitable timers
 * More accurate than Sleep() for short durations
 */
static inline int ncclNanoSleep(LONGLONG nanoseconds)
{
    HANDLE hTimer;
    LARGE_INTEGER dueTime;

    if (nanoseconds <= 0)
        return 0;

    /* CreateWaitableTimer for high precision */
    hTimer = CreateWaitableTimerEx(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (hTimer == NULL)
    {
        /* Fall back to regular waitable timer */
        hTimer = CreateWaitableTimer(NULL, TRUE, NULL);
        if (hTimer == NULL)
        {
            /* Last resort: use Sleep */
            DWORD ms = (DWORD)((nanoseconds + 999999) / 1000000);
            if (ms == 0)
                ms = 1;
            Sleep(ms);
            return 0;
        }
    }

    /* Negative value = relative time in 100ns intervals */
    dueTime.QuadPart = -(nanoseconds / 100);
    if (dueTime.QuadPart == 0)
        dueTime.QuadPart = -1;

    SetWaitableTimer(hTimer, &dueTime, 0, NULL, NULL, FALSE);
    WaitForSingleObject(hTimer, INFINITE);
    CloseHandle(hTimer);

    return 0;
}

/*
 * Microsecond-precision sleep
 */
static inline int ncclMicroSleep(LONGLONG microseconds)
{
    return ncclNanoSleep(microseconds * 1000);
}

/*
 * Busy-wait for very short delays (< 1 microsecond)
 * Uses QueryPerformanceCounter for precision
 */
static inline void ncclBusyWaitNanos(LONGLONG nanoseconds)
{
    LARGE_INTEGER freq, start, current;

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    LONGLONG targetTicks = (nanoseconds * freq.QuadPart) / 1000000000LL;
    if (targetTicks <= 0)
        return;

    do
    {
        ncclCpuPause();
        QueryPerformanceCounter(&current);
    } while ((current.QuadPart - start.QuadPart) < targetTicks);
}

/* ===================== Processor Group Support ===================== */

/*
 * Get the number of processor groups (for >64 CPU systems)
 */
static inline int ncclGetProcessorGroupCount(void)
{
    return (int)GetActiveProcessorGroupCount();
}

/*
 * Get the number of processors in a specific group
 */
static inline int ncclGetProcessorCountInGroup(int group)
{
    return (int)GetActiveProcessorCount((WORD)group);
}

/*
 * Get total logical processor count across all groups
 */
static inline int ncclGetTotalProcessorCount(void)
{
    int total = 0;
    int groups = ncclGetProcessorGroupCount();
    for (int i = 0; i < groups; i++)
    {
        total += ncclGetProcessorCountInGroup(i);
    }
    return total;
}

/*
 * Set thread affinity to a specific processor in a specific group
 * Required for systems with more than 64 logical processors
 */
static inline int ncclSetThreadGroupAffinity(HANDLE thread, int group, KAFFINITY mask)
{
    GROUP_AFFINITY affinity = {0};
    affinity.Group = (WORD)group;
    affinity.Mask = mask;

    return SetThreadGroupAffinity(thread, &affinity, NULL) ? 0 : -1;
}

/*
 * Get the current processor group and number for the calling thread
 */
static inline int ncclGetCurrentProcessorInfo(int *group, int *processor)
{
    PROCESSOR_NUMBER procNum;
    GetCurrentProcessorNumberEx(&procNum);

    if (group)
        *group = procNum.Group;
    if (processor)
        *processor = procNum.Number;

    return 0;
}

/* Define ETIMEDOUT if not defined */
#ifndef ETIMEDOUT
#define ETIMEDOUT 110
#endif

#ifndef EBUSY
#define EBUSY 16
#endif

#ifndef ENOMEM
#define ENOMEM 12
#endif

#ifndef EAGAIN
#define EAGAIN 11
#endif

#ifndef EINVAL
#define EINVAL 22
#endif

#endif /* _WIN32 */

#endif /* NCCL_WIN32_THREAD_H_ */
