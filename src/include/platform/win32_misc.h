/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_WIN32_MISC_H_
#define NCCL_WIN32_MISC_H_

#ifdef _WIN32

#include "win32_defs.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

/*
 * Miscellaneous Windows compatibility functions
 */

/* ===================== Time Functions ===================== */

/* Clock types for clock_gettime compatibility */
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

/*
 * Timespec structure - modern Windows SDK (UCRT 10.0.17134+) defines this in time.h
 * We rely on the SDK definition - if using older SDK, it will be defined under _CRT_NO_TIME_T guard
 * No need to define our own since we require Windows 10 SDK
 */

/*
 * clock_gettime implementation for Windows
 */
#ifndef HAVE_CLOCK_GETTIME
static inline int clock_gettime(int clk_id, struct timespec *tp)
{
    if (tp == NULL)
        return -1;

    if (clk_id == CLOCK_MONOTONIC)
    {
        LARGE_INTEGER freq, counter;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&counter);

        tp->tv_sec = (time_t)(counter.QuadPart / freq.QuadPart);
        tp->tv_nsec = (long)(((counter.QuadPart % freq.QuadPart) * 1000000000LL) / freq.QuadPart);
    }
    else
    {
        /* CLOCK_REALTIME */
        FILETIME ft;
        ULARGE_INTEGER uli;

        GetSystemTimeAsFileTime(&ft);
        uli.LowPart = ft.dwLowDateTime;
        uli.HighPart = ft.dwHighDateTime;

        /* Convert from 100ns intervals since Jan 1, 1601 to Unix epoch */
        uli.QuadPart -= 116444736000000000ULL;

        tp->tv_sec = (time_t)(uli.QuadPart / 10000000ULL);
        tp->tv_nsec = (long)((uli.QuadPart % 10000000ULL) * 100);
    }

    return 0;
}
#endif /* HAVE_CLOCK_GETTIME */

/*
 * nanosleep implementation for Windows
 */
static inline int nanosleep(const struct timespec *req, struct timespec *rem)
{
    DWORD ms;

    (void)rem; /* Remaining time not implemented */

    if (req == NULL)
        return -1;

    ms = (DWORD)(req->tv_sec * 1000 + req->tv_nsec / 1000000);
    if (ms == 0 && (req->tv_sec > 0 || req->tv_nsec > 0))
    {
        ms = 1; /* Minimum sleep of 1ms */
    }

    Sleep(ms);
    return 0;
}

/*
 * usleep implementation for Windows
 */
static inline int usleep(unsigned int usec)
{
    struct timespec req;
    req.tv_sec = usec / 1000000;
    req.tv_nsec = (usec % 1000000) * 1000;
    return nanosleep(&req, NULL);
}

/* ===================== Dynamic Library Loading ===================== */

/* On POSIX, these are defined in dlfcn.h */
#define RTLD_LAZY 0x0001
#define RTLD_NOW 0x0002
#define RTLD_LOCAL 0x0000
#define RTLD_GLOBAL 0x0100

/*
 * dlopen implementation for Windows
 */
static inline void *dlopen(const char *filename, int flags)
{
    HMODULE handle;
    (void)flags;

    if (filename == NULL)
    {
        return (void *)GetModuleHandle(NULL);
    }

    handle = LoadLibraryA(filename);
    return (void *)handle;
}

/*
 * dlsym implementation for Windows
 */
static inline void *dlsym(void *handle, const char *symbol)
{
    return (void *)GetProcAddress((HMODULE)handle, symbol);
}

/*
 * dlclose implementation for Windows
 */
static inline int dlclose(void *handle)
{
    return FreeLibrary((HMODULE)handle) ? 0 : -1;
}

/*
 * dlerror implementation for Windows
 */
static inline const char *dlerror(void)
{
    static char errbuf[256];
    DWORD err = GetLastError();

    if (err == 0)
        return NULL;

    FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        errbuf, sizeof(errbuf), NULL);

    /* Remove trailing newline */
    size_t len = strlen(errbuf);
    while (len > 0 && (errbuf[len - 1] == '\n' || errbuf[len - 1] == '\r'))
    {
        errbuf[--len] = '\0';
    }

    return errbuf;
}

/* ===================== Resource Limits ===================== */

/* Resource limit types */
#define RLIMIT_STACK 3

/* Resource limit values */
#define RLIM_INFINITY (~(size_t)0)

struct rlimit
{
    size_t rlim_cur; /* Soft limit */
    size_t rlim_max; /* Hard limit */
};

/*
 * getrlimit implementation for Windows
 */
static inline int getrlimit(int resource, struct rlimit *rlim)
{
    if (rlim == NULL)
        return -1;

    if (resource == RLIMIT_STACK)
    {
        /* Return default Windows stack size (1MB) */
        rlim->rlim_cur = 1024 * 1024;
        rlim->rlim_max = RLIM_INFINITY;
        return 0;
    }

    return -1;
}

/*
 * setrlimit implementation for Windows (no-op)
 */
static inline int setrlimit(int resource, const struct rlimit *rlim)
{
    (void)resource;
    (void)rlim;
    return 0; /* Silently succeed */
}

/* ===================== Hostname Functions ===================== */

/*
 * Get hostname with optional delimiter truncation
 */
static inline int ncclGetHostName(char *hostname, int maxlen, char delim)
{
    DWORD size = (DWORD)maxlen;

    if (!GetComputerNameExA(ComputerNameDnsHostname, hostname, &size))
    {
        if (gethostname(hostname, maxlen) != 0)
        {
            strncpy(hostname, "localhost", maxlen);
        }
    }

    hostname[maxlen - 1] = '\0';

    /* Truncate at delimiter if specified */
    if (delim != '\0')
    {
        char *p = strchr(hostname, delim);
        if (p)
            *p = '\0';
    }

    return 0;
}

/* ===================== Random Data ===================== */

/* NT_SUCCESS macro - should be defined in win32_defs.h but ensure it's available */
#ifndef NT_SUCCESS
#define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)
#endif

/*
 * Get random data from system
 * Uses BCrypt on Windows (cryptographically secure)
 */
static inline int ncclGetRandomData(void *buffer, size_t bytes)
{
    NTSTATUS status = BCryptGenRandom(
        NULL,
        (PUCHAR)buffer,
        (ULONG)bytes,
        BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    return NT_SUCCESS(status) ? 0 : -1;
}

/* ===================== Hash Functions ===================== */

/*
 * Get host hash (used for unique identification)
 */
static inline uint64_t ncclGetHostHash(void)
{
    char hostname[256];
    uint64_t hash = 5381;
    char *p;

    if (ncclGetHostName(hostname, sizeof(hostname), '\0') != 0)
    {
        strcpy(hostname, "localhost");
    }

    /* djb2 hash */
    for (p = hostname; *p; p++)
    {
        hash = ((hash << 5) + hash) + (unsigned char)*p;
    }

    return hash;
}

/*
 * Get process ID hash
 */
static inline uint64_t ncclGetPidHash(void)
{
    DWORD pid = GetCurrentProcessId();
    uint64_t hash = 5381;

    hash = ((hash << 5) + hash) + (pid & 0xFF);
    hash = ((hash << 5) + hash) + ((pid >> 8) & 0xFF);
    hash = ((hash << 5) + hash) + ((pid >> 16) & 0xFF);
    hash = ((hash << 5) + hash) + ((pid >> 24) & 0xFF);

    return hash;
}

/* ===================== File System ===================== */

/*
 * Create a unique temporary file
 * Equivalent to mkstemp
 */
static inline int mkstemp(char *template_name)
{
    char *xxxpos = strstr(template_name, "XXXXXX");
    static volatile long counter = 0;

    if (xxxpos == NULL)
    {
        return -1;
    }

    /* Generate unique suffix */
    DWORD pid = GetCurrentProcessId();
    long cnt = InterlockedIncrement(&counter);
    LARGE_INTEGER perfCounter;
    QueryPerformanceCounter(&perfCounter);

    snprintf(xxxpos, 7, "%02x%04x",
             (unsigned int)(pid & 0xFF),
             (unsigned int)((cnt ^ perfCounter.LowPart) & 0xFFFF));

    /* Create the file */
    HANDLE hFile = CreateFileA(
        template_name,
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        CREATE_NEW,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (hFile == INVALID_HANDLE_VALUE)
    {
        return -1;
    }

    /* Convert HANDLE to file descriptor */
    int fd = _open_osfhandle((intptr_t)hFile, 0);
    if (fd == -1)
    {
        CloseHandle(hFile);
        return -1;
    }

    return fd;
}

/*
 * Extend file to specified size
 * Equivalent to fallocate
 */
static inline int fallocate(int fd, int mode, off_t offset, off_t len)
{
    HANDLE hFile;
    LARGE_INTEGER newSize;
    LARGE_INTEGER currentPos;
    BOOL result;

    (void)mode;
    (void)offset;

    hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE)
    {
        return -1;
    }

    /* Save current position */
    currentPos.QuadPart = 0;
    SetFilePointerEx(hFile, currentPos, &currentPos, FILE_CURRENT);

    /* Set new size */
    newSize.QuadPart = offset + len;
    if (!SetFilePointerEx(hFile, newSize, NULL, FILE_BEGIN))
    {
        return -1;
    }

    result = SetEndOfFile(hFile);

    /* Restore position */
    SetFilePointerEx(hFile, currentPos, NULL, FILE_BEGIN);

    return result ? 0 : -1;
}

/*
 * fsync implementation
 */
static inline int fsync(int fd)
{
    HANDLE hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE)
        return -1;
    return FlushFileBuffers(hFile) ? 0 : -1;
}

/* ===================== Environment Variables ===================== */

/*
 * Get environment variable
 */
static inline const char *ncclGetEnv(const char *name)
{
    static __declspec(thread) char envBuffer[32768];

    DWORD ret = GetEnvironmentVariableA(name, envBuffer, sizeof(envBuffer));
    if (ret == 0 || ret >= sizeof(envBuffer))
    {
        return NULL;
    }

    return envBuffer;
}

/*
 * Set environment variable
 */
static inline int setenv(const char *name, const char *value, int overwrite)
{
    if (!overwrite && GetEnvironmentVariableA(name, NULL, 0) > 0)
    {
        return 0;
    }
    return SetEnvironmentVariableA(name, value) ? 0 : -1;
}

/*
 * Unset environment variable
 */
static inline int unsetenv(const char *name)
{
    return SetEnvironmentVariableA(name, NULL) ? 0 : -1;
}

/* ===================== Signal Handling (Stubs) ===================== */

/* Windows doesn't have Unix signals, provide stubs */
#ifndef SIGHUP
#define SIGHUP 1
#endif
#ifndef SIGQUIT
#define SIGQUIT 3
#endif
#ifndef SIGKILL
#define SIGKILL 9
#endif
#ifndef SIGTERM
#define SIGTERM 15
#endif
#ifndef SIGUSR1
#define SIGUSR1 10
#endif
#ifndef SIGUSR2
#define SIGUSR2 12
#endif

typedef void (*nccl_sighandler_t)(int);

static inline nccl_sighandler_t nccl_signal_handler(int signum, nccl_sighandler_t handler)
{
    (void)signum;
    (void)handler;
    return NULL; /* No-op on Windows */
}

/* Note: signal() is already handled in win32_defs.h */

/* ===================== CPU Affinity Support ===================== */

/*
 * Note: getpid() and gettid() are defined in win32_defs.h
 */

/*
 * Windows implementation of cpu_set_t and related macros
 * Uses KAFFINITY which supports up to 64 processors per group
 * For systems with more than 64 processors, use processor groups
 */

#ifndef CPU_SETSIZE
#define CPU_SETSIZE 1024 /* Support up to 1024 CPUs (16 processor groups) */
#endif

#define NCCL_CPUSET_WORDS ((CPU_SETSIZE + 63) / 64)

typedef struct
{
    DWORD_PTR mask[NCCL_CPUSET_WORDS];
} cpu_set_t;

/* Initialize CPU set to empty */
#define CPU_ZERO(set)                                  \
    do                                                 \
    {                                                  \
        for (int _i = 0; _i < NCCL_CPUSET_WORDS; _i++) \
            (set)->mask[_i] = 0;                       \
    } while (0)

/* Add CPU to set */
#define CPU_SET(cpu, set)                                 \
    do                                                    \
    {                                                     \
        int _word = (cpu) / 64;                           \
        int _bit = (cpu) % 64;                            \
        if (_word < NCCL_CPUSET_WORDS)                    \
            (set)->mask[_word] |= ((DWORD_PTR)1 << _bit); \
    } while (0)

/* Remove CPU from set */
#define CPU_CLR(cpu, set)                                  \
    do                                                     \
    {                                                      \
        int _word = (cpu) / 64;                            \
        int _bit = (cpu) % 64;                             \
        if (_word < NCCL_CPUSET_WORDS)                     \
            (set)->mask[_word] &= ~((DWORD_PTR)1 << _bit); \
    } while (0)

/* Check if CPU is in set */
#define CPU_ISSET(cpu, set) \
    (((cpu) / 64 < NCCL_CPUSET_WORDS) ? (((set)->mask[(cpu) / 64] >> ((cpu) % 64)) & 1) : 0)

/* Count number of CPUs in set */
static inline int CPU_COUNT(const cpu_set_t *set)
{
    int count = 0;
    for (int i = 0; i < NCCL_CPUSET_WORDS; i++)
    {
        DWORD_PTR v = set->mask[i];
        /* Brian Kernighan's algorithm */
        while (v)
        {
            v &= v - 1;
            count++;
        }
    }
    return count;
}

/* Copy CPU set */
#define CPU_COPY(dest, src)                            \
    do                                                 \
    {                                                  \
        for (int _i = 0; _i < NCCL_CPUSET_WORDS; _i++) \
            (dest)->mask[_i] = (src)->mask[_i];        \
    } while (0)

/* Check if two CPU sets are equal */
static inline int CPU_EQUAL(const cpu_set_t *a, const cpu_set_t *b)
{
    for (int i = 0; i < NCCL_CPUSET_WORDS; i++)
    {
        if (a->mask[i] != b->mask[i])
            return 0;
    }
    return 1;
}

/* AND two CPU sets */
#define CPU_AND(dest, src1, src2)                                   \
    do                                                              \
    {                                                               \
        for (int _i = 0; _i < NCCL_CPUSET_WORDS; _i++)              \
            (dest)->mask[_i] = (src1)->mask[_i] & (src2)->mask[_i]; \
    } while (0)

/* OR two CPU sets */
#define CPU_OR(dest, src1, src2)                                    \
    do                                                              \
    {                                                               \
        for (int _i = 0; _i < NCCL_CPUSET_WORDS; _i++)              \
            (dest)->mask[_i] = (src1)->mask[_i] | (src2)->mask[_i]; \
    } while (0)

/*
 * Get thread affinity
 */
static inline int ncclGetThreadAffinity(cpu_set_t *set)
{
    DWORD_PTR processAffinity, systemAffinity;

    CPU_ZERO(set);

    if (!GetProcessAffinityMask(GetCurrentProcess(), &processAffinity, &systemAffinity))
    {
        return -1;
    }

    /* For simplicity, use first processor group only */
    set->mask[0] = processAffinity;
    return 0;
}

/*
 * Set thread affinity (equivalent to pthread_setaffinity_np)
 */
static inline int ncclSetThreadAffinity(HANDLE thread, const cpu_set_t *set)
{
    /* On Windows, we can only set affinity within first 64 CPUs easily */
    DWORD_PTR mask = set->mask[0];

    if (mask == 0)
    {
        return -1; /* Cannot set empty affinity */
    }

    DWORD_PTR result = SetThreadAffinityMask(thread, mask);
    return (result != 0) ? 0 : -1;
}

/*
 * sched_setaffinity equivalent for Windows
 */
static inline int sched_setaffinity(pid_t pid, size_t cpusetsize, const cpu_set_t *mask)
{
    HANDLE hThread;
    int result;

    (void)cpusetsize;

    if (pid == 0)
    {
        hThread = GetCurrentThread();
    }
    else
    {
        hThread = OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE, (DWORD)pid);
        if (hThread == NULL)
        {
            return -1;
        }
    }

    result = ncclSetThreadAffinity(hThread, mask);

    if (pid != 0)
    {
        CloseHandle(hThread);
    }

    return result;
}

/*
 * sched_getaffinity equivalent for Windows
 */
static inline int sched_getaffinity(pid_t pid, size_t cpusetsize, cpu_set_t *mask)
{
    HANDLE hProcess;
    DWORD_PTR processAffinity, systemAffinity;

    (void)cpusetsize;

    CPU_ZERO(mask);

    if (pid == 0)
    {
        hProcess = GetCurrentProcess();
    }
    else
    {
        hProcess = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, (DWORD)pid);
        if (hProcess == NULL)
        {
            return -1;
        }
    }

    if (!GetProcessAffinityMask(hProcess, &processAffinity, &systemAffinity))
    {
        if (pid != 0)
            CloseHandle(hProcess);
        return -1;
    }

    mask->mask[0] = processAffinity;

    if (pid != 0)
    {
        CloseHandle(hProcess);
    }

    return 0;
}

/*
 * Get number of processors in the system
 */
static inline int ncclGetNumProcessors(void)
{
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return (int)sysInfo.dwNumberOfProcessors;
}

#endif /* _WIN32 */

#endif /* NCCL_WIN32_MISC_H_ */
