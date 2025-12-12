/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_WIN32_MISC_H_
#define NCCL_WIN32_MISC_H_

#ifdef _WIN32

#include "win32_defs.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <bcrypt.h>

/* Link with bcrypt library */
#pragma comment(lib, "bcrypt.lib")

/*
 * NTSTATUS type (needed for BCrypt functions)
 * Normally defined in winternl.h, but we define it here to avoid extra includes
 */
#ifndef _NTSTATUS_DEFINED
#define _NTSTATUS_DEFINED
typedef LONG NTSTATUS;
#endif

/* ========================================================================== */
/*                           String Functions                                 */
/* ========================================================================== */

/* strtok_r - reentrant version of strtok */
#ifndef strtok_r
#define strtok_r(str, delim, saveptr) strtok_s(str, delim, saveptr)
#endif

/* ========================================================================== */
/*                            Time Functions                                  */
/* ========================================================================== */

/* Clock types for clock_gettime compatibility */
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

/* clock_gettime implementation for Windows */
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

/* nanosleep implementation for Windows */
static inline int nanosleep(const struct timespec *req, struct timespec *rem)
{
    DWORD ms;
    (void)rem;

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

/* usleep implementation for Windows */
static inline int usleep(unsigned int usec)
{
    struct timespec req;
    req.tv_sec = usec / 1000000;
    req.tv_nsec = (usec % 1000000) * 1000;
    return nanosleep(&req, NULL);
}

/* timeval structure for gettimeofday */
#ifndef _WINSOCK2API_
#ifndef _TIMEVAL_DEFINED
#define _TIMEVAL_DEFINED
struct timeval
{
    long tv_sec;
    long tv_usec;
};
#endif
#endif

/* gettimeofday implementation for Windows */
static inline int gettimeofday(struct timeval *tv, void *tz)
{
    (void)tz;
    if (tv == NULL)
        return -1;

    FILETIME ft;
    ULARGE_INTEGER uli;
    GetSystemTimeAsFileTime(&ft);
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    uli.QuadPart -= 116444736000000000ULL;
    tv->tv_sec = (long)(uli.QuadPart / 10000000ULL);
    tv->tv_usec = (long)((uli.QuadPart % 10000000ULL) / 10);
    return 0;
}

/* ========================================================================== */
/*                       Dynamic Library Loading                              */
/* ========================================================================== */

#define RTLD_LAZY 0x0001
#define RTLD_NOW 0x0002
#define RTLD_LOCAL 0x0000
#define RTLD_GLOBAL 0x0100

static inline void *dlopen(const char *filename, int flags)
{
    HMODULE handle;
    (void)flags;

    if (filename == NULL)
        return (void *)GetModuleHandle(NULL);

    handle = LoadLibraryA(filename);
    return (void *)handle;
}

static inline void *dlsym(void *handle, const char *symbol)
{
    return (void *)GetProcAddress((HMODULE)handle, symbol);
}

static inline int dlclose(void *handle)
{
    return FreeLibrary((HMODULE)handle) ? 0 : -1;
}

static inline const char *dlerror(void)
{
    static char errbuf[256];
    DWORD err = GetLastError();

    if (err == 0)
        return NULL;

    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   errbuf, sizeof(errbuf), NULL);

    size_t len = strlen(errbuf);
    while (len > 0 && (errbuf[len - 1] == '\n' || errbuf[len - 1] == '\r'))
    {
        errbuf[--len] = '\0';
    }
    return errbuf;
}

/* dlinfo constants and structures for Windows */
#define RTLD_DI_LINKMAP 2

struct link_map
{
    void *l_addr;
    const char *l_name;
    void *l_ld;
    struct link_map *l_next, *l_prev;
};

static inline int dlinfo(void *handle, int request, void *info)
{
    if (request == RTLD_DI_LINKMAP)
    {
        static struct link_map lm;
        static char module_path[MAX_PATH];

        if (handle == NULL)
            return -1;

        if (GetModuleFileNameA((HMODULE)handle, module_path, MAX_PATH) == 0)
            return -1;

        lm.l_addr = handle;
        lm.l_name = module_path;
        lm.l_ld = NULL;
        lm.l_next = NULL;
        lm.l_prev = NULL;

        *(struct link_map **)info = &lm;
        return 0;
    }
    return -1;
}

/* realpath - resolve a pathname */
static inline char *realpath(const char *path, char *resolved_path)
{
    static char static_buf[MAX_PATH];
    char *buf = resolved_path ? resolved_path : static_buf;

    if (_fullpath(buf, path, MAX_PATH) == NULL)
        return NULL;
    return buf;
}

/* ========================================================================== */
/*                          Resource Limits                                   */
/* ========================================================================== */

#define RLIMIT_STACK 3
#define RLIM_INFINITY (~(size_t)0)

struct rlimit
{
    size_t rlim_cur;
    size_t rlim_max;
};

static inline int getrlimit(int resource, struct rlimit *rlim)
{
    if (rlim == NULL)
        return -1;

    if (resource == RLIMIT_STACK)
    {
        rlim->rlim_cur = 1024 * 1024;
        rlim->rlim_max = RLIM_INFINITY;
        return 0;
    }
    return -1;
}

static inline int setrlimit(int resource, const struct rlimit *rlim)
{
    (void)resource;
    (void)rlim;
    return 0;
}

/* ========================================================================== */
/*                         Hostname Functions                                 */
/* ========================================================================== */

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

    if (delim != '\0')
    {
        char *p = strchr(hostname, delim);
        if (p)
            *p = '\0';
    }
    return 0;
}

/* ========================================================================== */
/*                            Random Data                                     */
/* ========================================================================== */

#ifndef NT_SUCCESS
#define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)
#endif

static inline int ncclGetRandomData(void *buffer, size_t bytes)
{
    NTSTATUS status = BCryptGenRandom(NULL, (PUCHAR)buffer, (ULONG)bytes,
                                      BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    return NT_SUCCESS(status) ? 0 : -1;
}

/* ========================================================================== */
/*                           Hash Functions                                   */
/* ========================================================================== */

static inline uint64_t ncclGetHostHash(void)
{
    char hostname[256];
    uint64_t hash = 5381;
    char *p;

    if (ncclGetHostName(hostname, sizeof(hostname), '\0') != 0)
    {
        strcpy(hostname, "localhost");
    }

    for (p = hostname; *p; p++)
    {
        hash = ((hash << 5) + hash) + (unsigned char)*p;
    }
    return hash;
}

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

/* ========================================================================== */
/*                           File System                                      */
/* ========================================================================== */

static inline int mkstemp(char *template_name)
{
    char *xxxpos = strstr(template_name, "XXXXXX");
    static volatile long counter = 0;

    if (xxxpos == NULL)
        return -1;

    DWORD pid = GetCurrentProcessId();
    long cnt = InterlockedIncrement(&counter);
    LARGE_INTEGER perfCounter;
    QueryPerformanceCounter(&perfCounter);

    snprintf(xxxpos, 7, "%02x%04x",
             (unsigned int)(pid & 0xFF),
             (unsigned int)((cnt ^ perfCounter.LowPart) & 0xFFFF));

    HANDLE hFile = CreateFileA(template_name,
                               GENERIC_READ | GENERIC_WRITE,
                               0, NULL, CREATE_NEW,
                               FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE)
        return -1;

    int fd = _open_osfhandle((intptr_t)hFile, 0);
    if (fd == -1)
    {
        CloseHandle(hFile);
        return -1;
    }
    return fd;
}

static inline int fallocate(int fd, int mode, off_t offset, off_t len)
{
    HANDLE hFile;
    LARGE_INTEGER newSize, currentPos;
    BOOL result;

    (void)mode;
    (void)offset;

    hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE)
        return -1;

    currentPos.QuadPart = 0;
    SetFilePointerEx(hFile, currentPos, &currentPos, FILE_CURRENT);

    newSize.QuadPart = offset + len;
    if (!SetFilePointerEx(hFile, newSize, NULL, FILE_BEGIN))
        return -1;

    result = SetEndOfFile(hFile);
    SetFilePointerEx(hFile, currentPos, NULL, FILE_BEGIN);

    return result ? 0 : -1;
}

static inline int fsync(int fd)
{
    HANDLE hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE)
        return -1;
    return FlushFileBuffers(hFile) ? 0 : -1;
}

/* ========================================================================== */
/*                       Environment Variables                                */
/* ========================================================================== */

static inline int setenv(const char *name, const char *value, int overwrite)
{
    if (!overwrite && GetEnvironmentVariableA(name, NULL, 0) > 0)
        return 0;
    return SetEnvironmentVariableA(name, value) ? 0 : -1;
}

static inline int unsetenv(const char *name)
{
    return SetEnvironmentVariableA(name, NULL) ? 0 : -1;
}

/* ========================================================================== */
/*                       Signal Handling (Stubs)                              */
/* ========================================================================== */

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
    return NULL;
}

/* ========================================================================== */
/*                        CPU Affinity Support                                */
/* ========================================================================== */

#ifndef CPU_SETSIZE
#define CPU_SETSIZE 1024
#endif

#define NCCL_CPUSET_WORDS ((CPU_SETSIZE + 63) / 64)

typedef struct
{
    DWORD_PTR mask[NCCL_CPUSET_WORDS];
} cpu_set_t;

#define CPU_ZERO(set)                                  \
    do                                                 \
    {                                                  \
        for (int _i = 0; _i < NCCL_CPUSET_WORDS; _i++) \
            (set)->mask[_i] = 0;                       \
    } while (0)

#define CPU_SET(cpu, set)                                 \
    do                                                    \
    {                                                     \
        int _word = (cpu) / 64;                           \
        int _bit = (cpu) % 64;                            \
        if (_word < NCCL_CPUSET_WORDS)                    \
            (set)->mask[_word] |= ((DWORD_PTR)1 << _bit); \
    } while (0)

#define CPU_CLR(cpu, set)                                  \
    do                                                     \
    {                                                      \
        int _word = (cpu) / 64;                            \
        int _bit = (cpu) % 64;                             \
        if (_word < NCCL_CPUSET_WORDS)                     \
            (set)->mask[_word] &= ~((DWORD_PTR)1 << _bit); \
    } while (0)

#define CPU_ISSET(cpu, set) \
    (((cpu) / 64 < NCCL_CPUSET_WORDS) ? (((set)->mask[(cpu) / 64] >> ((cpu) % 64)) & 1) : 0)

static inline int CPU_COUNT(const cpu_set_t *set)
{
    int count = 0;
    for (int i = 0; i < NCCL_CPUSET_WORDS; i++)
    {
        DWORD_PTR v = set->mask[i];
        while (v)
        {
            v &= v - 1;
            count++;
        }
    }
    return count;
}

#define CPU_COPY(dest, src)                            \
    do                                                 \
    {                                                  \
        for (int _i = 0; _i < NCCL_CPUSET_WORDS; _i++) \
            (dest)->mask[_i] = (src)->mask[_i];        \
    } while (0)

static inline int CPU_EQUAL(const cpu_set_t *a, const cpu_set_t *b)
{
    for (int i = 0; i < NCCL_CPUSET_WORDS; i++)
    {
        if (a->mask[i] != b->mask[i])
            return 0;
    }
    return 1;
}

#define CPU_AND(dest, src1, src2)                                   \
    do                                                              \
    {                                                               \
        for (int _i = 0; _i < NCCL_CPUSET_WORDS; _i++)              \
            (dest)->mask[_i] = (src1)->mask[_i] & (src2)->mask[_i]; \
    } while (0)

#define CPU_OR(dest, src1, src2)                                    \
    do                                                              \
    {                                                               \
        for (int _i = 0; _i < NCCL_CPUSET_WORDS; _i++)              \
            (dest)->mask[_i] = (src1)->mask[_i] | (src2)->mask[_i]; \
    } while (0)

static inline int ncclGetThreadAffinity(cpu_set_t *set)
{
    DWORD_PTR processAffinity, systemAffinity;
    CPU_ZERO(set);
    if (!GetProcessAffinityMask(GetCurrentProcess(), &processAffinity, &systemAffinity))
        return -1;
    set->mask[0] = processAffinity;
    return 0;
}

static inline int ncclSetThreadAffinity(HANDLE thread, const cpu_set_t *set)
{
    DWORD_PTR mask = set->mask[0];
    if (mask == 0)
        return -1;
    DWORD_PTR result = SetThreadAffinityMask(thread, mask);
    return (result != 0) ? 0 : -1;
}

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
            return -1;
    }

    result = ncclSetThreadAffinity(hThread, mask);

    if (pid != 0)
        CloseHandle(hThread);
    return result;
}

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
            return -1;
    }

    if (!GetProcessAffinityMask(hProcess, &processAffinity, &systemAffinity))
    {
        if (pid != 0)
            CloseHandle(hProcess);
        return -1;
    }

    mask->mask[0] = processAffinity;

    if (pid != 0)
        CloseHandle(hProcess);
    return 0;
}

static inline int ncclGetNumProcessors(void)
{
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return (int)sysInfo.dwNumberOfProcessors;
}

/* ========================================================================== */
/*                        Scheduler Functions                                 */
/* ========================================================================== */

static inline int sched_yield(void)
{
    SwitchToThread();
    return 0;
}

/* ========================================================================== */
/*                     Memory Allocation Functions                            */
/* ========================================================================== */

#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE 30
#endif
#ifndef _SC_NPROCESSORS_CONF
#define _SC_NPROCESSORS_CONF 83
#endif
#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 84
#endif

static inline long sysconf(int name)
{
    switch (name)
    {
    case _SC_PAGESIZE:
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long)si.dwPageSize;
    }
    case _SC_NPROCESSORS_CONF:
    case _SC_NPROCESSORS_ONLN:
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long)si.dwNumberOfProcessors;
    }
    default:
        return -1;
    }
}

static inline int posix_memalign(void **memptr, size_t alignment, size_t size)
{
    if (memptr == NULL)
        return EINVAL;

    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || alignment < sizeof(void *))
    {
        return EINVAL;
    }

    *memptr = _aligned_malloc(size, alignment);
    if (*memptr == NULL)
        return ENOMEM;

    return 0;
}

#endif /* _WIN32 */

#endif /* NCCL_WIN32_MISC_H_ */
