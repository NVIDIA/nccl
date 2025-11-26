/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_WIN32_DEFS_H_
#define NCCL_WIN32_DEFS_H_

#ifdef _WIN32

/* Prevent Windows headers from defining min/max macros */
#ifndef NOMINMAX
#define NOMINMAX
#endif

/* Reduce Windows header bloat */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <mswsock.h>
#include <io.h>
#include <process.h>
#include <stdint.h>
#include <sys/types.h> /* For off_t, ino_t, dev_t */
#include <time.h>      /* For struct timespec on newer Windows SDK */
#include <errno.h>     /* For error codes on newer Windows SDK */

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")

/* POSIX compatibility definitions */

/* File operations */
#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif
#ifndef O_WRONLY
#define O_WRONLY _O_WRONLY
#endif
#ifndef O_RDWR
#define O_RDWR _O_RDWR
#endif
#ifndef O_CREAT
#define O_CREAT _O_CREAT
#endif
#ifndef O_EXCL
#define O_EXCL _O_EXCL
#endif
#ifndef O_TRUNC
#define O_TRUNC _O_TRUNC
#endif

/* Process/Thread IDs */
#ifndef _PID_T_DEFINED
typedef DWORD pid_t;
#define _PID_T_DEFINED
#endif

/* off_t for file offsets */
#ifndef _OFF_T_DEFINED
typedef long off_t;
#define _OFF_T_DEFINED
#endif

/* getpid/gettid as inline functions (not macros to avoid conflicts) */
static inline pid_t nccl_getpid(void) { return (pid_t)GetCurrentProcessId(); }
static inline pid_t nccl_gettid(void) { return (pid_t)GetCurrentThreadId(); }
#ifndef getpid
#define getpid nccl_getpid
#endif
#ifndef gettid
#define gettid nccl_gettid
#endif

/* Network interface flags (IFF_*) */
#ifndef IFF_UP
#define IFF_UP 0x1
#endif
#ifndef IFF_RUNNING
#define IFF_RUNNING 0x40
#endif
#ifndef IFF_LOOPBACK
#define IFF_LOOPBACK 0x8
#endif

/* Socket constants */
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#ifndef MSG_DONTWAIT
#define MSG_DONTWAIT 0
#endif

/* NI_* constants for getnameinfo */
#ifndef NI_MAXHOST
#define NI_MAXHOST 1025
#endif
#ifndef NI_MAXSERV
#define NI_MAXSERV 32
#endif

/* Error number compatibility - only define if not already defined by errno.h */
#ifndef EWOULDBLOCK
#define EWOULDBLOCK WSAEWOULDBLOCK
#endif
#ifndef EINPROGRESS
#define EINPROGRESS WSAEINPROGRESS
#endif
#ifndef ECONNRESET
#define ECONNRESET WSAECONNRESET
#endif
/* Note: EPIPE may already be defined in errno.h, only define if missing */
#ifndef EPIPE
#define EPIPE WSAENOTCONN
#endif

/* Signal handling (minimal support) */
typedef void (*sighandler_t)(int);
#define SIGPIPE 0 /* Windows doesn't have SIGPIPE */

static inline sighandler_t signal_noop(int sig, sighandler_t handler)
{
    (void)sig;
    (void)handler;
    return NULL;
}
/* Only define signal macro if needed */
#ifndef NCCL_NO_SIGNAL_OVERRIDE
#define nccl_signal(sig, handler) signal_noop(sig, handler)
#endif

/* String functions */
#ifndef strcasecmp
#define strcasecmp _stricmp
#endif
#ifndef strncasecmp
#define strncasecmp _strnicmp
#endif
#ifndef strdup
#define strdup _strdup
#endif
/* Note: snprintf is standard in C99/C++11 */

/* File descriptor operations */
#ifndef close
#define close(fd) _close(fd)
#endif
#ifndef read
#define read(fd, buf, count) _read(fd, buf, (unsigned int)(count))
#endif
#ifndef write
#define write(fd, buf, count) _write(fd, buf, (unsigned int)(count))
#endif
#ifndef unlink
#define unlink(path) _unlink(path)
#endif
#ifndef access
#define access(path, mode) _access(path, mode)
#endif
#ifndef F_OK
#define F_OK 0
#endif
#ifndef R_OK
#define R_OK 4
#endif
#ifndef W_OK
#define W_OK 2
#endif

/* Socket close function */
#define closesocket_compat(s) closesocket(s)

/* fcntl replacement for non-blocking sockets */
#ifndef F_GETFL
#define F_GETFL 0
#endif
#ifndef F_SETFL
#define F_SETFL 1
#endif
#ifndef O_NONBLOCK
#define O_NONBLOCK 0x0004
#endif

static inline int fcntl_socket(SOCKET s, int cmd, int arg)
{
    if (cmd == F_SETFL)
    {
        u_long nonblocking = (arg & O_NONBLOCK) ? 1 : 0;
        return ioctlsocket(s, FIONBIO, &nonblocking);
    }
    return 0;
}
#define fcntl(fd, cmd, ...) fcntl_socket((SOCKET)(fd), cmd, __VA_ARGS__)

/*
 * Poll support - use WSAPoll on newer Windows SDKs
 * Only define our own pollfd if the SDK doesn't provide it
 */
#ifndef POLLIN
#define POLLIN 0x0001
#endif
#ifndef POLLOUT
#define POLLOUT 0x0004
#endif
#ifndef POLLERR
#define POLLERR 0x0008
#endif
#ifndef POLLHUP
#define POLLHUP 0x0010
#endif
#ifndef POLLNVAL
#define POLLNVAL 0x0020
#endif

/* Use WSAPOLLFD from winsock2.h as pollfd */
#ifndef _STRUCT_POLLFD_DEFINED
#define _STRUCT_POLLFD_DEFINED
typedef WSAPOLLFD pollfd;
#endif

/* Use WSAPoll directly */
#ifndef poll
#define poll(fds, nfds, timeout) WSAPoll((LPWSAPOLLFD)(fds), (ULONG)(nfds), (INT)(timeout))
#endif

/* strerror_r is not available on Windows, use strerror_s */
static inline int strerror_r_compat(int errnum, char *buf, size_t buflen)
{
    return strerror_s(buf, buflen, errnum);
}
#define strerror_r(errnum, buf, buflen) strerror_r_compat(errnum, buf, buflen)

/* Random data generation */
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

/* NT_SUCCESS macro if not defined (normally in ntdef.h/ntddk.h) */
#ifndef NT_SUCCESS
#define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)
#endif

static inline int getRandomData_win32(void *buffer, size_t bytes)
{
    NTSTATUS status = BCryptGenRandom(NULL, (PUCHAR)buffer, (ULONG)bytes, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    return NT_SUCCESS(status) ? 0 : -1;
}

/* High-resolution timing */
static inline uint64_t clockNano_win32()
{
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (uint64_t)((counter.QuadPart * 1000000000LL) / freq.QuadPart);
}

/* Sleep function (milliseconds) */
static inline void msleep_win32(unsigned int time_msec)
{
    Sleep(time_msec);
}

/* Hostname */
static inline int gethostname_compat(char *name, int namelen)
{
    DWORD size = (DWORD)namelen;
    if (GetComputerNameExA(ComputerNameDnsHostname, name, &size))
    {
        return 0;
    }
    /* Fallback to Winsock gethostname */
    return gethostname(name, namelen);
}

/* Dynamic library loading */
typedef HMODULE ncclDynLib_t;
#define NCCL_DYNLIB_INVALID NULL

static inline ncclDynLib_t ncclDynLibOpen(const char *path)
{
    return LoadLibraryA(path);
}

static inline void *ncclDynLibSym(ncclDynLib_t lib, const char *symbol)
{
    return (void *)GetProcAddress(lib, symbol);
}

static inline int ncclDynLibClose(ncclDynLib_t lib)
{
    return FreeLibrary(lib) ? 0 : -1;
}

static inline const char *ncclDynLibError()
{
    static char errbuf[256];
    DWORD err = GetLastError();
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   errbuf, sizeof(errbuf), NULL);
    return errbuf;
}

/* Environment variables */
static inline const char *ncclGetEnv_win32(const char *name)
{
    static char buffer[32768]; /* Windows max env var size */
    DWORD ret = GetEnvironmentVariableA(name, buffer, sizeof(buffer));
    if (ret == 0 || ret >= sizeof(buffer))
        return NULL;
    return buffer;
}

/* Windows Socket initialization helper */
static inline int ncclWinsockInit()
{
    static int initialized = 0;
    if (!initialized)
    {
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (result != 0)
            return result;
        initialized = 1;
    }
    return 0;
}

static inline void ncclWinsockCleanup()
{
    WSACleanup();
}

/* ===================== CPU Intrinsics for Spin Loops ===================== */

/*
 * CPU pause instruction for spin-wait loops
 * Reduces power consumption and improves performance in busy loops
 * Equivalent to _mm_pause() / __builtin_ia32_pause()
 */
#if defined(_M_X64) || defined(_M_IX86) || defined(_M_AMD64)
#include <intrin.h>
#define ncclCpuPause() _mm_pause()
#elif defined(_M_ARM64)
#define ncclCpuPause() __yield()
#else
#define ncclCpuPause() ((void)0)
#endif

/*
 * Yield to other threads - use when spinning for longer periods
 */
static inline void ncclCpuYield(void)
{
    SwitchToThread();
}

/*
 * Adaptive spin-wait with exponential backoff
 * Efficient for contended resources
 */
static inline void ncclSpinWait(volatile LONG *flag, LONG expected, int maxSpins)
{
    int spins = 0;

    while (*flag != expected)
    {
        if (spins < 16)
        {
            /* Initial fast spins with pause */
            ncclCpuPause();
        }
        else if (spins < maxSpins)
        {
            /* Yield to other threads */
            ncclCpuYield();
        }
        else
        {
            /* Fall back to sleep for very long waits */
            Sleep(0);
            spins = maxSpins - 1; /* Don't let spins overflow */
        }
        spins++;
    }
}

/* ===================== Memory Barriers ===================== */

/*
 * Full memory fence - ensures all memory operations complete
 * Equivalent to __sync_synchronize() / std::atomic_thread_fence(memory_order_seq_cst)
 */
#if defined(_M_X64) || defined(_M_AMD64)
#define ncclMemoryFence() _mm_mfence()
#define ncclLoadFence() _mm_lfence()
#define ncclStoreFence() _mm_sfence()
#elif defined(_M_ARM64)
#define ncclMemoryFence() __dmb(_ARM64_BARRIER_SY)
#define ncclLoadFence() __dmb(_ARM64_BARRIER_LD)
#define ncclStoreFence() __dmb(_ARM64_BARRIER_ST)
#else
#define ncclMemoryFence() MemoryBarrier()
#define ncclLoadFence() MemoryBarrier()
#define ncclStoreFence() MemoryBarrier()
#endif

/*
 * Compiler barrier - prevents compiler reordering
 */
#define ncclCompilerBarrier() _ReadWriteBarrier()

/* ===================== Cache Line Alignment ===================== */

/*
 * Cache line size for avoiding false sharing
 * Most modern x86/ARM64 CPUs use 64-byte cache lines
 */
#define NCCL_CACHE_LINE_SIZE 64

/*
 * Align data to cache line boundary
 */
#define NCCL_CACHE_ALIGN __declspec(align(NCCL_CACHE_LINE_SIZE))

/*
 * Pad structure to cache line size
 */
#define NCCL_CACHE_PAD(name, size) char name##_pad[NCCL_CACHE_LINE_SIZE - ((size) % NCCL_CACHE_LINE_SIZE)]

/* ===================== Atomic Operations ===================== */

/*
 * Atomic load with acquire semantics
 */
static inline LONG ncclAtomicLoadAcquire(volatile LONG *ptr)
{
    LONG value = *ptr;
    ncclLoadFence();
    return value;
}

/*
 * Atomic store with release semantics
 */
static inline void ncclAtomicStoreRelease(volatile LONG *ptr, LONG value)
{
    ncclStoreFence();
    *ptr = value;
}

/*
 * Atomic load 64-bit with acquire semantics
 */
static inline LONGLONG ncclAtomicLoad64Acquire(volatile LONGLONG *ptr)
{
    LONGLONG value = InterlockedCompareExchange64(ptr, 0, 0);
    ncclLoadFence();
    return value;
}

/*
 * Atomic store 64-bit with release semantics
 */
static inline void ncclAtomicStore64Release(volatile LONGLONG *ptr, LONGLONG value)
{
    ncclStoreFence();
    InterlockedExchange64(ptr, value);
}

/*
 * Atomic add and return old value
 */
#define ncclAtomicAdd(ptr, val) InterlockedExchangeAdd((ptr), (val))
#define ncclAtomicAdd64(ptr, val) InterlockedExchangeAdd64((ptr), (val))

/*
 * Atomic compare-and-swap
 */
#define ncclAtomicCAS(ptr, expected, desired) \
    InterlockedCompareExchange((ptr), (desired), (expected))
#define ncclAtomicCAS64(ptr, expected, desired) \
    InterlockedCompareExchange64((ptr), (desired), (expected))

#endif /* _WIN32 */

#endif /* NCCL_WIN32_DEFS_H_ */
