/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PLATFORM_H_
#define NCCL_PLATFORM_H_

/*
 * Platform detection and abstraction layer for NCCL
 * Provides cross-platform compatibility between Linux and Windows
 */

/* Platform detection */
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
#define NCCL_PLATFORM_WINDOWS 1
#define NCCL_PLATFORM_LINUX 0
#define NCCL_PLATFORM_POSIX 0
#elif defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#define NCCL_PLATFORM_WINDOWS 0
#define NCCL_PLATFORM_LINUX 1
#define NCCL_PLATFORM_POSIX 1
#else
#error "Unsupported platform"
#endif

/* Compiler detection */
#if defined(_MSC_VER)
#define NCCL_COMPILER_MSVC 1
#define NCCL_COMPILER_GCC 0
#define NCCL_COMPILER_CLANG 0
#elif defined(__clang__)
#define NCCL_COMPILER_MSVC 0
#define NCCL_COMPILER_GCC 0
#define NCCL_COMPILER_CLANG 1
#elif defined(__GNUC__)
#define NCCL_COMPILER_MSVC 0
#define NCCL_COMPILER_GCC 1
#define NCCL_COMPILER_CLANG 0
#else
#define NCCL_COMPILER_MSVC 0
#define NCCL_COMPILER_GCC 0
#define NCCL_COMPILER_CLANG 0
#endif

/* Export/Import macros for DLL visibility */
#if NCCL_PLATFORM_WINDOWS
#ifdef NCCL_EXPORTS
#define NCCL_EXPORT __declspec(dllexport)
#else
#define NCCL_EXPORT __declspec(dllimport)
#endif
#define NCCL_HIDDEN
#else
#define NCCL_EXPORT __attribute__((visibility("default")))
#define NCCL_HIDDEN __attribute__((visibility("hidden")))
#endif

/* Common type definitions */
#if NCCL_PLATFORM_WINDOWS
#include <stdint.h>
#include <basetsd.h>
#ifndef _SSIZE_T_DEFINED
typedef SSIZE_T ssize_t;
#define _SSIZE_T_DEFINED
#endif
#ifndef PATH_MAX
#define PATH_MAX 260
#endif
#else
#include <sys/types.h>
#include <limits.h>
#endif

/* Thread-local storage */
#if NCCL_PLATFORM_WINDOWS
#define NCCL_THREAD_LOCAL __declspec(thread)
#else
#define NCCL_THREAD_LOCAL __thread
#endif

/* Function attributes */
#if NCCL_COMPILER_MSVC
#define NCCL_INLINE __forceinline
#define NCCL_NOINLINE __declspec(noinline)
#define NCCL_ALIGN(x) __declspec(align(x))
#define NCCL_PRINTF_FORMAT(fmt, args)

/*
 * GCC __attribute__ compatibility for MSVC
 * MSVC doesn't support __attribute__ syntax, so we define it to expand to nothing.
 * Specific attributes that have MSVC equivalents are handled separately above.
 * This must be defined before any headers that use __attribute__.
 */
#ifndef __attribute__
#define __attribute__(x)
#endif

#else
#define NCCL_INLINE inline __attribute__((always_inline))
#define NCCL_NOINLINE __attribute__((noinline))
#define NCCL_ALIGN(x) __attribute__((aligned(x)))
#define NCCL_PRINTF_FORMAT(fmt, args) __attribute__((format(printf, fmt, args)))
#endif

/* Atomic operations */
#if NCCL_PLATFORM_WINDOWS
#include <intrin.h>
#include <windows.h>

/* Memory order constants (matching GCC values) - only define if not already defined by CCCL */
#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#endif
#ifndef __ATOMIC_CONSUME
#define __ATOMIC_CONSUME 1
#endif
#ifndef __ATOMIC_ACQUIRE
#define __ATOMIC_ACQUIRE 2
#endif
#ifndef __ATOMIC_RELEASE
#define __ATOMIC_RELEASE 3
#endif
#ifndef __ATOMIC_ACQ_REL
#define __ATOMIC_ACQ_REL 4
#endif
#ifndef __ATOMIC_SEQ_CST
#define __ATOMIC_SEQ_CST 5
#endif

/* GCC builtins not available on MSVC */
#ifndef __builtin_expect
#define __builtin_expect(expr, val) (expr)
#endif

/*
 * NOTE: __builtin_ffs, __builtin_popcount, __builtin_popcountll, and
 * byte-swap functions are defined in bitops.h with MSVC implementations.
 * Don't define them here to avoid duplicate definition errors.
 */

/* __sync_synchronize - Full memory barrier (GCC builtin) */
#ifndef __sync_synchronize
#define __sync_synchronize() MemoryBarrier()
#endif

/*
 * __builtin_clz, __builtin_clzl, __builtin_clzll are defined in bitops.h
 * which also provides MSVC-compatible implementations. We don't define them
 * here to avoid duplicate symbol errors.
 */

/*
 * Basic NCCL atomic macros - these are our own abstraction layer
 * Use these instead of __atomic_* functions for simple atomic operations.
 * These macros work on both Windows and Linux without requiring CCCL.
 *
 * NOTE: For __atomic_* functions on MSVC, rely on CCCL's msvc_to_builtins.h
 * which is included as part of CUDA 13.0+. We do NOT define __atomic_*
 * functions here to avoid conflicts with CCCL.
 */
#define NCCL_ATOMIC_LOAD(ptr) _InterlockedCompareExchange((volatile long *)(ptr), 0, 0)
#define NCCL_ATOMIC_STORE(ptr, val) _InterlockedExchange((volatile long *)(ptr), (long)(val))
#define NCCL_ATOMIC_ADD(ptr, val) _InterlockedExchangeAdd((volatile long *)(ptr), (long)(val))
#define NCCL_ATOMIC_SUB(ptr, val) _InterlockedExchangeAdd((volatile long *)(ptr), -(long)(val))
#define NCCL_ATOMIC_CAS(ptr, expected, desired) \
    (_InterlockedCompareExchange((volatile long *)(ptr), (long)(desired), (long)(expected)) == (long)(expected))

/* 64-bit atomic operations */
#define NCCL_ATOMIC_LOAD64(ptr) _InterlockedCompareExchange64((volatile long long *)(ptr), 0, 0)
#define NCCL_ATOMIC_STORE64(ptr, val) _InterlockedExchange64((volatile long long *)(ptr), (long long)(val))
#define NCCL_ATOMIC_ADD64(ptr, val) _InterlockedExchangeAdd64((volatile long long *)(ptr), (long long)(val))

/*
 * GCC-compatible __atomic_* implementations for MSVC
 *
 * These functions provide atomic operations needed by NCCL code on Windows.
 * We provide them here since CCCL's msvc_to_builtins.h isn't automatically included.
 */

/* Memory order constants (matching GCC values) */
#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5
#endif

/*
 * __atomic_load_n - Atomic load with memory ordering
 */
template <typename T>
static __forceinline T __atomic_load_n(const volatile T *ptr, int /*memorder*/)
{
    T val;
    if constexpr (sizeof(T) == 4)
    {
        val = (T)_InterlockedOr((volatile long *)ptr, 0);
    }
    else if constexpr (sizeof(T) == 8)
    {
        val = (T)_InterlockedOr64((volatile long long *)ptr, 0);
    }
    else if constexpr (sizeof(T) == 1)
    {
        val = (T)_InterlockedOr8((volatile char *)ptr, 0);
    }
    else if constexpr (sizeof(T) == 2)
    {
        val = (T)_InterlockedOr16((volatile short *)ptr, 0);
    }
    else
    {
        val = *ptr; // Fallback for other sizes
    }
    return val;
}

/*
 * __atomic_load - Atomic load with memory ordering (GCC-style, stores to ptr)
 */
template <typename T>
static __forceinline void __atomic_load(const volatile T *ptr, T *ret, int memorder)
{
    *ret = __atomic_load_n(ptr, memorder);
}

/*
 * __atomic_store_n - Atomic store with memory ordering
 * Note: Uses separate template parameter for value to allow implicit conversions
 * (e.g., storing int literal 1 to uint32_t*)
 */
template <typename T, typename U>
static __forceinline void __atomic_store_n(volatile T *ptr, U val, int /*memorder*/)
{
    if constexpr (sizeof(T) == 4)
    {
        _InterlockedExchange((volatile long *)ptr, (long)(T)val);
    }
    else if constexpr (sizeof(T) == 8)
    {
        _InterlockedExchange64((volatile long long *)ptr, (long long)(T)val);
    }
    else if constexpr (sizeof(T) == 1)
    {
        _InterlockedExchange8((volatile char *)ptr, (char)(T)val);
    }
    else if constexpr (sizeof(T) == 2)
    {
        _InterlockedExchange16((volatile short *)ptr, (short)(T)val);
    }
    else
    {
        *ptr = (T)val; // Fallback for other sizes
    }
}

/*
 * __atomic_fetch_add - Atomic add, returns old value
 * Note: Uses separate template parameter for value to allow implicit conversions
 * (e.g., adding int literal 1 to uint64_t*)
 */
template <typename T, typename U>
static __forceinline T __atomic_fetch_add(volatile T *ptr, U val, int /*memorder*/)
{
    if constexpr (sizeof(T) == 4)
    {
        return (T)_InterlockedExchangeAdd((volatile long *)ptr, (long)(T)val);
    }
    else if constexpr (sizeof(T) == 8)
    {
        return (T)_InterlockedExchangeAdd64((volatile long long *)ptr, (long long)(T)val);
    }
    else
    {
        // No intrinsic for 1/2 byte add, use CAS loop
        T old_val;
        T add_val = (T)val;
        do
        {
            old_val = *ptr;
        } while (!__atomic_compare_exchange_n(ptr, &old_val, old_val + add_val, false, 0, 0));
        return old_val;
    }
}

/*
 * __atomic_fetch_sub - Atomic subtract, returns old value
 * Note: Uses separate template parameter for value to allow implicit conversions
 */
template <typename T, typename U>
static __forceinline T __atomic_fetch_sub(volatile T *ptr, U val, int /*memorder*/)
{
    return __atomic_fetch_add(ptr, (T)(-(long long)val), 0);
}

/*
 * __atomic_add_fetch - Atomic add, returns new value
 * Note: Uses separate template parameter for value to allow implicit conversions
 */
template <typename T, typename U>
static __forceinline T __atomic_add_fetch(volatile T *ptr, U val, int /*memorder*/)
{
    T tval = (T)val;
    return __atomic_fetch_add(ptr, tval, 0) + tval;
}

/*
 * __atomic_sub_fetch - Atomic subtract, returns new value
 * Note: Uses separate template parameter for value to allow implicit conversions
 */
template <typename T, typename U>
static __forceinline T __atomic_sub_fetch(volatile T *ptr, U val, int /*memorder*/)
{
    T tval = (T)val;
    return __atomic_fetch_sub(ptr, tval, 0) - tval;
}

/*
 * __atomic_exchange_n - Atomic exchange
 */
template <typename T>
static __forceinline T __atomic_exchange_n(volatile T *ptr, T val, int /*memorder*/)
{
    if constexpr (sizeof(T) == 4)
    {
        return (T)_InterlockedExchange((volatile long *)ptr, (long)val);
    }
    else if constexpr (sizeof(T) == 8)
    {
        return (T)_InterlockedExchange64((volatile long long *)ptr, (long long)val);
    }
    else if constexpr (sizeof(T) == 1)
    {
        return (T)_InterlockedExchange8((volatile char *)ptr, (char)val);
    }
    else if constexpr (sizeof(T) == 2)
    {
        return (T)_InterlockedExchange16((volatile short *)ptr, (short)val);
    }
    else
    {
        T old = *ptr;
        *ptr = val;
        return old;
    }
}

/*
 * __atomic_compare_exchange_n - Atomic compare and exchange
 * Returns true if exchange succeeded, false otherwise.
 * If failed, *expected is updated with the current value.
 */
template <typename T>
static __forceinline bool __atomic_compare_exchange_n(volatile T *ptr, T *expected, T desired,
                                                      bool /*weak*/, int /*success_order*/, int /*fail_order*/)
{
    T old_expected = *expected;
    T old_val;
    if constexpr (sizeof(T) == 4)
    {
        old_val = (T)_InterlockedCompareExchange((volatile long *)ptr, (long)desired, (long)old_expected);
    }
    else if constexpr (sizeof(T) == 8)
    {
        old_val = (T)_InterlockedCompareExchange64((volatile long long *)ptr, (long long)desired, (long long)old_expected);
    }
    else if constexpr (sizeof(T) == 1)
    {
        old_val = (T)_InterlockedCompareExchange8((volatile char *)ptr, (char)desired, (char)old_expected);
    }
    else if constexpr (sizeof(T) == 2)
    {
        old_val = (T)_InterlockedCompareExchange16((volatile short *)ptr, (short)desired, (short)old_expected);
    }
    else
    {
        // Fallback - not atomic
        old_val = *ptr;
        if (old_val == old_expected)
        {
            *ptr = desired;
        }
    }
    if (old_val == old_expected)
    {
        return true;
    }
    else
    {
        *expected = old_val;
        return false;
    }
}

/*
 * __atomic_thread_fence - Memory fence
 */
static __forceinline void __atomic_thread_fence(int memorder)
{
    (void)memorder;
    MemoryBarrier();
}

/*
 * __builtin_prefetch - Cache prefetch hint
 * On Windows/MSVC, this is a no-op (MSVC has _mm_prefetch but requires SSE headers)
 */
#ifndef __builtin_prefetch
#define __builtin_prefetch(addr, ...) ((void)0)
#endif

/*
 * sched_getcpu - Get the CPU number on which the calling thread is running
 * Windows equivalent using GetCurrentProcessorNumber()
 */
static __forceinline int sched_getcpu(void)
{
    return (int)GetCurrentProcessorNumber();
}

/*
 * Linux syscall compatibility
 * SYS_gettid and syscall(SYS_gettid) - get thread ID
 */
#ifndef SYS_gettid
#define SYS_gettid 0
#endif

static __forceinline long syscall(long number, ...)
{
    (void)number;
    /* On Windows, just return the thread ID for SYS_gettid */
    return (long)GetCurrentThreadId();
}

/*
 * pthread shared attribute functions are defined in win32_thread.h
 * since they require pthread types to be defined first.
 */

#else /* !NCCL_PLATFORM_WINDOWS */
#define NCCL_ATOMIC_LOAD(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
#define NCCL_ATOMIC_STORE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#define NCCL_ATOMIC_ADD(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED)
#define NCCL_ATOMIC_SUB(ptr, val) __atomic_fetch_sub(ptr, val, __ATOMIC_RELAXED)
#define NCCL_ATOMIC_CAS(ptr, expected, desired) \
    __atomic_compare_exchange_n(ptr, &(expected), desired, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)
#endif

/* Memory barriers */
#if NCCL_PLATFORM_WINDOWS
/* Use MemoryBarrier() from Windows.h - full hardware fence */
#define NCCL_MEMORY_BARRIER() MemoryBarrier()
/* Compiler barrier - use compiler-specific intrinsic or inline assembly equivalent */
#define NCCL_COMPILER_BARRIER() \
    _ReadBarrier();             \
    _WriteBarrier()
#else
#define NCCL_MEMORY_BARRIER() __sync_synchronize()
#define NCCL_COMPILER_BARRIER() asm volatile("" ::: "memory")
#endif

/* Socket handle type */
#if NCCL_PLATFORM_WINDOWS
#include <winsock2.h>
typedef SOCKET ncclSocketHandle_t;
#define NCCL_INVALID_SOCKET INVALID_SOCKET
#define NCCL_SOCKET_ERROR SOCKET_ERROR
#else
typedef int ncclSocketHandle_t;
#define NCCL_INVALID_SOCKET (-1)
#define NCCL_SOCKET_ERROR (-1)
#endif

/* Error code handling */
#if NCCL_PLATFORM_WINDOWS
#define NCCL_GET_LAST_ERROR() GetLastError()
#define NCCL_GET_SOCKET_ERROR() WSAGetLastError()
#define NCCL_ERRNO_EINTR WSAEINTR
#define NCCL_ERRNO_EWOULDBLOCK WSAEWOULDBLOCK
#define NCCL_ERRNO_EAGAIN WSAEWOULDBLOCK
#define NCCL_ERRNO_EINPROGRESS WSAEINPROGRESS
#define NCCL_ERRNO_ECONNRESET WSAECONNRESET
#else
#include <errno.h>
#define NCCL_GET_LAST_ERROR() errno
#define NCCL_GET_SOCKET_ERROR() errno
#define NCCL_ERRNO_EINTR EINTR
#define NCCL_ERRNO_EWOULDBLOCK EWOULDBLOCK
#define NCCL_ERRNO_EAGAIN EAGAIN
#define NCCL_ERRNO_EINPROGRESS EINPROGRESS
#define NCCL_ERRNO_ECONNRESET ECONNRESET
#endif

/* Include platform-specific headers */
#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_defs.h"
#include "platform/win32_misc.h" /* Must be before win32_thread.h for cpu_set_t */
#include "platform/win32_thread.h"
#include "platform/win32_dl.h"
#include "platform/win32_socket.h"
#endif

#endif /* NCCL_PLATFORM_H_ */
