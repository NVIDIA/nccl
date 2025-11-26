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
#define NCCL_DEPRECATED __declspec(deprecated)
#define NCCL_PRINTF_FORMAT(fmt, args)
#else
#define NCCL_INLINE inline __attribute__((always_inline))
#define NCCL_NOINLINE __attribute__((noinline))
#define NCCL_ALIGN(x) __attribute__((aligned(x)))
#define NCCL_DEPRECATED __attribute__((deprecated))
#define NCCL_PRINTF_FORMAT(fmt, args) __attribute__((format(printf, fmt, args)))
#endif

/* Atomic operations */
#if NCCL_PLATFORM_WINDOWS
#include <intrin.h>
#define NCCL_ATOMIC_LOAD(ptr) _InterlockedCompareExchange((volatile long *)(ptr), 0, 0)
#define NCCL_ATOMIC_STORE(ptr, val) _InterlockedExchange((volatile long *)(ptr), (long)(val))
#define NCCL_ATOMIC_ADD(ptr, val) _InterlockedExchangeAdd((volatile long *)(ptr), (long)(val))
#define NCCL_ATOMIC_SUB(ptr, val) _InterlockedExchangeAdd((volatile long *)(ptr), -(long)(val))
#define NCCL_ATOMIC_CAS(ptr, expected, desired) \
    (_InterlockedCompareExchange((volatile long *)(ptr), (long)(desired), (long)(expected)) == (long)(expected))
#else
#define NCCL_ATOMIC_LOAD(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
#define NCCL_ATOMIC_STORE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#define NCCL_ATOMIC_ADD(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED)
#define NCCL_ATOMIC_SUB(ptr, val) __atomic_fetch_sub(ptr, val, __ATOMIC_RELAXED)
#define NCCL_ATOMIC_CAS(ptr, expected, desired) \
    __atomic_compare_exchange_n(ptr, &(expected), desired, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)
#endif

/* Memory barriers */
#if NCCL_PLATFORM_WINDOWS
#define NCCL_MEMORY_BARRIER() _ReadWriteBarrier()
#define NCCL_COMPILER_BARRIER() _ReadWriteBarrier()
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
