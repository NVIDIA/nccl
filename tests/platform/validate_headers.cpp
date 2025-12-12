/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Comprehensive header inclusion test for Windows platform abstraction
 * This verifies all platform headers compile correctly together
 */

#include "platform.h"

/* Test that platform detection works */
#if !defined(NCCL_PLATFORM_WINDOWS) && !defined(NCCL_PLATFORM_LINUX)
#error "Platform not detected"
#endif

#if NCCL_PLATFORM_WINDOWS + NCCL_PLATFORM_LINUX != 1
#error "Exactly one platform should be detected"
#endif

/* Include standard Windows headers that we need to coexist with */
#ifdef _WIN32
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <bcrypt.h>
#endif

/* Test type definitions */
void test_types(void)
{
    /* Basic types */
    ssize_t ss = -1;
    (void)ss;

    /* Thread types */
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    (void)thread;
    (void)mutex;
    (void)cond;

    /* CPU affinity */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    (void)cpuset;

    /* Time types */
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 0;
    (void)ts;

    /* Socket types */
    struct pollfd pfd;
    pfd.fd = 0;
    pfd.events = POLLIN;
    pfd.revents = 0;
    (void)pfd;
}

/* Test function availability */
void test_functions(void)
{
    /* Time functions */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    /* Thread functions */
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_lock(&mutex);
    pthread_mutex_unlock(&mutex);
    pthread_mutex_destroy(&mutex);

    /* Process functions */
    int pid = getpid();
    int tid = gettid();
    (void)pid;
    (void)tid;

    /* CPU affinity */
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    int count = CPU_COUNT(&cpuset);
    (void)count;
}

/* Test macro definitions */
void test_macros(void)
{
    /* Error codes */
    int e1 = EINTR;
    int e2 = EWOULDBLOCK;
    int e3 = EINPROGRESS;
    int e4 = ECONNRESET;
    (void)e1;
    (void)e2;
    (void)e3;
    (void)e4;

    /* Clock types */
    int c1 = CLOCK_REALTIME;
    int c2 = CLOCK_MONOTONIC;
    (void)c1;
    (void)c2;

    /* Poll events */
    short p1 = POLLIN;
    short p2 = POLLOUT;
    short p3 = POLLERR;
    (void)p1;
    (void)p2;
    (void)p3;

    /* Path max */
    char path[PATH_MAX];
    (void)path;
}

int main(void)
{
    printf("Platform header validation\n");
#if NCCL_PLATFORM_WINDOWS
    printf("  Platform: Windows\n");
#elif NCCL_PLATFORM_LINUX
    printf("  Platform: Linux\n");
#endif
    printf("  PATH_MAX: %d\n", PATH_MAX);
    printf("  CPU_SETSIZE: %d\n", CPU_SETSIZE);

    test_types();
    printf("  Types: OK\n");

    test_functions();
    printf("  Functions: OK\n");

    test_macros();
    printf("  Macros: OK\n");

    printf("All validation checks passed!\n");
    return 0;
}
