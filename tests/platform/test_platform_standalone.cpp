/*************************************************************************
 * NCCL Platform Abstraction - Standalone Validation Test
 *
 * This test validates the Windows platform abstraction layer without
 * requiring CUDA or the full NCCL library to be built.
 *
 * Compile on Windows:
 *   cl /EHsc /I..\..\src\include test_platform_standalone.cpp ws2_32.lib iphlpapi.lib
 *
 * Compile on Linux:
 *   g++ -I../../src/include test_platform_standalone.cpp -lpthread -ldl -o test_platform
 *
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Include platform header */
#include "platform.h"

#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_misc.h"
#include "platform/win32_thread.h"
#include "platform/win32_socket.h"
#include "platform/win32_dl.h"
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#else
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <dlfcn.h>
#include <sched.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <poll.h>
#endif

/* Test counters */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(cond, msg)                   \
    do                                    \
    {                                     \
        tests_run++;                      \
        if (cond)                         \
        {                                 \
            tests_passed++;               \
            printf("  [PASS] %s\n", msg); \
        }                                 \
        else                              \
        {                                 \
            tests_failed++;               \
            printf("  [FAIL] %s\n", msg); \
        }                                 \
    } while (0)

/* =================================================================== */
/*                        TEST FUNCTIONS                               */
/* =================================================================== */

void test_platform_macros(void)
{
    printf("\n=== Platform Macros ===\n");

    TEST(NCCL_PLATFORM_WINDOWS == 0 || NCCL_PLATFORM_WINDOWS == 1,
         "NCCL_PLATFORM_WINDOWS is boolean");
    TEST(NCCL_PLATFORM_LINUX == 0 || NCCL_PLATFORM_LINUX == 1,
         "NCCL_PLATFORM_LINUX is boolean");
    TEST(NCCL_PLATFORM_WINDOWS + NCCL_PLATFORM_LINUX == 1,
         "Exactly one platform detected");

#if NCCL_PLATFORM_WINDOWS
    printf("  [INFO] Running on Windows\n");
#else
    printf("  [INFO] Running on Linux/Unix\n");
#endif

    TEST(PATH_MAX > 0, "PATH_MAX is defined and positive");
    printf("  [INFO] PATH_MAX = %d\n", PATH_MAX);
}

void test_time_functions(void)
{
    printf("\n=== Time Functions ===\n");

    struct timespec ts1, ts2;

    int ret = clock_gettime(CLOCK_MONOTONIC, &ts1);
    TEST(ret == 0, "clock_gettime(CLOCK_MONOTONIC) succeeds");
    TEST(ts1.tv_sec >= 0, "tv_sec is non-negative");
    TEST(ts1.tv_nsec >= 0 && ts1.tv_nsec < 1000000000L, "tv_nsec is valid");

    /* Sleep 10ms */
#if NCCL_PLATFORM_WINDOWS
    Sleep(10);
#else
    usleep(10000);
#endif

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    long long elapsed = (ts2.tv_sec - ts1.tv_sec) * 1000000000LL +
                        (ts2.tv_nsec - ts1.tv_nsec);
    TEST(elapsed > 0, "Time advances after sleep");
    TEST(elapsed >= 5000000LL, "At least 5ms elapsed");
    printf("  [INFO] Elapsed: %lld ns\n", elapsed);
}

void test_thread_functions(void)
{
    printf("\n=== Thread Functions ===\n");

    pthread_mutex_t mutex;
    int ret = pthread_mutex_init(&mutex, NULL);
    TEST(ret == 0, "pthread_mutex_init succeeds");

    ret = pthread_mutex_lock(&mutex);
    TEST(ret == 0, "pthread_mutex_lock succeeds");

    ret = pthread_mutex_unlock(&mutex);
    TEST(ret == 0, "pthread_mutex_unlock succeeds");

    ret = pthread_mutex_destroy(&mutex);
    TEST(ret == 0, "pthread_mutex_destroy succeeds");

    pthread_cond_t cond;
    ret = pthread_cond_init(&cond, NULL);
    TEST(ret == 0, "pthread_cond_init succeeds");

    ret = pthread_cond_destroy(&cond);
    TEST(ret == 0, "pthread_cond_destroy succeeds");

    pthread_t self = pthread_self();
#if NCCL_PLATFORM_WINDOWS
    TEST(self != NULL, "pthread_self returns valid handle");
#else
    TEST(self != 0, "pthread_self returns valid id");
#endif
}

void test_cpu_affinity(void)
{
    printf("\n=== CPU Affinity ===\n");

    TEST(CPU_SETSIZE > 0, "CPU_SETSIZE is positive");
    TEST(CPU_SETSIZE >= 64, "CPU_SETSIZE supports at least 64 CPUs");
    printf("  [INFO] CPU_SETSIZE = %d\n", CPU_SETSIZE);

    cpu_set_t set;
    CPU_ZERO(&set);
    TEST(CPU_COUNT(&set) == 0, "CPU_ZERO creates empty set");

    CPU_SET(0, &set);
    TEST(CPU_ISSET(0, &set), "CPU_SET sets bit correctly");
    TEST(CPU_COUNT(&set) == 1, "CPU_COUNT returns 1");

    CPU_SET(5, &set);
    CPU_SET(10, &set);
    TEST(CPU_COUNT(&set) == 3, "CPU_COUNT returns 3 after setting 3 CPUs");

    CPU_CLR(5, &set);
    TEST(!CPU_ISSET(5, &set), "CPU_CLR clears bit correctly");
    TEST(CPU_COUNT(&set) == 2, "CPU_COUNT returns 2 after clearing 1 CPU");

    /* Test sched_getaffinity */
    cpu_set_t process_affinity;
    CPU_ZERO(&process_affinity);
    int ret = sched_getaffinity(0, sizeof(cpu_set_t), &process_affinity);
    TEST(ret == 0, "sched_getaffinity succeeds");
    int count = CPU_COUNT(&process_affinity);
    TEST(count > 0, "Process has at least one CPU in affinity");
    printf("  [INFO] Process affinity has %d CPUs\n", count);
}

void test_socket_functions(void)
{
    printf("\n=== Socket Functions ===\n");

#if NCCL_PLATFORM_WINDOWS
    WSADATA wsaData;
    int ret = WSAStartup(MAKEWORD(2, 2), &wsaData);
    TEST(ret == 0, "WSAStartup succeeds");
#endif

    ncclSocketHandle_t sock = socket(AF_INET, SOCK_STREAM, 0);
    TEST(sock != NCCL_INVALID_SOCKET, "socket() succeeds");

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    int bindret = bind(sock, (struct sockaddr *)&addr, sizeof(addr));
    TEST(bindret == 0, "bind() to loopback succeeds");

    socklen_t addrlen = sizeof(addr);
    getsockname(sock, (struct sockaddr *)&addr, &addrlen);
    TEST(ntohs(addr.sin_port) != 0, "Port assigned by OS");
    printf("  [INFO] Bound to port %d\n", ntohs(addr.sin_port));

    int listenret = listen(sock, 5);
    TEST(listenret == 0, "listen() succeeds");

    struct pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLIN;
    pfd.revents = 0;
    int pollret = poll(&pfd, 1, 10);
    TEST(pollret == 0, "poll() times out correctly");

#if NCCL_PLATFORM_WINDOWS
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif
}

void test_dl_functions(void)
{
    printf("\n=== Dynamic Loading ===\n");

#if NCCL_PLATFORM_WINDOWS
    void *handle = dlopen("kernel32.dll", RTLD_NOW);
    TEST(handle != NULL, "dlopen(kernel32.dll) succeeds");

    if (handle)
    {
        void *func = dlsym(handle, "GetCurrentProcessId");
        TEST(func != NULL, "dlsym(GetCurrentProcessId) succeeds");
        dlclose(handle);
    }

    handle = dlopen("nonexistent_xyz123.dll", RTLD_NOW);
    TEST(handle == NULL, "dlopen of non-existent library fails");

    const char *err = dlerror();
    TEST(err != NULL, "dlerror returns error message");
#else
    void *handle = dlopen(NULL, RTLD_NOW);
    TEST(handle != NULL, "dlopen(NULL) succeeds");
    if (handle)
        dlclose(handle);
#endif
}

void test_atomic_operations(void)
{
    printf("\n=== Atomic Operations ===\n");

    volatile long val = 0;

    NCCL_ATOMIC_STORE(&val, 100);
    long loaded = NCCL_ATOMIC_LOAD(&val);
    TEST(loaded == 100, "Atomic store/load works");

    NCCL_ATOMIC_ADD(&val, 50);
    loaded = NCCL_ATOMIC_LOAD(&val);
    TEST(loaded == 150, "Atomic add works");

    NCCL_ATOMIC_SUB(&val, 25);
    loaded = NCCL_ATOMIC_LOAD(&val);
    TEST(loaded == 125, "Atomic sub works");

    long expected = 125;
    int cas_result = NCCL_ATOMIC_CAS(&val, expected, 200);
    TEST(cas_result, "CAS with correct expected succeeds");
    loaded = NCCL_ATOMIC_LOAD(&val);
    TEST(loaded == 200, "Value updated after CAS");

    NCCL_MEMORY_BARRIER();
    NCCL_COMPILER_BARRIER();
    TEST(1, "Memory barriers don't crash");
}

void test_misc_functions(void)
{
    printf("\n=== Miscellaneous ===\n");

    pid_t pid = getpid();
    TEST(pid > 0, "getpid returns positive value");
    printf("  [INFO] PID = %d\n", (int)pid);

#if NCCL_PLATFORM_WINDOWS
    pid_t tid = gettid();
    TEST(tid > 0, "gettid returns positive value");
    printf("  [INFO] TID = %d\n", (int)tid);

    int nprocs = ncclGetNumProcessors();
    TEST(nprocs > 0, "ncclGetNumProcessors returns positive value");
    printf("  [INFO] Processors = %d\n", nprocs);
#endif

    /* Environment variable test */
    const char *home = getenv("PATH");
    TEST(home != NULL, "getenv(PATH) returns value");
}

#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_shm.h"

void test_socket_optimizations(void)
{
    printf("\n=== Socket Optimizations (NCCL Paper-based) ===\n");

    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    TEST(sock != INVALID_SOCKET, "Socket creation succeeds");

    /* Test high-throughput optimization (Simple protocol) */
    int ret = ncclSocketOptimize(sock);
    TEST(ret == 0, "ncclSocketOptimize succeeds (4MB buffers)");

    /* Verify buffer sizes were set */
    int bufSize = 0;
    int len = sizeof(bufSize);
    getsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char *)&bufSize, &len);
    TEST(bufSize >= 256 * 1024, "Send buffer size increased");
    printf("  [INFO] Send buffer size: %d bytes\n", bufSize);

    getsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char *)&bufSize, &len);
    TEST(bufSize >= 256 * 1024, "Receive buffer size increased");
    printf("  [INFO] Receive buffer size: %d bytes\n", bufSize);

    /* Verify TCP_NODELAY */
    int nodelay = 0;
    len = sizeof(nodelay);
    getsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char *)&nodelay, &len);
    TEST(nodelay == 1, "TCP_NODELAY enabled");
    printf("  [INFO] TCP_NODELAY: %d\n", nodelay);

    closesocket(sock);

    /* Test low-latency optimization (LL/LL128 protocol) */
    sock = socket(AF_INET, SOCK_STREAM, 0);
    TEST(sock != INVALID_SOCKET, "Socket creation succeeds");

    ret = ncclSocketOptimizeLowLatency(sock);
    TEST(ret == 0, "ncclSocketOptimizeLowLatency succeeds (256KB buffers)");

    closesocket(sock);
    WSACleanup();
}

void test_overlapped_io(void)
{
    printf("\n=== Overlapped I/O (Async Socket Operations) ===\n");

    struct ncclSocketOverlapped ov;
    char buffer[1024] = {0};

    int ret = ncclSocketOverlappedInit(&ov, buffer, sizeof(buffer));
    TEST(ret == 0, "ncclSocketOverlappedInit succeeds");
    TEST(ov.overlapped.hEvent != NULL, "Event handle created");
    TEST(ov.wsaBuf.buf == buffer, "Buffer pointer set correctly");
    TEST(ov.wsaBuf.len == sizeof(buffer), "Buffer size set correctly");

    ncclSocketOverlappedFree(&ov);
    TEST(ov.overlapped.hEvent == NULL, "Event handle freed");
    printf("  [INFO] Overlapped I/O structures initialized correctly\n");
}

void test_shm_advanced(void)
{
    printf("\n=== Shared Memory Enhancements ===\n");

    /* Test NUMA node detection */
    int numaNode = ncclShmGetCurrentNumaNode();
    TEST(numaNode >= 0, "ncclShmGetCurrentNumaNode returns valid node");
    printf("  [INFO] Current NUMA node: %d\n", numaNode);

    int numaCount = ncclShmGetNumaNodeCount();
    TEST(numaCount >= 1, "ncclShmGetNumaNodeCount returns at least 1");
    printf("  [INFO] System has %d NUMA node(s)\n", numaCount);

    /* Test large page size query */
    size_t largePageSize = ncclShmGetLargePageSize();
    printf("  [INFO] Large page size: %zu bytes\n", largePageSize);
    if (largePageSize > 0)
    {
        TEST(largePageSize >= 4096, "Large page size is reasonable");
        TEST((largePageSize & (largePageSize - 1)) == 0, "Large page size is power of 2");
    }
    else
    {
        printf("  [INFO] Large pages not supported on this system\n");
    }

    /* Test basic shared memory (without advanced features) */
    ncclShmHandle_win32_t handle = NULL;
    void *ptr = NULL;
    const char *name = "test_shm_advanced";
    size_t size = 4096;

    int ret = ncclShmOpen(name, size, 1, &handle, &ptr);
    TEST(ret == 0, "ncclShmOpen succeeds");
    TEST(handle != NULL, "Handle is valid");
    TEST(ptr != NULL, "Mapped pointer is valid");

    /* Test read/write */
    if (ptr != NULL)
    {
        memset(ptr, 0xAB, size);
        TEST(((unsigned char *)ptr)[0] == 0xAB, "Memory write succeeds");
        TEST(((unsigned char *)ptr)[size - 1] == 0xAB, "Memory write covers full region");
    }

    /* Test NUMA-aware allocation (may not have elevated privileges for large pages) */
    ncclShmHandle_win32_t numaHandle = NULL;
    void *numaPtr = NULL;
    const char *numaName = "test_shm_numa";

    ret = ncclShmOpenAdvanced(numaName, size, NCCL_SHM_NUMA_AWARE, -1, &numaHandle, &numaPtr);
    if (ret == 0)
    {
        TEST(numaHandle != NULL, "NUMA-aware handle is valid");
        TEST(numaPtr != NULL, "NUMA-aware pointer is valid");
        printf("  [INFO] NUMA-aware allocation succeeded on node %d\n", numaHandle->numaNode);

        ncclShmClose_win32(numaHandle);
    }
    else
    {
        printf("  [INFO] NUMA-aware allocation not available (this is OK)\n");
    }

    /* Clean up */
    ncclShmClose_win32(handle);
}
#endif

/* =================================================================== */
/*                           MAIN                                      */
/* =================================================================== */

int main(int argc, char *argv[])
{
    printf("========================================\n");
    printf("NCCL Platform Abstraction Standalone Test\n");
    printf("========================================\n");

    test_platform_macros();
    test_time_functions();
    test_thread_functions();
    test_cpu_affinity();
    test_socket_functions();
    test_dl_functions();
    test_atomic_operations();
    test_misc_functions();

#if NCCL_PLATFORM_WINDOWS
    test_socket_optimizations();
    test_overlapped_io();
    test_shm_advanced();
#endif

    printf("\n========================================\n");
    printf("Results: %d run, %d passed, %d failed\n",
           tests_run, tests_passed, tests_failed);
    printf("========================================\n");

    return (tests_failed > 0) ? 1 : 0;
}
