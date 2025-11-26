/*************************************************************************
 * NCCL Windows Platform Optimizations Benchmark
 *
 * Benchmarks the socket and shared memory optimizations based on
 * findings from "Demystifying NCCL" (arXiv:2507.04786v2)
 *
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"

#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_misc.h"
#include "platform/win32_thread.h"
#include "platform/win32_socket.h"
#include "platform/win32_shm.h"
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")

/* Number of iterations for benchmarks */
#define WARMUP_ITERATIONS 100
#define BENCHMARK_ITERATIONS 10000

/* Message sizes to test (matching NCCL protocol boundaries) */
static const size_t message_sizes[] = {
    64,              /* Tiny message */
    1024,            /* 1 KB - LL protocol */
    8 * 1024,        /* 8 KB - LL protocol */
    64 * 1024,       /* 64 KB - transition point */
    256 * 1024,      /* 256 KB - LL128 buffer size */
    512 * 1024,      /* 512 KB - FIFO buffer size */
    1024 * 1024,     /* 1 MB */
    4 * 1024 * 1024, /* 4 MB - Simple protocol buffer */
};
static const int num_sizes = sizeof(message_sizes) / sizeof(message_sizes[0]);

/* High-resolution timer */
static inline double get_time_us(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000000.0 / (double)freq.QuadPart;
}

/* =================================================================== */
/*                    SOCKET CONFIGURATION BENCHMARK                    */
/* =================================================================== */

void benchmark_socket_configuration(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Socket Configuration Benchmark\n");
    printf("============================================================\n");

    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    double time_default = 0, time_optimized = 0, time_lowlatency = 0;

    /* Benchmark default socket creation */
    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
        closesocket(sock);
    }
    time_default = get_time_us() - start;

    /* Benchmark optimized socket creation (Simple protocol) */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
        ncclSocketOptimize(sock);
        closesocket(sock);
    }
    time_optimized = get_time_us() - start;

    /* Benchmark low-latency socket creation (LL protocol) */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
        ncclSocketOptimizeLowLatency(sock);
        closesocket(sock);
    }
    time_lowlatency = get_time_us() - start;

    printf("\nSocket Configuration (%d iterations):\n", BENCHMARK_ITERATIONS);
    printf("  Default socket:        %8.2f us/op\n", time_default / BENCHMARK_ITERATIONS);
    printf("  Optimized (4MB buf):   %8.2f us/op (+%.1f%%)\n",
           time_optimized / BENCHMARK_ITERATIONS,
           (time_optimized - time_default) / time_default * 100);
    printf("  Low-latency (256KB):   %8.2f us/op (+%.1f%%)\n",
           time_lowlatency / BENCHMARK_ITERATIONS,
           (time_lowlatency - time_default) / time_default * 100);

    WSACleanup();
}

/* =================================================================== */
/*                    OVERLAPPED I/O BENCHMARK                          */
/* =================================================================== */

void benchmark_overlapped_io(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Overlapped I/O Setup/Teardown Benchmark\n");
    printf("============================================================\n");

    char buffer[4096];
    double time_init = 0, time_free = 0;

    /* Benchmark overlapped I/O initialization */
    struct ncclSocketOverlapped ov[100];

    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        ncclSocketOverlappedInit(&ov[i % 100], buffer, sizeof(buffer));
    }
    time_init = get_time_us() - start;

    /* Benchmark overlapped I/O cleanup */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        ncclSocketOverlappedFree(&ov[i % 100]);
        /* Re-initialize for next iteration */
        if (i < BENCHMARK_ITERATIONS - 100)
            ncclSocketOverlappedInit(&ov[i % 100], buffer, sizeof(buffer));
    }
    time_free = get_time_us() - start;

    /* Clean up remaining */
    for (int i = 0; i < 100; i++)
    {
        if (ov[i].overlapped.hEvent != NULL)
            ncclSocketOverlappedFree(&ov[i]);
    }

    printf("\nOverlapped I/O (%d iterations):\n", BENCHMARK_ITERATIONS);
    printf("  Init (CreateEvent):    %8.2f us/op\n", time_init / BENCHMARK_ITERATIONS);
    printf("  Free (CloseHandle):    %8.2f us/op\n", time_free / BENCHMARK_ITERATIONS);
}

/* =================================================================== */
/*                    SHARED MEMORY BENCHMARK                           */
/* =================================================================== */

void benchmark_shared_memory(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Shared Memory Allocation Benchmark\n");
    printf("============================================================\n");

    const int iterations = 1000; /* Fewer iterations due to resource limits */

    for (int s = 0; s < num_sizes; s++)
    {
        size_t size = message_sizes[s];
        double time_basic = 0, time_numa = 0;
        char name[64];

        /* Skip very large allocations for many iterations */
        int iter = (size > 1024 * 1024) ? 100 : iterations;

        /* Benchmark basic shared memory */
        double start = get_time_us();
        for (int i = 0; i < iter; i++)
        {
            ncclShmHandle_win32_t handle;
            void *ptr;
            snprintf(name, sizeof(name), "bench_basic_%d_%d", (int)size, i);
            if (ncclShmOpen(name, size, 1, &handle, &ptr) == 0)
            {
                ncclShmClose_win32(handle);
            }
        }
        time_basic = get_time_us() - start;

        /* Benchmark NUMA-aware shared memory */
        start = get_time_us();
        for (int i = 0; i < iter; i++)
        {
            ncclShmHandle_win32_t handle;
            void *ptr;
            snprintf(name, sizeof(name), "bench_numa_%d_%d", (int)size, i);
            if (ncclShmOpenAdvanced(name, size, NCCL_SHM_NUMA_AWARE, -1, &handle, &ptr) == 0)
            {
                ncclShmClose_win32(handle);
            }
        }
        time_numa = get_time_us() - start;

        const char *size_str;
        if (size >= 1024 * 1024)
            printf("  %4zu MB:  Basic: %8.2f us/op  NUMA-aware: %8.2f us/op\n",
                   size / (1024 * 1024), time_basic / iter, time_numa / iter);
        else if (size >= 1024)
            printf("  %4zu KB:  Basic: %8.2f us/op  NUMA-aware: %8.2f us/op\n",
                   size / 1024, time_basic / iter, time_numa / iter);
        else
            printf("  %4zu B:   Basic: %8.2f us/op  NUMA-aware: %8.2f us/op\n",
                   size, time_basic / iter, time_numa / iter);
    }
}

/* =================================================================== */
/*                    MEMORY ACCESS BENCHMARK                           */
/* =================================================================== */

void benchmark_memory_access(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Shared Memory Access Pattern Benchmark\n");
    printf("============================================================\n");

    const size_t size = 4 * 1024 * 1024; /* 4 MB - NCCL Simple protocol buffer */
    const int iterations = 100;

    /* Use unique names with timestamp to avoid conflicts */
    char name_basic[64], name_numa[64];
    DWORD pid = GetCurrentProcessId();
    snprintf(name_basic, sizeof(name_basic), "bench_mem_%lu_basic", (unsigned long)pid);
    snprintf(name_numa, sizeof(name_numa), "bench_mem_%lu_numa", (unsigned long)pid);

    /* Create basic shared memory */
    ncclShmHandle_win32_t handle_basic;
    void *ptr_basic;
    if (ncclShmOpen(name_basic, size, 1, &handle_basic, &ptr_basic) != 0)
    {
        printf("  Failed to create basic shared memory (error %lu)\n", GetLastError());
        return;
    }

    /* Create NUMA-aware shared memory */
    ncclShmHandle_win32_t handle_numa;
    void *ptr_numa;
    if (ncclShmOpenAdvanced(name_numa, size, NCCL_SHM_NUMA_AWARE, -1,
                            &handle_numa, &ptr_numa) != 0)
    {
        printf("  Failed to create NUMA-aware shared memory (error %lu)\n", GetLastError());
        ncclShmClose_win32(handle_basic);
        return;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++)
    {
        memset(ptr_basic, i, size);
        memset(ptr_numa, i, size);
    }

    /* Benchmark sequential write (memset) */
    double time_basic_write, time_numa_write;

    double start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        memset(ptr_basic, i & 0xFF, size);
    }
    time_basic_write = get_time_us() - start;

    start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        memset(ptr_numa, i & 0xFF, size);
    }
    time_numa_write = get_time_us() - start;

    /* Benchmark sequential read (memcmp) */
    double time_basic_read, time_numa_read;
    char *temp = (char *)malloc(size);

    start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        memcpy(temp, ptr_basic, size);
    }
    time_basic_read = get_time_us() - start;

    start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        memcpy(temp, ptr_numa, size);
    }
    time_numa_read = get_time_us() - start;

    free(temp);

    /* Calculate bandwidth */
    double bw_basic_write = (double)size * iterations / (time_basic_write / 1000000.0) / (1024 * 1024 * 1024);
    double bw_numa_write = (double)size * iterations / (time_numa_write / 1000000.0) / (1024 * 1024 * 1024);
    double bw_basic_read = (double)size * iterations / (time_basic_read / 1000000.0) / (1024 * 1024 * 1024);
    double bw_numa_read = (double)size * iterations / (time_numa_read / 1000000.0) / (1024 * 1024 * 1024);

    printf("\nMemory Access (4 MB buffer, %d iterations):\n", iterations);
    printf("  Sequential Write:\n");
    printf("    Basic:       %8.2f GB/s\n", bw_basic_write);
    printf("    NUMA-aware:  %8.2f GB/s (%.1f%% vs basic)\n",
           bw_numa_write, (bw_numa_write - bw_basic_write) / bw_basic_write * 100);
    printf("  Sequential Read:\n");
    printf("    Basic:       %8.2f GB/s\n", bw_basic_read);
    printf("    NUMA-aware:  %8.2f GB/s (%.1f%% vs basic)\n",
           bw_numa_read, (bw_numa_read - bw_basic_read) / bw_basic_read * 100);

    ncclShmClose_win32(handle_basic);
    ncclShmClose_win32(handle_numa);
}

/* =================================================================== */
/*                    LOOPBACK THROUGHPUT BENCHMARK                     */
/* =================================================================== */

struct LoopbackArgs
{
    SOCKET server_sock;
    SOCKET client_sock;
    size_t msg_size;
    int iterations;
    volatile int ready;
    double throughput;
};

DWORD WINAPI loopback_server_thread(LPVOID arg)
{
    struct LoopbackArgs *args = (struct LoopbackArgs *)arg;
    char *buffer = (char *)malloc(args->msg_size);

    /* Accept connection */
    struct sockaddr_in client_addr;
    int addr_len = sizeof(client_addr);
    SOCKET client = accept(args->server_sock, (struct sockaddr *)&client_addr, &addr_len);

    args->ready = 1;

    /* Receive data */
    for (int i = 0; i < args->iterations; i++)
    {
        size_t received = 0;
        while (received < args->msg_size)
        {
            int n = recv(client, buffer + received, (int)(args->msg_size - received), 0);
            if (n <= 0)
                break;
            received += n;
        }
    }

    closesocket(client);
    free(buffer);
    return 0;
}

void benchmark_loopback_throughput(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Loopback Socket Throughput Benchmark\n");
    printf("============================================================\n");

    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    printf("\n%-12s %12s %12s %12s\n", "Size", "Default", "Optimized", "Improvement");
    printf("%-12s %12s %12s %12s\n", "----", "-------", "---------", "-----------");

    /* Test a subset of sizes for throughput */
    size_t throughput_sizes[] = {64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024};
    int num_throughput_sizes = 4;

    for (int s = 0; s < num_throughput_sizes; s++)
    {
        size_t msg_size = throughput_sizes[s];
        int iterations = (msg_size >= 1024 * 1024) ? 100 : 1000;

        double throughput_default = 0, throughput_optimized = 0;

        /* Test with default sockets */
        {
            SOCKET server_sock = socket(AF_INET, SOCK_STREAM, 0);
            struct sockaddr_in addr = {0};
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            addr.sin_port = 0;
            bind(server_sock, (struct sockaddr *)&addr, sizeof(addr));

            int addr_len = sizeof(addr);
            getsockname(server_sock, (struct sockaddr *)&addr, &addr_len);
            listen(server_sock, 1);

            struct LoopbackArgs args = {0};
            args.server_sock = server_sock;
            args.msg_size = msg_size;
            args.iterations = iterations;
            args.ready = 0;

            HANDLE thread = CreateThread(NULL, 0, loopback_server_thread, &args, 0, NULL);

            SOCKET client_sock = socket(AF_INET, SOCK_STREAM, 0);
            connect(client_sock, (struct sockaddr *)&addr, sizeof(addr));

            while (!args.ready)
                Sleep(1);

            char *buffer = (char *)malloc(msg_size);
            memset(buffer, 'A', msg_size);

            double start = get_time_us();
            for (int i = 0; i < iterations; i++)
            {
                send(client_sock, buffer, (int)msg_size, 0);
            }
            double elapsed = get_time_us() - start;

            throughput_default = (double)msg_size * iterations / (elapsed / 1000000.0) / (1024 * 1024);

            WaitForSingleObject(thread, INFINITE);
            CloseHandle(thread);
            closesocket(client_sock);
            closesocket(server_sock);
            free(buffer);
        }

        /* Test with optimized sockets */
        {
            SOCKET server_sock = socket(AF_INET, SOCK_STREAM, 0);
            ncclSocketOptimize(server_sock);
            struct sockaddr_in addr = {0};
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            addr.sin_port = 0;
            bind(server_sock, (struct sockaddr *)&addr, sizeof(addr));

            int addr_len = sizeof(addr);
            getsockname(server_sock, (struct sockaddr *)&addr, &addr_len);
            listen(server_sock, 1);

            struct LoopbackArgs args = {0};
            args.server_sock = server_sock;
            args.msg_size = msg_size;
            args.iterations = iterations;
            args.ready = 0;

            HANDLE thread = CreateThread(NULL, 0, loopback_server_thread, &args, 0, NULL);

            SOCKET client_sock = socket(AF_INET, SOCK_STREAM, 0);
            ncclSocketOptimize(client_sock);
            connect(client_sock, (struct sockaddr *)&addr, sizeof(addr));

            while (!args.ready)
                Sleep(1);

            char *buffer = (char *)malloc(msg_size);
            memset(buffer, 'A', msg_size);

            double start = get_time_us();
            for (int i = 0; i < iterations; i++)
            {
                send(client_sock, buffer, (int)msg_size, 0);
            }
            double elapsed = get_time_us() - start;

            throughput_optimized = (double)msg_size * iterations / (elapsed / 1000000.0) / (1024 * 1024);

            WaitForSingleObject(thread, INFINITE);
            CloseHandle(thread);
            closesocket(client_sock);
            closesocket(server_sock);
            free(buffer);
        }

        double improvement = (throughput_optimized - throughput_default) / throughput_default * 100;

        if (msg_size >= 1024 * 1024)
            printf("%4zu MB      %8.1f MB/s %8.1f MB/s    %+.1f%%\n",
                   msg_size / (1024 * 1024), throughput_default, throughput_optimized, improvement);
        else
            printf("%4zu KB      %8.1f MB/s %8.1f MB/s    %+.1f%%\n",
                   msg_size / 1024, throughput_default, throughput_optimized, improvement);
    }

    WSACleanup();
}

/* =================================================================== */
/*                    NUMA DETECTION BENCHMARK                          */
/* =================================================================== */

void benchmark_numa_detection(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("NUMA Detection Benchmark\n");
    printf("============================================================\n");

    double time_node = 0, time_count = 0, time_page = 0;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        ncclShmGetCurrentNumaNode();
        ncclShmGetNumaNodeCount();
        ncclShmGetLargePageSize();
    }

    /* Benchmark NUMA node detection */
    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        volatile int node = ncclShmGetCurrentNumaNode();
        (void)node;
    }
    time_node = get_time_us() - start;

    /* Benchmark NUMA node count */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        volatile int count = ncclShmGetNumaNodeCount();
        (void)count;
    }
    time_count = get_time_us() - start;

    /* Benchmark large page size query */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        volatile size_t size = ncclShmGetLargePageSize();
        (void)size;
    }
    time_page = get_time_us() - start;

    printf("\nNUMA/Large Page Queries (%d iterations):\n", BENCHMARK_ITERATIONS);
    printf("  GetCurrentNumaNode:    %8.3f us/op\n", time_node / BENCHMARK_ITERATIONS);
    printf("  GetNumaNodeCount:      %8.3f us/op\n", time_count / BENCHMARK_ITERATIONS);
    printf("  GetLargePageSize:      %8.3f us/op\n", time_page / BENCHMARK_ITERATIONS);

    printf("\nSystem Information:\n");
    printf("  Current NUMA node:     %d\n", ncclShmGetCurrentNumaNode());
    printf("  Total NUMA nodes:      %d\n", ncclShmGetNumaNodeCount());
    printf("  Large page size:       %zu KB\n", ncclShmGetLargePageSize() / 1024);
}

/* =================================================================== */
/*                           MAIN                                       */
/* =================================================================== */

void benchmark_thread_priority(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Thread Priority Optimization Benchmark\n");
    printf("============================================================\n");

    double time_setprio = 0, time_getprio = 0, time_boost = 0;
    HANDLE thread = GetCurrentThread();

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        ncclSetThreadPriority(thread, 1);
        ncclGetThreadPriority(thread);
        ncclSetThreadPriority(thread, 0);
    }

    /* Benchmark set priority */
    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        ncclSetThreadPriority(thread, i % 4);
    }
    time_setprio = get_time_us() - start;
    ncclSetThreadPriority(thread, 0); /* Reset */

    /* Benchmark get priority */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        volatile int p = ncclGetThreadPriority(thread);
        (void)p;
    }
    time_getprio = get_time_us() - start;

    /* Benchmark priority boost toggle */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        ncclThreadPriorityBoost(i & 1);
    }
    time_boost = get_time_us() - start;
    ncclThreadPriorityBoost(1); /* Re-enable */

    printf("\nThread Priority Operations (%d iterations):\n", BENCHMARK_ITERATIONS);
    printf("  SetThreadPriority:     %8.3f us/op\n", time_setprio / BENCHMARK_ITERATIONS);
    printf("  GetThreadPriority:     %8.3f us/op\n", time_getprio / BENCHMARK_ITERATIONS);
    printf("  PriorityBoost toggle:  %8.3f us/op\n", time_boost / BENCHMARK_ITERATIONS);
}

void benchmark_timer_resolution(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Timer Resolution Benchmark\n");
    printf("============================================================\n");

    ULONG minRes = 0, maxRes = 0, currentRes = 0;

    /* Get current resolution */
    if (ncclGetTimerResolution(&minRes, &maxRes, &currentRes) == 0)
    {
        printf("\nTimer Resolution (100ns units):\n");
        printf("  Minimum (best):        %lu (%.3f ms)\n", minRes, minRes / 10000.0);
        printf("  Maximum (worst):       %lu (%.3f ms)\n", maxRes, maxRes / 10000.0);
        printf("  Current:               %lu (%.3f ms)\n", currentRes, currentRes / 10000.0);
    }
    else
    {
        printf("  Timer resolution query not available\n");
    }

    /* Test high resolution timer impact */
    ULONG actualRes = 0;
    if (ncclEnableHighResolutionTimer(&actualRes) == 0)
    {
        printf("\nHigh Resolution Timer Enabled:\n");
        printf("  Actual resolution:     %lu (%.3f ms)\n", actualRes, actualRes / 10000.0);

        /* Measure sleep accuracy */
        double sleepTotal = 0;
        int sleepCount = 100;
        for (int i = 0; i < sleepCount; i++)
        {
            double start = get_time_us();
            Sleep(1);
            sleepTotal += get_time_us() - start;
        }
        printf("  Sleep(1) actual avg:   %.2f ms\n", sleepTotal / sleepCount / 1000.0);

        ncclDisableHighResolutionTimer();
    }
    else
    {
        printf("\n  High resolution timer not available\n");
    }

    /* Measure without high res timer */
    double sleepTotal = 0;
    int sleepCount = 100;
    for (int i = 0; i < sleepCount; i++)
    {
        double start = get_time_us();
        Sleep(1);
        sleepTotal += get_time_us() - start;
    }
    printf("  Sleep(1) default avg:  %.2f ms\n", sleepTotal / sleepCount / 1000.0);
}

void benchmark_advanced_socket(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Advanced Socket Optimizations Benchmark\n");
    printf("============================================================\n");

    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    /* Test loopback fast path */
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    int fastpath_result = ncclSocketEnableLoopbackFastPath(sock);
    printf("\nLoopback Fast Path: %s\n",
           fastpath_result == 0 ? "Enabled" : "Not available");
    closesocket(sock);

    /* Test TCP Fast Open */
    sock = socket(AF_INET, SOCK_STREAM, 0);
    int fastopen_result = ncclSocketEnableFastOpen(sock);
    printf("TCP Fast Open:      %s\n",
           fastopen_result == 0 ? "Enabled" : "Not available");
    closesocket(sock);

    /* Benchmark different socket optimization modes */
    double time_ultra = 0, time_max = 0;

    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
        ncclSocketOptimizeUltraLowLatency(s);
        closesocket(s);
    }
    time_ultra = get_time_us() - start;

    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
        ncclSocketOptimizeMaxThroughput(s);
        closesocket(s);
    }
    time_max = get_time_us() - start;

    printf("\nSocket Optimization Setup (%d iterations):\n", BENCHMARK_ITERATIONS);
    printf("  Ultra-low latency:     %8.2f us/op\n", time_ultra / BENCHMARK_ITERATIONS);
    printf("  Max throughput:        %8.2f us/op\n", time_max / BENCHMARK_ITERATIONS);

    WSACleanup();
}

void benchmark_memory_prefetch(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Memory Prefetch and Touch Benchmark\n");
    printf("============================================================\n");

    const size_t size = 4 * 1024 * 1024; /* 4 MB */
    const int iterations = 100;

    void *buffer = malloc(size);
    if (buffer == NULL)
    {
        printf("  Failed to allocate buffer\n");
        return;
    }

    /* Benchmark memory touch */
    double time_touch = 0, time_touchwrite = 0, time_prefetch = 0;

    double start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        ncclShmTouch(buffer, size);
    }
    time_touch = get_time_us() - start;

    start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        ncclShmTouchWrite(buffer, size);
    }
    time_touchwrite = get_time_us() - start;

    start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        ncclShmPrefetch(buffer, size);
    }
    time_prefetch = get_time_us() - start;

    printf("\nMemory Operations (4 MB buffer, %d iterations):\n", iterations);
    printf("  Touch (read):          %8.2f us/op\n", time_touch / iterations);
    printf("  Touch (write):         %8.2f us/op\n", time_touchwrite / iterations);
    printf("  Prefetch:              %8.2f us/op\n", time_prefetch / iterations);

    /* Benchmark memory copy with prefetch vs without */
    void *src = malloc(size);
    void *dst = malloc(size);
    if (src && dst)
    {
        memset(src, 'A', size);

        /* Without prefetch */
        start = get_time_us();
        for (int i = 0; i < iterations; i++)
        {
            memcpy(dst, src, size);
        }
        double time_copy = get_time_us() - start;

        /* With prefetch copy */
        start = get_time_us();
        for (int i = 0; i < iterations; i++)
        {
            ncclShmCopy(dst, src, size);
        }
        double time_prefetch_copy = get_time_us() - start;

        double bw_copy = (double)size * iterations / (time_copy / 1000000.0) / (1024 * 1024 * 1024);
        double bw_prefetch = (double)size * iterations / (time_prefetch_copy / 1000000.0) / (1024 * 1024 * 1024);

        printf("\nMemory Copy (4 MB):\n");
        printf("  Standard memcpy:       %8.2f GB/s\n", bw_copy);
        printf("  With prefetch:         %8.2f GB/s (%.1f%%)\n",
               bw_prefetch, (bw_prefetch - bw_copy) / bw_copy * 100);

        free(src);
        free(dst);
    }

    free(buffer);
}

void benchmark_iocp(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("I/O Completion Port Benchmark\n");
    printf("============================================================\n");

    struct ncclSocketIOCP iocp;
    double time_create = 0, time_associate = 0, time_destroy = 0;
    int iterations = 1000;

    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    /* Benchmark IOCP creation */
    double start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        ncclSocketIOCPCreate(&iocp, 0);
        ncclSocketIOCPDestroy(&iocp);
    }
    time_create = get_time_us() - start;

    /* Create IOCP for association test */
    ncclSocketIOCPCreate(&iocp, 0);

    /* Benchmark socket association */
    start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
        ncclSocketIOCPAssociate(&iocp, sock, (ULONG_PTR)i);
        closesocket(sock);
    }
    time_associate = get_time_us() - start;

    ncclSocketIOCPDestroy(&iocp);

    printf("\nIOCP Operations (%d iterations):\n", iterations);
    printf("  Create/Destroy:        %8.2f us/op\n", time_create / iterations);
    printf("  Associate socket:      %8.2f us/op\n", time_associate / iterations);

    WSACleanup();
}

void benchmark_spinlock(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Spinlock Benchmark\n");
    printf("============================================================\n");

    pthread_spinlock_t spinlock;
    pthread_mutex_t mutex;
    CRITICAL_SECTION cs;
    double time_spin = 0, time_mutex = 0, time_cs = 0;

    pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);
    pthread_mutex_init(&mutex, NULL);
    InitializeCriticalSection(&cs);

    /* Benchmark spinlock */
    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        pthread_spin_lock(&spinlock);
        pthread_spin_unlock(&spinlock);
    }
    time_spin = get_time_us() - start;

    /* Benchmark mutex (CRITICAL_SECTION wrapper) */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        pthread_mutex_lock(&mutex);
        pthread_mutex_unlock(&mutex);
    }
    time_mutex = get_time_us() - start;

    /* Benchmark raw CRITICAL_SECTION */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        EnterCriticalSection(&cs);
        LeaveCriticalSection(&cs);
    }
    time_cs = get_time_us() - start;

    pthread_spin_destroy(&spinlock);
    pthread_mutex_destroy(&mutex);
    DeleteCriticalSection(&cs);

    printf("\nLock Operations (%d iterations, uncontended):\n", BENCHMARK_ITERATIONS);
    printf("  Spinlock:              %8.3f ns/op\n", time_spin / BENCHMARK_ITERATIONS * 1000);
    printf("  Mutex (CS wrapper):    %8.3f ns/op\n", time_mutex / BENCHMARK_ITERATIONS * 1000);
    printf("  Raw CRITICAL_SECTION:  %8.3f ns/op\n", time_cs / BENCHMARK_ITERATIONS * 1000);
}

void benchmark_memory_barriers(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Memory Barrier and Atomic Operations Benchmark\n");
    printf("============================================================\n");

    volatile LONG value = 0;
    volatile LONGLONG value64 = 0;
    double time_fence = 0, time_load = 0, time_store = 0;
    double time_add = 0, time_cas = 0, time_pause = 0;

    /* Benchmark full memory fence */
    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        ncclMemoryFence();
    }
    time_fence = get_time_us() - start;

    /* Benchmark atomic load with acquire */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        volatile LONG v = ncclAtomicLoadAcquire(&value);
        (void)v;
    }
    time_load = get_time_us() - start;

    /* Benchmark atomic store with release */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        ncclAtomicStoreRelease(&value, (LONG)i);
    }
    time_store = get_time_us() - start;

    /* Benchmark atomic add */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        ncclAtomicAdd(&value, 1);
    }
    time_add = get_time_us() - start;

    /* Benchmark CAS */
    value = 0;
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        ncclAtomicCAS(&value, (LONG)i, (LONG)(i + 1));
    }
    time_cas = get_time_us() - start;

    /* Benchmark CPU pause */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        ncclCpuPause();
    }
    time_pause = get_time_us() - start;

    printf("\nMemory/Atomic Operations (%d iterations):\n", BENCHMARK_ITERATIONS);
    printf("  Memory fence:          %8.3f ns/op\n", time_fence / BENCHMARK_ITERATIONS * 1000);
    printf("  Atomic load (acquire): %8.3f ns/op\n", time_load / BENCHMARK_ITERATIONS * 1000);
    printf("  Atomic store (release):%8.3f ns/op\n", time_store / BENCHMARK_ITERATIONS * 1000);
    printf("  Atomic add:            %8.3f ns/op\n", time_add / BENCHMARK_ITERATIONS * 1000);
    printf("  Atomic CAS:            %8.3f ns/op\n", time_cas / BENCHMARK_ITERATIONS * 1000);
    printf("  CPU pause:             %8.3f ns/op\n", time_pause / BENCHMARK_ITERATIONS * 1000);
}

void benchmark_precision_sleep(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("High-Precision Sleep Benchmark\n");
    printf("============================================================\n");

    int iterations = 100;
    double time_sleep = 0, time_nano = 0, time_busy = 0;

    /* Test target delays */
    LONGLONG delays[] = {100, 500, 1000, 5000, 10000}; /* nanoseconds */
    int num_delays = sizeof(delays) / sizeof(delays[0]);

    printf("\nSleep Accuracy (target vs actual):\n");
    printf("%-12s %12s %12s %12s\n", "Target", "Sleep(0)", "NanoSleep", "BusyWait");

    for (int d = 0; d < num_delays; d++)
    {
        LONGLONG target_ns = delays[d];

        /* Sleep(0) */
        double start = get_time_us();
        for (int i = 0; i < iterations; i++)
        {
            Sleep(0);
        }
        double sleep0_actual = (get_time_us() - start) / iterations * 1000; /* ns */

        /* ncclNanoSleep */
        start = get_time_us();
        for (int i = 0; i < iterations; i++)
        {
            ncclNanoSleep(target_ns);
        }
        double nano_actual = (get_time_us() - start) / iterations * 1000;

        /* ncclBusyWaitNanos */
        start = get_time_us();
        for (int i = 0; i < iterations; i++)
        {
            ncclBusyWaitNanos(target_ns);
        }
        double busy_actual = (get_time_us() - start) / iterations * 1000;

        printf("%8lld ns  %8.0f ns  %8.0f ns  %8.0f ns\n",
               target_ns, sleep0_actual, nano_actual, busy_actual);
    }
}

void benchmark_processor_groups(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Processor Group Information\n");
    printf("============================================================\n");

    int groups = ncclGetProcessorGroupCount();
    int total = ncclGetTotalProcessorCount();

    printf("\nProcessor Topology:\n");
    printf("  Processor groups:      %d\n", groups);
    printf("  Total logical CPUs:    %d\n", total);

    for (int g = 0; g < groups; g++)
    {
        int cpus = ncclGetProcessorCountInGroup(g);
        printf("  Group %d processors:    %d\n", g, cpus);
    }

    int curGroup, curProc;
    ncclGetCurrentProcessorInfo(&curGroup, &curProc);
    printf("\nCurrent Thread:\n");
    printf("  Processor group:       %d\n", curGroup);
    printf("  Processor number:      %d\n", curProc);
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    printf("================================================================\n");
    printf("NCCL Windows Platform Optimizations Benchmark\n");
    printf("Based on 'Demystifying NCCL' (arXiv:2507.04786v2)\n");
    printf("================================================================\n");

    benchmark_socket_configuration();
    benchmark_overlapped_io();
    benchmark_numa_detection();
    benchmark_shared_memory();
    benchmark_memory_access();
    benchmark_loopback_throughput();

    /* Thread and timer benchmarks */
    benchmark_thread_priority();
    benchmark_timer_resolution();
    benchmark_advanced_socket();
    benchmark_memory_prefetch();
    benchmark_iocp();

    /* New low-level benchmarks */
    benchmark_spinlock();
    benchmark_memory_barriers();
    benchmark_precision_sleep();
    benchmark_processor_groups();

    printf("\n================================================================\n");
    printf("Benchmark Complete\n");
    printf("================================================================\n");

    return 0;
}

#else /* !NCCL_PLATFORM_WINDOWS */

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    printf("This benchmark is Windows-specific.\n");
    return 0;
}

#endif /* NCCL_PLATFORM_WINDOWS */
