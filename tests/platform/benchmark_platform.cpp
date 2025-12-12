/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Performance Benchmarks for Windows Platform Abstraction
 * Measures latency and throughput of key operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "platform.h"

#define WARMUP_ITERATIONS 1000
#define BENCHMARK_ITERATIONS 100000

/* Get current time in nanoseconds */
static inline int64_t get_time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Statistics structure */
typedef struct
{
    double min_ns;
    double max_ns;
    double avg_ns;
    double stddev_ns;
    double ops_per_sec;
} BenchmarkStats;

/* Calculate statistics from samples */
static void calculate_stats(int64_t *samples, int count, BenchmarkStats *stats)
{
    if (count == 0)
        return;

    double sum = 0;
    stats->min_ns = (double)samples[0];
    stats->max_ns = (double)samples[0];

    for (int i = 0; i < count; i++)
    {
        double val = (double)samples[i];
        sum += val;
        if (val < stats->min_ns)
            stats->min_ns = val;
        if (val > stats->max_ns)
            stats->max_ns = val;
    }

    stats->avg_ns = sum / count;

    /* Calculate stddev */
    double variance_sum = 0;
    for (int i = 0; i < count; i++)
    {
        double diff = (double)samples[i] - stats->avg_ns;
        variance_sum += diff * diff;
    }
    stats->stddev_ns = sqrt(variance_sum / count);

    stats->ops_per_sec = 1000000000.0 / stats->avg_ns;
}

/* Print benchmark results */
static void print_stats(const char *name, BenchmarkStats *stats)
{
    printf("  %-35s avg: %8.1f ns  min: %8.1f ns  max: %8.1f ns  stddev: %7.1f ns  (%.2fM ops/sec)\n",
           name, stats->avg_ns, stats->min_ns, stats->max_ns, stats->stddev_ns,
           stats->ops_per_sec / 1000000.0);
}

/* ===================== Time Function Benchmarks ===================== */

void benchmark_time_functions()
{
    printf("\n=== Time Function Benchmarks ===\n");

    int64_t *samples = (int64_t *)malloc(BENCHMARK_ITERATIONS * sizeof(int64_t));
    BenchmarkStats stats;
    struct timespec ts;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &ts);
    }

    /* Benchmark clock_gettime(CLOCK_MONOTONIC) */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        clock_gettime(CLOCK_MONOTONIC, &ts);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("clock_gettime(CLOCK_MONOTONIC)", &stats);

    /* Benchmark clock_gettime(CLOCK_REALTIME) */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        clock_gettime(CLOCK_REALTIME, &ts);
    }
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        clock_gettime(CLOCK_REALTIME, &ts);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("clock_gettime(CLOCK_REALTIME)", &stats);

    free(samples);
}

/* ===================== Mutex Benchmarks ===================== */

void benchmark_mutex()
{
    printf("\n=== Mutex Benchmarks ===\n");

    int64_t *samples = (int64_t *)malloc(BENCHMARK_ITERATIONS * sizeof(int64_t));
    BenchmarkStats stats;
    pthread_mutex_t mutex;

    pthread_mutex_init(&mutex, NULL);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        pthread_mutex_lock(&mutex);
        pthread_mutex_unlock(&mutex);
    }

    /* Benchmark uncontended lock/unlock */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        pthread_mutex_lock(&mutex);
        pthread_mutex_unlock(&mutex);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("mutex lock/unlock (uncontended)", &stats);

    /* Benchmark trylock success */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        pthread_mutex_trylock(&mutex);
        pthread_mutex_unlock(&mutex);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("mutex trylock/unlock (uncontended)", &stats);

    pthread_mutex_destroy(&mutex);

    /* Benchmark mutex init/destroy */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        pthread_mutex_init(&mutex, NULL);
        pthread_mutex_destroy(&mutex);
    }
    for (int i = 0; i < BENCHMARK_ITERATIONS / 10; i++)
    { /* Fewer iterations - slower op */
        int64_t start = get_time_ns();
        pthread_mutex_init(&mutex, NULL);
        pthread_mutex_destroy(&mutex);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS / 10, &stats);
    print_stats("mutex init/destroy", &stats);

    free(samples);
}

/* ===================== CPU Affinity Benchmarks ===================== */

void benchmark_cpu_affinity()
{
    printf("\n=== CPU Affinity Benchmarks ===\n");

    int64_t *samples = (int64_t *)malloc(BENCHMARK_ITERATIONS * sizeof(int64_t));
    BenchmarkStats stats;
    cpu_set_t cpuset;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        CPU_ZERO(&cpuset);
    }

    /* Benchmark CPU_ZERO */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        CPU_ZERO(&cpuset);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("CPU_ZERO", &stats);

    /* Benchmark CPU_SET */
    CPU_ZERO(&cpuset);
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        CPU_SET(i % CPU_SETSIZE, &cpuset);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("CPU_SET", &stats);

    /* Benchmark CPU_ISSET */
    for (int i = 0; i < CPU_SETSIZE; i++)
        CPU_SET(i, &cpuset);
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        volatile int result = CPU_ISSET(i % CPU_SETSIZE, &cpuset);
        (void)result;
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("CPU_ISSET", &stats);

    /* Benchmark CPU_COUNT */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        volatile int count = CPU_COUNT(&cpuset);
        (void)count;
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("CPU_COUNT (full set)", &stats);

    /* Benchmark sched_getaffinity */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        sched_getaffinity(0, sizeof(cpuset), &cpuset);
    }
    for (int i = 0; i < BENCHMARK_ITERATIONS / 10; i++)
    { /* System call - slower */
        int64_t start = get_time_ns();
        sched_getaffinity(0, sizeof(cpuset), &cpuset);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS / 10, &stats);
    print_stats("sched_getaffinity", &stats);

    free(samples);
}

/* ===================== Atomic Benchmarks ===================== */

void benchmark_atomics()
{
    printf("\n=== Atomic Operation Benchmarks ===\n");

    int64_t *samples = (int64_t *)malloc(BENCHMARK_ITERATIONS * sizeof(int64_t));
    BenchmarkStats stats;

#if NCCL_PLATFORM_WINDOWS
    volatile LONG64 val64 = 0;
    volatile LONG val32 = 0;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        InterlockedIncrement64(&val64);
    }

    /* Benchmark InterlockedIncrement64 */
    val64 = 0;
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        InterlockedIncrement64(&val64);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("InterlockedIncrement64", &stats);

    /* Benchmark InterlockedIncrement (32-bit) */
    val32 = 0;
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        InterlockedIncrement(&val32);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("InterlockedIncrement (32-bit)", &stats);

    /* Benchmark InterlockedExchange64 + read */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        InterlockedExchange64(&val64, i);
        volatile LONG64 loaded = val64;
        (void)loaded;
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("InterlockedExchange64 + load", &stats);

    /* Benchmark InterlockedCompareExchange64 success */
    val64 = 0;
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        val64 = i;
        int64_t start = get_time_ns();
        InterlockedCompareExchange64(&val64, i + 1, i);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("InterlockedCompareExchange64", &stats);

    /* Benchmark memory barrier */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        MemoryBarrier();
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("MemoryBarrier", &stats);
#else
    volatile int64_t val64 = 0;
    volatile int32_t val32 = 0;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        __atomic_add_fetch(&val64, 1, __ATOMIC_SEQ_CST);
    }

    /* Benchmark atomic_add_fetch (64-bit) */
    val64 = 0;
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        __atomic_add_fetch(&val64, 1, __ATOMIC_SEQ_CST);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("atomic_add_fetch (64-bit)", &stats);

    /* Benchmark atomic_add_fetch (32-bit) */
    val32 = 0;
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        __atomic_add_fetch(&val32, 1, __ATOMIC_SEQ_CST);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("atomic_add_fetch (32-bit)", &stats);

    /* Benchmark atomic_store/load */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        __atomic_store_n(&val64, i, __ATOMIC_SEQ_CST);
        volatile int64_t loaded = __atomic_load_n(&val64, __ATOMIC_SEQ_CST);
        (void)loaded;
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("atomic store + load (64-bit)", &stats);

    /* Benchmark CAS success */
    val64 = 0;
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t expected = i;
        val64 = i;
        int64_t start = get_time_ns();
        __atomic_compare_exchange_n(&val64, &expected, i + 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("CAS (success case)", &stats);

    /* Benchmark memory barrier */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("memory fence (SEQ_CST)", &stats);
#endif

    free(samples);
}

/* ===================== Socket Benchmarks ===================== */

void benchmark_sockets()
{
    printf("\n=== Socket Benchmarks ===\n");

    int64_t *samples = (int64_t *)malloc(BENCHMARK_ITERATIONS * sizeof(int64_t));
    BenchmarkStats stats;

#if NCCL_PLATFORM_WINDOWS
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    /* Benchmark socket creation/close */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        closesocket(s);
    }
    for (int i = 0; i < BENCHMARK_ITERATIONS / 100; i++)
    { /* Slow operation */
        int64_t start = get_time_ns();
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        closesocket(s);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS / 100, &stats);
    print_stats("socket create/close", &stats);

    /* Benchmark poll with timeout 0 */
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    struct pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLIN;

    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        poll(&pfd, 1, 0);
    }
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        poll(&pfd, 1, 0);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("poll (1 fd, timeout=0)", &stats);

    closesocket(sock);

#if NCCL_PLATFORM_WINDOWS
    WSACleanup();
#endif

    free(samples);
}

/* ===================== Dynamic Loading Benchmarks ===================== */

void benchmark_dl()
{
    printf("\n=== Dynamic Loading Benchmarks ===\n");

    int64_t *samples = (int64_t *)malloc(BENCHMARK_ITERATIONS * sizeof(int64_t));
    BenchmarkStats stats;

    /* Load library once for symbol lookup benchmark */
    void *handle = ncclDlOpen("kernel32.dll", NCCL_RTLD_NOW);

    /* Benchmark dlsym */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        ncclDlSym(handle, "GetCurrentProcessId");
    }
    for (int i = 0; i < BENCHMARK_ITERATIONS / 10; i++)
    {
        int64_t start = get_time_ns();
        void *sym = ncclDlSym(handle, "GetCurrentProcessId");
        (void)sym;
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS / 10, &stats);
    print_stats("dlsym", &stats);

    ncclDlClose(handle);

    /* Benchmark dlopen/dlclose cycle */
    for (int i = 0; i < 100; i++)
    {
        void *h = ncclDlOpen("kernel32.dll", NCCL_RTLD_NOW);
        ncclDlClose(h);
    }
    for (int i = 0; i < 1000; i++)
    { /* Very slow operation */
        int64_t start = get_time_ns();
        void *h = ncclDlOpen("kernel32.dll", NCCL_RTLD_NOW);
        ncclDlClose(h);
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, 1000, &stats);
    print_stats("dlopen/dlclose cycle", &stats);

    free(samples);
}

/* ===================== Getpid/Gettid Benchmarks ===================== */

void benchmark_process_info()
{
    printf("\n=== Process Info Benchmarks ===\n");

    int64_t *samples = (int64_t *)malloc(BENCHMARK_ITERATIONS * sizeof(int64_t));
    BenchmarkStats stats;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        getpid();
    }

    /* Benchmark getpid */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        volatile int pid = getpid();
        (void)pid;
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("getpid", &stats);

    /* Benchmark gettid */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        gettid();
    }
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        volatile int tid = gettid();
        (void)tid;
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("gettid", &stats);

    /* Benchmark getenv */
    _putenv("NCCL_BENCH_VAR=test_value");
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        getenv("NCCL_BENCH_VAR");
    }
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int64_t start = get_time_ns();
        volatile const char *val = getenv("NCCL_BENCH_VAR");
        (void)val;
        int64_t end = get_time_ns();
        samples[i] = end - start;
    }
    calculate_stats(samples, BENCHMARK_ITERATIONS, &stats);
    print_stats("getenv", &stats);

    _putenv("NCCL_BENCH_VAR=");

    free(samples);
}

/* ===================== Main ===================== */

int main()
{
    printf("========================================\n");
    printf("NCCL Platform Abstraction Benchmarks\n");
    printf("========================================\n");
    printf("Iterations: %d (warmup: %d)\n", BENCHMARK_ITERATIONS, WARMUP_ITERATIONS);
    printf("CPU: %d processors\n", ncclGetNumProcessors());

    benchmark_time_functions();
    benchmark_mutex();
    benchmark_cpu_affinity();
    benchmark_atomics();
    benchmark_sockets();
    benchmark_dl();
    benchmark_process_info();

    printf("\n========================================\n");
    printf("Benchmarks Complete\n");
    printf("========================================\n");

    return 0;
}
