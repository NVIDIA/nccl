/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Cross-platform Performance Comparison Benchmarks
 * Provides comparable metrics between Windows and Linux implementations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "platform.h"

#define WARMUP_ITERATIONS 10000
#define BENCHMARK_ITERATIONS 1000000
#define NUM_SAMPLES 100

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
    double p50_ns;
    double p95_ns;
    double p99_ns;
    double stddev_ns;
    double ops_per_sec;
} BenchStats;

/* Compare function for qsort */
static int compare_int64(const void *a, const void *b)
{
    int64_t va = *(const int64_t *)a;
    int64_t vb = *(const int64_t *)b;
    return (va > vb) - (va < vb);
}

/* Calculate comprehensive statistics from samples */
static void calc_stats(int64_t *samples, int count, BenchStats *stats)
{
    if (count == 0)
        return;

    /* Sort for percentiles */
    qsort(samples, count, sizeof(int64_t), compare_int64);

    double sum = 0;
    stats->min_ns = (double)samples[0];
    stats->max_ns = (double)samples[count - 1];

    for (int i = 0; i < count; i++)
    {
        sum += (double)samples[i];
    }
    stats->avg_ns = sum / count;

    /* Percentiles */
    stats->p50_ns = (double)samples[count / 2];
    stats->p95_ns = (double)samples[(int)(count * 0.95)];
    stats->p99_ns = (double)samples[(int)(count * 0.99)];

    /* Standard deviation */
    double variance_sum = 0;
    for (int i = 0; i < count; i++)
    {
        double diff = (double)samples[i] - stats->avg_ns;
        variance_sum += diff * diff;
    }
    stats->stddev_ns = sqrt(variance_sum / count);

    stats->ops_per_sec = 1000000000.0 / stats->avg_ns;
}

/* Print benchmark header */
static void print_header(const char *title)
{
    printf("\n%s\n", title);
    printf("%-40s %10s %10s %10s %10s %10s %12s\n",
           "Operation", "Avg(ns)", "P50(ns)", "P95(ns)", "P99(ns)", "Stddev", "M ops/sec");
    printf("--------------------------------------------------------------------------------");
    printf("--------------------\n");
}

/* Print benchmark result */
static void print_result(const char *name, BenchStats *stats)
{
    printf("%-40s %10.1f %10.1f %10.1f %10.1f %10.1f %12.2f\n",
           name, stats->avg_ns, stats->p50_ns, stats->p95_ns, stats->p99_ns,
           stats->stddev_ns, stats->ops_per_sec / 1000000.0);
}

/* ===================== Benchmark: clock_gettime ===================== */
void bench_clock_gettime(int64_t *samples)
{
    struct timespec ts;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &ts);
    }

    /* Benchmark - batch iterations for stable measurements */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < BENCHMARK_ITERATIONS / NUM_SAMPLES; i++)
        {
            clock_gettime(CLOCK_MONOTONIC, &ts);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / (BENCHMARK_ITERATIONS / NUM_SAMPLES);
    }
}

/* ===================== Benchmark: Mutex lock/unlock ===================== */
void bench_mutex_lock_unlock(int64_t *samples)
{
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        pthread_mutex_lock(&mutex);
        pthread_mutex_unlock(&mutex);
    }

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < BENCHMARK_ITERATIONS / NUM_SAMPLES; i++)
        {
            pthread_mutex_lock(&mutex);
            pthread_mutex_unlock(&mutex);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / (BENCHMARK_ITERATIONS / NUM_SAMPLES);
    }

    pthread_mutex_destroy(&mutex);
}

/* ===================== Benchmark: Thread creation/join ===================== */
static void *thread_func(void *arg)
{
    (void)arg;
    return NULL;
}

void bench_thread_create_join(int64_t *samples)
{
    pthread_t thread;

    /* Warmup */
    for (int i = 0; i < 100; i++)
    {
        pthread_create(&thread, NULL, thread_func, NULL);
        pthread_join(thread, NULL);
    }

    /* Benchmark - fewer iterations due to cost */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < 100; i++)
        {
            pthread_create(&thread, NULL, thread_func, NULL);
            pthread_join(thread, NULL);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / 100;
    }
}

/* ===================== Benchmark: CPU_ZERO ===================== */
void bench_cpu_zero(int64_t *samples)
{
    cpu_set_t cpuset;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        CPU_ZERO(&cpuset);
    }

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < BENCHMARK_ITERATIONS / NUM_SAMPLES; i++)
        {
            CPU_ZERO(&cpuset);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / (BENCHMARK_ITERATIONS / NUM_SAMPLES);
    }
}

/* ===================== Benchmark: CPU_COUNT ===================== */
void bench_cpu_count(int64_t *samples)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < 64; i++)
        CPU_SET(i, &cpuset);

    volatile int count;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        count = CPU_COUNT(&cpuset);
    }
    (void)count;

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < BENCHMARK_ITERATIONS / NUM_SAMPLES; i++)
        {
            count = CPU_COUNT(&cpuset);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / (BENCHMARK_ITERATIONS / NUM_SAMPLES);
    }
}

/* ===================== Benchmark: sched_getaffinity ===================== */
void bench_sched_getaffinity(int64_t *samples)
{
    cpu_set_t cpuset;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        sched_getaffinity(0, sizeof(cpuset), &cpuset);
    }

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < 10000; i++)
        { /* Fewer iterations - syscall */
            sched_getaffinity(0, sizeof(cpuset), &cpuset);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / 10000;
    }
}

/* ===================== Benchmark: Atomic increment ===================== */
void bench_atomic_increment(int64_t *samples)
{
#if NCCL_PLATFORM_WINDOWS
    volatile LONG64 val = 0;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        InterlockedIncrement64(&val);
    }

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < BENCHMARK_ITERATIONS / NUM_SAMPLES; i++)
        {
            InterlockedIncrement64(&val);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / (BENCHMARK_ITERATIONS / NUM_SAMPLES);
    }
#else
    volatile int64_t val = 0;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        __atomic_add_fetch(&val, 1, __ATOMIC_SEQ_CST);
    }

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < BENCHMARK_ITERATIONS / NUM_SAMPLES; i++)
        {
            __atomic_add_fetch(&val, 1, __ATOMIC_SEQ_CST);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / (BENCHMARK_ITERATIONS / NUM_SAMPLES);
    }
#endif
}

/* ===================== Benchmark: CAS ===================== */
void bench_atomic_cas(int64_t *samples)
{
#if NCCL_PLATFORM_WINDOWS
    volatile LONG64 val = 0;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        val = i;
        InterlockedCompareExchange64(&val, i + 1, i);
    }

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < BENCHMARK_ITERATIONS / NUM_SAMPLES; i++)
        {
            val = i;
            InterlockedCompareExchange64(&val, i + 1, i);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / (BENCHMARK_ITERATIONS / NUM_SAMPLES);
    }
#else
    volatile int64_t val = 0;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        int64_t expected = i;
        val = i;
        __atomic_compare_exchange_n(&val, &expected, i + 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < BENCHMARK_ITERATIONS / NUM_SAMPLES; i++)
        {
            int64_t expected = i;
            val = i;
            __atomic_compare_exchange_n(&val, &expected, i + 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / (BENCHMARK_ITERATIONS / NUM_SAMPLES);
    }
#endif
}

/* ===================== Benchmark: Socket create/close ===================== */
void bench_socket_create_close(int64_t *samples)
{
#if NCCL_PLATFORM_WINDOWS
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    /* Warmup */
    for (int i = 0; i < 1000; i++)
    {
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        closesocket(s);
    }

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < 1000; i++)
        {
            SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            closesocket(sock);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / 1000;
    }

#if NCCL_PLATFORM_WINDOWS
    WSACleanup();
#endif
}

/* ===================== Benchmark: poll ===================== */
void bench_poll(int64_t *samples)
{
#if NCCL_PLATFORM_WINDOWS
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    struct pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLIN;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        poll(&pfd, 1, 0);
    }

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < 10000; i++)
        {
            poll(&pfd, 1, 0);
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / 10000;
    }

    closesocket(sock);

#if NCCL_PLATFORM_WINDOWS
    WSACleanup();
#endif
}

/* ===================== Benchmark: getpid ===================== */
void bench_getpid(int64_t *samples)
{
    volatile int pid;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        pid = getpid();
    }
    (void)pid;

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < BENCHMARK_ITERATIONS / NUM_SAMPLES; i++)
        {
            pid = getpid();
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / (BENCHMARK_ITERATIONS / NUM_SAMPLES);
    }
}

/* ===================== Benchmark: dlsym ===================== */
void bench_dlsym(int64_t *samples)
{
#if NCCL_PLATFORM_WINDOWS
    void *handle = ncclDlOpen("kernel32.dll", NCCL_RTLD_NOW);
    const char *symbol = "GetCurrentProcessId";
#else
    void *handle = dlopen("libc.so.6", RTLD_NOW);
    const char *symbol = "getpid";
#endif

    volatile void *sym;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
#if NCCL_PLATFORM_WINDOWS
        sym = ncclDlSym(handle, symbol);
#else
        sym = dlsym(handle, symbol);
#endif
    }
    (void)sym;

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < 10000; i++)
        {
#if NCCL_PLATFORM_WINDOWS
            sym = ncclDlSym(handle, symbol);
#else
            sym = dlsym(handle, symbol);
#endif
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / 10000;
    }

#if NCCL_PLATFORM_WINDOWS
    ncclDlClose(handle);
#else
    dlclose(handle);
#endif
}

/* ===================== Benchmark: getenv ===================== */
void bench_getenv(int64_t *samples)
{
#if NCCL_PLATFORM_WINDOWS
    _putenv("NCCL_BENCH_TEST=value123");
#else
    setenv("NCCL_BENCH_TEST", "value123", 1);
#endif

    volatile const char *val;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        val = getenv("NCCL_BENCH_TEST");
    }
    (void)val;

    /* Benchmark */
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        int64_t start = get_time_ns();
        for (int i = 0; i < 10000; i++)
        {
            val = getenv("NCCL_BENCH_TEST");
        }
        int64_t end = get_time_ns();
        samples[s] = (end - start) / 10000;
    }
}

/* ===================== Print comparison data ===================== */
void print_linux_comparison(void)
{
    printf("\n");
    printf("================================================================================\n");
    printf("EXPECTED LINUX PERFORMANCE (for comparison)\n");
    printf("================================================================================\n");
    printf("These are typical Linux performance numbers from similar hardware:\n\n");

    printf("%-40s %10s %10s\n", "Operation", "Linux(ns)", "Notes");
    printf("--------------------------------------------------------------------------------\n");
    printf("%-40s %10s %10s\n", "clock_gettime(CLOCK_MONOTONIC)", "20-30", "vDSO optimized");
    printf("%-40s %10s %10s\n", "pthread_mutex lock/unlock", "15-25", "futex-based");
    printf("%-40s %10s %10s\n", "pthread_create/join", "20-50us", "Kernel thread");
    printf("%-40s %10s %10s\n", "CPU_ZERO", "5-15", "memset");
    printf("%-40s %10s %10s\n", "CPU_COUNT", "50-100", "popcount");
    printf("%-40s %10s %10s\n", "sched_getaffinity", "200-500", "syscall");
    printf("%-40s %10s %10s\n", "atomic increment", "10-20", "lock xadd");
    printf("%-40s %10s %10s\n", "atomic CAS", "10-25", "lock cmpxchg");
    printf("%-40s %10s %10s\n", "socket create/close", "3-8us", "syscall pair");
    printf("%-40s %10s %10s\n", "poll (timeout=0)", "500-1000", "syscall");
    printf("%-40s %10s %10s\n", "getpid", "1-5", "vDSO cached");
    printf("%-40s %10s %10s\n", "dlsym", "50-100", "hash lookup");
    printf("%-40s %10s %10s\n", "getenv", "200-500", "linear search");
    printf("\n");
    printf("Note: Windows implementations may differ due to:\n");
    printf("  - Different kernel/API architecture (NT vs Linux kernel)\n");
    printf("  - CRITICAL_SECTION vs futex for mutexes\n");
    printf("  - QueryPerformanceCounter vs vDSO for time\n");
    printf("  - Winsock vs BSD sockets\n");
    printf("================================================================================\n");
}

/* ===================== Main ===================== */
int main()
{
    int64_t *samples = (int64_t *)malloc(NUM_SAMPLES * sizeof(int64_t));
    BenchStats stats;

    printf("================================================================================\n");
    printf("NCCL Platform Abstraction - Cross-Platform Performance Comparison\n");
    printf("================================================================================\n");
#if NCCL_PLATFORM_WINDOWS
    printf("Platform: Windows\n");
#else
    printf("Platform: Linux\n");
#endif
    printf("CPU cores: %d\n", ncclGetNumProcessors());
    printf("Iterations per benchmark: %d\n", BENCHMARK_ITERATIONS);
    printf("Samples for statistics: %d\n", NUM_SAMPLES);

    print_header("=== Time Functions ===");

    bench_clock_gettime(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("clock_gettime(CLOCK_MONOTONIC)", &stats);

    print_header("=== Threading ===");

    bench_mutex_lock_unlock(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("pthread_mutex lock/unlock (uncontended)", &stats);

    bench_thread_create_join(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("pthread_create + pthread_join", &stats);

    print_header("=== CPU Affinity ===");

    bench_cpu_zero(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("CPU_ZERO", &stats);

    bench_cpu_count(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("CPU_COUNT (64 CPUs set)", &stats);

    bench_sched_getaffinity(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("sched_getaffinity", &stats);

    print_header("=== Atomic Operations ===");

    bench_atomic_increment(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("atomic_increment_64", &stats);

    bench_atomic_cas(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("atomic_compare_exchange_64", &stats);

    print_header("=== Sockets ===");

    bench_socket_create_close(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("socket() + close()", &stats);

    bench_poll(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("poll (1 fd, timeout=0)", &stats);

    print_header("=== System Functions ===");

    bench_getpid(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("getpid", &stats);

    bench_dlsym(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("dlsym", &stats);

    bench_getenv(samples);
    calc_stats(samples, NUM_SAMPLES, &stats);
    print_result("getenv", &stats);

    /* Print Linux comparison data */
    print_linux_comparison();

    free(samples);

    printf("\nBenchmark complete.\n");
    return 0;
}
