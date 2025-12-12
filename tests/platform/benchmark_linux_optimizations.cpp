/*************************************************************************
 * NCCL Linux Platform Optimizations Benchmark
 *
 * Benchmarks the socket and shared memory optimizations for Linux.
 * Based on findings from "Demystifying NCCL" (arXiv:2507.04786v2)
 *
 * Compile:
 *   g++ -std=c++17 -O2 -pthread benchmark_linux_optimizations.cpp \
 *       -o benchmark_linux_optimizations -ldl -lrt -lnuma
 *
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <dlfcn.h>
#include <errno.h>

/* Number of iterations for benchmarks */
#define WARMUP_ITERATIONS 100
#define BENCHMARK_ITERATIONS 10000

/* Message sizes to test */
static const size_t message_sizes[] = {
    64,              /* Tiny message */
    1024,            /* 1 KB */
    8 * 1024,        /* 8 KB */
    64 * 1024,       /* 64 KB */
    256 * 1024,      /* 256 KB */
    512 * 1024,      /* 512 KB */
    1024 * 1024,     /* 1 MB */
    4 * 1024 * 1024, /* 4 MB */
};
static const int num_sizes = sizeof(message_sizes) / sizeof(message_sizes[0]);

/* High-resolution timer */
static inline int64_t get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline double get_time_us(void)
{
    return get_time_ns() / 1000.0;
}

/* Socket options */
#ifndef TCP_QUICKACK
#define TCP_QUICKACK 12
#endif

#ifndef SO_BUSY_POLL
#define SO_BUSY_POLL 46
#endif

/* ========================================================================== */
/*                    Socket Optimization Functions                           */
/* ========================================================================== */

static inline int socket_optimize_default(int sock)
{
    /* Just set TCP_NODELAY */
    int optval = 1;
    return setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval));
}

static inline int socket_optimize_throughput(int sock)
{
    int result = 0;
    int optval;

    /* TCP_NODELAY */
    optval = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)) != 0)
        result = -1;

    /* Large buffers (4MB) */
    optval = 4 * 1024 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &optval, sizeof(optval));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &optval, sizeof(optval));

    /* TCP_QUICKACK */
    optval = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_QUICKACK, &optval, sizeof(optval));

    return result;
}

static inline int socket_optimize_low_latency(int sock)
{
    int result = 0;
    int optval;

    /* TCP_NODELAY */
    optval = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)) != 0)
        result = -1;

    /* Smaller buffers (256KB) */
    optval = 256 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &optval, sizeof(optval));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &optval, sizeof(optval));

    /* TCP_QUICKACK */
    optval = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_QUICKACK, &optval, sizeof(optval));

    /* Busy polling */
    optval = 50;
    setsockopt(sock, SOL_SOCKET, SO_BUSY_POLL, &optval, sizeof(optval));

    return result;
}

/* ========================================================================== */
/*                    SOCKET CONFIGURATION BENCHMARK                          */
/* ========================================================================== */

void benchmark_socket_configuration(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Socket Configuration Benchmark\n");
    printf("============================================================\n");

    double time_default = 0, time_throughput = 0, time_lowlatency = 0;

    /* Benchmark default socket creation */
    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        socket_optimize_default(sock);
        close(sock);
    }
    time_default = get_time_us() - start;

    /* Benchmark throughput-optimized socket */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        socket_optimize_throughput(sock);
        close(sock);
    }
    time_throughput = get_time_us() - start;

    /* Benchmark low-latency socket */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        socket_optimize_low_latency(sock);
        close(sock);
    }
    time_lowlatency = get_time_us() - start;

    printf("\nSocket Configuration (%d iterations):\n", BENCHMARK_ITERATIONS);
    printf("  Default (nodelay only): %8.2f us/op\n", time_default / BENCHMARK_ITERATIONS);
    printf("  Throughput (4MB buf):   %8.2f us/op (+%.1f%%)\n",
           time_throughput / BENCHMARK_ITERATIONS,
           (time_throughput - time_default) / time_default * 100);
    printf("  Low-latency (256KB):    %8.2f us/op (+%.1f%%)\n",
           time_lowlatency / BENCHMARK_ITERATIONS,
           (time_lowlatency - time_default) / time_default * 100);
}

/* ========================================================================== */
/*                    LOOPBACK THROUGHPUT BENCHMARK                           */
/* ========================================================================== */

struct thread_args
{
    int port;
    size_t msg_size;
    int iterations;
    int use_optimized;
    double throughput;
};

void *server_thread(void *arg)
{
    struct thread_args *args = (struct thread_args *)arg;
    char *buffer = (char *)malloc(args->msg_size);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int optval = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(args->port);

    bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(server_fd, 1);

    int client_fd = accept(server_fd, NULL, NULL);

    if (args->use_optimized)
        socket_optimize_throughput(client_fd);
    else
        socket_optimize_default(client_fd);

    /* Receive data */
    for (int i = 0; i < args->iterations; i++)
    {
        size_t total = 0;
        while (total < args->msg_size)
        {
            ssize_t n = recv(client_fd, buffer + total, args->msg_size - total, 0);
            if (n <= 0)
                break;
            total += n;
        }
    }

    close(client_fd);
    close(server_fd);
    free(buffer);
    return NULL;
}

void benchmark_loopback_throughput(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Loopback Socket Throughput Benchmark\n");
    printf("============================================================\n");

    printf("\nSize              Default    Optimized  Improvement\n");
    printf("----              -------    ---------  -----------\n");

    for (int s = 4; s < num_sizes; s++) /* Start from 64KB */
    {
        size_t size = message_sizes[s];
        int iterations = (size >= 1024 * 1024) ? 100 : 1000;
        int port_default = 19000 + s * 2;
        int port_optimized = 19000 + s * 2 + 1;

        char *buffer = (char *)malloc(size);
        memset(buffer, 'X', size);

        /* Test default configuration */
        struct thread_args args_default = {port_default, size, iterations, 0, 0};
        pthread_t server_default;
        pthread_create(&server_default, NULL, server_thread, &args_default);
        usleep(10000); /* Let server start */

        int client_default = socket(AF_INET, SOCK_STREAM, 0);
        socket_optimize_default(client_default);

        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(port_default);
        connect(client_default, (struct sockaddr *)&addr, sizeof(addr));

        int64_t start = get_time_ns();
        for (int i = 0; i < iterations; i++)
        {
            size_t total = 0;
            while (total < size)
            {
                ssize_t n = send(client_default, buffer + total, size - total, 0);
                if (n <= 0)
                    break;
                total += n;
            }
        }
        int64_t time_default = get_time_ns() - start;

        close(client_default);
        pthread_join(server_default, NULL);

        /* Test optimized configuration */
        struct thread_args args_optimized = {port_optimized, size, iterations, 1, 0};
        pthread_t server_optimized;
        pthread_create(&server_optimized, NULL, server_thread, &args_optimized);
        usleep(10000);

        int client_optimized = socket(AF_INET, SOCK_STREAM, 0);
        socket_optimize_throughput(client_optimized);

        addr.sin_port = htons(port_optimized);
        connect(client_optimized, (struct sockaddr *)&addr, sizeof(addr));

        start = get_time_ns();
        for (int i = 0; i < iterations; i++)
        {
            size_t total = 0;
            while (total < size)
            {
                ssize_t n = send(client_optimized, buffer + total, size - total, 0);
                if (n <= 0)
                    break;
                total += n;
            }
        }
        int64_t time_optimized = get_time_ns() - start;

        close(client_optimized);
        pthread_join(server_optimized, NULL);

        /* Calculate throughput */
        double bytes_total = (double)size * iterations;
        double throughput_default = bytes_total / (time_default / 1e9) / (1024 * 1024);
        double throughput_optimized = bytes_total / (time_optimized / 1e9) / (1024 * 1024);
        double improvement = (throughput_optimized - throughput_default) / throughput_default * 100;

        if (size >= 1024 * 1024)
            printf("%4zu MB     %8.1f MB/s   %8.1f MB/s    %+.1f%%\n",
                   size / (1024 * 1024), throughput_default, throughput_optimized, improvement);
        else
            printf("%4zu KB     %8.1f MB/s   %8.1f MB/s    %+.1f%%\n",
                   size / 1024, throughput_default, throughput_optimized, improvement);

        free(buffer);
    }
}

/* ========================================================================== */
/*                    SHARED MEMORY BENCHMARK                                 */
/* ========================================================================== */

void benchmark_shared_memory(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Shared Memory Allocation Benchmark\n");
    printf("============================================================\n");

    const int iterations = 1000;

    for (int s = 0; s < num_sizes; s++)
    {
        size_t size = message_sizes[s];
        double time_basic = 0, time_prefault = 0;
        char name[64];

        int iter = (size > 1024 * 1024) ? 100 : iterations;

        /* Benchmark basic shared memory */
        double start = get_time_us();
        for (int i = 0; i < iter; i++)
        {
            snprintf(name, sizeof(name), "/bench_basic_%d_%d", (int)size, i);
            int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
            if (fd >= 0)
            {
                ftruncate(fd, size);
                void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (ptr != MAP_FAILED)
                    munmap(ptr, size);
                close(fd);
                shm_unlink(name);
            }
        }
        time_basic = get_time_us() - start;

        /* Benchmark prefaulted shared memory */
        start = get_time_us();
        for (int i = 0; i < iter; i++)
        {
            snprintf(name, sizeof(name), "/bench_pf_%d_%d", (int)size, i);
            int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
            if (fd >= 0)
            {
                ftruncate(fd, size);
                void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
                if (ptr != MAP_FAILED)
                    munmap(ptr, size);
                close(fd);
                shm_unlink(name);
            }
        }
        time_prefault = get_time_us() - start;

        if (size >= 1024 * 1024)
            printf("  %4zu MB:  Basic: %8.2f us/op  Prefault: %8.2f us/op\n",
                   size / (1024 * 1024), time_basic / iter, time_prefault / iter);
        else if (size >= 1024)
            printf("  %4zu KB:  Basic: %8.2f us/op  Prefault: %8.2f us/op\n",
                   size / 1024, time_basic / iter, time_prefault / iter);
        else
            printf("  %4zu B:   Basic: %8.2f us/op  Prefault: %8.2f us/op\n",
                   size, time_basic / iter, time_prefault / iter);
    }
}

/* ========================================================================== */
/*                    MEMORY ACCESS BENCHMARK                                 */
/* ========================================================================== */

void benchmark_memory_access(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Memory Access Pattern Benchmark\n");
    printf("============================================================\n");

    const size_t size = 4 * 1024 * 1024; /* 4 MB */
    const int iterations = 100;

    /* Create basic shared memory */
    int fd = shm_open("/bench_mem_basic", O_CREAT | O_RDWR, 0600);
    ftruncate(fd, size);
    void *ptr_basic = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    /* Create prefaulted shared memory */
    fd = shm_open("/bench_mem_pf", O_CREAT | O_RDWR, 0600);
    ftruncate(fd, size);
    void *ptr_pf = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    close(fd);

    if (ptr_basic == MAP_FAILED || ptr_pf == MAP_FAILED)
    {
        printf("  Failed to create shared memory\n");
        return;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++)
    {
        memset(ptr_basic, i, size);
        memset(ptr_pf, i, size);
    }

    /* Benchmark sequential write */
    double start = get_time_us();
    for (int i = 0; i < iterations; i++)
        memset(ptr_basic, i & 0xFF, size);
    double time_basic_write = get_time_us() - start;

    start = get_time_us();
    for (int i = 0; i < iterations; i++)
        memset(ptr_pf, i & 0xFF, size);
    double time_pf_write = get_time_us() - start;

    /* Benchmark sequential read */
    volatile uint64_t sum = 0;
    start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        uint64_t *p = (uint64_t *)ptr_basic;
        for (size_t j = 0; j < size / sizeof(uint64_t); j++)
            sum += p[j];
    }
    double time_basic_read = get_time_us() - start;

    start = get_time_us();
    for (int i = 0; i < iterations; i++)
    {
        uint64_t *p = (uint64_t *)ptr_pf;
        for (size_t j = 0; j < size / sizeof(uint64_t); j++)
            sum += p[j];
    }
    double time_pf_read = get_time_us() - start;
    (void)sum;

    /* Calculate bandwidth */
    double bytes = (double)size * iterations;
    double bw_basic_write = bytes / (time_basic_write / 1e6) / (1024 * 1024 * 1024);
    double bw_pf_write = bytes / (time_pf_write / 1e6) / (1024 * 1024 * 1024);
    double bw_basic_read = bytes / (time_basic_read / 1e6) / (1024 * 1024 * 1024);
    double bw_pf_read = bytes / (time_pf_read / 1e6) / (1024 * 1024 * 1024);

    printf("\nMemory Access (4 MB buffer, %d iterations):\n", iterations);
    printf("  Sequential Write:\n");
    printf("    Basic:       %8.2f GB/s\n", bw_basic_write);
    printf("    Prefault:    %8.2f GB/s (%.1f%% vs basic)\n",
           bw_pf_write, (bw_pf_write - bw_basic_write) / bw_basic_write * 100);
    printf("  Sequential Read:\n");
    printf("    Basic:       %8.2f GB/s\n", bw_basic_read);
    printf("    Prefault:    %8.2f GB/s (%.1f%% vs basic)\n",
           bw_pf_read, (bw_pf_read - bw_basic_read) / bw_basic_read * 100);

    munmap(ptr_basic, size);
    munmap(ptr_pf, size);
    shm_unlink("/bench_mem_basic");
    shm_unlink("/bench_mem_pf");
}

/* ========================================================================== */
/*                    THREAD PRIORITY BENCHMARK                               */
/* ========================================================================== */

void benchmark_thread_priority(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Thread Priority Benchmark\n");
    printf("============================================================\n");

    double time_getpriority = 0, time_setnice = 0;

    /* Benchmark getpriority */
    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        volatile int prio = getpriority(PRIO_PROCESS, 0);
        (void)prio;
    }
    time_getpriority = get_time_us() - start;

    /* Benchmark setpriority (nice) */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        setpriority(PRIO_PROCESS, 0, 0);
    }
    time_setnice = get_time_us() - start;

    printf("\nThread Priority Operations (%d iterations):\n", BENCHMARK_ITERATIONS);
    printf("  getpriority:          %8.3f us/op\n", time_getpriority / BENCHMARK_ITERATIONS);
    printf("  setpriority (nice):   %8.3f us/op\n", time_setnice / BENCHMARK_ITERATIONS);
}

/* ========================================================================== */
/*                    SPINLOCK BENCHMARK                                      */
/* ========================================================================== */

typedef volatile int spinlock_t;

static inline void spinlock_init(spinlock_t *lock) { *lock = 0; }
static inline void spinlock_lock(spinlock_t *lock)
{
    while (__sync_lock_test_and_set(lock, 1))
        while (*lock)
            __builtin_ia32_pause();
}
static inline void spinlock_unlock(spinlock_t *lock) { __sync_lock_release(lock); }

void benchmark_spinlock(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Spinlock Benchmark\n");
    printf("============================================================\n");

    spinlock_t spinlock;
    spinlock_init(&spinlock);

    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    /* Benchmark spinlock */
    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        spinlock_lock(&spinlock);
        spinlock_unlock(&spinlock);
    }
    double time_spinlock = get_time_us() - start;

    /* Benchmark mutex */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        pthread_mutex_lock(&mutex);
        pthread_mutex_unlock(&mutex);
    }
    double time_mutex = get_time_us() - start;

    printf("\nLock Operations (%d iterations, uncontended):\n", BENCHMARK_ITERATIONS);
    printf("  Spinlock:              %8.3f ns/op\n", time_spinlock / BENCHMARK_ITERATIONS * 1000);
    printf("  Mutex (pthread):       %8.3f ns/op\n", time_mutex / BENCHMARK_ITERATIONS * 1000);

    pthread_mutex_destroy(&mutex);
}

/* ========================================================================== */
/*                    ATOMIC OPERATIONS BENCHMARK                             */
/* ========================================================================== */

void benchmark_atomics(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Atomic Operations Benchmark\n");
    printf("============================================================\n");

    volatile int64_t counter = 0;

    /* Memory fence */
    double start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
        __sync_synchronize();
    double time_fence = get_time_us() - start;

    /* Atomic load */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        volatile int64_t val = __atomic_load_n(&counter, __ATOMIC_ACQUIRE);
        (void)val;
    }
    double time_load = get_time_us() - start;

    /* Atomic store */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
        __atomic_store_n(&counter, i, __ATOMIC_RELEASE);
    double time_store = get_time_us() - start;

    /* Atomic add */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
        __sync_fetch_and_add(&counter, 1);
    double time_add = get_time_us() - start;

    /* Atomic CAS */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
        __sync_bool_compare_and_swap(&counter, counter, counter + 1);
    double time_cas = get_time_us() - start;

    /* CPU pause */
    start = get_time_us();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
        __builtin_ia32_pause();
    double time_pause = get_time_us() - start;

    printf("\nMemory/Atomic Operations (%d iterations):\n", BENCHMARK_ITERATIONS);
    printf("  Memory fence:          %8.3f ns/op\n", time_fence / BENCHMARK_ITERATIONS * 1000);
    printf("  Atomic load (acquire): %8.3f ns/op\n", time_load / BENCHMARK_ITERATIONS * 1000);
    printf("  Atomic store (release):%8.3f ns/op\n", time_store / BENCHMARK_ITERATIONS * 1000);
    printf("  Atomic add:            %8.3f ns/op\n", time_add / BENCHMARK_ITERATIONS * 1000);
    printf("  Atomic CAS:            %8.3f ns/op\n", time_cas / BENCHMARK_ITERATIONS * 1000);
    printf("  CPU pause:             %8.3f ns/op\n", time_pause / BENCHMARK_ITERATIONS * 1000);
}

/* ========================================================================== */
/*                    SLEEP ACCURACY BENCHMARK                                */
/* ========================================================================== */

void benchmark_sleep(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("Sleep Accuracy Benchmark\n");
    printf("============================================================\n");

    printf("\nSleep Accuracy (target vs actual):\n");
    printf("Target           usleep      nanosleep    clock_nanosleep\n");

    int targets[] = {100, 500, 1000, 5000, 10000};

    for (int t = 0; t < 5; t++)
    {
        int target_ns = targets[t];

        /* usleep */
        int64_t start = get_time_ns();
        usleep(target_ns / 1000);
        int64_t actual_usleep = get_time_ns() - start;

        /* nanosleep */
        struct timespec req = {0, target_ns};
        start = get_time_ns();
        nanosleep(&req, NULL);
        int64_t actual_nanosleep = get_time_ns() - start;

        /* clock_nanosleep */
        start = get_time_ns();
        clock_nanosleep(CLOCK_MONOTONIC, 0, &req, NULL);
        int64_t actual_clock = get_time_ns() - start;

        printf("%6d ns    %8lld ns  %8lld ns     %8lld ns\n",
               target_ns, (long long)actual_usleep, (long long)actual_nanosleep, (long long)actual_clock);
    }
}

/* ========================================================================== */
/*                    MAIN                                                    */
/* ========================================================================== */

int main(void)
{
    printf("================================================================\n");
    printf("NCCL Linux Platform Optimizations Benchmark\n");
    printf("Based on 'Demystifying NCCL' (arXiv:2507.04786v2)\n");
    printf("================================================================\n");

    printf("\nSystem Information:\n");
    printf("  CPU cores:          %d\n", (int)sysconf(_SC_NPROCESSORS_ONLN));
    printf("  Page size:          %ld bytes\n", sysconf(_SC_PAGESIZE));

    benchmark_socket_configuration();
    benchmark_loopback_throughput();
    benchmark_shared_memory();
    benchmark_memory_access();
    benchmark_thread_priority();
    benchmark_spinlock();
    benchmark_atomics();
    benchmark_sleep();

    printf("\n================================================================\n");
    printf("Benchmark Complete\n");
    printf("================================================================\n");

    return 0;
}
