/*
 * Linux Platform Benchmark
 * Compile: g++ -std=c++17 -O2 -pthread linux_bench.cpp -o linux_bench -ldl
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <poll.h>
#include <sys/syscall.h>

static inline int64_t get_time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline pid_t gettid_syscall()
{
    return syscall(SYS_gettid);
}

const int WARMUP = 1000;
const int ITERATIONS = 100000;

void bench_clock_gettime()
{
    struct timespec ts;
    int64_t start, end;

    for (int i = 0; i < WARMUP; i++)
        clock_gettime(CLOCK_MONOTONIC, &ts);
    start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++)
        clock_gettime(CLOCK_MONOTONIC, &ts);
    end = get_time_ns();

    double avg_ns = (double)(end - start) / ITERATIONS;
    printf("  clock_gettime(MONOTONIC)     avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);
}

void bench_mutex()
{
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);
    int64_t start, end;

    for (int i = 0; i < WARMUP; i++)
    {
        pthread_mutex_lock(&mutex);
        pthread_mutex_unlock(&mutex);
    }
    start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++)
    {
        pthread_mutex_lock(&mutex);
        pthread_mutex_unlock(&mutex);
    }
    end = get_time_ns();

    double avg_ns = (double)(end - start) / ITERATIONS;
    printf("  mutex lock/unlock            avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);
    pthread_mutex_destroy(&mutex);
}

void bench_cpu_affinity()
{
    cpu_set_t cpuset;
    int64_t start, end;

    for (int i = 0; i < WARMUP; i++)
        CPU_ZERO(&cpuset);
    start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++)
        CPU_ZERO(&cpuset);
    end = get_time_ns();
    double avg_ns = (double)(end - start) / ITERATIONS;
    printf("  CPU_ZERO                     avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);

    for (int i = 0; i < WARMUP; i++)
        sched_getaffinity(0, sizeof(cpuset), &cpuset);
    start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++)
        sched_getaffinity(0, sizeof(cpuset), &cpuset);
    end = get_time_ns();
    avg_ns = (double)(end - start) / ITERATIONS;
    printf("  sched_getaffinity            avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);
}

void bench_atomics()
{
    volatile int64_t counter = 0;
    int64_t start, end;

    for (int i = 0; i < WARMUP; i++)
        __sync_fetch_and_add(&counter, 1);
    start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++)
        __sync_fetch_and_add(&counter, 1);
    end = get_time_ns();
    double avg_ns = (double)(end - start) / ITERATIONS;
    printf("  atomic_add                   avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);

    counter = 0;
    for (int i = 0; i < WARMUP; i++)
        __sync_bool_compare_and_swap(&counter, counter, counter + 1);
    start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++)
        __sync_bool_compare_and_swap(&counter, counter, counter + 1);
    end = get_time_ns();
    avg_ns = (double)(end - start) / ITERATIONS;
    printf("  atomic_cas                   avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);
}

void bench_socket()
{
    int64_t start, end;
    int SOCKET_ITERATIONS = 10000;

    for (int i = 0; i < 100; i++)
    {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        close(s);
    }
    start = get_time_ns();
    for (int i = 0; i < SOCKET_ITERATIONS; i++)
    {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        close(s);
    }
    end = get_time_ns();
    double avg_ns = (double)(end - start) / SOCKET_ITERATIONS;
    printf("  socket create/close          avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct pollfd pfd = {sock, POLLIN, 0};
    for (int i = 0; i < WARMUP; i++)
        poll(&pfd, 1, 0);
    start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++)
        poll(&pfd, 1, 0);
    end = get_time_ns();
    avg_ns = (double)(end - start) / ITERATIONS;
    printf("  poll (1 fd, timeout=0)       avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);
    close(sock);
}

void bench_process()
{
    int64_t start, end;
    volatile int pid;

    for (int i = 0; i < WARMUP; i++)
        pid = getpid();
    start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++)
        pid = getpid();
    end = get_time_ns();
    double avg_ns = (double)(end - start) / ITERATIONS;
    printf("  getpid                       avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);

    for (int i = 0; i < WARMUP; i++)
        pid = gettid_syscall();
    start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++)
        pid = gettid_syscall();
    end = get_time_ns();
    avg_ns = (double)(end - start) / ITERATIONS;
    printf("  gettid                       avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);
    (void)pid;
}

void bench_dlsym()
{
    void *handle = dlopen(NULL, RTLD_NOW);
    void *sym;
    int64_t start, end;

    for (int i = 0; i < WARMUP; i++)
        sym = dlsym(handle, "printf");
    start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++)
        sym = dlsym(handle, "printf");
    end = get_time_ns();
    double avg_ns = (double)(end - start) / ITERATIONS;
    printf("  dlsym                        avg: %8.1f ns  (%.2fM ops/sec)\n", avg_ns, 1000.0 / avg_ns);
    (void)sym;
    dlclose(handle);
}

int main()
{
    printf("========================================\n");
    printf("Linux Platform Benchmark\n");
    printf("========================================\n");
    printf("Iterations: %d (warmup: %d)\n", ITERATIONS, WARMUP);
    printf("CPU: %d processors\n\n", (int)sysconf(_SC_NPROCESSORS_ONLN));

    printf("=== Time Functions ===\n");
    bench_clock_gettime();

    printf("\n=== Mutex Operations ===\n");
    bench_mutex();

    printf("\n=== CPU Affinity ===\n");
    bench_cpu_affinity();

    printf("\n=== Atomic Operations ===\n");
    bench_atomics();

    printf("\n=== Socket Operations ===\n");
    bench_socket();

    printf("\n=== Process Info ===\n");
    bench_process();

    printf("\n=== Dynamic Loading ===\n");
    bench_dlsym();

    printf("\n========================================\n");
    printf("Benchmark Complete\n");
    printf("========================================\n");
    return 0;
}
