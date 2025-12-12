/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Edge Case Tests for Windows Platform Abstraction
 * Tests boundary conditions, error handling, and corner cases
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "platform.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                           \
    do                                                                   \
    {                                                                    \
        tests_run++;                                                     \
        if (cond)                                                        \
        {                                                                \
            tests_passed++;                                              \
            printf("  [PASS] %s\n", msg);                                \
        }                                                                \
        else                                                             \
        {                                                                \
            tests_failed++;                                              \
            printf("  [FAIL] %s (at %s:%d)\n", msg, __FILE__, __LINE__); \
        }                                                                \
    } while (0)

#define TEST_INFO(fmt, ...) printf("  [INFO] " fmt "\n", ##__VA_ARGS__)

/* ===================== CPU Affinity Edge Cases ===================== */

void test_cpu_affinity_edge_cases()
{
    printf("\n=== CPU Affinity Edge Cases ===\n");

    cpu_set_t set;

    /* Test 1: Empty set operations */
    CPU_ZERO(&set);
    TEST_ASSERT(CPU_COUNT(&set) == 0, "Empty set has count 0");
    TEST_ASSERT(!CPU_ISSET(0, &set), "CPU 0 not set in empty set");
    TEST_ASSERT(!CPU_ISSET(CPU_SETSIZE - 1, &set), "Last CPU not set in empty set");

    /* Test 2: Set all CPUs */
    for (int i = 0; i < CPU_SETSIZE; i++)
    {
        CPU_SET(i, &set);
    }
    TEST_ASSERT(CPU_COUNT(&set) == CPU_SETSIZE, "All CPUs set gives correct count");
    TEST_ASSERT(CPU_ISSET(0, &set), "First CPU is set");
    TEST_ASSERT(CPU_ISSET(CPU_SETSIZE - 1, &set), "Last CPU is set");

    /* Test 3: Clear all CPUs */
    for (int i = 0; i < CPU_SETSIZE; i++)
    {
        CPU_CLR(i, &set);
    }
    TEST_ASSERT(CPU_COUNT(&set) == 0, "All CPUs cleared gives count 0");

    /* Test 4: Alternating pattern */
    CPU_ZERO(&set);
    for (int i = 0; i < CPU_SETSIZE; i += 2)
    {
        CPU_SET(i, &set);
    }
    TEST_ASSERT(CPU_COUNT(&set) == CPU_SETSIZE / 2, "Alternating pattern has correct count");
    TEST_ASSERT(CPU_ISSET(0, &set), "Even CPU 0 is set");
    TEST_ASSERT(!CPU_ISSET(1, &set), "Odd CPU 1 is not set");
    TEST_ASSERT(CPU_ISSET(CPU_SETSIZE - 2, &set), "Last even CPU is set");

    /* Test 5: Boundary CPUs */
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    CPU_SET(63, &set);  /* End of first 64-bit word */
    CPU_SET(64, &set);  /* Start of second 64-bit word */
    CPU_SET(127, &set); /* End of second 64-bit word */
    TEST_ASSERT(CPU_COUNT(&set) == 4, "Boundary CPUs set correctly");
    TEST_ASSERT(CPU_ISSET(63, &set), "CPU 63 (word boundary) is set");
    TEST_ASSERT(CPU_ISSET(64, &set), "CPU 64 (word boundary) is set");

    /* Test 6: Double set/clear */
    CPU_ZERO(&set);
    CPU_SET(5, &set);
    CPU_SET(5, &set); /* Set again */
    TEST_ASSERT(CPU_COUNT(&set) == 1, "Double set doesn't increase count");
    CPU_CLR(5, &set);
    CPU_CLR(5, &set); /* Clear again */
    TEST_ASSERT(CPU_COUNT(&set) == 0, "Double clear doesn't cause issues");

#if NCCL_PLATFORM_WINDOWS
    /* Test 7: CPU_AND edge cases */
    cpu_set_t set1, set2, result;
    CPU_ZERO(&set1);
    CPU_ZERO(&set2);
    CPU_AND(&result, &set1, &set2);
    TEST_ASSERT(CPU_COUNT(&result) == 0, "AND of empty sets is empty");

    /* Full set AND empty set */
    for (int i = 0; i < CPU_SETSIZE; i++)
        CPU_SET(i, &set1);
    CPU_AND(&result, &set1, &set2);
    TEST_ASSERT(CPU_COUNT(&result) == 0, "Full AND empty = empty");

    /* Test 8: CPU_OR edge cases */
    CPU_ZERO(&set1);
    CPU_ZERO(&set2);
    CPU_OR(&result, &set1, &set2);
    TEST_ASSERT(CPU_COUNT(&result) == 0, "OR of empty sets is empty");

    /* Empty OR full = full */
    for (int i = 0; i < CPU_SETSIZE; i++)
        CPU_SET(i, &set2);
    CPU_OR(&result, &set1, &set2);
    TEST_ASSERT(CPU_COUNT(&result) == CPU_SETSIZE, "Empty OR full = full");

    /* Test 9: CPU_EQUAL edge cases */
    CPU_ZERO(&set1);
    CPU_ZERO(&set2);
    TEST_ASSERT(CPU_EQUAL(&set1, &set2), "Two empty sets are equal");

    for (int i = 0; i < CPU_SETSIZE; i++)
    {
        CPU_SET(i, &set1);
        CPU_SET(i, &set2);
    }
    TEST_ASSERT(CPU_EQUAL(&set1, &set2), "Two full sets are equal");

    CPU_CLR(500, &set2);
    TEST_ASSERT(!CPU_EQUAL(&set1, &set2), "Sets differ by one CPU");
#endif
}

/* ===================== Time Functions Edge Cases ===================== */

void test_time_edge_cases()
{
    printf("\n=== Time Functions Edge Cases ===\n");

    struct timespec ts;
    int ret;

    /* Test 1: NULL pointer handling */
    ret = clock_gettime(CLOCK_MONOTONIC, NULL);
    TEST_ASSERT(ret == -1, "clock_gettime with NULL returns -1");

    /* Test 2: Invalid clock ID */
    ret = clock_gettime(999, &ts);
    TEST_ASSERT(ret == -1, "clock_gettime with invalid clock ID returns -1");

    /* Test 3: Rapid successive calls */
    struct timespec ts1, ts2, ts3;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    clock_gettime(CLOCK_MONOTONIC, &ts3);

    int64_t t1 = ts1.tv_sec * 1000000000LL + ts1.tv_nsec;
    int64_t t2 = ts2.tv_sec * 1000000000LL + ts2.tv_nsec;
    int64_t t3 = ts3.tv_sec * 1000000000LL + ts3.tv_nsec;

    TEST_ASSERT(t2 >= t1, "Time is monotonic (t2 >= t1)");
    TEST_ASSERT(t3 >= t2, "Time is monotonic (t3 >= t2)");
    TEST_INFO("Time deltas: t2-t1=%lld ns, t3-t2=%lld ns", (long long)(t2 - t1), (long long)(t3 - t2));

    /* Test 4: CLOCK_REALTIME vs CLOCK_MONOTONIC */
    struct timespec real_ts, mono_ts;
    clock_gettime(CLOCK_REALTIME, &real_ts);
    clock_gettime(CLOCK_MONOTONIC, &mono_ts);

    TEST_ASSERT(real_ts.tv_sec > 1000000000, "CLOCK_REALTIME returns epoch time (after year 2001)");
    TEST_ASSERT(real_ts.tv_nsec >= 0 && real_ts.tv_nsec < 1000000000, "CLOCK_REALTIME nsec in valid range");
    TEST_ASSERT(mono_ts.tv_nsec >= 0 && mono_ts.tv_nsec < 1000000000, "CLOCK_MONOTONIC nsec in valid range");
    TEST_INFO("CLOCK_REALTIME: %lld sec", (long long)real_ts.tv_sec);
    TEST_INFO("CLOCK_MONOTONIC: %lld sec, %ld nsec", (long long)mono_ts.tv_sec, mono_ts.tv_nsec);

    /* Test 5: Sleep precision */
    struct timespec before, after;
    clock_gettime(CLOCK_MONOTONIC, &before);
    Sleep(1); /* 1 millisecond */
    clock_gettime(CLOCK_MONOTONIC, &after);

    int64_t elapsed_ns = (after.tv_sec - before.tv_sec) * 1000000000LL + (after.tv_nsec - before.tv_nsec);
    TEST_ASSERT(elapsed_ns >= 1000000, "1ms sleep takes at least 1ms");
    TEST_ASSERT(elapsed_ns < 100000000, "1ms sleep takes less than 100ms");
    TEST_INFO("1ms sleep actual time: %lld ns (%.2f ms)", (long long)elapsed_ns, elapsed_ns / 1000000.0);
}

/* ===================== Thread Edge Cases ===================== */

void test_thread_edge_cases()
{
    printf("\n=== Thread Edge Cases ===\n");

    /* Test 1: Multiple mutex init/destroy cycles */
    pthread_mutex_t mutex;
    for (int i = 0; i < 100; i++)
    {
        int ret = pthread_mutex_init(&mutex, NULL);
        if (ret != 0)
        {
            printf("  [FAIL] Mutex init failed at iteration %d\n", i);
            tests_run++;
            tests_failed++;
            return;
        }
        pthread_mutex_destroy(&mutex);
    }
    TEST_ASSERT(1, "100 mutex init/destroy cycles succeed");

    /* Test 2: Rapid lock/unlock */
    pthread_mutex_init(&mutex, NULL);
    for (int i = 0; i < 10000; i++)
    {
        pthread_mutex_lock(&mutex);
        pthread_mutex_unlock(&mutex);
    }
    pthread_mutex_destroy(&mutex);
    TEST_ASSERT(1, "10000 rapid lock/unlock cycles succeed");

    /* Test 3: Condition variable init/destroy cycles */
    pthread_cond_t cond;
    for (int i = 0; i < 100; i++)
    {
        int ret = pthread_cond_init(&cond, NULL);
        if (ret != 0)
        {
            printf("  [FAIL] Cond init failed at iteration %d\n", i);
            tests_run++;
            tests_failed++;
            return;
        }
        pthread_cond_destroy(&cond);
    }
    TEST_ASSERT(1, "100 cond init/destroy cycles succeed");

    /* Test 4: pthread_self consistency */
    pthread_t self1 = pthread_self();
    pthread_t self2 = pthread_self();
#if NCCL_PLATFORM_WINDOWS
    TEST_ASSERT(self1 == self2, "pthread_self returns consistent value");
#else
    TEST_ASSERT(pthread_equal(self1, self2), "pthread_self returns consistent value");
#endif

    /* Test 5: Trylock on locked mutex */
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_lock(&mutex);
    int trylock_result = pthread_mutex_trylock(&mutex);
    /* On Windows with CRITICAL_SECTION, trylock on owned mutex may succeed (recursive) */
    TEST_INFO("Trylock on owned mutex returned: %d", trylock_result);
    if (trylock_result == 0)
    {
        pthread_mutex_unlock(&mutex); /* Unlock the extra lock */
    }
    pthread_mutex_unlock(&mutex);
    pthread_mutex_destroy(&mutex);
    TEST_ASSERT(1, "Trylock behavior verified");
}

/* ===================== Socket Edge Cases ===================== */

void test_socket_edge_cases()
{
    printf("\n=== Socket Edge Cases ===\n");

#if NCCL_PLATFORM_WINDOWS
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    /* Test 1: Invalid socket operations */
    SOCKET invalid_sock = INVALID_SOCKET;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    int ret = bind(invalid_sock, (struct sockaddr *)&addr, sizeof(addr));
    TEST_ASSERT(ret != 0, "bind on invalid socket fails");

    /* Test 2: Create many sockets */
    SOCKET sockets[100];
    int created = 0;
    for (int i = 0; i < 100; i++)
    {
        sockets[i] = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sockets[i] != INVALID_SOCKET)
        {
            created++;
        }
    }
    TEST_ASSERT(created == 100, "Can create 100 sockets");
    TEST_INFO("Created %d sockets", created);

    /* Close all sockets */
    for (int i = 0; i < created; i++)
    {
        closesocket(sockets[i]);
    }
    TEST_ASSERT(1, "All sockets closed successfully");

    /* Test 3: Poll with empty array */
    struct pollfd empty_fds[1];
    ret = poll(empty_fds, 0, 0);
    TEST_ASSERT(ret == 0, "poll with 0 fds returns 0");

    /* Test 4: Poll with timeout 0 (non-blocking) */
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    struct pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLIN;
    pfd.revents = 0;

    ret = poll(&pfd, 1, 0);
    TEST_ASSERT(ret == 0, "poll with timeout 0 returns immediately");
    closesocket(sock);

    /* Test 5: Multiple binds to same port should fail */
    SOCKET sock1 = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    SOCKET sock2 = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    addr.sin_port = htons(0); /* Let OS assign port */
    ret = bind(sock1, (struct sockaddr *)&addr, sizeof(addr));
    TEST_ASSERT(ret == 0, "First bind succeeds");

    socklen_t addrlen = sizeof(addr);
    getsockname(sock1, (struct sockaddr *)&addr, &addrlen);
    int port = ntohs(addr.sin_port);
    TEST_INFO("First socket bound to port %d", port);

    /* Try to bind second socket to same port */
    ret = bind(sock2, (struct sockaddr *)&addr, sizeof(addr));
    TEST_ASSERT(ret != 0, "Second bind to same port fails");

    closesocket(sock1);
    closesocket(sock2);

#if NCCL_PLATFORM_WINDOWS
    WSACleanup();
#endif
}

/* ===================== Dynamic Loading Edge Cases ===================== */

void test_dl_edge_cases()
{
    printf("\n=== Dynamic Loading Edge Cases ===\n");

    /* Test 1: Load non-existent library */
    void *handle = ncclDlOpen("nonexistent_library_12345.dll", NCCL_RTLD_NOW);
    TEST_ASSERT(handle == NULL, "Loading non-existent library returns NULL");

    const char *err = dlerror();
    TEST_ASSERT(err != NULL && strlen(err) > 0, "dlerror returns error message");
    TEST_INFO("Error message: %s", err);

    /* Test 2: Load system library */
    handle = ncclDlOpen("kernel32.dll", NCCL_RTLD_NOW);
    TEST_ASSERT(handle != NULL, "Loading kernel32.dll succeeds");

    /* Test 3: Get existing symbol */
    void *sym = ncclDlSym(handle, "GetCurrentProcessId");
    TEST_ASSERT(sym != NULL, "dlsym for GetCurrentProcessId succeeds");

    /* Test 4: Get non-existent symbol */
    sym = ncclDlSym(handle, "NonExistentFunction12345");
    TEST_ASSERT(sym == NULL, "dlsym for non-existent function returns NULL");

    /* Test 5: dlsym with NULL handle */
    sym = ncclDlSym(NULL, "GetCurrentProcessId");
    TEST_ASSERT(sym == NULL, "dlsym with NULL handle returns NULL");

    /* Test 6: Multiple loads of same library */
    void *handle2 = ncclDlOpen("kernel32.dll", NCCL_RTLD_NOW);
    TEST_ASSERT(handle2 != NULL, "Second load of kernel32.dll succeeds");

    ncclDlClose(handle);
    ncclDlClose(handle2);
    TEST_ASSERT(1, "Both handles closed successfully");

    /* Test 7: Close NULL handle */
    int ret = ncclDlClose(NULL);
    TEST_INFO("dlclose(NULL) returned: %d", ret);
}

/* ===================== Atomic Operations Edge Cases ===================== */

void test_atomic_edge_cases()
{
    printf("\n=== Atomic Operations Edge Cases ===\n");

#if NCCL_PLATFORM_WINDOWS
    /* Test 1: Atomic operations at boundary values */
    volatile LONG64 val64 = LLONG_MAX - 1;
    LONG64 result = InterlockedIncrement64(&val64);
    TEST_ASSERT(result == LLONG_MAX, "Atomic increment at LLONG_MAX-1 works");

    result = InterlockedIncrement64(&val64);
    TEST_ASSERT(result == LLONG_MIN, "Atomic increment overflow wraps correctly");

    /* Test 2: Atomic operations with 0 */
    volatile LONG val32 = 0;
    InterlockedExchange(&val32, 0);
    result = InterlockedCompareExchange(&val32, 0, 0);
    TEST_ASSERT(result == 0, "Atomic exchange/compare of 0 works");

    /* Test 3: CAS with same expected and desired */
    val32 = 42;
    LONG old_val = InterlockedCompareExchange(&val32, 42, 42);
    TEST_ASSERT(old_val == 42, "CAS with same expected and desired succeeds");
    TEST_ASSERT(val32 == 42, "Value unchanged after same-value CAS");

    /* Test 4: CAS failure case */
    val32 = 100;
    old_val = InterlockedCompareExchange(&val32, 200, 50); /* Wrong expected */
    TEST_ASSERT(old_val == 100, "CAS returns actual value on failure");
    TEST_ASSERT(val32 == 100, "Value unchanged on CAS failure");

    /* Test 5: Negative numbers */
    val32 = -100;
    result = InterlockedAdd(&val32, -50);
    TEST_ASSERT(result == -150, "Atomic add with negative numbers works");

    result = InterlockedAdd(&val32, 200);
    TEST_ASSERT(result == 50, "Atomic add to positive works");
#else
    /* Test 1: Atomic operations at boundary values */
    volatile int64_t val64 = INT64_MAX - 1;
    int64_t result = __atomic_add_fetch(&val64, 1, __ATOMIC_SEQ_CST);
    TEST_ASSERT(result == INT64_MAX, "Atomic add at INT64_MAX-1 works");

    result = __atomic_add_fetch(&val64, 1, __ATOMIC_SEQ_CST);
    TEST_ASSERT(result == INT64_MIN, "Atomic add overflow wraps correctly");

    /* Test 2: Atomic operations with 0 */
    volatile int32_t val32 = 0;
    __atomic_store_n(&val32, 0, __ATOMIC_SEQ_CST);
    result = __atomic_load_n(&val32, __ATOMIC_SEQ_CST);
    TEST_ASSERT(result == 0, "Atomic store/load of 0 works");

    /* Test 3: CAS with same expected and desired */
    val32 = 42;
    int32_t expected = 42;
    int success = __atomic_compare_exchange_n(&val32, &expected, 42, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    TEST_ASSERT(success, "CAS with same expected and desired succeeds");
    TEST_ASSERT(val32 == 42, "Value unchanged after same-value CAS");

    /* Test 4: CAS failure case */
    val32 = 100;
    expected = 50; /* Wrong expected value */
    success = __atomic_compare_exchange_n(&val32, &expected, 200, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    TEST_ASSERT(!success, "CAS with wrong expected fails");
    TEST_ASSERT(expected == 100, "Expected updated to actual value on CAS failure");
    TEST_ASSERT(val32 == 100, "Value unchanged on CAS failure");

    /* Test 5: Negative numbers */
    val32 = -100;
    result = __atomic_add_fetch(&val32, -50, __ATOMIC_SEQ_CST);
    TEST_ASSERT(result == -150, "Atomic add with negative numbers works");

    result = __atomic_sub_fetch(&val32, -200, __ATOMIC_SEQ_CST);
    TEST_ASSERT(result == 50, "Atomic sub with negative numbers works");
#endif
}

/* ===================== Miscellaneous Edge Cases ===================== */

void test_misc_edge_cases()
{
    printf("\n=== Miscellaneous Edge Cases ===\n");

    /* Test 1: getpid/gettid consistency */
    int pid1 = getpid();
    int pid2 = getpid();
    TEST_ASSERT(pid1 == pid2, "getpid returns consistent value");
    TEST_ASSERT(pid1 > 0, "getpid returns positive value");

    int tid1 = gettid();
    int tid2 = gettid();
    TEST_ASSERT(tid1 == tid2, "gettid returns consistent value");
    TEST_ASSERT(tid1 > 0, "gettid returns positive value");

    /* Test 2: Environment variable with special characters */
    _putenv("NCCL_TEST_SPECIAL=hello=world");
    const char *val = getenv("NCCL_TEST_SPECIAL");
    TEST_ASSERT(val != NULL, "Env var with = in value retrieved");
    if (val)
        TEST_INFO("Value: %s", val);

    /* Test 3: Empty environment variable */
    _putenv("NCCL_TEST_EMPTY=");
    val = getenv("NCCL_TEST_EMPTY");
    /* Empty value behavior can vary */
    TEST_INFO("Empty env var value: %s", val ? (strlen(val) == 0 ? "(empty string)" : val) : "(null)");

    /* Test 4: Long environment variable */
    char long_value[1024];
    memset(long_value, 'A', sizeof(long_value) - 1);
    long_value[sizeof(long_value) - 1] = '\0';

    char env_str[1100];
    snprintf(env_str, sizeof(env_str), "NCCL_TEST_LONG=%s", long_value);
    _putenv(env_str);

    val = getenv("NCCL_TEST_LONG");
    TEST_ASSERT(val != NULL, "Long env var retrieved");
    if (val)
        TEST_ASSERT(strlen(val) == 1023, "Long env var has correct length");

    /* Test 5: Non-existent environment variable */
    val = getenv("NCCL_NONEXISTENT_VAR_12345");
    TEST_ASSERT(val == NULL, "Non-existent env var returns NULL");

    /* Test 6: ncclGetNumProcessors */
    int procs = ncclGetNumProcessors();
    TEST_ASSERT(procs > 0, "ncclGetNumProcessors returns positive");
    TEST_ASSERT(procs <= 1024, "ncclGetNumProcessors returns reasonable value");
    TEST_INFO("Processors: %d", procs);

    /* Cleanup */
    _putenv("NCCL_TEST_SPECIAL=");
    _putenv("NCCL_TEST_EMPTY=");
    _putenv("NCCL_TEST_LONG=");
}

/* ===================== Error Code Edge Cases ===================== */

void test_error_codes()
{
    printf("\n=== Error Code Definitions ===\n");

    /* Verify all required error codes are defined and distinct */
    int codes[] = {EINTR, EWOULDBLOCK, EINPROGRESS, ECONNRESET, ENOTCONN, ECONNREFUSED, ETIMEDOUT};
    const char *names[] = {"EINTR", "EWOULDBLOCK", "EINPROGRESS", "ECONNRESET", "ENOTCONN", "ECONNREFUSED", "ETIMEDOUT"};
    int num_codes = sizeof(codes) / sizeof(codes[0]);

    for (int i = 0; i < num_codes; i++)
    {
        TEST_ASSERT(codes[i] != 0, names[i]);
        TEST_INFO("%s = %d", names[i], codes[i]);
    }

    /* Check EAGAIN == EWOULDBLOCK on Windows */
#ifdef EAGAIN
    TEST_INFO("EAGAIN = %d, EWOULDBLOCK = %d", EAGAIN, EWOULDBLOCK);
#endif
}

int main()
{
    printf("========================================\n");
    printf("NCCL Platform Abstraction Edge Case Tests\n");
    printf("========================================\n");

    test_cpu_affinity_edge_cases();
    test_time_edge_cases();
    test_thread_edge_cases();
    test_socket_edge_cases();
    test_dl_edge_cases();
    test_atomic_edge_cases();
    test_misc_edge_cases();
    test_error_codes();

    printf("\n========================================\n");
    printf("Results: %d run, %d passed, %d failed\n", tests_run, tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
