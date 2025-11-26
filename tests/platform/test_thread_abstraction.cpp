/*************************************************************************
 * NCCL Platform Abstraction Tests - Thread Abstraction
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#include "test_framework.h"
#include "platform.h"

#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_thread.h"
#else
#include <pthread.h>
#include <unistd.h>
#endif

/* Shared data for thread tests */
static pthread_mutex_t g_test_mutex;
static pthread_cond_t g_test_cond;
static int g_thread_counter = 0;
static int g_thread_done = 0;

static void *thread_increment(void *arg)
{
    int *count = (int *)arg;

    pthread_mutex_lock(&g_test_mutex);
    (*count)++;
    g_thread_done = 1;
    pthread_cond_signal(&g_test_cond);
    pthread_mutex_unlock(&g_test_mutex);

    return NULL;
}

static void *thread_counter(void *arg)
{
    int iterations = *(int *)arg;

    for (int i = 0; i < iterations; i++)
    {
        pthread_mutex_lock(&g_test_mutex);
        g_thread_counter++;
        pthread_mutex_unlock(&g_test_mutex);
    }

    return NULL;
}

void test_thread_abstraction(void)
{
    TEST_SECTION("Thread Abstraction Tests");

    /* Test mutex initialization */
    int ret = pthread_mutex_init(&g_test_mutex, NULL);
    TEST_ASSERT_EQ(0, ret, "Mutex init should succeed");

    /* Test condition variable initialization */
    ret = pthread_cond_init(&g_test_cond, NULL);
    TEST_ASSERT_EQ(0, ret, "Cond init should succeed");

    /* Test basic mutex lock/unlock */
    ret = pthread_mutex_lock(&g_test_mutex);
    TEST_ASSERT_EQ(0, ret, "Mutex lock should succeed");

    ret = pthread_mutex_unlock(&g_test_mutex);
    TEST_ASSERT_EQ(0, ret, "Mutex unlock should succeed");

    /* Test thread creation and join */
    pthread_t thread;
    int counter = 0;
    g_thread_done = 0;

    ret = pthread_create(&thread, NULL, thread_increment, &counter);
    TEST_ASSERT_EQ(0, ret, "Thread create should succeed");

    /* Wait for thread using condition variable */
    pthread_mutex_lock(&g_test_mutex);
    while (!g_thread_done)
    {
        pthread_cond_wait(&g_test_cond, &g_test_mutex);
    }
    pthread_mutex_unlock(&g_test_mutex);

    ret = pthread_join(thread, NULL);
    TEST_ASSERT_EQ(0, ret, "Thread join should succeed");
    TEST_ASSERT_EQ(1, counter, "Thread should have incremented counter");

    /* Test multiple threads with mutex synchronization */
    g_thread_counter = 0;
    const int num_threads = 4;
    const int iterations = 1000;
    pthread_t threads[num_threads];
    int iter_arg = iterations;

    for (int i = 0; i < num_threads; i++)
    {
        ret = pthread_create(&threads[i], NULL, thread_counter, &iter_arg);
        TEST_ASSERT_EQ(0, ret, "Creating thread should succeed");
    }

    for (int i = 0; i < num_threads; i++)
    {
        ret = pthread_join(threads[i], NULL);
        TEST_ASSERT_EQ(0, ret, "Joining thread should succeed");
    }

    TEST_ASSERT_EQ(num_threads * iterations, g_thread_counter,
                   "Counter should equal threads * iterations (mutex protecting)");
    printf("  [INFO] Counter after %d threads x %d iterations: %d\n",
           num_threads, iterations, g_thread_counter);

    /* Test mutex trylock */
    ret = pthread_mutex_trylock(&g_test_mutex);
    TEST_ASSERT_EQ(0, ret, "Trylock on unlocked mutex should succeed");

    int trylock_ret = pthread_mutex_trylock(&g_test_mutex);
    /* Note: Windows CRITICAL_SECTION allows recursive locking, so this may succeed */
    /* Just test it doesn't crash */
    if (trylock_ret == 0)
    {
        pthread_mutex_unlock(&g_test_mutex);
    }
    printf("  [INFO] Recursive trylock returned: %d\n", trylock_ret);

    pthread_mutex_unlock(&g_test_mutex);

    /* Test pthread_self */
    pthread_t self = pthread_self();
#if NCCL_PLATFORM_WINDOWS
    TEST_ASSERT_NE((pthread_t)NULL, self, "pthread_self should return valid handle");
#else
    TEST_ASSERT_NE((pthread_t)0, self, "pthread_self should return valid id");
#endif

    /* Cleanup */
    ret = pthread_cond_destroy(&g_test_cond);
    TEST_ASSERT_EQ(0, ret, "Cond destroy should succeed");

    ret = pthread_mutex_destroy(&g_test_mutex);
    TEST_ASSERT_EQ(0, ret, "Mutex destroy should succeed");
}
