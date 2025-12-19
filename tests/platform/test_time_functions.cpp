/*************************************************************************
 * NCCL Platform Abstraction Tests - Time Functions
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#include "test_framework.h"
#include "platform.h"

#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_misc.h"
#else
#include <time.h>
#include <unistd.h>
#endif

void test_time_functions(void)
{
    TEST_SECTION("Time Functions Tests");

    /* Test clock_gettime with CLOCK_MONOTONIC */
    struct timespec ts1, ts2;
    int ret;

    ret = clock_gettime(CLOCK_MONOTONIC, &ts1);
    TEST_ASSERT_EQ(0, ret, "clock_gettime(CLOCK_MONOTONIC) should succeed");
    TEST_ASSERT_GE(ts1.tv_sec, 0, "tv_sec should be non-negative");
    TEST_ASSERT_GE(ts1.tv_nsec, 0, "tv_nsec should be non-negative");
    TEST_ASSERT(ts1.tv_nsec < 1000000000L, "tv_nsec should be less than 1 billion");

    /* Sleep briefly and check time advances */
#if NCCL_PLATFORM_WINDOWS
    Sleep(10); /* 10 milliseconds */
#else
    usleep(10000); /* 10 milliseconds */
#endif

    ret = clock_gettime(CLOCK_MONOTONIC, &ts2);
    TEST_ASSERT_EQ(0, ret, "Second clock_gettime should succeed");

    /* Calculate elapsed time in nanoseconds */
    long long elapsed_ns = (ts2.tv_sec - ts1.tv_sec) * 1000000000LL +
                           (ts2.tv_nsec - ts1.tv_nsec);
    TEST_ASSERT_GT(elapsed_ns, 0, "Time should advance after sleep");
    TEST_ASSERT_GE(elapsed_ns, 5000000LL, "At least 5ms should have elapsed");
    printf("  [INFO] Elapsed time: %lld ns\n", elapsed_ns);

    /* Test clock_gettime with CLOCK_REALTIME */
    struct timespec ts_real;
    ret = clock_gettime(CLOCK_REALTIME, &ts_real);
    TEST_ASSERT_EQ(0, ret, "clock_gettime(CLOCK_REALTIME) should succeed");
    TEST_ASSERT_GT(ts_real.tv_sec, 0, "Real time seconds should be positive");
    printf("  [INFO] Current time: %lld seconds since epoch\n", (long long)ts_real.tv_sec);

#if NCCL_PLATFORM_WINDOWS
    /* Test NULL pointer handling - only on Windows where our wrapper handles it */
    /* On Linux, passing NULL to clock_gettime causes SIGSEGV (not caught by libc) */
    ret = clock_gettime(CLOCK_MONOTONIC, NULL);
    TEST_ASSERT_EQ(-1, ret, "clock_gettime with NULL should fail");
#else
    printf("  [INFO] Skipping NULL pointer test (would crash on native Linux syscall)\n");
#endif
}
