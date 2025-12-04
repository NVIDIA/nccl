/*************************************************************************
 * NCCL Platform Abstraction Tests - Platform Detection
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#include "test_framework.h"
#include "platform.h"

void test_platform_detection(void)
{
    TEST_SECTION("Platform Detection Tests");

    /* Test that exactly one platform is detected */
    int platform_count = NCCL_PLATFORM_WINDOWS + NCCL_PLATFORM_LINUX;
    TEST_ASSERT_EQ(1, platform_count, "Exactly one platform should be detected");

    /* Test platform macros are defined and boolean */
    TEST_ASSERT(NCCL_PLATFORM_WINDOWS == 0 || NCCL_PLATFORM_WINDOWS == 1,
                "NCCL_PLATFORM_WINDOWS should be 0 or 1");
    TEST_ASSERT(NCCL_PLATFORM_LINUX == 0 || NCCL_PLATFORM_LINUX == 1,
                "NCCL_PLATFORM_LINUX should be 0 or 1");
    TEST_ASSERT(NCCL_PLATFORM_POSIX == 0 || NCCL_PLATFORM_POSIX == 1,
                "NCCL_PLATFORM_POSIX should be 0 or 1");

#if NCCL_PLATFORM_WINDOWS
    printf("  [INFO] Detected platform: Windows\n");
    TEST_ASSERT_EQ(1, NCCL_PLATFORM_WINDOWS, "Windows platform detected correctly");
    TEST_ASSERT_EQ(0, NCCL_PLATFORM_LINUX, "Linux should not be detected on Windows");
    TEST_ASSERT_EQ(0, NCCL_PLATFORM_POSIX, "POSIX should not be detected on Windows");
#else
    printf("  [INFO] Detected platform: Linux/Unix\n");
    TEST_ASSERT_EQ(0, NCCL_PLATFORM_WINDOWS, "Windows should not be detected on Linux");
    TEST_ASSERT_EQ(1, NCCL_PLATFORM_LINUX, "Linux platform detected correctly");
    TEST_ASSERT_EQ(1, NCCL_PLATFORM_POSIX, "POSIX should be detected on Linux");
#endif

    /* Test compiler detection */
    int compiler_count = NCCL_COMPILER_MSVC + NCCL_COMPILER_GCC + NCCL_COMPILER_CLANG;
    TEST_ASSERT(compiler_count <= 1, "At most one compiler should be detected");

#if NCCL_COMPILER_MSVC
    printf("  [INFO] Detected compiler: MSVC\n");
#elif NCCL_COMPILER_GCC
    printf("  [INFO] Detected compiler: GCC\n");
#elif NCCL_COMPILER_CLANG
    printf("  [INFO] Detected compiler: Clang\n");
#else
    printf("  [INFO] Detected compiler: Unknown\n");
#endif

    /* Test PATH_MAX is defined */
    TEST_ASSERT_GT(PATH_MAX, 0, "PATH_MAX should be defined and positive");
    printf("  [INFO] PATH_MAX = %d\n", PATH_MAX);

    /* Test ssize_t is available */
    ssize_t test_ssize = -1;
    TEST_ASSERT_EQ(-1, test_ssize, "ssize_t type should work correctly");
}
