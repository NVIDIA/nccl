/*************************************************************************
 * NCCL Platform Abstraction Tests - Test Framework Header
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#ifndef NCCL_TEST_FRAMEWORK_H_
#define NCCL_TEST_FRAMEWORK_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple test framework */
static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(condition, msg)                                      \
    do                                                                   \
    {                                                                    \
        g_tests_run++;                                                   \
        if (condition)                                                   \
        {                                                                \
            g_tests_passed++;                                            \
            printf("  [PASS] %s\n", msg);                                \
        }                                                                \
        else                                                             \
        {                                                                \
            g_tests_failed++;                                            \
            printf("  [FAIL] %s (at %s:%d)\n", msg, __FILE__, __LINE__); \
        }                                                                \
    } while (0)

#define TEST_ASSERT_EQ(expected, actual, msg)                          \
    do                                                                 \
    {                                                                  \
        g_tests_run++;                                                 \
        if ((expected) == (actual))                                    \
        {                                                              \
            g_tests_passed++;                                          \
            printf("  [PASS] %s\n", msg);                              \
        }                                                              \
        else                                                           \
        {                                                              \
            g_tests_failed++;                                          \
            printf("  [FAIL] %s (expected %lld, got %lld at %s:%d)\n", \
                   msg, (long long)(expected), (long long)(actual),    \
                   __FILE__, __LINE__);                                \
        }                                                              \
    } while (0)

#define TEST_ASSERT_NE(not_expected, actual, msg)                        \
    do                                                                   \
    {                                                                    \
        g_tests_run++;                                                   \
        if ((not_expected) != (actual))                                  \
        {                                                                \
            g_tests_passed++;                                            \
            printf("  [PASS] %s\n", msg);                                \
        }                                                                \
        else                                                             \
        {                                                                \
            g_tests_failed++;                                            \
            printf("  [FAIL] %s (got unexpected value %lld at %s:%d)\n", \
                   msg, (long long)(actual), __FILE__, __LINE__);        \
        }                                                                \
    } while (0)

#define TEST_ASSERT_GT(actual, threshold, msg)                       \
    do                                                               \
    {                                                                \
        g_tests_run++;                                               \
        if ((actual) > (threshold))                                  \
        {                                                            \
            g_tests_passed++;                                        \
            printf("  [PASS] %s\n", msg);                            \
        }                                                            \
        else                                                         \
        {                                                            \
            g_tests_failed++;                                        \
            printf("  [FAIL] %s (%lld not > %lld at %s:%d)\n",       \
                   msg, (long long)(actual), (long long)(threshold), \
                   __FILE__, __LINE__);                              \
        }                                                            \
    } while (0)

#define TEST_ASSERT_GE(actual, threshold, msg)                       \
    do                                                               \
    {                                                                \
        g_tests_run++;                                               \
        if ((actual) >= (threshold))                                 \
        {                                                            \
            g_tests_passed++;                                        \
            printf("  [PASS] %s\n", msg);                            \
        }                                                            \
        else                                                         \
        {                                                            \
            g_tests_failed++;                                        \
            printf("  [FAIL] %s (%lld not >= %lld at %s:%d)\n",      \
                   msg, (long long)(actual), (long long)(threshold), \
                   __FILE__, __LINE__);                              \
        }                                                            \
    } while (0)

#define TEST_SECTION(name) printf("\n=== %s ===\n", name)

#define TEST_SUMMARY()                                          \
    do                                                          \
    {                                                           \
        printf("\n========================================\n"); \
        printf("Test Summary: %d run, %d passed, %d failed\n",  \
               g_tests_run, g_tests_passed, g_tests_failed);    \
        printf("========================================\n");   \
    } while (0)

/* Function declarations for test modules */
void test_platform_detection(void);
void test_time_functions(void);
void test_socket_abstraction(void);
void test_thread_abstraction(void);
void test_cpu_affinity(void);
void test_misc_functions(void);

#endif /* NCCL_TEST_FRAMEWORK_H_ */
