/*************************************************************************
 * NCCL Platform Abstraction Tests - Main Entry Point
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#include "test_framework.h"

#include <stdio.h>

int main(int argc, char *argv[])
{
    printf("========================================\n");
    printf("NCCL Platform Abstraction Test Suite\n");
    printf("========================================\n");

    /* Run all test modules */
    test_platform_detection();
    test_time_functions();
    test_socket_abstraction();
    test_thread_abstraction();
    test_cpu_affinity();
    test_misc_functions();

    /* Print summary */
    TEST_SUMMARY();

    /* Return 0 if all tests passed, 1 otherwise */
    return (g_tests_failed > 0) ? 1 : 0;
}
