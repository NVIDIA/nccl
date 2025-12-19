/*************************************************************************
 * NCCL Platform Abstraction Tests - CPU Affinity
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#include "test_framework.h"
#include "platform.h"

#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_misc.h"
#else
#include <sched.h>
#include <unistd.h>
#endif

void test_cpu_affinity(void)
{
    TEST_SECTION("CPU Affinity Tests");

    /* Test CPU_SETSIZE is defined and reasonable */
    TEST_ASSERT_GT(CPU_SETSIZE, 0, "CPU_SETSIZE should be positive");
    TEST_ASSERT_GE(CPU_SETSIZE, 64, "CPU_SETSIZE should support at least 64 CPUs");
    printf("  [INFO] CPU_SETSIZE = %d\n", CPU_SETSIZE);

    /* Test cpu_set_t operations */
    cpu_set_t set1, set2;

    /* Test CPU_ZERO */
    CPU_ZERO(&set1);
    int count = CPU_COUNT(&set1);
    TEST_ASSERT_EQ(0, count, "CPU_ZERO should create empty set");

    /* Test CPU_SET */
    CPU_SET(0, &set1);
    TEST_ASSERT(CPU_ISSET(0, &set1), "CPU 0 should be set after CPU_SET");
    count = CPU_COUNT(&set1);
    TEST_ASSERT_EQ(1, count, "Set should have 1 CPU after setting CPU 0");

    /* Test setting multiple CPUs */
    CPU_SET(1, &set1);
    CPU_SET(3, &set1);
    TEST_ASSERT(CPU_ISSET(1, &set1), "CPU 1 should be set");
    TEST_ASSERT(CPU_ISSET(3, &set1), "CPU 3 should be set");
    TEST_ASSERT(!CPU_ISSET(2, &set1), "CPU 2 should not be set");
    count = CPU_COUNT(&set1);
    TEST_ASSERT_EQ(3, count, "Set should have 3 CPUs");

    /* Test CPU_CLR */
    CPU_CLR(1, &set1);
    TEST_ASSERT(!CPU_ISSET(1, &set1), "CPU 1 should be cleared after CPU_CLR");
    count = CPU_COUNT(&set1);
    TEST_ASSERT_EQ(2, count, "Set should have 2 CPUs after clearing CPU 1");

    /* Test CPU_ZERO again */
    CPU_ZERO(&set2);
    CPU_SET(0, &set2);
    CPU_SET(3, &set2);
    TEST_ASSERT(CPU_ISSET(0, &set2), "CPU 0 should be in set2");
    TEST_ASSERT(CPU_ISSET(3, &set2), "CPU 3 should be in set2");

#if NCCL_PLATFORM_WINDOWS
    /* Test CPU_EQUAL */
    int equal = CPU_EQUAL(&set1, &set2);
    TEST_ASSERT(equal, "Sets with same CPUs should be equal");

    /* Test CPU_AND */
    cpu_set_t set_and;
    CPU_ZERO(&set_and);
    CPU_ZERO(&set1); /* Reset set1 for clean test */
    CPU_SET(0, &set1);
    CPU_SET(1, &set1);
    CPU_SET(2, &set1);
    CPU_ZERO(&set2);
    CPU_SET(1, &set2);
    CPU_SET(2, &set2);
    CPU_SET(3, &set2);
    CPU_AND(&set_and, &set1, &set2);
    TEST_ASSERT(!CPU_ISSET(0, &set_and), "CPU 0 should not be in AND result");
    TEST_ASSERT(CPU_ISSET(1, &set_and), "CPU 1 should be in AND result");
    TEST_ASSERT(CPU_ISSET(2, &set_and), "CPU 2 should be in AND result");
    TEST_ASSERT(!CPU_ISSET(3, &set_and), "CPU 3 should not be in AND result");

    /* Test CPU_OR */
    cpu_set_t set_or;
    CPU_OR(&set_or, &set1, &set2);
    TEST_ASSERT(CPU_ISSET(0, &set_or), "CPU 0 should be in OR result");
    TEST_ASSERT(CPU_ISSET(1, &set_or), "CPU 1 should be in OR result");
    TEST_ASSERT(CPU_ISSET(2, &set_or), "CPU 2 should be in OR result");
    TEST_ASSERT(CPU_ISSET(3, &set_or), "CPU 3 should be in OR result");
#endif

    /* Test sched_getaffinity */
    cpu_set_t process_affinity;
    CPU_ZERO(&process_affinity);
    int ret = sched_getaffinity(0, sizeof(cpu_set_t), &process_affinity);
    TEST_ASSERT_EQ(0, ret, "sched_getaffinity should succeed");

    count = CPU_COUNT(&process_affinity);
    TEST_ASSERT_GT(count, 0, "Process should have at least one CPU in affinity mask");
    printf("  [INFO] Process affinity has %d CPUs\n", count);

    /* Print which CPUs are in the affinity mask */
    printf("  [INFO] CPUs in affinity mask: ");
    int printed = 0;
    for (int i = 0; i < CPU_SETSIZE && printed < 16; i++)
    {
        if (CPU_ISSET(i, &process_affinity))
        {
            printf("%d ", i);
            printed++;
        }
    }
    if (count > 16)
        printf("... (%d more)", count - 16);
    printf("\n");

#if NCCL_PLATFORM_WINDOWS
    /* Test ncclGetNumProcessors */
    int num_procs = ncclGetNumProcessors();
    TEST_ASSERT_GT(num_procs, 0, "ncclGetNumProcessors should return positive value");
    printf("  [INFO] System has %d processors\n", num_procs);
#endif

    /* Test setting affinity (be careful not to restrict too much) */
    cpu_set_t new_affinity;
    CPU_ZERO(&new_affinity);

    /* Find first available CPU and set affinity to it */
    int first_cpu = -1;
    for (int i = 0; i < CPU_SETSIZE; i++)
    {
        if (CPU_ISSET(i, &process_affinity))
        {
            first_cpu = i;
            break;
        }
    }

    if (first_cpu >= 0)
    {
        CPU_SET(first_cpu, &new_affinity);
        ret = sched_setaffinity(0, sizeof(cpu_set_t), &new_affinity);
        /* Note: This may fail due to permissions, which is OK */
        if (ret == 0)
        {
            printf("  [INFO] Successfully set affinity to CPU %d\n", first_cpu);

            /* Restore original affinity */
            ret = sched_setaffinity(0, sizeof(cpu_set_t), &process_affinity);
            TEST_ASSERT_EQ(0, ret, "Restoring original affinity should succeed");
        }
        else
        {
            printf("  [INFO] Could not set affinity (may require elevated privileges)\n");
        }
    }
}
