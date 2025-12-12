/*************************************************************************
 * NCCL Platform Abstraction Tests - Miscellaneous Functions
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#include "test_framework.h"
#include "platform.h"

#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_misc.h"
#include "platform/win32_dl.h"
#else
#include <unistd.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <sys/types.h>
#endif

void test_misc_functions(void)
{
    TEST_SECTION("Miscellaneous Functions Tests");

    /* Test getpid */
    pid_t pid = getpid();
    TEST_ASSERT_GT(pid, 0, "getpid should return positive process ID");
    printf("  [INFO] Current process ID: %d\n", (int)pid);

#if NCCL_PLATFORM_WINDOWS
    /* Test gettid */
    pid_t tid = gettid();
    TEST_ASSERT_GT(tid, 0, "gettid should return positive thread ID");
    printf("  [INFO] Current thread ID: %d\n", (int)tid);
#endif

    /* Test environment variables */
    const char *test_env_name = "NCCL_TEST_ENV_VAR";
    const char *test_env_value = "test_value_12345";

    /* Set environment variable */
#if NCCL_PLATFORM_WINDOWS
    char setenv_cmd[256];
    snprintf(setenv_cmd, sizeof(setenv_cmd), "%s=%s", test_env_name, test_env_value);
    int setret = _putenv(setenv_cmd);
#else
    int setret = setenv(test_env_name, test_env_value, 1);
#endif
    TEST_ASSERT_EQ(0, setret, "Setting environment variable should succeed");

    /* Get environment variable */
    const char *got_value = getenv(test_env_name);
    TEST_ASSERT_NE((const char *)NULL, got_value, "getenv should find the variable");
    if (got_value)
    {
        TEST_ASSERT_EQ(0, strcmp(got_value, test_env_value),
                       "getenv should return correct value");
        printf("  [INFO] Environment variable %s = %s\n", test_env_name, got_value);
    }

    /* Unset environment variable */
#if NCCL_PLATFORM_WINDOWS
    snprintf(setenv_cmd, sizeof(setenv_cmd), "%s=", test_env_name);
    _putenv(setenv_cmd);
#else
    unsetenv(test_env_name);
#endif
    got_value = getenv(test_env_name);
    /* After unsetting, should be NULL or empty */
#if NCCL_PLATFORM_WINDOWS
    TEST_ASSERT(got_value == NULL || strlen(got_value) == 0,
                "After unset, env var should be NULL or empty");
#else
    TEST_ASSERT_EQ((const char *)NULL, got_value,
                   "After unsetenv, getenv should return NULL");
#endif

    /* Test NCCL_DEBUG environment variable handling */
    const char *nccl_debug = getenv("NCCL_DEBUG");
    if (nccl_debug)
    {
        printf("  [INFO] NCCL_DEBUG = %s\n", nccl_debug);
    }
    else
    {
        printf("  [INFO] NCCL_DEBUG not set\n");
    }

    /* Test dynamic library loading */
    void *handle = NULL;

#if NCCL_PLATFORM_WINDOWS
    /* Try to load kernel32.dll (always available on Windows) */
    handle = dlopen("kernel32.dll", RTLD_NOW);
    TEST_ASSERT_NE((void *)NULL, handle, "dlopen kernel32.dll should succeed");

    if (handle)
    {
        /* Try to get a known function */
        void *func = dlsym(handle, "GetCurrentProcessId");
        TEST_ASSERT_NE((void *)NULL, func, "dlsym GetCurrentProcessId should succeed");

        /* Close the library */
        int closeret = dlclose(handle);
        TEST_ASSERT_EQ(0, closeret, "dlclose should succeed");
    }

    /* Test loading non-existent library */
    handle = dlopen("nonexistent_library_xyz123.dll", RTLD_NOW);
    TEST_ASSERT_EQ((void *)NULL, handle, "dlopen non-existent library should fail");

    const char *error = dlerror();
    TEST_ASSERT_NE((const char *)NULL, error, "dlerror should return error message");
    printf("  [INFO] dlerror for non-existent library: %s\n", error ? error : "(null)");

#else
    /* Try to load libc (always available on Linux) */
    handle = dlopen(NULL, RTLD_NOW); /* NULL loads the main program */
    TEST_ASSERT_NE((void *)NULL, handle, "dlopen NULL should succeed");

    if (handle)
    {
        dlclose(handle);
    }
#endif

    /* Test atomic operations */
    volatile long atomic_val = 0;

    NCCL_ATOMIC_STORE(&atomic_val, 42);
    long loaded = NCCL_ATOMIC_LOAD(&atomic_val);
    TEST_ASSERT_EQ(42, loaded, "Atomic store/load should work");

    NCCL_ATOMIC_ADD(&atomic_val, 10);
    loaded = NCCL_ATOMIC_LOAD(&atomic_val);
    TEST_ASSERT_EQ(52, loaded, "Atomic add should work");

    NCCL_ATOMIC_SUB(&atomic_val, 2);
    loaded = NCCL_ATOMIC_LOAD(&atomic_val);
    TEST_ASSERT_EQ(50, loaded, "Atomic sub should work");

    /* Test compare-and-swap */
    long expected = 50;
    int cas_result = NCCL_ATOMIC_CAS(&atomic_val, expected, 100);
    TEST_ASSERT(cas_result, "CAS with correct expected value should succeed");
    loaded = NCCL_ATOMIC_LOAD(&atomic_val);
    TEST_ASSERT_EQ(100, loaded, "Value should be updated after successful CAS");

    /* Test memory barriers (just verify they compile and don't crash) */
    NCCL_MEMORY_BARRIER();
    NCCL_COMPILER_BARRIER();
    TEST_ASSERT(1, "Memory barriers should not crash");

    printf("  [INFO] Atomic operations test completed successfully\n");
}
