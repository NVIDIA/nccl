/*************************************************************************
 * NCCL CUDA Shared Memory Tests
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Tests for CUDA kernel launch with various shared memory sizes,
 * particularly verifying the fix for >48KB shared memory.
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* Test framework macros */
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

#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess)                                     \
        {                                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                        \
            return false;                                           \
        }                                                           \
    } while (0)

#define CU_CHECK(call)                                                     \
    do                                                                     \
    {                                                                      \
        CUresult err = call;                                               \
        if (err != CUDA_SUCCESS)                                           \
        {                                                                  \
            const char *errStr;                                            \
            cuGetErrorString(err, &errStr);                                \
            printf("CUDA driver error at %s:%d: %s\n", __FILE__, __LINE__, \
                   errStr ? errStr : "unknown");                           \
            return false;                                                  \
        }                                                                  \
    } while (0)

/* Simple kernel that uses dynamic shared memory */
__global__ void sharedMemoryKernel(int *output, int smemSize)
{
    extern __shared__ char sharedMem[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Write to shared memory
    if (threadIdx.x < smemSize / sizeof(int))
    {
        ((int *)sharedMem)[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();

    // Read back and write to global memory
    if (tid == 0)
    {
        *output = ((int *)sharedMem)[0];
    }
}

/* Kernel that specifically requires large shared memory */
__global__ void largeSharedMemoryKernel(int *output, int smemSize)
{
    extern __shared__ char sharedMem[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize shared memory pattern
    for (int i = threadIdx.x; i < smemSize / sizeof(int); i += blockDim.x)
    {
        ((int *)sharedMem)[i] = i;
    }
    __syncthreads();

    // Compute checksum
    int checksum = 0;
    for (int i = threadIdx.x; i < smemSize / sizeof(int); i += blockDim.x)
    {
        checksum += ((int *)sharedMem)[i];
    }

    // Reduce within block (simplified)
    __shared__ int blockSum;
    if (threadIdx.x == 0)
        blockSum = 0;
    __syncthreads();
    atomicAdd(&blockSum, checksum);
    __syncthreads();

    if (tid == 0)
    {
        *output = blockSum;
    }
}

/* Test launching kernel with shared memory below 48KB */
bool test_small_shared_memory()
{
    printf("\n--- Test: Small Shared Memory (<48KB) ---\n");

    int *d_output;
    int h_output = -1;
    const int smemSize = 16 * 1024; // 16KB

    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(int)));

    // Launch with small shared memory - should work without special attributes
    sharedMemoryKernel<<<1, 256, smemSize>>>(d_output, smemSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));

    printf("  Kernel executed with %d KB shared memory\n", smemSize / 1024);
    printf("  Output: %d (expected: 0)\n", h_output);

    return h_output == 0;
}

/* Test launching kernel with shared memory at exactly 48KB */
bool test_boundary_shared_memory()
{
    printf("\n--- Test: Boundary Shared Memory (48KB) ---\n");

    int *d_output;
    int h_output = -1;
    const int smemSize = 48 * 1024; // Exactly 48KB

    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(int)));

    // Launch at the boundary - may or may not need special attributes
    sharedMemoryKernel<<<1, 256, smemSize>>>(d_output, smemSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));

    printf("  Kernel executed with %d KB shared memory\n", smemSize / 1024);
    printf("  Output: %d (expected: 0)\n", h_output);

    return h_output == 0;
}

/* Test launching kernel with shared memory above 48KB using driver API */
bool test_large_shared_memory_driver_api()
{
    printf("\n--- Test: Large Shared Memory (>48KB) with Driver API ---\n");

    // Get device properties to check max shared memory
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int maxSharedMem = prop.sharedMemPerBlockOptin;
    printf("  Device: %s\n", prop.name);
    printf("  Max shared memory per block (optin): %d KB\n", maxSharedMem / 1024);

    if (maxSharedMem < 64 * 1024)
    {
        printf("  SKIP: Device doesn't support >48KB shared memory\n");
        return true; // Skip test on devices that don't support large smem
    }

    // Test sizes: 64KB, 80KB (like the 82KB that was failing)
    int testSizes[] = {64 * 1024, 80 * 1024};
    int numTests = sizeof(testSizes) / sizeof(testSizes[0]);

    for (int t = 0; t < numTests; t++)
    {
        int smemSize = testSizes[t];

        if (smemSize > maxSharedMem)
        {
            printf("  SKIP: %d KB exceeds device limit\n", smemSize / 1024);
            continue;
        }

        printf("\n  Testing %d KB shared memory:\n", smemSize / 1024);

        int *d_output;
        int h_output = -1;

        CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(int)));

        // Get CUfunction handle
        CUfunction fn;
        cudaError_t cudaRes = cudaGetFuncBySymbol(&fn, (const void *)largeSharedMemoryKernel);
        if (cudaRes != cudaSuccess)
        {
            printf("    Failed to get CUfunction handle: %s\n", cudaGetErrorString(cudaRes));
            cudaFree(d_output);
            return false;
        }

        // Set max dynamic shared memory attribute - THIS IS THE KEY FIX
        CUresult cuRes = cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smemSize);
        if (cuRes != CUDA_SUCCESS)
        {
            const char *errStr;
            cuGetErrorString(cuRes, &errStr);
            printf("    Failed to set shared memory attribute: %s\n", errStr ? errStr : "unknown");
            cudaFree(d_output);
            return false;
        }
        printf("    Set max dynamic shared memory to %d KB\n", smemSize / 1024);

        // Launch using cuLaunchKernel
        void *args[] = {&d_output, &smemSize};
        cuRes = cuLaunchKernel(fn,
                               1, 1, 1,   // grid
                               256, 1, 1, // block
                               smemSize,  // shared memory
                               0,         // stream
                               args,      // kernel args
                               nullptr);  // extra

        if (cuRes != CUDA_SUCCESS)
        {
            const char *errStr;
            cuGetErrorString(cuRes, &errStr);
            printf("    cuLaunchKernel failed: %s\n", errStr ? errStr : "unknown");
            cudaFree(d_output);
            return false;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_output));

        printf("    Kernel executed successfully with %d KB shared memory\n", smemSize / 1024);
        printf("    Output checksum: %d\n", h_output);
    }

    return true;
}

/* Test that launching WITHOUT setting attribute fails for large smem */
bool test_large_shared_memory_without_attribute()
{
    printf("\n--- Test: Large Shared Memory WITHOUT Attribute (Expected Failure) ---\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int maxSharedMem = prop.sharedMemPerBlockOptin;
    if (maxSharedMem <= 48 * 1024)
    {
        printf("  SKIP: Device doesn't support >48KB shared memory\n");
        return true;
    }

    const int smemSize = 64 * 1024; // 64KB - above default limit

    int *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(int)));

    // Get CUfunction handle
    CUfunction fn;
    cudaError_t cudaRes = cudaGetFuncBySymbol(&fn, (const void *)largeSharedMemoryKernel);
    if (cudaRes != cudaSuccess)
    {
        printf("  Failed to get CUfunction handle: %s\n", cudaGetErrorString(cudaRes));
        cudaFree(d_output);
        return false;
    }

    // Intentionally DO NOT set the shared memory attribute
    printf("  Attempting to launch with %d KB shared memory WITHOUT setting attribute...\n", smemSize / 1024);

    void *args[] = {&d_output, (void *)&smemSize};
    CUresult cuRes = cuLaunchKernel(fn,
                                    1, 1, 1,   // grid
                                    256, 1, 1, // block
                                    smemSize,  // shared memory > 48KB
                                    0,         // stream
                                    args,      // kernel args
                                    nullptr);  // extra

    cudaFree(d_output);

    if (cuRes == CUDA_ERROR_INVALID_VALUE)
    {
        printf("  Got expected CUDA_ERROR_INVALID_VALUE - this confirms the bug scenario\n");
        return true; // This is the expected behavior without the fix
    }
    else if (cuRes == CUDA_SUCCESS)
    {
        // Some newer drivers/GPUs might not require this
        printf("  Launch succeeded (driver may not require attribute)\n");
        cudaDeviceSynchronize();
        return true;
    }
    else
    {
        const char *errStr;
        cuGetErrorString(cuRes, &errStr);
        printf("  Got unexpected error: %s\n", errStr ? errStr : "unknown");
        return false;
    }
}

/* Test the exact configuration from the bug report */
bool test_exact_bug_configuration()
{
    printf("\n--- Test: Exact Bug Configuration (grid=2, block=640, smem=82240) ---\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int maxSharedMem = prop.sharedMemPerBlockOptin;
    printf("  Device: %s\n", prop.name);
    printf("  Max shared memory per block (optin): %d bytes (%d KB)\n",
           maxSharedMem, maxSharedMem / 1024);

    const int smemSize = 82240; // Exact size from bug report
    const int gridX = 2;
    const int blockX = 640;

    if (smemSize > maxSharedMem)
    {
        printf("  SKIP: %d bytes exceeds device limit of %d bytes\n", smemSize, maxSharedMem);
        return true;
    }

    printf("  Testing: grid=(%d,1,1) block=(%d,1,1) smem=%d\n", gridX, blockX, smemSize);

    int *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(int)));

    // Get CUfunction handle
    CUfunction fn;
    cudaError_t cudaRes = cudaGetFuncBySymbol(&fn, (const void *)largeSharedMemoryKernel);
    if (cudaRes != cudaSuccess)
    {
        printf("  Failed to get CUfunction handle: %s\n", cudaGetErrorString(cudaRes));
        cudaFree(d_output);
        return false;
    }

    // Set max dynamic shared memory attribute - THE FIX
    CUresult cuRes = cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smemSize);
    if (cuRes != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(cuRes, &errStr);
        printf("  Failed to set shared memory attribute: %s\n", errStr ? errStr : "unknown");
        cudaFree(d_output);
        return false;
    }
    printf("  Set CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = %d\n", smemSize);

    // Launch with exact bug configuration
    void *args[] = {&d_output, (void *)&smemSize};
    cuRes = cuLaunchKernel(fn,
                           gridX, 1, 1,  // grid - exact from bug
                           blockX, 1, 1, // block - exact from bug
                           smemSize,     // shared memory - exact from bug
                           0,            // stream
                           args,         // kernel args
                           nullptr);     // extra

    if (cuRes != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(cuRes, &errStr);
        printf("  cuLaunchKernel failed: %s (error %d)\n", errStr ? errStr : "unknown", cuRes);
        cudaFree(d_output);
        return false;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess)
    {
        printf("  cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(syncErr));
        cudaFree(d_output);
        return false;
    }

    int h_output;
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));

    printf("  SUCCESS: Kernel executed with exact bug configuration!\n");
    printf("  Output checksum: %d\n", h_output);

    return true;
}

/* Test cuLaunchKernelEx with large shared memory (CUDA 11.8+) */
#if CUDART_VERSION >= 11080
bool test_launch_kernel_ex_large_smem()
{
    printf("\n--- Test: cuLaunchKernelEx with Large Shared Memory ---\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int maxSharedMem = prop.sharedMemPerBlockOptin;
    if (maxSharedMem <= 48 * 1024)
    {
        printf("  SKIP: Device doesn't support >48KB shared memory\n");
        return true;
    }

    const int smemSize = 64 * 1024;

    printf("  Testing cuLaunchKernelEx with %d KB shared memory\n", smemSize / 1024);

    int *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(int)));

    // Get CUfunction handle
    CUfunction fn;
    cudaError_t cudaRes = cudaGetFuncBySymbol(&fn, (const void *)largeSharedMemoryKernel);
    if (cudaRes != cudaSuccess)
    {
        printf("  Failed to get CUfunction handle: %s\n", cudaGetErrorString(cudaRes));
        cudaFree(d_output);
        return false;
    }

    // Set max dynamic shared memory attribute
    CUresult cuRes = cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smemSize);
    if (cuRes != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(cuRes, &errStr);
        printf("  Failed to set shared memory attribute: %s\n", errStr ? errStr : "unknown");
        cudaFree(d_output);
        return false;
    }

    // Set up launch config for cuLaunchKernelEx
    CUlaunchConfig launchConfig = {0};
    launchConfig.gridDimX = 1;
    launchConfig.gridDimY = 1;
    launchConfig.gridDimZ = 1;
    launchConfig.blockDimX = 256;
    launchConfig.blockDimY = 1;
    launchConfig.blockDimZ = 1;
    launchConfig.sharedMemBytes = smemSize;
    launchConfig.hStream = 0;
    launchConfig.attrs = nullptr;
    launchConfig.numAttrs = 0;

    // Use standard kernel argument passing
    void *kernelArgs[] = {&d_output, (void *)&smemSize};

    cuRes = cuLaunchKernelEx(&launchConfig, fn, kernelArgs, nullptr);

    if (cuRes != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(cuRes, &errStr);
        printf("  cuLaunchKernelEx failed: %s\n", errStr ? errStr : "unknown");
        cudaFree(d_output);
        return false;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    int h_output;
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));

    printf("  SUCCESS: cuLaunchKernelEx executed with %d KB shared memory!\n", smemSize / 1024);
    printf("  Output checksum: %d\n", h_output);

    return true;
}
#endif

/* Print test summary */
void print_summary()
{
    printf("\n========================================\n");
    printf("Test Summary: %d/%d passed", g_tests_passed, g_tests_run);
    if (g_tests_failed > 0)
    {
        printf(" (%d FAILED)", g_tests_failed);
    }
    printf("\n========================================\n");
}

int main(int argc, char **argv)
{
    printf("NCCL CUDA Shared Memory Tests\n");
    printf("==============================\n");
    printf("Testing fix for CUDA kernel launch with >48KB shared memory\n");

    // Initialize CUDA
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0)
    {
        printf("No CUDA devices found. Skipping tests.\n");
        return 0;
    }

    cudaSetDevice(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice 0: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Shared memory per block: %d KB\n", (int)(prop.sharedMemPerBlock / 1024));
    printf("Shared memory per block (optin): %d KB\n", prop.sharedMemPerBlockOptin / 1024);

    // Run tests
    TEST_ASSERT(test_small_shared_memory(), "Small shared memory (<48KB)");
    TEST_ASSERT(test_boundary_shared_memory(), "Boundary shared memory (48KB)");
    TEST_ASSERT(test_large_shared_memory_driver_api(), "Large shared memory with driver API");
    TEST_ASSERT(test_large_shared_memory_without_attribute(), "Large shared memory without attribute (expected failure)");
    TEST_ASSERT(test_exact_bug_configuration(), "Exact bug configuration (grid=2, block=640, smem=82240)");

#if CUDART_VERSION >= 11080
    TEST_ASSERT(test_launch_kernel_ex_large_smem(), "cuLaunchKernelEx with large shared memory");
#endif

    print_summary();

    return g_tests_failed > 0 ? 1 : 0;
}
