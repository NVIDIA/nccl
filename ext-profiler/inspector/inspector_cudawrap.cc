/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "inspector_cudawrap.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// Function pointer storage
PFN_cuGetErrorString pfn_cuGetErrorString = nullptr;
PFN_cuDeviceGet pfn_cuDeviceGet = nullptr;
PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid = nullptr;

// Handle to the CUDA driver library
static void* cudaDriverLib = nullptr;

/*
 * Description:
 *
 *   Loads a CUDA driver function symbol from the library.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during initialization).
 *
 * Input:
 *   const char* symbol_name - name of the symbol to load.
 *
 * Output:
 *   None.
 *
 * Return:
 *   void* - pointer to the loaded symbol, or nullptr on failure.
 */
static void* loadCudaSymbol(const char* symbol_name) {
  if (!cudaDriverLib) {
    return nullptr;
  }

  void* symbol = dlsym(cudaDriverLib, symbol_name);
  if (!symbol) {
    INFO_INSPECTOR("Inspector: Failed to load CUDA symbol '%s': %s",
         symbol_name, dlerror());
    return nullptr;
  }

  return symbol;
}

/*
 * Description:
 *
 *   Initializes the CUDA wrapper by loading the CUDA driver library
 *   and resolving required function pointers.
 *
 * Thread Safety:
 *   Not thread-safe (should be called once during initialization).
 *
 * Input:
 *   None.
 *
 * Output:
 *   None.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorCudaWrapInit(void) {
  // Clear any previous dlopen errors
  dlerror();

  // Try to load CUDA driver library
  cudaDriverLib = dlopen("libcuda.so", RTLD_LAZY);
  if (!cudaDriverLib) {
    // Try alternative name
    cudaDriverLib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!cudaDriverLib) {
      INFO_INSPECTOR("Inspector: Failed to load CUDA driver library: %s", dlerror());
      return inspectorCudaError;
    }
  }

  // Load required CUDA driver functions
  pfn_cuGetErrorString = (PFN_cuGetErrorString)loadCudaSymbol("cuGetErrorString");
  if (!pfn_cuGetErrorString) {
    INFO_INSPECTOR("Inspector: Failed to load cuGetErrorString");
    inspectorCudaWrapCleanup();
    return inspectorCudaError;
  }

  pfn_cuDeviceGet = (PFN_cuDeviceGet)loadCudaSymbol("cuDeviceGet");
  if (!pfn_cuDeviceGet) {
    INFO_INSPECTOR("Inspector: Failed to load cuDeviceGet");
    inspectorCudaWrapCleanup();
    return inspectorCudaError;
  }

  pfn_cuDeviceGetUuid = (PFN_cuDeviceGetUuid)loadCudaSymbol("cuDeviceGetUuid");
  if (!pfn_cuDeviceGetUuid) {
    INFO_INSPECTOR("Inspector: Failed to load cuDeviceGetUuid");
    inspectorCudaWrapCleanup();
    return inspectorCudaError;
  }

  INFO(NCCL_INIT, "Inspector: CUDA wrapper initialized successfully");
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Cleans up the CUDA wrapper by closing the driver library.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during cleanup).
 *
 * Input:
 *   None.
 *
 * Output:
 *   None.
 *
 * Return:
 *   None.
 */
void inspectorCudaWrapCleanup(void) {
  // Clear function pointers
  pfn_cuGetErrorString = nullptr;
  pfn_cuDeviceGet = nullptr;
  pfn_cuDeviceGetUuid = nullptr;

  // Close library handle
  if (cudaDriverLib) {
    dlclose(cudaDriverLib);
    cudaDriverLib = nullptr;
  }
}
