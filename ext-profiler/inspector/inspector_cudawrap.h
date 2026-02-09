/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef INSPECTOR_CUDAWRAP_H_
#define INSPECTOR_CUDAWRAP_H_

#include <cuda.h>
#include <cuda_runtime.h>

// Inspector-specific CUDA wrapper for standalone compilation

// Include inspector.h for proper type definitions
#include "inspector.h"

// Function pointer types for CUDA driver API functions
typedef CUresult (*PFN_cuGetErrorString)(CUresult error, const char **pStr);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice *device, int ordinal);
typedef CUresult (*PFN_cuDeviceGetUuid)(CUuuid *uuid, CUdevice dev);

// Function pointers - externally defined
extern PFN_cuGetErrorString pfn_cuGetErrorString;
extern PFN_cuDeviceGet pfn_cuDeviceGet;
extern PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid;

// Convenience macro for calling function pointers
#define INSPECTOR_CUPFN(symbol) pfn_##symbol

// Initialize the CUDA wrapper (load function pointers)
inspectorResult_t inspectorCudaWrapInit(void);

// Cleanup the CUDA wrapper
void inspectorCudaWrapCleanup(void);

#endif // INSPECTOR_CUDAWRAP_H_
