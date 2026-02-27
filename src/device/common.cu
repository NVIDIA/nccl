/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"
#include "nccl_device.h"
#include "comm.h"

__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

struct RunWorkNop {
  __device__ void run() {}
};

__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {
  ncclKernelMain<-1, RunWorkNop>(&args4K.args);
}

__global__ void ncclDevKernelGinResetSignalsAndCounters(ncclDevComm devComm) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalThreads = gridDim.x * blockDim.x;

  int signalCount = devComm.ginSignalCount;
  int counterCount = devComm.ginCounterCount;

  // Reset signals and counters for all contexts
  for (int contextIdx = 0; contextIdx < devComm.ginContextCount; contextIdx++) {
    ncclGin gin(devComm, contextIdx);

    for (int i = tid; i < signalCount; i += totalThreads) {
      gin.resetSignal(i);
    }

    for (int i = tid; i < counterCount; i += totalThreads) {
      gin.resetCounter(i);
    }
  }
}

ncclResult_t ncclGinResetSignalsAndCounters(struct ncclComm* comm, ncclDevComm_t const* devComm) {
  int deviceWork = std::max(devComm->ginSignalCount, devComm->ginCounterCount);

  if (deviceWork == 0) {
    return ncclSuccess;
  }

  // Ensure we run on the comm's device (important when called from async reclaim thread)
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  dim3 grid(1);
  dim3 block(ncclSymkMaxThreads);

  // NOTE: Use a dedicated stream so we only wait for the reset kernel, not other device work.
  cudaStream_t stream = nullptr;
  CUDACHECK(cudaStreamCreate(&stream));

  void* args[] = { (void*)devComm };
  CUDACHECK(cudaLaunchKernel((void*)ncclDevKernelGinResetSignalsAndCounters, grid, block, args, 0,
                             stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaStreamDestroy(stream));

  return ncclSuccess;
}

__device__ void ncclDevFunc_Nop() {}
