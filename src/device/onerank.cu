/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "alloc.h"
#include "collectives.h"
#include "cudawrap.h"
#include "common_kernel.h"
#include "common.h"
#include <cuda_runtime.h>

namespace {
template <typename RedOp>
__global__ __launch_bounds__(512, 1) void oneRankReduce(void* dst, void* src, size_t nElts, uint64_t redOpArg,
                                                        bool redOpArgIsPtr) {
  using T = typename RedOp::EltType;
  int tid = threadIdx.x;
  int tn = blockDim.x;
  int bid = blockIdx.x;
  int bn = gridDim.x;

  // each block/channel gets a roughly equal segment of 16 byte packs
  constexpr int EltPerPack = 16 / sizeof(T);
  intptr_t i0 = (bid + 0) * alignUp(divUp(nElts, bn), EltPerPack);
  intptr_t i1 = (bid + 1) * alignUp(divUp(nElts, bn), EltPerPack);
  i0 = min(i0, nElts);
  i1 = min(i1, nElts);
  src = (T*)src + i0;
  dst = (T*)dst + i0;

  if (redOpArgIsPtr) {
    if (redOpArg % 2 != 0) {
      redOpArg = *reinterpret_cast<uint8_t*>(redOpArg);
    } else if (redOpArg % 4 != 0) {
      redOpArg = *reinterpret_cast<uint16_t*>(redOpArg);
    } else if (redOpArg % 8 != 0) {
      redOpArg = *reinterpret_cast<uint32_t*>(redOpArg);
    } else {
      redOpArg = *reinterpret_cast<uint64_t*>(redOpArg);
    }
  }
  reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/1>(tid, tn, redOpArg, true, 1, &src, 1, &dst,
                                                                       i1 - i0);
}
} // namespace

ncclResult_t ncclLaunchOneRank(void* dst, void const* src, size_t nElts, struct ncclDevRedOpFull redOp,
                               ncclDataType_t eltType, cudaStream_t stream, cudaEvent_t launchCompletionEvent) {
  size_t eltSize = ncclTypeSize(eltType);
  if (redOp.op != ncclDevPreMulSum) {
    if (dst != src) {
      NCCLCHECK(ncclCudaMemcpyAsync((char*)dst, (char*)src, nElts * eltSize, stream));
    }
    return ncclSuccess;
  }

  void const* kernel;
  switch (eltType) {
  case ncclInt8:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<int8_t>>;
    break;
  case ncclUint8:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<uint8_t>>;
    break;
  case ncclInt32:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<int32_t>>;
    break;
  case ncclUint32:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<uint32_t>>;
    break;
  case ncclInt64:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<int64_t>>;
    break;
  case ncclUint64:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<uint64_t>>;
    break;
#if defined(__CUDA_FP8_TYPES_EXIST__) && __CUDA_ARCH__ >= 900
  case ncclFloat8e4m3:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<__nv_fp8_e4m3>>;
    break;
  case ncclFloat8e5m2:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<__nv_fp8_e5m2>>;
    break;
#endif
  case ncclFloat16:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<half>>;
    break;
#if defined(__CUDA_BF16_TYPES_EXIST__)
  case ncclBfloat16:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<__nv_bfloat16>>;
    break;
#endif
  case ncclFloat32:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<float>>;
    break;
  case ncclFloat64:
    kernel = (void const*)&oneRankReduce<FuncPreMulSum<double>>;
    break;
  default:
    return ncclInvalidArgument;
  }
  dim3 grid = {0, 1, 1};
  grid.x = std::min(32, (int)divUp(nElts * eltSize, 16 << 10));
  dim3 block = {512, 1, 1};
  void* args[5] = {&dst, &src, &nElts, &redOp.scalarArg, &redOp.scalarArgIsPtr};

#if CUDART_VERSION >= 12030
  if (launchCompletionEvent != nullptr) {
    int driverVersion;
    NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
    if (driverVersion >= 12030) {
      cudaLaunchAttribute attr = {};
      attr.id = cudaLaunchAttributeLaunchCompletionEvent;
      attr.val.launchCompletionEvent.event = launchCompletionEvent;
      attr.val.launchCompletionEvent.flags = 0;
      cudaLaunchConfig_t config = {};
      config.gridDim = grid;
      config.blockDim = block;
      config.stream = stream;
      config.attrs = &attr;
      config.numAttrs = 1;
      CUDACHECK(cudaLaunchKernelExC(&config, kernel, args));
      return ncclSuccess;
    }
  }
#endif
  if (launchCompletionEvent != nullptr) {
    WARN("CUDA launch-completion events require CUDA 12.3 or newer; recording the user event before launch");
    CUDACHECK(cudaEventRecord(launchCompletionEvent, stream));
  }
  CUDACHECK(cudaLaunchKernel(kernel, grid, block, args, 0, stream));
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Profiler GPU-clock probe.
//
// A 1-thread kernel that samples the GPU globaltimer -- the same
// clock the device profiler stamps into KernelCh/slab timestamps --
// into a host-visible location. The profiler thread launches this
// around CPU clock reads to build a GPU->CPU time mapping, so it can
// convert GPU-derived event timestamps (KernelCh, KernelPhase) to
// CLOCK_MONOTONIC/CLOCK_REALTIME. Only used when a plugin asks for
// NCCL-sourced timings.
// ---------------------------------------------------------------------------
namespace {
__global__ void ncclProfClockProbeKernel(unsigned long long* out) {
  *out = globaltimer();
}
} // namespace

ncclResult_t ncclProfilerProbeGpuClock(unsigned long long* devOut, cudaStream_t stream) {
  void* args[1] = {(void*)&devOut};
  // Deliberately not CUDACHECK: this probe runs continuously on the
  // profiler thread and is often the first CUDA call to observe a
  // context poisoned by unrelated GPU work (e.g. a driver-contained
  // NVLink/hardware fault). A WARN here would wrongly implicate the
  // profiler during triage, so fail quietly, clear the sticky error,
  // and let the caller skip this calibration anchor.
  cudaError_t err = cudaLaunchKernel((void const*)&ncclProfClockProbeKernel,
                                     dim3(1, 1, 1),
                                     dim3(1, 1, 1),
                                     args, 0, stream);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}
