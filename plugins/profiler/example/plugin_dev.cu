/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*************************************************************************
 * Device-side profiler hook for the example plugin (profiler interface v7).
 *
 * exampleProfilerGetDeviceHook() returns a __device__ callback that NCCL invokes
 * once per work item from inside ncclKernelMain's profiler(). The hook counts
 * events per device into its own device context (the plugin owns all telemetry);
 * exampleProfilerDrainDeviceHook() copies the counters back and prints them at
 * finalize, resolving the last funcId via the libnccl-exported
 * ncclProfilerDevFuncName().
 *
 * Opt-in: the hook is only returned when NCCL_PROFILER_EXAMPLE_DEV_HOOK is set,
 * so loading this example otherwise keeps its host-side behavior unchanged.
 *
 * Built with nvcc + relocatable device code (see Makefile): the __device__ hook
 * must have a real, callable global address for NCCL's kernel to invoke it.
 *************************************************************************/
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "err.h" // ncclResult_t
#include "profiler_dev.h"

#define EXAMPLE_MAX_DEVICES 64

// Per-device telemetry owned by the hook (its ctx). A real plugin would use its
// own ring/buffer; here a counter plus the last event's identity suffices.
struct ExampleDevCtx {
  unsigned long long count;
  unsigned int lastFuncId;
  unsigned int lastPhase;
};

// The device hook: cheap, leaf, touches only its own ctx. Runs on lane 0, once
// per profiler-enabled work item, directly inside the NCCL kernel.
__device__ void exampleDevHook(const ncclProfilerDevEvent_t* ev, void* ctx) {
  ExampleDevCtx* c = (ExampleDevCtx*)ctx;
  atomicAdd(&c->count, 1ULL);
  c->lastFuncId = ev->funcId;
  c->lastPhase = ev->phase;
}

// Capture exampleDevHook's callable device address at runtime. Requires
// relocatable device code (-rdc=true) so the __device__ function has a real
// global address that NCCL's kernel can call indirectly across modules.
__global__ void exampleCaptureHookAddr(void** out) { *out = (void*)exampleDevHook; }

static ExampleDevCtx* gDevCtx[EXAMPLE_MAX_DEVICES] = {nullptr};

extern "C" ncclResult_t exampleProfilerGetDeviceHook(int cudaDev, void** devHook, void** devCtx) {
  if (devHook) *devHook = nullptr;
  if (devCtx) *devCtx = nullptr;
  // Opt-in: no device hook unless explicitly enabled.
  if (getenv("NCCL_PROFILER_EXAMPLE_DEV_HOOK") == nullptr) return ncclSuccess;
  if (cudaDev < 0 || cudaDev >= EXAMPLE_MAX_DEVICES) return ncclSuccess;
  if (cudaSetDevice(cudaDev) != cudaSuccess) return ncclSuccess;

  // Resolve the hook's device address (see exampleCaptureHookAddr).
  void** dOut = nullptr;
  if (cudaMalloc(&dOut, sizeof(void*)) != cudaSuccess) return ncclSuccess;
  exampleCaptureHookAddr<<<1, 1>>>(dOut);
  cudaDeviceSynchronize();
  void* fn = nullptr;
  cudaMemcpy(&fn, dOut, sizeof(fn), cudaMemcpyDeviceToHost);
  cudaFree(dOut);
  if (!fn) return ncclSuccess;

  // Allocate this device's telemetry context once.
  if (!gDevCtx[cudaDev]) {
    if (cudaMalloc(&gDevCtx[cudaDev], sizeof(ExampleDevCtx)) != cudaSuccess) return ncclSuccess;
    cudaMemset(gDevCtx[cudaDev], 0, sizeof(ExampleDevCtx));
  }
  *devHook = fn;
  *devCtx = gDevCtx[cudaDev];
  return ncclSuccess;
}

// Print each device's captured counts. Called from the plugin's finalize; a
// no-op unless a device hook was handed out (gDevCtx allocated).
extern "C" void exampleProfilerDrainDeviceHook(void) {
  for (int d = 0; d < EXAMPLE_MAX_DEVICES; d++) {
    if (!gDevCtx[d]) continue;
    ExampleDevCtx h = {};
    cudaSetDevice(d);
    cudaMemcpy(&h, gDevCtx[d], sizeof(h), cudaMemcpyDeviceToHost);
    fprintf(stderr, "PROFILER/DeviceHook: dev=%d events=%llu lastFunc=%s phase=%s\n", d, h.count,
            ncclProfilerDevFuncName((int)h.lastFuncId), h.lastPhase == NCCL_PROFILER_DEV_STOP ? "STOP" : "START");
  }
}
