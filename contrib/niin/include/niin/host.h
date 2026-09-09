/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_HOST_H_
#define NIIN_HOST_H_

#include <nccl.h>
#include <nccl_device.h>
#include <cuda_runtime.h>

#ifndef NIIN_CHECK_NCCL
#define NIIN_CHECK_NCCL(cmd) do {                                             \
  ncclResult_t r = (cmd);                                                     \
  if (r != ncclSuccess) {                                                     \
    fprintf(stderr, "NIIN: NCCL error %s at %s:%d\n",                        \
            ncclGetErrorString(r), __FILE__, __LINE__);                       \
    return r;                                                                  \
  }                                                                            \
} while(0)
#endif

#ifndef NIIN_CHECK_CUDA
#define NIIN_CHECK_CUDA(cmd) do {                                             \
  cudaError_t e = (cmd);                                                      \
  if (e != cudaSuccess) {                                                     \
    fprintf(stderr, "NIIN: CUDA error %s at %s:%d\n",                        \
            cudaGetErrorString(e), __FILE__, __LINE__);                       \
    return ncclInternalError;                                                  \
  }                                                                            \
} while(0)
#endif

// niinContext_host: host-side staging struct for two-phase init.
// Phase 1 (niinInit) performs NCCL collective calls and stores results here.
// Phase 2 (niinCommit) copies results to device memory after ncclGroupEnd().
struct niinContext_host {
  ncclDevComm devComm;
  ncclWindow_t heapWindow;
  void* heapBase;
  size_t heapSize;
  int ginContextIndex;
  bool peerNativeAtomic;  // Whether peer GPUs support native system-scope atomics
  bool forceSeparatePutSignal; // Force put+fence+signal instead of fused put_signal
};

// niinInit: host-side initialization — phase 1 (NCCL collective calls).
//
// Registers heapBuf as a symmetric window and creates a device communicator
// with GIN resources. Results are stored in hostCtx. This function may be
// called inside ncclGroupStart/End. After ncclGroupEnd(), call niinCommit()
// to copy the context to device memory.
//
// Arguments:
//   comm       - NCCL communicator
//   heapBuf    - Device memory buffer to use as symmetric heap (must be
//                aligned to NCCL_WIN_REQUIRED_ALIGNMENT and same size on all ranks)
//   heapSize   - Size of heapBuf in bytes
//   hostCtx    - [OUT] Host-side staging struct to populate
//   ginContextIndex - Which GIN context index to use (default 0)
//   barrierCount    - Number of barrier sessions to request (default 1)
//   ginSignalCount  - Number of GIN signals to request (default 0)
//
// Returns ncclSuccess on success.
inline ncclResult_t niinInit(ncclComm_t comm,
                             void* heapBuf,
                             size_t heapSize,
                             niinContext_host* hostCtx,
                             bool enableGin = true,
                             int ginContextIndex = 0,
                             int barrierCount = 1,
                             int ginSignalCount = 0) {
  // Register the symmetric heap as a window
  NIIN_CHECK_NCCL(ncclCommWindowRegister(comm, heapBuf, heapSize,
                                          &hostCtx->heapWindow,
                                          NCCL_WIN_COLL_SYMMETRIC));

  // Create a device communicator with GIN resources
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  if (enableGin) {
    reqs.barrierCount = barrierCount;
    reqs.ginForceEnable = true;
    reqs.ginContextCount = 1;
    reqs.ginSignalCount = ginSignalCount + barrierCount;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  } else {
    reqs.lsaBarrierCount = barrierCount;
    reqs.lsaMultimem = false;
  }

  NIIN_CHECK_NCCL(ncclDevCommCreate(comm, &reqs, &hostCtx->devComm));

  hostCtx->heapBase = heapBuf;
  hostCtx->heapSize = heapSize;
  hostCtx->ginContextIndex = ginContextIndex;

  return ncclSuccess;
}

// niinCommit: host-side initialization — phase 2 (copy to device).
//
// Must be called AFTER ncclGroupEnd() so that NCCL collective results are
// finalized. Copies the device communicator and context to device memory.
//
// Arguments:
//   hostCtx - Host-side staging struct populated by niinInit
//   devCtx  - [OUT] Device pointer to niinContext (must be cudaMalloc'd)
//
// Returns ncclSuccess on success.
inline ncclResult_t niinCommit(const niinContext_host* hostCtx,
                               niinContext* devCtx) {
  // Copy devComm to device memory
  ncclDevComm* d_devComm;
  NIIN_CHECK_CUDA(cudaMalloc(&d_devComm, sizeof(ncclDevComm)));
  NIIN_CHECK_CUDA(cudaMemcpy(d_devComm, &hostCtx->devComm, sizeof(ncclDevComm),
                              cudaMemcpyHostToDevice));

  // Build and copy niinContext to device
  niinContext ctx;
  ctx.comm = d_devComm;
  ctx.heapWindow = hostCtx->heapWindow;
  ctx.heapBase = hostCtx->heapBase;
  ctx.heapSize = hostCtx->heapSize;
  ctx.ginContextIndex = hostCtx->ginContextIndex;
  ctx.peerNativeAtomic = hostCtx->peerNativeAtomic;
  ctx.forceSeparatePutSignal = hostCtx->forceSeparatePutSignal;

  NIIN_CHECK_CUDA(cudaMemcpy(devCtx, &ctx, sizeof(niinContext), cudaMemcpyHostToDevice));

  return ncclSuccess;
}

// niinFinalize: host-side cleanup.
//
// Deregisters the window and destroys the device communicator.
// May be called inside ncclGroupStart/End for the NCCL calls.
//
// Arguments:
//   comm   - NCCL communicator
//   devCtx - Device pointer to the niinContext (will be read back to host)
//
// Returns ncclSuccess on success.
inline ncclResult_t niinFinalize(ncclComm_t comm, niinContext* devCtx) {
  // Read context back from device
  niinContext ctx;
  cudaError_t ce = cudaMemcpy(&ctx, devCtx, sizeof(niinContext), cudaMemcpyDeviceToHost);
  if (ce != cudaSuccess) return ncclInternalError;

  // Read back the devComm so we can pass it to destroy
  ncclDevComm devComm;
  ce = cudaMemcpy(&devComm, (void*)ctx.comm, sizeof(ncclDevComm), cudaMemcpyDeviceToHost);
  if (ce != cudaSuccess) return ncclInternalError;

  // Destroy device communicator (best-effort — may fail during teardown)
  ncclDevCommDestroy(comm, &devComm);

  // Free the device-side devComm copy
  cudaFree((void*)ctx.comm);

  // Deregister the window (best-effort)
  ncclCommWindowDeregister(comm, ctx.heapWindow);

  return ncclSuccess;
}

#endif // NIIN_HOST_H_
