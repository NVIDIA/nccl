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
#include <cstdio>
#include <cstdlib>
#include <climits>

// Host helpers construct and copy the device context but do not need the
// device-side accessor definitions in niin/context.h.  Include the stable ABI
// directly so this header remains self-contained for low-level users.
#include "niin/context_abi.h"

#include "niin/context.h"

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
  void** peerHeapBaseP2p;
  int nodeRank;
  int nodeSize;
  bool peerNativeAtomic;  // Whether peer GPUs support native system-scope atomics
  bool forceSeparatePutSignal; // Force put+fence+signal instead of fused put_signal
  unsigned int* ginPendingOps;  // Device scalar tracking unquieted GIN ops
  unsigned int* lsaStorePending; // Device per-CTA flags for local/LSA stores
  size_t lsaStorePendingLen;    // Entries in lsaStorePending
  // TMA state, owned by niinTmaEnable/niinTmaDisable. niinInit clears these, so
  // callers that manage a niinContext_host themselves get TMA off by default.
  int tmaPolicy;               // nvshmemx_tma_policy_t
  uintptr_t* tmaSmemBases;     // Device array of per-CTA registered smem bases
  size_t tmaSmemBasesLen;      // Entries in tmaSmemBases
  size_t* tmaSmemSize;         // Device scalar holding the given smem size
};

// Default number of GIN contexts to request. Each context is one QP per peer,
// so this is the number of QPs NIIN round-robins operations across. NCCL rounds
// it up to a multiple of the connection count, so the effective value can be
// higher on multi-NIC nodes.
#define NIIN_DEFAULT_NUM_QPS 4

// Read the QP count from NIIN_NUM_QPS. Anything unparseable or < 1 warns and
// falls back to the default.
inline int niinParseNumQps() {
  const char* env = getenv("NIIN_NUM_QPS");
  if (env == nullptr || env[0] == '\0') return NIIN_DEFAULT_NUM_QPS;

  char* end = nullptr;
  const long value = strtol(env, &end, 10);
  if (end == env || *end != '\0' || value < 1 || value > INT_MAX) {
    fprintf(stderr, "NIIN: NIIN_NUM_QPS must be a positive integer\n");
    return NIIN_DEFAULT_NUM_QPS;
  }
  return static_cast<int>(value);
}

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
//   barrierCount    - Number of barrier sessions to request (default 1)
//   ginSignalCount  - Number of GIN signals to request (default 0)
//   ginContextCount - Number of GIN contexts (QPs per peer) to provision. 0
//                     takes the value from NIIN_NUM_QPS, else the default.
//                     Device operations round-robin over them per operation.
//
// Returns ncclSuccess on success.
inline ncclResult_t niinInit(ncclComm_t comm,
                             void* heapBuf,
                             size_t heapSize,
                             niinContext_host* hostCtx,
                             bool enableGin = true,
                             int barrierCount = 1,
                             int ginSignalCount = 0,
                             int ginContextCount = 0,
                             int nodeRank = 0,
                             int nodeSize = 1) {
  // Register the symmetric heap as a window
  // The NVSHMEM heap contains both RMA payloads and signal locations.  GIN
  // requires strict ordering for a window whose remote signal makes a prior
  // payload visible to an independent consumer CTA.
  NIIN_CHECK_NCCL(ncclCommWindowRegister(comm, heapBuf, heapSize,
                                          &hostCtx->heapWindow,
                                          NCCL_WIN_COLL_SYMMETRIC |
                                          NCCL_WIN_STRICT_ORDERING));

  // Create a device communicator with GIN resources
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  if (enableGin) {
    // barrierCount provisions the hybrid LSA+GIN barrier.  NIIN selects the
    // cheaper pure-LSA path when every PE is local, so provision that handle
    // as well.  Without it, a same-node communicator has an empty
    // devComm.lsaBarrier even though the device wrapper selects it.
    reqs.lsaBarrierCount = barrierCount;
    reqs.barrierCount = barrierCount;
    // Device barriers fence every GIN context, which uses the dedicated
    // world-GIN barrier handle rather than the legacy hybrid handle.
    reqs.worldGinBarrierCount = barrierCount;
    reqs.ginForceEnable = true;
    reqs.ginContextCount = ginContextCount > 0 ? ginContextCount : niinParseNumQps();
    reqs.ginSignalCount = ginSignalCount + barrierCount;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  } else {
    reqs.lsaBarrierCount = barrierCount;
    reqs.lsaMultimem = false;
  }

  NIIN_CHECK_NCCL(ncclDevCommCreate(comm, &reqs, &hostCtx->devComm));

  hostCtx->heapBase = heapBuf;
  hostCtx->heapSize = heapSize;
  hostCtx->peerHeapBaseP2p = nullptr;
  hostCtx->nodeRank = nodeRank;
  hostCtx->nodeSize = nodeSize;
  hostCtx->ginPendingOps = nullptr;
  hostCtx->lsaStorePending = nullptr;
  hostCtx->lsaStorePendingLen = 0;

  // TMA is opt-in; niinTmaEnable() turns it on after this call.
  hostCtx->tmaPolicy = NVSHMEMX_TMA_DISABLE;
  hostCtx->tmaSmemBases = nullptr;
  hostCtx->tmaSmemBasesLen = 0;
  hostCtx->tmaSmemSize = nullptr;

  return ncclSuccess;
}

inline void niinPendingStateDisable(niinContext_host* hostCtx) {
  if (hostCtx->ginPendingOps != nullptr) cudaFree(hostCtx->ginPendingOps);
  if (hostCtx->lsaStorePending != nullptr) cudaFree(hostCtx->lsaStorePending);
  hostCtx->ginPendingOps = nullptr;
  hostCtx->lsaStorePending = nullptr;
  hostCtx->lsaStorePendingLen = 0;
}

inline ncclResult_t niinPendingStateEnable(niinContext_host* hostCtx) {
  if (hostCtx->ginPendingOps == nullptr) {
    cudaError_t e = cudaMalloc(&hostCtx->ginPendingOps, sizeof(unsigned int));
    if (e == cudaSuccess) e = cudaMemset(hostCtx->ginPendingOps, 0, sizeof(unsigned int));
    if (e != cudaSuccess) {
      fprintf(stderr, "NIIN: CUDA error %s allocating pending GIN counter at %s:%d\n",
              cudaGetErrorString(e), __FILE__, __LINE__);
      niinPendingStateDisable(hostCtx);
      return ncclInternalError;
    }
  }

  if (hostCtx->lsaStorePending == nullptr) {
    size_t pendingBytes = (size_t)NIIN_TMA_MAX_BLOCKS * sizeof(unsigned int);
    cudaError_t e = cudaMalloc(&hostCtx->lsaStorePending, pendingBytes);
    if (e == cudaSuccess) e = cudaMemset(hostCtx->lsaStorePending, 0, pendingBytes);
    if (e != cudaSuccess) {
      fprintf(stderr, "NIIN: CUDA error %s allocating LSA pending table at %s:%d\n",
              cudaGetErrorString(e), __FILE__, __LINE__);
      niinPendingStateDisable(hostCtx);
      return ncclInternalError;
    }
    hostCtx->lsaStorePendingLen = NIIN_TMA_MAX_BLOCKS;
  }

  return ncclSuccess;
}

// niinTmaDisable: release the tables allocated by niinTmaEnable. Safe to call
// when TMA was never enabled.
inline void niinTmaDisable(niinContext_host* hostCtx) {
  if (hostCtx->tmaSmemBases != nullptr) cudaFree(hostCtx->tmaSmemBases);
  if (hostCtx->tmaSmemSize != nullptr) cudaFree(hostCtx->tmaSmemSize);
  hostCtx->tmaSmemBases = nullptr;
  hostCtx->tmaSmemSize = nullptr;
  hostCtx->tmaSmemBasesLen = 0;
  hostCtx->tmaPolicy = NVSHMEMX_TMA_DISABLE;
}

// niinTmaEnable: allocate the per-CTA TMA registration tables.
//
// Call between niinInit() and niinCommit(). Once enabled, kernels that call
// nvshmemx_give_smem() route their LSA put/get traffic through cp.async.bulk.
// Requires compute capability 9.0 or newer; the caller is responsible for
// checking that (nvshmem_init() does so from NVSHMEM_TMA_POLICY).
//
// Arguments:
//   hostCtx - Host-side staging struct populated by niinInit
//   policy  - NVSHMEMX_TMA_ENABLE or NVSHMEMX_TMA_FORCE; DISABLE is a no-op
//
// Returns ncclSuccess on success; leaves TMA off on failure.
inline ncclResult_t niinTmaEnable(niinContext_host* hostCtx, nvshmemx_tma_policy_t policy) {
  if (policy == NVSHMEMX_TMA_DISABLE) return ncclSuccess;

  // niinInit() left these null; clear them again so a partial allocation below
  // is never mistaken for a live table.
  hostCtx->tmaSmemBases = nullptr;
  hostCtx->tmaSmemSize = nullptr;

  size_t basesBytes = (size_t)NIIN_TMA_MAX_BLOCKS * sizeof(uintptr_t);
  cudaError_t e = cudaMalloc(&hostCtx->tmaSmemBases, basesBytes);
  if (e == cudaSuccess) e = cudaMemset(hostCtx->tmaSmemBases, 0, basesBytes);
  if (e == cudaSuccess) e = cudaMalloc(&hostCtx->tmaSmemSize, sizeof(size_t));
  if (e == cudaSuccess) e = cudaMemset(hostCtx->tmaSmemSize, 0, sizeof(size_t));
  if (e != cudaSuccess) {
    fprintf(stderr, "NIIN: CUDA error %s allocating TMA tables at %s:%d\n",
            cudaGetErrorString(e), __FILE__, __LINE__);
    niinTmaDisable(hostCtx);
    return ncclInternalError;
  }

  hostCtx->tmaSmemBasesLen = NIIN_TMA_MAX_BLOCKS;
  hostCtx->tmaPolicy = policy;
  return ncclSuccess;
}

inline ncclResult_t niinPublishContextCache(const niinContext* ctx) {
#if NIIN_USE_DEVICE_CONTEXT_POINTER
  (void)ctx;
  return ncclSuccess;
#else
  NIIN_CHECK_CUDA(cudaMemcpyToSymbol(niin_g_ctx_constant, ctx, sizeof(niinContext),
                                     0, cudaMemcpyHostToDevice));
  return ncclSuccess;
#endif
}

inline ncclResult_t niinRefreshContextCache(niinContext* devCtx) {
#if NIIN_USE_DEVICE_CONTEXT_POINTER
  (void)devCtx;
  return ncclSuccess;
#else
  niinContext ctx;
  NIIN_CHECK_CUDA(cudaMemcpy(&ctx, devCtx, sizeof(niinContext), cudaMemcpyDeviceToHost));
  return niinPublishContextCache(&ctx);
#endif
}

inline ncclResult_t niinBuildPeerHeapBaseP2p(niinContext_host* hostCtx) {
  const int nRanks = hostCtx->devComm.nRanks;
  if (nRanks <= 0) return ncclSuccess;

  void** peerHeapBaseP2pHost = static_cast<void**>(calloc(static_cast<size_t>(nRanks), sizeof(void*)));
  if (peerHeapBaseP2pHost == nullptr) return ncclInternalError;

  ncclResult_t result = ncclSuccess;
  for (int pe = 0; pe < nRanks; pe++) {
    result = ncclGetPeerDevicePointer(hostCtx->heapWindow, 0, pe, &peerHeapBaseP2pHost[pe]);
    if (result != ncclSuccess) break;
  }

  void** peerHeapBaseP2p = nullptr;
  if (result == ncclSuccess) {
    cudaError_t e = cudaMalloc(&peerHeapBaseP2p, static_cast<size_t>(nRanks) * sizeof(void*));
    if (e == cudaSuccess) {
      e = cudaMemcpy(peerHeapBaseP2p, peerHeapBaseP2pHost,
                     static_cast<size_t>(nRanks) * sizeof(void*),
                     cudaMemcpyHostToDevice);
    }
    if (e != cudaSuccess) {
      fprintf(stderr, "NIIN: CUDA error %s building peer heap table at %s:%d\n",
              cudaGetErrorString(e), __FILE__, __LINE__);
      if (peerHeapBaseP2p != nullptr) cudaFree(peerHeapBaseP2p);
      result = ncclInternalError;
    }
  }

  free(peerHeapBaseP2pHost);
  if (result != ncclSuccess) return result;
  hostCtx->peerHeapBaseP2p = peerHeapBaseP2p;
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
inline ncclResult_t niinCommit(niinContext_host* hostCtx,
                               niinContext* devCtx) {
  NIIN_CHECK_NCCL(niinPendingStateEnable(hostCtx));
  NIIN_CHECK_NCCL(niinBuildPeerHeapBaseP2p(hostCtx));

  // Copy devComm to device memory
  ncclDevComm* d_devComm;
  NIIN_CHECK_CUDA(cudaMalloc(&d_devComm, sizeof(ncclDevComm)));
  NIIN_CHECK_CUDA(cudaMemcpy(d_devComm, &hostCtx->devComm, sizeof(ncclDevComm),
                              cudaMemcpyHostToDevice));

  // Build and copy niinContext to device
  niinContext ctx;
  ctx.comm = d_devComm;
  ctx.commValue = hostCtx->devComm;
  ctx.heapWindow = hostCtx->heapWindow;
  ctx.heapBase = hostCtx->heapBase;
  ctx.heapSize = hostCtx->heapSize;
  ctx.peerHeapBaseP2p = hostCtx->peerHeapBaseP2p;
  ctx.rank = hostCtx->devComm.rank;
  ctx.nRanks = hostCtx->devComm.nRanks;
  ctx.lsaRank = hostCtx->devComm.lsaRank;
  ctx.lsaSize = hostCtx->devComm.lsaSize;
  ctx.ginConnectionCount = hostCtx->devComm.ginConnectionCount;
  ctx.worldIsLsaOnly = hostCtx->devComm.lsaSize == hostCtx->devComm.nRanks;
  ctx.nodeRank = hostCtx->nodeRank;
  ctx.nodeSize = hostCtx->nodeSize;
  ctx.gpunetioAtomicContext = nullptr;
  ctx.peerNativeAtomic = hostCtx->peerNativeAtomic;
  ctx.forceSeparatePutSignal = hostCtx->forceSeparatePutSignal;
  ctx.ginPendingOps = hostCtx->ginPendingOps;
  ctx.lsaStorePending = hostCtx->lsaStorePending;
  ctx.lsaStorePendingLen = hostCtx->lsaStorePendingLen;
  ctx.tmaPolicy = hostCtx->tmaPolicy;
  ctx.tmaSmemBases = hostCtx->tmaSmemBases;
  ctx.tmaSmemBasesLen = hostCtx->tmaSmemBasesLen;
  ctx.tmaSmemSize = hostCtx->tmaSmemSize;

  NIIN_CHECK_CUDA(cudaMemcpy(devCtx, &ctx, sizeof(niinContext), cudaMemcpyHostToDevice));
  NIIN_CHECK_NCCL(niinPublishContextCache(&ctx));

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
  cudaFree(ctx.peerHeapBaseP2p);
  cudaFree(ctx.ginPendingOps);
  cudaFree(ctx.lsaStorePending);

  // Deregister the window (best-effort)
  ncclCommWindowDeregister(comm, ctx.heapWindow);

  return ncclSuccess;
}

#endif // NIIN_HOST_H_
