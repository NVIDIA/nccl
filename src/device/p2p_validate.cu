/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "checks.h"
#include "p2p.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr size_t kP2pValidateWords = 64; // 256 bytes; matches issue #2335 discriminator size class

__global__ void p2pValidateWriteKernel(uint32_t* dst, size_t nWords) {
  for (size_t i = threadIdx.x; i < nWords; i += blockDim.x) {
    dst[i] = 0xC0FFEE00u ^ (0x9E3779B9u * (uint32_t)i);
  }
  // Match the #2335 reproducer: publish the store before the owner may observe it.
  __threadfence_system();
}

static cudaError_t enablePeerAccess(int peerDev) {
  cudaError_t err = cudaDeviceEnablePeerAccess(peerDev, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
    return cudaSuccess;
  }
  return err;
}

} // namespace

/* Device-side peer store + owner-local read. Requires cudaDeviceEnablePeerAccess so the
 * write uses the same direct peer aperture NCCL SM protocols use — not a host-staged
 * cudaMemcpyPeer fallback path.
 */
ncclResult_t ncclP2pValidatePeerMapping(int cudaDevFrom, int cudaDevTo, int* valid) {
  *valid = 1;
  if (cudaDevFrom == cudaDevTo) return ncclSuccess;

  int prevDev = 0;
  CUDACHECK(cudaGetDevice(&prevDev));

  uint32_t hostPat[kP2pValidateWords];
  uint32_t hostOut[kP2pValidateWords];
  for (size_t i = 0; i < kP2pValidateWords; i++) {
    hostPat[i] = 0xC0FFEE00u ^ (0x9E3779B9u * (uint32_t)i);
  }
  memset(hostOut, 0, sizeof(hostOut));

  uint32_t* dst = nullptr;
  size_t const nBytes = kP2pValidateWords * sizeof(uint32_t);

  // Any probe failure disables P2P for the pair; never fail communicator init.
  do {
    if (!CUDASUCCESS(cudaSetDevice(cudaDevTo))) { *valid = 0; break; }
    if (!CUDASUCCESS(cudaMalloc(&dst, nBytes))) { *valid = 0; break; }
    if (!CUDASUCCESS(cudaMemset(dst, 0, nBytes))) { *valid = 0; break; }

    if (!CUDASUCCESS(cudaSetDevice(cudaDevFrom))) { *valid = 0; break; }
    if (enablePeerAccess(cudaDevTo) != cudaSuccess) {
      INFO(NCCL_INIT | NCCL_P2P, "P2P validate enablePeerAccess failed between dev %d and %d",
           cudaDevFrom, cudaDevTo);
      *valid = 0;
      break;
    }

    // SM store through the peer mapping (same visibility contract NCCL P2P protocols need).
    p2pValidateWriteKernel<<<1, 64>>>(dst, kP2pValidateWords);
    if (!CUDASUCCESS(cudaGetLastError())) { *valid = 0; break; }
    if (!CUDASUCCESS(cudaDeviceSynchronize())) { *valid = 0; break; }

    // Owner-local read — do not read back through the peer window (mirrors look correct that way).
    if (!CUDASUCCESS(cudaSetDevice(cudaDevTo))) { *valid = 0; break; }
    if (!CUDASUCCESS(cudaDeviceSynchronize())) { *valid = 0; break; }
    if (!CUDASUCCESS(cudaMemcpy(hostOut, dst, nBytes, cudaMemcpyDeviceToHost))) { *valid = 0; break; }

    *valid = (memcmp(hostPat, hostOut, nBytes) == 0) ? 1 : 0;
    if (*valid == 0) {
      INFO(NCCL_INIT | NCCL_P2P,
           "P2P validate failed between dev %d and %d: peer write not visible to owner "
           "(possible private peer-aperture mirror). Disabling P2P for this pair.",
           cudaDevFrom, cudaDevTo);
    }
  } while (0);

  if (dst) {
    (void)cudaSetDevice(cudaDevTo);
    (void)cudaFree(dst);
  }
  (void)cudaSetDevice(prevDev);
  (void)cudaGetLastError(); // clear sticky errors from the probe
  return ncclSuccess;
}
