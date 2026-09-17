/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_GPUNETIO_HOST_H_
#define NIIN_GPUNETIO_HOST_H_

#include <stddef.h>
#include <stdint.h>

#include <nccl.h>

// The optional SRQ/GDRCopy proxy-all provider has a separate implementation
// and runtime dependency.  Expose its host API only when the CMake target was
// built with that implementation; otherwise these declarations would promise
// symbols that the direct-only archive deliberately does not contain.
#if defined(NIIN_GPUNETIO_SRQ_PROXY_ATOMICS_ENABLE) && \
    NIIN_GPUNETIO_SRQ_PROXY_ATOMICS_ENABLE
#include "niin/gpunetio/proxy_atomics/host.h"
#endif

struct niinContext;
struct niinGpunetioAtomicHostContext;

// NIIN brings the provider up itself during nvshmem_init(), so these entry
// points are declared weak: an application that does not link the provider
// archive still links, and NIIN sees a null symbol and leaves network AMOs
// fail-closed exactly as before.
#ifndef NIIN_GPUNETIO_WEAK
#define NIIN_GPUNETIO_WEAK __attribute__((weak))
#endif

// The host provider deliberately has a small, independent configuration
// surface.  It must not select, inspect, or modify a GIN context: existing
// NIIN RMA and signaling continue to use GIN exactly as before.
enum niinGpunetioAtomicNicHandler : int {
  // Let GPUNetIO select GPU doorbells where available. Initialization fails
  // collectively if this provider revision instead selects CPU proxy
  // doorbells, which the direct atomic path has not validated.
  NIIN_GPUNETIO_ATOMIC_NIC_HANDLER_AUTO = 0,
  // Require GPU SM doorbells. Initialization fails rather than falling back.
  NIIN_GPUNETIO_ATOMIC_NIC_HANDLER_GPU_SM_DB = 1,
  // Request GPUNetIO's CPU doorbell proxy. This is currently rejected during
  // setup rather than permitting a potentially non-terminating device AMO.
  NIIN_GPUNETIO_ATOMIC_NIC_HANDLER_CPU_PROXY = 2,
};

struct niinGpunetioAtomicOptions {
  // HCA name, for example "mlx5_0". nullptr selects NIIN_GPUNETIO_IB_DEV,
  // then the first active verbs device.
  const char* ibDevice;
  // One-based verbs port. Zero selects the first active port on the HCA.
  int ibPort;
  // GID index used for GRH/RoCE addressing. -1 uses
  // NIIN_GPUNETIO_GID_INDEX, then index 0.
  int gidIndex;
  // Local PKey table index for the provider's private RC QPs. The caller is
  // responsible for selecting matching PKeys on every PE; zero is the
  // conventional full-membership/default table entry.
  uint16_t pkeyIndex;
  // Service level for the private RC QPs (0-15). Zero is the default fabric
  // service level.
  uint8_t serviceLevel;
  uint8_t reserved0;
  // GPUNetIO SQ depth per destination PE. It must be a power of two.
  uint32_t sqDepth;
  enum niinGpunetioAtomicNicHandler nicHandler;
};

#define NIIN_GPUNETIO_ATOMIC_OPTIONS_INITIALIZER \
  { nullptr, 0, -1, 0, 0, 0, 256, NIIN_GPUNETIO_ATOMIC_NIC_HANDLER_AUTO }

// Create a NIIN-owned, atomic-only GPUNetIO provider. Every PE in `comm`
// must call this routine once, outside ncclGroupStart/End. The implementation
// performs a public NCCL all-gather only to exchange this provider's QP and MR
// metadata; it does not use GIN for that exchange or for data movement.
//
// `heapBase`/`heapBytes` must name NIIN's existing symmetric heap. NIIN
// registers it directly with this provider only for remote atomic access; no
// GPUNetIO put/get/RMA operation is exposed or implemented here.
NIIN_GPUNETIO_WEAK ncclResult_t niinGpunetioAtomicInit(
    ncclComm_t comm, void* heapBase, size_t heapBytes,
    const struct niinGpunetioAtomicOptions* options,
    struct niinGpunetioAtomicHostContext** out);

// Bind a ready provider to a context that has already been initialized by
// niinCommit(). This copies only NIIN's private atomic context pointer into
// `deviceContext`; it leaves its ncclDevComm, GIN context, window, and all
// existing RMA/signaling state untouched.
NIIN_GPUNETIO_WEAK ncclResult_t niinGpunetioAtomicBind(struct niinGpunetioAtomicHostContext* provider,
                                    struct niinContext* deviceContext);

// Make one bounded forward-progress pass for a future supported GPUNetIO CPU
// doorbell configuration. Current direct-provider initialization rejects that
// configuration, so this is retained only as a diagnostics/future-extension
// hook.
NIIN_GPUNETIO_WEAK ncclResult_t niinGpunetioAtomicProgress(struct niinGpunetioAtomicHostContext* provider);

// Destroy the NIIN-owned QPs, private registrations, response ring, and
// device export. Call after all kernels using the provider have completed and
// before niinFinalize() destroys the device context / NCCL communicator.
NIIN_GPUNETIO_WEAK ncclResult_t niinGpunetioAtomicFinalize(struct niinGpunetioAtomicHostContext* provider);

#endif  // NIIN_GPUNETIO_HOST_H_
