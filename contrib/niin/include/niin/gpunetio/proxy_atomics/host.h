/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_GPUNETIO_PROXY_ATOMICS_HOST_H_
#define NIIN_GPUNETIO_PROXY_ATOMICS_HOST_H_

#include "niin/gpunetio/proxy_atomics/protocol.h"

#include <cstddef>
#include <cstdint>

#include <nccl.h>

struct niinContext;
struct niinGpunetioProxyAtomicHostContext;

// Configuration for NIIN's experimental raw-verbs/SRQ proxy. Its current
// ProxyAll protocol is a development prototype, not a supported replacement
// for the native direct provider: CPU BAR RMWs cannot be transparently ordered
// with a concurrently running target GPU kernel. Native GPUNetIO QP atomics
// are initialized separately with niinGpunetioAtomicInit().
struct niinGpunetioProxyAtomicOptions {
  enum niinGpunetioProxyAtomicExecutionMode mode;
  const char* ibDevice;
  int ibPort;
  int gidIndex;
  uint16_t pkeyIndex;
  uint8_t serviceLevel;
  uint8_t reserved0;
  // The first implementation leases a single slot per destination. Keep this
  // field explicit so an application cannot silently assume a ticketed ring;
  // values other than one are rejected until the allocator is implemented.
  uint32_t requestSlotsPerPe;
  uint32_t srqDepth;
  uint32_t sendQueueDepth;
  // Optional dlopen path for libgdrapi. nullptr tries libgdrapi.so.2 then
  // libgdrapi.so, retaining no link-time GDRCopy dependency for normal NIIN.
  const char* gdrCopyLibrary;
};

#define NIIN_GPUNETIO_PROXY_ATOMIC_OPTIONS_INITIALIZER                         \
  {                                                                             \
    NIIN_GPUNETIO_PROXY_ATOMIC_EXECUTION_PROXY_ALL, nullptr, 0, -1, 0, 0, 0, \
        1, 256, 256, nullptr                                                    \
  }

// Collectively create the NIIN-only proxy plane:
//
// - a GDRCopy CPU mapping of the existing symmetric heap on each PE;
// - mapped host-memory request/completion slots visible to GPU and CPU;
// - one raw ibverbs RC SEND QP per remote PE plus a target-owned SRQ;
// - a NIIN host progress thread that serializes target RMW operations.
//
// The setup uses public NCCL only for bootstrap metadata. It does not inspect,
// borrow, or modify GIN state, GPUNetIO objects, or NCCL-core resources.
ncclResult_t niinGpunetioProxyAtomicInit(
    ncclComm_t comm, void* heapBase, size_t heapBytes,
    const struct niinGpunetioProxyAtomicOptions* options,
    struct niinGpunetioProxyAtomicHostContext** out);

// Bind the proxy-all device context after niinCommit(). A context can be bound
// to either this proxy provider or the native direct provider, not both.
ncclResult_t niinGpunetioProxyAtomicBind(struct niinGpunetioProxyAtomicHostContext* provider,
                                         struct niinContext* deviceContext);

// Make bounded progress when the application owns host progress scheduling.
// Init also starts one NIIN-owned progress thread, so callers normally do not
// need this routine; it is useful for diagnostics and test harnesses.
ncclResult_t niinGpunetioProxyAtomicProgress(struct niinGpunetioProxyAtomicHostContext* provider);

// Collectively destroy the proxy only after all PEs have completed every
// kernel that can issue AMOs. Finalize performs a private-NCCL quiesce before
// releasing RC QPs, the SRQ, and GDRCopy mappings; every PE in the communicator
// must call it in the same lifecycle phase.
ncclResult_t niinGpunetioProxyAtomicFinalize(struct niinGpunetioProxyAtomicHostContext* provider);

#endif  // NIIN_GPUNETIO_PROXY_ATOMICS_HOST_H_
