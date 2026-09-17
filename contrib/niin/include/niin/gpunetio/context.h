/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_GPUNETIO_CONTEXT_H_
#define NIIN_GPUNETIO_CONTEXT_H_

#include <stddef.h>
#include <stdint.h>

// This is deliberately a NIIN-owned ABI.  The host implementation creates
// the QPs, registers the heap and response ring, exchanges endpoint metadata,
// and copies one of these contexts to device memory.  Nothing in this ABI
// relies on a GIN context or a NCCL transport-private object.
#define NIIN_GPUNETIO_ATOMIC_CONTEXT_VERSION 3u

enum niinGpunetioAtomicContextFlags : uint32_t {
  // The host has completed bootstrap and every endpoint below is safe to use.
  NIIN_GPUNETIO_ATOMIC_CONTEXT_READY = 1u << 0,
  // The endpoint QPs may be submitted by GPU code.
  NIIN_GPUNETIO_ATOMIC_CONTEXT_DIRECT = 1u << 1,
  // The endpoint QPs were transitioned with 4/8-byte masked atomic support.
  NIIN_GPUNETIO_ATOMIC_CONTEXT_EXTENDED = 1u << 2,
  // A host/SRQ fallback is attached.  Native code does not dereference it.
  NIIN_GPUNETIO_ATOMIC_CONTEXT_PROXY = 1u << 3,
  // The provider has verified that the CQ completion makes atomic response
  // data visible without a terminal fenced DUMP.  Leave this clear by default
  // for the conservative completion path.
  NIIN_GPUNETIO_ATOMIC_CONTEXT_MAY_SKIP_CST = 1u << 4,
  // The selected HCA reports 8-byte atomic responses in host/device byte
  // order. Without this bit, the direct path converts 8-byte fetch results
  // from network byte order just as it always does for 4-byte results.
  NIIN_GPUNETIO_ATOMIC_CONTEXT_8B_RESULT_HOST_ENDIAN = 1u << 5,
  // Reserved for a future supported GPUNetIO CPU doorbell configuration. This
  // is only a doorbell transport detail (not the future SRQ software-atomic
  // fallback). Current direct-provider setup rejects CPU-doorbell QPs.
  NIIN_GPUNETIO_ATOMIC_CONTEXT_CPU_PROXY_DOORBELL = 1u << 6,
};

enum niinGpunetioAtomicEndpointFlags : uint32_t {
  // `deviceQp` is a connected `doca_gpu_dev_verbs_qp*` for this destination.
  NIIN_GPUNETIO_ATOMIC_ENDPOINT_CONNECTED = 1u << 0,
  NIIN_GPUNETIO_ATOMIC_ENDPOINT_ATOMIC_4B = 1u << 1,
  NIIN_GPUNETIO_ATOMIC_ENDPOINT_ATOMIC_8B = 1u << 2,
  NIIN_GPUNETIO_ATOMIC_ENDPOINT_EXTENDED = 1u << 3,
};

// Bootstrap metadata for one destination PE.  `deviceQp` is intentionally a
// void pointer here so this stable NIIN context stays independent of the
// vendor device-header layout.  The device adapter casts it only after the
// runtime capability checks pass.  Key fields use the byte order required by
// the GPUNetIO device header, not the host ibverbs value. When
// `DOCA_GPUNETIO_VERBS_MKEY_SWAPPED=1`, the host provider stores
// `htobe32(ibv_mr->rkey)` here.
struct niinGpunetioAtomicEndpoint {
  void* deviceQp;
  uint64_t remoteHeapBase;
  size_t remoteHeapBytes;
  uint32_t remoteHeapRkey;
  uint32_t flags;
};

// Device-resident state for NIIN's GPUNetIO atomics provider.
//
// `responseBase` / `responseBytes` identify a GPU allocation registered with
// `responseLkey`.  It is a one-slot-per-destination-PE response ring: slot
// `pe` starts at `responseBase + pe * responseStride`.  The first eight bytes
// of every slot are reserved for the atomic result, so the same context can
// serve both 4- and 8-byte operations.  `endpointLocks[pe]` serializes that
// slot and its dedicated QP; the array is GPU memory, has `nPes` zeroed
// 32-bit entries, and is owned by the provider for the context lifetime.
//
// `cstScratchOffset` is relative to *each* response slot.  It names a
// provider-reserved byte used only by a terminal fenced DUMP on systems that
// cannot skip the consistency step.  It must be inside `responseStride` and
// outside the first eight result bytes.  Keeping it per PE avoids unrelated
// QPs contending on one DUMP address.
//
// `responseLkey` uses the same device-header byte-order contract as
// `remoteHeapRkey`: with the pinned GPUNetIO headers it is
// `htobe32(ibv_mr->lkey)`.  A provider built against a GPUNetIO release with
// `DOCA_GPUNETIO_VERBS_MKEY_SWAPPED=0` must instead preserve the host-order
// ibverbs key, because the device WQE builders perform the conversion there.
// `proxyContext` is owned by the optional NIIN SRQ fallback. The front-end
// dispatcher checks it before attempting a direct WQE, but the direct WQE
// construction code itself never dereferences it.
struct niinGpunetioAtomicContext {
  uint32_t version;
  uint32_t flags;
  int nPes;
  int localPe;
  const struct niinGpunetioAtomicEndpoint* endpoints;
  void* responseBase;
  size_t responseBytes;
  size_t responseStride;
  size_t cstScratchOffset;
  uint32_t responseLkey;
  uint32_t reserved;
  int* endpointLocks;
  void* proxyContext;
};

#endif  // NIIN_GPUNETIO_CONTEXT_H_
