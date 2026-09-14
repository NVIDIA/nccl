/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_CONTEXT_ABI_H_
#define NIIN_CONTEXT_ABI_H_

// Keep the layout consumed by host-side binders separate from device helper
// functions. The direct GPUNetIO atomic provider needs this ABI to update its
// one optional pointer, but must not pull in GIN/LSA helper implementations or
// NCCL transport internals.
#include <nccl_device.h>

#include "niin/gpunetio/context.h"

struct niinContext {
  ncclDevComm const* comm;       // NCCL device communicator
  ncclWindow_t heapWindow;       // The single symmetric heap window
  void* heapBase;                // Local base pointer of the heap
  size_t heapSize;               // Size of the symmetric heap
  int nodeRank;                  // Rank among PEs on this physical node
  int nodeSize;                  // Number of PEs on this physical node
  // Optional, NIIN-owned AMO provider (native GPUNetIO direct WQEs or the
  // separate raw-verbs/SRQ proxy-all service). This stays separate from the GIN
  // contexts NIIN rotates through: RMA and signaling always continue on GIN.
  const struct niinGpunetioAtomicContext* gpunetioAtomicContext;
  bool peerNativeAtomic;         // True if peer GPUs support native system-scope atomics
  bool forceSeparatePutSignal;   // Force put+fence+signal instead of fused put_signal
};

#endif  // NIIN_CONTEXT_ABI_H_
