/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// NIIN: NVSHMEM Implemented In NCCL
//
// Drop-in replacement for nvshmem.h. Include this instead of the real
// nvshmem.h to use NCCL's communication infrastructure (LSA, GIN, barriers)
// with the familiar NVSHMEM device API.
//
// Usage (NVSHMEM-compatible):
//   1. Host: nvshmem_init() or nvshmemx_init_attr()
//   2. Host: nvshmem_malloc() to allocate symmetric memory
//   3. Launch kernels: pass niin_get_device_ctx() as kernel arg
//   4. Kernel entry: niin_set_context(ctx)
//   5. All threads: use nvshmem_* functions as normal
//   6. Host: nvshmem_finalize()
//
// Unsupported operations (scalar network gets, most collectives, network
// atomics) invoke NIIN_NOT_IMPLEMENTED behavior, controlled by
// NIIN_ON_NOT_IMPLEMENTED.

#ifndef NVSHMEM_H_NIIN_
#define NVSHMEM_H_NIIN_

// NCCL device headers -- provides ncclDevComm, ncclWindow_t, ncclGin, etc.
// nccl_device.h is the umbrella header that includes all impl types and funcs.
#include <nccl_device.h>

// NIIN internal headers
#include "niin/config.h"
#include "niin/types.h"
#include "niin/context.h"
#include "niin/query.h"
#include "niin/rma.h"
#include "niin/signaling.h"
#include "niin/sync.h"
#include "niin/atomics.h"
#include "niin/collectives.h"

// Host-side helpers (niinInit, niinCommit, niinFinalize — low-level API)
#include "niin/host.h"

// Team management (split, translate, destroy)
#include "niin/teams.h"

// NVSHMEM-compatible host API (nvshmem_init, nvshmem_malloc, etc.)
#include "niin/nvshmem_host.h"

#endif // NVSHMEM_H_NIIN_
