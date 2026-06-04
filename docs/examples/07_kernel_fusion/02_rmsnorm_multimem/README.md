<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Fused RMSNorm with Multimem

**Multimem RMSNorm Implementation**

This example demonstrates fused computation and communication using NCCL's Multimem capability. Multimem leverages hardware multicast memory operations (SM 9.0+) to perform reduce-scatter and gather with a single pointer, simplifying the kernel logic compared to explicit peer iteration.

See the [category README](../README.md) for the shared background on distributed RMSNorm (the reduce-scatter → RMSNorm → all-gather pattern), the host program setup flow, and prerequisites.

## Overview

Implement the fused distributed RMSNorm using hardware multicast (multimem) memory operations, so a single multimem pointer reduces from and broadcasts to all LSA peers — simplifying the kernel relative to the explicit peer iteration of the pure LSA example.

## Target Architecture / Topology

Single-node / single NVLink domain (Hopper+). Requires GPU compute capability 9.0 or higher (Hopper H100+).

## Key Features

- Uses `ncclGetLsaMultimemPointer` for multicast memory access
- `multimemLoadSum()` for Phase 1: hardware-accelerated sum across all LSA peers
- `multimemStore()` for Phase 3: broadcast normalized results to all peers
- Each rank processes only its own tokens (reduce-scatter semantics)
- Requires GPU compute capability 9.0 or higher (Hopper H100+)

## What This Example Does

The kernel fuses the three logical phases described in the [category README](../README.md):

1. **Phase 1 (Reduce-Scatter via Multimem)**: each block loads and sums its token's data from all peers with a single multimem pointer, storing the reduced result locally
2. **Phase 2 (RMSNorm)**: identical to the LSA example — `blockRMSNorm()` on the local reduced data
3. **Phase 3 (All-Gather via Multimem)**: each block broadcasts its normalized token to all peers with `multimemStore`

## Device Communicator Requirements

| Requirement | Value | Purpose |
|-------------|-------|---------|
| `lsaBarrierCount` | `tokens_per_gpu` | One LSA barrier per block |
| `lsaMultimem` | `true` | Enables Multimem for `ncclGetLsaMultimemPointer` and PTX multimem instructions (requires SM 9.0+) |

The host checks GPU compute capability (SM 9.0+) before creating the communicator, then verifies multimem support via `ncclCommQueryProperties` (`props.multimemSupport`). Requires a single LSA team (`nLsaTeams == 1`) and Device API support.

## Variants

- **C/CUDA**: `c/` (uses `ncclGetLsaMultimemPointer` + `multimemLoadSum`/`multimemStore`)
  - How to run + walkthrough: `c/README.md`

Note: This example is C/CUDA-only as it demonstrates device-side (kernel) communication, which is not exposed by the nccl4py host-side API.

## Expected Output

You should see:

- GPU compute-capability (SM 9.0+) and multimem capability checks per rank
- The fused RMSNorm kernel launching across `tokens_per_gpu` blocks
- Verification of the GPU output against a CPU reference, with a pass/fail result

## Performance Characteristics

**Performance considerations:**
- **Best Performance**: On Hopper+ GPUs with NVLink/NVSwitch
- **Memory Access**: Hardware multicast reduce and store
- **Scalability**: Limited to single-node / single NVLink domain
