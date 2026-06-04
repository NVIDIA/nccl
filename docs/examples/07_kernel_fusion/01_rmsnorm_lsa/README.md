<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Fused RMSNorm with LSA

**Pure Load Store Accessible (LSA) RMSNorm Implementation**

This example demonstrates fused computation and communication using NCCL's Load Store Accessible (LSA) mechanism. LSA enables GPUs to directly access memory on peer GPUs within a single-node / single NVLink domain using high-bandwidth NVLink connections.

See the [category README](../README.md) for the shared background on distributed RMSNorm (the reduce-scatter → RMSNorm → all-gather pattern), the host program setup flow, and prerequisites.

## Overview

Implement the complete distributed RMSNorm operation — reduce-scatter, local RMSNorm, then all-gather — fused inside a single GPU kernel, using LSA for direct peer memory access instead of separate host-launched collectives.

## Target Architecture / Topology

Single-node / single NVLink domain.

## Key Features

- Uses `ncclLsaBarrierSession` for synchronization
- Leverages `ncclGetLsaPointer` for direct peer memory access

## What This Example Does

The kernel fuses the three logical phases described in the [category README](../README.md):

1. **Phase 1 (Reduce-Scatter via LSA)**: each block owns one token and accumulates that token's contributions directly from all peers' windows
2. **Phase 2 (RMSNorm)**: each block normalizes its assigned token with the shared `blockRMSNorm()` helper
3. **Phase 3 (All-Gather via LSA)**: each block writes the normalized token back to all peer GPUs

## Device Communicator Requirements

| Requirement | Value | Purpose |
|-------------|-------|---------|
| `lsaBarrierCount` | `tokens_per_gpu` | One LSA barrier per block; synchronizes across LSA peers between phases |

Requires a single LSA team (`nLsaTeams == 1`) and Device API support (verified via `ncclCommQueryProperties`).

## Variants

- **C/CUDA**: `c/` (uses `ncclGetLsaPointer` + `ncclLsaBarrierSession`)
  - How to run + walkthrough: `c/README.md`

Note: This example is C/CUDA-only as it demonstrates device-side (kernel) communication, which is not exposed by the nccl4py host-side API.

## Expected Output

You should see:

- Device API and LSA topology capability checks per rank
- The fused RMSNorm kernel launching across `tokens_per_gpu` blocks
- Verification of the GPU output against a CPU reference, with a pass/fail result

## Performance Characteristics

**Advantages of LSA:**
- **High Bandwidth**: Utilizes full NVLink bandwidth
- **No Intermediate Copies**: Data moves directly from peer memory to registers

**Block-level Parallelism:**
- Each block operates independently on one token
- `tokens_per_gpu` blocks execute concurrently (limited by GPU resources)
- Barriers synchronize only when necessary for correctness

**Performance considerations:**
- **Best Performance**: On systems with NVLink/NVSwitch
- **Memory Access**: Direct peer-to-peer with low latency
- **Scalability**: Limited to single-node / single NVLink domain
