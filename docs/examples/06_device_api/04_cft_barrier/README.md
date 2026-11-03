<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Device API CFT barrier

This example shows how to create and use a CFT barrier session.

## Overview

Large multi-node NVLink domains need the Compute Fabric Transport (CFT) programming model
to share and access peer GPU memory across the NCCL ranks. NCCL exposes this new
programming model through appropriate CFT handles and CFT synchronization primitives.
This example shows users how resources for CFT barriers are created and used.

## Runtime Requirements

The C variant can be built with either pthreads or MPI. It can run on a single node but is most relevant when local and remote peers are both present (multiple nodes with multiple GPUs per node). It is most useful with multiple GPUs, but can still run with one visible GPU.

The CFT programming model requires CUDA TK 13.3 and SM\_100 architectures.

## What This Example Does

1. **Request CFT capabilities** with `cftCaps = NCCL_CFT | NCCL_CFT_MULTIMEM`
2. **Request barriers** with `cftBarrierCount = NCCL_DEVICE_CTA_COUNT`, one per CTA
3. **Create device communicator** to allocate the requested CFT resources
4. **Launch GPU kernels** that perform:
   - Create a barrier session using the device communicator and `blockIdx.x` as its barrier index
   - Set `multimem=true` for the multicast variant
   - Use the barrier session to synchronize ranks
5. **Report kernel completion** and clean up resources

## Key concepts
