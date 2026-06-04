<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Symmetric Memory AllGather with Copy Engine

This example demonstrates how to configure NCCL to use the GPU's Copy Engine
instead of SMs for collective operations. By offloading communication to the
Copy Engine, GPU compute resources remain fully available for application
kernels, enabling true overlap of computation and communication.

## Overview

Symmetric memory windows provide a way to register memory buffers that benefit
from optimized collective operations. When all ranks provide symmetric buffers,
NCCL can apply optimized communication patterns. Configuring the copy engine
(`CTAPolicy=2`) makes those collectives use zero SMs, freeing GPU compute
resources for other work.

## Runtime Requirements

The C variant can be built with either pthreads or MPI. Copy-engine benefits apply inside a single (MN)NVL domain. It is most useful with multiple GPUs, but can still run with one visible GPU.

## What This Example Does

1. **Configure the copy engine** by setting `CTAPolicy=2` so the collective uses zero SMs
2. **Allocate memory** with NCCL's allocator (required for symmetric windows)
3. **Register symmetric windows** with the communicator (collective call - all ranks participate)
4. **Perform AllGather** using the symmetric buffers with the copy engine
5. **Deregister and free** windows and buffers in the correct cleanup order

## Variants

- **C**: `c/` (uses `ncclCommInitRankConfig` + `ncclMemAlloc` + `ncclCommWindowRegister` + `ncclAllGather`)
  - How to run + walkthrough: `c/README.md`
- **Python (nccl4py + mpi4py)**: `python/` (uses `NCCLConfig(cta_policy=CTAPolicy.ZERO)` + `nccl.cupy.empty()` + `comm.register_window()` + `comm.allgather()`)
  - How to run + walkthrough: `python/README.md`

## Expected Output

You should see:

- Communicator initialization for each rank (with `CTAPolicy=2`)
- Symmetric memory allocation and window registration
- AllGather operation completion
- Verification showing each segment holds the contributing rank's value
- Clean window deregistration and resource cleanup

## Performance Benefits

The copy engine provides several advantages:

- **Zero SM usage**: Inside a single (MN)NVL domain, the collective uses the copy
  engine instead of SMs, freeing up compute resources
- **Computation overlap**: Enables true overlap of communication with GPU
  computation kernels
- **Better peak bandwidth**: Achieves higher peak bandwidth for large message
  sizes (with higher latency for small message sizes)

For more information see the [Fusing Communication and Compute with New Device API and Copy Engine Collectives in NVIDIA NCCL 2.28](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)
blog.

Copy Engine mode is most beneficial for:

- Applications that need to overlap communication with computation
- Scenarios where SM resources are at a premium
- Large-scale collectives without arithmetic operations (AllGather, AlltoAll, Gather, Scatter)

**Important**: Buffers must be allocated using the CUDA Virtual Memory Management (VMM) API.
NCCL provides the `ncclMemAlloc` convenience function for symmetric memory registration. The
`NCCL_WIN_COLL_SYMMETRIC` flag requires all ranks to provide symmetric buffers consistently.

## Common Issues and Solutions

1. **Window registration failure**: Buffers must be from a VMM-compatible allocator (e.g., `ncclMemAlloc`, not `cudaMalloc`)
2. **Allocation error**: If `ncclMemAlloc` fails, check NCCL version (requires 2.27+) and available memory
3. **Deregistration order**: Always deregister windows before freeing memory or destroying communicators
4. **Symmetric requirement**: All ranks must use `NCCL_WIN_COLL_SYMMETRIC` consistently
5. **Memory leaks**: Always use `ncclMemFree` for buffers allocated with `ncclMemAlloc`
6. **Copy engine not supported**: `CTAPolicy=2` only works inside a single (MN)NVL domain
