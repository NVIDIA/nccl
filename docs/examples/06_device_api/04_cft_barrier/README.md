<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Device API CFT barrier

This example shows how to create and use a CFT barrier session.

## Overview

Large multi-node NVLink domains need the Cuda Fabric Transport (CFT) programming model
to share and access peer GPU memory across the NCCL ranks. NCCL exposes this new
programming model through appropriate CFT handles and CFT synchronization primitives.
This example shows users how resources for CFT barriers are created and used.

## Runtime Requirements

The C variant can be built with either pthreads or MPI. It can run on a single node but is most relevant when local and remote peers are both present (multiple nodes with multiple GPUs per node). It is most useful with multiple GPUs, but can still run with one visible GPU.

The CFT programming model requires CUDA TK 13.3 and SM\_100 architectures.

## What This Example Does

1. **Create barrier requirement** for CFT barrier session
2. **Create team requirement** for CFT ranks with unicast or multicast capabilities
3. **Create device communicator** to allocate barrier and team requirements
4. **Launch GPU kernel** that performs:
   - Create barrier session with the resource handle for the barrier provided by the user
   - Use the barrier session to synchronize all the ranks in the communicator
5. **Print test result** and clean up resources

## Key concepts
