<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Device API Examples

## Overview
This directory contains minimal examples that demonstrate NCCL's device API,
enabling users to perform inter-GPU communication within their own kernels.

## Examples

### [01_allreduce_lsa](01_allreduce_lsa/)
**AllReduce with Device Kernel Implementation**
- **Pattern**: GPU kernel performs collectives using device communicators
- **API**: `ncclDevCommCreate`, `ncclCommWindowRegister`, device-side LSA
  barriers, `ncclAllReduce`
- **Use case**: Allreduce operations with custom operations, fusing allreduce
  operation with previous/next compute operation.
- **Key features**:
  - Device communicator creation with LSA barrier support
  - Symmetric memory windows for peer memory access
  - Device kernels coordinating via LSA barriers
  - Host launches kernel; kernel performs AllReduce on-device

### [02_alltoall_gin](02_alltoall_gin/)
**Pure GIN AlltoAll - Network-Only Communication**
- **Pattern**: GPU kernel performs AlltoAll using only GIN for all peers
- **API**: `ncclDevCommCreate` with GIN support, `ncclGin`, GIN barriers and signals
- **Use case**: Multi-node AlltoAll with consistent network-based communication
- **Key features**:
  - Pure GIN implementation (no LSA optimizations)
  - Network barriers for cross-node synchronization
  - Signal-based completion detection
  - Baseline network performance measurements

### [03_alltoall_hybrid](03_alltoall_hybrid/)
**Hybrid AlltoAll - Optimized Communication**
- **Pattern**: GPU kernel performs AlltoAll using LSA for local peers, GIN for remote
- **API**: `ncclDevCommCreate` with both LSA and GIN support, peer classification
- **Use case**: Multi-node AlltoAll with optimal performance across topologies
- **Key features**:
  - Hybrid implementation for optimal performance
  - Intelligent peer classification (local vs remote)
  - Combined LSA and GIN synchronization
  - Production-ready optimized communication patterns

## Choosing the Right Pattern

*Scenario* : Custom kernels fusing computation and communication.
*Addresses* : Schedule communication from inside a CUDA kernel.
*Dependencies* : pthread or MPI

### Why the Device API?
The device API allows NCCL communication within CUDA kernels, fusing communication and computation steps:
```cpp
// Host:
// 1) Create device communicator + requirements
// 2) Register symmetric memory window for peer access
ncclDevComm devComm;
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;
NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));

// Device:
// - Use barriers for cross-GPU synchronization
// - Access peers via symmetric window (LSA pointers)
myAllReduceKernel<<<grid, block>>>(win, devComm);
```

### Checking for Device API and GIN Support

To check for the required Device API and GIN support, the communicator properties are queried via `ncclCommQueryProperties`:

```cpp
ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
NCCLCHECK(ncclCommQueryProperties(comm, &props));

// Check for Device API support
if (!props.deviceApiSupport) {
    printf("ERROR: communicator does not support Device API!\n");
    // Exit gracefully...
}

// Check for GIN support (ginType should not be NCCL_GIN_TYPE_NONE)
if (props.ginType == NCCL_GIN_TYPE_NONE) {
    printf("ERROR: communicator does not support GIN!\n");
    // Exit gracefully...
}

// For pure LSA examples, ensure a single team where all ranks can access each other
if (props.nLsaTeams != 1) {
    printf("ERROR: expected 1 LSA team for pure LSA example!\n");
    // Exit gracefully...
}
```

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce_lsa
make 02_alltoall_gin
make 03_alltoall_hybrid
```

### **Individual Examples**
```shell
# Build and run the device API AllReduce
cd 01_allreduce_lsa && make
./allreduce_lsa

# Build and run the Pure GIN AlltoAll example
cd 02_alltoall_gin && make
./allreduce_gin

# Build and run the Hybrid AlltoAll example
cd 03_alltoall_hybrid && make
./allreduce_hybrid
```

## References
- [NCCL User Guide:
  Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
- [NCCL API
  Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [CUDA Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
