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
  barriers, `ncclGetLsaPointer` (manual reduction in-kernel)
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
  - Network barriers for cross-node synchronization (acquire before puts, release after flush)
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

### [05_allgather_cft_counted](05_allgather_cft_counted/)
**CFT Counted AllGather**
- **Pattern**: GPU kernel performs AllGather with counted CFT puts
- **API**: `ncclCft::putCounted`, CFT logical endpoints, symmetric windows
- **Use case**: Peer completion using a counter instead of barrier sync
- **Key features**:
  - Device-side allgather implemented with CFT counted writes
  - Counter base completion
  - One counter per rank shared across all peers

### [06_allgather_cft_multimem_counted](06_allgather_cft_multimem_counted/)
**CFT Multimem Counted AllGather**
- **Pattern**: GPU kernel performs AllGather with counted CFT multimem puts
- **API**: `ncclCft::putMultimemCounted`, CFT multimem logical endpoint,
  symmetric windows
- **Use case**: Counter-based completion with multicast writes
- **Key features**:
  - Device-side allgather implemented with CFT multimem counted writes
  - One multicast put per chunk instead of one put per peer
  - One counter per rank shared across all multicast senders

## Choosing the Right Pattern

*Scenario* : Custom kernels fusing computation and communication.
*Addresses* : Schedule communication from inside a CUDA kernel.
*Dependencies* : pthread or MPI

### Why the Device API?
The device API allows NCCL communication within CUDA kernels, fusing
communication and computation steps. This eliminates host-device synchronization
bottlenecks and enables lower-latency collective operations. See each example's
`c/README.md` for detailed code walkthroughs.

### Checking for Device API and GIN Support

These examples require communicator support for the relevant device-side
capabilities. The C/CUDA examples check communicator properties with
`ncclCommQueryProperties` before launching the device kernels:

```cpp
ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
NCCLCHECK(ncclCommQueryProperties(comm, &props));

if (!props.deviceApiSupport) {
    printf("ERROR: communicator does not support Device API!\n");
    // Exit gracefully...
}

// GIN-based examples also require network support.
if (props.ginType == NCCL_GIN_TYPE_NONE) {
    printf("ERROR: communicator does not support GIN!\n");
    // Exit gracefully...
}

// Pure LSA examples require a single LSA team.
if (props.nLsaTeams != 1) {
    printf("ERROR: expected 1 LSA team for pure LSA example!\n");
    // Exit gracefully...
}
```

Note: These examples are C/CUDA-only as they demonstrate GPU kernel programming.

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce_lsa
make 02_alltoall_gin
make 03_alltoall_hybrid
make 05_allgather_cft_counted
make 06_allgather_cft_multimem_counted
```

### **Individual Examples**
```shell
# Build and run the device API AllReduce
cd 01_allreduce_lsa/c && make
./allreduce_lsa

# Build and run the Pure GIN AlltoAll example
cd 02_alltoall_gin/c && make
./alltoall_gin

# Build and run the Hybrid AlltoAll example
cd 03_alltoall_hybrid/c && make
./alltoall_hybrid

# Build and run the CFT counted AllGather example
cd 05_allgather_cft_counted/c && make
./allgather_cft_counted

# Build and run the CFT multimem counted AllGather example
cd 06_allgather_cft_multimem_counted/c && make
./allgather_cft_multimem_counted
```

## References
- [NCCL User Guide:
  Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
- [NCCL API
  Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [CUDA Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
