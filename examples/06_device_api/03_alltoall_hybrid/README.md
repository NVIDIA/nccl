<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Device API Hybrid AlltoAll Example

This example shows how to implement AlltoAll operations using a hybrid approach that combines Load Store Access (LSA) for local peers with GPU-Initiated Networking (GIN) for remote peers. We create a device communicator with `ncclDevCommCreate` supporting both LSA and GIN capabilities, enabling optimal communication performance across different peer types.

## Overview

This example showcases **hybrid communication** that intelligently selects the optimal communication method for each peer:

- **LSA (Load Store Access)** for local peers (same node/memory space)
- **GIN (GPU-Initiated Networking)** for remote peers (different nodes)

## What This Example Does

1. **Creates hybrid device communicators** using `ncclDevCommCreate` with both LSA and GIN support for optimal peer communication
2. **Registers symmetric memory windows** with `ncclCommWindowRegister` for both LSA direct access and GIN network operations
3. **Launches GPU kernel** that performs AlltoAll operations using LSA for local peers and GIN for remote peers
4. **Demonstrates hybrid synchronization** coordinating both LSA barriers and GIN signals for correctness

## Building and Running

The advanced examples can be built using either pthread or MPI for parallelization. pthread is the default choice. To use MPI the user needs to set `MPI=1` at build time and can optionally provide a valid MPI installation under `MPI_HOME`.

### Build
```bash
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run when compiled for pthreads (default)
```bash
[NTHREADS=N] ./alltoall_hybrid
```

### Run when compiled for MPI
```bash
mpirun -np <num_processes> ./alltoall_hybrid
```

## Code Walk-through

### Device Communicator Creation (Host-side)
The `ncclDevComm` is the core component enabling GPU kernels to perform both local and remote communication. For hybrid communication, we configure the device communicator with both LSA and GIN resources. The `ncclDevCommRequirements` specifies LSA barriers for local synchronization, GIN barriers for network synchronization, and GIN signals for completion detection. This dual setup enables optimal communication for each peer type.

```cpp
ncclDevComm devComm;
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
// LSA barriers enable direct memory access coordination for local peers
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;
// GIN barriers enable cross-node synchronization over the network
reqs.railGinBarrierCount = NCCL_DEVICE_CTA_COUNT;
// GIN signals provide completion notifications for asynchronous network operations
reqs.ginSignalCount = 1;

// Create device communicator with hybrid LSA+GIN support
NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
```

### Memory Window Registration (Host-side)
The device API requires symmetric memory windows registered using `NCCL_WIN_COLL_SYMMETRIC`. These windows enable both LSA direct access for local peers and GIN network operations for remote peers. The same memory windows support both communication methods, with the kernel automatically selecting the appropriate access pattern based on peer locality.

```cpp
ncclWindow_t send_win;
ncclWindow_t recv_win;

// Register symmetric windows for both LSA and GIN access
NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));
```

### Hybrid Barriers (Device-side)
Hybrid barriers coordinate both local LSA operations and remote GIN operations. The barrier session uses the world team and GIN context to ensure synchronization across all ranks, regardless of their communication method. This unified barrier approach ensures all peers reach the same synchronization point before proceeding with data exchange.

```cpp
// Hybrid barriers coordinate both LSA and GIN operations across all ranks
ncclBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(),              // Barrier scope: entire CTA (thread block)
    ncclTeamTagWorld(),         // Team spanning all ranks (local + remote)
    gin,                        // GIN context for network coordination
    blockIdx.x                  // Barrier index: matches our CTA index
};
bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
```

### Peer Classification (Device-side)
The hybrid kernel intelligently classifies peers into local (LSA-accessible) and remote (GIN-only) categories. This classification determines the optimal communication method for each peer. Local peers benefit from direct memory access, while remote peers use network communication.

```cpp
// Classify peers into local (LSA) and remote (GIN) categories
ncclTeam world = ncclTeamWorld(devComm);  // All ranks
ncclTeam lsa = ncclTeamLsa(devComm);      // Local ranks only
const int startLsa = world.rank - lsa.rank;  // First local rank in world
const int lsaSize = lsa.nRanks;              // Number of local peers
```

### Memory Access (Device-side)
`ncclGetLsaPointer` allows CUDA kernels to directly access other GPUs' memory within the LSA team, while `gin.put` handles remote communication over the network. The hybrid approach uses the most efficient method for each peer type.

```cpp
// Handle local peers using direct memory access (LSA)
T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);
T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);

// Handle remote peers using network operations (GIN)
gin.put(world, r, recvwin, recvoffset + world.rank * size,
        sendwin, sendoffset + r * size, size, ncclGin_SignalInc{signalIndex});
```

## Building and Running

### Build
```bash
make
```

### Run with pthread mode (default)
```bash
# Run with all available GPUs
./alltoall_hybrid

# Run with specific number of GPUs
NTHREADS=4 ./alltoall_hybrid
```

### Run with MPI mode
```bash
# Build with MPI support
make MPI=1

# Run with MPI across multiple nodes
mpirun -np 4 --hostfile hosts ./alltoall_hybrid
```

### Test
```bash
make test
```

## Expected Output

```
Starting Hybrid AlltoAll initialization
  Rank 0 using GPU device 0
  Rank 1 using GPU device 1
  Rank 2 using GPU device 2
  Rank 3 using GPU device 3
  Rank 0 initialized NCCL communicator for 4 total ranks
  Rank 1 initialized NCCL communicator for 4 total ranks
  Rank 2 initialized NCCL communicator for 4 total ranks
  Rank 3 initialized NCCL communicator for 4 total ranks
  Rank 0 initialized send data
  Rank 1 initialized send data
  Rank 2 initialized send data
  Rank 3 initialized send data
  Rank 0 created device communicator with hybrid support
  Rank 1 created device communicator with hybrid support
  Rank 2 created device communicator with hybrid support
  Rank 3 created device communicator with hybrid support
Starting Hybrid AlltoAll with 1024 elements per rank (4096 total elements, 0 MB)
Using LSA for local peers and GIN for remote peers

=== Executing Hybrid AlltoAll ===
  Rank 0 completed hybrid AlltoAll kernel
  Rank 1 completed hybrid AlltoAll kernel
  Rank 2 completed hybrid AlltoAll kernel
  Rank 3 completed hybrid AlltoAll kernel
Hybrid AlltoAll result: PASSED
✓ All 4096 elements correctly exchanged using hybrid communication
```

## When to Use

- **Multi-node usage**: Mixed local/remote communication patterns
- **Production workloads**: Where performance is critical
- **Heterogeneous clusters**: Different node configurations

## Performance Considerations

**Advantages:**
- **Reduced Latency**: LSA provides low latency for local communication
- **Optimal Bandwidth**: GIN efficiently handles remote communication
- **Reduced Network Load**: Local traffic stays off the network
- **Scalable Design**: Efficient across different node configurations

**Disadvantages:**
- More complex programming model requiring coordination of both LSA and GIN
- Requires careful synchronization between different communication methods
- Higher development complexity compared to pure approaches

## Common Issues and Solutions

### Issue: LSA barriers not supported
**Cause:** GPUs not connected through NVLink or PCIe for direct memory access
**Solution:** Verify GPU topology with `nvidia-smi topo -m` and ensure proper LSA-capable connections

### Issue: Hybrid synchronization failures
**Solution:** Ensure both `lsaBarrierCount` and `railGinBarrierCount` match the number of thread blocks in kernel launch configuration

## Performance Notes

- These are educational examples, not optimized for performance
- Real implementations should consider:
  - Optimal balance between LSA and GIN operations based on topology
  - Memory coalescing patterns for both LSA and GIN operations
  - Barrier synchronization overhead minimization
  - Signal pool management for high-throughput GIN scenarios

## Error Handling

The example uses comprehensive error checking for CUDA, NCCL, LSA, and GIN operations. Device kernels should implement proper error handling for both direct memory access patterns and network operations.

## Next Steps

After understanding this example, explore:
- **Topology-aware optimization**: Fine-tune LSA/GIN balance based on hardware topology
- **Custom hybrid patterns**: Implement specialized communication strategies
- **Performance profiling**: Analyze LSA vs GIN performance characteristics
- **Advanced synchronization**: Optimize barrier usage for complex communication patterns
