<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Device API Pure GIN AlltoAll Example

This example demonstrates NCCL's GPU-Initiated Networking (GIN) capabilities for performing AlltoAll collective operations directly from GPU kernels using only network-based communication.

## Overview

This example showcases **pure GIN communication** where all data exchange happens through the network, without any Load Store Access (LSA) optimizations. This is particularly useful for:

- Multi-node environments where ranks cannot use LSA
- Testing network performance without local optimizations
- Understanding the baseline GIN communication patterns
- Scenarios where all communication must go through the network

## What This Example Does

1. **Creates device communicators** using `ncclDevCommCreate` for GPU kernel access to NCCL operations
2. **Registers symmetric memory windows** with `ncclCommWindowRegister` for direct peer-to-peer access
3. **Launches GPU kernel** that performs AlltoAll operations using pure GIN for all peer communication

## Building and Running

The advanced examples can be built using either pthread or MPI for parallelization. pthread is the default choice. To use MPI the user needs to set `MPI=1` at build time and can optionally provide a valid MPI installation under `MPI_HOME`.

### Build
```bash
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run when compiled for pthreads (default)
```bash
[NTHREADS=N] ./alltoall_gin
```

### Run when compiled for MPI
```bash
mpirun -np <num_processes> ./alltoall_gin
```

## Code Walk-through

### Device Communicator Creation (Host-side)
The `ncclDevComm` is the core component enabling GPU kernels to perform network communication directly. For pure GIN communication, we configure the device communicator with GIN-specific resources. The `ncclDevCommRequirements` specifies GIN barriers for network synchronization and signals for completion detection. Unlike LSA-based examples, we don't need LSA barriers since all communication goes through the network.

```cpp
ncclDevComm devComm;
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
// GIN barriers enable cross-node synchronization over the network
reqs.railGinBarrierCount = NCCL_DEVICE_CTA_COUNT;
// GIN signals provide completion notifications for asynchronous operations
reqs.ginSignalCount = 1;

// Create device communicator with pure GIN support
NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
```

### Memory Window Registration (Host-side)
The device API requires symmetric memory windows registered using `NCCL_WIN_COLL_SYMMETRIC`. These windows enable GPU kernels to access remote memory through GIN operations. Unlike LSA which provides direct memory access, GIN windows are accessed through network put/get operations.

```cpp
ncclWindow_t send_win;
ncclWindow_t recv_win;

// Register symmetric windows for GIN network access
NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));
```

### GIN Barriers (Device-side)
GIN barriers enable cross-node synchronization from device code over the network. Each thread block uses `blockIdx.x` to select its dedicated barrier, allowing blocks to progress independently while coordinating with corresponding blocks on other nodes. This is crucial for ensuring all ranks are ready before starting the AlltoAll exchange.

```cpp
// GIN barriers coordinate GPU threads across different nodes over network
ncclGinBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(),                    // Barrier scope: entire CTA (thread block)
    gin,                              // GIN context for network operations
    ncclTeamWorld(devComm),          // Team spanning all ranks
    devComm.railGinBarrier,          // GIN barrier handle
    blockIdx.x                       // Barrier index: matches our CTA index
};
bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
```

### GIN Put Operations (Device-side)
GIN provides one-sided put operations for direct remote memory writes over the network. Each thread handles a subset of destination ranks, writing its rank's data to the appropriate location in each peer's receive buffer. The `ncclGin_SignalInc` parameter increments a signal counter, enabling asynchronous completion detection.

```cpp
// Send data to all peers via GIN network operations
const size_t size = count * sizeof(T);
for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,  // Destination: peer r's buffer
        sendwin, sendoffset + r * size,             // Source: data for peer r
        size, ncclGin_SignalInc{signalIndex});      // Signal increment for completion
}
```

### Signal-based Completion (Device-side)
GIN uses signals for asynchronous completion detection of network operations. The kernel waits for the signal value to reach the expected count (initial value + number of ranks), indicating all put operations have completed. The `gin.flush()` ensures all pending operations are committed before proceeding.

```cpp
// Wait for all remote puts to complete
gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
gin.flush(ncclCoopCta());  // Ensure all operations are committed
```

## Expected Output

```
Starting Pure GIN AlltoAll initialization
  Rank 0 using GPU device 0
  Rank 1 using GPU device 1
  Rank 0 initialized NCCL communicator for 2 total ranks
  Rank 1 initialized NCCL communicator for 2 total ranks
  Rank 0 initialized send data
  Rank 1 initialized send data
  Rank 0 created device communicator with GIN support
  Rank 1 created device communicator with GIN support
Starting Pure GIN AlltoAll with 1024 elements per rank (2048 total elements, 0 MB)

=== Executing Pure GIN AlltoAll ===
  Rank 0 completed pure GIN AlltoAll kernel
  Rank 1 completed pure GIN AlltoAll kernel
Pure GIN AlltoAll result: PASSED
```

## When to Use

- **Multi-node environments**: When ranks cannot use LSA
- **Testing network performance**: Without local optimizations

## Performance Considerations

- **Network overhead**: All communication goes through the network stack
- **Signal-based completion**: Enables asynchronous operation patterns
- **Barrier synchronization**: Ensures proper ordering of network operations
- **Multiple GIN contexts**: Can improve parallel communication performance

## Common Issues and Solutions

### Issue: Deadlock at util_broadcast
**Solution:** Ensure you're running with multiple GPUs/processes
```bash
NTHREADS=2 ./alltoall_gin  # For 2 GPUs
```

### Issue: CUDA out of memory
**Solution:** Reduce the data size in the example

### Issue: Network errors
**Solution:** Ensure proper network configuration for multi-node setups

## Performance Notes

- These are educational examples, not optimized for performance
- Real implementations should consider:
  - Optimal GIN context usage for parallel operations
  - Signal pool management for high-throughput scenarios
  - Memory coalescing patterns for network operations
  - Network topology-aware communication strategies

## Error Handling

The example uses comprehensive error checking for CUDA, NCCL, and GIN operations. Device kernels should implement proper error handling for network operations and signal management.

## Next Steps

After understanding this example, explore:
- **Performance optimization**: Fine-tune GIN context usage and signal management
- **Hybrid approaches**: Combine GIN with LSA for topology-aware optimizations
- **Integration with compute**: Fuse network communication with computation kernels
