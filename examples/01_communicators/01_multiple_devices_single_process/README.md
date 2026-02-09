<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Example: Multiple Devices Single Process

This example demonstrates how to use `ncclCommInitAll` to create NCCL
communicators for multiple GPUs within a single process, without requiring MPI
or threading.

## Overview

The `ncclCommInitAll` function provides a simplified way to initialize NCCL
communicators when:
- All GPUs are managed by a single process
- Running on a single node
- No multi-process coordination is needed

This approach is ideal for single-node multi-GPU applications where simplicity
is preferred over the flexibility of multi-process setups.

## What This Example Does

1. **Device Detection**:
   - Queries available CUDA devices
   - Lists device properties for each GPU

2. **Communicator Creation**:
   - Uses `ncclCommInitAll` to create all communicators in one call
   - Automatically assigns NCCL ranks 0 through n-1
   - No NCCL unique ID distribution needed

3. **Verification**:
   - Displays communicator information for each GPU
   - Shows rank assignments and device mappings
   - Confirms successful initialization

4. **Cleanup**:
   - Properly destroys communicators and streams
   - Demonstrates correct resource management

## Building and Running

### Build
```shell
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run with all available GPUs
```shell
./multiple_devices_single_process
```

### Run with specific GPUs
```shell
# Use only GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 ./multiple_devices_single_process
```

### Run with NCCL debug output
```shell
NCCL_DEBUG=INFO ./multiple_devices_single_process
```

## Code Walk-through

### Key Function: ncclCommInitAll
For single-node collective examples we use `ncclCommInitAll` as it creates a clique of communicators in one call.
```c
int num_gpus; // num_gpus is set by querying the CUDA devices
ncclComm_t* comms;
int* devices; // devices needs to be populated with CUDA devices used

// Create communicators for all devices in one call
NCCLCHECK(ncclCommInitAll(comms, num_gpus, devices));
```

This single function call:
- Creates `num_gpus` communicators
- Assigns ranks 0 to (num_gpus-1)
- Sets up internal communication paths
- No unique ID needed

### Comparison with ncclCommInitRank
`ncclCommInitAll` is a convenience function and has the same functionality as:
```c
ncclUniqueId id;

ncclGetUniqueId(&id);

ncclGroupStart();
for(int i = 0; i < num_gpus; i++) {
  cudaSetDevice(i);
  ncclCommInitRank(comms[i], num_gpus, id, devices[i]);
}
ncclGroupEnd();
```

## Expected Output

```
Found 4 CUDA device(s) available

Available GPU devices:
  GPU 0: NVIDIA A100-SXM4-40GB (CUDA Device 0)
    Compute Capability: 8.0
    Memory: 40.0 GB
  GPU 1: NVIDIA A100-SXM4-40GB (CUDA Device 1)
    Compute Capability: 8.0
    Memory: 40.0 GB
  GPU 2: NVIDIA A100-SXM4-40GB (CUDA Device 2)
    Compute Capability: 8.0
    Memory: 40.0 GB
  GPU 3: NVIDIA A100-SXM4-40GB (CUDA Device 3)
    Compute Capability: 8.0
    Memory: 40.0 GB
Using ncclCommInitAll() to create all communicators simultaneously
All 4 NCCL communicators initialized successfully

Communicator Details:
  Communicator 0: Rank 0/4 on CUDA device 0
  Communicator 1: Rank 1/4 on CUDA device 1
  Communicator 2: Rank 2/4 on CUDA device 2
  Communicator 3: Rank 3/4 on CUDA device 3
All communicators have the expected size of 4

Synchronizing all CUDA streams...
All streams synchronized
Destroying NCCL communicators...
All NCCL communicators destroyed
Destroying CUDA streams...
All CUDA streams destroyed

=============================================================
SUCCESS: Multiple devices single process example completed!
=============================================================
```

## When to Use ncclCommInitAll

### Ideal Use Cases
- **Single-node workloads**: All GPUs on one machine
- **Simple applications**: No multi-process complexity needed
- **Testing/Development**: Quick setup for experiments

### When NOT to Use
- **Multi-node clusters**: Need MPI for cross-node communication
- **Process isolation**: When GPUs should be in separate processes

## Performance Considerations

- **Advantages**:
  - Lower overhead (no inter process communication)
  - Simpler memory management
  - Direct access to all GPUs

- **Disadvantages**:
  - Limited by single process resources
  - Cannot scale beyond one node

## Common Issues and Solutions

1. **Not all GPUs visible**:
   - Check `CUDA_VISIBLE_DEVICES`
   - Ensure user has permissions for all GPUs
   - Verify no other process is using GPUs exclusively

2. **Out of memory**:
   - Single process must handle memory for all GPUs
   - Consider using multiple processes if memory limited

## Next Steps

After understanding this example:
1. Try the collective operation examples using `ncclCommInitAll`
2. Compare performance with MPI-based multi-process approach
3. Experiment with different GPU combinations
