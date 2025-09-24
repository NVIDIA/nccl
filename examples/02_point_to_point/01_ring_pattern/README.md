<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Ring Communication Pattern Example

This example demonstrates a ring communication pattern using NCCL P2P
operations. It runs on a single node where a single process manages all GPUs and
data flows in a circular pattern.

## Overview

The ring communication pattern creates a circular data flow where each GPU sends
data to its "next" neighbor and receives from its "previous" neighbor in the
ring. This example uses `ncclCommInitAll` for simplified single-threaded,
single-process multi-GPU setup.

## What This Example Does

1. **Detects and initializes all available GPUs** using `ncclCommInitAll` for
   simplified single-process setup
2. **Creates ring topology** where each GPU calculates its next and previous
   neighbors using modulo
3. **Executes simultaneous point-to-point communication** with each GPU sending
   to next and receiving from previous
4. **Verifies data correctness** by checking that each GPU received the expected
   data from its predecessor

## Building and Running

### Build the Example
```bash
cd examples/02_point_to_point/01_ring_pattern
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run with All Available GPUs
```bash
./ring_pattern
```

## Code Walk-through

### Ring Topology Setup

The example calculates ring neighbors using modulo arithmetic:

```cpp
for (int i = 0; i < num_gpus; i++) {
    int next = (i + 1) % num_gpus;        // Next neighbor in ring
    int prev = (i - 1 + num_gpus) % num_gpus;  // Previous neighbor in ring
}
```

### Simultaneous Communication

Uses `ncclGroupStart/End` to prevent deadlocks when scheduling all send and
receive operations:

```cpp
float **d_sendbuff; // device side send and receive buffer are allocated through cudaMalloc
float **d_recvbuff;
size_t count;       // count is set to the number of floats to be sent (usually the size of the buffers)
ncclComm_t *comms;  // comms are set during ncclCommInitAll
cudaStream_t *streams; // streams are set in cudaStreamCreate

// Each GPU simultaneously sends to next and receives from previous
NCCLCHECK(ncclGroupStart());
for (int i = 0; i < num_gpus; i++) {
    int next = (i + 1) % num_gpus;
    int prev = (i - 1 + num_gpus) % num_gpus;

    NCCLCHECK(ncclSend(d_sendbuff[i], count, ncclFloat, next, comms[i], streams[i]));
    NCCLCHECK(ncclRecv(d_recvbuff[i], count, ncclFloat, prev, comms[i], streams[i]));
}
NCCLCHECK(ncclGroupEnd());
```

## Expected Output

```
Starting NCCL ring communication example
Using 4 GPUs for ring communication
Preparing data structures
Initializing NCCL communicators
All communicators initialized successfully
Creating CUDA streams and verifying setup
  GPU 0 -> NCCL rank 0/4 on CUDA device 0
  GPU 1 -> NCCL rank 1/4 on CUDA device 1
  GPU 2 -> NCCL rank 2/4 on CUDA device 2
  GPU 3 -> NCCL rank 3/4 on CUDA device 3
Setting up ring topology
Data flow -> GPU 0 -> GPU 1 -> ... -> GPU 3 -> GPU 0
Ring transfer with 268435456 elements (1.00 GB per GPU)
Allocating and initializing buffers
Executing ring communication
  GPU 0 sends to GPU 1, receives from GPU 3
  GPU 1 sends to GPU 2, receives from GPU 0
  GPU 2 sends to GPU 3, receives from GPU 1
  GPU 3 sends to GPU 0, receives from GPU 2
Ring communication completed successfully
Verifying data correctness
  GPU 0 received data from GPU 3: CORRECT
  GPU 1 received data from GPU 0: CORRECT
  GPU 2 received data from GPU 1: CORRECT
  GPU 3 received data from GPU 2: CORRECT
SUCCESS - All GPUs received correct data
Cleaning up resources
Example completed successfully!
```

## When to Use

- **Learning NCCL fundamentals**: Understanding point-to-point communication
  patterns
- **Algorithm development**: Building custom collective operations based on
  point to point communications
- **Single-node applications**: Pipeline parallelism or custom data distribution
  patterns

## Key Insights
- `ncclCommInitAll` simplifies single-node multi-GPU setup
- No MPI or pthreads needed for single-node patterns
- Ring pattern enables circular data flow among all GPUs
- `ncclGroupStart/End` prevents deadlock in simultaneous operations
- Each GPU both sends and receives in parallel

## Common Issues and Solutions

### Issue: Deadlock without group operations
**Solution:** Always use `ncclGroupStart()` and `ncclGroupEnd()` when performing
simultaneous send/recv operations.

### Issue: Verification failures
**Solution:** Check ring topology calculations and data initialization patterns.
Ensure correct neighbor calculations.

## Error Handling

This example uses comprehensive error checking with `NCCLCHECK` and `CUDACHECK`
macros that immediately exit on any failure. In production code, consider more
graceful error handling and recovery mechanisms.

## Next Steps

After this example, try:
- **Collective operations**: Examples in `03_collectives/`
- **Multi-node approach**: Use the MPI implementation from `01_communicators` to
  send data across nodes.
