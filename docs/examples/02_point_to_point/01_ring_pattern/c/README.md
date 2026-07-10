<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: Ring Communication Pattern

This is the C implementation using `ncclSend` and `ncclRecv`.

## Build

From this directory:

```shell
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```
## Run

From this directory:

```shell
./ring_pattern
```

Run with NCCL debug output:

```shell
NCCL_DEBUG=INFO ./ring_pattern
```

## Expected output

```
Starting NCCL ring communication example
Using 4 GPUs for ring communication
Preparing data structures
Initializing NCCL communicators
All communicators initialized successfully
Verifying communicator setup
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

## Code walk-through

### Key functions: `ncclSend` and `ncclRecv`

The ring pattern uses point-to-point operations:

```c
// Use ncclGroupStart/End to prevent deadlocks
ncclGroupStart();
for (int i = 0; i < num_gpus; i++) {
  int next = (i + 1) % num_gpus;
  int prev = (i - 1 + num_gpus) % num_gpus;

  ncclSend(d_sendbuff[i], count, ncclFloat, next, comms[i], streams[i]);
  ncclRecv(d_recvbuff[i], count, ncclFloat, prev, comms[i], streams[i]);
}
ncclGroupEnd();
```

### Ring topology

- Each GPU sends to `(rank + 1) % num_gpus` (next neighbor)
- Each GPU receives from `(rank - 1 + num_gpus) % num_gpus` (previous neighbor)
- Data flows in a circular pattern: GPU 0 → GPU 1 → ... → GPU N-1 → GPU 0

### Deadlock avoidance

`ncclGroupStart/End` is critical to avoid deadlocks:
- Without grouping, a send might block waiting for a receive that hasn't been posted yet
- Grouping allows NCCL to see all operations and schedule them appropriately
- All sends and receives are executed simultaneously
