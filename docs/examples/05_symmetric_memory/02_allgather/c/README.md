<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: AllGather with Symmetric Memory and Copy Engine

This is the C implementation using symmetric memory windows with the copy engine
(`CTAPolicy=2`) so the collective uses zero SMs.

## Build

From this directory:

```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

## Run

### When compiled for pthreads (default)

```shell
[NTHREADS=N] ./allgather_ce
```

### When compiled for MPI

```shell
mpirun -np <num_processes> ./allgather_ce
```

## Expected output

### With 4 GPUs (using pthreads/MPI)
```
Starting AllGather example with 4 ranks (Copy Engine enabled)
  Rank 0 communicator initialized using device 0 (CTAPolicy=2)
  Rank 1 communicator initialized using device 1 (CTAPolicy=2)
  Rank 2 communicator initialized using device 2 (CTAPolicy=2)
  Rank 3 communicator initialized using device 3 (CTAPolicy=2)
Symmetric Memory allocation
  Rank 0 allocating 4.00 MB send buffer, 16.00 MB recv buffer
  Rank 1 allocating 4.00 MB send buffer, 16.00 MB recv buffer
  Rank 2 allocating 4.00 MB send buffer, 16.00 MB recv buffer
  Rank 3 allocating 4.00 MB send buffer, 16.00 MB recv buffer
  Rank 0 data initialized (value: 0)
  Rank 1 data initialized (value: 1)
  Rank 2 data initialized (value: 2)
  Rank 3 data initialized (value: 3)
Starting AllGather with 1048576 elements per rank (16 MB total)
AllGather completed successfully
Verification - Segment 0: Expected: 0.0, Got: 0.0
Verification - Segment 1: Expected: 1.0, Got: 1.0
Verification - Segment 2: Expected: 2.0, Got: 2.0
Verification - Segment 3: Expected: 3.0, Got: 3.0
Results verified correctly
  Rank 0 symmetric memory windows deregistered
  Rank 1 symmetric memory windows deregistered
  Rank 2 symmetric memory windows deregistered
  Rank 3 symmetric memory windows deregistered
All resources cleaned up successfully
Example completed - demonstrated symmetric memory + copy engine allgather
```

## Code walk-through

### NCCL configuration for copy engine

Set `CTAPolicy=2` so the collective runs on the copy engine instead of SMs:

```c
// Configure NCCL to use copy engine (CTAPolicy=2) for zero SM usage
// Alternatively, this can also be done by setting Environment Variable NCCL_CTA_POLICY=ZERO
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.CTAPolicy = 2;

// Initialize communicator with config
ncclCommInitRankConfig(&comm, total_ranks, nccl_unique_id, my_rank, &config);
```

### Buffer allocation and window registration

```c
size_t sendcount;       // Elements per rank
size_t send_size_bytes; // Size of send buffer
size_t recv_size_bytes; // Size of recv buffer (sendcount * total_ranks * sizeof(type))
void *d_sendbuff;
void *d_recvbuff;

// Allocate buffers using ncclMemAlloc (compatible with symmetric memory)
ncclMemAlloc(&d_sendbuff, send_size_bytes);
ncclMemAlloc(&d_recvbuff, recv_size_bytes);

ncclComm_t comm;
ncclWindow_t send_win;
ncclWindow_t recv_win;

// Register buffers as symmetric windows
ncclCommWindowRegister(comm, d_sendbuff, send_size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC);
ncclCommWindowRegister(comm, d_recvbuff, recv_size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC);
```

### AllGather operation

```c
cudaStream_t stream; // stream is set in cudaStreamCreate

// Perform AllGather with symmetric memory and copy engine
ncclAllGather(d_sendbuff, d_recvbuff, sendcount, ncclFloat,
              comm, stream);
```

### Window deregistration and cleanup

```c
// Deregister symmetric memory windows
ncclCommWindowDeregister(comm, send_win);
ncclCommWindowDeregister(comm, recv_win);

// Free buffers allocated with ncclMemAlloc
ncclMemFree(d_sendbuff);
ncclMemFree(d_recvbuff);
```
