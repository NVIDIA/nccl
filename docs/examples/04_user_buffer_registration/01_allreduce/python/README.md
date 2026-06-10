<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# Python (nccl4py + mpi4py): User Buffer Registration AllReduce

This is the nccl4py implementation using `nccl.core` with CuPy arrays. It uses
one GPU per MPI process, matching the C example.

## Run

From this directory:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cu12.txt  # or: requirements-cu13.txt
mpirun -np <num_processes> python allreduce_ub.py
```

## Code walk-through

- **One GPU per process**: each MPI rank picks its GPU from the node-local rank
  (`MPI.COMM_TYPE_SHARED` split) and makes it current with `cp.cuda.Device(local_rank).use()`.
- **Unique ID distribution**: rank 0 calls `nccl.get_unique_id()` and broadcasts it with
  `comm_world.bcast()` (matches C's `ncclGetUniqueId` + broadcast).
- **Communicator init**: `nccl.Communicator.init(nranks=mpi_size, rank=mpi_rank, unique_id=...)`
  joins the distributed communicator on this rank's GPU, matching C's `ncclCommInitRank`.
- **NCCL-backed CuPy arrays**: `nccl.cupy.empty(...)` creates CuPy arrays backed by NCCL-managed
  memory so they can be registered for zero-copy.
- **Buffer registration**: `comm.register_buffer(buf)` pre-registers this rank's buffers
  (matches C's `ncclCommRegister`); it returns a `RegisteredBufferHandle`.
- **AllReduce**: `comm.allreduce(sendbuf, recvbuf, nccl.SUM)` performs AllReduce using the
  registered buffers.
- **Deregistration**: `comm.destroy()` automatically deregisters all registered buffers
  (matches C's `ncclCommDeregister`).
