<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# Python (nccl4py + mpi4py): Symmetric Memory AllReduce

This is the nccl4py implementation using `nccl.core` with CuPy arrays. It uses
one GPU per MPI process, matching the C example.

## Run

From this directory:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cu12.txt  # or: requirements-cu13.txt
mpirun -np <num_processes> python allreduce_sm.py
```

## Code walk-through

- **One GPU per process**: each MPI rank picks its GPU from the node-local rank
  (`MPI.COMM_TYPE_SHARED` split) and makes it current with `cp.cuda.Device(local_rank).use()`.
- **Unique ID distribution**: rank 0 calls `nccl.get_unique_id()` and broadcasts it with
  `comm_world.bcast()` (matches C's `ncclGetUniqueId` + broadcast).
- **Communicator init**: `nccl.Communicator.init(nranks=mpi_size, rank=mpi_rank, unique_id=...)`
  joins the distributed communicator on this rank's GPU, matching C's `ncclCommInitRank`.
- **NCCL-backed CuPy arrays**: `nccl.cupy.empty(...)` creates CuPy arrays backed by NCCL-managed
  memory. These arrays can be registered directly as symmetric windows.
- **Window registration**: `comm.register_window(buf, flags=nccl.WindowFlag.CollSymmetric)`
  registers this rank's symmetric windows (matches C's `ncclCommWindowRegister` with
  `NCCL_WIN_COLL_SYMMETRIC`). It is a collective call - all ranks participate.
- **AllReduce**: `comm.allreduce(sendbuf, recvbuf, nccl.SUM)` performs AllReduce using the
  symmetric memory for optimized communication.
- **Deregistration**: `comm.destroy()` automatically deregisters all registered windows
  (matches C's `ncclCommWindowDeregister`).
