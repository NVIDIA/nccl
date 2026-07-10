<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# Python (nccl4py + mpi4py): Symmetric Memory AllGather with Copy Engine

This is the nccl4py implementation using `nccl.core` with CuPy arrays. It runs an
AllGather over symmetric memory windows on the GPU's copy engine (zero SM usage),
using one GPU per MPI process, matching the C example.

## Run

From this directory:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cu12.txt  # or: requirements-cu13.txt
mpirun -np <num_processes> python allgather_ce.py
```

## Code walk-through

- **One GPU per process**: each MPI rank picks its GPU from the node-local rank
  (`MPI.COMM_TYPE_SHARED` split) and makes it current with `cp.cuda.Device(local_rank).use()`.
- **Copy-engine config**: `nccl.NCCLConfig(cta_policy=nccl.CTAPolicy.ZERO)` selects the
  zero-SM (copy engine) policy, matching C's `config.CTAPolicy = 2`.
- **Unique ID distribution**: rank 0 calls `nccl.get_unique_id()` and broadcasts it with
  `comm_world.bcast()` (matches C's `ncclGetUniqueId` + broadcast).
- **Communicator init**: `nccl.Communicator.init(nranks=mpi_size, rank=mpi_rank, unique_id=..., config=config)`
  applies the copy-engine config on this rank's GPU, matching C's `ncclCommInitRankConfig`.
- **NCCL-backed CuPy arrays**: `nccl.cupy.empty(...)` creates CuPy arrays backed by NCCL-managed
  memory so they can be registered as symmetric windows. The receive buffer is `nranks * sendcount`
  elements (it holds every rank's slice).
- **Window registration**: `comm.register_window(buf, flags=nccl.WindowFlag.CollSymmetric)`
  registers this rank's symmetric windows (matches C's `ncclCommWindowRegister` with
  `NCCL_WIN_COLL_SYMMETRIC`). It is a collective call - all ranks participate.
- **AllGather**: `comm.allgather(sendbuf, recvbuf)` gathers `sendcount` elements from every rank
  into each rank's `recvbuf` (matches C's `ncclAllGather`); with symmetric memory and
  `CTAPolicy.ZERO` it runs on the copy engine.
- **Deregistration**: `comm.destroy()` automatically deregisters all registered windows
  (matches C's `ncclCommWindowDeregister`).
