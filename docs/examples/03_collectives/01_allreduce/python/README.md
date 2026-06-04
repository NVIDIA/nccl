<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# Python (nccl4py): AllReduce

This is the nccl4py implementation using `nccl.core` with CuPy arrays.

## Run

From this directory:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cu12.txt  # or: requirements-cu13.txt
python allreduce.py
```

## Code walk-through

- **Communicator init**: `nccl.Communicator.init_all()` creates a host-local communicator,
  returning one object per visible device, matching C's `ncclCommInitAll`.
- **Device buffers**: CuPy arrays are used for buffer allocation, initialization, and host-side
  verification.
- **AllReduce**: `comm.allreduce(sendbuff, recvbuff, nccl.SUM)` performs sum reduction
  across all GPUs (matches C's `ncclAllReduce`).
- **Group operations**: `with nccl.group():` wraps all collective calls to avoid deadlock
  (matches C's `ncclGroupStart`/`ncclGroupEnd`).
