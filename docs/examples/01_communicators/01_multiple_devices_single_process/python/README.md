<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# Python (nccl4py): Multiple Devices Single Process

This is the nccl4py implementation of the example using `nccl.core`.

## Run

From this directory:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cu12.txt  # or: requirements-cu13.txt
python multiple_devices_single_process.py
```

## Code walk-through

- **Communicator init**: `nccl.Communicator.init_all()` creates a host-local communicator,
  returning one object per visible device, matching C's `ncclCommInitAll`.
- **Querying properties**: Each communicator exposes `.rank`, `.nranks`, and `.device` properties.
- **Cleanup**: `comm.finalize()` for each communicator within a single `nccl.group()`,
  then `comm.destroy()` for each. With multiple GPUs per thread, `finalize()` is a
  collective and must be grouped to avoid a hang.
