# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A reduce-scatter that fails mid-protocol (TIDE_FAULT=abort: a device wait nothing will satisfy is
queued, then the call raises) must leave a communicator that close() and the arena's release abort
instead of draining: the process returns instead of hanging. Needs the fault-hook build.

TIDE_FAULT=abort torchrun --standalone --nproc_per_node 8 test/test_abort.py
"""
import os
import time

import torch
import torch.distributed as dist

import tide


def main() -> int:
    if "abort" not in os.environ.get("TIDE_FAULT", ""):
        raise SystemExit("run with TIDE_FAULT=abort")
    dist.init_process_group("nccl")
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(dev)
    comm = tide.TideComm(device=dev)
    grads = [torch.ones(4096, 1024, dtype=torch.bfloat16, device=dev)]
    arena = comm.alloc_arena(comm.rs_stage_bytes(grads))
    h = comm.register_rs(arena, 0, grads)
    try:
        h.reduce_scatter(grads)
        raise SystemExit("FAIL: the injected failure did not raise")
    except RuntimeError as e:
        if "TIDE_FAULT=abort" not in str(e):
            raise
    t0 = time.monotonic()
    comm.close()
    del h, arena  # the window's release must not drain either
    dt = time.monotonic() - t0
    ok = dt < 30.0
    print(f"rank {dist.get_rank()}: close after a failed call returned in {dt:.2f} s", flush=True)
    if dist.get_rank() == 0:
        print("PASS" if ok else "FAIL", flush=True)
    # The failed call's device wait stays queued: no further work on this device, no process-group teardown.
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
