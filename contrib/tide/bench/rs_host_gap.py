# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host time of back-to-back reduce-scatters separated by compute: before each call the current stream
runs `gap` x (the call's device time) of GPU work (a spin kernel), as a layer's backward would. Shows at
which compute gap the issuing thread stops waiting on the copy-engine queue.

torchrun ... bench/rs_host_gap.py --shard-mib 32 --gaps 0 0.5 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

import tide

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_bench import make_grads  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, default=32.0)
    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--gaps", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    comm = tide.TideComm(device=dev)
    grads = make_grads(world, int(args.shard_mib * 2**20) // 4, args.segments, torch.bfloat16, dev, 7)
    arena = comm.alloc_arena(comm.rs_stage_bytes(grads))
    h = comm.register_rs(arena, 0, grads)
    out = torch.empty(h.out_numel, dtype=torch.float32, device=dev)

    # device time of one call, and spin-kernel cycles per ms on this GPU
    for _ in range(5):
        h.reduce_scatter(grads, out=out)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):
        h.reduce_scatter(grads, out=out)
    b.record()
    torch.cuda.synchronize()
    call_ms = a.elapsed_time(b) / 10
    a.record()
    torch.cuda._sleep(100_000_000)
    b.record()
    torch.cuda.synchronize()
    cycles_per_ms = 100_000_000 / a.elapsed_time(b)
    t = torch.tensor([call_ms, cycles_per_ms], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    call_ms, cycles_per_ms = float(t[0]), float(t[1])

    rows = []
    for gap in args.gaps:
        cycles = int(gap * call_ms * cycles_per_ms)
        torch.cuda.synchronize()
        dist.barrier()
        host = []
        a.record()
        for _ in range(args.iters):
            if cycles:
                torch.cuda._sleep(cycles)
            t0 = time.perf_counter()
            h.reduce_scatter(grads, out=out)
            host.append((time.perf_counter() - t0) * 1e3)
        b.record()
        torch.cuda.synchronize()
        per_iter = a.elapsed_time(b) / args.iters
        rows.append({"gap": gap, "gap_ms": gap * call_ms, "call_ms": call_ms, "host_ms_med": statistics.median(host),
                     "host_ms_max": max(host), "iter_ms": per_iter, "iter_minus_gap_ms": per_iter - gap * call_ms})
    if rank == 0:
        for r in rows:
            print(json.dumps({"nodes": comm.n_nodes, "world": world, "shard_mib": args.shard_mib,
                              "rs_tuning": comm.rs_tuning, **{k: round(v, 3) for k, v in r.items()}}), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                for r in rows:
                    f.write(json.dumps({"nodes": comm.n_nodes, "shard_mib": args.shard_mib, **r}) + "\n")
    dist.barrier()
    del h
    comm.free_arena(arena)
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
