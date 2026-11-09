# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host time of back-to-back all-gathers separated by compute: before each call the current stream runs
`gap` x (the call's device time) of GPU work (a spin kernel), as a layer's forward would. Two landing
regions alternate, as FSDP's two ring slots do. The all-gather counterpart of rs_host_gap.py.

torchrun ... bench/ag_host_gap.py --shard-mib 32 --segments 30 --gaps 0 0.5 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

import tide
from tide.testing import make_segments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, default=32.0)
    ap.add_argument("--segments", type=int, default=30)
    ap.add_argument("--small-frac", type=float, default=0.5)
    ap.add_argument("--gaps", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    comm = tide.TideComm(device=dev)
    S = int(args.shard_mib * 2**20) // 16 * 16
    staged = comm.n_nodes > 1
    lane = (S * world + (comm.stage_bytes(S) if staged else 0) + 4095) // 4096 * 4096
    arena = comm.alloc_arena(2 * lane)
    sizes, offsets = tide.byte_table(make_segments(S, args.segments, args.small_frac, seed=args.segments * 31 + 7))
    handles = [comm.register_ag(arena, i * lane, sizes, offsets, stage_off=(i * lane + S * world) if staged else -1)
               for i in range(2)]
    stream = torch.cuda.current_stream()
    it = [0]

    def call():
        handles[it[0] % 2].all_gather(stream)
        it[0] += 1

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):
        call()
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
            call()
            host.append((time.perf_counter() - t0) * 1e3)
        b.record()
        torch.cuda.synchronize()
        per_iter = a.elapsed_time(b) / args.iters
        rows.append({"gap": gap, "gap_ms": gap * call_ms, "call_ms": call_ms, "host_ms_med": statistics.median(host),
                     "host_ms_max": max(host), "iter_ms": per_iter, "iter_minus_gap_ms": per_iter - gap * call_ms})
    if rank == 0:
        tag = {"nodes": comm.n_nodes, "world": world, "shard_mib": args.shard_mib, "segments": args.segments,
               "descriptors": handles[0].num_descriptors}
        for r in rows:
            print(json.dumps({**tag, **{k: round(v, 3) for k, v in r.items()}}), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                for r in rows:
                    f.write(json.dumps({**tag, **r}) + "\n")
    dist.barrier()
    del handles
    comm.free_arena(arena)
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
