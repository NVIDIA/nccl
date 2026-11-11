# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fence and credit test under an FSDP-like two-stream schedule, no host synchronization.

AG stream:      wait release(lane) -> copy-in -> all_gather(k) -> record ag_ev(k)
compute stream: wait ag_ev(k) -> (slow rank sleeps) -> checksum(lane) -> record release(lane)
(--decouple: the call takes release(lane) for `ready` and records ag_ev(k) itself; the AG stream goes on
without waiting for the other ranks' writes.)

Lanes are reused in a configurable order (`alt`: 0 1 0 1 ..., `burst`: 0 0 1 1 ...; `burst` is the
non-round-robin order that a per-plan credit count would get wrong). Peers may only write into my
lane after my ready(k), which is ordered behind release(lane), i.e. behind my checksum of the
lane's previous tenant; on multi-node a source may only overwrite my stage slot after my credit.
Every recorded checksum must equal the reference recomputed afterwards.

Negatives (exit 0 iff the outcome matches the mode); `slow` delays one rank's data movement by
TIDE_FAULT_US on TIDE_FAULT_RANK, and on its own is the matching positive:
  TIDE_FAULT=nofence          the entry wait is skipped -> the slow consumer's lane is overwritten while read
  TIDE_FAULT=nodone,slow      the exit wait is skipped  -> checksums see the slow writer's pieces missing
  TIDE_FAULT=nocredit,slow    multi-node: the slow rank's delayed scatter is overwritten by the next put into
                              the same stage slot (use --pattern burst)
  TIDE_FAULT=nofree           --decouple: `ready` ignores the release -> the slow consumer's lane is overwritten

torchrun --standalone --nproc_per_node 8 test/test_fence.py
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

import tide
from tide.testing import make_segments, param_major_permute, sub_group


def shard_for(rank: int, it: int, nbytes: int, dev, seed: int) -> torch.Tensor:
    g = torch.Generator(device=dev).manual_seed(seed * 7919 + it * 131 + rank)
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=dev, generator=g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, default=16.0)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--pattern", choices=["alt", "burst"], default="alt")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--sleep-ms", type=float, default=20.0)
    ap.add_argument("--slow-rank", type=int, default=0)
    ap.add_argument("--staged", action="store_true", help="packed-shard stage (forced on multi-node)")
    ap.add_argument("--num-rma-ctx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--ep", type=int, default=1, help="run over the subgroup r %% ep (expert-group layout)")
    ap.add_argument("--decouple", action="store_true")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    fault = os.environ.get("TIDE_FAULT", "")
    expect_bad = bool({"nofence", "nodone", "nocredit", "nofree"} & set(fault.split(",")))

    group = sub_group(args.ep)
    world = dist.get_world_size(group)
    comm = tide.TideComm(group=group, device=dev, num_rma_ctx=args.num_rma_ctx)
    S = int(args.shard_mib * 2**20) // 16 * 16
    sizes, offsets = tide.byte_table(make_segments(S, args.segments, 0.5, args.seed))
    staged = args.staged or comm.n_nodes > 1
    region = S * world
    stage = comm.stage_bytes(S) if staged else 0
    lane_bytes = (region + stage + 4095) // 4096 * 4096
    arena = comm.alloc_arena(lane_bytes * args.lanes)
    handles = [comm.register_ag(arena, lane * lane_bytes, sizes, offsets,
                                stage_off=(lane * lane_bytes + region) if staged else -1)
               for lane in range(args.lanes)]
    order = [(k if args.pattern == "alt" else k // 2) % args.lanes for k in range(args.iters)]

    ag_stream, compute_stream = torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev)
    release = [torch.cuda.Event() for _ in range(args.lanes)]
    done = [torch.cuda.Event() for _ in range(args.lanes)]
    sums = torch.zeros(args.iters, dtype=torch.int64, device=dev)
    sleep_cycles = int(args.sleep_ms * 1e-3 * 1.5e9)  # ~1.5 GHz SM clock, order of magnitude is enough

    dist.barrier()
    for k, lane in enumerate(order):
        h = handles[lane]
        shard = shard_for(rank, k, S, dev, args.seed)
        ag_stream.wait_stream(torch.cuda.current_stream())  # the shard is generated there
        with torch.cuda.stream(ag_stream):
            ag_stream.wait_event(release[lane])  # lane free (my previous compute on it finished)
            torch._foreach_copy_(h.own_pieces(arena), [shard.narrow(0, offsets[p], sizes[p]) for p in range(len(sizes))])
            if args.decouple:
                h.all_gather_on(ag_stream, release[lane], done[lane])
                ag_ev = done[lane]
            else:
                h.all_gather(ag_stream)
                ag_ev = torch.cuda.Event()
                ag_ev.record(ag_stream)
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(ag_ev)
            if rank == args.slow_rank:
                torch.cuda._sleep(sleep_cycles)  # the slow consumer of its lane
            sums[k] = h.region(arena).view(torch.int64).sum()
            release[lane].record(compute_stream)
        shard.record_stream(ag_stream)
    torch.cuda.synchronize()
    dist.barrier()

    # Reference checksums, recomputed deterministically after the fact.
    ref = torch.zeros_like(sums)
    for k in range(args.iters):
        exp = param_major_permute(shard_for(rank, k, S, dev, args.seed), sizes, offsets, group)
        ref[k] = exp.view(torch.int64).sum()
    torch.cuda.synchronize()
    bad = torch.tensor([int((sums != ref).sum().item())], device=dev)
    dist.all_reduce(bad)
    bad_all = int(bad.item())
    ok = (bad_all > 0) if expect_bad else (bad_all == 0)
    if rank == 0:
        print(json.dumps({"fault": fault, "pattern": args.pattern, "lanes": args.lanes, "staged": staged,
                          "nodes": comm.n_nodes, "bad_checksums_all_ranks": bad_all, "iters": args.iters, "world": world}))
        print(("PASS" if ok else "FAIL") + (f" (fault '{fault}' detected)" if expect_bad and ok else ""), flush=True)
    dist.barrier()
    del handles, arena
    comm.close()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
