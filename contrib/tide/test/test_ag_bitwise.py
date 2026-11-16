# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise check of TIDE all-gather against stock all-gather + parameter-major permutation.

torchrun --standalone --nproc_per_node 8 test/test_ag_bitwise.py --shard-mib 155 --segments 30 --iters 20
TIDE_FAULT=nowrite ... must report a detected fault (exit 0 only if the check catches it).
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
from tide.testing import make_segments, param_major_permute, sub_group


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, default=155.0)
    ap.add_argument("--segments", type=int, default=30)
    ap.add_argument("--small-frac", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-rma-ctx", type=int, default=0)
    ap.add_argument("--staged", action="store_true", help="packed-shard stage + copy-in (forced on multi-node)")
    ap.add_argument("--own-sep", action="store_true", help="own shard at its own offset (own_off), as the FSDP adapter places it")
    ap.add_argument("--ep", type=int, default=1, help="run over the subgroup r %% ep (expert-group layout)")
    ap.add_argument("--decouple", action="store_true", help="decoupled calls (timed to the done event)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    fault = os.environ.get("TIDE_FAULT", "")

    group = sub_group(args.ep)
    world = dist.get_world_size(group)
    comm = tide.TideComm(group=group, device=dev, num_rma_ctx=args.num_rma_ctx)
    shard_bytes = int(args.shard_mib * 2**20) // 16 * 16
    sizes = make_segments(shard_bytes, args.segments, args.small_frac, args.seed)
    sizes, offsets = tide.byte_table(sizes)
    region_bytes = shard_bytes * world
    staged = args.staged or comm.n_nodes > 1
    stage_bytes = comm.stage_bytes(shard_bytes, own_separate=args.own_sep) if staged else 0
    own_bytes = shard_bytes if args.own_sep else 0
    lane_bytes = (region_bytes + stage_bytes + own_bytes + 4095) // 4096 * 4096
    arena = comm.alloc_arena(lane_bytes * args.lanes)
    handles = [comm.register_ag(arena, lane * lane_bytes, sizes, offsets,
                                stage_off=(lane * lane_bytes + region_bytes) if staged else -1,
                                own_off=(lane * lane_bytes + region_bytes + stage_bytes) if args.own_sep else -1)
               for lane in range(args.lanes)]

    if rank == 0:
        print(json.dumps({
            "nccl_runtime": tide._C.nccl_runtime_version(), "nccl_compiled": tide._C.nccl_compiled_version,
            "props": comm.props, "lsa_peers": comm.lsa_peers, "world": world, "nodes": comm.n_nodes, "ep": args.ep, "direct": handles[0].plan.direct,
            "shard_bytes": shard_bytes, "segments": len(sizes), "min_seg": min(sizes), "max_seg": max(sizes),
            "descriptors": handles[0].num_descriptors, "lanes": args.lanes, "staged": staged,
            "rma_ctx": comm.rma_ctx, "rounds": handles[0].plan.rounds,
            "nic_groups": comm.nic_groups, "nic_chunks": handles[0].plan.nic_chunks, "fault": fault,
        }), flush=True)

    # Non-default stream: cudaMemcpyBatchAsync rejects the legacy NULL stream (FSDP uses side streams too).
    stream = torch.cuda.Stream(device=dev)
    torch.cuda.set_stream(stream)
    mism, host_us, gpu_ms, stock_ms = [], [], [], []
    for it in range(args.iters):
        h = handles[it % args.lanes]
        g = torch.Generator(device=dev).manual_seed(args.seed * 7919 + it * 131 + rank)
        shard = torch.randint(0, 256, (shard_bytes,), dtype=torch.uint8, device=dev, generator=g)
        # copy-in: this rank's pieces to their landing slots (in place) or to its packed stage slot
        torch._foreach_copy_(h.own_pieces(arena), [shard.narrow(0, offsets[p], sizes[p]) for p in range(len(sizes))])
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        e0.record(stream)
        if args.decouple:
            h.all_gather_on(stream, done_event=e1)
        else:
            h.all_gather(stream)
            e1.record(stream)
        host_us.append((time.perf_counter() - t0) * 1e6)
        # reference on the stock path (torch's own communicator)
        s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s0.record(stream)
        exp = param_major_permute(shard, sizes, offsets, group)
        s1.record(stream)
        torch.cuda.synchronize()
        gpu_ms.append(e0.elapsed_time(e1))
        stock_ms.append(s0.elapsed_time(s1))
        bad = int((h.region(arena) != exp).sum().item())
        mism.append(bad)
        del shard, exp

    total_bad = sum(mism)
    bad_t = torch.tensor([total_bad], device=dev)
    dist.all_reduce(bad_t)
    total_bad_all = int(bad_t.item())
    moved = (world - 1) * shard_bytes
    res = {
        "rank": rank, "mismatch_bytes_all_ranks": total_bad_all,
        "tide_gpu_ms_med": statistics.median(gpu_ms[1:] or gpu_ms), "tide_busbw_GBs": moved / (statistics.median(gpu_ms[1:] or gpu_ms) * 1e-3) / 1e9,
        "stock_ag_plus_perm_ms_med": statistics.median(stock_ms[1:] or stock_ms),
        "host_enqueue_us_med": statistics.median(host_us[1:] or host_us), "host_enqueue_us_max": max(host_us),
        "calls": handles[0].calls + handles[-1].calls,
    }
    if rank == 0:
        print(json.dumps(res), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps({"args": vars(args), **res}) + "\n")
    ok = (total_bad_all == 0) if not fault else (total_bad_all > 0)
    if rank == 0:
        print(("PASS" if ok else "FAIL") + (f" (fault '{fault}' detected)" if fault and ok else ""), flush=True)
    dist.barrier()
    del handles, arena
    comm.close()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
