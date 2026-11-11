# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All-gather interference probe: how much an all-gather running on its own stream slows compute on
another stream, and how much the compute slows the all-gather. The all-gather form is the FSDP2 one
(one call per parameter group, param-major landing).

Same phase structure and JSON row style as bench/interference.py (the reduce-scatter version, whose
Workload / iter_stats this imports): compute alone -> compute while all-gathers are issued
back-to-back on a separate stream -> compute alone again (drift check).

torchrun ... bench/ag_interference.py --arm tide --shard-mib 64 --segments 16 --workloads gemm_big,ewise
Arms: none (no all-gather; sanity), tide (TIDE_* env select mode / pacing), stock
(dist.all_gather_into_tensor, rank-major, no copy-out). PROBE_EWISE_MIB sizes the ewise workload.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from interference import Workload, iter_stats

import tide
from tide.testing import make_segments


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="tide", choices=["none", "tide", "stock"])
    ap.add_argument("--shard-mib", type=float, default=64)
    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--small-frac", type=float, default=0.5)
    ap.add_argument("--workloads", default="gemm_big,gemm_small,ewise")
    ap.add_argument("--ag-iters", type=int, default=40, help="all-gathers issued during each workload")
    ap.add_argument("--alone-ms", type=float, default=150, help="compute-alone window per phase")
    ap.add_argument("--compute-priority", type=int, default=0, help="torch stream priority for compute (0 lowest, -1 higher)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    compute = torch.cuda.Stream(device=dev, priority=args.compute_priority)
    ag_stream = torch.cuda.Stream(device=dev)

    S = int(args.shard_mib * 2**20) // 16 * 16
    sizes, offsets = tide.byte_table(make_segments(S, args.segments, args.small_frac, seed=args.segments * 31 + 7))
    shard = torch.randint(0, 256, (S,), dtype=torch.uint8, device=dev)
    comm = arena = None
    if args.arm == "tide":
        comm = tide.TideComm(device=dev, num_rma_ctx=int(os.environ.get("TIDE_RMA_CTX", "0")))
        staged = comm.n_nodes > 1 or os.environ.get("TIDE_MC_STAGE", "0") == "1"
        stage = comm.stage_bytes(S) if staged else 0
        arena = comm.alloc_arena((S * world + stage + 4095) // 4096 * 4096)
        h = comm.register_ag(arena, 0, sizes, offsets, stage_off=(S * world) if staged else -1)
        torch._foreach_copy_(h.own_pieces(arena), [shard.narrow(0, o, s) for s, o in zip(sizes, offsets)])
        ag = lambda: h.all_gather(ag_stream)  # noqa: E731
    elif args.arm == "stock":
        out = torch.empty(S * world, dtype=torch.uint8, device=dev)

        def ag():
            with torch.cuda.stream(ag_stream):
                dist.all_gather_into_tensor(out, shard)
    else:
        ag = None

    row = {"arm": args.arm, "world": world, "nodes": comm.n_nodes if comm else None, "shard_mib": args.shard_mib,
           "segments": args.segments,
           "nic_pace": os.environ.get("TIDE_NIC_PACE", "auto"),
           "compute_priority": args.compute_priority, "ag_iters": args.ag_iters,
           "ewise_mib": int(os.environ.get("PROBE_EWISE_MIB", "256"))}

    # all-gather alone
    if ag is not None:
        for _ in range(3):
            ag()
        torch.cuda.synchronize()
        dist.barrier()
        s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s0.record(ag_stream)
        for _ in range(10):
            ag()
        e0.record(ag_stream)
        torch.cuda.synchronize()
        ag_alone_ms = s0.elapsed_time(e0) / 10
        row["ag_alone_ms"] = ag_alone_ms
        row["ag_alone_busbw_GBs"] = (world - 1) * S / (ag_alone_ms * 1e-3) / 1e9
    else:
        ag_alone_ms = 5.0

    for name in args.workloads.split(","):
        w = Workload(name, dev)
        # clocks ramp under load: warm the workload for ~300 ms before any measurement
        with torch.cuda.stream(compute):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(compute)
            w.fn()
            e.record(compute)
            e.synchronize()
            for _ in range(max(20, int(300 / max(s.elapsed_time(e), 0.005)))):
                w.fn()
        torch.cuda.synchronize()

        def alone_phase():
            starts, ends = [], []
            with torch.cuda.stream(compute):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(compute)
                w.fn()
                e.record(compute)
                e.synchronize()
                per = max(s.elapsed_time(e), 0.005)
                for _ in range(max(20, int(args.alone_ms / per))):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(compute)
                    w.fn()
                    e.record(compute)
                    starts.append(s)
                    ends.append(e)
            torch.cuda.synchronize()
            return iter_stats(starts, ends)[:2]

        dist.barrier()
        alone1, alone1_mean = alone_phase()
        per_ag = max(1, int(ag_alone_ms / max(alone1, 0.005)))  # compute iterations per all-gather call

        during = during_mean = ag_during_ms = float("nan")
        n_in = 0
        if ag is not None:
            dist.barrier()
            torch.cuda.synchronize()
            ref = torch.cuda.Event(enable_timing=True)
            ref.record(compute)
            ag_s, ag_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starts, ends = [], []
            ag_s.record(ag_stream)
            for _ in range(args.ag_iters):
                ag()
                with torch.cuda.stream(compute):
                    for _ in range(per_ag):
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record(compute)
                        w.fn()
                        e.record(compute)
                        starts.append(s)
                        ends.append(e)
            ag_e.record(ag_stream)
            torch.cuda.synchronize()
            lo, hi = ref.elapsed_time(ag_s), ref.elapsed_time(ag_e)
            # drop the first and last all-gather worth of the window (pipeline fill / drain)
            during, during_mean, n_in = iter_stats(starts, ends, lo + ag_alone_ms, hi - ag_alone_ms, ref)
            ag_during_ms = (hi - lo) / args.ag_iters
        dist.barrier()
        alone2, alone2_mean = alone_phase()
        base = (alone1 + alone2) / 2
        base_mean = (alone1_mean + alone2_mean) / 2
        row[f"{name}_alone_ms"] = base
        row[f"{name}_alone_drift_pct"] = 100 * (alone2 / alone1 - 1)
        row[f"{name}_during_ms"] = during
        row[f"{name}_slowdown_pct"] = 100 * (during / base - 1)
        row[f"{name}_slowdown_mean_pct"] = 100 * (during_mean / base_mean - 1)
        row[f"{name}_n_in_window"] = n_in
        row[f"{name}_ag_during_ms"] = ag_during_ms
        row[f"{name}_ag_slowdown_pct"] = 100 * (ag_during_ms / ag_alone_ms - 1) if ag is not None else float("nan")
        del w
        torch.cuda.empty_cache()

    # worst rank matters for a step: gather slowdowns (both directions)
    keys = [k for k in row if k.endswith("_slowdown_pct")]
    vals = torch.tensor([row[k] for k in keys], dtype=torch.float64, device=dev)
    gathered = [torch.empty_like(vals) for _ in range(world)]
    dist.all_gather(gathered, vals)
    allv = torch.stack(gathered)
    for i, k in enumerate(keys):
        col = allv[:, i]
        col = col[~torch.isnan(col)]
        if col.numel():
            row[k + "_max"] = col.max().item()
            row[k + "_med"] = col.median().item()
    if rank == 0:
        print(json.dumps(row))
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
    if comm is not None:
        del arena, h  # drop the views so close() releases the window now
        comm.close()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
