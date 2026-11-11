# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute-stream interference probe: how much a reduce-scatter running on its own streams slows
compute on the current stream, per workload, with the RS kept busy for the whole measurement.

For each workload: compute alone (median per iteration) -> compute while reduce-scatters are issued
back-to-back on a separate stream (host interleaves one RS per ~one RS worth of compute, so both
streams stay busy; only iterations that ran entirely inside the RS window count) -> compute alone
again (drift check). Also reports the RS time alone vs during compute.

torchrun ... bench/interference.py --arm tide --shard-mib 32 --workloads gemm_big,gemm_small,ewise
Arms: none (RS not run; sanity), tide (TIDE_RS_* env select CTAs / mode), stock_rs (fp32 packed
input, reduce_scatter_tensor), stock (torch._chunk_cat bf16->fp32 + reduce_scatter_tensor), symm_rs
(packed fp32 in a symmetric window: NCCL's symmetric kernel), symm (the _chunk_cat into that window
first: FSDP2 with symmetric memory, benchmarked with NCCL_CTA_POLICY=2).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from rs_bench import make_grads, padded_out_elems

import tide


class Workload:
    def __init__(self, name: str, dev):
        self.name = name
        if name == "gemm_big":  # SM-bound: 2*16384*8192*8192 = 2.2 TFLOP, ~1 ms
            self.a = torch.randn(16384, 8192, dtype=torch.bfloat16, device=dev)
            self.b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
            self.fn = lambda: torch.matmul(self.a, self.b)
        elif name == "gemm_small":  # weight-streaming: M=256, 134 MB of weights per call
            self.a = torch.randn(256, 8192, dtype=torch.bfloat16, device=dev)
            self.b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
            self.fn = lambda: torch.matmul(self.a, self.b)
        elif name == "ewise":  # pure HBM: PROBE_EWISE_MIB (256) read + write; 64 fits L2, 4096 streams
            mib = int(os.environ.get("PROBE_EWISE_MIB", "256"))
            self.x = torch.randn(mib * 2**19, dtype=torch.bfloat16, device=dev)
            self.fn = lambda: self.x.mul_(1.0001)
        else:
            raise ValueError(name)


def iter_stats(starts, ends, lo=None, hi=None, ref=None):
    """Per-iteration ms from event pairs (median, mean, count); if lo/hi (ms relative to ref) are
    given, only iterations entirely inside [lo, hi]. The mean is the throughput view (a burst that
    hits a minority of iterations moves the mean, not the median)."""
    vals = []
    for s, e in zip(starts, ends):
        if lo is not None:
            t0, t1 = ref.elapsed_time(s), ref.elapsed_time(e)
            if t0 < lo or t1 > hi:
                continue
        vals.append(s.elapsed_time(e))
    if not vals:
        return float("nan"), float("nan"), 0
    return statistics.median(vals), statistics.fmean(vals), len(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="tide", choices=["none", "tide", "stock_rs", "stock", "symm_rs", "symm"])
    ap.add_argument("--shard-mib", type=float, default=32)
    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--workloads", default="gemm_big,gemm_small,ewise")
    ap.add_argument("--rs-iters", type=int, default=40, help="reduce-scatters issued during each workload")
    ap.add_argument("--alone-ms", type=float, default=150, help="compute-alone window per phase")
    ap.add_argument("--compute-priority", type=int, default=0, help="torch stream priority for compute (0 lowest, -1 higher)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    compute = torch.cuda.Stream(device=dev, priority=args.compute_priority)
    rs_stream = torch.cuda.Stream(device=dev)

    out_elems = int(args.shard_mib * 2**20) // 4
    grads = make_grads(world, out_elems, args.segments, torch.bfloat16, dev, 7)
    out_n = padded_out_elems(grads, world)
    out = torch.empty(out_n, dtype=torch.float32, device=dev)
    comm = None
    if args.arm == "tide":
        comm = tide.TideComm(device=dev)
        stage = comm.rs_stage_bytes(grads)
        arena = comm.alloc_arena(stage)
        h = comm.register_rs(arena, 0, grads)
        rs = lambda: h.reduce_scatter(grads, out=out, stream=rs_stream)  # noqa: E731
    elif args.arm == "stock_rs":
        inp = torch.randn(out_n * world, dtype=torch.float32, device=dev)
        rs = lambda: dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)  # noqa: E731
    elif args.arm == "stock":
        inp = torch.empty(out_n * world, dtype=torch.float32, device=dev).view(world, -1)

        def rs():
            torch._chunk_cat(grads, dim=0, num_chunks=world, out=inp)  # bf16 into fp32, as FSDP2 does
            dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
    elif args.arm in ("symm", "symm_rs"):  # buffers in a symmetric window: NCCL's symmetric-kernel reduce-scatter
        dist.barrier()  # the pool needs an initialized NCCL communicator
        backend = dist.group.WORLD._get_backend(dev)
        pool = torch.cuda.MemPool(backend.mem_allocator)
        with torch.cuda.use_mem_pool(pool):
            inp = torch.empty(out_n * world, dtype=torch.float32, device=dev).view(world, -1)
            out = torch.empty(out_n, dtype=torch.float32, device=dev)
        backend.register_mem_pool(pool, symm=True)
        if args.arm == "symm":
            def rs():
                torch._chunk_cat(grads, dim=0, num_chunks=world, out=inp)  # bf16 into fp32, as FSDP2 does
                dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
        else:
            inp.normal_()
            rs = lambda: dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)  # noqa: E731
    else:
        rs = None

    def run_rs():
        with torch.cuda.stream(rs_stream):
            rs()

    row = {"arm": args.arm, "world": world, "nodes": comm.n_nodes if comm else None, "shard_mib": args.shard_mib,
           "segments": args.segments, "rs_tuning": comm.rs_tuning if comm else None,
           "compute_priority": args.compute_priority, "rs_iters": args.rs_iters}

    # RS alone (min of 10 individual calls after warm-up, so a single RoCE tail spike cannot inflate the window)
    if rs is not None:
        for _ in range(5):
            run_rs()
        torch.cuda.synchronize()
        dist.barrier()
        alone_evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(10)]
        for s0, e0 in alone_evs:
            s0.record(rs_stream)
            run_rs()
            e0.record(rs_stream)
        torch.cuda.synchronize()
        alone_rs = sorted(s0.elapsed_time(e0) for s0, e0 in alone_evs)
        rs_alone_ms = alone_rs[0]
        row["rs_alone_ms"] = rs_alone_ms
        row["rs_alone_med_ms"] = statistics.median(alone_rs)
    else:
        rs_alone_ms = 5.0

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
                n = max(20, int(args.alone_ms / per))
                for _ in range(n):
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

        during = during_mean = float("nan")
        n_in = 0
        rs_during_ms = float("nan")
        if rs is not None:
            dist.barrier()
            torch.cuda.synchronize()
            ref = torch.cuda.Event(enable_timing=True)
            ref.record(compute)
            rs_s, rs_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starts, ends, rs_evs = [], [], []
            # Keep 2-3 RS calls and <= 128 compute iterations (< 400 driver queue entries, well below the ~1000-entry
            # CUDA host queue limit) in flight so neither stream ever starves or blocks the host from feeding the other.
            comp_inflight = min(128, max(8, int(20.0 / max(alone1, 0.005))))

            def step_compute():
                with torch.cuda.stream(compute):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(compute)
                    w.fn()
                    e.record(compute)
                    starts.append(s)
                    ends.append(e)

            def step_rs():
                run_rs()
                ev = torch.cuda.Event()
                ev.record(rs_stream)
                rs_evs.append(ev)
                if len(rs_evs) == args.rs_iters:
                    rs_e.record(rs_stream)

            rs_s.record(rs_stream)
            for _ in range(min(3, args.rs_iters)):
                step_rs()
            for _ in range(comp_inflight):
                step_compute()
            while len(rs_evs) < args.rs_iters or not rs_e.query():
                if len(rs_evs) < args.rs_iters and rs_evs[-2].query():
                    step_rs()
                for _ in range(8):
                    # rs_e is recorded only after the last RS is issued, and an unrecorded event queries as complete
                    rs_running = len(rs_evs) < args.rs_iters or not rs_e.query()
                    if rs_running and ends[-comp_inflight].query():
                        step_compute()
                    else:
                        break
            torch.cuda.synchronize()
            lo, hi = ref.elapsed_time(rs_s), ref.elapsed_time(rs_e)
            # drop the first and last RS worth of the window (pipeline fill / drain)
            during, during_mean, n_in = iter_stats(starts, ends, lo + rs_alone_ms, hi - rs_alone_ms, ref)
            rs_during_ms = (hi - lo) / args.rs_iters
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
        row[f"{name}_rs_during_ms"] = rs_during_ms
        row[f"{name}_rs_slowdown_pct"] = 100 * (rs_during_ms / rs_alone_ms - 1) if rs is not None else float("nan")
        del w
        torch.cuda.empty_cache()

    # worst rank matters for a step: gather slowdowns
    keys = [k for k in row if k.endswith("_slowdown_pct") or k.endswith("_slowdown_mean_pct")]
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
        comm.close()
    dist.barrier()
    if args.arm in ("symm", "symm_rs"):
        torch.cuda.synchronize()
        dist.barrier()
        backend.deregister_mem_pool(pool)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
