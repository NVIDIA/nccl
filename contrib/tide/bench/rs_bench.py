# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduce-scatter benchmark: TIDE (gradients read in place) vs stock FSDP2 (chunk_cat + NCCL
reduce_scatter_tensor on the process group's library) vs the bare NCCL reduce-scatter.

busbw = (nranks-1) * out_bytes / t (same convention as the all-gather bench, S = fp32 output per
rank). --gemm N runs N bf16 8192^3 GEMMs on a second stream during the timed loop and reports their
slowdown against the same GEMMs run alone (SM interference of the collective).

torchrun --standalone --nproc_per_node 8 bench/rs_bench.py --shard-mib 32 128 --arms tide stock
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
from tide.testing import sub_group


def make_grads(world: int, out_elems: int, segments: int, dtype, dev, seed: int) -> list[torch.Tensor]:
    g = torch.Generator(device=dev).manual_seed(seed)
    grads = []
    total = out_elems * world
    rest = 2048
    for i in range(segments):
        share = total // (segments - i)
        dim0 = max(world, share // rest)
        grads.append(torch.randn(dim0, rest, generator=g, device=dev).to(dtype))
        total -= dim0 * rest
    return grads


def padded_out_elems(grads, world):
    return sum(((g.shape[0] + world - 1) // world) * (g.numel() // g.shape[0]) for g in grads)


class Gemm:
    def __init__(self, dev, n=8192):
        self.a = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
        self.b = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
        self.stream = torch.cuda.Stream(device=dev)

    def step(self):
        torch.matmul(self.a, self.b)

    def run(self, count: int) -> float:
        """Wall ms for `count` load steps on the side stream."""
        with torch.cuda.stream(self.stream):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(self.stream)
            for _ in range(count):
                self.step()
            e.record(self.stream)
        e.synchronize()
        return s.elapsed_time(e)


class H2D(Gemm):
    """256 MiB pinned host -> device copies: copy engine + the GPU's PCIe ingress."""

    def __init__(self, dev, mib=256):
        self.src = torch.empty(mib << 20, dtype=torch.uint8, pin_memory=True)
        self.dst = torch.empty(mib << 20, dtype=torch.uint8, device=dev)
        self.stream = torch.cuda.Stream(device=dev)

    def step(self):
        self.dst.copy_(self.src, non_blocking=True)


class HBM(Gemm):
    """bf16 adds over 1 GiB operands: HBM traffic issued by SMs, no copy engine."""

    def __init__(self, dev, mib=1024):
        n = (mib << 20) // 2
        self.a, self.b, self.c = (torch.ones(n, dtype=torch.bfloat16, device=dev) for _ in range(3))
        self.stream = torch.cuda.Stream(device=dev)

    def step(self):
        torch.add(self.a, self.b, out=self.c)


def timed(fn, iters: int, warmup: int, dev, gemm: Gemm | None, gemm_n: int, sync_each: bool = False):
    stream = torch.cuda.current_stream(dev)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    host = []
    gemm_ms = None
    if gemm is not None:
        # GEMMs alone (after a warm-up), then the same GEMMs issued right after the collective loop
        # starts; the loop must outlast them (pick --iters accordingly)
        gemm.run(5)
        alone = gemm.run(gemm_n)
        torch.cuda.synchronize()
        dist.barrier()
    for i in range(iters):
        ev[i][0].record(stream)
        t0 = time.perf_counter()
        fn()
        host.append((time.perf_counter() - t0) * 1e6)
        ev[i][1].record(stream)
        if sync_each:
            torch.cuda.synchronize()
        if gemm is not None and i == 2:
            with torch.cuda.stream(gemm.stream):
                gs, ge = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                gs.record(gemm.stream)
                for _ in range(gemm_n):
                    gemm.step()
                ge.record(gemm.stream)
    torch.cuda.synchronize()
    if gemm is not None:
        gemm_ms = (alone, gs.elapsed_time(ge))
    ms = sorted(a.elapsed_time(b) for a, b in ev)
    return ms, host, gemm_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, nargs="+", default=[32.0])
    ap.add_argument("--segments", type=int, nargs="+", default=[16])
    ap.add_argument("--arms", nargs="+", default=["tide", "stock", "stock_rs"],
                    help="tide | stock (chunk_cat + reduce_scatter_tensor) | stock_rs (bare) | symm / symm_rs (the same on "
                         "buffers in a symmetric NCCL window: NCCL's symmetric-kernel reduce-scatter)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--gemm", type=int, default=0, help="side-stream load steps during the loop (0 = off)")
    ap.add_argument("--load", choices=["gemm", "h2d", "hbm"], default="gemm", help="kind of side-stream load step")
    ap.add_argument("--ep", type=int, default=1, help="reduce over the expert-style subgroup r %% ep (tide / stock / stock_rs)")
    ap.add_argument("--out", default="")
    ap.add_argument("--profile", action="store_true", help="rank 0: kineto table of GPU ops for the timed loop")
    ap.add_argument("--sync-each", action="store_true", help="synchronize after every call (true host issue time)")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    if args.ep > 1 and set(args.arms) - {"tide", "stock", "stock_rs"}:
        raise SystemExit("--ep supports the arms tide, stock, stock_rs")
    group = sub_group(args.ep)
    world = dist.get_world_size(group)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    comm = tide.TideComm(group, device=dev) if "tide" in args.arms else None
    gemm = {"gemm": Gemm, "h2d": H2D, "hbm": HBM}[args.load](dev) if args.gemm else None
    results = []
    pools = []  # symmetric pools stay registered until the end (deregistration is collective)
    for mib in args.shard_mib:
        for segs in args.segments:
            out_elems = int(mib * 2**20) // 4
            grads = make_grads(world, out_elems, segs, dtype, dev, 7)
            out_n = padded_out_elems(grads, world)
            out_bytes = out_n * 4
            row = {"shard_mib": mib, "segments": segs, "dtype": args.dtype, "world": world, "ep": args.ep,
                   "nodes": comm.n_nodes if comm else None, "out_mib": out_bytes / 2**20, "gemm_n": args.gemm, "load": args.load if args.gemm else None,
                   "rs_tuning": comm.rs_tuning if comm else None,  # (sum CTAs, landing CTAs, k, chunk bytes)
                   "nic_pace": os.environ.get("TIDE_NIC_PACE", "auto"), "pace_lead_mib": os.environ.get("TIDE_NIC_PACE_LEAD_MIB", "8")}
            for arm in args.arms:
                torch.cuda.synchronize()
                dist.barrier()
                free0 = torch.cuda.mem_get_info(dev)[0]  # device bytes the arm itself takes (arena, windows, buffers)
                if arm == "tide":
                    stage = comm.rs_stage_bytes(grads)
                    arena = comm.alloc_arena(stage)
                    h = comm.register_rs(arena, 0, grads)
                    out = torch.empty(h.out_numel, dtype=torch.float32, device=dev)
                    fn = lambda: h.reduce_scatter(grads, out=out)  # noqa: E731
                    row["tide_stage_mib"] = stage / 2**20
                    row["tide_chunks"] = h.plan.chunks
                elif arm == "stock":
                    inp = torch.empty(out_n * world, dtype=torch.float32, device=dev).view(world, -1)
                    out = torch.empty(out_n, dtype=torch.float32, device=dev)

                    def fn():
                        torch._chunk_cat(grads, dim=0, num_chunks=world, out=inp)  # bf16 into fp32, as FSDP2 does
                        dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM, group=group)
                elif arm == "stock_rs":
                    inp = torch.randn(out_n * world, dtype=torch.float32, device=dev)
                    out = torch.empty(out_n, dtype=torch.float32, device=dev)
                    fn = lambda: dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM, group=group)  # noqa: E731
                elif arm in ("symm", "symm_rs"):
                    backend = dist.group.WORLD._get_backend(dev)
                    pool = torch.cuda.MemPool(backend.mem_allocator)
                    with torch.cuda.use_mem_pool(pool):
                        inp = torch.empty(out_n * world, dtype=torch.float32, device=dev).view(world, -1)
                        out = torch.empty(out_n, dtype=torch.float32, device=dev)
                    backend.register_mem_pool(pool, symm=True)
                    pools.append(pool)
                    if arm == "symm":
                        def fn():
                            torch._chunk_cat(grads, dim=0, num_chunks=world, out=inp)  # bf16 into fp32, as FSDP2 does
                            dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
                    else:
                        inp.normal_()
                        fn = lambda: dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)  # noqa: E731
                else:
                    raise SystemExit(f"unknown arm {arm}")
                if args.profile:
                    from torch.profiler import ProfilerActivity, profile
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        ms, host, gemm_ms = timed(fn, args.iters, args.warmup, dev, gemm, args.gemm, args.sync_each)
                    if rank == 0:
                        print(f"--- {arm} shard={mib} segs={segs}")
                        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
                        if args.out:
                            prof.export_chrome_trace(f"{args.out}.{arm}_{int(mib)}_{segs}.trace.json")
                else:
                    ms, host, gemm_ms = timed(fn, args.iters, args.warmup, dev, gemm, args.gemm, args.sync_each)
                med = statistics.median(ms)
                row[f"{arm}_ms"] = med
                row[f"{arm}_min_ms"] = ms[0]
                row[f"{arm}_p90_ms"] = ms[int(0.9 * (len(ms) - 1))]
                row[f"{arm}_busbw_GBs"] = (world - 1) * out_bytes / (med * 1e-3) / 1e9
                row[f"{arm}_host_us_med"] = statistics.median(host)
                torch.cuda.synchronize()
                row[f"{arm}_dev_mib"] = (free0 - torch.cuda.mem_get_info(dev)[0]) / 2**20
                if gemm_ms:
                    row[f"{arm}_gemm_alone_ms"] = gemm_ms[0]
                    row[f"{arm}_gemm_during_ms"] = gemm_ms[1]
                    row[f"{arm}_gemm_slowdown_pct"] = 100 * (gemm_ms[1] / gemm_ms[0] - 1)
                torch.cuda.synchronize()
                dist.barrier()
                if arm == "tide":
                    del h, arena
                inp = out = fn = None  # the next arm's device accounting starts from what the grads alone take
                torch.cuda.empty_cache()
            results.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
    if rank == 0 and args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    dist.barrier()
    for pool in pools:  # before the communicator goes (collective)
        dist.group.WORLD._get_backend(dev).deregister_mem_pool(pool)
    pools.clear()
    if comm is not None:
        comm.close()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
