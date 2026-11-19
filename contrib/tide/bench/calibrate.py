# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the rates tide.solver plans with, on this fleet, and write them as calibration.json.

Single node: K_sum bytes moved per CTA (reduce-scatter with one sum CTA). Multi-node adds the NIC
rate and the per-round fixed cost (all-gather at two shard sizes) and the multicast landing rate per
CTA (reduce-scatter with 1 vs 4 landing CTAs). Knobs the communicator reads at construction are set
through the environment before each TideComm.

    torchrun --standalone --nproc_per_node 8 bench/calibrate.py --out calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

import tide
from tide.solver import Calib

MIB = 2**20


def timed(fn, iters: int, warmup: int) -> float:
    """Median ms of `fn` on the current stream, after a barrier."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    s = torch.cuda.current_stream()
    for a, b in ev:
        a.record(s)
        fn()
        b.record(s)
    torch.cuda.synchronize()
    ms = statistics.median(a.elapsed_time(b) for a, b in ev)
    t = torch.tensor([ms], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # the collective is as slow as its slowest rank
    return float(t)


def rs_time(dev, out_mib: float, segments: int, iters: int, env: dict[str, str]) -> float:
    for k, v in env.items():
        os.environ[k] = v
    comm = tide.TideComm(device=dev)
    world = dist.get_world_size()
    out_elems = int(out_mib * MIB) // 4
    grads, total, rest = [], out_elems * world, 2048
    for i in range(segments):
        dim0 = max(world, (total // (segments - i)) // rest)
        grads.append(torch.randn(dim0, rest, device=dev).to(torch.bfloat16))
        total -= dim0 * rest
    arena = comm.alloc_arena(comm.rs_area_bytes([grads]))
    h = comm.register_rs(arena, 0, grads)
    ms = timed(lambda: h.reduce_scatter(grads), iters, 3)
    del h, arena
    comm.close()
    for k in env:
        del os.environ[k]
    return ms


def ag_time(dev, shard_mib: float, segments: int, iters: int) -> tuple[float, int]:
    comm = tide.TideComm(device=dev)
    world = dist.get_world_size()
    S = int(shard_mib * MIB) // 4096 * 4096
    sizes = [S // segments // 4096 * 4096] * segments
    sizes[-1] += S - sum(sizes)
    offsets = [sum(sizes[:p]) for p in range(segments)]
    region = S * world
    stage = comm.stage_bytes(S)
    arena = comm.alloc_arena(region + stage)
    h = comm.register_ag(arena, 0, sizes, offsets, stage_off=region)
    shard = torch.randn(S // 2, device=dev).to(torch.bfloat16).view(torch.uint8)
    torch._foreach_copy_(h.own_pieces(arena), [shard.narrow(0, offsets[p], sizes[p]) for p in range(segments)])
    ms = timed(lambda: comm.all_gather_on(h, torch.cuda.current_stream()), iters, 3)
    rounds = int(h.plan.rounds)
    del h, arena
    comm.close()
    return ms, rounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--rs-mib", type=float, default=128.0, help="fp32 output per rank for the K_sum / landing legs")
    ap.add_argument("--ag-mib", type=float, nargs=2, default=[8.0, 64.0], help="small and large all-gather shards")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    probe = tide.TideComm(device=dev)
    nodes, local = probe.n_nodes, probe.local_ranks
    probe.close()
    c = Calib()
    report = {"nodes": nodes, "local": local}

    # K_sum per CTA: single-node reduce-scatter is one level-1 sum into the output; bytes moved per
    # rank = 7 staged bf16 pieces + own piece + fp32 write = 5 x output bytes
    if nodes == 1:
        t1 = rs_time(dev, a.rs_mib, a.segments, a.iters, {"TIDE_RS_CTAS": "1"})
        t2 = rs_time(dev, a.rs_mib, a.segments, a.iters, {"TIDE_RS_CTAS": "2"})
        moved = 5 * a.rs_mib * MIB
        c.ksum_gbs_per_cta = moved / (t1 / 1e3) / 1e9
        report["ksum_ms_1cta"], report["ksum_ms_2cta"] = t1, t2
        report["ksum_scaling_2cta"] = t1 / t2
    else:
        # NIC rate and per-round fixed cost from two shard sizes (same round count)
        s0, s1 = a.ag_mib
        t0, R = ag_time(dev, s0, a.segments, a.iters)
        t1, _ = ag_time(dev, s1, a.segments, a.iters)
        wire = (nodes - 1) * (s1 - s0) * MIB
        c.nic_gbs = wire / ((t1 - t0) / 1e3) / 1e9
        c.round_fixed_ms = max(0.0, (t0 - (nodes - 1) * s0 * MIB / (c.nic_gbs * 1e9) * 1e3) / R)
        report.update({"ag_ms_small": t0, "ag_ms_large": t1, "ag_rounds": R})
        # landing per CTA: the last wave's landing is exposed; 1 vs 4 CTAs differ by S / r x (1 - 1/4)
        m1 = rs_time(dev, a.rs_mib, a.segments, a.iters, {"TIDE_RS_MC_CTAS": "1", "TIDE_RS_CTAS": "8"})
        m4 = rs_time(dev, a.rs_mib, a.segments, a.iters, {"TIDE_RS_MC_CTAS": "4", "TIDE_RS_CTAS": "8"})
        if m1 > m4:
            c.land_gbs_per_cta = a.rs_mib * MIB * 0.75 / ((m1 - m4) / 1e3) / 1e9
        report.update({"rs_ms_land1": m1, "rs_ms_land4": m4})
        # K_sum pace: the fewest sum CTAs (with 4 landing CTAs) within 3% of 8
        t8 = rs_time(dev, a.rs_mib, a.segments, a.iters, {"TIDE_RS_CTAS": "8", "TIDE_RS_MC_CTAS": "4"})
        for ctas in (2, 4, 6):
            t = rs_time(dev, a.rs_mib, a.segments, a.iters, {"TIDE_RS_CTAS": str(ctas), "TIDE_RS_MC_CTAS": "4"})
            report[f"rs_ms_sum{ctas}"] = t
            if t <= t8 * 1.03:
                report["ksum_ctas_at_pace"] = ctas
                break
        report["rs_ms_sum8"] = t8
    if rank == 0:
        out = {k: v for k, v in c.__dict__.items()}
        json.dump(out, open(a.out, "w"), indent=1)
        print(json.dumps({"calib": out, "measured": report}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
