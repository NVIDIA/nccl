# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state all-gather bandwidth: TIDE vs stock NCCL (rank-major) vs stock + param-major permutation.

torchrun --standalone --nproc_per_node 8 bench/ag_bench.py --shard-mib 64 155 256 --segments 1 16 64
busbw = (nranks-1) * shard_bytes / t_call   (bytes each rank must receive)
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


def timed_loop(fn, iters: int, warmup: int, stream, label: str) -> tuple[float, list[float], list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(iters + 1)]
    host = []
    torch.cuda.nvtx.range_push(label)
    evs[0].record(stream)
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        host.append((time.perf_counter() - t0) * 1e6)
        evs[i + 1].record(stream)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    per = [evs[i].elapsed_time(evs[i + 1]) for i in range(iters)]
    return evs[0].elapsed_time(evs[-1]) / iters, per, host


def mismatches(a: torch.Tensor, b: torch.Tensor, step: int = 2**28) -> int:
    # chunked: a bool .sum() widens to int64, 8x the compared bytes
    a, b = a.reshape(-1), b.reshape(-1)
    return sum(int((a[i : i + step] != b[i : i + step]).sum().item()) for i in range(0, a.numel(), step))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, nargs="+", default=[64.0, 155.0])
    ap.add_argument("--segments", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--small-frac", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--arms", type=str, nargs="+", default=["tide", "stock", "stock_perm"],
                    help="tide | stock (ring kernel) | stock_perm (ring + copy-out) | "
                         "stock_ce (symmetric-window buffers: FSDP2's set_symm_mem_for_comm; "
                         "needs NCCL_CTA_POLICY=2 in the environment for the copy-engine path) | "
                         "stock_ce_perm (stock_ce + copy-out: what FSDP2 pays with symm_mem)")
    ap.add_argument("--ep", type=int, default=1, help="gather over the expert-style subgroup r %% ep (tide / stock / stock_perm)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if args.ep > 1 and set(args.arms) - {"tide", "stock", "stock_perm"}:
        raise SystemExit("--ep supports the arms tide, stock, stock_perm")
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    group = sub_group(args.ep)
    world = dist.get_world_size(group)
    stream = torch.cuda.Stream(device=dev)
    torch.cuda.set_stream(stream)
    num_ctx = int(os.environ.get("TIDE_RMA_CTX", "0"))  # 0: the communicator's default
    comm = tide.TideComm(group, device=dev, num_rma_ctx=num_ctx) if "tide" in args.arms else None
    multinode = comm is not None and comm.n_nodes > 1
    ce_pool = None
    if "stock_ce" in args.arms or "stock_ce_perm" in args.arms:
        # Buffers from a MemPool registered as NCCL symmetric windows on torch's own communicator.
        backend = dist.group.WORLD._get_backend(dev)
        ce_pool = torch.cuda.MemPool(backend.mem_allocator)
        max_S = int(max(args.shard_mib) * 2**20) // 16 * 16
        with torch.cuda.use_mem_pool(ce_pool):
            ce_in = torch.empty(max_S, dtype=torch.uint8, device=dev)
            ce_out = torch.empty(max_S * world, dtype=torch.uint8, device=dev)
        torch.cuda.synchronize(); dist.barrier()
        backend.register_mem_pool(ce_pool, symm=True)
        torch.cuda.synchronize(); dist.barrier()

    max_shard = int(max(args.shard_mib) * 2**20) // 16 * 16
    # staged: the packed shard is written to a stage slot instead of in place (required on multi-node;
    # TIDE_MC_STAGE=1 forces it on one node)
    mc_stage = (multinode and not comm.rail) or os.environ.get("TIDE_MC_STAGE", "0") == "1"
    stage_bytes = comm.stage_bytes(max_shard) if (comm and mc_stage) else 0
    lane_bytes = (max_shard * world + stage_bytes + 4095) // 4096 * 4096
    arena = comm.alloc_arena(lane_bytes * args.lanes) if comm else None
    tag = {"world": world, "ep": args.ep, "nodes": comm.n_nodes if comm else None, "nccl": tide._C.nccl_runtime_version(),
           "nic_groups": comm.nic_groups if comm else None,
           "nic_pace": os.environ.get("TIDE_NIC_PACE", "auto"), "pace_lead_mib": os.environ.get("TIDE_NIC_PACE_LEAD_MIB", "8"),
           "mc_stage": mc_stage, "rma_ctx": comm.rma_ctx if multinode else None,
           "multimem": bool(comm._window_of(arena)[0].has_multimem) if comm else None}

    for mib in args.shard_mib:
        S = int(mib * 2**20) // 16 * 16
        # two shards alternate over the lanes, so a lane holding the other lane's (stale) data is caught
        shards = [torch.randint(0, 256, (S,), dtype=torch.uint8, device=dev) for _ in range(2)]
        shard = shards[0]
        moved = (world - 1) * S
        for nseg in args.segments:
            sizes, offsets = tide.byte_table(make_segments(S, nseg, args.small_frac, seed=nseg * 31 + 7))
            row = {**tag, "shard_mib": mib, "segments": nseg, "min_seg": min(sizes), "max_seg": max(sizes)}
            if "tide" in args.arms:
                handles = []
                for lane in range(args.lanes):
                    # stage at a fixed offset per lane (past the largest landing), as the FSDP adapter does
                    roff = lane * lane_bytes
                    h = comm.register_ag(arena, roff, sizes, offsets, stage_off=(roff + max_shard * world) if mc_stage else -1)
                    src = shards[lane % 2]
                    torch._foreach_copy_(h.own_pieces(arena), [src.narrow(0, offsets[p], sizes[p]) for p in range(nseg)])
                    handles.append(h)
                it = [0]

                def tide_call():
                    handles[it[0] % args.lanes].all_gather(stream)
                    it[0] += 1

                t, per, host = timed_loop(tide_call, args.iters, args.warmup, stream, "tide_all_gather")
                bad = 0
                for lane, h in enumerate(handles):
                    exp = param_major_permute(shards[lane % 2], sizes, offsets, group)
                    bad += mismatches(h.region(arena), exp)
                row.update({"tide_ms": t, "tide_ms_med": statistics.median(per), "tide_ms_min": min(per),
                            "tide_ms_p90": sorted(per)[int(0.9 * (len(per) - 1))], "tide_busbw_GBs": moved / (t * 1e-3) / 1e9,
                            "tide_host_us_med": statistics.median(host), "tide_host_us_max": max(host),
                            "tide_descriptors": handles[0].num_descriptors, "tide_mismatch": bad})
                trace = comm.trace_ag() if os.environ.get("TIDE_TRACE_AG") == "1" else []
                if trace:  # GPU-side phase medians, same names as the FSDP adapter's trace log
                    n = len(trace[0][1])
                    trace = [x for x in trace if len(x[1]) == n]
                    names = ["credits", "fence"] + [f"r{j}" for j in range(1, n - 3)] + ["scatter", "done"]
                    row["tide_trace_med_ms"] = {k: statistics.median(x[1][p] for x in trace) for p, k in enumerate(names)}
            if "stock" in args.arms:
                out = torch.empty(S * world, dtype=torch.uint8, device=dev)
                t, per, host = timed_loop(lambda: dist.all_gather_into_tensor(out, shard, group=group), args.iters, args.warmup,
                                          stream, "stock_all_gather")
                row.update({"stock_ms": t, "stock_ms_med": statistics.median(per), "stock_ms_min": min(per),
                            "stock_ms_p90": sorted(per)[int(0.9 * (len(per) - 1))],
                            "stock_busbw_GBs": moved / (t * 1e-3) / 1e9,
                            "stock_host_us_med": statistics.median(host), "stock_host_us_max": max(host)})
                del out
            if "stock_ce" in args.arms:
                cin, cout = ce_in.narrow(0, 0, S), ce_out.narrow(0, 0, S * world)
                cin.copy_(shard)
                t, per, host = timed_loop(lambda: dist.all_gather_into_tensor(cout, cin), args.iters, args.warmup,
                                          stream, "stock_ce_all_gather")
                ref = torch.empty(S * world, dtype=torch.uint8, device=dev)
                dist.all_gather_into_tensor(ref, shard)
                row.update({"stock_ce_ms": t, "stock_ce_ms_med": statistics.median(per), "stock_ce_ms_min": min(per),
                            "stock_ce_busbw_GBs": moved / (t * 1e-3) / 1e9,
                            "stock_ce_host_us_med": statistics.median(host), "stock_ce_mismatch": mismatches(cout, ref),
                            "cta_policy": os.environ.get("NCCL_CTA_POLICY", "")})
                del ref
            if "stock_ce_perm" in args.arms:
                cin, cout = ce_in.narrow(0, 0, S), ce_out.narrow(0, 0, S * world)
                cin.copy_(shard)
                pm = torch.empty(S * world, dtype=torch.uint8, device=dev)
                pm_views = [pm[o * world : (o + s) * world].view(world, s) for s, o in zip(sizes, offsets)]

                def stock_ce_perm():
                    dist.all_gather_into_tensor(cout, cin)
                    # FSDP2's copy-out: one fused split_with_sizes_copy over the rank-major output
                    torch.split_with_sizes_copy(cout.view(world, S), list(sizes), dim=1, out=pm_views)

                t, per, host = timed_loop(stock_ce_perm, args.iters, args.warmup, stream, "stock_ce_perm_all_gather")
                row.update({"stock_ce_perm_ms": t, "stock_ce_perm_ms_med": statistics.median(per),
                            "stock_ce_perm_ms_min": min(per), "stock_ce_perm_host_us_med": statistics.median(host),
                            "cta_policy": os.environ.get("NCCL_CTA_POLICY", "")})
                del pm
            if "stock_perm" in args.arms:
                out = torch.empty(S * world, dtype=torch.uint8, device=dev)
                pm = torch.empty_like(out)
                pm_views = [pm[o * world : (o + s) * world].view(world, s) for s, o in zip(sizes, offsets)]

                dummy = os.environ.get("AG_BENCH_PERM_DUMMY", "0") == "1"  # same bytes, no permutation (control)

                def stock_perm():
                    dist.all_gather_into_tensor(out, shard, group=group)
                    if dummy:
                        pm.copy_(out)
                        return
                    # FSDP2's copy-out: one fused split_with_sizes_copy over the rank-major output
                    torch.split_with_sizes_copy(out.view(world, S), list(sizes), dim=1, out=pm_views)

                t, per, host = timed_loop(stock_perm, args.iters, args.warmup, stream, "stock_perm_all_gather")
                row.update({"stock_perm_ms": t, "stock_perm_ms_med": statistics.median(per), "stock_perm_ms_min": min(per),
                            "stock_perm_host_us_med": statistics.median(host)})
                del out, pm
            if rank == 0:
                print(json.dumps(row), flush=True)
                if args.out:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
        del shard
    dist.barrier()
    if comm is not None:
        del arena
        handles = h = None  # noqa: F841 - drop the last views so close() releases the window now
        comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
