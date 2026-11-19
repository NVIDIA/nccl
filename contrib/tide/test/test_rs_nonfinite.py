# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-finite gradients propagate like IEEE sums, as they do through stock FSDP2 (loss scalers and
anomaly checks rely on it): NaN anywhere -> NaN; +Inf -> +Inf; +Inf meeting -Inf -> NaN; the rest
matches an fp64 reduce-scatter. Checked on the plain output and on the scale/accumulate landing,
through the node-local sum and, multi-node, the switch sum of the multicast landing.

torchrun --standalone --nproc_per_node 8 test/test_rs_nonfinite.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.distributed as dist

import tide

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_rs_bitwise import make_shapes, stock_reduce_scatter  # noqa: E402


def classes(x: torch.Tensor) -> torch.Tensor:
    """0 finite, 1 NaN, 2 +Inf, 3 -Inf."""
    c = torch.zeros_like(x, dtype=torch.int8)
    c[torch.isnan(x)] = 1
    c[torch.isposinf(x)] = 2
    c[torch.isneginf(x)] = 3
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, default=8.0, help="fp32 output bytes per rank")
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--hits", type=int, default=64, help="positions per kind of injection")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    shapes = make_shapes(random.Random(args.seed), world, int(args.shard_mib * 2**20) // 4, args.segments)
    comm = tide.TideComm(device=dev)

    bad = {"class": 0, "value": 0, "acc_class": 0, "acc_value": 0}
    counts = {}
    for dtype in (torch.bfloat16, torch.float32):
        table = [(d0 * rest, d0) for d0, rest in shapes]
        arena = comm.alloc_arena(comm.rs_stage_bytes((table, dtype.itemsize)))
        h = comm.register_rs(arena, 0, (table, dtype.itemsize))
        g = torch.Generator(device=dev).manual_seed(args.seed + rank)
        grads = [torch.randn(d0, rest, generator=g, device=dev).to(dtype) for d0, rest in shapes]
        # the same positions on every rank (rank-independent generator), different values per rank
        pos = torch.Generator(device="cpu").manual_seed(args.seed * 3 + dtype.itemsize)
        flat = [t.view(-1) for t in grads]

        def pick(n):
            out = []
            for _ in range(n):
                p = int(torch.randint(len(flat), (1,), generator=pos))
                out.append((p, int(torch.randint(flat[p].numel(), (1,), generator=pos))))
            return out

        nan_at, inf_at, clash_at = pick(args.hits), pick(args.hits), pick(args.hits)
        if rank == 0:
            for p, i in nan_at:
                flat[p][i] = float("nan")
        if rank == world - 1:
            for p, i in inf_at + clash_at:
                flat[p][i] = float("inf")
        if rank == world // 2 and world > 1:
            for p, i in clash_at:
                flat[p][i] = float("-inf")
        ref = stock_reduce_scatter(grads, world, torch.float64, dist.group.WORLD)
        out = h.reduce_scatter(grads)[0]
        acc = torch.full((h.out_numel,), 1.0, dtype=torch.float32, device=dev)
        h.reduce_scatter(grads, out=acc, scale=0.5, accumulate=True)
        ref_acc = 1.0 + 0.5 * ref
        torch.cuda.synchronize()
        for o, r, kc, kv in ((out, ref, "class", "value"), (acc, ref_acc, "acc_class", "acc_value")):
            co, cr = classes(o), classes(r)
            bad[kc] += int((co != cr).sum().item())
            fin = cr == 0
            tol = 1e-5 * r[fin].abs().max().clamp_min(1.0) if fin.any() else 0.0
            bad[kv] += int(((o.double()[fin] - r[fin]).abs() > tol).sum().item())
        cr = classes(ref)
        counts[str(dtype)] = [int((cr == k).sum().item()) for k in (1, 2, 3)]
        del h
        comm.free_arena(arena)
    tot = torch.tensor(list(bad.values()), device=dev)
    dist.all_reduce(tot)
    seen = torch.tensor([sum(v) for v in counts.values()], device=dev)
    dist.all_reduce(seen)
    # the injections must reach the outputs at all (NaN and clash hits somewhere in the world)
    ok = int(tot.sum().item()) == 0 and int(seen.min().item()) > 0
    if rank == 0:
        print(json.dumps({"world": world, "nodes": comm.n_nodes, **{k: int(v) for k, v in zip(bad, tot.tolist())},
                          "nonfinite_ref_rank0": counts}))
        print("PASS" if ok else "FAIL", flush=True)
    dist.barrier()
    comm.close()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
