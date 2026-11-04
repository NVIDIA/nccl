# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The reduce-scatter's sizing does not change its result: sum / landing CTAs, NIC round width k, the
staging chunk and the number of passes over the sub-slices only schedule the same additions (K_sum
sums each element over the sources in a fixed order, the landing is one multimem load per 16 B, and
an element's waves land in the same order in whichever pass carries it), so real-valued inputs
reduced through communicators tuned differently must give bitwise identical outputs, plain and
scale/accumulate. The landing wave size is the one knob that regroups the cross-node sum and is not
varied here.

torchrun --standalone --nproc_per_node 8 test/test_rs_invariance.py
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
from test_rs_bitwise import make_shapes  # noqa: E402

MIB = 2**20
# (sum CTAs, landing CTAs, k, chunk bytes[, passes]); None = the communicator's defaults
TUNINGS = [None, (1, 1, 1, 1 * MIB), (8, 8, 2, 4 * MIB), (3, 2, 8, 256 * 1024), (8, 4, 1, 8 * MIB, 4),
           (3, 2, 2, 256 * 1024, 3)]
_ENV = ("TIDE_RS_CTAS", "TIDE_RS_MC_CTAS", "TIDE_RS_KWAY", "TIDE_RS_CHUNK_MIB", "TIDE_RS_PARTS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, default=16.0, help="fp32 output bytes per rank")
    ap.add_argument("--segments", type=int, default=12)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4321)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    overridden = [e for e in _ENV if e in os.environ]
    if overridden:
        if rank == 0:
            print(f"FAIL: {overridden} set in the environment would pin every communicator to one tuning", flush=True)
        dist.destroy_process_group()
        return 1

    shapes = make_shapes(random.Random(args.seed), world, int(args.shard_mib * MIB) // 4, args.segments)
    table = [(d0 * rest, d0) for d0, rest in shapes]
    comms, handles = [], []
    for t in TUNINGS:
        comm = tide.TideComm(device=dev)
        if t is not None:
            comm._c.set_rs_tuning(*t)
        arena = comm.alloc_arena(comm.rs_area_bytes([(table, 2), (table, 4)]))
        handles.append((comm.register_rs(arena, 0, (table, 2)), comm.register_rs(arena, 0, (table, 4))))
        comms.append((comm, arena, comm.rs_tuning))

    bad = {"plain": 0, "accum": 0}
    for it in range(args.iters):
        for dt_i, dtype in enumerate((torch.bfloat16, torch.float32)):
            g = torch.Generator(device=dev).manual_seed(args.seed * 7 + it * 101 + dt_i * 13 + rank)
            grads = [torch.randn(d0, rest, generator=g, device=dev).to(dtype) for d0, rest in shapes]
            outs, accs = [], []
            for (comm, _, _), hs in zip(comms, handles):
                h = hs[dt_i]
                outs.append(h.reduce_scatter(grads)[0])
                acc = torch.full((h.out_numel,), 0.25, dtype=torch.float32, device=dev)
                h.reduce_scatter(grads, out=acc, scale=0.5, accumulate=True)
                accs.append(acc)
            torch.cuda.synchronize()
            for o, a in zip(outs[1:], accs[1:]):
                bad["plain"] += int((o != outs[0]).sum().item())
                bad["accum"] += int((a != accs[0]).sum().item())
    tot = torch.tensor([bad["plain"], bad["accum"]], device=dev)
    dist.all_reduce(tot)
    ok = int(tot.sum().item()) == 0
    if rank == 0:
        print(json.dumps({"world": world, "nodes": comms[0][0].n_nodes, "tunings": [list(c[2]) for c in comms],
                          "chunks": [hs[0].plan.chunks for hs in handles], "parts": [hs[0].plan.parts for hs in handles],
                          "bad_plain": int(tot[0]), "bad_accum": int(tot[1])}))
        print("PASS" if ok else "FAIL", flush=True)
    dist.barrier()
    del handles
    for comm, arena, _ in comms:
        comm.free_arena(arena)
        comm.close()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
