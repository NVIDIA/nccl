# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TIDE all-gather on one stream while torch/NCCL reduce-scatter kernels run on another
(the FSDP backward shape). Checks for deadlock (bounded by the launcher timeout) and that both
results stay bitwise correct.

torchrun --standalone --nproc_per_node 8 test/test_concurrent_rs.py --iters 200
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

import tide
from tide.testing import make_segments, param_major_permute


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, default=64.0)
    ap.add_argument("--rs-mib", type=float, default=256.0)
    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)

    comm = tide.TideComm(device=dev)
    S = int(args.shard_mib * 2**20) // 16 * 16
    sizes, offsets = tide.byte_table(make_segments(S, args.segments, 0.5, 5))
    staged = comm.n_nodes > 1
    lane_bytes = (S * world + (comm.stage_bytes(S) if staged else 0) + 4095) // 4096 * 4096
    arena = comm.alloc_arena(lane_bytes * 2)
    handles = [comm.register_ag(arena, lane * lane_bytes, sizes, offsets,
                                stage_off=(lane * lane_bytes + S * world) if staged else -1) for lane in range(2)]
    shard = torch.randint(0, 256, (S,), dtype=torch.uint8, device=dev)
    for h in handles:
        torch._foreach_copy_(h.own_pieces(arena), [shard.narrow(0, offsets[p], sizes[p]) for p in range(len(sizes))])

    rs_n = int(args.rs_mib * 2**20) // 4 // world * world
    rs_in = torch.full((rs_n,), float(rank + 1), device=dev)
    rs_out = torch.empty(rs_n // world, device=dev)
    rs_expected = float(sum(range(1, world + 1)))

    ag_stream, rs_stream = torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev)
    dist.barrier()
    t0 = time.perf_counter()
    for k in range(args.iters):
        with torch.cuda.stream(ag_stream):
            handles[k % 2].all_gather(ag_stream)
        with torch.cuda.stream(rs_stream):
            dist.reduce_scatter_tensor(rs_out, rs_in)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    dist.barrier()

    exp = param_major_permute(shard, sizes, offsets, dist.group.WORLD)
    bad_ag = int((handles[0].region(arena) != exp).sum().item()) + int((handles[1].region(arena) != exp).sum().item())
    bad_rs = int((rs_out != rs_expected).sum().item())
    bad = torch.tensor([bad_ag, bad_rs], device=dev)
    dist.all_reduce(bad)
    ok = int(bad.sum().item()) == 0
    if rank == 0:
        print(json.dumps({"iters": args.iters, "wall_s": round(wall, 3), "per_iter_ms": round(wall / args.iters * 1e3, 3),
                          "bad_ag_bytes": int(bad[0].item()), "bad_rs_elems": int(bad[1].item()), "world": world}))
        print("PASS" if ok else "FAIL", flush=True)
    dist.barrier()
    del handles, arena
    comm.close()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
