# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inter-node NIC leg in isolation: every rank puts its shard to the same-index GPU on every other
node (rail-aligned, no intra-node fan-out) with NCCL host RMA, then waits for the incoming shards.

Patterns:
  all     -- fire all N-1 puts at once (what NCCL's hierarchical CE all-gather does)
  rotate  -- round s: put to node+s, wait for node-s; one flow in / one flow out per NIC at a time
Chunks are striped over RMA contexts (ctx = chunk % num_ctx). Reports per-rank NIC bandwidth and a
byte-exact check of every received shard.

NNODES=8 scripts/run_multinode.sh rma bench/rma_bench.py --shard-mib 256 --chunk-mib 64 --num-ctx 4 --pattern all rotate
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
from tide import _C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, nargs="+", default=[256.0])
    ap.add_argument("--chunk-mib", type=float, nargs="+", default=[64.0])
    ap.add_argument("--num-ctx", type=int, default=1)
    ap.add_argument("--pattern", type=str, nargs="+", default=["all", "rotate"],
                    help="all | rotate | rotate_lock (receiver credits gate the next round on signal index 1)")
    ap.add_argument("--width", type=int, default=1, help="rotate*: destinations per round (concurrent flows per NIC)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    stream = torch.cuda.Stream(device=dev)
    torch.cuda.set_stream(stream)

    comm = tide.TideComm(device=dev, num_rma_ctx=args.num_ctx)
    C = comm._c
    node, N, L = C.node, C.n_nodes, C.local_ranks
    local = rank % L
    if N < 2:
        if rank == 0:
            print(json.dumps({"error": "single node; nothing to measure"}))
        return
    if rank == 0:
        print(json.dumps({"props": comm.props, "nodes": N, "local_ranks": L, "num_ctx": args.num_ctx}), flush=True)

    max_S = int(max(args.shard_mib) * 2**20) // 4096 * 4096
    arena = comm.alloc_arena(max_S * N)  # slot m <- shard of my rail peer on node m
    win, _ = comm._window_of(arena)
    rail_peer = lambda m: m * L + local  # noqa: E731

    for mib in args.shard_mib:
        S = int(mib * 2**20) // 4096 * 4096
        # my shard lives in my own slot; pattern = rank id so receivers can verify
        mine = arena.narrow(0, node * max_S, S)
        mine.fill_(rank % 251)
        for m in range(N):
            if m != node:
                arena.narrow(0, m * max_S, S).zero_()
        torch.cuda.synchronize()
        dist.barrier()
        for cmib in args.chunk_mib:
            chunk = int(cmib * 2**20) // 4096 * 4096
            nchunks = (S + chunk - 1) // chunk
            per_ctx = [sum(1 for j in range(nchunks) if j % args.num_ctx == c) for c in range(args.num_ctx)]

            def puts_to(m: int, s=stream):
                peer = rail_peer(m)
                _C.group_start()
                for j in range(nchunks):
                    off = node * max_S + j * chunk
                    C.put_signal(win, off, min(chunk, S - j * chunk), peer, off, 0, j % args.num_ctx, s.cuda_stream)
                _C.group_end()

            def wait_from(ms, s=stream):
                descs = [(per_ctx[c], rail_peer(m), 0, c) for m in ms for c in range(args.num_ctx) if per_ctx[c] > 0]
                C.wait_signal(descs, s.cuda_stream)

            for pattern in args.pattern:
                W = max(1, args.width)
                rounds = [list(range(s_, min(s_ + W, N))) for s_ in range(1, N, W)]  # shifts per round

                def one_iter():
                    if pattern == "all":
                        for m in range(N):
                            if m != node:
                                puts_to(m)
                        wait_from([m for m in range(N) if m != node])
                        return
                    for ri, shifts in enumerate(rounds):
                        if pattern == "rotate_lock" and ri > 0:
                            # credits from this round's destinations: they finished receiving the previous round
                            C.wait_signal([(1, rail_peer((node + sh) % N), 1, 0) for sh in shifts], stream.cuda_stream)
                        for sh in shifts:
                            puts_to((node + sh) % N)
                        wait_from([(node - sh) % N for sh in shifts])
                        if pattern == "rotate_lock" and ri + 1 < len(rounds):
                            # tell next round's sources (the nodes that will put to me) that I am drained
                            for sh in rounds[ri + 1]:
                                C.signal(rail_peer((node - sh) % N), 1, 0, stream.cuda_stream)

                for _ in range(args.warmup):
                    one_iter()
                torch.cuda.synchronize()
                dist.barrier()
                evs = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters + 1)]
                host = []
                evs[0].record(stream)
                for i in range(args.iters):
                    t0 = time.perf_counter()
                    one_iter()
                    host.append((time.perf_counter() - t0) * 1e6)
                    evs[i + 1].record(stream)
                torch.cuda.synchronize()
                per = [evs[i].elapsed_time(evs[i + 1]) for i in range(args.iters)]
                t_ms = evs[0].elapsed_time(evs[-1]) / args.iters
                # verify every received shard
                bad = 0
                for m in range(N):
                    if m != node:
                        got = arena.narrow(0, m * max_S, S)
                        bad += int((got != (rail_peer(m) % 251)).sum().item())
                tot = torch.tensor([bad], device=dev, dtype=torch.float64)
                dist.all_reduce(tot)  # sum of mismatches
                t_all = torch.zeros(world, device=dev, dtype=torch.float64)
                t_all[rank] = t_ms
                dist.all_reduce(t_all)  # per-rank time: exposes a slow rail or GPU
                slowest = torch.topk(t_all, min(3, world))
                nic_bytes = (N - 1) * S
                row = {"pattern": pattern, "width": W, "nodes": N, "shard_mib": mib, "chunk_mib": cmib, "num_ctx": args.num_ctx,
                       "chunks": nchunks, "t_ms_rank0": round(t_ms, 3), "t_ms_max": round(float(t_all.max()), 3),
                       "t_ms_med_rank0": round(statistics.median(per), 3),
                       "nic_GBs_rank0": round(nic_bytes / (t_ms * 1e-3) / 1e9, 1),
                       "nic_GBs_at_max_t": round(nic_bytes / (float(t_all.max()) * 1e-3) / 1e9, 1),
                       "slowest_ranks": [(int(r), round(float(t), 2)) for t, r in zip(slowest.values, slowest.indices)],
                       "host_us_med": round(statistics.median(host), 1), "mismatch_all": int(tot[0].item())}
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    if args.out:
                        with open(args.out, "a") as f:
                            f.write(json.dumps(row) + "\n")
    dist.barrier()
    del arena, win
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
