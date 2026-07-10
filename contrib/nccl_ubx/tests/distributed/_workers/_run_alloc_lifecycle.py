"""Worker for the SymmAllocator lifecycle / leak regression test.

Loops N times: create SymmAllocator → fill a tensor → drop. After each
iter, sample GPU free memory. If destroy() is releasing properly, the free
memory stays flat. If the bootstrapped ncclComm_t leaks (the bug this
test guards), free memory decreases monotonically.

Pass criterion: max(free_baseline - free_iter_i) below LEAK_THRESHOLD_MB.

Usage:
    python _run_alloc_lifecycle.py --iters 20 --pool-mib 64 --leak-mb 100
"""

import argparse
import gc
import os
import sys
import torch
import torch.distributed as dist


def get_rank_info():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"]); world = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"]); world = int(os.environ["SLURM_NTASKS"])
        local = int(os.environ.get("SLURM_LOCALID", rank))
    else:
        raise RuntimeError("Cannot determine rank")
    return rank, world, local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--pool-mib", type=int, default=64)
    ap.add_argument("--leak-mb", type=int, default=100,
                    help="Per-rank allowed memory delta after N iters (MB)")
    args = ap.parse_args()

    rank, world, local = get_rank_info()
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=world, rank=rank,
        )

    from ubx import SymmAllocator
    pg = dist.group.WORLD

    pool_bytes = args.pool_mib * 1024 * 1024
    count = pool_bytes // 24  # leave headroom for allocator metadata

    # Warm up the parent PG's NCCL channels by running the same two
    # collectives SymmAllocator.__init__ prewarms — allgather + alltoall.
    # The first call to each of these allocates ~hundreds of MiB of
    # NCCL channel buffers + loads device kernels into the driver; that
    # cost is process-persistent and reused across every SymmAllocator,
    # so we want it paid BEFORE sampling free_baseline. Otherwise iter 0
    # records that static infrastructure cost as a "leak" against
    # baseline-before-first-collective.
    _src = torch.zeros(262144, dtype=torch.bfloat16, device=device)
    _dst = torch.zeros(262144 * world, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(_dst, _src, group=pg)
    _a2a = torch.zeros(world * 16, dtype=torch.bfloat16, device=device)
    _a2a_recv = torch.empty_like(_a2a)
    dist.all_to_all_single(_a2a_recv, _a2a, group=pg)
    torch.cuda.synchronize()
    dist.barrier()
    del _src, _dst, _a2a, _a2a_recv
    gc.collect()

    free_baseline, _ = torch.cuda.mem_get_info(device)

    max_leak_seen = 0
    for i in range(args.iters):
        alloc = SymmAllocator(pool_bytes, device, pg)
        t = alloc.create_tensor([count], torch.bfloat16)
        t.fill_(1.0)
        torch.cuda.synchronize()
        del t, alloc
        gc.collect()
        torch.cuda.synchronize()

        free_now, _ = torch.cuda.mem_get_info(device)
        leak = free_baseline - free_now
        if leak > max_leak_seen:
            max_leak_seen = leak
        if rank == 0:
            print(f"[rank0] iter {i}: free={free_now/(1<<30):.2f} GiB leak={leak/(1<<20):.1f} MiB",
                  flush=True)

    leak_mib = max_leak_seen / (1 << 20)
    threshold_mib = args.leak_mb
    if rank == 0:
        if leak_mib < threshold_mib:
            print(f"PASS leak={leak_mib:.1f} MiB < threshold={threshold_mib} MiB across {args.iters} iters",
                  flush=True)
        else:
            print(f"FAIL leak={leak_mib:.1f} MiB >= threshold={threshold_mib} MiB across {args.iters} iters",
                  flush=True)
    sys.exit(0 if leak_mib < threshold_mib else 1)


if __name__ == "__main__":
    main()
