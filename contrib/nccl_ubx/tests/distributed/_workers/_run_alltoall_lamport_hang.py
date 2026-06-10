"""Hang-repro worker for alltoall_lamport at large sizes (>=4MB).

For each (mode, size) combination:
  - mode = eager | graph
  - size in --sizes (bytes total tensor)
Runs N consecutive lamport alltoall calls, optionally validates against NCCL,
and prints a heartbeat every call so a hang is visible in the slurm log.

The kernel is launched into a child stream so a per-iteration sync below
catches a hang at the right call count rather than waiting for graph-replay
to finish.

Usage:
    torchrun --nproc_per_node=2 _run_alltoall_lamport_hang.py \
        --sizes 256K,1M,4M,8M,16M --mode eager --calls 8
"""

import argparse
import gc
import os
import signal
import sys
import time

import torch
import torch.distributed as dist


def parse_size(s):
    s = s.strip().upper()
    if s.endswith("K"):
        return int(s[:-1]) * 1024
    if s.endswith("M"):
        return int(s[:-1]) * 1024 * 1024
    if s.endswith("G"):
        return int(s[:-1]) * 1024 * 1024 * 1024
    return int(s)


def get_rank_info():
    if "RANK" in os.environ:
        return (int(os.environ["RANK"]),
                int(os.environ["WORLD_SIZE"]),
                int(os.environ.get("LOCAL_RANK", os.environ["RANK"])))
    if "SLURM_PROCID" in os.environ:
        return (int(os.environ["SLURM_PROCID"]),
                int(os.environ["SLURM_NTASKS"]),
                int(os.environ.get("SLURM_LOCALID", os.environ["SLURM_PROCID"])))
    raise RuntimeError("Cannot determine rank")


def log(rank, msg):
    print(f"[r{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_eager(allocator, dtype, count, calls, do_check, rank, world_size, device):
    from ubx import SymmAllocator  # noqa: F401
    symm_in = allocator.create_tensor(torch.Size([count]), dtype)
    symm_in.fill_(float(rank) + 1.0)
    log(rank, f"eager: input symm allocated, count={count}, dtype={dtype}")

    if do_check:
        # NCCL reference: a uniform fill means rank j's chunk on rank i is (j+1).
        ref = torch.empty_like(symm_in)
        chunks_in = list(symm_in.split(count // world_size))
        chunks_out = list(ref.split(count // world_size))
        dist.all_to_all(chunks_out, chunks_in)
        torch.cuda.synchronize()
        log(rank, "eager: NCCL reference computed")

    cur = symm_in
    for c in range(calls):
        t0 = time.perf_counter()
        log(rank, f"eager: call {c} START")
        result = allocator.alltoall_lamport(cur)
        # Fence at the end of the kernel — a hang inside the kernel manifests
        # as cudaSync hanging here, with the prior log telling us which call.
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(rank, f"eager: call {c} END, elapsed={elapsed_ms:.1f}ms")

        if do_check and c == 0:
            ok = torch.allclose(result.float(), ref.float(), atol=1e-3, rtol=2e-2)
            log(rank, f"eager: NCCL match call0 = {ok}")
            if not ok:
                max_err = (result.float() - ref.float()).abs().max().item()
                log(rank, f"eager: MAX ERR = {max_err}")
                return False

        # Re-stage input for next call so we don't depend on result's lifecycle.
        cur = allocator.create_tensor(result.shape, result.dtype)
        cur.copy_(result)
    return True


def run_graph(allocator, dtype, count, calls, captures_per_graph, rank, device):
    """Capture `captures_per_graph` lamport calls into a single graph, replay it `calls` times."""
    symm_in = allocator.create_tensor(torch.Size([count]), dtype)
    symm_in.fill_(float(rank) + 1.0)

    # Warmup outside graph (puts triple-buf into steady state).
    for _ in range(3):
        allocator.alltoall_lamport(symm_in)
    torch.cuda.synchronize()
    log(rank, f"graph: warmup done")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    gc.disable()
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cur = symm_in
            for _ in range(captures_per_graph):
                cur = allocator.alltoall_lamport(cur)
    gc.enable()
    gc.collect()
    torch.cuda.current_stream().wait_stream(stream)
    log(rank, f"graph: captured {captures_per_graph} ops, replays={calls}")

    # Sync per replay so a hang manifests at the exact replay number.
    t0 = time.perf_counter()
    for c in range(calls):
        log(rank, f"graph: replay {c} START")
        graph.replay()
        torch.cuda.synchronize()
        log(rank, f"graph: replay {c} END")
    elapsed = time.perf_counter() - t0
    log(rank, f"graph: ALL {calls} replays DONE, total={elapsed:.2f}s")
    del graph
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="256K,1M,4M,8M,16M",
                        help="Comma-separated total tensor sizes (bytes; K/M/G ok)")
    parser.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
    parser.add_argument("--calls", type=int, default=8,
                        help="Eager: # of consecutive calls. Graph: # of replays.")
    parser.add_argument("--captures-per-graph", type=int, default=4,
                        help="# of lamport ops captured into one graph (graph mode).")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--check", action="store_true",
                        help="Validate first eager call against NCCL.")
    args = parser.parse_args()

    rank, world_size, local_rank = get_rank_info()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    device = torch.device(f"cuda:{local_rank}")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    sizes_b = [parse_size(s) for s in args.sizes.split(",")]
    log(rank, f"start: world={world_size}, sizes={[s//1024 for s in sizes_b]}KB, "
              f"mode={args.mode}, calls={args.calls}")

    from ubx import SymmAllocator
    # Pool size: 3 triple-bufs + headroom + per-size overhead. Make it big.
    max_size = max(sizes_b)
    # graph + eager pools both — graph share 0.5
    os.environ["UBX_GRAPH_POOL_SHARE"] = "0.5"
    pool_size = max(max_size * 32, 256 * 1024 * 1024)
    log(rank, f"allocating pool: {pool_size//(1024*1024)}MB")

    failure = 0
    for sz in sizes_b:
        elem_size = torch.tensor(0, dtype=dtype).element_size()
        count = sz // elem_size
        log(rank, f"=== size={sz//1024}KB count={count} ===")

        if args.mode in ("eager", "both"):
            allocator = SymmAllocator(pool_size, device, dist.group.WORLD)
            try:
                ok = run_eager(allocator, dtype, count, args.calls, args.check,
                               rank, world_size, device)
                log(rank, f"eager size={sz//1024}KB: {'OK' if ok else 'FAIL'}")
                if not ok:
                    failure += 1
            except Exception as e:
                log(rank, f"eager size={sz//1024}KB: EXCEPTION {e}")
                failure += 1
            allocator.close()
            del allocator
            gc.collect()

        if args.mode in ("graph", "both"):
            allocator = SymmAllocator(pool_size, device, dist.group.WORLD)
            try:
                ok = run_graph(allocator, dtype, count, args.calls,
                               args.captures_per_graph, rank, device)
                log(rank, f"graph size={sz//1024}KB: {'OK' if ok else 'FAIL'}")
                if not ok:
                    failure += 1
            except Exception as e:
                log(rank, f"graph size={sz//1024}KB: EXCEPTION {e}")
                failure += 1
            allocator.close()
            del allocator
            gc.collect()

    log(rank, f"DONE failures={failure}")
    dist.destroy_process_group()
    sys.exit(failure)


if __name__ == "__main__":
    main()
