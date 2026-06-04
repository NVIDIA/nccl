"""E2E worker: compare UBX alltoall_lamport against NCCL with random inputs.

For each seed × call, generates random input, runs both NCCL and UBX,
compares outputs. Tests the full triple-buffer lifecycle: warmup (barrier)
and steady state (barrier-free).

Usage:
    torchrun --nproc_per_node=4 _run_alltoall_e2e.py --size 1024 --calls 10 --seeds 5
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist

os.environ.setdefault("UBX_GRAPH_POOL_SHARE", "0.1")


def get_rank_info():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))
    else:
        raise RuntimeError("Cannot determine rank")
    return rank, world_size, local_rank


def nccl_alltoall(tensor_input, world_size, device):
    """Reference alltoall via NCCL (torch.distributed)."""
    chunk_size = tensor_input.numel() // world_size
    send_chunks = list(tensor_input.split(chunk_size))
    recv_chunks = [torch.empty(chunk_size, dtype=tensor_input.dtype, device=device)
                   for _ in range(world_size)]
    dist.all_to_all(recv_chunks, send_chunks)
    return torch.cat(recv_chunks, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024, help="Total elements (divisible by world_size)")
    parser.add_argument("--calls", type=int, default=1, help="Consecutive Lamport calls per seed")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds to test")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    rank, world_size, local_rank = get_rank_info()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    device = torch.device(f"cuda:{local_rank}")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    atol = 0.001 if dtype == torch.bfloat16 else 0.0001
    rtol = 0.02 if dtype == torch.bfloat16 else 0.001

    from ubx import SymmAllocator
    # Lamport needs 3 extra triple buffers + input + output headroom
    pool_size = max(args.size * 2 * 16, 64 * 1024 * 1024)
    allocator = SymmAllocator(pool_size, device, dist.group.WORLD)

    total_tests = 0
    total_pass = 0

    for seed in range(args.seeds):
        # Create fresh input for this seed
        torch.manual_seed(seed * 1000 + rank)
        tensor_input = torch.randn(args.size, dtype=dtype, device=device)

        # NCCL reference
        ref = nccl_alltoall(tensor_input, world_size, device)

        # UBX Lamport — run `calls` consecutive calls, each feeding output to next
        symm_in = allocator.create_tensor(tensor_input.shape, dtype)
        symm_in.copy_(tensor_input)
        current_input = tensor_input.clone()  # track input for semantic check

        for call_i in range(args.calls):
            result = allocator.alltoall_lamport(symm_in)
            torch.cuda.synchronize()

            # Compare against NCCL on the FIRST call (same input)
            if call_i == 0:
                total_tests += 1
                try:
                    torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)
                    total_pass += 1
                    if rank == 0:
                        print(f"PASS seed={seed} call={call_i} size={args.size} "
                              f"nccl_match", flush=True)
                except AssertionError as e:
                    max_err = (result.float() - ref.float()).abs().max().item()
                    print(f"FAIL rank={rank} seed={seed} call={call_i} size={args.size} "
                          f"max_err={max_err:.6f}: {e}", flush=True)
                    sys.exit(1)

            # Cross-rank semantic validation: after alltoall, rank i's chunk j
            # should contain rank j's original chunk i.
            # Gather all ranks' inputs, then verify each chunk.
            chunk_size = args.size // world_size
            all_inputs = [torch.empty(args.size, dtype=dtype, device=device)
                          for _ in range(world_size)]
            dist.all_gather(all_inputs, current_input)
            total_tests += 1
            semantic_ok = True
            for src_rank in range(world_size):
                expected_chunk = all_inputs[src_rank][rank * chunk_size:(rank + 1) * chunk_size]
                actual_chunk = result[src_rank * chunk_size:(src_rank + 1) * chunk_size]
                if not torch.allclose(actual_chunk.float(), expected_chunk.float(),
                                      atol=atol, rtol=rtol):
                    max_err = (actual_chunk.float() - expected_chunk.float()).abs().max().item()
                    print(f"FAIL rank={rank} seed={seed} call={call_i} size={args.size} "
                          f"semantic: chunk from rank {src_rank} mismatch, "
                          f"max_err={max_err:.6f}", flush=True)
                    semantic_ok = False
            if semantic_ok:
                total_pass += 1
                if rank == 0:
                    print(f"PASS seed={seed} call={call_i} size={args.size} "
                          f"semantic_ok", flush=True)
            else:
                sys.exit(1)

            # Feed output back as input for next call.
            # Copy to a fresh SymmTensor — the result buffer will be used as
            # clear_ptr (poison target) in a future call, so its .w fields
            # will contain 0xFFFAFFFA (NaN in bf16). A fresh copy avoids this.
            symm_in = allocator.create_tensor(result.shape, result.dtype)
            symm_in.copy_(result)
            current_input = result.clone()

    if rank == 0:
        print(f"ALL PASS ({total_pass}/{total_tests} tests, {args.seeds} seeds, "
              f"{args.calls} calls/seed, {world_size} GPUs)", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
