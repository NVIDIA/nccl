"""Worker script for UB-X alltoallv correctness tests.

Generates power-law distributed split sizes, runs alltoallv, and verifies
against torch.distributed.all_to_all as reference.

Usage (4 GPUs):
    torchrun --nproc_per_node=4 _run_alltoallv.py --total-elems 4096
    torchrun --nproc_per_node=4 _run_alltoallv.py --total-elems 4096 --alpha 1.0
    torchrun --nproc_per_node=4 _run_alltoallv.py --total-elems 4096 --alpha 0.0
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist


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


def init_distributed():
    rank, world_size, local_rank = get_rank_info()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=world_size, rank=rank)
    return rank, world_size, local_rank


def powerlaw_split(total_elems, nranks, alpha, seed, device):
    """Generate power-law distributed split sizes.

    alpha=0: uniform
    alpha=0.5: moderately skewed (default)
    alpha=1.0: Zipf-like
    """
    torch.manual_seed(seed)
    if alpha == 0:
        weights = torch.ones(nranks, device=device)
    else:
        weights = torch.arange(1, nranks + 1, dtype=torch.float32, device=device) ** (-alpha)
        # Shuffle so skew isn't always rank-0-heavy
        perm = torch.randperm(nranks, device=device)
        weights = weights[perm]

    # Distribute total_elems proportionally, ensure sum matches
    raw = (weights / weights.sum() * total_elems).floor().to(torch.int32)
    remainder = total_elems - int(raw.sum().item())
    for i in range(remainder):
        raw[i % nranks] += 1

    # Ensure 16-byte (uint4) alignment: each split must be divisible by
    # line_size / elem_size. For bf16: 16/2 = 8 elements per uint4 line.
    align = 8  # bf16 elements per uint4
    aligned = ((raw + align - 1) // align) * align
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-elems", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Power-law skew: 0=uniform, 0.5=moderate, 1.0=Zipf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    os.environ.setdefault("UBX_GRAPH_POOL_SHARE", "0.1")

    # Each rank generates the SAME split matrix [nranks, nranks]:
    # split_matrix[src, dst] = elements src sends to dst
    split_matrix = torch.zeros(world_size, world_size, dtype=torch.int32, device=device)
    for src in range(world_size):
        split_matrix[src] = powerlaw_split(
            args.total_elems, world_size, args.alpha, args.seed + src, device)

    input_split_sizes = split_matrix[rank]  # what this rank sends to each dst
    output_split_sizes = split_matrix[:, rank]  # what this rank receives from each src

    total_send = int(input_split_sizes.sum().item())
    total_recv = int(output_split_sizes.sum().item())

    print(f"[rank{rank}] send_splits={input_split_sizes.cpu().tolist()} "
          f"recv_splits={output_split_sizes.cpu().tolist()} "
          f"total_send={total_send} total_recv={total_recv}", flush=True)

    # Create input with rank-unique pattern
    torch.manual_seed(args.seed + rank * 1000)
    input_data = torch.randn(total_send, dtype=torch.bfloat16, device=device)

    # --- Reference: torch.distributed.all_to_all ---
    send_chunks = list(input_data.split(input_split_sizes.tolist()))
    recv_chunks = [torch.empty(s, dtype=torch.bfloat16, device=device)
                   for s in output_split_sizes.tolist()]
    dist.all_to_all(recv_chunks, send_chunks)
    ref_output = torch.cat(recv_chunks)

    # --- UBX: alltoallv ---
    from ubx import SymmAllocator
    pool_size = max((total_send + total_recv) * 4, 16 * 1024 * 1024)
    allocator = SymmAllocator(pool_size, device, dist.group.WORLD)

    ubx_input = allocator.create_tensor(torch.Size([total_send]), torch.bfloat16)
    ubx_input.copy_(input_data)

    ubx_output = allocator.alltoallv(ubx_input, output_split_sizes, input_split_sizes)
    torch.cuda.synchronize()

    # Compare
    ubx_result = ubx_output[:total_recv].clone()
    abs_diff = (ubx_result.float() - ref_output.float()).abs()
    max_err = abs_diff.max().item()
    num_diff = (ubx_result != ref_output).sum().item()

    if max_err > 0.001 or num_diff > 0:
        print(f"FAIL rank={rank}: max_err={max_err:.6f} num_diff={num_diff}/{total_recv}",
              flush=True)
        sys.exit(1)
    else:
        print(f"PASS rank={rank}: total_recv={total_recv} max_err={max_err:.6f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
