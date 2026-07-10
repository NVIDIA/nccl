"""UB-X AllToAll — minimal end-to-end example.

Shows direct `SymmAllocator` usage (instead of the convenience registry)
and the auto AllToAll that selects between Lamport (small) and UC (large)
based on total tensor size.

After the alltoall, rank `i`'s `j`-th chunk contains rank `j`'s `i`-th
chunk — the standard "transpose by ranks" semantic, validated here
against `torch.distributed.all_to_all` (NCCL).

Launch (2 or more GPUs on a single node):

    torchrun --nproc-per-node=2 examples/alltoall.py
"""

import os
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")

    from ubx import SymmAllocator

    # Total elements (must be divisible by world_size). 64K bf16 = 128 KB,
    # so the auto wrapper picks Lamport. Bump to ~256K elements (= 0.5 MB
    # in bf16) to see the UC path.
    total_count = 65536
    dtype = torch.bfloat16
    pool_bytes = max(64 * 1024 * 1024, total_count * 2 * 16)  # headroom for Lamport triple buffer
    allocator = SymmAllocator(pool_bytes, device, dist.group.WORLD)

    # Fill input with rank-distinct data so the alltoall transpose is observable.
    src = allocator.create_tensor(torch.Size([total_count]), dtype)
    torch.manual_seed(42 + rank)
    src.copy_(torch.randn(total_count, dtype=dtype, device=device))

    # Run the auto alltoall (Lamport ≤ 0.25 MB, otherwise UC).
    out = allocator.alltoall_auto(src)
    torch.cuda.synchronize()

    # NCCL reference: build a list of per-destination chunks, dispatch via
    # torch.distributed.all_to_all, concatenate the received chunks.
    chunk = total_count // world_size
    send = list(src.split(chunk))
    recv = [torch.empty(chunk, dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_to_all(recv, send)
    ref = torch.cat(recv, dim=0)

    torch.testing.assert_close(out, ref, atol=0.001, rtol=0.02)
    if rank == 0:
        print(f"OK: alltoall_auto on {world_size} GPUs matches NCCL reference "
              f"(total_count={total_count}, dtype={dtype})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
