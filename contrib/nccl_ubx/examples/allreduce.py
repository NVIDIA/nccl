"""UB-X AllReduce — minimal end-to-end example.

Demonstrates the high-level convenience API:
  - `request_allocator` declares a process group and the largest tensor
    shape that will pass through it;
  - `get_sym_tensor` lazily allocates a symmetric-memory tensor;
  - `allreduce` runs the auto-selecting allreduce (Lamport ≤ 0.25 MB,
    otherwise MC).

Result is validated against `torch.distributed.all_reduce` (NCCL).

Launch (2 or more GPUs on a single node):

    torchrun --nproc-per-node=2 examples/allreduce.py
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

    from ubx.ops import request_allocator, get_sym_tensor, allreduce

    shape = (4096,)
    dtype = torch.bfloat16
    group = dist.group.WORLD

    # Declare the process group + max shape to the allocator registry.
    request_allocator(group, shape=shape, dtype=dtype)

    # Allocate a symmetric tensor and fill it with rank-distinct data.
    sym = get_sym_tensor(shape, dtype, group)
    torch.manual_seed(42 + rank)
    sym.copy_(torch.randn(shape, dtype=dtype, device=device))

    # Run the auto allreduce (kernel chosen by total tensor size).
    out = allreduce(sym)
    torch.cuda.synchronize()

    # NCCL reference for validation. Reduce in fp32 to avoid the
    # accumulation-order error inherent in NCCL's bf16 ring reduction.
    ref = sym.detach().clone().float()
    dist.all_reduce(ref)

    torch.testing.assert_close(out.float(), ref, atol=0.0625, rtol=0.02)
    if rank == 0:
        print(f"OK: allreduce on {world_size} GPUs matches NCCL reference "
              f"(shape={tuple(shape)}, dtype={dtype})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
