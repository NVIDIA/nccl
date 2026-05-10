# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch interop helpers for nccl.ep.

Currently exposes :func:`get_nccl_comm_from_group`, which mirrors vLLM's
pattern: always create a fresh NCCL communicator (rather than extracting
one from a PyTorch ``ProcessGroup``, which is fragile across torch
versions).
"""

from __future__ import annotations

import nccl.core as nccl

try:
    import torch
    import torch.distributed as dist

    _torch_enabled = True
except ImportError:
    _torch_enabled = False


__all__ = ["get_nccl_comm_from_group"]


def get_nccl_comm_from_group(group=None) -> nccl.Communicator:
    """Create a fresh NCCL communicator that mirrors *group*'s membership.

    Args:
        group: A torch ``ProcessGroup``; ``None`` for the default group.

    Returns:
        Initialized :class:`nccl.core.Communicator`.

    Raises:
        ModuleNotFoundError: If PyTorch is not installed.
        RuntimeError: If ``torch.distributed`` is not initialized.
    """
    if not _torch_enabled:
        raise ModuleNotFoundError("PyTorch is not installed")

    if group is not None:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        src = dist.get_process_group_ranks(group)[0]
    elif dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        src = 0
    else:
        raise RuntimeError(
            "PyTorch distributed is not initialized. "
            "Call torch.distributed.init_process_group(...) first."
        )

    device = torch.cuda.current_device()
    unique_id = nccl.get_unique_id(empty=(rank != 0))

    try:
        backend_name = dist.get_backend(group) if group else dist.get_backend()
    except Exception:
        backend_name = "nccl"

    # bytearray (not bytes) — torch.frombuffer needs a writable buffer.
    buf = bytearray(unique_id.as_bytes)
    tensor = torch.frombuffer(buf, dtype=torch.uint8)
    if backend_name == "nccl":
        tensor = tensor.to(device)
    dist.broadcast(tensor, src=src, group=group)
    unique_id = nccl.UniqueId.from_bytes(tensor.cpu().numpy().tobytes())

    with torch.cuda.device(device):
        return nccl.Communicator.init(world_size, rank, unique_id)
