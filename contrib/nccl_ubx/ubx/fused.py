"""Fused operations (residual + RMSNorm).

This module is DETACHABLE: it can be removed without affecting the core
collective API in allocator.py. It calls the same CUDA kernels but passes
non-null gamma/residual_in parameters to enable fusion paths.
"""

from __future__ import annotations

import torch
from typing import Optional

from .tensor import SymmTensor
from .allocator import SymmAllocator, AUTO_SWITCH_BYTES


def allreduce_fused(
    allocator: SymmAllocator,
    tensor_in: SymmTensor,
    hidden_size: int,
    residual_in: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    gamma: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
    smlimit: int = 0,
    cgasize: int = 0,
) -> torch.Tensor:
    """Allreduce with optional fused residual add and RMSNorm.

    Fusion is controlled by parameter presence:
    - gamma != None and eps != None: enables RMSNorm fusion
    - residual_in != None: enables residual addition
    - residual_out != None: writes updated residual

    Args:
        allocator: The SymmAllocator managing the memory pool.
        tensor_in: Input SymmTensor to allreduce.
        hidden_size: Hidden dimension size for RMSNorm.
        residual_in: Optional input residual to add after reduction.
        residual_out: Optional output buffer for updated residual.
        gamma: RMSNorm weight parameter. Enables RMSNorm when non-None.
        eps: RMSNorm epsilon. Required when gamma is provided.
        smlimit: SM count limit for communication kernels (0 = no limit).
        cgasize: CGA cluster size (0 = no clustering).

    Returns:
        The allreduced (and optionally fused) tensor.
    """
    fuse_layernorm = gamma is not None and eps is not None

    if tensor_in.numel() * tensor_in.element_size() > AUTO_SWITCH_BYTES:
        return allocator.allreduce_mc(
            tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm,
            gamma, eps, smlimit, cgasize,
        )
    else:
        return allocator.allreduce_lamport(
            tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm,
            gamma, eps, smlimit, cgasize,
        )
