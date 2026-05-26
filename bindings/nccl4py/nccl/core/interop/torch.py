# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""PyTorch interoperability for NCCL4Py.

This module provides utilities for creating PyTorch tensors backed by NCCL-allocated
memory, enabling zero-copy integration between NCCL operations and PyTorch workflows.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from nccl.core.buffer import mem_alloc
from nccl.core.typing import NcclInvalid, NcclDataType

__all__ = ["empty", "resolve_tensor"]


try:
    import torch

    _torch_enabled = True
except ImportError:
    _torch_enabled = False


def _to_nccl_dtype(torch_dtype) -> NcclDataType:
    """Converts a PyTorch dtype to NcclDataType.

    torch.bool is mapped to UINT8 for convenience. float8_e4m3fn and
    float8_e5m2 are supported on PyTorch 2.1+. PyTorch's same-singleton
    aliases (``torch.half`` is ``torch.float16``, ``torch.float`` is
    ``torch.float32``, ``torch.double`` is ``torch.float64``,
    ``torch.long`` is ``torch.int64``) need no separate map entries.

    Args:
        torch_dtype: PyTorch data type to convert.

    Returns:
        Corresponding NcclDataType global constant.

    Raises:
        ModuleNotFoundError: If PyTorch is not installed.
        NcclInvalid: If torch_dtype has no NCCL equivalent (complex,
            int16/short, or quantized types are explicitly unsupported).
    """
    if not _torch_enabled:
        raise ModuleNotFoundError("PyTorch is not installed")

    _torch_to_nccl = {
        torch.float16: NcclDataType.FLOAT16,
        torch.float32: NcclDataType.FLOAT32,
        torch.float64: NcclDataType.FLOAT64,
        torch.bfloat16: NcclDataType.BFLOAT16,
        torch.int8: NcclDataType.INT8,
        torch.int32: NcclDataType.INT32,
        torch.int64: NcclDataType.INT64,
        torch.uint8: NcclDataType.UINT8,
        torch.bool: NcclDataType.UINT8,  # NCCL has no bool; map to its natural byte-width.
    }
    # PyTorch 2.1+ float8 types (optional).
    if hasattr(torch, "float8_e4m3fn"):
        _torch_to_nccl[torch.float8_e4m3fn] = NcclDataType.FLOAT8E4M3
    if hasattr(torch, "float8_e5m2"):
        _torch_to_nccl[torch.float8_e5m2] = NcclDataType.FLOAT8E5M2

    try:
        return _torch_to_nccl[torch_dtype]
    except KeyError:
        raise NcclInvalid(
            f"Unsupported torch dtype {torch_dtype} has no NCCL equivalent. "
            f"NCCL does not support complex, int16, or quantized types."
        ) from None


def _parse_device(device: torch.device | int | str | None) -> int:
    """Parses a device specification into a CUDA device index.

    Args:
        device: Device specification (torch.device, int, or str), or
            ``None`` to use the current CUDA device.

    Returns:
        CUDA device index.

    Raises:
        ModuleNotFoundError: If PyTorch is not installed.
        NcclInvalid: If device does not refer to a CUDA device.
    """
    if not _torch_enabled:
        raise ModuleNotFoundError("PyTorch is not installed")

    if device is None:
        # Use current CUDA device
        return torch.cuda.current_device()
    elif isinstance(device, torch.device):
        if device.type != "cuda":
            raise NcclInvalid(f"NCCL tensors must be on CUDA device, got {device}")
        return device.index if device.index is not None else torch.cuda.current_device()
    elif isinstance(device, str):
        device_obj = torch.device(device)
        if device_obj.type != "cuda":
            raise NcclInvalid(f"NCCL tensors must be on CUDA device, got {device}")
        return device_obj.index if device_obj.index is not None else torch.cuda.current_device()
    else:
        # device is int, use it directly
        return device


def _allocate_nccl_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, device: int, morder: str
) -> torch.Tensor:
    """Allocates an NCCL-backed PyTorch tensor.

    Args:
        shape: Shape of the tensor.
        dtype: PyTorch data type.
        device: CUDA device index.
        morder: Memory order, 'C' (row-major) or 'F' (column-major).

    Returns:
        Allocated tensor backed by NCCL-managed memory.

    Raises:
        ModuleNotFoundError: If PyTorch is not installed.
    """
    if not _torch_enabled:
        raise ModuleNotFoundError("PyTorch is not installed")

    # Get itemsize for the dtype
    itemsize = torch.tensor([], dtype=dtype).element_size()
    size_bytes = int(np.prod(shape) * itemsize)

    # Allocate buffer on the specified device
    buf = mem_alloc(size_bytes, device=device)

    # Create tensor via DLPack
    torch_tensor = torch.from_dlpack(buf)  # type: ignore[attr-defined]
    view = torch_tensor.view(dtype).view(shape)

    # Handle Fortran order if requested
    if morder == "F":
        # Compute Fortran-style (column-major) strides
        if len(shape) > 0:
            strides = [1]
            for dim in shape[:-1]:
                strides.append(strides[-1] * dim)
            view = view.as_strided(size=shape, stride=strides)

    return view


def empty(
    *size,
    dtype: torch.dtype | None = None,
    device: torch.device | int | str | None = None,
    morder: Literal["C", "F"] = "C",
) -> torch.Tensor:
    """Creates an uninitialized PyTorch tensor backed by NCCL-allocated memory.

    Returns a tensor filled with uninitialized data using NCCL's memory
    allocator. This provides a PyTorch-compatible interface while using NCCL's
    memory allocator for efficient GPU memory management in distributed
    scenarios. Unlike torch.empty, the underlying memory is allocated through
    NCCL.

    Memory is automatically freed when the tensor is garbage collected; no
    explicit free call is required. For zero-copy optimization, register the
    tensor using :py:meth:`~nccl.core.Communicator.register_buffer` or
    :py:meth:`~nccl.core.Communicator.register_window`.

    Args:
        *size: A sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a single
            list/tuple.
        dtype: Desired data type of the tensor. If ``None``, uses
            torch.get_default_dtype(). Defaults to ``None``.
        device: Device of the tensor. If ``None``, uses the current CUDA
            device. Defaults to ``None``.
        morder: Memory layout. 'C' for row-major (C-style), 'F' for
            column-major (Fortran-style). Defaults to 'C'.

    Returns:
        An uninitialized PyTorch tensor backed by NCCL-allocated memory.

    Raises:
        NcclInvalid: If morder is not 'C' or 'F', or device is not a CUDA
            device.
        ModuleNotFoundError: If PyTorch is not installed.
    """
    if not _torch_enabled:
        raise ModuleNotFoundError("PyTorch is not installed")

    # Validate morder parameter
    if morder not in ("C", "F"):
        raise NcclInvalid(f"Invalid memory order: must be 'C' or 'F', got '{morder}'")

    # Parse size arguments (can be *args or a single tuple/list)
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        shape = tuple(size[0])
    else:
        shape = size

    # Handle dtype - use PyTorch's default if not specified
    if dtype is None:
        dtype = torch.get_default_dtype()

    # Parse device specification
    device_idx = _parse_device(device)

    # Allocate NCCL-backed tensor
    view = _allocate_nccl_tensor(shape, dtype, device_idx, morder)

    return view


def resolve_tensor(tensor: torch.Tensor) -> tuple[int, int, NcclDataType, int]:
    """Resolves a PyTorch tensor to its NCCL buffer descriptor.

    Args:
        tensor: PyTorch tensor to resolve.

    Returns:
        Tuple of (ptr, count, dtype, device_id): device pointer, element
        count, NCCL data type, and CUDA device ID.

    Raises:
        ModuleNotFoundError: If PyTorch is not installed.
        NcclInvalid: If tensor is not a PyTorch tensor or its dtype has no
            NCCL equivalent.
    """
    if not _torch_enabled:
        raise ModuleNotFoundError("PyTorch is not installed")

    if not isinstance(tensor, torch.Tensor):
        raise NcclInvalid(f"tensor must be a PyTorch tensor, got {type(tensor).__name__}")

    return (tensor.data_ptr(), tensor.numel(), _to_nccl_dtype(tensor.dtype), tensor.device.index)
