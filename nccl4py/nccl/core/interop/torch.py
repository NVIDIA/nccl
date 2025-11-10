# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See LICENSE.txt for license information

"""
PyTorch interoperability for NCCL4Py.

This module provides utilities for creating PyTorch tensors backed by NCCL-allocated
memory, enabling zero-copy integration between NCCL operations and PyTorch workflows.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from nccl.core.buffer import mem_alloc
from nccl.core.typing import (
    NcclInvalid,
    NcclDataType,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    INT8,
    INT32,
    INT64,
    UINT8,
    BFLOAT16,
    FLOAT8E4M3,
    FLOAT8E5M2,
)

__all__ = ["empty", "resolve_tensor"]


try:
    import torch

    _torch_enabled = True
except ImportError:
    _torch_enabled = False


def _to_nccl_dtype(torch_dtype) -> NcclDataType:
    """
    Converts PyTorch dtype to NcclDataType.

    Args:
        - torch_dtype (torch.dtype): PyTorch data type.

    Returns:
        ``NcclDataType``: Corresponding NCCL data type (global constant).

    Raises:
        - ``ModuleNotFoundError``: If PyTorch is not installed.
        - ``NcclInvalid``: If torch_dtype has no NCCL equivalent.
    """
    if not _torch_enabled:
        raise ModuleNotFoundError("PyTorch is not installed")

    _torch_to_nccl = {
        # Float types (all NCCL-supported)
        torch.float16: FLOAT16,
        torch.half: FLOAT16,  # Alias
        torch.float32: FLOAT32,
        torch.float: FLOAT32,  # Alias
        torch.float64: FLOAT64,
        torch.double: FLOAT64,  # Alias
        torch.bfloat16: BFLOAT16,
        # Signed integer types
        torch.int8: INT8,
        torch.int32: INT32,
        torch.int64: INT64,
        torch.long: INT64,  # Alias
        # Unsigned types
        torch.uint8: UINT8,
        torch.bool: UINT8,  # Map bool to uint8 for convenience
    }

    # Add float8 types if available (PyTorch 2.1+)
    if hasattr(torch, "float8_e4m3fn"):
        _torch_to_nccl[torch.float8_e4m3fn] = FLOAT8E4M3
    if hasattr(torch, "float8_e5m2"):
        _torch_to_nccl[torch.float8_e5m2] = FLOAT8E5M2

    # Explicitly unsupported PyTorch dtypes (no NCCL equivalent)
    _unsupported_dtypes = set()

    # Integer types not in NCCL
    if hasattr(torch, "int16"):
        _unsupported_dtypes.add(torch.int16)
    if hasattr(torch, "short"):
        _unsupported_dtypes.add(torch.short)

    # Complex types (NCCL doesn't support)
    if hasattr(torch, "complex64"):
        _unsupported_dtypes.add(torch.complex64)
        _unsupported_dtypes.add(torch.cfloat)
    if hasattr(torch, "complex128"):
        _unsupported_dtypes.add(torch.complex128)
        _unsupported_dtypes.add(torch.cdouble)

    # Quantized types (NCCL doesn't support)
    for qtype in ["qint8", "quint8", "qint32", "quint4x2", "quint2x4"]:
        if hasattr(torch, qtype):
            _unsupported_dtypes.add(getattr(torch, qtype))

    # Check if dtype is explicitly unsupported
    if torch_dtype in _unsupported_dtypes:
        raise NcclInvalid(
            f"Unsupported data type: torch dtype {torch_dtype} has no NCCL equivalent. "
            f"NCCL does not support complex, int16, or quantized types."
        )

    # Check if dtype is supported
    if torch_dtype in _torch_to_nccl:
        return _torch_to_nccl[torch_dtype]
    else:
        # Unknown dtype - list what we do support
        supported_list = [str(dt) for dt in sorted(_torch_to_nccl.keys(), key=str)]
        raise NcclInvalid(
            f"Unknown torch dtype {torch_dtype}. "
            f"Supported types: {', '.join(supported_list[:12])}..."
        )


def _parse_device(device: torch.device | int | str | None) -> int:
    """
    Parses device specification into device index.

    Args:
        - device (torch.device | int | str, optional): Device specification.

    Returns:
        ``int``: CUDA device index.

    Raises:
        - ``ModuleNotFoundError``: If PyTorch is not installed.
        - ``NcclInvalid``: If device is not a CUDA device.
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
    """
    Allocates NCCL-backed tensor with specified parameters.

    Args:
        - shape (tuple[int, ...]): Shape of tensor.
        - dtype (torch.dtype): Data type.
        - device (int): CUDA device index.
        - morder (str): Memory order ("C" or "F").

    Returns:
        ``torch.Tensor``: Allocated tensor with NCCL-backed memory.

    Raises:
        - ``ModuleNotFoundError``: If PyTorch is not installed.
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
    """
    Creates an uninitialized PyTorch tensor backed by NCCL-allocated memory.

    Returns a tensor filled with uninitialized data using NCCL's memory allocator.
    This provides a PyTorch-compatible interface while using NCCL's memory allocator
    for efficient GPU memory management in distributed scenarios.

    Args:
        - *size (int...): A sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.
        - dtype (torch.dtype, optional): The desired data type of returned tensor. If None, uses global default (see ``torch.set_default_dtype()``). Defaults to None.
        - device (torch.device | int | str, optional): The device of the constructed tensor. If None, uses the current CUDA device. Defaults to None.
        - morder (Literal["C", "F"], optional): Memory layout - "C" for row-major (C-style), "F" for column-major (Fortran-style). Defaults to "C".

    Returns:
        ``torch.Tensor``: An uninitialized PyTorch tensor backed by NCCL-allocated memory.

    Raises:
        - ``NcclInvalid``: If morder is invalid (must be "C" or "F"), or device is not a CUDA device.
        - ``ModuleNotFoundError``: If PyTorch is not installed.

    Notes:
        - Unlike ``torch.empty()``, this allocates memory using NCCL's memory allocator.
        - For zero-copy optimization, register using ``Communicator.register_buffer()`` or ``Communicator.register_window()``.
        - Memory is automatically freed when the tensor is garbage collected. No explicit free call is required.
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
    """
    Resolves a PyTorch tensor to a tuple of (ptr, count, dtype, device_id).

    Args:
        - tensor (torch.Tensor): PyTorch tensor to resolve.

    Returns:
        ``tuple[int, int, NcclDataType, int]``: Tuple of (ptr, count, dtype, device_id).
    """
    if not _torch_enabled:
        raise ModuleNotFoundError("PyTorch is not installed")

    if not isinstance(tensor, torch.Tensor):
        raise NcclInvalid(f"tensor must be a PyTorch tensor, got {type(tensor).__name__}")

    return (tensor.data_ptr(), tensor.numel(), _to_nccl_dtype(tensor.dtype), tensor.device.index)
