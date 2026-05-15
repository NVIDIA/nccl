# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""CuPy interoperability for NCCL4Py.

This module provides utilities for creating CuPy arrays backed by NCCL-allocated
memory, enabling zero-copy integration between NCCL operations and CuPy workflows.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from nccl.core.buffer import mem_alloc
from nccl.core.typing import NcclInvalid, NcclDataType

__all__ = ["empty", "resolve_array"]


try:
    import cupy

    _cupy_enabled = True
except ImportError:
    _cupy_enabled = False


def _to_nccl_dtype(cupy_dtype) -> NcclDataType:
    """Converts a CuPy dtype to NcclDataType.

    CuPy dtypes are NumPy dtype objects, so resolution is delegated to
    :py:meth:`NcclDataType.from_numpy_dtype`. NumPy ``bool`` has no NCCL
    equivalent and is mapped to ``UINT8``. For bfloat16, float8_e4m3fn,
    and float8_e5m2, the optional ml-dtypes package is required.

    Args:
        cupy_dtype: CuPy / NumPy data type to convert.

    Returns:
        Corresponding :py:class:`NcclDataType` member.

    Raises:
        ModuleNotFoundError: If CuPy is not installed.
        NcclInvalid: If the dtype has no NCCL equivalent (e.g. complex,
            int16, uint16, structured, datetime, or string types).
    """
    if not _cupy_enabled:
        raise ModuleNotFoundError("CuPy is not installed")

    try:
        np_dtype = np.dtype(cupy_dtype)
    except (TypeError, ValueError) as e:
        raise NcclInvalid(
            f"Invalid data type: could not convert {cupy_dtype} "
            f"(type: {type(cupy_dtype).__name__}) to numpy dtype: {e}"
        )

    if np_dtype.kind == "b":
        return NcclDataType.UINT8

    return NcclDataType.from_numpy_dtype(np_dtype)


def _allocate_nccl_array(shape: tuple[int, ...], dtype: np.dtype, order: str) -> cupy.ndarray:
    """Allocates an NCCL-backed CuPy array on the current device.

    Args:
        shape: Shape of the array.
        dtype: NumPy data type.
        order: Memory order, 'C' (row-major) or 'F' (column-major).

    Returns:
        Allocated CuPy array backed by NCCL-managed memory.

    Raises:
        ModuleNotFoundError: If CuPy is not installed.
    """
    if not _cupy_enabled:
        raise ModuleNotFoundError("CuPy is not installed")

    size = int(np.prod(shape) * dtype.itemsize)

    # Allocate buffer on current device (CuPy manages device context)
    buf = mem_alloc(size)

    # Important! Disable copy to force allocation to stay in NCCL memory
    cupy_array = cupy.from_dlpack(buf, copy=False)
    return cupy_array.view(dtype).reshape(shape, order=order)


def empty(
    shape: int | tuple[int, ...],
    dtype: str | np.dtype | cupy.dtype | type = float,
    order: Literal["C", "F"] = "C",
) -> cupy.ndarray:
    """Creates an uninitialized CuPy array backed by NCCL-allocated memory.

    Returns an array filled with uninitialized data using NCCL's memory
    allocator. This provides a CuPy-compatible interface while using NCCL's
    memory allocator for efficient GPU memory management in distributed
    scenarios. Unlike cupy.empty, the underlying memory is allocated through
    NCCL.

    Memory is automatically freed when the array is garbage collected; no
    explicit free call is required. For zero-copy optimization, register the
    array using :py:meth:`~nccl.core.Communicator.register_buffer` or
    :py:meth:`~nccl.core.Communicator.register_window`.

    Args:
        shape: Shape of the array.
        dtype: Data type specifier. Defaults to ``float``.
        order: Memory layout. 'C' for row-major (C-style), 'F' for
            column-major (Fortran-style). Defaults to 'C'.

    Returns:
        An uninitialized CuPy array backed by NCCL-allocated memory.

    Raises:
        NcclInvalid: If order is not 'C' or 'F'.
        ModuleNotFoundError: If CuPy is not installed.
    """
    if not _cupy_enabled:
        raise ModuleNotFoundError("CuPy is not installed")

    # Validate order parameter
    if order not in ("C", "F"):
        raise NcclInvalid(
            f"Invalid memory order: must be 'C' or 'F', got '{order}'. "
            f"NCCL arrays support row-major (C) and column-major (F) layouts only."
        )

    # Parse shape
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    # Parse dtype
    np_dtype = np.dtype(dtype)

    # Allocate NCCL-backed array on current device
    view = _allocate_nccl_array(shape, np_dtype, order)

    return view


def resolve_array(array: cupy.ndarray) -> tuple[int, int, NcclDataType, int]:
    """Resolves a CuPy array to its NCCL buffer descriptor.

    Args:
        array: CuPy array to resolve.

    Returns:
        Tuple of (ptr, count, dtype, device_id): device pointer, element
        count, NCCL data type, and CUDA device ID.

    Raises:
        ModuleNotFoundError: If CuPy is not installed.
        NcclInvalid: If array is not a CuPy ndarray or its dtype has no
            NCCL equivalent.
    """
    if not _cupy_enabled:
        raise ModuleNotFoundError("CuPy is not installed")

    if not isinstance(array, cupy.ndarray):
        raise NcclInvalid(f"array must be a CuPy array, got {type(array).__name__}")

    return (array.data.ptr, array.size, _to_nccl_dtype(array.dtype), array.device.id)
