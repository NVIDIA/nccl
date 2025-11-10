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
CuPy interoperability for NCCL4Py.

This module provides utilities for creating CuPy arrays backed by NCCL-allocated
memory, enabling zero-copy integration between NCCL operations and CuPy workflows.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from nccl.core.buffer import mem_alloc
from nccl.core.typing import (
    NcclInvalid,
    NcclDataType,
    INT8,
    UINT8,
    INT32,
    UINT32,
    INT64,
    UINT64,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    BFLOAT16,
    FLOAT8E4M3,
    FLOAT8E5M2,
)

__all__ = ["empty", "resolve_array"]


try:
    import cupy

    _cupy_enabled = True
except ImportError:
    _cupy_enabled = False


def _to_nccl_dtype(cupy_dtype) -> NcclDataType:
    """
    Converts CuPy dtype to NcclDataType.

    Note:
        CuPy dtypes are numpy.dtype objects. This function maps them to
        global NCCL data type constants.

        ml-dtypes is needed for bfloat16, float8_e4m3fn, float8_e5m2.

    Args:
        - cupy_dtype (cupy.dtype): CuPy data type.

    Returns:
        ``NcclDataType``: Corresponding NCCL data type (global constant).

    Raises:
        - ``ModuleNotFoundError``: If CuPy is not installed.
        - ``NcclInvalid``: If cupy_dtype has no NCCL equivalent.
    """
    if not _cupy_enabled:
        raise ModuleNotFoundError("CuPy is not installed")

    # CuPy dtypes are numpy dtypes - convert to numpy dtype first
    try:
        np_dtype = np.dtype(cupy_dtype)
    except (TypeError, ValueError) as e:
        raise NcclInvalid(
            f"Invalid data type: could not convert {cupy_dtype} (type: {type(cupy_dtype).__name__}) "
            f"to numpy dtype: {e}"
        )

    # Map numpy dtype to global NcclDataType constants
    # First try name-based for ml-dtypes (higher priority)
    if np_dtype.name == "bfloat16":
        return BFLOAT16
    elif np_dtype.name == "float8_e4m3fn":
        return FLOAT8E4M3
    elif np_dtype.name == "float8_e5m2":
        return FLOAT8E5M2

    # Explicitly unsupported types - detect and give clear error
    _unsupported_kinds = {
        "c": "complex",
        "V": "void/structured",
        "O": "object",
        "S": "byte string",
        "U": "unicode string",
        "M": "datetime",
        "m": "timedelta",
    }

    if np_dtype.kind in _unsupported_kinds:
        kind_name = _unsupported_kinds[np_dtype.kind]
        raise NcclInvalid(
            f"Unsupported data type: numpy dtype {np_dtype} ({kind_name} type) has no NCCL equivalent. "
            f"NCCL only supports numeric types."
        )

    # Map standard numpy types by kind+size
    _numpy_to_nccl = {
        ("f", 2): FLOAT16,
        ("f", 4): FLOAT32,
        ("f", 8): FLOAT64,
        ("i", 1): INT8,
        ("i", 2): None,  # int16 - explicitly not supported by NCCL
        ("i", 4): INT32,
        ("i", 8): INT64,
        ("u", 1): UINT8,
        ("u", 2): None,  # uint16 - explicitly not supported by NCCL
        ("u", 4): UINT32,
        ("u", 8): UINT64,
        ("b", 1): UINT8,  # bool -> uint8
    }

    kind_size = (np_dtype.kind, np_dtype.itemsize)
    if kind_size in _numpy_to_nccl:
        result = _numpy_to_nccl[kind_size]
        if result is None:
            raise NcclInvalid(
                f"Unsupported data type: numpy dtype {np_dtype} (kind={np_dtype.kind}, "
                f"itemsize={np_dtype.itemsize}). NCCL does not support {np_dtype.kind}{np_dtype.itemsize * 8}-bit types."
            )
        return result
    else:
        raise NcclInvalid(
            f"Unsupported data type: numpy dtype {np_dtype} (kind={np_dtype.kind}, "
            f"itemsize={np_dtype.itemsize}, name={np_dtype.name}) has no NCCL equivalent. "
            f"NCCL supports: int8/32/64, uint8/32/64, float16/32/64, bfloat16, float8_e4m3fn, float8_e5m2."
        )


def _allocate_nccl_array(shape: tuple[int, ...], dtype: np.dtype, order: str) -> cupy.ndarray:
    """
    Allocates NCCL-backed CuPy array with specified parameters.

    Args:
        - shape (tuple[int, ...]): Shape of array.
        - dtype (np.dtype): Data type.
        - order (str): Memory order ("C" or "F").

    Returns:
        ``cupy.ndarray``: Allocated array with NCCL-backed memory.

    Raises:
        - ``ModuleNotFoundError``: If CuPy is not installed.

    Notes:
        Uses current CuPy device for allocation.
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
    """
    Creates an uninitialized CuPy array backed by NCCL-allocated memory.

    Returns an array filled with uninitialized data using NCCL's memory allocator.
    This provides a CuPy-compatible interface while using NCCL's memory allocator
    for efficient GPU memory management in distributed scenarios.

    Args:
        - shape (int | tuple[int, ...]): Dimensionalities of the array.
        - dtype (str | np.dtype | cupy.dtype | type, optional): Data type specifier. Defaults to float.
        - order (Literal['C', 'F'], optional): Row-major (C-style) or column-major (Fortran-style) order. Only 'C' and 'F' are supported. Defaults to 'C'.

    Returns:
        ``cupy.ndarray``: An uninitialized CuPy array backed by NCCL-allocated memory.

    Raises:
        - ``NcclInvalid``: If order is not 'C' or 'F'.
        - ``ModuleNotFoundError``: If CuPy is not installed.

    Notes:
        - Unlike ``cupy.empty()``, this allocates memory using NCCL's memory allocator.
        - For zero-copy optimization, register using ``Communicator.register_buffer()`` or ``Communicator.register_window()``.
        - Memory is automatically freed when the array is garbage collected. No explicit free call is required.
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
    """
    Resolves a CuPy array to a tuple of (ptr, count, dtype, device_id).

    Args:
        - array (cupy.ndarray): CuPy array to resolve.

    Returns:
        ``tuple[int, int, NcclDataType, int]``: Tuple of (ptr, count, dtype, device_id).
    """
    if not _cupy_enabled:
        raise ModuleNotFoundError("CuPy is not installed")

    if not isinstance(array, cupy.ndarray):
        raise NcclInvalid(f"array must be a CuPy array, got {type(array).__name__}")

    return (array.data.ptr, array.size, _to_nccl_dtype(array.dtype), array.device.id)
