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
Type definitions, protocols, and type aliases for NCCL4Py.

This module defines all type specifications used throughout NCCL4Py including
data types, reduction operators, buffer specifications, and protocol classes
for DLPack and CUDA Array Interface compatibility. It provides type-safe
interfaces for NCCL operations with comprehensive type hints.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, TypeAlias

import numpy as _np
from cuda.core.experimental import Buffer, Device, Stream

try:
    from cuda.core.experimental._stream import IsStreamT
except ImportError:
    try:
        from cuda.core import IsStreamT
    except ImportError:
        try:
            from cuda.core._stream import IsStreamT
        except ImportError:
            # ---- Fallback definition ----
            @runtime_checkable
            class IsStreamT(Protocol):
                def __cuda_stream__(self) -> tuple[int, int]:
                    """
                    Fallback Protocol for CUDA stream objects.

                    Returns:
                        (version: int, cudaStream_t address: int)
                    """
                    ...


from nccl import bindings as _nccl_bindings
from nccl.bindings import DataType, RedOp

__all__ = [
    "NcclDataType",
    "NcclRedOp",
    "NcclBufferSpec",
    "NcclScalarSpec",
    "NcclDeviceSpec",
    "NcclStreamSpec",
    "NcclInvalid",
    # Data type constants
    "INT8",
    "CHAR",
    "UINT8",
    "INT32",
    "INT",
    "UINT32",
    "INT64",
    "UINT64",
    "FLOAT16",
    "HALF",
    "FLOAT32",
    "FLOAT",
    "FLOAT64",
    "DOUBLE",
    "BFLOAT16",
    "FLOAT8E4M3",
    "FLOAT8E5M2",
    # Reduction op constants
    "SUM",
    "PROD",
    "MAX",
    "MIN",
    "AVG",
]

_PyCapsule: TypeAlias = object
_DeviceType: TypeAlias = int
_DeviceID: TypeAlias = int


class NcclInvalid(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return f"<NcclInvalid: {self.msg}>"


class NcclDataType:
    def __init__(self, datatype: int | _np.dtype):
        if isinstance(datatype, _np.dtype):
            # First try name-based mapping for ml-dtypes (has higher priority)
            _name_mapping = {
                "bfloat16": DataType.Bfloat16,
                "float8_e4m3fn": DataType.Float8e4m3,
                "float8_e5m2": DataType.Float8e5m2,
            }
            if datatype.name in _name_mapping:
                self._datatype = _name_mapping[datatype.name]
            else:
                # Fall back to kind+size mapping for standard NumPy types
                _mapping = {
                    ("f", 2): DataType.Float16,
                    ("f", 4): DataType.Float32,
                    ("f", 8): DataType.Float64,
                    ("i", 1): DataType.Int8,
                    ("i", 4): DataType.Int32,
                    ("i", 8): DataType.Int64,
                    ("u", 1): DataType.Uint8,
                    ("u", 4): DataType.Uint32,
                    ("u", 8): DataType.Uint64,
                }
                kind_size = (datatype.kind, datatype.itemsize)
                if kind_size in _mapping:
                    self._datatype = _mapping[kind_size]
                else:
                    raise NcclInvalid(
                        f"Unsupported data type: numpy dtype {datatype} (kind={datatype.kind}, "
                        f"itemsize={datatype.itemsize}, name={datatype.name}) has no NCCL equivalent"
                    )
        else:
            try:
                self._datatype = DataType(int(datatype))
            except Exception:
                raise NcclInvalid(
                    f"Unsupported data type: NCCL datatype value {datatype} is invalid"
                )

    def __int__(self) -> int:
        return int(self._datatype)

    def __str__(self) -> str:
        return self._datatype.name

    def __repr__(self) -> str:
        return f"NcclDataType({self._datatype.name})"

    def __eq__(self, other: object) -> bool:
        """
        Compares two NcclDataType instances for equality.

        Returns:
            ``bool``: True if data types are equal.
        """
        if not isinstance(other, NcclDataType):
            return NotImplemented
        return self._datatype == other._datatype

    def __hash__(self) -> int:
        """
        Makes NcclDataType hashable.

        Returns:
            ``int``: Hash value.
        """
        return hash(self._datatype)

    @property
    def value(self) -> int:
        """
        Integer value of the NCCL data type.

        Returns:
            ``int``: Data type value.
        """
        return int(self._datatype)

    @property
    def name(self) -> str:
        """
        Name of the NCCL data type.

        Returns:
            ``str``: Data type name (e.g., "Float32", "Int64").
        """
        return self._datatype.name

    @property
    def itemsize(self) -> int:
        """
        Size in bytes of this data type.

        Returns:
            ``int``: Byte size (1, 2, 4, or 8).

        Raises:
            - ``NcclInvalid``: If data type is unsupported.
        """
        if self._datatype in [
            DataType.Int8,
            DataType.Char,
            DataType.Uint8,
            DataType.Float8e4m3,
            DataType.Float8e5m2,
        ]:
            return 1
        elif self._datatype in [DataType.Float16, DataType.Half, DataType.Bfloat16]:
            return 2
        elif self._datatype in [
            DataType.Int32,
            DataType.Int,
            DataType.Uint32,
            DataType.Float32,
            DataType.Float,
        ]:
            return 4
        elif self._datatype in [DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Double]:
            return 8
        else:
            raise NcclInvalid(
                f"Unsupported data type: NCCL datatype {self._datatype} has no byte size mapping"
            )

    @property
    def numpy_dtype(self) -> _np.dtype:
        """
        NumPy dtype corresponding to this NCCL data type.

        Returns:
            ``np.dtype``: Equivalent NumPy data type.

        Raises:
            - ``NcclInvalid``: If data type is unsupported.
        """
        # Mapping from NCCL DataType to numpy dtype string
        _dtype_to_numpy = {
            DataType.Int8: "int8",
            DataType.Char: "int8",
            DataType.Uint8: "uint8",
            DataType.Int32: "int32",
            DataType.Int: "int32",
            DataType.Uint32: "uint32",
            DataType.Int64: "int64",
            DataType.Uint64: "uint64",
            DataType.Float16: "float16",
            DataType.Half: "float16",
            DataType.Float32: "float32",
            DataType.Float: "float32",
            DataType.Float64: "float64",
            DataType.Double: "float64",
            # ml-dtypes
            DataType.Bfloat16: "bfloat16",
            DataType.Float8e4m3: "float8_e4m3fn",
            DataType.Float8e5m2: "float8_e5m2",
        }

        if self._datatype not in _dtype_to_numpy:
            raise NcclInvalid(
                f"Unsupported data type: NCCL datatype {self._datatype} has no numpy dtype mapping"
            )

        dtype_str = _dtype_to_numpy[self._datatype]

        # For ml-dtypes, provide helpful error if package is missing (optional dependency)
        if dtype_str in ("bfloat16", "float8_e4m3fn", "float8_e5m2"):
            try:
                return _np.dtype(dtype_str)
            except TypeError as e:
                raise NcclInvalid(
                    f"Cannot create numpy dtype '{dtype_str}': ml-dtypes package is not installed. "
                    f"ml-dtypes is required for bfloat16 and float8 support. "
                    f"Install with: pip install ml-dtypes"
                ) from e
        else:
            return _np.dtype(dtype_str)


INT8 = NcclDataType(DataType.Int8)
CHAR = NcclDataType(DataType.Char)
UINT8 = NcclDataType(DataType.Uint8)
INT32 = NcclDataType(DataType.Int32)
INT = NcclDataType(DataType.Int)
UINT32 = NcclDataType(DataType.Uint32)
INT64 = NcclDataType(DataType.Int64)
UINT64 = NcclDataType(DataType.Uint64)
FLOAT16 = NcclDataType(DataType.Float16)
HALF = NcclDataType(DataType.Half)
FLOAT32 = NcclDataType(DataType.Float32)
FLOAT = NcclDataType(DataType.Float)
FLOAT64 = NcclDataType(DataType.Float64)
DOUBLE = NcclDataType(DataType.Double)
BFLOAT16 = NcclDataType(DataType.Bfloat16)
FLOAT8E4M3 = NcclDataType(DataType.Float8e4m3)
FLOAT8E5M2 = NcclDataType(DataType.Float8e5m2)


class NcclRedOp:
    """
    NCCL reduction operator wrapper with validation.

    Wraps NCCL reduction operator values and validates them against
    the built-in operators (Sum, Prod, Max, Min, Avg).
    """

    def __init__(self, value: int):
        """
        Initializes NcclRedOp with validation.

        Args:
            - value (int): Integer value of the reduction operator.

        Raises:
            - ``NcclInvalid``: If the reduction operator value is invalid.
        """
        try:
            _ro = getattr(_nccl_bindings, "RedOp")
            # Access a few known attributes to ensure class exists
            _ = getattr(_ro, "Sum")
        except Exception:
            raise NcclInvalid("NCCL bindings error: NcclRedOp bindings not found")

        # Validate that the value corresponds to a valid reduction operator
        try:
            self._redop_value = _ro(value)
            self._redop_name = self._redop_value.name
        except Exception:
            raise NcclInvalid(
                f"Invalid reduction operator: value {value} is not a valid NCCL reduction operator"
            )

    def __int__(self) -> int:
        return int(self._redop_value)

    def __str__(self) -> str:
        return self._redop_name

    def __repr__(self) -> str:
        return f"<NcclRedOp: {self._redop_name}>"

    @property
    def value(self) -> int:
        """
        Integer value of the reduction operator.

        Returns:
            ``int``: Operator value.
        """
        return int(self._redop_value)

    @property
    def name(self) -> str:
        """
        Gets the name of the reduction operator.

        Returns:
            ``str``: Operator name (e.g., "Sum", "Max").
        """
        return self._redop_name


SUM = NcclRedOp(RedOp.Sum)
PROD = NcclRedOp(RedOp.Prod)
MAX = NcclRedOp(RedOp.Max)
MIN = NcclRedOp(RedOp.Min)
AVG = NcclRedOp(RedOp.Avg)


class SupportsDLPack(Protocol):
    def __dlpack__(self, /, *, stream: Any | None = None) -> _PyCapsule: ...
    def __dlpack_device__(self) -> tuple[_DeviceType, _DeviceID]: ...


class SupportsCAI(Protocol):
    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...


NcclSupportedBuffer: TypeAlias = Buffer | SupportsDLPack | SupportsCAI

NcclBufferSpec: TypeAlias = NcclSupportedBuffer

NcclScalarSpec: TypeAlias = (
    int
    | float
    | _np.ndarray  # NumPy array for host scalars
    | NcclSupportedBuffer  # Device buffer (Buffer, DLPack, or CUDA Array Interface)
)

NcclDeviceSpec: TypeAlias = Device | int
NcclStreamSpec: TypeAlias = Stream | IsStreamT | int
