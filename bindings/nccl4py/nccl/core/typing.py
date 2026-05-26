# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Type definitions, protocols, and type aliases for NCCL4Py.

This module defines all type specifications used throughout NCCL4Py including
data types, reduction operators, buffer specifications, and protocol classes
for DLPack and CUDA Array Interface compatibility. It provides type-safe
interfaces for NCCL operations with comprehensive type hints.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Protocol, TypeAlias, Union

import numpy as _np
from cuda.core import Buffer, Device, Stream
from cuda.core.typing import IsStreamType

__all__ = [
    "NcclDataType",
    "NcclRedOp",
    "NcclGinType",
    "NcclGinConnectionType",
    "NcclCommMemStat",
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
    """Raised when an argument provided to an NCCL4Py API is invalid.

    Used for argument validation errors that the Python layer detects before
    forwarding the call to NCCL (e.g. unsupported dtype, mismatched buffer
    counts, wrong device). Errors raised by NCCL itself are reported as
    NCCLError from the bindings layer.
    """

    def __init__(self, msg):
        self.msg = msg
        super().__init__(msg)

    def __reduce__(self):
        return (type(self), (self.msg,))

    def __repr__(self):
        return f"<NcclInvalid: {self.msg}>"


###############################################################################
# Enums (mirror nccl.h)
###############################################################################


class NcclCommMemStat(IntEnum):
    """Memory-statistic selector, mirroring :c:type:`ncclCommMemStat_t`.

    Used as the ``stat`` argument of :py:meth:`Communicator.get_mem_stat`
    to identify which memory statistic to query. All values are returned
    in bytes except :py:attr:`GPU_MEM_SUSPENDED`, which is a 0/1 flag.
    """

    GPU_MEM_SUSPEND = 0
    """Communicator-allocated GPU memory that can be released by
    :py:meth:`Communicator.suspend` (bytes)."""
    GPU_MEM_SUSPENDED = 1
    """Whether communicator-allocated GPU memory is currently suspended
    (``0`` = active, ``1`` = suspended)."""
    GPU_MEM_PERSIST = 2
    """Communicator-allocated GPU memory that cannot be suspended
    (bytes)."""
    GPU_MEM_TOTAL = 3
    """Total communicator-allocated GPU memory tracked by NCCL (bytes)."""

    # Backward-compat aliases mirroring nccl.bindings.CommMemStat camelCase.
    GpuMemSuspend = 0
    GpuMemSuspended = 1
    GpuMemPersist = 2
    GpuMemTotal = 3


class NcclRedOp(IntEnum):
    """NCCL reduction operator, mirroring :c:type:`ncclRedOp_t`.

    Used as the ``op`` argument of reduction collectives
    (:py:meth:`Communicator.allreduce`, :py:meth:`Communicator.reduce`,
    :py:meth:`Communicator.reduce_scatter`).
    """

    SUM = 0
    """Element-wise sum (``+``)."""
    PROD = 1
    """Element-wise product (``*``)."""
    MAX = 2
    """Element-wise maximum."""
    MIN = 3
    """Element-wise minimum."""
    AVG = 4
    """Sum across all ranks divided by the number of ranks."""


class NcclGinType(IntEnum):
    """GIN transport type, mirroring :c:type:`ncclGinType_t`.

    Reported by :py:attr:`Communicator.gin_type` and
    :py:attr:`Communicator.railed_gin_type` to indicate which device-side
    network transport, if any, is available on the communicator.
    """

    NONE = 0
    """GIN not available on this communicator."""
    PROXY = 2
    """Proxy-based GIN. Network operations issued from a device kernel are
    relayed through a CPU proxy thread."""
    GDAKI = 3
    """GPUDirect Async Kernel-Initiated (GDA-KI). The kernel directly
    issues network operations to the NIC, bypassing the CPU proxy."""
    GPI = 4
    """GPU-Push Interface. GPU threads push network descriptors directly
    to a NIC-visible MMIO queue, with no CPU involvement and no memory
    barriers."""


class NcclGinConnectionType(IntEnum):
    """GIN connection topology, mirroring :c:type:`ncclGinConnectionType_t`.

    Set on the ``gin_connection_type`` field of
    :py:class:`NCCLDevCommRequirements` before calling
    :py:meth:`Communicator.create_dev_comm` to declare which peers must be
    reachable via GIN from device code.
    """

    NONE = 0
    """No GIN connection requested."""
    FULL = 1
    """Fully connected. Every rank in the communicator must be reachable
    from every other rank via GIN."""
    RAIL = 2
    """Rail-restricted. Ranks must be reachable via GIN only within the
    same rail (network plane)."""


class NcclDataType(IntEnum):
    """NCCL data type, mirroring :c:type:`ncclDataType_t`.

    Used as the ``dtype`` of buffer specs and as the ``datatype`` argument
    of NCCL collective operations. Supports conversion to/from NumPy
    dtypes via :py:meth:`from_numpy_dtype` and :py:attr:`numpy_dtype`.
    """

    INT8 = 0
    """Signed 8-bit integer."""
    CHAR = 0
    """Alias of :py:attr:`INT8`."""
    UINT8 = 1
    """Unsigned 8-bit integer."""
    INT32 = 2
    """Signed 32-bit integer."""
    INT = 2
    """Alias of :py:attr:`INT32`."""
    UINT32 = 3
    """Unsigned 32-bit integer."""
    INT64 = 4
    """Signed 64-bit integer."""
    UINT64 = 5
    """Unsigned 64-bit integer."""
    FLOAT16 = 6
    """IEEE half-precision floating point (2 bytes)."""
    HALF = 6
    """Alias of :py:attr:`FLOAT16`."""
    FLOAT32 = 7
    """IEEE single-precision floating point (4 bytes)."""
    FLOAT = 7
    """Alias of :py:attr:`FLOAT32`."""
    FLOAT64 = 8
    """IEEE double-precision floating point (8 bytes)."""
    DOUBLE = 8
    """Alias of :py:attr:`FLOAT64`."""
    BFLOAT16 = 9
    """Brain floating-point (16-bit truncated single precision; CUDA 11+)."""
    FLOAT8E4M3 = 10
    """8-bit floating point, 4 exponent + 3 mantissa bits (CUDA >= 11.8, SM >= 90)."""
    FLOAT8E5M2 = 11
    """8-bit floating point, 5 exponent + 2 mantissa bits (CUDA >= 11.8, SM >= 90)."""

    @classmethod
    def _missing_(cls, value):
        # Anything ``_np.dtype()`` can normalize -- numpy dtype objects,
        # dtype strings ("float32"), Python type objects (np.float32,
        # float, bool), etc. -- gets routed through ``from_numpy_dtype``. Plain
        # integer values that aren't valid members fall through to
        # IntEnum's default ``ValueError``.
        try:
            dtype = _np.dtype(value)
        except (TypeError, ValueError):
            return None
        return cls.from_numpy_dtype(dtype)

    @classmethod
    def from_numpy_dtype(cls, dtype: _np.dtype) -> "NcclDataType":
        """Maps a NumPy dtype to its NCCL equivalent.

        Args:
            dtype: A NumPy dtype. Mapped first by name (for ``ml-dtypes``
                like ``bfloat16``, ``float8_e4m3fn``, ``float8_e5m2``)
                and then by ``(kind, itemsize)`` for standard types.

        Returns:
            Corresponding :py:class:`NcclDataType` member.

        Raises:
            NcclInvalid: If the dtype has no NCCL equivalent.
        """
        dtype = _np.dtype(dtype)
        # ml-dtypes are matched by name first because they share (kind,
        # itemsize) with native float types.
        if dtype.name in _NAME_TO_NCCL:
            return _NAME_TO_NCCL[dtype.name]
        kind_size = (dtype.kind, dtype.itemsize)
        if kind_size in _KIND_SIZE_TO_NCCL:
            return _KIND_SIZE_TO_NCCL[kind_size]
        raise NcclInvalid(
            f"Unsupported data type: numpy dtype {dtype} "
            f"(kind={dtype.kind}, itemsize={dtype.itemsize}, "
            f"name={dtype.name}) has no NCCL equivalent"
        )

    @property
    def itemsize(self) -> int:
        """Size in bytes of a single element of this data type."""
        return _ITEMSIZE[self]

    @property
    def numpy_dtype(self) -> _np.dtype:
        """Equivalent NumPy dtype.

        Returns:
            NumPy dtype corresponding to this NCCL data type. For
            ``BFLOAT16`` and the float8 variants, ``ml-dtypes`` must be
            installed.

        Raises:
            NcclInvalid: If ``ml-dtypes`` is required but not installed.
        """
        dtype_str = _NCCL_TO_NUMPY_NAME[self]
        if dtype_str in _ML_DTYPE_NAMES:
            try:
                return _np.dtype(dtype_str)
            except TypeError as e:
                raise NcclInvalid(
                    f"Cannot create numpy dtype '{dtype_str}': ml-dtypes "
                    f"package is not installed. ml-dtypes is required for "
                    f"bfloat16 and float8 support. "
                    f"Install with: pip install ml-dtypes"
                ) from e
        return _np.dtype(dtype_str)


_ITEMSIZE: dict[NcclDataType, int] = {
    NcclDataType.INT8: 1,
    NcclDataType.UINT8: 1,
    NcclDataType.FLOAT8E4M3: 1,
    NcclDataType.FLOAT8E5M2: 1,
    NcclDataType.FLOAT16: 2,
    NcclDataType.BFLOAT16: 2,
    NcclDataType.INT32: 4,
    NcclDataType.UINT32: 4,
    NcclDataType.FLOAT32: 4,
    NcclDataType.INT64: 8,
    NcclDataType.UINT64: 8,
    NcclDataType.FLOAT64: 8,
}

_NAME_TO_NCCL: dict[str, NcclDataType] = {
    "bfloat16": NcclDataType.BFLOAT16,
    "float8_e4m3fn": NcclDataType.FLOAT8E4M3,
    "float8_e5m2": NcclDataType.FLOAT8E5M2,
}

_KIND_SIZE_TO_NCCL: dict[tuple[str, int], NcclDataType] = {
    ("f", 2): NcclDataType.FLOAT16,
    ("f", 4): NcclDataType.FLOAT32,
    ("f", 8): NcclDataType.FLOAT64,
    ("i", 1): NcclDataType.INT8,
    ("i", 4): NcclDataType.INT32,
    ("i", 8): NcclDataType.INT64,
    ("u", 1): NcclDataType.UINT8,
    ("u", 4): NcclDataType.UINT32,
    ("u", 8): NcclDataType.UINT64,
}

_NCCL_TO_NUMPY_NAME: dict[NcclDataType, str] = {
    NcclDataType.INT8: "int8",
    NcclDataType.UINT8: "uint8",
    NcclDataType.INT32: "int32",
    NcclDataType.UINT32: "uint32",
    NcclDataType.INT64: "int64",
    NcclDataType.UINT64: "uint64",
    NcclDataType.FLOAT16: "float16",
    NcclDataType.FLOAT32: "float32",
    NcclDataType.FLOAT64: "float64",
    NcclDataType.BFLOAT16: "bfloat16",
    NcclDataType.FLOAT8E4M3: "float8_e4m3fn",
    NcclDataType.FLOAT8E5M2: "float8_e5m2",
}

_ML_DTYPE_NAMES: frozenset[str] = frozenset({"bfloat16", "float8_e4m3fn", "float8_e5m2"})


###############################################################################
# Module-level constants
###############################################################################

INT8 = NcclDataType.INT8
CHAR = NcclDataType.CHAR
UINT8 = NcclDataType.UINT8
INT32 = NcclDataType.INT32
INT = NcclDataType.INT
UINT32 = NcclDataType.UINT32
INT64 = NcclDataType.INT64
UINT64 = NcclDataType.UINT64
FLOAT16 = NcclDataType.FLOAT16
HALF = NcclDataType.HALF
FLOAT32 = NcclDataType.FLOAT32
FLOAT = NcclDataType.FLOAT
FLOAT64 = NcclDataType.FLOAT64
DOUBLE = NcclDataType.DOUBLE
BFLOAT16 = NcclDataType.BFLOAT16
FLOAT8E4M3 = NcclDataType.FLOAT8E4M3
FLOAT8E5M2 = NcclDataType.FLOAT8E5M2

SUM = NcclRedOp.SUM
PROD = NcclRedOp.PROD
MAX = NcclRedOp.MAX
MIN = NcclRedOp.MIN
AVG = NcclRedOp.AVG


###############################################################################
# Buffer / device / stream protocols and aliases
###############################################################################


class SupportsDLPack(Protocol):
    """Protocol for objects implementing the DLPack data interchange protocol."""

    def __dlpack__(self, /, *, stream: Any | None = None) -> _PyCapsule: ...
    def __dlpack_device__(self) -> tuple[_DeviceType, _DeviceID]: ...


class SupportsCAI(Protocol):
    """Protocol for objects implementing the CUDA Array Interface."""

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...


NcclSupportedBuffer: TypeAlias = Union[Buffer, SupportsDLPack, SupportsCAI]

NcclBufferSpec: TypeAlias = NcclSupportedBuffer
"""A buffer object accepted by NCCL operations: a :py:class:`cuda.core.Buffer`,
an object implementing the DLPack protocol, or an object exposing
``__cuda_array_interface__``."""

NcclScalarSpec: TypeAlias = Union[int, float, _np.ndarray, NcclSupportedBuffer]
"""A scalar value: a Python ``int`` / ``float``, a one-element NumPy array,
or a one-element device buffer in any of the forms accepted by
:py:data:`NcclBufferSpec`."""

NcclDeviceSpec: TypeAlias = Union[Device, int]
"""A CUDA device: a :py:class:`cuda.core.Device` or an integer device ID."""

NcclStreamSpec: TypeAlias = Union[Stream, IsStreamType, int]
"""A CUDA stream: a :py:class:`cuda.core.Stream`, an object implementing
``__cuda_stream__``, or an integer stream handle."""
