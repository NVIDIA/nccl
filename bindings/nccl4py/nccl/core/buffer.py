# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Buffer specification and memory allocation for NCCL operations.

This module provides utilities for allocating NCCL-optimized device memory and
resolving various buffer specifications (arrays, tensors, tuples) into the raw
pointers, counts, and dtypes required by NCCL operations.
"""

from __future__ import annotations
import math

from cuda.core import Buffer
from cuda.core.utils import StridedMemoryView, args_viewable_as_strided_memory

from nccl.core.cuda import get_device_id, get_stream_ptr
from nccl.core.memory import get_memory_resource
from nccl.core.typing import (
    NcclBufferSpec,
    NcclDataType,
    NcclDeviceSpec,
    NcclInvalid,
    NcclStreamSpec,
)


__all__ = ["mem_alloc", "mem_free"]


def mem_alloc(size: int, device: NcclDeviceSpec | None = None) -> Buffer:
    """Allocates GPU buffer memory using NCCL's memory allocator.

    The actual allocated size may be larger than requested due to buffer
    granularity requirements from NCCL optimizations. The returned buffer can
    be explicitly freed with :py:func:`mem_free` or automatically freed when
    garbage collected.

    Args:
        size: Number of bytes to allocate.
        device: Target CUDA device. Defaults to the current device.

    Returns:
        A CUDA buffer object backed by NCCL-managed memory. The buffer is
        allocated on the specified device; the current device is restored
        after allocation.
    """
    device_id = get_device_id(device)

    mr = get_memory_resource(device_id)
    return mr.allocate(size)


def mem_free(buf: Buffer) -> None:
    """Frees memory allocated by :py:func:`mem_alloc`.

    Explicit deallocation is optional. Memory is automatically freed when the
    Buffer object is garbage collected.

    Args:
        buf: The buffer to free.
    """
    buf.close()


@args_viewable_as_strided_memory((0,))
def _resolve_buffer(buffer, stream_ptr: int | None) -> StridedMemoryView:
    return buffer.view(stream_ptr)


class NcclBuffer:
    """Resolves user-provided buffer specifications to raw pointer, count, dtype, and device.

    This class handles various buffer input formats and extracts the necessary
    metadata for NCCL operations. Buffer specifications can be a CUDA Array
    Interface or DLPack compatible object, or a ``cuda.core.Buffer`` (any
    PyTorch tensor or CuPy array works via DLPack). Element count is
    automatically derived from buffer size and dtype; use slicing to control
    element count.

    When stream is ``None``, the special sentinel value -1 is used internally.
    This sentinel is required by cuda.core's StridedMemoryView.view to
    indicate that no stream synchronization should be performed between the
    producer stream (where the buffer was created) and the consumer stream
    (where NCCL operations will use it). This avoids implicit synchronization
    overhead when the user is managing stream dependencies manually. When a
    stream is provided, its handle (typically 0 for the default stream or a
    valid stream pointer) is used for proper synchronization.

    Attributes:
        ptr: Raw device pointer address.
        count: Number of elements in the buffer.
        dtype: NCCL data type of buffer elements.
        device_id: CUDA device ID where the buffer resides.
    """

    def __init__(self, buffer: NcclBufferSpec, stream: NcclStreamSpec | None = None) -> None:
        """Initializes an NcclBuffer from a buffer specification.

        Args:
            buffer: Buffer specification to resolve.
            stream: CUDA stream for synchronization. Defaults to ``None`` (no
                synchronization between producer and consumer streams).

        Raises:
            NcclInvalid: If the buffer specification is invalid or malformed.
        """
        self.stream_ptr = -1 if stream is None else get_stream_ptr(stream)

        resolved = None

        try:
            from nccl.core.interop.torch import resolve_tensor

            resolved = resolve_tensor(buffer)
        except (ImportError, ModuleNotFoundError, NcclInvalid):
            pass

        if resolved is None:
            try:
                from nccl.core.interop.cupy import resolve_array

                resolved = resolve_array(buffer)
            except (ImportError, ModuleNotFoundError, NcclInvalid):
                pass

        if resolved is not None:
            self._ptr = resolved[0]
            self._count = resolved[1]
            self._dtype = resolved[2]
            self._device_id = resolved[3]
        else:
            view = _resolve_buffer(buffer, self.stream_ptr)
            self._ptr = view.ptr
            self._count = math.prod(view.shape)
            self._dtype = NcclDataType(view.dtype)
            self._device_id = view.device_id

    @property
    def ptr(self) -> int:
        """Raw device pointer address."""
        return self._ptr

    @property
    def count(self) -> int:
        """Number of elements in the buffer."""
        return self._count

    @property
    def dtype(self) -> NcclDataType:
        """NCCL data type of buffer elements."""
        return self._dtype

    @property
    def device_id(self) -> int:
        """CUDA device ID where the buffer resides."""
        return self._device_id
