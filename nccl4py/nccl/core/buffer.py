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
Buffer specification and memory allocation for NCCL operations.

This module provides utilities for allocating NCCL-optimized device memory and
resolving various buffer specifications (arrays, tensors, tuples) into the raw
pointers, counts, and dtypes required by NCCL operations.
"""

from __future__ import annotations
import math

from cuda.core.experimental import Buffer
from cuda.core.experimental.utils import StridedMemoryView, args_viewable_as_strided_memory

from nccl.core.cuda import get_device_id, get_stream_ptr
from nccl.core.memory import get_memory_resource
from nccl.core.typing import (
    NcclBufferSpec,
    NcclDataType,
    NcclDeviceSpec,
    NcclInvalid,
    NcclStreamSpec,
)


__all__ = ["mem_alloc", "mem_free", "NcclBuffer"]


def mem_alloc(size: int, device: NcclDeviceSpec | None = None) -> Buffer:
    """
    Allocates GPU buffer memory using NCCL's memory allocator.

    The actual allocated size may be larger than requested due to buffer granularity
    requirements from NCCL optimizations.

    Args:
        - size (int): Number of bytes to allocate.
        - device (NcclDeviceSpec, optional): Target CUDA device. Defaults to current device.

    Returns:
        ``Buffer``: A CUDA buffer object backed by NCCL-managed memory.

    Notes:
        - The returned buffer size may exceed the requested size due to alignment requirements.
        - Buffer is allocated on the specified device, if provided; current device is restored after allocation.

    Memory Lifecycle:
        Memory can be explicitly freed using ``mem_free(buf)`` or automatically freed when
        the Buffer object is garbage collected.

    See Also:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclmemalloc
    """
    device_id = get_device_id(device)

    mr = get_memory_resource(device_id)
    return mr.allocate(size)


def mem_free(buf: Buffer) -> None:
    """
    Frees memory allocated by ``mem_alloc()``.

    Args:
        - buf (Buffer): The buffer to free.

    Notes:
        Explicit deallocation is optional. Memory is automatically freed when the Buffer
        object is garbage collected.

    See Also:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclmemfree
    """
    buf.close()


@args_viewable_as_strided_memory((0,))
def _resolve_buffer(buffer, stream_ptr: int | None) -> StridedMemoryView:
    return buffer.view(stream_ptr)


class NcclBuffer:
    """
    Resolves user-provided buffer specifications to raw pointer, count, dtype, and device.

    This class handles various buffer input formats and extracts the necessary metadata
    for NCCL operations.

    Buffer specifications can be:
        - An object supporting DLPack or CUDA Array Interface
        - A ``Buffer`` object
        - A tuple: ``(buffer, dtype)``

    Stream Handling:
        When stream is None, the special sentinel value -1 is used internally.
        This sentinel value (-1) is required by cuda.core's StridedMemoryView.view()
        to indicate that no stream synchronization should be performed between the
        producer stream (where the buffer was created) and the consumer stream
        (where NCCL operations will use it). This avoids implicit synchronization
        overhead when the user is managing stream dependencies manually.

        When a stream is provided, its handle (typically 0 for default stream or
        a valid stream pointer) is used for proper synchronization.

    Notes:
        - Element count is automatically derived from buffer size and dtype. Use slicing to control element count.

    Attributes:
        ptr (int): Raw device pointer address.
        count (int): Number of elements in the buffer.
        dtype (NcclDataType): Data type of buffer elements.
        device_id (int): CUDA device ID where buffer resides.
    """

    def __init__(self, buffer: NcclBufferSpec, stream: NcclStreamSpec | None = None) -> None:
        """
        Initializes an NcclBuffer from a buffer specification.

        Args:
            - buffer (NcclBufferSpec): Buffer specification to resolve.
            - stream (NcclStreamSpec, optional): CUDA stream for synchronization. Defaults to None.

        Raises:
            - ``NcclInvalid``: If the buffer specification is invalid or malformed.
        """
        self.stream_ptr = -1 if stream is None else get_stream_ptr(stream)

        resolved = None

        try:
            from nccl.core.interop.torch import resolve_tensor

            resolved = resolve_tensor(buffer)
        except (ImportError, ModuleNotFoundError, NcclInvalid):
            pass

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
        """
        Raw device pointer address.

        Returns:
            ``int``: Device memory pointer.
        """
        return self._ptr

    @property
    def count(self) -> int:
        """
        Number of elements in the buffer.

        Returns:
            ``int``: Element count.
        """
        return self._count

    @property
    def dtype(self) -> NcclDataType:
        """
        Data type of buffer elements.

        Returns:
            ``NcclDataType``: NCCL data type.
        """
        return self._dtype

    @property
    def device_id(self) -> int:
        """
        CUDA device ID where buffer resides.

        Returns:
            ``int``: Device ID.
        """
        return self._device_id
