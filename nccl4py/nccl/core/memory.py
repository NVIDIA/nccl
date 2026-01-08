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
NCCL-backed memory resource management.

This module provides NcclMemoryResource, a MemoryResource implementation that uses
NCCL's memory allocation functions to allocate device memory optimized for NCCL operations.
Memory resources are thread-local and device-specific.
"""

from __future__ import annotations

import threading

from cuda.core.experimental import Buffer, Device, MemoryResource, Stream

from nccl import bindings as _nccl_bindings

from nccl.core.cuda import CudaDeviceContext

try:
    from cuda.core.experimental._memory import DevicePointerT
except ImportError:
    try:
        from cuda.core import DevicePointerT
    except ImportError:
        try:
            from cuda.core._memory._buffer import DevicePointerT
        except ImportError:
            DevicePointerT = int


__all__ = ["NcclMemoryResource", "get_memory_resource"]

_mr_instances = threading.local()


class NcclMemoryResource(MemoryResource):
    """
    NCCL-backed device memory resource for Buffer allocations.

    This memory resource uses NCCL's memory allocation functions to allocate
    device memory that is optimized for NCCL operations. The resource is tied
    to a specific CUDA device and ensures proper device context during allocation.
    """

    def __init__(self, device_id: int) -> None:
        """
        Initializes NCCL memory resource for a specific device.

        Args:
            - device_id (int): CUDA device ID to allocate memory on.
        """
        self.device = Device(device_id)

    def allocate(self, size: int, stream: Stream | None = None) -> Buffer:
        """
        Allocates device memory using NCCL.

        Args:
            - size (int): Number of bytes to allocate.
            - stream (Stream, optional): CUDA stream for allocation (currently unused). Defaults to None.

        Returns:
            ``Buffer``: Buffer object wrapping the allocated memory.

        Raises:
            - ``NCCLError``: If allocation fails.
        """
        with CudaDeviceContext(self.device):
            ptr = _nccl_bindings.mem_alloc(size)  # mem_alloc raises NCCLError if ptr is 0
            buf = Buffer.from_handle(ptr=ptr, size=size, mr=self)
            return buf

    def deallocate(self, ptr: DevicePointerT, size: int, stream: Stream | None = None) -> None:
        """
        Deallocates device memory using NCCL.

        Args:
            - ptr (DevicePointerT): Device pointer to deallocate.
            - size (int): Size of the allocation (for compatibility).
            - stream (Stream, optional): CUDA stream for deallocation (currently unused). Defaults to None.
        """
        with CudaDeviceContext(self.device):
            _nccl_bindings.mem_free(int(ptr))

    @property
    def device_id(self) -> int:
        """
        CUDA device ID this resource is associated with.

        Returns:
            ``int``: Device ID.
        """
        return self.device.device_id

    @property
    def is_device_accessible(self) -> bool:
        """
        Whether this memory is accessible from device code.

        Returns:
            ``bool``: Always True for NCCL-allocated memory.
        """
        return True

    @property
    def is_host_accessible(self) -> bool:
        """
        Whether this memory is accessible from host code.

        Returns:
            ``bool``: Always False for device memory.
        """
        return False


def get_memory_resource(device_id: int) -> NcclMemoryResource:
    """
    Gets a thread-local ``NcclMemoryResource`` for the given device.

    This function provides a cached, thread-local memory resource for each device.
    Multiple calls with the same device will return the same resource instance,
    ensuring efficient memory management.

    Args:
        - device_id (int): CUDA device ID to get memory resource for.

    Returns:
        ``NcclMemoryResource``: Thread-local NcclMemoryResource for the device.
    """
    if not hasattr(_mr_instances, "memory_resources"):
        _mr_instances.memory_resources = {}

    if device_id not in _mr_instances.memory_resources:
        _mr_instances.memory_resources[device_id] = NcclMemoryResource(device_id)

    return _mr_instances.memory_resources[device_id]
