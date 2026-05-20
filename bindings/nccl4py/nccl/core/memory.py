# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL-backed memory resource management.

This module provides NcclMemoryResource, a MemoryResource implementation that
uses NCCL's memory allocation functions to allocate device memory optimized
for NCCL operations. Memory resources are thread-local and device-specific.
"""

from __future__ import annotations

import threading

from cuda.core import Buffer, Device, MemoryResource, Stream
from cuda.core.typing import DevicePointerType

from nccl.bindings import nccl as _nccl_bindings

from nccl.core.cuda import CudaDeviceContext


__all__ = ["NcclMemoryResource", "get_memory_resource"]

_mr_instances = threading.local()


class NcclMemoryResource(MemoryResource):
    """NCCL-backed device memory resource for Buffer allocations.

    Uses NCCL's memory allocation functions to allocate device memory that is
    optimized for NCCL operations. The resource is tied to a specific CUDA
    device and ensures the proper device context during allocation.
    """

    def __init__(self, device_id: int) -> None:
        """Initializes an NCCL memory resource for a specific device.

        Args:
            device_id: CUDA device ID to allocate memory on.
        """
        self.device = Device(device_id)

    def allocate(self, size: int, *, stream: Stream | None = None) -> Buffer:
        """Allocates device memory using NCCL.

        Args:
            size: Number of bytes to allocate.
            stream: CUDA stream for the allocation. Currently unused; reserved
                for API compatibility with MemoryResource.

        Returns:
            Buffer wrapping the allocated memory.
        """
        with CudaDeviceContext(self.device):
            ptr = _nccl_bindings.mem_alloc(size)  # mem_alloc raises NCCLError if ptr is 0
            buf = Buffer.from_handle(ptr=ptr, size=size, mr=self)
            return buf

    def deallocate(self, ptr: DevicePointerType, size: int, *, stream: Stream | None = None) -> None:
        """Deallocates device memory using NCCL.

        Args:
            ptr: Device pointer to deallocate.
            size: Size of the allocation, kept for API compatibility.
            stream: CUDA stream for the deallocation. Currently unused;
                reserved for API compatibility with MemoryResource.
        """
        with CudaDeviceContext(self.device):
            _nccl_bindings.mem_free(int(ptr))

    @property
    def device_id(self) -> int:
        """CUDA device ID this resource is associated with."""
        return self.device.device_id

    @property
    def is_device_accessible(self) -> bool:
        """Whether this memory is accessible from device code (always True)."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """Whether this memory is accessible from host code (always False)."""
        return False


def get_memory_resource(device_id: int) -> NcclMemoryResource:
    """Returns a thread-local NcclMemoryResource for the given device.

    Provides a cached, thread-local memory resource for each device. Multiple
    calls with the same device on the same thread return the same resource
    instance, ensuring efficient memory management.

    Args:
        device_id: CUDA device ID to get a memory resource for.

    Returns:
        Thread-local NcclMemoryResource for the device.
    """
    if not hasattr(_mr_instances, "memory_resources"):
        _mr_instances.memory_resources = {}

    if device_id not in _mr_instances.memory_resources:
        _mr_instances.memory_resources[device_id] = NcclMemoryResource(device_id)

    return _mr_instances.memory_resources[device_id]
