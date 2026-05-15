# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""CUDA device and stream utilities for NCCL operations.

This module provides context managers and helper functions for working with
CUDA devices and streams in NCCL operations, including device context management
and stream resolution. These helpers are used internally to translate user
device/stream specifications into the concrete forms NCCL expects.
"""

from __future__ import annotations

from cuda.core import Device, Stream

from nccl.core.typing import NcclDeviceSpec, NcclStreamSpec


class CudaDeviceContext:
    """Context manager that temporarily switches the current CUDA device.

    On enter, sets device as the current device if it differs from the
    previously active one; on exit, restores the original device. No-op when
    device already matches the current device.
    """

    def __init__(self, device: Device) -> None:
        self._device = device
        self._old_device = Device()

    def __enter__(self):
        if self._device.device_id != self._old_device.device_id:
            self._device.set_current()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._device.device_id != self._old_device.device_id:
            self._old_device.set_current()
        return False  # Re-raise any exception


def get_cuda_device(device: NcclDeviceSpec | None = None) -> Device:
    """Resolves a device specification to a cuda.core.Device.

    Args:
        device: A Device instance, an integer device ID, or ``None`` to use
            the current device.

    Returns:
        Resolved cuda.core.Device.
    """
    if device is None:
        return Device()
    elif isinstance(device, Device):
        return device
    else:
        return Device(device)


def get_device_id(device: NcclDeviceSpec | None = None) -> int:
    """Resolves a device specification to its CUDA device ID.

    Args:
        device: A Device instance, an integer device ID, or ``None`` to use
            the current device.

    Returns:
        CUDA device ID.
    """
    if isinstance(device, int):
        return device
    else:
        return get_cuda_device(device).device_id


def get_cuda_stream(
    stream: NcclStreamSpec | None = None, device: NcclDeviceSpec | None = None
) -> Stream:
    """Resolves a stream specification to a cuda.core.Stream.

    Args:
        stream: A Stream instance, an integer stream handle, an object
            implementing __cuda_stream__, or ``None`` to use the device's
            default stream.
        device: Device used to resolve the default stream or wrap a foreign
            stream object. Only consulted when needed. Defaults to the
            current device.

    Returns:
        Resolved cuda.core.Stream.
    """
    if stream is None:
        device = get_cuda_device(device)
        return device.default_stream
    elif isinstance(stream, Stream):
        return stream
    elif isinstance(stream, int):
        return Stream.from_handle(handle=stream)
    else:
        device = get_cuda_device(device)
        return device.create_stream(stream)


def get_stream_ptr(stream: NcclStreamSpec | None = None) -> int:
    """Resolves a stream specification to its raw stream handle (as int).

    Args:
        stream: A Stream instance, an integer stream handle, an object
            implementing __cuda_stream__, or ``None`` to use the default
            stream (handle 0).

    Returns:
        Raw CUDA stream handle as int. Returns 0 for the default stream.
    """
    if stream is None:
        return 0
    elif isinstance(stream, int):
        return stream
    elif isinstance(stream, Stream):
        return int(stream.handle)
    else:
        return int(stream.__cuda_stream__()[1])
