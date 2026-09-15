# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""CUDA device, stream, and event utilities for NCCL operations.

This module provides context managers and helper functions for working with
CUDA devices, streams, and events in NCCL operations, including device context
management and stream/event handle resolution. These helpers are used
internally to translate user specifications into the concrete forms NCCL
expects.
"""

from __future__ import annotations

from cuda.core import Device, Event, Stream

from nccl.core.typing import NcclDeviceSpec, NcclEventSpec, NcclStreamSpec


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

    Raises:
        TypeError: If ``stream`` is a boolean.
    """
    if isinstance(stream, bool):
        raise TypeError("stream must not be a bool")
    if stream is None:
        device = get_cuda_device(device)
        return device.default_stream
    if isinstance(stream, Stream):
        return stream
    if isinstance(stream, int):
        return Stream.from_handle(handle=stream)
    device = get_cuda_device(device)
    return device.create_stream(stream)


def get_stream_ptr(stream: NcclStreamSpec | None = None) -> int:
    """Resolves a CUDA stream specification to an integer ``cudaStream_t`` handle.

    Args:
        stream: A ``cuda.core.Stream``, an object implementing
            ``__cuda_stream__``, an integer ``cudaStream_t`` handle, or
            ``None`` for the default stream (handle 0).

    Returns:
        The CUDA stream handle as an integer, or 0 for the default stream.

    Raises:
        TypeError: If ``stream`` is a boolean.
    """
    if isinstance(stream, bool):
        raise TypeError("stream must not be a bool")
    if stream is None:
        return 0
    if isinstance(stream, int):
        return stream
    if isinstance(stream, Stream):
        return int(stream.handle)
    return int(stream.__cuda_stream__()[1])


def get_event_ptr(event: NcclEventSpec | None = None) -> int:
    """Resolves a CUDA event specification to an integer ``cudaEvent_t`` handle.

    Args:
        event: A ``cuda.core.Event``, a nonzero integer ``cudaEvent_t`` handle,
            or ``None`` for no event. Convert other integer-convertible handle
            objects explicitly with ``int()`` before passing them.

    Returns:
        The CUDA event handle as an integer, or 0 when no event is given.

    Raises:
        TypeError: If ``event`` is a boolean or is not a ``cuda.core.Event``
            or ``int``.
        RuntimeError: If ``event`` is a closed ``cuda.core.Event``.
        ValueError: If ``event`` is an integer with value 0.
    """
    if event is None:
        return 0
    if isinstance(event, Event):
        handle = int(event.handle)
        if handle == 0:
            raise RuntimeError("Event has been closed")
        return handle
    if isinstance(event, bool) or not isinstance(event, int):
        raise TypeError(
            "event must be a cuda.core.Event or a nonzero integer cudaEvent_t handle"
        )
    if event == 0:
        raise ValueError(
            "CUDA event has a null handle; initialize it before passing it to NCCL"
        )
    return event
