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
CUDA device and stream utilities for NCCL operations.

This module provides context managers and helper functions for working with
CUDA devices and streams in NCCL operations, including device context management
and stream resolution.
"""

from __future__ import annotations

from cuda.core.experimental import Device, Stream

from nccl.core.typing import NcclDeviceSpec, NcclStreamSpec


class CudaDeviceContext:
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
    if device is None:
        return Device()
    elif isinstance(device, Device):
        return device
    else:
        return Device(device)


def get_device_id(device: NcclDeviceSpec | None = None) -> int:
    if isinstance(device, int):
        return device
    else:
        return get_cuda_device(device).device_id


def get_cuda_stream(
    stream: NcclStreamSpec | None = None, device: NcclDeviceSpec | None = None
) -> Stream:
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
    if stream is None:
        return 0
    elif isinstance(stream, int):
        return stream
    elif isinstance(stream, Stream):
        return int(stream.handle)
    else:
        return int(stream.__cuda_stream__()[1])
