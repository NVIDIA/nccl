# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Shared public types used across the NCCL device API.

- :class:`CftLeInfo` -- logical endpoint address returned by CFT queries.
- :class:`MemoryOrder` -- memory ordering for GIN and barrier operations.
"""

from enum import IntEnum
from typing import NamedTuple

import cutlass

from . import _nccl_device_ir as _ir

__all__ = [
    "CftLeInfo",
    "MemoryOrder",
]


class CftLeInfo(NamedTuple):
    """Logical endpoint ID and byte offset returned by CFT queries."""

    le_id: cutlass.Uint32
    le_offset: cutlass.Uint64


class MemoryOrder(IntEnum):
    """CUDA memory ordering for NCCL GIN and barrier operations.

    Not interchangeable with ``cutlass.cute.nvgpu.MemoryOrder``, which is for
    CuTe operations.
    """

    RELAXED = _ir._CUDA_MEMORY_ORDER_RELAXED
    CONSUME = _ir._CUDA_MEMORY_ORDER_CONSUME
    ACQUIRE = _ir._CUDA_MEMORY_ORDER_ACQUIRE
    RELEASE = _ir._CUDA_MEMORY_ORDER_RELEASE
    ACQ_REL = _ir._CUDA_MEMORY_ORDER_ACQ_REL
    SEQ_CST = _ir._CUDA_MEMORY_ORDER_SEQ_CST
