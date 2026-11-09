# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Device-side cooperative groups for NCCL operations."""

import cutlass
from cutlass.cutlass_dsl import ir

from . import _nccl_device_ir as _ir

__all__ = ["Coop"]


class Coop:
    """A cooperative group created inside device code through the class factories.

    All group members must participate in collective operations, and each
    caller must belong to the group. The group's lifetime is limited to the
    device function that creates it.

    Range checks apply only to Python integers. CuTeDSL integer arguments
    must satisfy the same constraints.
    """

    def __init__(self) -> None:
        raise TypeError("Use Coop.thread(), warp(), lanes(), warp_span(), or cta()")

    def __extract_mlir_values__(self) -> list[ir.Value]:
        return [self._ptr]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "Coop":
        return type(self)._from_ptr(values[0])

    @classmethod
    def _from_ptr(cls, ptr: ir.Value) -> "Coop":
        obj = object.__new__(cls)
        obj._ptr = ptr
        return obj

    def _to_ptr(self) -> ir.Value:
        return self._ptr

    @classmethod
    def thread(cls) -> "Coop":
        """Create a group containing only the calling thread."""
        ptr = _ir._NcclIrCoop._allocate()
        _ir._nccl_ir_coop_init_thread(ptr)
        return cls._from_ptr(ptr)

    @classmethod
    def warp(cls) -> "Coop":
        """Create a group spanning the calling warp."""
        ptr = _ir._NcclIrCoop._allocate()
        _ir._nccl_ir_coop_init_warp(ptr)
        return cls._from_ptr(ptr)

    @classmethod
    def lanes(cls, lane_mask: int | cutlass.Integer) -> "Coop":
        """Create a group of selected lanes within the calling warp.

        ``lane_mask`` is a nonzero 32-bit mask. Collective operations require
        every selected lane to participate with the same mask, with each
        caller's own bit set.
        """
        if isinstance(lane_mask, int) and not 0 < lane_mask <= 0xFFFFFFFF:
            raise ValueError("lane_mask must be a nonzero 32-bit mask")

        ptr = _ir._NcclIrCoop._allocate()
        _ir._nccl_ir_coop_init_lanes(ptr, cutlass.Uint32(lane_mask))
        return cls._from_ptr(ptr)

    @classmethod
    def warp_span(
        cls,
        warp0: int | cutlass.Integer,
        n_warps: int | cutlass.Integer,
        id: int | cutlass.Integer,
    ) -> "Coop":
        """Create a nonempty span of consecutive warps within a 1D CTA.

        ``warp0`` must be in [0, 255], ``n_warps`` in [1, 255], and ``id``
        in [0, 14]. The span must fit within the CTA.
        Concurrent spans in the same CTA must use distinct IDs for
        overlapping collectives.
        """
        if isinstance(warp0, int) and not 0 <= warp0 <= 255:
            raise ValueError("warp0 must be in [0, 255]")
        if isinstance(n_warps, int) and not 1 <= n_warps <= 255:
            raise ValueError("n_warps must be in [1, 255]")
        if isinstance(id, int) and not 0 <= id <= 14:
            raise ValueError("id must be in [0, 14]")

        ptr = _ir._NcclIrCoop._allocate()
        _ir._nccl_ir_coop_init_warp_span(
            ptr, cutlass.Int32(warp0), cutlass.Int32(n_warps), cutlass.Int32(id)
        )
        return cls._from_ptr(ptr)

    @classmethod
    def cta(cls) -> "Coop":
        """Create a group spanning a one-dimensional CTA."""
        ptr = _ir._NcclIrCoop._allocate()
        _ir._nccl_ir_coop_init_cta(ptr)
        return cls._from_ptr(ptr)

    @property
    def thread_rank(self) -> cutlass.Int32:
        """Return the calling thread's rank within the group."""
        return cutlass.Int32(_ir._nccl_ir_coop_thread_rank(self._ptr))

    @property
    def num_threads(self) -> cutlass.Int32:
        """Return the number of threads in the group."""
        return cutlass.Int32(_ir._nccl_ir_coop_num_threads(self._ptr))

    def sync(self) -> None:
        """Synchronize all threads in the group."""
        _ir._nccl_ir_coop_sync(self._ptr)
