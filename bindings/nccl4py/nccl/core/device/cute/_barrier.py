# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""LSA, GIN, and hybrid barrier sessions bound to their construction Coop.

Sessions reference kernel-local state. Call close() exactly once after final
use, uniformly across the original Coop. Closing writes back the epoch of any
LSA stage for slot reuse; it does not free storage or host resources. Keep the
referenced DevComm storage valid until the session is closed. Omitting close()
from an LSA or hybrid session can make later barriers on the same slot stop
synchronizing, without hanging.

Independent concurrent sessions need distinct barrier slots. Indices must
identify corresponding logical slots across ranks and be uniform within each
Coop. Passing a session through a helper or control flow preserves its state;
it does not create an independent copy.
"""

from __future__ import annotations

from enum import IntFlag

import cutlass
from cutlass.cutlass_dsl import ir

from . import _nccl_device_ir as _ir
from ._comm import DevComm
from ._coop import Coop
from ._gin import Gin, GinAllContexts
from ._handles import GinBarrierHandle, LsaBarrierHandle, MultimemHandle
from ._team import Team
from ._types import MemoryOrder

__all__ = [
    "GinBarrierSession",
    "GinFenceLevel",
    "HybridBarrierSession",
    "LsaBarrierSession",
]


class GinFenceLevel(IntFlag):
    """GIN completion guarantees established by a barrier."""

    NONE = _ir._NCCL_GIN_FENCE_LEVEL_NONE
    """Synchronize ranks without an additional GIN completion fence."""

    PUT = _ir._NCCL_GIN_FENCE_LEVEL_PUT
    """Make preceding puts from other team members to this rank visible locally."""

    GET = _ir._NCCL_GIN_FENCE_LEVEL_GET
    """Complete this rank's preceding gets into its local memory."""


class _BarrierSession:
    """Shared reconstruction of session storage and its original Coop."""

    _ptr: ir.Value
    _coop: Coop

    def __extract_mlir_values__(self) -> list[ir.Value]:
        return [self._ptr, *self._coop.__extract_mlir_values__()]

    def __new_from_mlir_values__(self, values: list[ir.Value]):
        obj = object.__new__(type(self))
        obj._ptr = values[0]
        obj._coop = self._coop.__new_from_mlir_values__(values[1:])
        return obj


class LsaBarrierSession(_BarrierSession):
    """An LSA barrier using a provisioned handle and an LSA-accessible team.

    Pair ``arrive()`` with ``wait()``, or call ``sync()`` for a complete round.
    Explicit construction requires ``mm_handle`` matching ``team`` unless
    ``multimem`` is statically false, including ``cutlass.Boolean(False)``.
    """

    def __init__(
        self,
        coop: Coop,
        dev_comm: DevComm,
        team: Team,
        handle: LsaBarrierHandle,
        *,
        index: int | cutlass.Integer,
        multimem: bool | cutlass.Boolean = False,
        mm_handle: MultimemHandle | None = None,
    ) -> None:
        multimem = cutlass.Boolean(multimem)
        if isinstance(multimem.value, int) and multimem.value == 0:
            mm_handle_ptr = cutlass.inttoptr(0, cutlass.AddressSpace.generic, cutlass.Int8)
        elif mm_handle is None:
            raise ValueError("mm_handle is required unless multimem is statically false")
        else:
            mm_handle_ptr = mm_handle._to_ptr()
        ptr = _ir._NcclIrLsaBarrierSession._allocate()
        _ir._nccl_ir_lsa_barrier_session_init(
            ptr,
            coop._to_ptr(),
            dev_comm._to_ptr(),
            team._to_ptr(),
            handle._to_ptr(),
            cutlass.Uint32(index),
            multimem,
            mm_handle_ptr,
        )
        self._ptr = ptr
        self._coop = coop

    @classmethod
    def lsa(
        cls,
        coop: Coop,
        dev_comm: DevComm,
        *,
        index: int | cutlass.Integer,
        multimem: bool | cutlass.Boolean = False,
    ) -> LsaBarrierSession:
        """Use the communicator's provisioned LSA team, barrier, and multimem handle."""
        ptr = _ir._NcclIrLsaBarrierSession._allocate()
        _ir._nccl_ir_lsa_barrier_session_init_lsa(
            ptr,
            coop._to_ptr(),
            dev_comm._to_ptr(),
            cutlass.Uint32(index),
            cutlass.Boolean(multimem),
        )
        obj = object.__new__(cls)
        obj._ptr = ptr
        obj._coop = coop
        return obj

    def arrive(self, order: MemoryOrder = MemoryOrder.RELEASE) -> None:
        """Publish this rank's arrival; pair with wait() before starting another round."""
        _ir._nccl_ir_lsa_barrier_session_arrive(
            self._ptr, self._coop._to_ptr(), _ir._CudaMemoryOrder(order)
        )

    def wait(self, order: MemoryOrder = MemoryOrder.ACQUIRE) -> None:
        """Wait for peer arrivals in the current round."""
        _ir._nccl_ir_lsa_barrier_session_wait(
            self._ptr, self._coop._to_ptr(), _ir._CudaMemoryOrder(order)
        )

    def sync(self, order: MemoryOrder = MemoryOrder.ACQ_REL) -> None:
        """Perform one arrive/wait round with the original Coop."""
        _ir._nccl_ir_lsa_barrier_session_sync(
            self._ptr, self._coop._to_ptr(), _ir._CudaMemoryOrder(order)
        )

    def close(self) -> None:
        """Persist the session epoch for slot reuse; call once across the original Coop."""
        _ir._nccl_ir_lsa_barrier_session_destroy(self._ptr)


class GinBarrierSession(_BarrierSession):
    """A GIN barrier on a compatible team with a provisioned handle.

    Gin selects one context; GinAllContexts selects every provisioned context
    on its communicator. Both use the same team and requested fence level.
    """

    def __init__(
        self,
        coop: Coop,
        gin: Gin | GinAllContexts,
        team: Team,
        handle: GinBarrierHandle,
        *,
        index: int | cutlass.Integer,
    ) -> None:
        ptr = _ir._NcclIrGinBarrierSession._allocate()
        if isinstance(gin, GinAllContexts):
            _ir._nccl_ir_gin_barrier_session_init_all_contexts(
                ptr,
                coop._to_ptr(),
                gin._dev_comm._to_ptr(),
                team._to_ptr(),
                handle._to_ptr(),
                cutlass.Uint32(index),
            )
        else:
            _ir._nccl_ir_gin_barrier_session_init(
                ptr,
                coop._to_ptr(),
                gin._to_ptr(),
                team._to_ptr(),
                handle._to_ptr(),
                cutlass.Uint32(index),
            )
        self._ptr = ptr
        self._coop = coop

    @classmethod
    def world(
        cls,
        coop: Coop,
        gin: Gin | GinAllContexts,
        *,
        index: int | cutlass.Integer,
    ) -> GinBarrierSession:
        """Use the communicator's provisioned world-team barrier."""
        ptr = _ir._NcclIrGinBarrierSession._allocate()
        if isinstance(gin, GinAllContexts):
            _ir._nccl_ir_gin_barrier_session_init_all_contexts_world(
                ptr,
                coop._to_ptr(),
                gin._dev_comm._to_ptr(),
                cutlass.Uint32(index),
            )
        else:
            _ir._nccl_ir_gin_barrier_session_init_world(
                ptr, coop._to_ptr(), gin._to_ptr(), cutlass.Uint32(index)
            )
        obj = object.__new__(cls)
        obj._ptr = ptr
        obj._coop = coop
        return obj

    @classmethod
    def rail(
        cls,
        coop: Coop,
        gin: Gin | GinAllContexts,
        *,
        index: int | cutlass.Integer,
    ) -> GinBarrierSession:
        """Use the communicator's provisioned rail-team barrier."""
        ptr = _ir._NcclIrGinBarrierSession._allocate()
        if isinstance(gin, GinAllContexts):
            _ir._nccl_ir_gin_barrier_session_init_all_contexts_rail(
                ptr,
                coop._to_ptr(),
                gin._dev_comm._to_ptr(),
                cutlass.Uint32(index),
            )
        else:
            _ir._nccl_ir_gin_barrier_session_init_rail(
                ptr, coop._to_ptr(), gin._to_ptr(), cutlass.Uint32(index)
            )
        obj = object.__new__(cls)
        obj._ptr = ptr
        obj._coop = coop
        return obj

    def sync(
        self,
        order: MemoryOrder = MemoryOrder.ACQ_REL,
        fence: GinFenceLevel = GinFenceLevel.PUT | GinFenceLevel.GET,
    ) -> None:
        """Synchronize the team and establish the selected GIN completion guarantees."""
        _ir._nccl_ir_gin_barrier_session_sync(
            self._ptr,
            self._coop._to_ptr(),
            _ir._CudaMemoryOrder(order),
            _ir._NcclGinFenceLevel(fence),
        )

    def close(self) -> None:
        """Finalize the session once across the original Coop."""
        _ir._nccl_ir_gin_barrier_session_destroy(self._ptr)


class HybridBarrierSession(_BarrierSession):
    """A barrier selecting LSA, GIN, or combined stages.

    Explicit construction requires ``inner_mm_handle`` matching ``inner_team``
    unless ``multimem`` is statically false, including ``cutlass.Boolean(False)``.

    PUT fencing with the explicit constructor requires GIN connections no
    denser than the rail team. Use ``world()`` to select the appropriate
    algorithm for a communicator-wide barrier, including denser connections.
    """

    def __init__(
        self,
        coop: Coop,
        inner_team: Team,
        outer_team: Team,
        gin: Gin,
        inner_handle: LsaBarrierHandle,
        outer_handle: GinBarrierHandle,
        *,
        index: int | cutlass.Integer,
        multimem: bool | cutlass.Boolean = False,
        inner_mm_handle: MultimemHandle | None = None,
    ) -> None:
        multimem = cutlass.Boolean(multimem)
        if isinstance(multimem.value, int) and multimem.value == 0:
            inner_mm_handle_ptr = cutlass.inttoptr(0, cutlass.AddressSpace.generic, cutlass.Int8)
        elif inner_mm_handle is None:
            raise ValueError("inner_mm_handle is required unless multimem is statically false")
        else:
            inner_mm_handle_ptr = inner_mm_handle._to_ptr()
        ptr = _ir._NcclIrBarrierSession._allocate()
        _ir._nccl_ir_barrier_session_init(
            ptr,
            coop._to_ptr(),
            inner_team._to_ptr(),
            outer_team._to_ptr(),
            gin._to_ptr(),
            inner_handle._to_ptr(),
            outer_handle._to_ptr(),
            cutlass.Uint32(index),
            multimem,
            inner_mm_handle_ptr,
        )
        self._ptr = ptr
        self._coop = coop

    @classmethod
    def world(
        cls,
        coop: Coop,
        gin: Gin,
        *,
        index: int | cutlass.Integer,
        multimem: bool | cutlass.Boolean = False,
    ) -> HybridBarrierSession:
        """Use the communicator's provisioned hybrid world-barrier resources."""
        ptr = _ir._NcclIrBarrierSession._allocate()
        _ir._nccl_ir_barrier_session_init_world(
            ptr, coop._to_ptr(), gin._to_ptr(), cutlass.Uint32(index), cutlass.Boolean(multimem)
        )
        obj = object.__new__(cls)
        obj._ptr = ptr
        obj._coop = coop
        return obj

    @classmethod
    def lsa(
        cls,
        coop: Coop,
        dev_comm: DevComm,
        *,
        index: int | cutlass.Integer,
        multimem: bool | cutlass.Boolean = False,
    ) -> HybridBarrierSession:
        """Use only the LSA stage with the communicator's hybrid LSA barrier resources."""
        ptr = _ir._NcclIrBarrierSession._allocate()
        _ir._nccl_ir_barrier_session_init_lsa(
            ptr,
            coop._to_ptr(),
            dev_comm._to_ptr(),
            cutlass.Uint32(index),
            cutlass.Boolean(multimem),
        )
        obj = object.__new__(cls)
        obj._ptr = ptr
        obj._coop = coop
        return obj

    @classmethod
    def rail(
        cls,
        coop: Coop,
        gin: Gin,
        *,
        index: int | cutlass.Integer,
    ) -> HybridBarrierSession:
        """Use only the GIN stage with the communicator's hybrid rail barrier resources."""
        ptr = _ir._NcclIrBarrierSession._allocate()
        _ir._nccl_ir_barrier_session_init_rail(
            ptr, coop._to_ptr(), gin._to_ptr(), cutlass.Uint32(index)
        )
        obj = object.__new__(cls)
        obj._ptr = ptr
        obj._coop = coop
        return obj

    def sync(
        self,
        order: MemoryOrder = MemoryOrder.ACQ_REL,
        fence: GinFenceLevel = GinFenceLevel.PUT | GinFenceLevel.GET,
    ) -> None:
        """Synchronize participating teams with the selected GIN fence level."""
        _ir._nccl_ir_barrier_session_sync(
            self._ptr,
            self._coop._to_ptr(),
            _ir._CudaMemoryOrder(order),
            _ir._NcclGinFenceLevel(fence),
        )

    def close(self) -> None:
        """Finalize the nested sessions once across the original Coop."""
        _ir._nccl_ir_barrier_session_destroy(self._ptr)
