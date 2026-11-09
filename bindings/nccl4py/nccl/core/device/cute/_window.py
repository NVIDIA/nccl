# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Opaque CuTeDSL view over NCCL registered windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cutlass
from cutlass import cute
from cutlass.cutlass_dsl import ir

from nccl.core.resources import RegisteredWindowHandle

from . import _nccl_device_ir as _ir
from ._handles import MultimemHandle
from ._team import Team
from ._types import CftLeInfo

if TYPE_CHECKING:
    from ._comm import DevComm

__all__ = ["SymmetricPointer", "Window"]


class Window:
    """CuTeDSL view over an NCCL :c:type:`ncclWindow_t` handle.

    Construct from a host :py:class:`~nccl.core.RegisteredWindowHandle` with
    ``runtime.to_cute(resource)`` or ``Window(resource)``.

    Address-query offsets are in bytes. Callers ensure dtype alignment and
    that accesses remain within the registered range. The registration and
    underlying allocation must remain valid until GPU work completes.

    Args:
        resource: Registered window resource.
    """

    def __init__(self, resource: RegisteredWindowHandle) -> None:
        if not isinstance(resource, RegisteredWindowHandle):
            raise TypeError(
                f"Window expects an nccl.core.RegisteredWindowHandle, got {type(resource).__name__}"
            )
        self._resource = resource
        self._value = None

    def __c_pointers__(self) -> list[int]:
        if self._resource is None:
            raise cutlass.DSLRuntimeError(
                "Window.__c_pointers__ is only available on a host-mode Window"
            )
        if not self._resource.is_valid:
            raise RuntimeError("RegisteredWindowHandle does not reference an active NCCL window")
        return [self._resource._window.handle_ptr]

    @staticmethod
    def __get_mlir_types__() -> list[ir.Type]:
        return [_ir._NcclWindowT.mlir_type()]

    def __extract_mlir_values__(self) -> list[ir.Value] | None:
        if self._value is None:
            return None
        return [self._value]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> Window:
        obj = object.__new__(type(self))
        obj._resource = None
        obj._value = values[0]
        return obj

    def _to_ptr(self) -> ir.Value:
        if self._value is None:
            raise cutlass.DSLRuntimeError("Window device handle is only available inside JIT")
        return self._value

    def local_pointer(
        self,
        dtype: type[cutlass.Numeric],
        byte_offset: int | cutlass.Integer = 0,
    ) -> cute.Pointer:
        """Return a typed pointer into the local registered buffer."""
        ptr = _ir._nccl_ir_get_local_pointer(self._to_ptr(), _ir._SizeT(byte_offset))
        return cute.make_ptr(dtype, ptr)

    def lsa_pointer(
        self,
        dtype: type[cutlass.Numeric],
        peer: int | cutlass.Integer,
        byte_offset: int | cutlass.Integer = 0,
    ) -> cute.Pointer:
        """Return a typed pointer on ``peer``, expressed as an LSA-team rank."""
        ptr = _ir._nccl_ir_get_lsa_pointer(
            self._to_ptr(), _ir._SizeT(byte_offset), cutlass.Int32(peer)
        )
        return cute.make_ptr(dtype, ptr)

    def peer_pointer(
        self,
        dtype: type[cutlass.Numeric],
        peer: int | cutlass.Integer,
        byte_offset: int | cutlass.Integer = 0,
        *,
        team: Team | None = None,
    ) -> cute.Pointer:
        """Return a typed pointer on a world rank or a rank within ``team``.

        The peer must belong to the window's accessible LSA domain. An
        unsupported peer can produce an invalid, non-null address.
        """
        if team is None:
            ptr = _ir._nccl_ir_get_peer_pointer(
                self._to_ptr(), _ir._SizeT(byte_offset), cutlass.Int32(peer)
            )
        else:
            ptr = _ir._nccl_ir_get_peer_pointer_team(
                self._to_ptr(),
                _ir._SizeT(byte_offset),
                team._to_ptr(),
                cutlass.Int32(peer),
            )
        return cute.make_ptr(dtype, ptr)

    def multimem_pointer(
        self,
        dtype: type[cutlass.Numeric],
        mm_handle: MultimemHandle,
        byte_offset: int | cutlass.Integer = 0,
    ) -> cute.Pointer:
        """Return a typed multicast pointer using the supplied mapping.

        The mapping must cover this window. Multimem requires supported
        topology and memory; it is not a fallback for peer addressing.
        """
        ptr = _ir._nccl_ir_get_multimem_pointer(
            self._to_ptr(), _ir._SizeT(byte_offset), mm_handle._to_ptr()
        )
        return cute.make_ptr(dtype, ptr)

    def lsa_multimem_pointer(
        self,
        dtype: type[cutlass.Numeric],
        dev_comm: DevComm,
        byte_offset: int | cutlass.Integer = 0,
    ) -> cute.Pointer:
        """Return a typed multicast pointer using the communicator's LSA mapping."""
        ptr = _ir._nccl_ir_get_lsa_multimem_pointer(
            self._to_ptr(), _ir._SizeT(byte_offset), dev_comm._to_ptr(at_entry=True)
        )
        return cute.make_ptr(dtype, ptr)

    def cft_le_info(
        self,
        byte_offset: int | cutlass.Integer,
        peer_cft: int | cutlass.Integer,
        cft_team: Team,
        dev_comm: DevComm,
    ) -> CftLeInfo:
        """Resolve a byte offset and rank within ``cft_team`` to a logical endpoint."""
        le_id = cutlass.Array(_ir._NcclCftLeId, 1)
        le_offset = cutlass.Array(_ir._SizeT, 1)
        _ir._nccl_ir_get_cft_le_info(
            self._to_ptr(),
            _ir._SizeT(byte_offset),
            cutlass.Int32(peer_cft),
            cft_team._to_ptr(),
            dev_comm._to_ptr(at_entry=True),
            le_id.data_ptr(),
            le_offset.data_ptr(),
        )
        return CftLeInfo(le_id[0], le_offset[0])

    def peer_le_info(
        self,
        byte_offset: int | cutlass.Integer,
        peer: int | cutlass.Integer,
        dev_comm: DevComm,
    ) -> CftLeInfo:
        """Resolve a byte offset and world rank to a logical endpoint.

        ``peer`` must belong to the flat CFT team; otherwise the endpoint is
        invalid. The device API does not check membership.
        """
        le_id = cutlass.Array(_ir._NcclCftLeId, 1)
        le_offset = cutlass.Array(_ir._SizeT, 1)
        _ir._nccl_ir_get_peer_le_info(
            self._to_ptr(),
            _ir._SizeT(byte_offset),
            cutlass.Int32(peer),
            dev_comm._to_ptr(at_entry=True),
            le_id.data_ptr(),
            le_offset.data_ptr(),
        )
        return CftLeInfo(le_id[0], le_offset[0])

    def multimem_le_info(
        self,
        byte_offset: int | cutlass.Integer,
        dev_comm: DevComm,
    ) -> CftLeInfo:
        """Resolve a byte offset to its multicast logical endpoint."""
        le_id = cutlass.Array(_ir._NcclCftLeId, 1)
        le_offset = cutlass.Array(_ir._SizeT, 1)
        _ir._nccl_ir_get_multimem_le_info(
            self._to_ptr(),
            _ir._SizeT(byte_offset),
            dev_comm._to_ptr(at_entry=True),
            le_id.data_ptr(),
            le_offset.data_ptr(),
        )
        return CftLeInfo(le_id[0], le_offset[0])

    def tensor(
        self,
        dtype: type[cutlass.Numeric],
        layout: cute.Layout | cute.ComposedLayout,
        byte_offset: int | cutlass.Integer = 0,
    ) -> cute.Tensor:
        """Create a local, non-owning tensor view without copying data.

        The caller ensures the layout's accesses are aligned and within the
        registered range. An arbitrary layout is not a contiguous transfer range.
        """
        return cute.make_tensor(self.local_pointer(dtype, byte_offset), layout)


@dataclass(frozen=True, eq=False)
class SymmetricPointer:
    """A registered window, static element type, and byte offset.

    Construct in device code. Construction uses bytes; addition and subtraction
    use element counts and return new values. Accessors select the team or peer
    without changing this descriptor or owning the underlying registration.
    Element arithmetic on sub-byte data requires a packed dtype with whole-byte storage.
    The dtype must remain unchanged when a value crosses dynamic control flow.
    """

    window: Window
    dtype: cutlass.Constexpr[type[cutlass.Numeric]]
    byte_offset: int | cutlass.Integer = 0

    # CuTeDSL's dataclass path mistakes Numeric dtype classes for dynamic values.
    def __extract_mlir_values__(self) -> list[ir.Value]:
        return [self.window._to_ptr(), cutlass.Uint64(self.byte_offset).ir_value()]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> SymmetricPointer:
        return type(self)(
            self.window.__new_from_mlir_values__(values[:1]), self.dtype, cutlass.Uint64(values[1])
        )

    @property
    def element_size(self) -> int:
        """The number of bytes occupied by one dtype element."""
        if self.dtype is not cutlass.Boolean and self.dtype.width % 8:
            raise ValueError("Sub-byte elements have no whole-byte size; use a packed dtype")
        return self.dtype.bytes

    def __add__(self, count: int | cutlass.Integer) -> SymmetricPointer:
        """Advance by ``count`` elements."""
        # Widen signed counts before applying size_t wraparound in IR.
        delta = cutlass.Int64(count).ir_value() * self.element_size
        byte_offset = cutlass.Uint64(self.byte_offset).ir_value() + delta
        return type(self)(self.window, self.dtype, cutlass.Uint64(byte_offset))

    def __sub__(self, count: int | cutlass.Integer) -> SymmetricPointer:
        """Move back by ``count`` elements."""
        delta = cutlass.Int64(count).ir_value() * self.element_size
        byte_offset = cutlass.Uint64(self.byte_offset).ir_value() - delta
        return type(self)(self.window, self.dtype, cutlass.Uint64(byte_offset))

    def local_pointer(self) -> cute.Pointer:
        """Return the local typed data pointer."""
        return self.window.local_pointer(self.dtype, self.byte_offset)

    def lsa_pointer(self, peer: int | cutlass.Integer) -> cute.Pointer:
        """Return the typed data pointer on an LSA-team rank."""
        return self.window.lsa_pointer(self.dtype, peer, self.byte_offset)

    def peer_pointer(
        self,
        peer: int | cutlass.Integer,
        *,
        team: Team | None = None,
    ) -> cute.Pointer:
        """Return the typed data pointer on a world rank or a rank within ``team``."""
        return self.window.peer_pointer(self.dtype, peer, self.byte_offset, team=team)

    def multimem_pointer(self, mm_handle: MultimemHandle) -> cute.Pointer:
        """Return the typed multicast pointer using the supplied mapping."""
        return self.window.multimem_pointer(self.dtype, mm_handle, self.byte_offset)

    def lsa_multimem_pointer(self, dev_comm: DevComm) -> cute.Pointer:
        """Return the typed multicast pointer using the communicator's LSA mapping."""
        return self.window.lsa_multimem_pointer(self.dtype, dev_comm, self.byte_offset)

    def tensor(self, layout: cute.Layout | cute.ComposedLayout) -> cute.Tensor:
        """Create a local tensor view with the supplied layout."""
        return self.window.tensor(self.dtype, layout, self.byte_offset)
