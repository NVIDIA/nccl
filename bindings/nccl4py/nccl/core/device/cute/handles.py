# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""CuTeDSL views over NCCL resource handles.

Each view marshals its handle struct by value into a kernel and registers
a JIT arg adapter, so the host-side handles from :mod:`nccl.core.resources`
can be passed as ``@cute.jit`` arguments directly.
:meth:`_Handle.from_native_struct` wraps the copies embedded in
:class:`DevComm`, so both sources present one type per handle.
"""

import cutlass
from cutlass.cutlass_dsl import ir

from ...resources import (
    GinBarrierHandle as _GinBarrierHandleResource,
    LsaBarrierHandle as _LsaBarrierHandleResource,
    MultimemHandle as _MultimemHandleResource,
)
from ._structs import (
    ncclGinBarrierHandle,
    ncclLsaBarrierHandle,
    ncclMultimemHandle,
)


class _Handle:
    """Base for a by-value CuTeDSL view over a handle struct.

    Subclasses set ``_struct_cls`` (the ``@cute.native_struct`` type) and
    ``_view_cls`` (the host-side :mod:`nccl.core.resources` type), then
    expose fields as properties over ``value``.
    """

    _struct_cls: type
    _view_cls: type

    def __init__(self, handle):
        if not isinstance(handle, type(self)._view_cls):
            raise TypeError(
                f"{type(self).__name__} expects an "
                f"nccl.core.{type(self)._view_cls.__name__}, "
                f"got {type(handle).__name__}"
            )
        self._handle = handle
        self._value = None

    def __c_pointers__(self) -> list[int]:
        if self._handle is None:
            raise cutlass.DSLRuntimeError(
                f"{type(self).__name__}.__c_pointers__ is only available on a "
                "host-mode handle"
            )
        return [self._handle._lowpp.ptr]

    @classmethod
    def __get_mlir_types__(cls) -> list[ir.Type]:
        return cls._struct_cls.__get_mlir_types__()

    def __extract_mlir_values__(self) -> list[ir.Value]:
        return self.value.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values: list[ir.Value]):
        obj = object.__new__(type(self))
        obj._handle = None
        obj._value = type(self)._struct_cls(values[0])
        return obj

    @classmethod
    def from_native_struct(cls, struct):
        """Wrap an already-traced native struct in this view.

        Args:
            struct: value-mode ``_struct_cls`` instance, such as the one
                behind :py:attr:`DevComm.lsa_barrier`.

        Returns:
            Value-mode view over ``struct``.
        """
        if not isinstance(struct, cls._struct_cls):
            raise TypeError(
                f"{cls.__name__}.from_native_struct expects a "
                f"{cls._struct_cls.__name__}, got {type(struct).__name__}"
            )
        obj = object.__new__(cls)
        obj._handle = None
        obj._value = struct
        return obj

    @property
    def value(self):
        """Value-mode native struct backing the field reads."""
        if self._value is None:
            raise cutlass.DSLRuntimeError(
                f"{type(self).__name__} fields are only available while "
                "tracing CuTeDSL code"
            )
        return self._value


class MultimemHandle(_Handle):
    """CuTeDSL view over an :py:class:`~nccl.core.MultimemHandle`.

    Pass one to :py:meth:`Window.multimem_pointer` or
    :py:meth:`DevComm.resource_buffer_multimem_pointer` to address a team
    other than LSA; for the LSA team use :py:attr:`DevComm.lsa_multimem`.
    """

    _struct_cls = ncclMultimemHandle
    _view_cls = _MultimemHandleResource

    @property
    def mc_base_pointer(self) -> ir.Value:
        return self.value.mcBasePtr


class LsaBarrierHandle(_Handle):
    """CuTeDSL view over an :py:class:`~nccl.core.LsaBarrierHandle` obtained
    from ``DevCommResource.resource_handles``."""

    _struct_cls = ncclLsaBarrierHandle
    _view_cls = _LsaBarrierHandleResource

    @property
    def buf_handle(self) -> cutlass.Uint32:
        return self.value.bufHandle

    @property
    def n_barriers(self) -> cutlass.Int32:
        return self.value.nBarriers


class GinBarrierHandle(_Handle):
    """CuTeDSL view over a :py:class:`~nccl.core.GinBarrierHandle` obtained
    from ``DevCommResource.resource_handles``."""

    _struct_cls = ncclGinBarrierHandle
    _view_cls = _GinBarrierHandleResource

    @property
    def signal0(self) -> cutlass.Uint32:
        return self.value.signal0


@cutlass.register_jit_arg_adapter(_MultimemHandleResource)
def _adapt_multimem_handle(handle: _MultimemHandleResource) -> MultimemHandle:
    """Adapt a host multimem handle to a CuTeDSL view."""
    return MultimemHandle(handle)


@cutlass.register_jit_arg_adapter(_LsaBarrierHandleResource)
def _adapt_lsa_barrier_handle(
    handle: _LsaBarrierHandleResource,
) -> LsaBarrierHandle:
    """Adapt a host LSA barrier handle to a CuTeDSL view."""
    return LsaBarrierHandle(handle)


@cutlass.register_jit_arg_adapter(_GinBarrierHandleResource)
def _adapt_gin_barrier_handle(
    handle: _GinBarrierHandleResource,
) -> GinBarrierHandle:
    """Adapt a host GIN barrier handle to a CuTeDSL view."""
    return GinBarrierHandle(handle)


__all__ = [
    "MultimemHandle",
    "LsaBarrierHandle",
    "GinBarrierHandle",
]
