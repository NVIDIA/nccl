# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""CuTeDSL views of opaque NCCL resource handles.

Convert host handles with ``runtime.to_cute`` or the view constructor.
Keep the owning device communicator open until device work completes.
"""

from typing import ClassVar, Generic, TypeVar

import cutlass
from cutlass.cutlass_dsl import ir

import nccl.core.resources as _resources

from . import _nccl_device_ir as _ir

__all__ = [
    "GinBarrierHandle",
    "LsaBarrierHandle",
    "MultimemHandle",
]

_HostHandle = TypeVar("_HostHandle")


class _Handle(Generic[_HostHandle]):
    """A host handle reference, reconstructed as a native struct SSA value in JIT."""

    _native_type: ClassVar[type]
    _host_type: ClassVar[type]

    def __init__(self, handle: _HostHandle) -> None:
        if not isinstance(handle, self._host_type):
            raise TypeError(
                f"{type(self).__name__} expects nccl.core.{self._host_type.__name__}, "
                f"got {type(handle).__name__}"
            )
        self._host_handle = handle
        self._value = None

    def __c_pointers__(self) -> list[int]:
        if self._host_handle is None:
            raise cutlass.DSLRuntimeError("Only host-backed handles can be passed from Python")
        return [self._host_handle._lowpp.ptr]

    @classmethod
    def __get_mlir_types__(cls) -> list[ir.Type]:
        return cls._native_type.__get_mlir_types__()

    def __extract_mlir_values__(self) -> list[ir.Value] | None:
        if self._value is None:
            return None
        return self._value.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values: list[ir.Value]):
        obj = object.__new__(type(self))
        obj._host_handle = None
        obj._value = type(self)._native_type(values[0])
        return obj

    def _to_ptr(self) -> ir.Value:
        if self._value is None:
            raise cutlass.DSLRuntimeError("Handle device storage is only available inside JIT")
        ptr = type(self)._native_type._allocate(at_entry=True)
        self._value._store(ptr)
        return ptr


class MultimemHandle(_Handle[_resources.MultimemHandle]):
    """CuTeDSL handle for a mapping from ``DevCommResource.multimem_handle``.

    Pass one to :py:meth:`Window.multimem_pointer` or
    :py:meth:`DevComm.resource_buffer_multimem_pointer` to address a team
    other than LSA; for the LSA team use :py:meth:`Window.lsa_multimem_pointer`.
    """

    _native_type = _ir._NcclMultimemHandle
    _host_type = _resources.MultimemHandle


class LsaBarrierHandle(_Handle[_resources.LsaBarrierHandle]):
    """CuTeDSL LSA barrier handle from ``DevCommResource.resource_handles``."""

    _native_type = _ir._NcclLsaBarrierHandle
    _host_type = _resources.LsaBarrierHandle


class GinBarrierHandle(_Handle[_resources.GinBarrierHandle]):
    """CuTeDSL GIN barrier handle from ``DevCommResource.resource_handles``."""

    _native_type = _ir._NcclGinBarrierHandle
    _host_type = _resources.GinBarrierHandle
