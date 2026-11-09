# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Host argument preparation for CuTeDSL NCCL views.

Use ``to_cute`` for explicit conversion of NCCL host objects to CuTeDSL values.
The ``make_fake_*`` factories provide type-only arguments for ``cute.compile``,
allowing compilation before NCCL resources are created.

.. warning::

    Compile-only placeholders must be replaced with real arguments before
    execution. CuTeDSL does not guarantee a runtime error if a placeholder
    is passed to a compiled callable.
"""

from typing import ClassVar, cast, overload

import cutlass

import nccl.core.resources as _resources
from nccl.core.team import NCCLTeam

from ._comm import DevComm
from ._handles import GinBarrierHandle, LsaBarrierHandle, MultimemHandle
from ._team import Team
from ._window import Window

__all__ = [
    "make_fake_dev_comm",
    "make_fake_gin_barrier_handle",
    "make_fake_lsa_barrier_handle",
    "make_fake_multimem_handle",
    "make_fake_window",
    "to_cute",
]


@cutlass.register_jit_arg_adapter(
    _resources.DevCommResource, scope=cutlass.JitArgAdapterRegistry.CUDA_DIALECT_SCOPE
)
def _adapt_dev_comm_resource(resource: _resources.DevCommResource) -> DevComm:
    return DevComm(resource)


@cutlass.register_jit_arg_adapter(
    _resources.RegisteredWindowHandle, scope=cutlass.JitArgAdapterRegistry.CUDA_DIALECT_SCOPE
)
def _adapt_registered_window_handle(resource: _resources.RegisteredWindowHandle) -> Window:
    return Window(resource)


@cutlass.register_jit_arg_adapter(
    _resources.MultimemHandle, scope=cutlass.JitArgAdapterRegistry.CUDA_DIALECT_SCOPE
)
def _adapt_multimem_handle(handle: _resources.MultimemHandle) -> MultimemHandle:
    return MultimemHandle(handle)


@cutlass.register_jit_arg_adapter(
    _resources.LsaBarrierHandle, scope=cutlass.JitArgAdapterRegistry.CUDA_DIALECT_SCOPE
)
def _adapt_lsa_barrier_handle(handle: _resources.LsaBarrierHandle) -> LsaBarrierHandle:
    return LsaBarrierHandle(handle)


@cutlass.register_jit_arg_adapter(
    _resources.GinBarrierHandle, scope=cutlass.JitArgAdapterRegistry.CUDA_DIALECT_SCOPE
)
def _adapt_gin_barrier_handle(handle: _resources.GinBarrierHandle) -> GinBarrierHandle:
    return GinBarrierHandle(handle)


@overload
def to_cute(value: NCCLTeam) -> Team: ...


@overload
def to_cute(value: _resources.DevCommResource) -> DevComm: ...


@overload
def to_cute(value: _resources.RegisteredWindowHandle) -> Window: ...


@overload
def to_cute(value: _resources.MultimemHandle) -> MultimemHandle: ...


@overload
def to_cute(value: _resources.LsaBarrierHandle) -> LsaBarrierHandle: ...


@overload
def to_cute(value: _resources.GinBarrierHandle) -> GinBarrierHandle: ...


def to_cute(value: object) -> object:
    """Convert a supported NCCL host object to its CuTeDSL view.

    Call outside JIT. Supports ``NCCLTeam``,
    ``DevCommResource``, ``RegisteredWindowHandle``, ``MultimemHandle``,
    ``LsaBarrierHandle`` and ``GinBarrierHandle``. Teams copy their fields;
    resource views retain the host object without taking ownership. Keep
    resources open until device work completes. Conversion allocates no
    NCCL resources or device storage.

    Raises:
        TypeError: If the object's type is not supported.
    """
    if isinstance(value, NCCLTeam):
        return Team(n_ranks=value.n_ranks, rank=value.rank, stride=value.stride)
    if isinstance(value, _resources.DevCommResource):
        return DevComm(value)
    if isinstance(value, _resources.RegisteredWindowHandle):
        return Window(value)
    if isinstance(value, _resources.MultimemHandle):
        return MultimemHandle(value)
    if isinstance(value, _resources.LsaBarrierHandle):
        return LsaBarrierHandle(value)
    if isinstance(value, _resources.GinBarrierHandle):
        return GinBarrierHandle(value)
    raise TypeError(
        f"to_cute does not support {type(value).__name__}; "
        "supported host types: nccl.core.NCCLTeam, nccl.core.DevCommResource, "
        "nccl.core.RegisteredWindowHandle, "
        "nccl.core.MultimemHandle, nccl.core.LsaBarrierHandle, nccl.core.GinBarrierHandle"
    )


class _CompileOnlyJitArgument:
    """Type-only JIT argument that deliberately has no ``__c_pointers__``.

    Omitting it is what marks the object compile-only: CuTeDSL takes a host
    argument that yields MLIR signature types but no execution argument as a
    placeholder rather than an unsupported type.
    """

    _real_cls: ClassVar[type]

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        """Make isinstance-based annotation checks see the real view class.

        Load-bearing: CuTeDSL validates an argument against its parameter
        annotation before it looks for an adapter, so a placeholder passed to
        a ``dev_comm: DevComm`` parameter is rejected without this.
        """
        return self._real_cls

    def __get_mlir_types__(self):
        """Return the MLIR argument types of the real view."""
        return self._real_cls.__get_mlir_types__()

    def __new_from_mlir_values__(self, values):
        """Reconstruct the real view used while tracing the function body.

        Called unbound on a bare prototype, since the real implementations
        populate an instance they derive from ``type(self)``.
        """
        prototype = object.__new__(self._real_cls)
        return self._real_cls.__new_from_mlir_values__(prototype, values)

    def __repr__(self) -> str:
        return f"<compile-only fake {self._real_cls.__name__}>"


class _FakeDevComm(_CompileOnlyJitArgument):
    _real_cls = DevComm


class _FakeWindow(_CompileOnlyJitArgument):
    _real_cls = Window


class _FakeMultimemHandle(_CompileOnlyJitArgument):
    _real_cls = MultimemHandle


class _FakeLsaBarrierHandle(_CompileOnlyJitArgument):
    _real_cls = LsaBarrierHandle


class _FakeGinBarrierHandle(_CompileOnlyJitArgument):
    _real_cls = GinBarrierHandle


def make_fake_dev_comm() -> DevComm:
    """Create a compile-only ``DevComm`` placeholder for ``cute.compile``.

    Replace it with ``to_cute(dev_comm_resource)`` before execution.
    """
    return cast(DevComm, _FakeDevComm())


def make_fake_window() -> Window:
    """Create a compile-only ``Window`` placeholder for ``cute.compile``.

    Replace it with ``to_cute(window_resource)`` before execution.
    """
    return cast(Window, _FakeWindow())


def make_fake_multimem_handle() -> MultimemHandle:
    """Create a compile-only ``MultimemHandle`` placeholder for ``cute.compile``.

    Replace it with ``to_cute(host_handle)`` before execution.
    """
    return cast(MultimemHandle, _FakeMultimemHandle())


def make_fake_lsa_barrier_handle() -> LsaBarrierHandle:
    """Create a compile-only ``LsaBarrierHandle`` placeholder for ``cute.compile``.

    Replace it with ``to_cute(host_handle)`` before execution.
    """
    return cast(LsaBarrierHandle, _FakeLsaBarrierHandle())


def make_fake_gin_barrier_handle() -> GinBarrierHandle:
    """Create a compile-only ``GinBarrierHandle`` placeholder for ``cute.compile``.

    Replace it with ``to_cute(host_handle)`` before execution.
    """
    return cast(GinBarrierHandle, _FakeGinBarrierHandle())
