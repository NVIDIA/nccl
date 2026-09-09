"""Compile-only representative arguments for CuTeDSL NCCL views.

Each factory builds a placeholder that carries a ``@cute.jit`` argument's
type without owning an NCCL resource, so an application can ``cute.compile``
before it creates a communicator or registers memory and invoke the compiled
callable with real resources later. The nccl4py analogue of
``cutlass.cute.runtime.make_fake_stream``.

.. warning::

    A placeholder is only valid as a ``cute.compile`` argument. Passing one
    to the compiled callable contributes no execution pointer, which shifts
    every following argument and launches with garbage instead of raising;
    only CuTeDSL's debugging mode reports it.
"""

from typing import ClassVar, cast

from .comm import DevComm
from .handles import GinBarrierHandle, LsaBarrierHandle, MultimemHandle
from .window import Window


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
    """Create a type-only ``DevComm`` placeholder for ``cute.compile``.

    Valid only as a ``cute.compile`` argument; invoke the compiled callable
    with a real ``DevCommResource``, or a :class:`DevComm` wrapping one.
    """
    return cast(DevComm, _FakeDevComm())


def make_fake_window() -> Window:
    """Create a type-only ``Window`` placeholder for ``cute.compile``.

    Valid only as a ``cute.compile`` argument; invoke the compiled callable
    with a real ``RegisteredWindowHandle``, or a :class:`Window` wrapping one.
    """
    return cast(Window, _FakeWindow())


def make_fake_multimem_handle() -> MultimemHandle:
    """Create a type-only ``MultimemHandle`` placeholder for ``cute.compile``.

    Valid only as a ``cute.compile`` argument; invoke the compiled callable
    with the real handle.
    """
    return cast(MultimemHandle, _FakeMultimemHandle())


def make_fake_lsa_barrier_handle() -> LsaBarrierHandle:
    """Create a type-only ``LsaBarrierHandle`` placeholder for ``cute.compile``.

    Valid only as a ``cute.compile`` argument; invoke the compiled callable
    with the real handle.
    """
    return cast(LsaBarrierHandle, _FakeLsaBarrierHandle())


def make_fake_gin_barrier_handle() -> GinBarrierHandle:
    """Create a type-only ``GinBarrierHandle`` placeholder for ``cute.compile``.

    Valid only as a ``cute.compile`` argument; invoke the compiled callable
    with the real handle.
    """
    return cast(GinBarrierHandle, _FakeGinBarrierHandle())


__all__ = [
    "make_fake_dev_comm",
    "make_fake_gin_barrier_handle",
    "make_fake_lsa_barrier_handle",
    "make_fake_multimem_handle",
    "make_fake_window",
]
