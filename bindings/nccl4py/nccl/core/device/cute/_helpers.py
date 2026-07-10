"""Hand-written internal utilities for the CuTeDSL bindings.

Holds the bitcode resolver, the module-level ``BitCode`` cache, the
``_ffi`` factory that injects ``source=_BC`` into every prototype, and
the ``_to_ptr`` / ``_to_coop_value`` coercion helpers used by
:mod:`_bindings`.
"""

import os
from pathlib import Path

import cutlass
from cutlass.cute import BitCode
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op

from cuda.pathfinder import find_bitcode_lib

from ._structs import ncclCoopAny


# === Device bitcode discovery ===

def device_bitcode_path() -> str:
    """Locate ``libnccl_device.bc`` for FFI linking.

    Resolution order:

        1. ``$NCCL_HOME/lib/libnccl_device.bc`` (in-repo source builds).
        2. ``cuda.pathfinder.find_bitcode_lib("nccl_device")`` for the
           installed ``nvidia-nccl-cuXX`` wheel (requires
           ``cuda-pathfinder >= 1.5.4``).

    Returns:
        Absolute path to ``libnccl_device.bc``.

    Raises:
        cuda.pathfinder.BitcodeLibNotFoundError: bitcode not found.
    """
    env = os.environ.get("NCCL_HOME")
    if env:
        p = Path(env) / "lib" / "libnccl_device.bc"
        if p.is_file():
            return str(p)
    return find_bitcode_lib("nccl_device")


_BC = BitCode(device_bitcode_path())


def _ffi(**kw):
    """``cute.ffi`` with ``source=_BC`` injected on every prototype.

    The DSL pipeline merges the bitcode into each module's
    ``link-libraries`` attribute automatically, so users never pass
    ``options="--link-libraries=..."`` to ``cute.compile``.

    Args:
        **kw: forwarded to ``cute.ffi``.

    Returns:
        ``cute.ffi`` prototype object.
    """
    return cute.ffi(source=_BC, **kw)


# === Coercion helpers ===

@dsl_user_op
def _to_ptr(x, *, loc=None, ip=None):
    """Coerce ``x`` to an ``!llvm.ptr`` ir.Value.

    Args:
        x: accepted forms —

            * ``!llvm.ptr`` ir.Value — passthrough.
            * ``@cute.native_struct`` with a ``!llvm.ptr`` ``.ptr`` field
              (``DevComm``, ``Window``, ``Gin``, ``Coop``) — extract.
            * cutlass numeric (has ``.ir_value()``) — ``inttoptr``.
            * integer ``ir.Value`` — ``inttoptr``.
            * Python int — wrap in ``cutlass.Int64``, then ``inttoptr``.

    Returns:
        ``!llvm.ptr`` ir.Value.
    """
    ptr_type = ir.Type.parse("!llvm.ptr")
    if isinstance(x, ir.Value) and x.type == ptr_type:
        return x
    # @cute.native_struct wrapper around a ptr field.
    if hasattr(x, "ptr"):
        inner = x.ptr
        if isinstance(inner, ir.Value) and inner.type == ptr_type:
            return inner
    if hasattr(x, "ir_value"):
        int_value = x.ir_value()
    elif isinstance(x, ir.Value):
        int_value = x
    else:
        int_value = cutlass.Int64(x).ir_value()
    return llvm.inttoptr(res=ptr_type, arg=int_value, loc=loc, ip=ip)


@dsl_user_op
def _to_coop_value(x, *, loc=None, ip=None):
    """Coerce ``x`` to an ``ncclCoopAny`` value for by-value FFI calls.

    :class:`Coop` only carries a pointer to alloca'd storage (cheap
    property access); by-value FFIs need the full struct loaded here.

    Args:
        x: accepted forms —

            * ``ncclCoopAny`` value — passthrough.
            * ``@cute.native_struct`` pointer-wrapper around alloca'd
              ``ncclCoopAny`` storage (e.g. :class:`Coop`) — load it.

    Returns:
        ``ncclCoopAny`` struct value.
    """
    if isinstance(x, ncclCoopAny):
        return x
    if hasattr(x, "ptr"):
        return ncclCoopAny(
            llvm.load(res=ncclCoopAny._struct_type, addr=x.ptr, loc=loc, ip=ip)
        )
    return x
