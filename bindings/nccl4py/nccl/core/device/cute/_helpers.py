"""Hand-written internal utilities for the CuTeDSL bindings.

Holds the ``_alloca_*`` stack-storage helpers, ``_load_cft_le_info``, and the
``_to_ptr`` / ``_to_coop_value`` / ``_to_value`` coercion helpers.
Device bitcode discovery lives in :mod:`_bindings`, next to the
``@cute.extern`` stubs that link against it.
"""

import cutlass
from cutlass.cutlass_dsl import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from ._structs import ncclCoopAny
from .types import CftLeInfo


@dsl_user_op
def _alloca_struct(struct_cls, *, alignment=None, loc=None, ip=None) -> ir.Value:
    """Alloca uninitialized storage for a native struct on the kernel stack.

    Args:
        struct_cls: ``@cute.native_struct`` class supplying ``_struct_type``.
        alignment: explicit alignment in bytes; natural if ``None``.

    Returns:
        ``!llvm.ptr`` ir.Value to the storage.
    """
    return llvm.alloca(
        res=ir.Type.parse("!llvm.ptr"),
        array_size=cutlass.Int32(1).ir_value(),
        elem_type=struct_cls._struct_type,
        alignment=alignment,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _alloca_cft_le_info(*, loc=None, ip=None) -> tuple[ir.Value, ir.Value]:
    """Alloca the ``(ncclCftLeId, size_t)`` out-params of ``ncclGet*LeInfo``.

    Returns:
        ``(le_id_ptr, le_offset_ptr)``, both ``!llvm.ptr`` ir.Values.
    """
    ptr_type = ir.Type.parse("!llvm.ptr")
    one = cutlass.Int32(1).ir_value()
    return (
        llvm.alloca(res=ptr_type, array_size=one,
                    elem_type=ir.IntegerType.get_signless(32), loc=loc, ip=ip),
        llvm.alloca(res=ptr_type, array_size=one,
                    elem_type=ir.IntegerType.get_signless(64), loc=loc, ip=ip),
    )


@dsl_user_op
def _load_cft_le_info(le_id_ptr, le_offset_ptr, *, loc=None, ip=None) -> CftLeInfo:
    """Load the out-params filled by ``ncclGet*LeInfo``.

    Returns:
        Logical endpoint information containing ``cutlass.Uint32`` and
        ``cutlass.Uint64`` values.
    """
    return CftLeInfo(
        le_id=cutlass.Uint32(
            llvm.load(ir.IntegerType.get_signless(32), le_id_ptr, loc=loc, ip=ip)
        ),
        le_offset=cutlass.Uint64(
            llvm.load(ir.IntegerType.get_signless(64), le_offset_ptr, loc=loc, ip=ip)
        ),
    )


# === Coercion helpers ===

@dsl_user_op
def _to_ptr(x, *, loc=None, ip=None):
    """Coerce an address ``x`` to an ``!llvm.ptr`` ir.Value.

    Callers pass materialized pointers (``.ptr``) to the bindings directly;
    this only handles the arguments given as an address — a null
    ``descriptor_ptr=0`` or an explicit descriptor pointer:

        * ``!llvm.ptr`` ir.Value — passthrough.
        * cutlass numeric (has ``.ir_value()``) — ``inttoptr``.
        * integer ``ir.Value`` — ``inttoptr``.
        * Python int — wrap in ``cutlass.Int64``, then ``inttoptr``.

    Returns:
        ``!llvm.ptr`` ir.Value.
    """
    ptr_type = ir.Type.parse("!llvm.ptr")
    if isinstance(x, ir.Value) and x.type == ptr_type:
        return x
    if hasattr(x, "ir_value"):
        int_value = x.ir_value()
    elif isinstance(x, ir.Value):
        int_value = x
    else:
        int_value = cutlass.Int64(x).ir_value()
    return llvm.inttoptr(res=ptr_type, arg=int_value, loc=loc, ip=ip)


@dsl_user_op
def _to_coop_value(x, *, loc=None, ip=None):
    """Coerce a :class:`Coop` to a bare ``ncclCoopAny`` struct ir.Value.

    :class:`Coop` only carries a pointer to alloca'd storage (cheap
    property access); by-value externs need the full struct loaded here.
    The result is a bare struct ir.Value (not a wrapper) so the
    ``@cute.extern`` matcher recognizes it against an ``ncclCoopAny``
    annotation.

    Args:
        x: accepted forms —

            * ``ncclCoopAny`` struct ir.Value — passthrough.
            * ``@cute.native_struct`` pointer-wrapper around alloca'd
              ``ncclCoopAny`` storage (e.g. :class:`Coop`) — load it.

    Returns:
        ``ncclCoopAny`` struct ir.Value.
    """
    if isinstance(x, ir.Value):
        return x
    if hasattr(x, "ptr"):
        return llvm.load(ncclCoopAny._struct_type, x.ptr, loc=loc, ip=ip)
    return _to_value(x)


def _to_value(x):
    """Unwrap a by-value ``@cute.native_struct`` argument to its bare ir.Value.

    ``@cute.extern`` matches a native struct only as a bare struct ir.Value
    (a value-mode wrapper is rejected), so teams and barrier handles passed
    by value — which ``cute.ffi`` accepted as wrappers — are unwrapped here.

    Args:
        x: a struct ir.Value (passthrough) or a value-mode native-struct
            wrapper exposing ``__extract_mlir_values__``.

    Returns:
        The underlying struct ir.Value.
    """
    if isinstance(x, ir.Value):
        return x
    if hasattr(x, "__extract_mlir_values__"):
        return x.__extract_mlir_values__()[0]
    return x
