"""Hand-written internal utilities for the CuTeDSL bindings.

Holds the ``_alloca_*`` stack-storage helpers, ``_load_cft_le_info``, and the
``_to_ptr`` / ``_to_coop_ptr`` / ``_to_struct_ptr`` coercion helpers.
Device bitcode discovery lives in :mod:`_bindings`, next to the
``@cute.extern`` stubs that link against it.
"""

import cutlass
from cutlass.cutlass_dsl import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .types import CftLeInfo


def _device_function_entry_block():
    """Return the enclosing device function's entry block."""
    op = ir.InsertionPoint.current.block.owner
    while op is not None:
        parent = op.parent
        if parent is not None and parent.name == "gpu.module":
            if len(op.regions) == 0 or len(op.regions[0].blocks) == 0:
                break
            return op.regions[0].blocks[0]
        op = parent
    raise cutlass.DSLRuntimeError(
        "Unable to find the device function entry block for IR storage"
    )


@dsl_user_op
def _alloca_struct(struct_cls, *, alignment=None, at_entry=False, loc=None, ip=None) -> ir.Value:
    """Alloca uninitialized storage for a native struct on the kernel stack.

    Args:
        struct_cls: ``@cute.native_struct`` class supplying ``_struct_type``.
        alignment: explicit alignment in bytes; natural if ``None``.
        at_entry: allocate at function entry, outside any runtime loops.

    Returns:
        ``!llvm.ptr`` ir.Value to the storage.
    """
    if at_entry:
        with ir.InsertionPoint.at_block_begin(_device_function_entry_block()):
            # Do not forward ip: both the size constant and alloca belong at entry.
            return _alloca_struct(struct_cls, alignment=alignment, loc=loc)
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
    this only handles the arguments given as an address -- a null
    ``descriptor_ptr=0`` or an explicit descriptor pointer:

        * ``!llvm.ptr`` ir.Value -- passthrough.
        * cutlass numeric (has ``.ir_value()``) -- ``inttoptr``.
        * integer ``ir.Value`` -- ``inttoptr``.
        * Python int -- wrap in ``cutlass.Int64``, then ``inttoptr``.
        * typed ``!llvm.ptr<N>`` (e.g. shared ``ptr<3>`` from smem) --
          ``addrspacecast`` to generic ``!llvm.ptr``.

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

    # ATP, we've converted x to some int value.
    # Normally we now just do llvm.inttoptr to interpret it as a pointer...
    # But SMEM allocators are special... instead of being !llvm.ptr, it's actually !llvm.ptr<3>
    if isinstance(int_value, ir.Value) and str(int_value.type).startswith("!llvm.ptr"):
        if int_value.type == ptr_type:
            return int_value
        return llvm.addrspacecast(res=ptr_type, arg=int_value, loc=loc, ip=ip)
    return llvm.inttoptr(res=ptr_type, arg=int_value, loc=loc, ip=ip)


@dsl_user_op
def _to_coop_ptr(x, *, loc=None, ip=None):
    """Return a Coop's storage pointer, or coerce a raw address to !llvm.ptr."""
    return _to_ptr(x.ptr if hasattr(x, "ptr") else x, loc=loc, ip=ip)


@dsl_user_op
def _to_struct_ptr(x, *, loc=None, ip=None) -> ir.Value:
    """Materialize a value-mode team or handle for a const-pointer IR call."""
    value = x if isinstance(x, ir.Value) else x.__extract_mlir_values__()[0]
    with ir.InsertionPoint.at_block_begin(_device_function_entry_block()):
        # The value can be loop-dependent; hoist only its storage, not its store.
        ptr = llvm.alloca(
            res=ir.Type.parse("!llvm.ptr"),
            array_size=cutlass.Int32(1).ir_value(),
            elem_type=value.type,
            loc=loc,
        )
    llvm.store(value, ptr, loc=loc, ip=ip)
    return ptr
