"""CuTeDSL bindings for C struct types declared in
``bindings/ir/nccl_device_wrapper.h`` and ``src/include/nccl_device/``.

Codegen target — each entry is 1:1 with a C struct definition.
``_LLVMPtrType`` and ``_array_i8`` are MLIR-type adapters used as
``@cute.native_struct`` field annotations.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import ir


# === MLIR type adapters ===

class _LLVMPtrType:
    """Wraps ``!llvm.ptr`` for use in ``@cute.extern`` signatures and as a
    ``@cute.native_struct`` field annotation."""

    @staticmethod
    def mlir_type():
        return ir.Type.parse("!llvm.ptr")

    @staticmethod
    def __get_mlir_types__():
        return [_LLVMPtrType.mlir_type()]

    @classmethod
    def isinstance(cls, value):
        """Match an ``!llvm.ptr`` ir.Value during ``@cute.extern`` dispatch.

        Callers pass pointers as bare ``!llvm.ptr`` ir.Values — a wrapper's
        ``.ptr``, or ``_to_ptr`` for an integer address — so the extern
        overload matcher needs this hook to recognize them against a
        ``_LLVMPtrType`` annotation.
        """
        return isinstance(value, ir.Value) and value.type == cls.mlir_type()


def _array_i8(n: int):
    """Build a fixed-size byte array type for opaque struct storage."""

    class T:
        @staticmethod
        def mlir_type():
            return ir.Type.parse(f"!llvm.array<{n} x i8>")

    return T


def _array_ptr(n: int):
    """Build a fixed-size opaque-pointer array type."""

    class T:
        @staticmethod
        def mlir_type():
            return ir.Type.parse(f"!llvm.array<{n} x ptr>")

    return T


# === Native structs ===

@cute.native_struct
class ncclTeam:
    """``struct ncclTeam { int nRanks, rank, stride; }``
    (src/include/nccl_device/core.h)."""

    nRanks: cutlass.Int32
    rank: cutlass.Int32
    stride: cutlass.Int32


@cute.native_struct
class ncclLsaBarrierHandle:
    """``struct ncclLsaBarrierHandle { ncclDevResourceHandle_t bufHandle; int nBarriers; }``
    (src/include/nccl_device/impl/lsa_barrier__types.h)."""

    bufHandle: cutlass.Uint32
    nBarriers: cutlass.Int32


@cute.native_struct
class ncclGinBarrierHandle:
    """``struct ncclGinBarrierHandle { ncclGinSignal_t signal0; ncclDevResourceHandle_t unused; }``
    (src/include/nccl_device/impl/gin_barrier__types.h)."""

    signal0: cutlass.Uint32
    unused: cutlass.Uint32


@cute.native_struct
class ncclCftBarrierHandle:
    """``struct ncclCftBarrierHandle { ncclDevResourceHandle_t bufHandle; int nBarriers; }``
    (src/include/nccl_device/impl/cft_barrier__types.h)."""

    bufHandle: cutlass.Uint32
    nBarriers: cutlass.Int32


@cute.native_struct
class ncclMultimemHandle:
    """``struct ncclMultimemHandle { void* mcBasePtr; }``
    (src/include/nccl_device/impl/core__types.h)."""

    mcBasePtr: _LLVMPtrType


@cute.native_struct
class ncclResourceWindow_vidmem:
    """``ncclResourceWindow_vidmem_t`` from ``impl/core__types.h``."""

    lsa_flat_base: _LLVMPtrType
    stride4g: cutlass.Uint32
    mc_offset4k: cutlass.Uint32


@cute.native_struct
class DevCommValue:
    """By-value ABI mirror of ``struct ncclDevComm``.

    Field order and types must stay synchronized with
    ``nccl_device_expanded.h`` used to generate the low-level bindings. The
    unpacked LLVM struct supplies the same natural padding as the C structure.
    """

    magic: cutlass.Uint32
    version: cutlass.Uint32

    rank: cutlass.Int32
    n_ranks: cutlass.Int32
    n_ranks_rcp32: cutlass.Uint32
    lsa_rank: cutlass.Int32
    lsa_size: cutlass.Int32
    lsa_size_rcp32: cutlass.Uint32

    window_table: _LLVMPtrType
    resource_window: _LLVMPtrType
    resource_window_inlined: ncclResourceWindow_vidmem

    hybrid_dense_gin_barrier: ncclGinBarrierHandle

    lsa_multimem: ncclMultimemHandle
    lsa_barrier: ncclLsaBarrierHandle
    rail_gin_barrier: ncclGinBarrierHandle

    gin_connection_count: cutlass.Uint8
    backend_index: cutlass.Uint8
    gin_net_device_types: _array_i8(4)
    gin_handles: _array_ptr(4)
    gin_signal_count: cutlass.Int32
    gin_counter_count: cutlass.Int32
    gin_signal_shadows: _LLVMPtrType
    gin_context_count: cutlass.Uint32
    gin_connection_stride: cutlass.Int32
    gin_context_stride: cutlass.Int32
    gin_strong_legacy_signals: cutlass.Uint8

    abort_flag: _LLVMPtrType

    hybrid_lsa_barrier: ncclLsaBarrierHandle
    hybrid_rail_gin_barrier: ncclGinBarrierHandle

    world_gin_barrier: ncclGinBarrierHandle
    gin_connection_stride_rcp32: cutlass.Uint32

    cft_rank: cutlass.Int32
    cft_size: cutlass.Int32
    cft_multimem_rank: cutlass.Int32
    cft_multimem_size: cutlass.Int32
    cft_multimem_size_rcp32: cutlass.Uint32
    uc_le_id: cutlass.Uint32
    mc_le_id: cutlass.Uint32
    cft_barrier: ncclCftBarrierHandle
    cft_multimem_barrier: ncclCftBarrierHandle


@cute.native_struct
class _ncclCoopStorage:
    space: _array_i8(16)


@cute.native_struct
class ncclCoopAny:
    """``struct ncclCoopAny`` — 16-byte aligned-to-ptr storage + vtable ptr
    (src/include/nccl_device/coop.h)."""

    storage: _ncclCoopStorage
    vtable: _LLVMPtrType


@cute.native_struct
class ncclGin_C:
    """``struct ncclGin_C`` (src/include/nccl_device/gin.h).

    The bitfield triple ``{nConnections:8, connectionId:8, _ginBackend:8}``
    is represented as a single ``Uint32`` ``flags`` field; padding fields
    match the natural alignment of the C++ layout.
    """

    comm: _LLVMPtrType
    flags: cutlass.Uint32
    contextId: cutlass.Uint32
    resourceSharingMode: cutlass.Int8
    _pad0: _array_i8(7)
    ginHandle: _LLVMPtrType
    signalShadows: _LLVMPtrType
    backendMask: cutlass.Uint32
    _pad1: _array_i8(4)
