"""CuTeDSL bindings for C struct types declared in
``bindings/ir/nccl_device_wrapper.h`` and ``src/include/nccl_device/``.

Codegen target — each entry is 1:1 with a C struct definition.
``_LLVMPtrType`` and ``_array_i8`` are MLIR-type adapters used as
``@cute.native_struct`` field annotations.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir


# === MLIR type adapters ===

class _LLVMPtrType:
    """Wraps ``!llvm.ptr`` for use in ``cute.ffi`` signatures and as a
    ``@cute.native_struct`` field annotation."""

    @staticmethod
    def mlir_type():
        return ir.Type.parse("!llvm.ptr")

    @staticmethod
    def __get_mlir_types__():
        return [_LLVMPtrType.mlir_type()]


def _array_i8(n: int):
    """Build a fixed-size byte array type for opaque struct storage."""

    class T:
        @staticmethod
        def mlir_type():
            return ir.Type.parse(f"!llvm.array<{n} x i8>")

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
class ncclMultimemHandle:
    """``struct ncclMultimemHandle { void* mcBasePtr; }``
    (src/include/nccl_device/impl/core__types.h)."""

    mcBasePtr: _LLVMPtrType


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
