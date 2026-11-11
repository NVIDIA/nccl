"""CuTeDSL bindings for C struct types declared in
``bindings/ir/nccl_device_ir.h`` and ``src/include/nccl_device/``.

Codegen target -- each entry is 1:1 with a C struct definition.
``_LLVMPtrType`` and ``_array_i64`` are MLIR-type adapters used as
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

        Callers pass pointers as bare ``!llvm.ptr`` ir.Values -- a wrapper's
        ``.ptr``, or ``_to_ptr`` for an integer address -- so the extern
        overload matcher needs this hook to recognize them against a
        ``_LLVMPtrType`` annotation.
        """
        return isinstance(value, ir.Value) and value.type == cls.mlir_type()


def _array_i64(n: int):
    """Build a fixed-size, naturally 8-byte-aligned opaque storage array."""

    class T:
        @staticmethod
        def mlir_type():
            return ir.Type.parse(f"!llvm.array<{n} x i64>")

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
class ncclIrDevComm:
    """Opaque DevComm storage from the matching NCCL device IR build."""

    storage: _array_i64(31)


@cute.native_struct
class ncclIrCoop:
    """Opaque 24-byte, 8-byte-aligned coop storage from the IR wrapper header."""

    storage: _array_i64(3)


@cute.native_struct
class ncclIrGin:
    """Opaque 40-byte, 8-byte-aligned GIN storage from the IR wrapper header."""

    storage: _array_i64(5)


@cute.native_struct
class ncclIrLsaBarrierSession:
    """Opaque LSA barrier session storage from the matching NCCL device IR build."""

    storage: _array_i64(10)


@cute.native_struct
class ncclIrGinBarrierSession:
    """Opaque GIN barrier session storage from the matching NCCL device IR build."""

    storage: _array_i64(12)


@cute.native_struct
class ncclIrBarrierSession:
    """Opaque hybrid barrier session storage from the matching NCCL device IR build."""

    storage: _array_i64(46)
