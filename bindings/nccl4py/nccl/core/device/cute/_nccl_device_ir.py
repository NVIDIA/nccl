# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Low-level CuTeDSL declarations for ``bindings/ir/nccl_device_ir.h``.

Single codegen target for NCCL IR types and function bindings. Complete C
structs use ``@cute.native_struct`` with explicit ABI size and alignment.
Semantic wrappers decide whether to carry a value or a storage pointer and
manage opaque objects' initialization and lifetime through the IR API.

Python identifiers follow Python naming conventions; private declarations
use a leading underscore, and C typedef suffix ``_t`` becomes ``T``.
Original C names remain in declaration documentation and symbol bindings.
Pointer aliases document C pointee types without enforcing them;
const-qualified and unqualified pointers share the same alias.

Private Python function names use the snake_case spelling of their C symbols.
Fixed symbols use ``@cute.extern``; ReduceCopy symbols are selected by dtype
through overloaded declarations. Callers handle argument conversion and
result wrapping.
"""

import os
from pathlib import Path
from typing import TypeAlias, TypeVar

import cutlass
from cuda.pathfinder import find_bitcode_lib
from cutlass import cute
from cutlass._mlir.dialects import cuda as cuda_dialect
from cutlass._mlir.dialects import llvm
from cutlass.cute import BitCode
from cutlass.cutlass_dsl import dsl_user_op, ir

try:
    from cutlass._mlir.dialects import lir as lir_dialect
except ImportError:
    lir_dialect = None


# === Scalar aliases ===

_NcclDevResourceHandle: TypeAlias = cutlass.Uint32
_NcclDevResourceHandleT: TypeAlias = _NcclDevResourceHandle  # noqa: PYI043
_NcclGinSignalT: TypeAlias = cutlass.Uint32  # noqa: PYI043
_NcclGinCounterT: TypeAlias = cutlass.Uint32  # noqa: PYI043
_NcclCftLeId: TypeAlias = cutlass.Uint32
_NcclCftLeIdT: TypeAlias = _NcclCftLeId  # noqa: PYI043, PYI047

# size_t in the 64-bit CUDA device ABI, independent of the Python host ABI.
_SizeT: TypeAlias = cutlass.Uint64  # noqa: PYI043


# === Enum aliases and constants ===

# Underlying types for the gnu++17 CUDA device ABI used by bindings/ir.
# Unfixed, nonnegative C++ enums use unsigned int in this build; explicitly
# sized enums retain their declared type. Plain Python integers represent
# enumerators; callers construct the corresponding alias for FFI arguments.
# Public IntEnum/IntFlag classes live in the semantic wrapper modules.

# ncclCftTeamMode_t (nccl_device/core.h).
_NcclCftTeamModeT: TypeAlias = cutlass.Uint32  # noqa: PYI043
_NCCL_CFT_TEAM_FLAT = 0
_NCCL_CFT_TEAM_HIER_MULTIMEM = 1
_NCCL_CFT_TEAM_HIER_LSA = 2

# ncclGinResourceSharingMode : uint8_t (nccl_device/gin/gin_device_common.h).
_NcclGinResourceSharingMode: TypeAlias = cutlass.Uint8
_NCCL_GIN_RESOURCE_SHARING_GPU = 0
_NCCL_GIN_RESOURCE_SHARING_CTA = 1
_NCCL_GIN_RESOURCE_SHARING_THREAD = 2

# ncclGinFenceLevel : uint32_t (nccl_device/gin_barrier.h).
_NcclGinFenceLevel: TypeAlias = cutlass.Uint32
_NCCL_GIN_FENCE_LEVEL_NONE = 0
_NCCL_GIN_FENCE_LEVEL_PUT = 1 << 0
_NCCL_GIN_FENCE_LEVEL_GET = 1 << 1
_NCCL_GIN_FENCE_LEVEL_RELAXED = _NCCL_GIN_FENCE_LEVEL_NONE

# ncclGinSignalType (nccl_device/gin/gin_device_common.h).
_NcclGinSignalType: TypeAlias = cutlass.Uint32
_NCCL_GIN_SIGNAL_TYPE_NONE = 0
_NCCL_GIN_SIGNAL_TYPE_VA = 1
_NCCL_GIN_SIGNAL_TYPE_INDEXED = 2

# ncclGinSignalOp_t (nccl_device/gin/gin_device_host_common.h).
_NcclGinSignalOpT: TypeAlias = cutlass.Uint32  # noqa: PYI043
_NCCL_GIN_SIGNAL_INC = 0
_NCCL_GIN_SIGNAL_ADD = 1

# ncclGinSegmentType and its typedef (nccl_device_ir.h).
_NcclGinSegmentType: TypeAlias = cutlass.Uint32
_NcclGinSegmentTypeT: TypeAlias = _NcclGinSegmentType  # noqa: PYI043
_NCCL_GIN_SEGMENT_TYPE_DEVICE = 0
_NCCL_GIN_SEGMENT_TYPE_MIXED = 1
_NCCL_GIN_SEGMENT_TYPE_HOST_NUMA = 2

# ncclGinOptFlags (nccl_device/gin/gin_device_common.h).
# IR optFlags parameters themselves are uint32_t, not the enum type.
_NcclGinOptFlags: TypeAlias = cutlass.Uint32  # noqa: PYI047
_NCCL_GIN_OPT_FLAGS_DEFAULT = 0
_NCCL_GIN_OPT_FLAGS_MAY_SKIP_CREDIT_CHECK = 1 << 0
_NCCL_GIN_OPT_FLAGS_AGGREGATE_REQUESTS = 1 << 1

# ncclResult_t, returned by timeout operations (nccl.h).
_NcclResultT: TypeAlias = cutlass.Uint32  # noqa: PYI043
_NCCL_SUCCESS = 0
_NCCL_UNHANDLED_CUDA_ERROR = 1
_NCCL_SYSTEM_ERROR = 2
_NCCL_INTERNAL_ERROR = 3
_NCCL_INVALID_ARGUMENT = 4
_NCCL_INVALID_USAGE = 5
_NCCL_REMOTE_ERROR = 6
_NCCL_IN_PROGRESS = 7
_NCCL_TIMEOUT = 8
_NCCL_NUM_RESULTS = 9

# cuda::memory_order (libcu++ <cuda/std/atomic>, C++17).
_CudaMemoryOrder: TypeAlias = cutlass.Uint32
_CUDA_MEMORY_ORDER_RELAXED = 0
_CUDA_MEMORY_ORDER_CONSUME = 1
_CUDA_MEMORY_ORDER_ACQUIRE = 2
_CUDA_MEMORY_ORDER_RELEASE = 3
_CUDA_MEMORY_ORDER_ACQ_REL = 4
_CUDA_MEMORY_ORDER_SEQ_CST = 5

# cuda::thread_scope (libcu++ <cuda/atomic>).
_CudaThreadScope: TypeAlias = cutlass.Uint32
_CUDA_THREAD_SCOPE_SYSTEM = 0
_CUDA_THREAD_SCOPE_DEVICE = 1
_CUDA_THREAD_SCOPE_BLOCK = 2
_CUDA_THREAD_SCOPE_THREAD = 10


# === MLIR type adapters ===


class _LLVMPtrType:
    """Wraps ``!llvm.ptr`` for use in ``@cute.extern`` signatures and as a
    ``@cute.native_struct`` field annotation."""

    @staticmethod
    def mlir_type() -> ir.Type:
        return ir.Type.parse("!llvm.ptr")

    @staticmethod
    def __get_mlir_types__() -> list[ir.Type]:
        return [_LLVMPtrType.mlir_type()]

    @classmethod
    def isinstance(cls, value: object) -> bool:
        """Match bare ``!llvm.ptr`` values during ``@cute.extern`` dispatch."""
        return isinstance(value, ir.Value) and value.type == cls.mlir_type()


# === IR pointer aliases ===

# ``Ptr`` denotes both T* and T const*, lowering to a generic LLVM pointer.
# Pointee constness remains in the C declarations; it is not encoded here.
_NcclTeamPtr: TypeAlias = _LLVMPtrType
_NcclIrDevCommPtr: TypeAlias = _LLVMPtrType
_NcclLsaBarrierHandlePtr: TypeAlias = _LLVMPtrType
_NcclGinBarrierHandlePtr: TypeAlias = _LLVMPtrType
_NcclMultimemHandlePtr: TypeAlias = _LLVMPtrType
_NcclIrCoopPtr: TypeAlias = _LLVMPtrType
_NcclIrGinPtr: TypeAlias = _LLVMPtrType
_NcclIrLsaBarrierSessionPtr: TypeAlias = _LLVMPtrType
_NcclIrGinBarrierSessionPtr: TypeAlias = _LLVMPtrType
_NcclIrBarrierSessionPtr: TypeAlias = _LLVMPtrType
_NcclGinRequestTPtr: TypeAlias = _LLVMPtrType
_NcclGinDescriptorSmemPtr: TypeAlias = _LLVMPtrType

# ncclWindow_t is already a pointer typedef. Its Ptr form is an output slot
# containing that handle, as used by ncclIrGetResourceBuffer.
_NcclWindowT: TypeAlias = _LLVMPtrType  # noqa: PYI043
_NcclWindowTPtr: TypeAlias = _LLVMPtrType  # noqa: PYI047

# Scalar output parameters.
_NcclCftLeIdPtr: TypeAlias = _LLVMPtrType
_SizeTPtr: TypeAlias = _LLVMPtrType
_Uint64TPtr: TypeAlias = _LLVMPtrType


def _array_type(element_type: type[cutlass.Numeric], n: int):
    """Describe a fixed-size numeric array, resolving its type in the active context."""

    class _ArrayType:
        @staticmethod
        def mlir_type():
            return ir.Type.parse(f"!llvm.array<{n} x {element_type.mlir_type}>")

    return _ArrayType


# === Native struct memory support ===

_FUNCTION_OP_NAMES = (
    cuda_dialect.KernelOp.OPERATION_NAME,
    cuda_dialect.FuncOp.OPERATION_NAME,
    "gpu.func",
    "func.func",
    "llvm.func",
)
if lir_dialect is not None:
    _FUNCTION_OP_NAMES += (lir_dialect.FuncOp.OPERATION_NAME,)


class _NativeStructMemoryMixin:
    """Memory operations for CuTeDSL native struct values.

    Allocation creates uninitialized storage without constructing an SSA value.
    Loading takes an SSA snapshot; storing copies an existing value to memory.
    No operation associates a cached pointer with a value. Semantic wrappers
    manage opaque objects through their storage pointers and IR operations.
    Each concrete struct defines its ABI alignment in ``_memory_alignment``;
    pointers passed to ``_load`` and ``_store`` must satisfy that alignment.
    """

    # CuTeDSL treats all annotations except Constexpr as ABI fields,
    # including inherited ClassVar annotations. Keep this metadata unannotated.
    _nbytes = None

    @staticmethod
    def _function_entry_block(ip=None) -> ir.Block:
        """Find the nearest enclosing function from the requested position."""
        insertion_point = ir.InsertionPoint.current if ip is None else ip
        op = insertion_point.block.owner.operation
        while op is not None:
            if op.name in _FUNCTION_OP_NAMES:
                if len(op.regions) > 0 and len(op.regions[0].blocks) > 0:
                    return op.regions[0].blocks[0]
                break
            op = op.parent
        raise cutlass.DSLRuntimeError(
            "Unable to find the enclosing function entry block for native struct storage"
        )

    @classmethod
    @dsl_user_op
    def _allocate(
        cls,
        *,
        memory_space: cutlass.AddressSpace = cutlass.AddressSpace.rmem,
        at_entry: bool = False,
        loc=None,
        ip=None,
    ) -> ir.Value:
        """Allocate fresh storage at the given position and return ``!llvm.ptr``.

        ``rmem`` (the default) uses thread-private stack storage; ``smem`` uses
        CTA shared storage. No value is constructed, initialized, or copied.
        Storage uses the type's ABI alignment. Callers manage initialization,
        sharing, and synchronization through the IR API.

        ``at_entry=True`` places ``rmem`` storage at the beginning of the
        nearest enclosing function selected by ``ip`` (or the current insertion
        point). CUDA kernels and device functions, optional LIR functions, and
        standard GPU, Func, and LLVM functions are supported. It does not change
        the caller's insertion point or move initialization. Combining it with
        another memory space raises ``ValueError``.

        Use ``at_entry=True`` for per-call temporaries, such as a by-value
        argument or an out-param whose lifetime ends within the call, so a loop
        reuses one slot instead of allocating on every iteration. Keep object
        storage (Coop, Gin, sessions, requests), and storage an object keeps a
        pointer to, such as the DevComm behind a Gin or session, where it is
        created: a pipelined loop can hold several live instances from one call site.
        """
        if at_entry:
            if memory_space != cutlass.AddressSpace.rmem:
                raise ValueError("at_entry=True supports only AddressSpace.rmem")
            # Both the count constant and alloca use this IP; the caller's
            # ambient IP remains unchanged for subsequent initialization.
            ip = ir.InsertionPoint.at_block_begin(cls._function_entry_block(ip))
        if memory_space == cutlass.AddressSpace.rmem:
            return llvm.alloca(
                res=_LLVMPtrType.mlir_type(),
                array_size=cutlass.Int32(1).ir_value(loc=loc, ip=ip),
                elem_type=cls.mlir_type,
                alignment=cls._memory_alignment,
                loc=loc,
                ip=ip,
            )
        if memory_space == cutlass.AddressSpace.smem:
            ptr = cute.arch.alloc_smem(
                cutlass.Int8,
                cls._nbytes,
                alignment=cls._memory_alignment,
                loc=loc,
                ip=ip,
            )
            # Generic addressing matches the IR ABI; storage remains shared.
            return cutlass.Pointer(ptr, loc=loc, ip=ip).tospace(
                cutlass.AddressSpace.generic, loc=loc, ip=ip
            )
        raise ValueError(
            "native struct allocation supports only AddressSpace.rmem and AddressSpace.smem"
        )

    @classmethod
    @dsl_user_op
    def _load(cls, ptr, *, loc=None, ip=None):
        """Load an SSA snapshot from a pointer satisfying this type's ABI alignment."""
        return cutlass.Pointer(ptr, dtype=cls, loc=loc, ip=ip).load(
            alignment=cls._memory_alignment,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _store(self, ptr, *, loc=None, ip=None) -> None:
        """Copy this value to storage satisfying this type's ABI alignment."""
        cutlass.Pointer(ptr, dtype=type(self), loc=loc, ip=ip).store(
            cutlass.extract_mlir_values(self)[0],
            alignment=type(self)._memory_alignment,
            loc=loc,
            ip=ip,
        )


# === Native structs ===


@cute.native_struct
class _NcclTeam(_NativeStructMemoryMixin):
    """``struct ncclTeam { int nRanks, rank, stride; }``
    (src/include/nccl_device/core.h)."""

    _nbytes = 12
    _memory_alignment = 4

    n_ranks: cutlass.Int32
    rank: cutlass.Int32
    stride: cutlass.Int32


@cute.native_struct
class _NcclLsaBarrierHandle(_NativeStructMemoryMixin):
    """``struct ncclLsaBarrierHandle { ncclDevResourceHandle_t bufHandle; int nBarriers; }``
    (src/include/nccl_device/impl/lsa_barrier__types.h)."""

    _nbytes = 8
    _memory_alignment = 4

    buf_handle: _NcclDevResourceHandleT
    n_barriers: cutlass.Int32


@cute.native_struct
class _NcclGinBarrierHandle(_NativeStructMemoryMixin):
    """``struct ncclGinBarrierHandle { ncclGinSignal_t signal0; ncclDevResourceHandle_t unused; }``
    (src/include/nccl_device/impl/gin_barrier__types.h)."""

    _nbytes = 8
    _memory_alignment = 4

    signal0: _NcclGinSignalT
    unused: _NcclDevResourceHandleT


@cute.native_struct
class _NcclMultimemHandle(_NativeStructMemoryMixin):
    """``struct ncclMultimemHandle { void* mcBasePtr; }``
    (src/include/nccl_device/impl/core__types.h)."""

    _nbytes = 8
    _memory_alignment = 8

    mc_base_ptr: _LLVMPtrType


@cute.native_struct
class _NcclIrDevComm(_NativeStructMemoryMixin):
    """Opaque ``ncclIrDevComm`` storage from the matching NCCL device IR build."""

    _nbytes = 248
    _memory_alignment = 8

    storage: _array_type(cutlass.Uint64, 31)


@cute.native_struct
class _NcclIrCoop(_NativeStructMemoryMixin):
    """Opaque ``ncclIrCoop`` storage: 24 bytes, aligned to 8 bytes."""

    _nbytes = 24
    _memory_alignment = 8

    storage: _array_type(cutlass.Uint64, 3)


@cute.native_struct
class _NcclIrGin(_NativeStructMemoryMixin):
    """Opaque ``ncclIrGin`` storage: 40 bytes, aligned to 8 bytes."""

    _nbytes = 40
    _memory_alignment = 8

    storage: _array_type(cutlass.Uint64, 5)


@cute.native_struct
class _NcclIrLsaBarrierSession(_NativeStructMemoryMixin):
    """Opaque ``ncclIrLsaBarrierSession`` storage from the matching device IR build."""

    _nbytes = 80
    _memory_alignment = 8

    storage: _array_type(cutlass.Uint64, 10)


@cute.native_struct
class _NcclIrGinBarrierSession(_NativeStructMemoryMixin):
    """Opaque ``ncclIrGinBarrierSession`` storage from the matching device IR build."""

    _nbytes = 96
    _memory_alignment = 8

    storage: _array_type(cutlass.Uint64, 12)


@cute.native_struct
class _NcclIrBarrierSession(_NativeStructMemoryMixin):
    """Opaque ``ncclIrBarrierSession`` storage from the matching device IR build."""

    _nbytes = 368
    _memory_alignment = 8

    storage: _array_type(cutlass.Uint64, 46)


@cute.native_struct
class _NcclGinRequestT(_NativeStructMemoryMixin):
    """Opaque ``ncclGinRequest_t`` storage declared in ``nccl_device/core.h``."""

    _nbytes = 16
    _memory_alignment = 8

    opaque: _array_type(cutlass.Uint64, 2)


@cute.native_struct
class _NcclGinDescriptorSmem(_NativeStructMemoryMixin):
    """Opaque ``ncclGinDescriptorSmem`` storage in CTA shared memory.

    Layout from ``nccl_device/gin/gin_device_common.h``. Semantic wrappers
    carry the allocated pointer and coordinate its use by the cooperative group.
    They must pass ``memory_space=cutlass.AddressSpace.smem`` when allocating.
    The C field's ``alignas(64)`` is retained as allocation metadata; the LLVM
    array field alone does not express this over-alignment.
    """

    _nbytes = 128
    _memory_alignment = 64

    space: _array_type(cutlass.Int8, 128)


# === Device bitcode discovery ===


def _device_bitcode_path() -> str:
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


# Module-level BitCode cache passed as ``@cute.extern(source=_BC)`` on every
# stub; the DSL merges it into each module's ``link-libraries`` attribute.
_BC = BitCode(_device_bitcode_path())


# === Coop API ===


@cute.extern(name="ncclIrCoopInitThread", source=_BC)
def _nccl_ir_coop_init_thread(coop: _NcclIrCoopPtr) -> None: ...


@cute.extern(name="ncclIrCoopInitWarp", source=_BC)
def _nccl_ir_coop_init_warp(coop: _NcclIrCoopPtr) -> None: ...


@cute.extern(name="ncclIrCoopInitLanes", source=_BC)
def _nccl_ir_coop_init_lanes(coop: _NcclIrCoopPtr, lane_mask: cutlass.Uint32) -> None: ...


@cute.extern(name="ncclIrCoopInitWarpSpan", source=_BC)
def _nccl_ir_coop_init_warp_span(
    coop: _NcclIrCoopPtr,
    warp0: cutlass.Int32,
    n_warps: cutlass.Int32,
    id: cutlass.Int32,
) -> None: ...


@cute.extern(name="ncclIrCoopInitCta", source=_BC)
def _nccl_ir_coop_init_cta(coop: _NcclIrCoopPtr) -> None: ...


@cute.extern(name="ncclIrCoopThreadRank", source=_BC)
def _nccl_ir_coop_thread_rank(coop: _NcclIrCoopPtr) -> cutlass.Int32: ...


@cute.extern(name="ncclIrCoopSize", source=_BC)
def _nccl_ir_coop_size(coop: _NcclIrCoopPtr) -> cutlass.Int32: ...


@cute.extern(name="ncclIrCoopNumThreads", source=_BC)
def _nccl_ir_coop_num_threads(coop: _NcclIrCoopPtr) -> cutlass.Int32: ...


@cute.extern(name="ncclIrCoopSync", source=_BC)
def _nccl_ir_coop_sync(coop: _NcclIrCoopPtr) -> None: ...


# === Core API (teams + window pointers) ===


@cute.extern(name="ncclIrDevCommGinContextCount", source=_BC)
def _nccl_ir_dev_comm_gin_context_count(
    dev_comm: _NcclIrDevCommPtr,
) -> cutlass.Uint32: ...


@cute.extern(name="ncclIrDevCommRank", source=_BC)
def _nccl_ir_dev_comm_rank(dev_comm: _NcclIrDevCommPtr) -> cutlass.Int32: ...


@cute.extern(name="ncclIrDevCommNRanks", source=_BC)
def _nccl_ir_dev_comm_n_ranks(dev_comm: _NcclIrDevCommPtr) -> cutlass.Int32: ...


@cute.extern(name="ncclIrDevCommLsaRank", source=_BC)
def _nccl_ir_dev_comm_lsa_rank(dev_comm: _NcclIrDevCommPtr) -> cutlass.Int32: ...


@cute.extern(name="ncclIrDevCommLsaSize", source=_BC)
def _nccl_ir_dev_comm_lsa_size(dev_comm: _NcclIrDevCommPtr) -> cutlass.Int32: ...


@cute.extern(name="ncclIrTeamWorld", source=_BC)
def _nccl_ir_team_world(
    dev_comm: _NcclIrDevCommPtr,
    out_team: _NcclTeamPtr,
) -> None: ...


@cute.extern(name="ncclIrTeamLsa", source=_BC)
def _nccl_ir_team_lsa(
    dev_comm: _NcclIrDevCommPtr,
    out_team: _NcclTeamPtr,
) -> None: ...


@cute.extern(name="ncclIrTeamRail", source=_BC)
def _nccl_ir_team_rail(
    dev_comm: _NcclIrDevCommPtr,
    out_team: _NcclTeamPtr,
) -> None: ...


@cute.extern(name="ncclIrTeamCft", source=_BC)
def _nccl_ir_team_cft(
    dev_comm: _NcclIrDevCommPtr,
    out_team: _NcclTeamPtr,
    mode: _NcclCftTeamModeT,
) -> None: ...


@cute.extern(name="ncclIrTeamCftMultimem", source=_BC)
def _nccl_ir_team_cft_multimem(
    dev_comm: _NcclIrDevCommPtr,
    out_team: _NcclTeamPtr,
) -> None: ...


@cute.extern(name="ncclIrTeamRankToWorld", source=_BC)
def _nccl_ir_team_rank_to_world(
    dev_comm: _NcclIrDevCommPtr,
    team: _NcclTeamPtr,
    rank: cutlass.Int32,
) -> cutlass.Int32: ...


@cute.extern(name="ncclIrTeamRankToLsa", source=_BC)
def _nccl_ir_team_rank_to_lsa(
    dev_comm: _NcclIrDevCommPtr,
    team: _NcclTeamPtr,
    rank: cutlass.Int32,
) -> cutlass.Int32: ...


@cute.extern(name="ncclIrGetLocalPointer", source=_BC)
def _nccl_ir_get_local_pointer(
    window: _NcclWindowT,
    offset: _SizeT,
) -> _LLVMPtrType: ...


@cute.extern(name="ncclIrGetLsaPointer", source=_BC)
def _nccl_ir_get_lsa_pointer(
    window: _NcclWindowT,
    offset: _SizeT,
    peer: cutlass.Int32,
) -> _LLVMPtrType: ...


@cute.extern(name="ncclIrGetPeerPointer", source=_BC)
def _nccl_ir_get_peer_pointer(
    window: _NcclWindowT,
    offset: _SizeT,
    peer: cutlass.Int32,
) -> _LLVMPtrType: ...


@cute.extern(name="ncclIrGetPeerPointerTeam", source=_BC)
def _nccl_ir_get_peer_pointer_team(
    window: _NcclWindowT,
    offset: _SizeT,
    team: _NcclTeamPtr,
    peer: cutlass.Int32,
) -> _LLVMPtrType: ...


@cute.extern(name="ncclIrGetMultimemPointer", source=_BC)
def _nccl_ir_get_multimem_pointer(
    window: _NcclWindowT,
    offset: _SizeT,
    mm_handle: _NcclMultimemHandlePtr,
) -> _LLVMPtrType: ...


@cute.extern(name="ncclIrGetLsaMultimemPointer", source=_BC)
def _nccl_ir_get_lsa_multimem_pointer(
    window: _NcclWindowT,
    offset: _SizeT,
    dev_comm: _NcclIrDevCommPtr,
) -> _LLVMPtrType: ...


# === Resource buffer pointers ===


@cute.extern(name="ncclIrGetResourceBufferLocalPointer", source=_BC)
def _nccl_ir_get_resource_buffer_local_pointer(
    dev_comm: _NcclIrDevCommPtr,
    handle: _NcclDevResourceHandle,
) -> _LLVMPtrType: ...


@cute.extern(name="ncclIrGetResourceBufferLsaPointer", source=_BC)
def _nccl_ir_get_resource_buffer_lsa_pointer(
    dev_comm: _NcclIrDevCommPtr,
    handle: _NcclDevResourceHandle,
    peer: cutlass.Int32,
) -> _LLVMPtrType: ...


@cute.extern(name="ncclIrGetResourceBufferPeerPointer", source=_BC)
def _nccl_ir_get_resource_buffer_peer_pointer(
    dev_comm: _NcclIrDevCommPtr,
    handle: _NcclDevResourceHandle,
    team: _NcclTeamPtr,
    peer: cutlass.Int32,
) -> _LLVMPtrType: ...


@cute.extern(name="ncclIrGetResourceBufferMultimemPointer", source=_BC)
def _nccl_ir_get_resource_buffer_multimem_pointer(
    dev_comm: _NcclIrDevCommPtr,
    handle: _NcclDevResourceHandle,
    mm_handle: _NcclMultimemHandlePtr,
) -> _LLVMPtrType: ...


@cute.extern(name="ncclIrGetResourceBufferLsaMultimemPointer", source=_BC)
def _nccl_ir_get_resource_buffer_lsa_multimem_pointer(
    dev_comm: _NcclIrDevCommPtr,
    handle: _NcclDevResourceHandle,
) -> _LLVMPtrType: ...


# === CFT logical endpoints ===
#
# `le_id` and `le_offset` are out-params: pointers to caller-owned
# ncclCftLeId (uint32) and size_t storage.


@cute.extern(name="ncclIrGetCftLeInfo", source=_BC)
def _nccl_ir_get_cft_le_info(
    window: _NcclWindowT,
    offset: _SizeT,
    peer_cft: cutlass.Int32,
    cft_team: _NcclTeamPtr,
    dev_comm: _NcclIrDevCommPtr,
    le_id: _NcclCftLeIdPtr,
    le_offset: _SizeTPtr,
) -> None: ...


@cute.extern(name="ncclIrGetPeerLeInfo", source=_BC)
def _nccl_ir_get_peer_le_info(
    window: _NcclWindowT,
    offset: _SizeT,
    peer_world: cutlass.Int32,
    dev_comm: _NcclIrDevCommPtr,
    le_id: _NcclCftLeIdPtr,
    le_offset: _SizeTPtr,
) -> None: ...


@cute.extern(name="ncclIrGetMultimemLeInfo", source=_BC)
def _nccl_ir_get_multimem_le_info(
    window: _NcclWindowT,
    offset: _SizeT,
    dev_comm: _NcclIrDevCommPtr,
    le_id: _NcclCftLeIdPtr,
    le_offset: _SizeTPtr,
) -> None: ...


@cute.extern(name="ncclIrGetResourceBufferCftLeInfo", source=_BC)
def _nccl_ir_get_resource_buffer_cft_le_info(
    dev_comm: _NcclIrDevCommPtr,
    handle: _NcclDevResourceHandle,
    peer_cft: cutlass.Int32,
    le_id: _NcclCftLeIdPtr,
    le_offset: _SizeTPtr,
) -> None: ...


@cute.extern(name="ncclIrGetResourceBufferPeerLeInfo", source=_BC)
def _nccl_ir_get_resource_buffer_peer_le_info(
    dev_comm: _NcclIrDevCommPtr,
    handle: _NcclDevResourceHandle,
    peer_world: cutlass.Int32,
    le_id: _NcclCftLeIdPtr,
    le_offset: _SizeTPtr,
) -> None: ...


@cute.extern(name="ncclIrGetResourceBufferMultimemLeInfo", source=_BC)
def _nccl_ir_get_resource_buffer_multimem_le_info(
    dev_comm: _NcclIrDevCommPtr,
    handle: _NcclDevResourceHandle,
    le_id: _NcclCftLeIdPtr,
    le_offset: _SizeTPtr,
) -> None: ...


_REDUCE_COPY_DTYPE_SUFFIXES = {
    cutlass.Int8: "I8",
    cutlass.Uint8: "U8",
    cutlass.Int32: "I32",
    cutlass.Uint32: "U32",
    cutlass.Int64: "I64",
    cutlass.Uint64: "U64",
    cutlass.Float16: "F16",
    cutlass.Float32: "F32",
    cutlass.Float64: "F64",
    cutlass.BFloat16: "BF16",
    cutlass.Float8E4M3: "F8E4M3",
    cutlass.Float8E5M2: "F8E5M2",
}

_REDUCE_COPY_ALL_DTYPES = tuple(_REDUCE_COPY_DTYPE_SUFFIXES)

_REDUCE_COPY_MULTIMEM_DTYPES = (
    cutlass.Int32,
    cutlass.Uint32,
    cutlass.Int64,
    cutlass.Uint64,
    cutlass.Float16,
    cutlass.Float32,
    cutlass.Float64,
    cutlass.BFloat16,
    cutlass.Float8E4M3,
    cutlass.Float8E5M2,
)

_ReduceCopyAllDTypes = TypeVar(
    "_ReduceCopyAllDTypes",
    *_REDUCE_COPY_ALL_DTYPES,
)
_ReduceCopyMultimemDTypes = TypeVar(
    "_ReduceCopyMultimemDTypes",
    *_REDUCE_COPY_MULTIMEM_DTYPES,
)


def _reduce_copy_name_mangler(ffi):
    dtype_param = next(
        (param for param in ffi.params_types if isinstance(param, cute.ConstValue)),
        None,
    )
    if dtype_param is None:
        raise TypeError(f"{ffi.name} requires a constexpr dtype")
    dtype = dtype_param.value

    if ffi.name.startswith("ncclIrMultimem") and dtype not in _REDUCE_COPY_MULTIMEM_DTYPES:
        raise TypeError(f"{ffi.name} does not support {dtype}")

    suffix = _REDUCE_COPY_DTYPE_SUFFIXES[dtype]
    return f"{ffi.name}_{suffix}"


@cute.extern(
    name="ncclIrLsaReduceSum",
    source=_BC,
    name_mangler=_reduce_copy_name_mangler,
)
def _nccl_ir_lsa_reduce_sum(
    coop: _NcclIrCoopPtr,
    src_window: _NcclWindowT,
    src_offset: _SizeT,
    dst: _LLVMPtrType,
    count: _SizeT,
    team: _NcclTeamPtr,
    dtype: cutlass.Constexpr[_ReduceCopyAllDTypes],
) -> None: ...


@cute.extern(
    name="ncclIrMultimemReduceSum",
    source=_BC,
    name_mangler=_reduce_copy_name_mangler,
)
def _nccl_ir_multimem_reduce_sum(
    coop: _NcclIrCoopPtr,
    mc_src: _LLVMPtrType,
    dst: _LLVMPtrType,
    count: _SizeT,
    dtype: cutlass.Constexpr[_ReduceCopyMultimemDTypes],
) -> None: ...


@cute.extern(
    name="ncclIrLsaCopy",
    source=_BC,
    name_mangler=_reduce_copy_name_mangler,
)
def _nccl_ir_lsa_copy(
    coop: _NcclIrCoopPtr,
    src: _LLVMPtrType,
    dst_window: _NcclWindowT,
    dst_offset: _SizeT,
    count: _SizeT,
    team: _NcclTeamPtr,
    dtype: cutlass.Constexpr[_ReduceCopyAllDTypes],
) -> None: ...


@cute.extern(
    name="ncclIrLsaCopyTma",
    source=_BC,
    name_mangler=_reduce_copy_name_mangler,
)
def _nccl_ir_lsa_copy_tma(
    coop: _NcclIrCoopPtr,
    src: _LLVMPtrType,
    dst_window: _NcclWindowT,
    dst_offset: _SizeT,
    count: _SizeT,
    team: _NcclTeamPtr,
    smem: _LLVMPtrType,
    smem_bytes: cutlass.Int32,
    dtype: cutlass.Constexpr[_ReduceCopyAllDTypes],
) -> None: ...


@cute.extern(
    name="ncclIrMultimemCopy",
    source=_BC,
    name_mangler=_reduce_copy_name_mangler,
)
def _nccl_ir_multimem_copy(
    coop: _NcclIrCoopPtr,
    src: _LLVMPtrType,
    mc_dst: _LLVMPtrType,
    count: _SizeT,
    dtype: cutlass.Constexpr[_ReduceCopyMultimemDTypes],
) -> None: ...


@cute.extern(
    name="ncclIrLsaReduceSumCopy",
    source=_BC,
    name_mangler=_reduce_copy_name_mangler,
)
def _nccl_ir_lsa_reduce_sum_copy(
    coop: _NcclIrCoopPtr,
    src_window: _NcclWindowT,
    src_offset: _SizeT,
    dst_window: _NcclWindowT,
    dst_offset: _SizeT,
    count: _SizeT,
    team: _NcclTeamPtr,
    dtype: cutlass.Constexpr[_ReduceCopyAllDTypes],
) -> None: ...


@cute.extern(
    name="ncclIrMultimemReduceSumCopy",
    source=_BC,
    name_mangler=_reduce_copy_name_mangler,
)
def _nccl_ir_multimem_reduce_sum_copy(
    coop: _NcclIrCoopPtr,
    mc_src: _LLVMPtrType,
    mc_dst: _LLVMPtrType,
    count: _SizeT,
    dtype: cutlass.Constexpr[_ReduceCopyMultimemDTypes],
) -> None: ...


@cute.extern(
    name="ncclIrLocalReduceSumCopy",
    source=_BC,
    name_mangler=_reduce_copy_name_mangler,
)
def _nccl_ir_local_reduce_sum_copy(
    coop: _NcclIrCoopPtr,
    n_src: cutlass.Int32,
    src_base: _LLVMPtrType,
    src_displ: _SizeT,
    n_dst: cutlass.Int32,
    dst_base: _LLVMPtrType,
    dst_displ: _SizeT,
    count: _SizeT,
    dtype: cutlass.Constexpr[_ReduceCopyAllDTypes],
) -> None: ...


# === GIN API ===


@cute.extern(name="ncclIrGinInit", source=_BC)
def _nccl_ir_gin_init(
    gin: _NcclIrGinPtr,
    dev_comm: _NcclIrDevCommPtr,
    context_id: cutlass.Int32,
    resource_sharing_mode: _NcclGinResourceSharingMode,
) -> None: ...


@cute.extern(name="ncclIrGinPut", source=_BC)
def _nccl_ir_gin_put(
    gin: _NcclIrGinPtr,
    team: _NcclTeamPtr,
    peer: cutlass.Int32,
    dst_win: _NcclWindowT,
    dst_offset: _SizeT,
    src_win: _NcclWindowT,
    src_offset: _SizeT,
    size: _SizeT,
    signal_type: _NcclGinSignalType,  # indexed, VA, or none
    signal_win: _NcclWindowT,
    signal_offset: _SizeT,
    signal_id: _NcclGinSignalT,
    is_strong: cutlass.Boolean,
    signal_op: _NcclGinSignalOpT,
    signal_op_arg: cutlass.Uint64,
    is_counter: cutlass.Boolean,
    counter_id: _NcclGinCounterT,
    coop: _NcclIrCoopPtr,
    is_descriptor: cutlass.Boolean,
    descriptor_ptr: _NcclGinDescriptorSmemPtr,
    given_release: _CudaThreadScope,
    required_release: _CudaThreadScope,
    opt_flags: cutlass.Uint32,
    segment_type: _NcclGinSegmentTypeT,
) -> None: ...


@cute.extern(name="ncclIrGinPutValue", source=_BC)
def _nccl_ir_gin_put_value(
    gin: _NcclIrGinPtr,
    team: _NcclTeamPtr,
    peer: cutlass.Int32,
    dst_win: _NcclWindowT,
    dst_offset: _SizeT,
    value: cutlass.Uint64,
    size: _SizeT,
    signal_type: _NcclGinSignalType,  # indexed, VA, or none
    signal_win: _NcclWindowT,
    signal_offset: _SizeT,
    signal_id: _NcclGinSignalT,
    is_strong: cutlass.Boolean,
    signal_op: _NcclGinSignalOpT,
    signal_op_arg: cutlass.Uint64,
    coop: _NcclIrCoopPtr,
    is_descriptor: cutlass.Boolean,
    descriptor_ptr: _NcclGinDescriptorSmemPtr,
    given_release: _CudaThreadScope,
    required_release: _CudaThreadScope,
    opt_flags: cutlass.Uint32,
) -> None: ...


@cute.extern(name="ncclIrGinGet", source=_BC)
def _nccl_ir_gin_get(
    gin: _NcclIrGinPtr,
    team: _NcclTeamPtr,
    peer: cutlass.Int32,
    remote_win: _NcclWindowT,
    remote_offset: _SizeT,
    local_win: _NcclWindowT,
    local_offset: _SizeT,
    size: _SizeT,
    coop: _NcclIrCoopPtr,
    is_descriptor: cutlass.Boolean,
    descriptor_ptr: _NcclGinDescriptorSmemPtr,
    opt_flags: cutlass.Uint32,
    segment_type: _NcclGinSegmentTypeT,
) -> None: ...


@cute.extern(name="ncclIrGinFlush", source=_BC)
def _nccl_ir_gin_flush(
    gin: _NcclIrGinPtr,
    coop: _NcclIrCoopPtr,
    ord: _CudaMemoryOrder,
    is_descriptor: cutlass.Boolean,
    descriptor_ptr: _NcclGinDescriptorSmemPtr,
) -> None: ...


@cute.extern(name="ncclIrGinFlushTimeout", source=_BC)
def _nccl_ir_gin_flush_timeout(
    gin: _NcclIrGinPtr,
    coop: _NcclIrCoopPtr,
    ord: _CudaMemoryOrder,
    is_descriptor: cutlass.Boolean,
    descriptor_ptr: _NcclGinDescriptorSmemPtr,
    timeout_cycles: cutlass.Uint64,
) -> _NcclResultT: ...


@cute.extern(name="ncclIrGinFlushAsync", source=_BC)
def _nccl_ir_gin_flush_async(
    gin: _NcclIrGinPtr,
    team: _NcclTeamPtr,
    peer: cutlass.Uint32,
    out_request: _NcclGinRequestTPtr,
    coop: _NcclIrCoopPtr,
    opt_flags: cutlass.Uint32,
    is_descriptor: cutlass.Boolean,
    descriptor_ptr: _NcclGinDescriptorSmemPtr,
) -> None: ...


@cute.extern(name="ncclIrGinWait", source=_BC)
def _nccl_ir_gin_wait(
    gin: _NcclIrGinPtr,
    request: _NcclGinRequestTPtr,
    coop: _NcclIrCoopPtr,
    is_descriptor: cutlass.Boolean,
    descriptor_ptr: _NcclGinDescriptorSmemPtr,
    ord: _CudaMemoryOrder,
) -> None: ...


@cute.extern(name="ncclIrGinWaitTimeout", source=_BC)
def _nccl_ir_gin_wait_timeout(
    gin: _NcclIrGinPtr,
    request: _NcclGinRequestTPtr,
    coop: _NcclIrCoopPtr,
    is_descriptor: cutlass.Boolean,
    descriptor_ptr: _NcclGinDescriptorSmemPtr,
    ord: _CudaMemoryOrder,
    timeout_cycles: cutlass.Uint64,
) -> _NcclResultT: ...


@cute.extern(name="ncclIrGinSignal", source=_BC)
def _nccl_ir_gin_signal(
    gin: _NcclIrGinPtr,
    team: _NcclTeamPtr,
    peer: cutlass.Int32,
    signal_type: _NcclGinSignalType,  # indexed, VA, or none
    signal_win: _NcclWindowT,
    signal_offset: _SizeT,
    signal_id: _NcclGinSignalT,
    is_strong: cutlass.Boolean,
    signal_op: _NcclGinSignalOpT,
    signal_op_arg: cutlass.Uint64,
    coop: _NcclIrCoopPtr,
    is_descriptor: cutlass.Boolean,
    descriptor_ptr: _NcclGinDescriptorSmemPtr,
    given_release: _CudaThreadScope,
    required_release: _CudaThreadScope,
    opt_flags: cutlass.Uint32,
) -> None: ...


@cute.extern(name="ncclIrGinReadSignal", source=_BC)
def _nccl_ir_gin_read_signal(
    gin: _NcclIrGinPtr,
    signal_id: _NcclGinSignalT,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
) -> cutlass.Uint64: ...


@cute.extern(name="ncclIrGinReadSignalVA", source=_BC)
def _nccl_ir_gin_read_signal_va(
    gin: _NcclIrGinPtr,
    signal_window: _NcclWindowT,
    signal_offset: _SizeT,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
) -> cutlass.Uint64: ...


@cute.extern(name="ncclIrGinWaitSignal", source=_BC)
def _nccl_ir_gin_wait_signal(
    gin: _NcclIrGinPtr,
    coop: _NcclIrCoopPtr,
    signal: _NcclGinSignalT,
    least: cutlass.Uint64,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
) -> None: ...


@cute.extern(name="ncclIrGinWaitSignalTimeout", source=_BC)
def _nccl_ir_gin_wait_signal_timeout(
    gin: _NcclIrGinPtr,
    coop: _NcclIrCoopPtr,
    signal: _NcclGinSignalT,
    least: cutlass.Uint64,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
    timeout_cycles: cutlass.Uint64,
) -> _NcclResultT: ...


@cute.extern(name="ncclIrGinWaitSignalVA", source=_BC)
def _nccl_ir_gin_wait_signal_va(
    gin: _NcclIrGinPtr,
    coop: _NcclIrCoopPtr,
    signal_window: _NcclWindowT,
    signal_offset: _SizeT,
    least: cutlass.Uint64,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
) -> None: ...


@cute.extern(name="ncclIrGinWaitSignalTimeoutVA", source=_BC)
def _nccl_ir_gin_wait_signal_timeout_va(
    gin: _NcclIrGinPtr,
    coop: _NcclIrCoopPtr,
    signal_window: _NcclWindowT,
    signal_offset: _SizeT,
    least: cutlass.Uint64,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
    timeout_cycles: cutlass.Uint64,
) -> _NcclResultT: ...


@cute.extern(name="ncclIrGinIncreaseSignalShadow", source=_BC)
def _nccl_ir_gin_increase_signal_shadow(
    gin: _NcclIrGinPtr,
    signal: _NcclGinSignalT,
    delta: cutlass.Uint64,
) -> None: ...


@cute.extern(name="ncclIrGinWaitSignalMeetShadow", source=_BC)
def _nccl_ir_gin_wait_signal_meet_shadow(
    gin: _NcclIrGinPtr,
    coop: _NcclIrCoopPtr,
    signal: _NcclGinSignalT,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
) -> None: ...


@cute.extern(name="ncclIrGinWaitSignalFollowShadow", source=_BC)
def _nccl_ir_gin_wait_signal_follow_shadow(
    gin: _NcclIrGinPtr,
    coop: _NcclIrCoopPtr,
    signal: _NcclGinSignalT,
    least_delta: cutlass.Uint64,
    before: _Uint64TPtr,
    delta: _Uint64TPtr,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
) -> None: ...


@cute.extern(name="ncclIrGinReadCounter", source=_BC)
def _nccl_ir_gin_read_counter(
    gin: _NcclIrGinPtr,
    counter: _NcclGinCounterT,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
) -> cutlass.Uint64: ...


@cute.extern(name="ncclIrGinWaitCounter", source=_BC)
def _nccl_ir_gin_wait_counter(
    gin: _NcclIrGinPtr,
    coop: _NcclIrCoopPtr,
    counter: _NcclGinCounterT,
    least: cutlass.Uint64,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
) -> None: ...


@cute.extern(name="ncclIrGinWaitCounterTimeout", source=_BC)
def _nccl_ir_gin_wait_counter_timeout(
    gin: _NcclIrGinPtr,
    coop: _NcclIrCoopPtr,
    counter: _NcclGinCounterT,
    least: cutlass.Uint64,
    bits: cutlass.Int32,
    ord: _CudaMemoryOrder,
    timeout_cycles: cutlass.Uint64,
) -> _NcclResultT: ...


@cute.extern(name="ncclIrGinResetCounter", source=_BC)
def _nccl_ir_gin_reset_counter(
    gin: _NcclIrGinPtr,
    counter: _NcclGinCounterT,
) -> None: ...


@cute.extern(name="ncclIrGinResetSignal", source=_BC)
def _nccl_ir_gin_reset_signal(
    gin: _NcclIrGinPtr,
    signal: _NcclGinSignalT,
) -> None: ...


@cute.extern(name="ncclIrGinResetSignalVA", source=_BC)
def _nccl_ir_gin_reset_signal_va(
    gin: _NcclIrGinPtr,
    signal_window: _NcclWindowT,
    signal_offset: _SizeT,
) -> None: ...


@cute.extern(name="ncclIrGinGetSignalShadowPtr", source=_BC)
def _nccl_ir_gin_get_signal_shadow_ptr(
    gin: _NcclIrGinPtr,
    signal: _NcclGinSignalT,
) -> _Uint64TPtr: ...


# === LSA Barrier Session API ===


@cute.extern(name="ncclIrLsaBarrierSessionInit", source=_BC)
def _nccl_ir_lsa_barrier_session_init(
    session: _NcclIrLsaBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    dev_comm: _NcclIrDevCommPtr,
    team: _NcclTeamPtr,
    handle: _NcclLsaBarrierHandlePtr,
    index: cutlass.Uint32,
    multimem: cutlass.Boolean,
    mm_handle: _NcclMultimemHandlePtr,
) -> None: ...


@cute.extern(name="ncclIrLsaBarrierSessionInitLsa", source=_BC)
def _nccl_ir_lsa_barrier_session_init_lsa(
    session: _NcclIrLsaBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    dev_comm: _NcclIrDevCommPtr,
    index: cutlass.Uint32,
    multimem: cutlass.Boolean,
) -> None: ...


@cute.extern(name="ncclIrLsaBarrierSessionArrive", source=_BC)
def _nccl_ir_lsa_barrier_session_arrive(
    session: _NcclIrLsaBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    order: _CudaMemoryOrder,
) -> None: ...


@cute.extern(name="ncclIrLsaBarrierSessionWait", source=_BC)
def _nccl_ir_lsa_barrier_session_wait(
    session: _NcclIrLsaBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    order: _CudaMemoryOrder,
) -> None: ...


@cute.extern(name="ncclIrLsaBarrierSessionSync", source=_BC)
def _nccl_ir_lsa_barrier_session_sync(
    session: _NcclIrLsaBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    order: _CudaMemoryOrder,
) -> None: ...


@cute.extern(name="ncclIrLsaBarrierSessionDestroy", source=_BC)
def _nccl_ir_lsa_barrier_session_destroy(session: _NcclIrLsaBarrierSessionPtr) -> None: ...


# === GIN Barrier Session API ===


@cute.extern(name="ncclIrGinBarrierSessionInit", source=_BC)
def _nccl_ir_gin_barrier_session_init(
    session: _NcclIrGinBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    gin: _NcclIrGinPtr,
    team: _NcclTeamPtr,
    handle: _NcclGinBarrierHandlePtr,
    index: cutlass.Uint32,
) -> None: ...


@cute.extern(name="ncclIrGinBarrierSessionInitAllContexts", source=_BC)
def _nccl_ir_gin_barrier_session_init_all_contexts(
    session: _NcclIrGinBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    dev_comm: _NcclIrDevCommPtr,
    team: _NcclTeamPtr,
    handle: _NcclGinBarrierHandlePtr,
    index: cutlass.Uint32,
) -> None: ...


@cute.extern(name="ncclIrGinBarrierSessionInitRail", source=_BC)
def _nccl_ir_gin_barrier_session_init_rail(
    session: _NcclIrGinBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    gin: _NcclIrGinPtr,
    index: cutlass.Uint32,
) -> None: ...


@cute.extern(name="ncclIrGinBarrierSessionInitWorld", source=_BC)
def _nccl_ir_gin_barrier_session_init_world(
    session: _NcclIrGinBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    gin: _NcclIrGinPtr,
    index: cutlass.Uint32,
) -> None: ...


@cute.extern(name="ncclIrGinBarrierSessionInitAllContextsRail", source=_BC)
def _nccl_ir_gin_barrier_session_init_all_contexts_rail(
    session: _NcclIrGinBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    dev_comm: _NcclIrDevCommPtr,
    index: cutlass.Uint32,
) -> None: ...


@cute.extern(name="ncclIrGinBarrierSessionInitAllContextsWorld", source=_BC)
def _nccl_ir_gin_barrier_session_init_all_contexts_world(
    session: _NcclIrGinBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    dev_comm: _NcclIrDevCommPtr,
    index: cutlass.Uint32,
) -> None: ...


@cute.extern(name="ncclIrGinBarrierSessionSync", source=_BC)
def _nccl_ir_gin_barrier_session_sync(
    session: _NcclIrGinBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    order: _CudaMemoryOrder,
    fence: _NcclGinFenceLevel,
) -> None: ...


@cute.extern(name="ncclIrGinBarrierSessionDestroy", source=_BC)
def _nccl_ir_gin_barrier_session_destroy(session: _NcclIrGinBarrierSessionPtr) -> None: ...


# === Hybrid Barrier Session API ===


@cute.extern(name="ncclIrBarrierSessionInit", source=_BC)
def _nccl_ir_barrier_session_init(
    session: _NcclIrBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    inner_team: _NcclTeamPtr,
    outer_team: _NcclTeamPtr,
    gin: _NcclIrGinPtr,
    inner_handle: _NcclLsaBarrierHandlePtr,
    outer_handle: _NcclGinBarrierHandlePtr,
    index: cutlass.Uint32,
    multimem: cutlass.Boolean,
    inner_mm_handle: _NcclMultimemHandlePtr,
) -> None: ...


@cute.extern(name="ncclIrBarrierSessionInitWorld", source=_BC)
def _nccl_ir_barrier_session_init_world(
    session: _NcclIrBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    gin: _NcclIrGinPtr,
    index: cutlass.Uint32,
    multimem: cutlass.Boolean,
) -> None: ...


@cute.extern(name="ncclIrBarrierSessionInitLsa", source=_BC)
def _nccl_ir_barrier_session_init_lsa(
    session: _NcclIrBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    dev_comm: _NcclIrDevCommPtr,
    index: cutlass.Uint32,
    multimem: cutlass.Boolean,
) -> None: ...


@cute.extern(name="ncclIrBarrierSessionInitRail", source=_BC)
def _nccl_ir_barrier_session_init_rail(
    session: _NcclIrBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    gin: _NcclIrGinPtr,
    index: cutlass.Uint32,
) -> None: ...


@cute.extern(name="ncclIrBarrierSessionSync", source=_BC)
def _nccl_ir_barrier_session_sync(
    session: _NcclIrBarrierSessionPtr,
    coop: _NcclIrCoopPtr,
    order: _CudaMemoryOrder,
    fence: _NcclGinFenceLevel,
) -> None: ...


@cute.extern(name="ncclIrBarrierSessionDestroy", source=_BC)
def _nccl_ir_barrier_session_destroy(session: _NcclIrBarrierSessionPtr) -> None: ...
