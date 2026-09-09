"""1:1 CuTeDSL ``@cute.extern`` prototypes for ``bindings/ir/nccl_device_wrapper.h``.

Codegen target. Each entry is a typed ``@cute.extern`` stub bound to the
device bitcode (``source=_BC``): parameter types and return type are
derived from the stub signature, and the C symbol is named explicitly via
``name=`` so the Python stub can keep a PEP 8 ``snake_case`` name. The
stubs are the raw FFI surface — argument coercion (``_to_ptr`` /
``_to_value`` / cutlass numeric casts) and return-value wrapping are the
caller's job.

Struct decls live in :mod:`_structs`; coercion helpers in :mod:`_helpers`.
"""

import os
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.cute import BitCode

from cuda.pathfinder import find_bitcode_lib

from ._structs import (
    _LLVMPtrType,
    ncclTeam,
    ncclLsaBarrierHandle,
    ncclGinBarrierHandle,
    ncclMultimemHandle,
    ncclCoopAny,
)


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

@cute.extern(name="ncclCoopAnyInitThread", source=_BC)
def nccl_coop_any_init_thread(coop: _LLVMPtrType) -> None: ...

@cute.extern(name="ncclCoopAnyInitWarp", source=_BC)
def nccl_coop_any_init_warp(coop: _LLVMPtrType) -> None: ...

@cute.extern(name="ncclCoopAnyInitLanes", source=_BC)
def nccl_coop_any_init_lanes(coop: _LLVMPtrType, lane_mask: cutlass.Uint32) -> None: ...

@cute.extern(name="ncclCoopAnyInitWarpSpan", source=_BC)
def nccl_coop_any_init_warp_span(
    coop: _LLVMPtrType, warp0: cutlass.Int32, n_warps: cutlass.Int32,
    id: cutlass.Int32,
) -> None: ...

@cute.extern(name="ncclCoopAnyInitCta", source=_BC)
def nccl_coop_any_init_cta(coop: _LLVMPtrType) -> None: ...

@cute.extern(name="ncclCoopThreadRank", source=_BC)
def nccl_coop_thread_rank(coop: _LLVMPtrType) -> cutlass.Int32: ...

@cute.extern(name="ncclCoopSize", source=_BC)
def nccl_coop_size(coop: _LLVMPtrType) -> cutlass.Int32: ...

@cute.extern(name="ncclCoopNumThreads", source=_BC)
def nccl_coop_num_threads(coop: _LLVMPtrType) -> cutlass.Int32: ...

@cute.extern(name="ncclCoopSync", source=_BC)
def nccl_coop_sync(coop: _LLVMPtrType) -> None: ...


# === Core API (teams + window pointers) ===

@cute.extern(name="ncclTeamWorld", source=_BC)
def nccl_team_world(dev_comm: _LLVMPtrType) -> ncclTeam: ...

@cute.extern(name="ncclTeamLsa", source=_BC)
def nccl_team_lsa(dev_comm: _LLVMPtrType) -> ncclTeam: ...

@cute.extern(name="ncclTeamRail", source=_BC)
def nccl_team_rail(dev_comm: _LLVMPtrType) -> ncclTeam: ...

# `mode` is ncclCftTeamMode_t, i.e. nccl.bindings.nccl.CftTeamMode. The C
# declaration defaults it to FLAT; the symbol takes it explicitly.
@cute.extern(name="ncclTeamCft", source=_BC)
def nccl_team_cft(
    dev_comm: _LLVMPtrType, mode: cutlass.Int32,
) -> ncclTeam: ...

@cute.extern(name="ncclTeamCftMultimem", source=_BC)
def nccl_team_cft_multimem(dev_comm: _LLVMPtrType) -> ncclTeam: ...

@cute.extern(name="ncclTeamRankToWorld", source=_BC)
def nccl_team_rank_to_world(
    dev_comm: _LLVMPtrType, team: ncclTeam, rank: cutlass.Int32,
) -> cutlass.Int32: ...

@cute.extern(name="ncclTeamRankToLsa", source=_BC)
def nccl_team_rank_to_lsa(
    dev_comm: _LLVMPtrType, team: ncclTeam, rank: cutlass.Int32,
) -> cutlass.Int32: ...

@cute.extern(name="ncclGetLocalPointer", source=_BC)
def nccl_get_local_pointer(
    window: _LLVMPtrType, offset: cutlass.Int64,
) -> _LLVMPtrType: ...

@cute.extern(name="ncclGetLsaPointer", source=_BC)
def nccl_get_lsa_pointer(
    window: _LLVMPtrType, offset: cutlass.Int64, peer: cutlass.Int32,
) -> _LLVMPtrType: ...

@cute.extern(name="ncclGetPeerPointer", source=_BC)
def nccl_get_peer_pointer(
    window: _LLVMPtrType, offset: cutlass.Int64, peer: cutlass.Int32,
) -> _LLVMPtrType: ...

@cute.extern(name="ncclGetPeerPointerTeam", source=_BC)
def nccl_get_peer_pointer_team(
    window: _LLVMPtrType, offset: cutlass.Int64, team: ncclTeam,
    peer: cutlass.Int32,
) -> _LLVMPtrType: ...

@cute.extern(name="ncclGetMultimemPointer", source=_BC)
def nccl_get_multimem_pointer(
    window: _LLVMPtrType, offset: cutlass.Int64,
    mm_handle: ncclMultimemHandle,
) -> _LLVMPtrType: ...

@cute.extern(name="ncclGetLsaMultimemPointer", source=_BC)
def nccl_get_lsa_multimem_pointer(
    window: _LLVMPtrType, offset: cutlass.Int64, dev_comm: _LLVMPtrType,
) -> _LLVMPtrType: ...


# === Resource buffer pointers ===

@cute.extern(name="ncclGetResourceBufferLocalPointer", source=_BC)
def nccl_get_resource_buffer_local_pointer(
    dev_comm: _LLVMPtrType, handle: cutlass.Uint32,
) -> _LLVMPtrType: ...

@cute.extern(name="ncclGetResourceBufferLsaPointer", source=_BC)
def nccl_get_resource_buffer_lsa_pointer(
    dev_comm: _LLVMPtrType, handle: cutlass.Uint32, peer: cutlass.Int32,
) -> _LLVMPtrType: ...

@cute.extern(name="ncclGetResourceBufferPeerPointer", source=_BC)
def nccl_get_resource_buffer_peer_pointer(
    dev_comm: _LLVMPtrType, handle: cutlass.Uint32, team: ncclTeam,
    peer: cutlass.Int32,
) -> _LLVMPtrType: ...

@cute.extern(name="ncclGetResourceBufferMultimemPointer", source=_BC)
def nccl_get_resource_buffer_multimem_pointer(
    dev_comm: _LLVMPtrType, handle: cutlass.Uint32,
    mm_handle: ncclMultimemHandle,
) -> _LLVMPtrType: ...

@cute.extern(name="ncclGetResourceBufferLsaMultimemPointer", source=_BC)
def nccl_get_resource_buffer_lsa_multimem_pointer(
    dev_comm: _LLVMPtrType, handle: cutlass.Uint32,
) -> _LLVMPtrType: ...


# === CFT logical endpoints ===
#
# `le_id` and `le_offset` are out-params: pointers to caller-owned
# ncclCftLeId (uint32) and size_t storage.

@cute.extern(name="ncclGetCftLeInfo", source=_BC)
def nccl_get_cft_le_info(
    window: _LLVMPtrType, offset: cutlass.Int64, peer_cft: cutlass.Int32,
    cft_team: ncclTeam, dev_comm: _LLVMPtrType, le_id: _LLVMPtrType,
    le_offset: _LLVMPtrType,
) -> None: ...

@cute.extern(name="ncclGetPeerLeInfo", source=_BC)
def nccl_get_peer_le_info(
    window: _LLVMPtrType, offset: cutlass.Int64, peer_world: cutlass.Int32,
    dev_comm: _LLVMPtrType, le_id: _LLVMPtrType, le_offset: _LLVMPtrType,
) -> None: ...

@cute.extern(name="ncclGetMultimemLeInfo", source=_BC)
def nccl_get_multimem_le_info(
    window: _LLVMPtrType, offset: cutlass.Int64, dev_comm: _LLVMPtrType,
    le_id: _LLVMPtrType, le_offset: _LLVMPtrType,
) -> None: ...

@cute.extern(name="ncclGetResourceBufferCftLeInfo", source=_BC)
def nccl_get_resource_buffer_cft_le_info(
    dev_comm: _LLVMPtrType, handle: cutlass.Uint32, peer_cft: cutlass.Int32,
    le_id: _LLVMPtrType, le_offset: _LLVMPtrType,
) -> None: ...

@cute.extern(name="ncclGetResourceBufferPeerLeInfo", source=_BC)
def nccl_get_resource_buffer_peer_le_info(
    dev_comm: _LLVMPtrType, handle: cutlass.Uint32, peer_world: cutlass.Int32,
    le_id: _LLVMPtrType, le_offset: _LLVMPtrType,
) -> None: ...

@cute.extern(name="ncclGetResourceBufferMultimemLeInfo", source=_BC)
def nccl_get_resource_buffer_multimem_le_info(
    dev_comm: _LLVMPtrType, handle: cutlass.Uint32, le_id: _LLVMPtrType,
    le_offset: _LLVMPtrType,
) -> None: ...


# === GIN API ===

@cute.extern(name="ncclGin_C_initWithResourceSharingMode", source=_BC)
def nccl_gin_c_init(
    gin: _LLVMPtrType, backend_mask: cutlass.Int32, dev_comm: _LLVMPtrType,
    context_id: cutlass.Int32, resource_sharing_mode: cutlass.Uint8,
) -> None: ...

@cute.extern(name="ncclGinPut_v2", source=_BC)
def nccl_gin_put(
    gin: _LLVMPtrType,             # ncclGin_C* gin
    team: ncclTeam,                # team
    peer: cutlass.Int32,           # peer
    dst_win: _LLVMPtrType,         # dst window
    dst_offset: cutlass.Int64,     # dst offset
    src_win: _LLVMPtrType,         # src window
    src_offset: cutlass.Int64,     # src offset
    size: cutlass.Int64,           # size
    is_signal: cutlass.Boolean,    # is_signal
    signal_id: cutlass.Int32,      # signal_id
    signal_op: cutlass.Int32,      # signal_op
    signal_op_arg: cutlass.Int64,  # signal_op_arg
    is_counter: cutlass.Boolean,   # is_counter
    counter_id: cutlass.Int32,     # counter_id
    coop: ncclCoopAny,             # coop
    is_descriptor: cutlass.Boolean,  # is_descriptor
    descriptor_ptr: _LLVMPtrType,  # descriptor_ptr
    given_release: cutlass.Int32,  # given_release
    required_release: cutlass.Int32,  # required_release
    opt_flags: cutlass.Int32,      # opt_flags
) -> None: ...

@cute.extern(name="ncclGinPutValue_v2", source=_BC)
def nccl_gin_put_value(
    gin: _LLVMPtrType,             # ncclGin_C* gin
    team: ncclTeam,                # team
    peer: cutlass.Int32,           # peer
    dst_win: _LLVMPtrType,         # dst window
    dst_offset: cutlass.Int64,     # dst offset
    value: cutlass.Int64,          # value
    size: cutlass.Int64,           # size
    is_signal: cutlass.Boolean,    # is_signal
    signal_id: cutlass.Int32,      # signal_id
    signal_op: cutlass.Int32,      # signal_op
    signal_op_arg: cutlass.Int64,  # signal_op_arg
    coop: ncclCoopAny,             # coop
    is_descriptor: cutlass.Boolean,  # is_descriptor
    descriptor_ptr: _LLVMPtrType,  # descriptor_ptr
    given_release: cutlass.Int32,  # given_release
    required_release: cutlass.Int32,  # required_release
    opt_flags: cutlass.Int32,      # opt_flags
) -> None: ...

@cute.extern(name="ncclGinGet", source=_BC)
def nccl_gin_get(
    gin: _LLVMPtrType,             # ncclGin_C* gin
    team: ncclTeam,                # team
    peer: cutlass.Int32,           # peer
    remote_win: _LLVMPtrType,      # remote window
    remote_offset: cutlass.Int64,  # remote offset
    local_win: _LLVMPtrType,       # local window
    local_offset: cutlass.Int64,   # local offset
    size: cutlass.Int64,           # size
    coop: ncclCoopAny,             # coop
    is_descriptor: cutlass.Boolean,  # is_descriptor
    descriptor_ptr: _LLVMPtrType,  # descriptor_ptr
    opt_flags: cutlass.Int32,      # opt_flags
) -> None: ...

@cute.extern(name="ncclGinFlush", source=_BC)
def nccl_gin_flush(
    gin: _LLVMPtrType, coop: ncclCoopAny, ord: cutlass.Int32,
) -> None: ...

@cute.extern(name="ncclGinSignal_v2", source=_BC)
def nccl_gin_signal(
    gin: _LLVMPtrType,             # ncclGin_C* gin
    team: ncclTeam,                # team
    peer: cutlass.Int32,           # peer
    is_signal: cutlass.Boolean,    # is_signal
    signal_id: cutlass.Int32,      # signal_id
    signal_op: cutlass.Int32,      # signal_op
    signal_op_arg: cutlass.Int64,  # signal_op_arg
    coop: ncclCoopAny,             # coop
    is_descriptor: cutlass.Boolean,  # is_descriptor
    descriptor_ptr: _LLVMPtrType,  # descriptor_ptr
    given_release: cutlass.Int32,  # given_release
    required_release: cutlass.Int32,  # required_release
    opt_flags: cutlass.Int32,      # opt_flags
) -> None: ...

@cute.extern(name="ncclGinReadSignal", source=_BC)
def nccl_gin_read_signal(
    gin: _LLVMPtrType, signal_id: cutlass.Int32, bits: cutlass.Int32,
    ord: cutlass.Int32,
) -> cutlass.Int64: ...

@cute.extern(name="ncclGinWaitSignal", source=_BC)
def nccl_gin_wait_signal(
    gin: _LLVMPtrType, coop: ncclCoopAny, signal: cutlass.Int32,
    least: cutlass.Int64, bits: cutlass.Int32, ord: cutlass.Int32,
) -> None: ...

@cute.extern(name="ncclGinReadCounter", source=_BC)
def nccl_gin_read_counter(
    gin: _LLVMPtrType, counter: cutlass.Int32, bits: cutlass.Int32,
    ord: cutlass.Int32,
) -> cutlass.Int64: ...

@cute.extern(name="ncclGinWaitCounter", source=_BC)
def nccl_gin_wait_counter(
    gin: _LLVMPtrType, coop: ncclCoopAny, counter: cutlass.Int32,
    least: cutlass.Int64, bits: cutlass.Int32, ord: cutlass.Int32,
) -> None: ...

@cute.extern(name="ncclGinResetCounter", source=_BC)
def nccl_gin_reset_counter(
    gin: _LLVMPtrType, counter: cutlass.Int32,
) -> None: ...

@cute.extern(name="ncclGinResetSignal", source=_BC)
def nccl_gin_reset_signal(
    gin: _LLVMPtrType, signal: cutlass.Int32,
) -> None: ...

@cute.extern(name="ncclGinGetSignalShadowPtr", source=_BC)
def nccl_gin_get_signal_shadow_ptr(
    gin: _LLVMPtrType, signal: cutlass.Int32,
) -> _LLVMPtrType: ...


# === LSA Barrier Session API ===

@cute.extern(name="ncclLsaBarrierSessionInit", source=_BC)
def nccl_lsa_barrier_session_init(
    session: _LLVMPtrType, coop: ncclCoopAny, dev_comm: _LLVMPtrType,
    team: ncclTeam, handle: ncclLsaBarrierHandle, index: cutlass.Uint32,
    multimem: cutlass.Boolean, mm_handle: ncclMultimemHandle,
) -> None: ...

@cute.extern(name="ncclLsaBarrierSessionArrive", source=_BC)
def nccl_lsa_barrier_session_arrive(
    session: _LLVMPtrType, coop: ncclCoopAny, order: cutlass.Int32,
) -> None: ...

@cute.extern(name="ncclLsaBarrierSessionWait", source=_BC)
def nccl_lsa_barrier_session_wait(
    session: _LLVMPtrType, coop: ncclCoopAny, order: cutlass.Int32,
) -> None: ...

@cute.extern(name="ncclLsaBarrierSessionSync", source=_BC)
def nccl_lsa_barrier_session_sync(
    session: _LLVMPtrType, coop: ncclCoopAny, order: cutlass.Int32,
) -> None: ...


# === GIN Barrier Session API ===

@cute.extern(name="ncclGinBarrierSessionInit", source=_BC)
def nccl_gin_barrier_session_init(
    session: _LLVMPtrType, coop: ncclCoopAny, gin: _LLVMPtrType,
    team: ncclTeam, handle: ncclGinBarrierHandle, index: cutlass.Uint32,
) -> None: ...

@cute.extern(name="ncclGinBarrierSessionInitAllContexts", source=_BC)
def nccl_gin_barrier_session_init_all_contexts(
    session: _LLVMPtrType, coop: ncclCoopAny, dev_comm: _LLVMPtrType,
    team: ncclTeam, handle: ncclGinBarrierHandle, index: cutlass.Uint32,
) -> None: ...

@cute.extern(name="ncclGinBarrierSessionSync", source=_BC)
def nccl_gin_barrier_session_sync(
    session: _LLVMPtrType, coop: ncclCoopAny, order: cutlass.Int32,
    fence: cutlass.Int32,
) -> None: ...


# === Hybrid Barrier Session API ===

@cute.extern(name="ncclBarrierSessionInit", source=_BC)
def nccl_barrier_session_init(
    session: _LLVMPtrType, coop: ncclCoopAny, inner_team: ncclTeam,
    outer_team: ncclTeam, gin: _LLVMPtrType, inner_handle: ncclLsaBarrierHandle,
    outer_handle: ncclGinBarrierHandle, index: cutlass.Uint32,
    multimem: cutlass.Boolean, inner_mm_handle: ncclMultimemHandle,
) -> None: ...

@cute.extern(name="ncclBarrierSessionSync", source=_BC)
def nccl_barrier_session_sync(
    session: _LLVMPtrType, coop: ncclCoopAny, order: cutlass.Int32,
    fence: cutlass.Int32,
) -> None: ...


# === Session size getters ===

@cute.extern(name="ncclLsaBarrierSession_C_size", source=_BC)
def nccl_lsa_barrier_session_c_size() -> cutlass.Int64: ...

@cute.extern(name="ncclGinBarrierSession_C_size", source=_BC)
def nccl_gin_barrier_session_c_size() -> cutlass.Int64: ...

@cute.extern(name="ncclBarrierSession_C_size", source=_BC)
def nccl_barrier_session_c_size() -> cutlass.Int64: ...
