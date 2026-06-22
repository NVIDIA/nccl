"""1:1 CuTeDSL FFI prototypes for ``bindings/ir/nccl_device_wrapper.h``.

Codegen target. Each bitcode symbol gets a paired
``_raw_<sym>`` (the ``cute.ffi`` prototype) and ``<sym>`` (a mechanical
Python wrapper that coerces arguments and wraps the return value).

Argument coercion is derivable from the declared FFI type:

  * ``_LLVMPtrType``         → ``_to_ptr(arg)``
  * ``ncclCoopAny`` by value → ``_to_coop_value(arg)``
  * cutlass numeric          → ``cutlass.<T>(arg)``
  * other native_struct      → passthrough

Coercion helpers live in :mod:`_helpers`; struct decls in :mod:`_structs`.
"""

import cutlass

from ._helpers import _ffi, _to_ptr, _to_coop_value
from ._structs import (
    _LLVMPtrType,
    ncclTeam,
    ncclLsaBarrierHandle,
    ncclGinBarrierHandle,
    ncclMultimemHandle,
    ncclCoopAny,
)


# === Coop API ===

_raw_ncclCoopAnyInitThread = _ffi(
    name="ncclCoopAnyInitThread", params_types=[_LLVMPtrType])

def ncclCoopAnyInitThread(coop_ptr):
    _raw_ncclCoopAnyInitThread(_to_ptr(coop_ptr))


_raw_ncclCoopAnyInitWarp = _ffi(
    name="ncclCoopAnyInitWarp", params_types=[_LLVMPtrType])

def ncclCoopAnyInitWarp(coop_ptr):
    _raw_ncclCoopAnyInitWarp(_to_ptr(coop_ptr))


_raw_ncclCoopAnyInitLanes = _ffi(
    name="ncclCoopAnyInitLanes", params_types=[_LLVMPtrType, cutlass.Uint32])

def ncclCoopAnyInitLanes(coop_ptr, lane_mask):
    _raw_ncclCoopAnyInitLanes(_to_ptr(coop_ptr), cutlass.Uint32(lane_mask))


_raw_ncclCoopAnyInitWarpSpan = _ffi(
    name="ncclCoopAnyInitWarpSpan",
    params_types=[_LLVMPtrType, cutlass.Int32, cutlass.Int32, cutlass.Int32])

def ncclCoopAnyInitWarpSpan(coop_ptr, warp0, n_warps, id):
    _raw_ncclCoopAnyInitWarpSpan(
        _to_ptr(coop_ptr),
        cutlass.Int32(warp0),
        cutlass.Int32(n_warps),
        cutlass.Int32(id),
    )


_raw_ncclCoopAnyInitCta = _ffi(
    name="ncclCoopAnyInitCta", params_types=[_LLVMPtrType])

def ncclCoopAnyInitCta(coop_ptr):
    _raw_ncclCoopAnyInitCta(_to_ptr(coop_ptr))


_raw_ncclCoopThreadRank = _ffi(
    name="ncclCoopThreadRank",
    params_types=[_LLVMPtrType], return_type=cutlass.Int32)

def ncclCoopThreadRank(coop_ptr):
    return cutlass.Int32(_raw_ncclCoopThreadRank(_to_ptr(coop_ptr)))


_raw_ncclCoopSize = _ffi(
    name="ncclCoopSize",
    params_types=[_LLVMPtrType], return_type=cutlass.Int32)

def ncclCoopSize(coop_ptr):
    return cutlass.Int32(_raw_ncclCoopSize(_to_ptr(coop_ptr)))


_raw_ncclCoopNumThreads = _ffi(
    name="ncclCoopNumThreads",
    params_types=[_LLVMPtrType], return_type=cutlass.Int32)

def ncclCoopNumThreads(coop_ptr):
    return cutlass.Int32(_raw_ncclCoopNumThreads(_to_ptr(coop_ptr)))


_raw_ncclCoopSync = _ffi(
    name="ncclCoopSync", params_types=[_LLVMPtrType])

def ncclCoopSync(coop_ptr):
    _raw_ncclCoopSync(_to_ptr(coop_ptr))


# === Core API (teams + window pointers) ===

_raw_ncclTeamWorld = _ffi(
    name="ncclTeamWorld", params_types=[_LLVMPtrType], return_type=ncclTeam)

def ncclTeamWorld(dev_comm):
    return ncclTeam(_raw_ncclTeamWorld(_to_ptr(dev_comm)))


_raw_ncclTeamLsa = _ffi(
    name="ncclTeamLsa", params_types=[_LLVMPtrType], return_type=ncclTeam)

def ncclTeamLsa(dev_comm):
    return ncclTeam(_raw_ncclTeamLsa(_to_ptr(dev_comm)))


_raw_ncclTeamRail = _ffi(
    name="ncclTeamRail", params_types=[_LLVMPtrType], return_type=ncclTeam)

def ncclTeamRail(dev_comm):
    return ncclTeam(_raw_ncclTeamRail(_to_ptr(dev_comm)))


_raw_ncclGetLocalPointer = _ffi(
    name="ncclGetLocalPointer",
    params_types=[_LLVMPtrType, cutlass.Int64], return_type=_LLVMPtrType)

def ncclGetLocalPointer(window, offset):
    return _raw_ncclGetLocalPointer(_to_ptr(window), cutlass.Int64(offset))


_raw_ncclGetLsaPointer = _ffi(
    name="ncclGetLsaPointer",
    params_types=[_LLVMPtrType, cutlass.Int64, cutlass.Int32],
    return_type=_LLVMPtrType)

def ncclGetLsaPointer(window, offset, peer):
    return _raw_ncclGetLsaPointer(
        _to_ptr(window), cutlass.Int64(offset), cutlass.Int32(peer))


_raw_ncclGetPeerPointer = _ffi(
    name="ncclGetPeerPointer",
    params_types=[_LLVMPtrType, cutlass.Int64, cutlass.Int32],
    return_type=_LLVMPtrType)

def ncclGetPeerPointer(window, offset, peer):
    return _raw_ncclGetPeerPointer(
        _to_ptr(window), cutlass.Int64(offset), cutlass.Int32(peer))


_raw_ncclGetPeerPointerTeam = _ffi(
    name="ncclGetPeerPointerTeam",
    params_types=[_LLVMPtrType, cutlass.Int64, ncclTeam, cutlass.Int32],
    return_type=_LLVMPtrType)

def ncclGetPeerPointerTeam(window, offset, team, peer):
    return _raw_ncclGetPeerPointerTeam(
        _to_ptr(window), cutlass.Int64(offset), team, cutlass.Int32(peer))


# === GIN API ===

_raw_ncclGin_C_init = _ffi(
    name="ncclGin_C_init",
    params_types=[_LLVMPtrType, cutlass.Int32, _LLVMPtrType, cutlass.Int32])

def ncclGin_C_init(gin_ptr, backend_mask, dev_comm, context_id):
    _raw_ncclGin_C_init(
        _to_ptr(gin_ptr),
        cutlass.Int32(int(backend_mask)),
        _to_ptr(dev_comm),
        cutlass.Int32(context_id),
    )


_raw_ncclGinPut = _ffi(
    name="ncclGinPut",
    params_types=[
        _LLVMPtrType,        # ncclGin_C* gin
        ncclTeam,            # team
        cutlass.Int32,       # peer
        _LLVMPtrType,        # dst window
        cutlass.Int64,       # dst offset
        _LLVMPtrType,        # src window
        cutlass.Int64,       # src offset
        cutlass.Int64,       # size
        cutlass.Boolean,     # is_signal
        cutlass.Int32,       # signal_id
        cutlass.Int32,       # signal_op
        cutlass.Int64,       # signal_op_arg
        cutlass.Boolean,     # is_counter
        cutlass.Int32,       # counter_id
        ncclCoopAny,         # coop
        cutlass.Boolean,     # is_descriptor
        _LLVMPtrType,        # descriptor_ptr
        cutlass.Int32,       # given_release
        cutlass.Int32,       # required_release
    ])

def ncclGinPut(gin_ptr, team, peer, dst_win, dst_offset, src_win, src_offset,
               size, is_signal, signal_id, signal_op, signal_op_arg,
               is_counter, counter_id, coop, is_descriptor, descriptor_ptr,
               given_release, required_release):
    _raw_ncclGinPut(
        _to_ptr(gin_ptr), team, cutlass.Int32(peer),
        _to_ptr(dst_win), cutlass.Int64(dst_offset),
        _to_ptr(src_win), cutlass.Int64(src_offset),
        cutlass.Int64(size),
        cutlass.Boolean(is_signal), cutlass.Int32(signal_id),
        cutlass.Int32(signal_op), cutlass.Int64(signal_op_arg),
        cutlass.Boolean(is_counter), cutlass.Int32(counter_id),
        _to_coop_value(coop),
        cutlass.Boolean(is_descriptor), _to_ptr(descriptor_ptr),
        cutlass.Int32(given_release), cutlass.Int32(required_release),
    )


_raw_ncclGinWaitSignal = _ffi(
    name="ncclGinWaitSignal",
    params_types=[
        _LLVMPtrType, ncclCoopAny, cutlass.Int32, cutlass.Int64,
        cutlass.Int32, cutlass.Int32,
    ])

def ncclGinWaitSignal(gin_ptr, coop, signal, least, bits, ord):
    _raw_ncclGinWaitSignal(
        _to_ptr(gin_ptr), _to_coop_value(coop),
        cutlass.Int32(signal), cutlass.Int64(least),
        cutlass.Int32(bits), cutlass.Int32(ord),
    )


# === LSA Barrier Session API ===

_raw_ncclLsaBarrierSessionInit = _ffi(
    name="ncclLsaBarrierSessionInit",
    params_types=[
        _LLVMPtrType, ncclCoopAny, _LLVMPtrType, ncclTeam,
        ncclLsaBarrierHandle, cutlass.Uint32, cutlass.Boolean,
        ncclMultimemHandle,
    ])

def ncclLsaBarrierSessionInit(session_ptr, coop, dev_comm, team, handle,
                               index, multimem, mm_handle):
    _raw_ncclLsaBarrierSessionInit(
        _to_ptr(session_ptr), _to_coop_value(coop), _to_ptr(dev_comm), team,
        handle, cutlass.Uint32(index), cutlass.Boolean(multimem), mm_handle,
    )


_raw_ncclLsaBarrierSessionArrive = _ffi(
    name="ncclLsaBarrierSessionArrive",
    params_types=[_LLVMPtrType, ncclCoopAny, cutlass.Int32])

def ncclLsaBarrierSessionArrive(session_ptr, coop, order):
    _raw_ncclLsaBarrierSessionArrive(
        _to_ptr(session_ptr), _to_coop_value(coop), cutlass.Int32(int(order)))


_raw_ncclLsaBarrierSessionWait = _ffi(
    name="ncclLsaBarrierSessionWait",
    params_types=[_LLVMPtrType, ncclCoopAny, cutlass.Int32])

def ncclLsaBarrierSessionWait(session_ptr, coop, order):
    _raw_ncclLsaBarrierSessionWait(
        _to_ptr(session_ptr), _to_coop_value(coop), cutlass.Int32(int(order)))


_raw_ncclLsaBarrierSessionSync = _ffi(
    name="ncclLsaBarrierSessionSync",
    params_types=[_LLVMPtrType, ncclCoopAny, cutlass.Int32])

def ncclLsaBarrierSessionSync(session_ptr, coop, order):
    _raw_ncclLsaBarrierSessionSync(
        _to_ptr(session_ptr), _to_coop_value(coop), cutlass.Int32(int(order)))


# === GIN Barrier Session API ===

_raw_ncclGinBarrierSessionInit = _ffi(
    name="ncclGinBarrierSessionInit",
    params_types=[
        _LLVMPtrType, ncclCoopAny, _LLVMPtrType, ncclTeam,
        ncclGinBarrierHandle, cutlass.Uint32,
    ])

def ncclGinBarrierSessionInit(session_ptr, coop, gin_ptr, team, handle, index):
    _raw_ncclGinBarrierSessionInit(
        _to_ptr(session_ptr), _to_coop_value(coop), _to_ptr(gin_ptr), team, handle,
        cutlass.Uint32(index),
    )


_raw_ncclGinBarrierSessionSync = _ffi(
    name="ncclGinBarrierSessionSync",
    params_types=[_LLVMPtrType, ncclCoopAny, cutlass.Int32, cutlass.Int32])

def ncclGinBarrierSessionSync(session_ptr, coop, order, fence):
    _raw_ncclGinBarrierSessionSync(
        _to_ptr(session_ptr), _to_coop_value(coop),
        cutlass.Int32(int(order)), cutlass.Int32(int(fence)),
    )


# === Hybrid Barrier Session API ===

_raw_ncclBarrierSessionInit = _ffi(
    name="ncclBarrierSessionInit",
    params_types=[
        _LLVMPtrType, ncclCoopAny, ncclTeam, ncclTeam, _LLVMPtrType,
        ncclLsaBarrierHandle, ncclGinBarrierHandle, cutlass.Uint32,
        cutlass.Boolean, ncclMultimemHandle,
    ])

def ncclBarrierSessionInit(session_ptr, coop, inner_team, outer_team,
                            gin_ptr, inner_handle, outer_handle, index,
                            multimem, inner_mm_handle):
    _raw_ncclBarrierSessionInit(
        _to_ptr(session_ptr), _to_coop_value(coop), inner_team, outer_team,
        _to_ptr(gin_ptr), inner_handle, outer_handle, cutlass.Uint32(index),
        cutlass.Boolean(multimem), inner_mm_handle,
    )


_raw_ncclBarrierSessionSync = _ffi(
    name="ncclBarrierSessionSync",
    params_types=[_LLVMPtrType, ncclCoopAny, cutlass.Int32, cutlass.Int32])

def ncclBarrierSessionSync(session_ptr, coop, order, fence):
    _raw_ncclBarrierSessionSync(
        _to_ptr(session_ptr), _to_coop_value(coop),
        cutlass.Int32(int(order)), cutlass.Int32(int(fence)),
    )


# === Session size getters ===

_raw_ncclLsaBarrierSession_C_size = _ffi(
    name="ncclLsaBarrierSession_C_size", return_type=cutlass.Int64)

def ncclLsaBarrierSession_C_size():
    return cutlass.Int64(_raw_ncclLsaBarrierSession_C_size())


_raw_ncclGinBarrierSession_C_size = _ffi(
    name="ncclGinBarrierSession_C_size", return_type=cutlass.Int64)

def ncclGinBarrierSession_C_size():
    return cutlass.Int64(_raw_ncclGinBarrierSession_C_size())


_raw_ncclBarrierSession_C_size = _ffi(
    name="ncclBarrierSession_C_size", return_type=cutlass.Int64)

def ncclBarrierSession_C_size():
    return cutlass.Int64(_raw_ncclBarrierSession_C_size())


# === ncclDevComm field accessors ===

_raw_ncclDevComm_Rank = _ffi(
    name="ncclDevComm_Rank",
    params_types=[_LLVMPtrType], return_type=cutlass.Int32)

def ncclDevComm_Rank(dev_comm):
    return cutlass.Int32(_raw_ncclDevComm_Rank(_to_ptr(dev_comm)))


_raw_ncclDevComm_NRanks = _ffi(
    name="ncclDevComm_NRanks",
    params_types=[_LLVMPtrType], return_type=cutlass.Int32)

def ncclDevComm_NRanks(dev_comm):
    return cutlass.Int32(_raw_ncclDevComm_NRanks(_to_ptr(dev_comm)))


_raw_ncclDevComm_LsaRank = _ffi(
    name="ncclDevComm_LsaRank",
    params_types=[_LLVMPtrType], return_type=cutlass.Int32)

def ncclDevComm_LsaRank(dev_comm):
    return cutlass.Int32(_raw_ncclDevComm_LsaRank(_to_ptr(dev_comm)))


_raw_ncclDevComm_LsaSize = _ffi(
    name="ncclDevComm_LsaSize",
    params_types=[_LLVMPtrType], return_type=cutlass.Int32)

def ncclDevComm_LsaSize(dev_comm):
    return cutlass.Int32(_raw_ncclDevComm_LsaSize(_to_ptr(dev_comm)))


_raw_ncclDevComm_LsaBarrier = _ffi(
    name="ncclDevComm_LsaBarrier",
    params_types=[_LLVMPtrType], return_type=ncclLsaBarrierHandle)

def ncclDevComm_LsaBarrier(dev_comm):
    return ncclLsaBarrierHandle(_raw_ncclDevComm_LsaBarrier(_to_ptr(dev_comm)))


_raw_ncclDevComm_RailGinBarrier = _ffi(
    name="ncclDevComm_RailGinBarrier",
    params_types=[_LLVMPtrType], return_type=ncclGinBarrierHandle)

def ncclDevComm_RailGinBarrier(dev_comm):
    return ncclGinBarrierHandle(
        _raw_ncclDevComm_RailGinBarrier(_to_ptr(dev_comm)))


_raw_ncclDevComm_HybridLsaBarrier = _ffi(
    name="ncclDevComm_HybridLsaBarrier",
    params_types=[_LLVMPtrType], return_type=ncclLsaBarrierHandle)

def ncclDevComm_HybridLsaBarrier(dev_comm):
    return ncclLsaBarrierHandle(
        _raw_ncclDevComm_HybridLsaBarrier(_to_ptr(dev_comm)))


_raw_ncclDevComm_HybridRailGinBarrier = _ffi(
    name="ncclDevComm_HybridRailGinBarrier",
    params_types=[_LLVMPtrType], return_type=ncclGinBarrierHandle)

def ncclDevComm_HybridRailGinBarrier(dev_comm):
    return ncclGinBarrierHandle(
        _raw_ncclDevComm_HybridRailGinBarrier(_to_ptr(dev_comm)))


_raw_ncclDevComm_WorldGinBarrier = _ffi(
    name="ncclDevComm_WorldGinBarrier",
    params_types=[_LLVMPtrType], return_type=ncclGinBarrierHandle)

def ncclDevComm_WorldGinBarrier(dev_comm):
    return ncclGinBarrierHandle(
        _raw_ncclDevComm_WorldGinBarrier(_to_ptr(dev_comm)))


_raw_ncclDevComm_LsaMultimem = _ffi(
    name="ncclDevComm_LsaMultimem",
    params_types=[_LLVMPtrType], return_type=ncclMultimemHandle)

def ncclDevComm_LsaMultimem(dev_comm):
    return ncclMultimemHandle(
        _raw_ncclDevComm_LsaMultimem(_to_ptr(dev_comm)))
