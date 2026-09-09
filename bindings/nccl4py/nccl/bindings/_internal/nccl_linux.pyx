# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.31.2. Do not modify it directly.



# <<<< PREAMBLE CONTENT >>>>

cdef extern from * nogil:
    """
    #if defined(_MSC_VER) && !defined(__clang__)
        #include <intrin.h>
        static __forceinline int atomic_int_load(int *p) {
            int v = *(int volatile *)p; _ReadBarrier(); return v;
        }
        static __forceinline void atomic_int_store(int *p, int v) {
            _WriteBarrier(); *(int volatile *)p = v;
        }
    #elif defined(__cplusplus)
        /* GCC/Clang __atomic builtins work in any C++ standard without headers */
        static inline int atomic_int_load(int *p) {
            return __atomic_load_n(p, __ATOMIC_ACQUIRE);
        }
        static inline void atomic_int_store(int *p, int v) {
            __atomic_store_n(p, v, __ATOMIC_RELEASE);
        }
    #else
        #include <stdatomic.h>
        static inline int atomic_int_load(int *p) {
            return (int)atomic_load_explicit((atomic_int *)p, memory_order_acquire);
        }
        static inline void atomic_int_store(int *p, int v) {
            atomic_store_explicit((atomic_int *)p, v, memory_order_release);
        }
    #endif

    """
    cdef int _cyb_atomic_int_load "atomic_int_load"(int *p) nogil
    cdef void _cyb_atomic_int_store "atomic_int_store"(int *p, int v) nogil

cdef extern from "<dlfcn.h>":
    void* _cyb_dlsym "dlsym"(void*, const char*) nogil
    const void * _cyb_RTLD_DEFAULT "RTLD_DEFAULT"

cimport cython as _cyb_cython
from libc.stdint cimport (
    int16_t,
    int32_t,
    int64_t,
    int8_t,
    intptr_t,
    uint16_t,
    uint32_t,
    uint64_t,
    uint8_t,
)

import threading as _cyb_threading

cdef int _cyb___py_nccl_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib

cdef extern from "<dlfcn.h>" nogil:
    ctypedef struct Dl_info:
        const char* dli_fname
        void* dli_fbase
        const char* dli_sname
        void* dli_saddr
    int dladdr(const void*, Dl_info*)


###############################################################################
# Wrapper init
###############################################################################


cdef void* __ncclMemAlloc = NULL
cdef void* __ncclMemFree = NULL
cdef void* __ncclGetVersion = NULL
cdef void* __ncclGetUniqueId = NULL
cdef void* __ncclCommInitRankConfig = NULL
cdef void* __ncclCommInitRank = NULL
cdef void* __ncclCommInitAll = NULL
cdef void* __ncclCommFinalize = NULL
cdef void* __ncclCommDestroy = NULL
cdef void* __ncclCommAbort = NULL
cdef void* __ncclCommRevoke = NULL
cdef void* __ncclCommSplit = NULL
cdef void* __ncclCommShrink = NULL
cdef void* __ncclCommGetUniqueId = NULL
cdef void* __ncclCommGrow = NULL
cdef void* __ncclCommInitRankScalable = NULL
cdef void* __ncclGetErrorString = NULL
cdef void* __ncclGetLastError = NULL
cdef void* __ncclCommGetAsyncError = NULL
cdef void* __ncclCommCount = NULL
cdef void* __ncclCommCuDevice = NULL
cdef void* __ncclCommUserRank = NULL
cdef void* __ncclCommRegister = NULL
cdef void* __ncclCommDeregister = NULL
cdef void* __ncclCommSuspend = NULL
cdef void* __ncclCommResume = NULL
cdef void* __ncclCommMemStats = NULL
cdef void* __ncclCommWindowRegister = NULL
cdef void* __ncclCommWindowDeregister = NULL
cdef void* __ncclWinGetUserPtr = NULL
cdef void* __ncclRedOpCreatePreMulSum = NULL
cdef void* __ncclRedOpDestroy = NULL
cdef void* __ncclReduce = NULL
cdef void* __ncclBcast = NULL
cdef void* __ncclBroadcast = NULL
cdef void* __ncclAllReduce = NULL
cdef void* __ncclReduceScatter = NULL
cdef void* __ncclAllGather = NULL
cdef void* __ncclAlltoAll = NULL
cdef void* __ncclGather = NULL
cdef void* __ncclScatter = NULL
cdef void* __ncclAllReduceConfig = NULL
cdef void* __ncclBroadcastConfig = NULL
cdef void* __ncclReduceConfig = NULL
cdef void* __ncclAllGatherConfig = NULL
cdef void* __ncclReduceScatterConfig = NULL
cdef void* __ncclAlltoAllConfig = NULL
cdef void* __ncclGatherConfig = NULL
cdef void* __ncclScatterConfig = NULL
cdef void* __ncclSend = NULL
cdef void* __ncclRecv = NULL
cdef void* __ncclPutSignal = NULL
cdef void* __ncclSignal = NULL
cdef void* __ncclWaitSignal = NULL
cdef void* __ncclGroupStart = NULL
cdef void* __ncclGroupEnd = NULL
cdef void* __ncclGroupSimulateEnd = NULL
cdef void* __ncclParamBind = NULL
cdef void* __ncclParamGetI8 = NULL
cdef void* __ncclParamGetI16 = NULL
cdef void* __ncclParamGetI32 = NULL
cdef void* __ncclParamGetI64 = NULL
cdef void* __ncclParamGetU8 = NULL
cdef void* __ncclParamGetU16 = NULL
cdef void* __ncclParamGetU32 = NULL
cdef void* __ncclParamGetU64 = NULL
cdef void* __ncclParamGetStr = NULL
cdef void* __ncclParamGet = NULL
cdef void* __ncclParamGetParameter = NULL
cdef void* __ncclParamGetAllParameterKeys = NULL
cdef void* __ncclParamDumpAll = NULL
cdef void* __ncclCommQueryProperties = NULL
cdef void* __ncclDevCommCreate = NULL
cdef void* __ncclDevCommDestroy = NULL
cdef void* __ncclGetLsaMultimemDevicePointer = NULL
cdef void* __ncclGetLsaDevicePointer = NULL
cdef void* __ncclGetMultimemDevicePointer = NULL
cdef void* __ncclGetPeerDevicePointer = NULL
cdef void* __ncclGetMultimemDeviceLeInfo = NULL
cdef void* __ncclGetCftDeviceLeInfo = NULL
cdef void* __ncclGetPeerDeviceLeInfo = NULL
cdef void* __ncclTeamWorld = NULL
cdef void* __ncclTeamLsa = NULL
cdef void* __ncclTeamCft = NULL
cdef void* __ncclTeamCftMultimem = NULL
cdef void* __ncclTeamRail = NULL
cdef void* __ncclTeamRankToWorld = NULL
cdef void* __ncclTeamRankToLsa = NULL
cdef void* __ncclLsaBarrierCreateRequirement = NULL
cdef void* __ncclGinBarrierCreateRequirement = NULL
cdef void* __ncclLLA2ACreateRequirement = NULL
cdef void* __ncclLLA2ACalcSlots = NULL

cdef int _init_nccl() except -1 nogil:
    global _cyb___py_nccl_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_nccl_init: return 0

        global __ncclMemAlloc
        __ncclMemAlloc = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclMemAlloc')
        if __ncclMemAlloc == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclMemAlloc = _cyb_dlsym(handle, 'ncclMemAlloc')

        global __ncclMemFree
        __ncclMemFree = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclMemFree')
        if __ncclMemFree == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclMemFree = _cyb_dlsym(handle, 'ncclMemFree')

        global __ncclGetVersion
        __ncclGetVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetVersion')
        if __ncclGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetVersion = _cyb_dlsym(handle, 'ncclGetVersion')

        global __ncclGetUniqueId
        __ncclGetUniqueId = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetUniqueId')
        if __ncclGetUniqueId == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetUniqueId = _cyb_dlsym(handle, 'ncclGetUniqueId')

        global __ncclCommInitRankConfig
        __ncclCommInitRankConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommInitRankConfig')
        if __ncclCommInitRankConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommInitRankConfig = _cyb_dlsym(handle, 'ncclCommInitRankConfig')

        global __ncclCommInitRank
        __ncclCommInitRank = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommInitRank')
        if __ncclCommInitRank == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommInitRank = _cyb_dlsym(handle, 'ncclCommInitRank')

        global __ncclCommInitAll
        __ncclCommInitAll = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommInitAll')
        if __ncclCommInitAll == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommInitAll = _cyb_dlsym(handle, 'ncclCommInitAll')

        global __ncclCommFinalize
        __ncclCommFinalize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommFinalize')
        if __ncclCommFinalize == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommFinalize = _cyb_dlsym(handle, 'ncclCommFinalize')

        global __ncclCommDestroy
        __ncclCommDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommDestroy')
        if __ncclCommDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommDestroy = _cyb_dlsym(handle, 'ncclCommDestroy')

        global __ncclCommAbort
        __ncclCommAbort = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommAbort')
        if __ncclCommAbort == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommAbort = _cyb_dlsym(handle, 'ncclCommAbort')

        global __ncclCommRevoke
        __ncclCommRevoke = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommRevoke')
        if __ncclCommRevoke == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommRevoke = _cyb_dlsym(handle, 'ncclCommRevoke')

        global __ncclCommSplit
        __ncclCommSplit = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommSplit')
        if __ncclCommSplit == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommSplit = _cyb_dlsym(handle, 'ncclCommSplit')

        global __ncclCommShrink
        __ncclCommShrink = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommShrink')
        if __ncclCommShrink == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommShrink = _cyb_dlsym(handle, 'ncclCommShrink')

        global __ncclCommGetUniqueId
        __ncclCommGetUniqueId = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommGetUniqueId')
        if __ncclCommGetUniqueId == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommGetUniqueId = _cyb_dlsym(handle, 'ncclCommGetUniqueId')

        global __ncclCommGrow
        __ncclCommGrow = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommGrow')
        if __ncclCommGrow == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommGrow = _cyb_dlsym(handle, 'ncclCommGrow')

        global __ncclCommInitRankScalable
        __ncclCommInitRankScalable = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommInitRankScalable')
        if __ncclCommInitRankScalable == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommInitRankScalable = _cyb_dlsym(handle, 'ncclCommInitRankScalable')

        global __ncclGetErrorString
        __ncclGetErrorString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetErrorString')
        if __ncclGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetErrorString = _cyb_dlsym(handle, 'ncclGetErrorString')

        global __ncclGetLastError
        __ncclGetLastError = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetLastError')
        if __ncclGetLastError == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetLastError = _cyb_dlsym(handle, 'ncclGetLastError')

        global __ncclCommGetAsyncError
        __ncclCommGetAsyncError = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommGetAsyncError')
        if __ncclCommGetAsyncError == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommGetAsyncError = _cyb_dlsym(handle, 'ncclCommGetAsyncError')

        global __ncclCommCount
        __ncclCommCount = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommCount')
        if __ncclCommCount == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommCount = _cyb_dlsym(handle, 'ncclCommCount')

        global __ncclCommCuDevice
        __ncclCommCuDevice = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommCuDevice')
        if __ncclCommCuDevice == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommCuDevice = _cyb_dlsym(handle, 'ncclCommCuDevice')

        global __ncclCommUserRank
        __ncclCommUserRank = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommUserRank')
        if __ncclCommUserRank == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommUserRank = _cyb_dlsym(handle, 'ncclCommUserRank')

        global __ncclCommRegister
        __ncclCommRegister = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommRegister')
        if __ncclCommRegister == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommRegister = _cyb_dlsym(handle, 'ncclCommRegister')

        global __ncclCommDeregister
        __ncclCommDeregister = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommDeregister')
        if __ncclCommDeregister == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommDeregister = _cyb_dlsym(handle, 'ncclCommDeregister')

        global __ncclCommSuspend
        __ncclCommSuspend = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommSuspend')
        if __ncclCommSuspend == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommSuspend = _cyb_dlsym(handle, 'ncclCommSuspend')

        global __ncclCommResume
        __ncclCommResume = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommResume')
        if __ncclCommResume == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommResume = _cyb_dlsym(handle, 'ncclCommResume')

        global __ncclCommMemStats
        __ncclCommMemStats = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommMemStats')
        if __ncclCommMemStats == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommMemStats = _cyb_dlsym(handle, 'ncclCommMemStats')

        global __ncclCommWindowRegister
        __ncclCommWindowRegister = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommWindowRegister')
        if __ncclCommWindowRegister == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommWindowRegister = _cyb_dlsym(handle, 'ncclCommWindowRegister')

        global __ncclCommWindowDeregister
        __ncclCommWindowDeregister = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommWindowDeregister')
        if __ncclCommWindowDeregister == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommWindowDeregister = _cyb_dlsym(handle, 'ncclCommWindowDeregister')

        global __ncclWinGetUserPtr
        __ncclWinGetUserPtr = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclWinGetUserPtr')
        if __ncclWinGetUserPtr == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclWinGetUserPtr = _cyb_dlsym(handle, 'ncclWinGetUserPtr')

        global __ncclRedOpCreatePreMulSum
        __ncclRedOpCreatePreMulSum = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclRedOpCreatePreMulSum')
        if __ncclRedOpCreatePreMulSum == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclRedOpCreatePreMulSum = _cyb_dlsym(handle, 'ncclRedOpCreatePreMulSum')

        global __ncclRedOpDestroy
        __ncclRedOpDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclRedOpDestroy')
        if __ncclRedOpDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclRedOpDestroy = _cyb_dlsym(handle, 'ncclRedOpDestroy')

        global __ncclReduce
        __ncclReduce = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclReduce')
        if __ncclReduce == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclReduce = _cyb_dlsym(handle, 'ncclReduce')

        global __ncclBcast
        __ncclBcast = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclBcast')
        if __ncclBcast == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclBcast = _cyb_dlsym(handle, 'ncclBcast')

        global __ncclBroadcast
        __ncclBroadcast = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclBroadcast')
        if __ncclBroadcast == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclBroadcast = _cyb_dlsym(handle, 'ncclBroadcast')

        global __ncclAllReduce
        __ncclAllReduce = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclAllReduce')
        if __ncclAllReduce == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclAllReduce = _cyb_dlsym(handle, 'ncclAllReduce')

        global __ncclReduceScatter
        __ncclReduceScatter = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclReduceScatter')
        if __ncclReduceScatter == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclReduceScatter = _cyb_dlsym(handle, 'ncclReduceScatter')

        global __ncclAllGather
        __ncclAllGather = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclAllGather')
        if __ncclAllGather == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclAllGather = _cyb_dlsym(handle, 'ncclAllGather')

        global __ncclAlltoAll
        __ncclAlltoAll = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclAlltoAll')
        if __ncclAlltoAll == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclAlltoAll = _cyb_dlsym(handle, 'ncclAlltoAll')

        global __ncclGather
        __ncclGather = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGather')
        if __ncclGather == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGather = _cyb_dlsym(handle, 'ncclGather')

        global __ncclScatter
        __ncclScatter = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclScatter')
        if __ncclScatter == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclScatter = _cyb_dlsym(handle, 'ncclScatter')

        global __ncclAllReduceConfig
        __ncclAllReduceConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclAllReduceConfig')
        if __ncclAllReduceConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclAllReduceConfig = _cyb_dlsym(handle, 'ncclAllReduceConfig')

        global __ncclBroadcastConfig
        __ncclBroadcastConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclBroadcastConfig')
        if __ncclBroadcastConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclBroadcastConfig = _cyb_dlsym(handle, 'ncclBroadcastConfig')

        global __ncclReduceConfig
        __ncclReduceConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclReduceConfig')
        if __ncclReduceConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclReduceConfig = _cyb_dlsym(handle, 'ncclReduceConfig')

        global __ncclAllGatherConfig
        __ncclAllGatherConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclAllGatherConfig')
        if __ncclAllGatherConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclAllGatherConfig = _cyb_dlsym(handle, 'ncclAllGatherConfig')

        global __ncclReduceScatterConfig
        __ncclReduceScatterConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclReduceScatterConfig')
        if __ncclReduceScatterConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclReduceScatterConfig = _cyb_dlsym(handle, 'ncclReduceScatterConfig')

        global __ncclAlltoAllConfig
        __ncclAlltoAllConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclAlltoAllConfig')
        if __ncclAlltoAllConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclAlltoAllConfig = _cyb_dlsym(handle, 'ncclAlltoAllConfig')

        global __ncclGatherConfig
        __ncclGatherConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGatherConfig')
        if __ncclGatherConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGatherConfig = _cyb_dlsym(handle, 'ncclGatherConfig')

        global __ncclScatterConfig
        __ncclScatterConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclScatterConfig')
        if __ncclScatterConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclScatterConfig = _cyb_dlsym(handle, 'ncclScatterConfig')

        global __ncclSend
        __ncclSend = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclSend')
        if __ncclSend == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclSend = _cyb_dlsym(handle, 'ncclSend')

        global __ncclRecv
        __ncclRecv = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclRecv')
        if __ncclRecv == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclRecv = _cyb_dlsym(handle, 'ncclRecv')

        global __ncclPutSignal
        __ncclPutSignal = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclPutSignal')
        if __ncclPutSignal == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclPutSignal = _cyb_dlsym(handle, 'ncclPutSignal')

        global __ncclSignal
        __ncclSignal = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclSignal')
        if __ncclSignal == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclSignal = _cyb_dlsym(handle, 'ncclSignal')

        global __ncclWaitSignal
        __ncclWaitSignal = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclWaitSignal')
        if __ncclWaitSignal == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclWaitSignal = _cyb_dlsym(handle, 'ncclWaitSignal')

        global __ncclGroupStart
        __ncclGroupStart = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGroupStart')
        if __ncclGroupStart == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGroupStart = _cyb_dlsym(handle, 'ncclGroupStart')

        global __ncclGroupEnd
        __ncclGroupEnd = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGroupEnd')
        if __ncclGroupEnd == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGroupEnd = _cyb_dlsym(handle, 'ncclGroupEnd')

        global __ncclGroupSimulateEnd
        __ncclGroupSimulateEnd = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGroupSimulateEnd')
        if __ncclGroupSimulateEnd == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGroupSimulateEnd = _cyb_dlsym(handle, 'ncclGroupSimulateEnd')

        global __ncclParamBind
        __ncclParamBind = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamBind')
        if __ncclParamBind == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamBind = _cyb_dlsym(handle, 'ncclParamBind')

        global __ncclParamGetI8
        __ncclParamGetI8 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetI8')
        if __ncclParamGetI8 == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetI8 = _cyb_dlsym(handle, 'ncclParamGetI8')

        global __ncclParamGetI16
        __ncclParamGetI16 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetI16')
        if __ncclParamGetI16 == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetI16 = _cyb_dlsym(handle, 'ncclParamGetI16')

        global __ncclParamGetI32
        __ncclParamGetI32 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetI32')
        if __ncclParamGetI32 == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetI32 = _cyb_dlsym(handle, 'ncclParamGetI32')

        global __ncclParamGetI64
        __ncclParamGetI64 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetI64')
        if __ncclParamGetI64 == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetI64 = _cyb_dlsym(handle, 'ncclParamGetI64')

        global __ncclParamGetU8
        __ncclParamGetU8 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetU8')
        if __ncclParamGetU8 == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetU8 = _cyb_dlsym(handle, 'ncclParamGetU8')

        global __ncclParamGetU16
        __ncclParamGetU16 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetU16')
        if __ncclParamGetU16 == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetU16 = _cyb_dlsym(handle, 'ncclParamGetU16')

        global __ncclParamGetU32
        __ncclParamGetU32 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetU32')
        if __ncclParamGetU32 == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetU32 = _cyb_dlsym(handle, 'ncclParamGetU32')

        global __ncclParamGetU64
        __ncclParamGetU64 = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetU64')
        if __ncclParamGetU64 == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetU64 = _cyb_dlsym(handle, 'ncclParamGetU64')

        global __ncclParamGetStr
        __ncclParamGetStr = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetStr')
        if __ncclParamGetStr == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetStr = _cyb_dlsym(handle, 'ncclParamGetStr')

        global __ncclParamGet
        __ncclParamGet = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGet')
        if __ncclParamGet == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGet = _cyb_dlsym(handle, 'ncclParamGet')

        global __ncclParamGetParameter
        __ncclParamGetParameter = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetParameter')
        if __ncclParamGetParameter == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetParameter = _cyb_dlsym(handle, 'ncclParamGetParameter')

        global __ncclParamGetAllParameterKeys
        __ncclParamGetAllParameterKeys = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamGetAllParameterKeys')
        if __ncclParamGetAllParameterKeys == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamGetAllParameterKeys = _cyb_dlsym(handle, 'ncclParamGetAllParameterKeys')

        global __ncclParamDumpAll
        __ncclParamDumpAll = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclParamDumpAll')
        if __ncclParamDumpAll == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclParamDumpAll = _cyb_dlsym(handle, 'ncclParamDumpAll')

        global __ncclCommQueryProperties
        __ncclCommQueryProperties = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclCommQueryProperties')
        if __ncclCommQueryProperties == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommQueryProperties = _cyb_dlsym(handle, 'ncclCommQueryProperties')

        global __ncclDevCommCreate
        __ncclDevCommCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclDevCommCreate')
        if __ncclDevCommCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclDevCommCreate = _cyb_dlsym(handle, 'ncclDevCommCreate')

        global __ncclDevCommDestroy
        __ncclDevCommDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclDevCommDestroy')
        if __ncclDevCommDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclDevCommDestroy = _cyb_dlsym(handle, 'ncclDevCommDestroy')

        global __ncclGetLsaMultimemDevicePointer
        __ncclGetLsaMultimemDevicePointer = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetLsaMultimemDevicePointer')
        if __ncclGetLsaMultimemDevicePointer == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetLsaMultimemDevicePointer = _cyb_dlsym(handle, 'ncclGetLsaMultimemDevicePointer')

        global __ncclGetLsaDevicePointer
        __ncclGetLsaDevicePointer = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetLsaDevicePointer')
        if __ncclGetLsaDevicePointer == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetLsaDevicePointer = _cyb_dlsym(handle, 'ncclGetLsaDevicePointer')

        global __ncclGetMultimemDevicePointer
        __ncclGetMultimemDevicePointer = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetMultimemDevicePointer')
        if __ncclGetMultimemDevicePointer == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetMultimemDevicePointer = _cyb_dlsym(handle, 'ncclGetMultimemDevicePointer')

        global __ncclGetPeerDevicePointer
        __ncclGetPeerDevicePointer = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetPeerDevicePointer')
        if __ncclGetPeerDevicePointer == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetPeerDevicePointer = _cyb_dlsym(handle, 'ncclGetPeerDevicePointer')

        global __ncclGetMultimemDeviceLeInfo
        __ncclGetMultimemDeviceLeInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetMultimemDeviceLeInfo')
        if __ncclGetMultimemDeviceLeInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetMultimemDeviceLeInfo = _cyb_dlsym(handle, 'ncclGetMultimemDeviceLeInfo')

        global __ncclGetCftDeviceLeInfo
        __ncclGetCftDeviceLeInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetCftDeviceLeInfo')
        if __ncclGetCftDeviceLeInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetCftDeviceLeInfo = _cyb_dlsym(handle, 'ncclGetCftDeviceLeInfo')

        global __ncclGetPeerDeviceLeInfo
        __ncclGetPeerDeviceLeInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGetPeerDeviceLeInfo')
        if __ncclGetPeerDeviceLeInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetPeerDeviceLeInfo = _cyb_dlsym(handle, 'ncclGetPeerDeviceLeInfo')

        global __ncclTeamWorld
        __ncclTeamWorld = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclTeamWorld')
        if __ncclTeamWorld == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclTeamWorld = _cyb_dlsym(handle, 'ncclTeamWorld')

        global __ncclTeamLsa
        __ncclTeamLsa = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclTeamLsa')
        if __ncclTeamLsa == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclTeamLsa = _cyb_dlsym(handle, 'ncclTeamLsa')

        global __ncclTeamCft
        __ncclTeamCft = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclTeamCft')
        if __ncclTeamCft == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclTeamCft = _cyb_dlsym(handle, 'ncclTeamCft')

        global __ncclTeamCftMultimem
        __ncclTeamCftMultimem = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclTeamCftMultimem')
        if __ncclTeamCftMultimem == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclTeamCftMultimem = _cyb_dlsym(handle, 'ncclTeamCftMultimem')

        global __ncclTeamRail
        __ncclTeamRail = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclTeamRail')
        if __ncclTeamRail == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclTeamRail = _cyb_dlsym(handle, 'ncclTeamRail')

        global __ncclTeamRankToWorld
        __ncclTeamRankToWorld = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclTeamRankToWorld')
        if __ncclTeamRankToWorld == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclTeamRankToWorld = _cyb_dlsym(handle, 'ncclTeamRankToWorld')

        global __ncclTeamRankToLsa
        __ncclTeamRankToLsa = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclTeamRankToLsa')
        if __ncclTeamRankToLsa == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclTeamRankToLsa = _cyb_dlsym(handle, 'ncclTeamRankToLsa')

        global __ncclLsaBarrierCreateRequirement
        __ncclLsaBarrierCreateRequirement = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclLsaBarrierCreateRequirement')
        if __ncclLsaBarrierCreateRequirement == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclLsaBarrierCreateRequirement = _cyb_dlsym(handle, 'ncclLsaBarrierCreateRequirement')

        global __ncclGinBarrierCreateRequirement
        __ncclGinBarrierCreateRequirement = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclGinBarrierCreateRequirement')
        if __ncclGinBarrierCreateRequirement == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGinBarrierCreateRequirement = _cyb_dlsym(handle, 'ncclGinBarrierCreateRequirement')

        global __ncclLLA2ACreateRequirement
        __ncclLLA2ACreateRequirement = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclLLA2ACreateRequirement')
        if __ncclLLA2ACreateRequirement == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclLLA2ACreateRequirement = _cyb_dlsym(handle, 'ncclLLA2ACreateRequirement')

        global __ncclLLA2ACalcSlots
        __ncclLLA2ACalcSlots = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'ncclLLA2ACalcSlots')
        if __ncclLLA2ACalcSlots == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclLLA2ACalcSlots = _cyb_dlsym(handle, 'ncclLLA2ACalcSlots')

        _cyb_atomic_int_store(<int *>&_cyb___py_nccl_init, 1)
        return 0

cdef inline int _check_or_init_nccl() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_nccl_init):
        return 0

    return _init_nccl()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_nccl()
    cdef dict data = {}
    global __ncclMemAlloc
    data["__ncclMemAlloc"] = <intptr_t>__ncclMemAlloc

    global __ncclMemFree
    data["__ncclMemFree"] = <intptr_t>__ncclMemFree

    global __ncclGetVersion
    data["__ncclGetVersion"] = <intptr_t>__ncclGetVersion

    global __ncclGetUniqueId
    data["__ncclGetUniqueId"] = <intptr_t>__ncclGetUniqueId

    global __ncclCommInitRankConfig
    data["__ncclCommInitRankConfig"] = <intptr_t>__ncclCommInitRankConfig

    global __ncclCommInitRank
    data["__ncclCommInitRank"] = <intptr_t>__ncclCommInitRank

    global __ncclCommInitAll
    data["__ncclCommInitAll"] = <intptr_t>__ncclCommInitAll

    global __ncclCommFinalize
    data["__ncclCommFinalize"] = <intptr_t>__ncclCommFinalize

    global __ncclCommDestroy
    data["__ncclCommDestroy"] = <intptr_t>__ncclCommDestroy

    global __ncclCommAbort
    data["__ncclCommAbort"] = <intptr_t>__ncclCommAbort

    global __ncclCommRevoke
    data["__ncclCommRevoke"] = <intptr_t>__ncclCommRevoke

    global __ncclCommSplit
    data["__ncclCommSplit"] = <intptr_t>__ncclCommSplit

    global __ncclCommShrink
    data["__ncclCommShrink"] = <intptr_t>__ncclCommShrink

    global __ncclCommGetUniqueId
    data["__ncclCommGetUniqueId"] = <intptr_t>__ncclCommGetUniqueId

    global __ncclCommGrow
    data["__ncclCommGrow"] = <intptr_t>__ncclCommGrow

    global __ncclCommInitRankScalable
    data["__ncclCommInitRankScalable"] = <intptr_t>__ncclCommInitRankScalable

    global __ncclGetErrorString
    data["__ncclGetErrorString"] = <intptr_t>__ncclGetErrorString

    global __ncclGetLastError
    data["__ncclGetLastError"] = <intptr_t>__ncclGetLastError

    global __ncclCommGetAsyncError
    data["__ncclCommGetAsyncError"] = <intptr_t>__ncclCommGetAsyncError

    global __ncclCommCount
    data["__ncclCommCount"] = <intptr_t>__ncclCommCount

    global __ncclCommCuDevice
    data["__ncclCommCuDevice"] = <intptr_t>__ncclCommCuDevice

    global __ncclCommUserRank
    data["__ncclCommUserRank"] = <intptr_t>__ncclCommUserRank

    global __ncclCommRegister
    data["__ncclCommRegister"] = <intptr_t>__ncclCommRegister

    global __ncclCommDeregister
    data["__ncclCommDeregister"] = <intptr_t>__ncclCommDeregister

    global __ncclCommSuspend
    data["__ncclCommSuspend"] = <intptr_t>__ncclCommSuspend

    global __ncclCommResume
    data["__ncclCommResume"] = <intptr_t>__ncclCommResume

    global __ncclCommMemStats
    data["__ncclCommMemStats"] = <intptr_t>__ncclCommMemStats

    global __ncclCommWindowRegister
    data["__ncclCommWindowRegister"] = <intptr_t>__ncclCommWindowRegister

    global __ncclCommWindowDeregister
    data["__ncclCommWindowDeregister"] = <intptr_t>__ncclCommWindowDeregister

    global __ncclWinGetUserPtr
    data["__ncclWinGetUserPtr"] = <intptr_t>__ncclWinGetUserPtr

    global __ncclRedOpCreatePreMulSum
    data["__ncclRedOpCreatePreMulSum"] = <intptr_t>__ncclRedOpCreatePreMulSum

    global __ncclRedOpDestroy
    data["__ncclRedOpDestroy"] = <intptr_t>__ncclRedOpDestroy

    global __ncclReduce
    data["__ncclReduce"] = <intptr_t>__ncclReduce

    global __ncclBcast
    data["__ncclBcast"] = <intptr_t>__ncclBcast

    global __ncclBroadcast
    data["__ncclBroadcast"] = <intptr_t>__ncclBroadcast

    global __ncclAllReduce
    data["__ncclAllReduce"] = <intptr_t>__ncclAllReduce

    global __ncclReduceScatter
    data["__ncclReduceScatter"] = <intptr_t>__ncclReduceScatter

    global __ncclAllGather
    data["__ncclAllGather"] = <intptr_t>__ncclAllGather

    global __ncclAlltoAll
    data["__ncclAlltoAll"] = <intptr_t>__ncclAlltoAll

    global __ncclGather
    data["__ncclGather"] = <intptr_t>__ncclGather

    global __ncclScatter
    data["__ncclScatter"] = <intptr_t>__ncclScatter

    global __ncclAllReduceConfig
    data["__ncclAllReduceConfig"] = <intptr_t>__ncclAllReduceConfig

    global __ncclBroadcastConfig
    data["__ncclBroadcastConfig"] = <intptr_t>__ncclBroadcastConfig

    global __ncclReduceConfig
    data["__ncclReduceConfig"] = <intptr_t>__ncclReduceConfig

    global __ncclAllGatherConfig
    data["__ncclAllGatherConfig"] = <intptr_t>__ncclAllGatherConfig

    global __ncclReduceScatterConfig
    data["__ncclReduceScatterConfig"] = <intptr_t>__ncclReduceScatterConfig

    global __ncclAlltoAllConfig
    data["__ncclAlltoAllConfig"] = <intptr_t>__ncclAlltoAllConfig

    global __ncclGatherConfig
    data["__ncclGatherConfig"] = <intptr_t>__ncclGatherConfig

    global __ncclScatterConfig
    data["__ncclScatterConfig"] = <intptr_t>__ncclScatterConfig

    global __ncclSend
    data["__ncclSend"] = <intptr_t>__ncclSend

    global __ncclRecv
    data["__ncclRecv"] = <intptr_t>__ncclRecv

    global __ncclPutSignal
    data["__ncclPutSignal"] = <intptr_t>__ncclPutSignal

    global __ncclSignal
    data["__ncclSignal"] = <intptr_t>__ncclSignal

    global __ncclWaitSignal
    data["__ncclWaitSignal"] = <intptr_t>__ncclWaitSignal

    global __ncclGroupStart
    data["__ncclGroupStart"] = <intptr_t>__ncclGroupStart

    global __ncclGroupEnd
    data["__ncclGroupEnd"] = <intptr_t>__ncclGroupEnd

    global __ncclGroupSimulateEnd
    data["__ncclGroupSimulateEnd"] = <intptr_t>__ncclGroupSimulateEnd

    global __ncclParamBind
    data["__ncclParamBind"] = <intptr_t>__ncclParamBind

    global __ncclParamGetI8
    data["__ncclParamGetI8"] = <intptr_t>__ncclParamGetI8

    global __ncclParamGetI16
    data["__ncclParamGetI16"] = <intptr_t>__ncclParamGetI16

    global __ncclParamGetI32
    data["__ncclParamGetI32"] = <intptr_t>__ncclParamGetI32

    global __ncclParamGetI64
    data["__ncclParamGetI64"] = <intptr_t>__ncclParamGetI64

    global __ncclParamGetU8
    data["__ncclParamGetU8"] = <intptr_t>__ncclParamGetU8

    global __ncclParamGetU16
    data["__ncclParamGetU16"] = <intptr_t>__ncclParamGetU16

    global __ncclParamGetU32
    data["__ncclParamGetU32"] = <intptr_t>__ncclParamGetU32

    global __ncclParamGetU64
    data["__ncclParamGetU64"] = <intptr_t>__ncclParamGetU64

    global __ncclParamGetStr
    data["__ncclParamGetStr"] = <intptr_t>__ncclParamGetStr

    global __ncclParamGet
    data["__ncclParamGet"] = <intptr_t>__ncclParamGet

    global __ncclParamGetParameter
    data["__ncclParamGetParameter"] = <intptr_t>__ncclParamGetParameter

    global __ncclParamGetAllParameterKeys
    data["__ncclParamGetAllParameterKeys"] = <intptr_t>__ncclParamGetAllParameterKeys

    global __ncclParamDumpAll
    data["__ncclParamDumpAll"] = <intptr_t>__ncclParamDumpAll

    global __ncclCommQueryProperties
    data["__ncclCommQueryProperties"] = <intptr_t>__ncclCommQueryProperties

    global __ncclDevCommCreate
    data["__ncclDevCommCreate"] = <intptr_t>__ncclDevCommCreate

    global __ncclDevCommDestroy
    data["__ncclDevCommDestroy"] = <intptr_t>__ncclDevCommDestroy

    global __ncclGetLsaMultimemDevicePointer
    data["__ncclGetLsaMultimemDevicePointer"] = <intptr_t>__ncclGetLsaMultimemDevicePointer

    global __ncclGetLsaDevicePointer
    data["__ncclGetLsaDevicePointer"] = <intptr_t>__ncclGetLsaDevicePointer

    global __ncclGetMultimemDevicePointer
    data["__ncclGetMultimemDevicePointer"] = <intptr_t>__ncclGetMultimemDevicePointer

    global __ncclGetPeerDevicePointer
    data["__ncclGetPeerDevicePointer"] = <intptr_t>__ncclGetPeerDevicePointer

    global __ncclGetMultimemDeviceLeInfo
    data["__ncclGetMultimemDeviceLeInfo"] = <intptr_t>__ncclGetMultimemDeviceLeInfo

    global __ncclGetCftDeviceLeInfo
    data["__ncclGetCftDeviceLeInfo"] = <intptr_t>__ncclGetCftDeviceLeInfo

    global __ncclGetPeerDeviceLeInfo
    data["__ncclGetPeerDeviceLeInfo"] = <intptr_t>__ncclGetPeerDeviceLeInfo

    global __ncclTeamWorld
    data["__ncclTeamWorld"] = <intptr_t>__ncclTeamWorld

    global __ncclTeamLsa
    data["__ncclTeamLsa"] = <intptr_t>__ncclTeamLsa

    global __ncclTeamCft
    data["__ncclTeamCft"] = <intptr_t>__ncclTeamCft

    global __ncclTeamCftMultimem
    data["__ncclTeamCftMultimem"] = <intptr_t>__ncclTeamCftMultimem

    global __ncclTeamRail
    data["__ncclTeamRail"] = <intptr_t>__ncclTeamRail

    global __ncclTeamRankToWorld
    data["__ncclTeamRankToWorld"] = <intptr_t>__ncclTeamRankToWorld

    global __ncclTeamRankToLsa
    data["__ncclTeamRankToLsa"] = <intptr_t>__ncclTeamRankToLsa

    global __ncclLsaBarrierCreateRequirement
    data["__ncclLsaBarrierCreateRequirement"] = <intptr_t>__ncclLsaBarrierCreateRequirement

    global __ncclGinBarrierCreateRequirement
    data["__ncclGinBarrierCreateRequirement"] = <intptr_t>__ncclGinBarrierCreateRequirement

    global __ncclLLA2ACreateRequirement
    data["__ncclLLA2ACreateRequirement"] = <intptr_t>__ncclLLA2ACreateRequirement

    global __ncclLLA2ACalcSlots
    data["__ncclLLA2ACalcSlots"] = <intptr_t>__ncclLLA2ACalcSlots
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("nccl")._handle_uint
    return <void*>handle


cdef object __nccl_loaded_so_path = None


cpdef object _inspect_loaded_library_path():
    import os
    # Path of the .so backing the loaded symbols, via dladdr() on a
    # resolved entry point. None if it cannot be determined.
    global __nccl_loaded_so_path
    if __nccl_loaded_so_path is not None:
        return __nccl_loaded_so_path

    cdef dict ptrs = _inspect_function_pointers()
    # Any resolved symbol maps to the same .so.
    cdef intptr_t addr = 0
    for value in ptrs.values():
        if value:
            addr = value
            break

    cdef Dl_info info
    if addr == 0:
        return None
    if dladdr(<void*>addr, &info) == 0 or info.dli_fname == NULL:
        return None
    __nccl_loaded_so_path = os.fsdecode(<bytes>info.dli_fname)
    return __nccl_loaded_so_path


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t _ncclMemAlloc(void** ptr, size_t size) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclMemAlloc
    _check_or_init_nccl()
    if __ncclMemAlloc == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclMemAlloc is not found")
    return (<ncclResult_t (*)(void**, size_t) noexcept nogil>__ncclMemAlloc)(
        ptr, size)


cdef ncclResult_t _ncclMemFree(void* ptr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclMemFree
    _check_or_init_nccl()
    if __ncclMemFree == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclMemFree is not found")
    return (<ncclResult_t (*)(void*) noexcept nogil>__ncclMemFree)(
        ptr)


cdef ncclResult_t _ncclGetVersion(int* version) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetVersion
    _check_or_init_nccl()
    if __ncclGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetVersion is not found")
    return (<ncclResult_t (*)(int*) noexcept nogil>__ncclGetVersion)(
        version)


cdef ncclResult_t _ncclGetUniqueId(ncclUniqueId* uniqueId) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetUniqueId
    _check_or_init_nccl()
    if __ncclGetUniqueId == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetUniqueId is not found")
    return (<ncclResult_t (*)(ncclUniqueId*) noexcept nogil>__ncclGetUniqueId)(
        uniqueId)


cdef ncclResult_t _ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommInitRankConfig
    _check_or_init_nccl()
    if __ncclCommInitRankConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommInitRankConfig is not found")
    return (<ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*) noexcept nogil>__ncclCommInitRankConfig)(
        comm, nranks, commId, rank, config)


cdef ncclResult_t _ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommInitRank
    _check_or_init_nccl()
    if __ncclCommInitRank == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommInitRank is not found")
    return (<ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int) noexcept nogil>__ncclCommInitRank)(
        comm, nranks, commId, rank)


cdef ncclResult_t _ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommInitAll
    _check_or_init_nccl()
    if __ncclCommInitAll == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommInitAll is not found")
    return (<ncclResult_t (*)(ncclComm_t*, int, const int*) noexcept nogil>__ncclCommInitAll)(
        comm, ndev, devlist)


cdef ncclResult_t _ncclCommFinalize(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommFinalize
    _check_or_init_nccl()
    if __ncclCommFinalize == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommFinalize is not found")
    return (<ncclResult_t (*)(ncclComm_t) noexcept nogil>__ncclCommFinalize)(
        comm)


cdef ncclResult_t _ncclCommDestroy(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommDestroy
    _check_or_init_nccl()
    if __ncclCommDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommDestroy is not found")
    return (<ncclResult_t (*)(ncclComm_t) noexcept nogil>__ncclCommDestroy)(
        comm)


cdef ncclResult_t _ncclCommAbort(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommAbort
    _check_or_init_nccl()
    if __ncclCommAbort == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommAbort is not found")
    return (<ncclResult_t (*)(ncclComm_t) noexcept nogil>__ncclCommAbort)(
        comm)


cdef ncclResult_t _ncclCommRevoke(ncclComm_t comm, int revokeFlags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommRevoke
    _check_or_init_nccl()
    if __ncclCommRevoke == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommRevoke is not found")
    return (<ncclResult_t (*)(ncclComm_t, int) noexcept nogil>__ncclCommRevoke)(
        comm, revokeFlags)


cdef ncclResult_t _ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommSplit
    _check_or_init_nccl()
    if __ncclCommSplit == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommSplit is not found")
    return (<ncclResult_t (*)(ncclComm_t, int, int, ncclComm_t*, ncclConfig_t*) noexcept nogil>__ncclCommSplit)(
        comm, color, key, newcomm, config)


cdef ncclResult_t _ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommShrink
    _check_or_init_nccl()
    if __ncclCommShrink == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommShrink is not found")
    return (<ncclResult_t (*)(ncclComm_t, int*, int, ncclComm_t*, ncclConfig_t*, int) noexcept nogil>__ncclCommShrink)(
        comm, excludeRanksList, excludeRanksCount, newcomm, config, shrinkFlags)


cdef ncclResult_t _ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommGetUniqueId
    _check_or_init_nccl()
    if __ncclCommGetUniqueId == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommGetUniqueId is not found")
    return (<ncclResult_t (*)(ncclComm_t, ncclUniqueId*) noexcept nogil>__ncclCommGetUniqueId)(
        comm, uniqueId)


cdef ncclResult_t _ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommGrow
    _check_or_init_nccl()
    if __ncclCommGrow == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommGrow is not found")
    return (<ncclResult_t (*)(ncclComm_t, int, const ncclUniqueId*, int, ncclComm_t*, ncclConfig_t*) noexcept nogil>__ncclCommGrow)(
        comm, nRanks, uniqueId, rank, newcomm, config)


cdef ncclResult_t _ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommInitRankScalable
    _check_or_init_nccl()
    if __ncclCommInitRankScalable == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommInitRankScalable is not found")
    return (<ncclResult_t (*)(ncclComm_t*, int, int, int, ncclUniqueId*, ncclConfig_t*) noexcept nogil>__ncclCommInitRankScalable)(
        newcomm, nranks, myrank, nId, commIds, config)


cdef const char* _ncclGetErrorString(ncclResult_t result) except?NULL nogil:
    global __ncclGetErrorString
    _check_or_init_nccl()
    if __ncclGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetErrorString is not found")
    return (<const char* (*)(ncclResult_t) noexcept nogil>__ncclGetErrorString)(
        result)


cdef const char* _ncclGetLastError(ncclComm_t comm) except?NULL nogil:
    global __ncclGetLastError
    _check_or_init_nccl()
    if __ncclGetLastError == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetLastError is not found")
    return (<const char* (*)(ncclComm_t) noexcept nogil>__ncclGetLastError)(
        comm)


cdef ncclResult_t _ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommGetAsyncError
    _check_or_init_nccl()
    if __ncclCommGetAsyncError == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommGetAsyncError is not found")
    return (<ncclResult_t (*)(ncclComm_t, ncclResult_t*) noexcept nogil>__ncclCommGetAsyncError)(
        comm, asyncError)


cdef ncclResult_t _ncclCommCount(const ncclComm_t comm, int* count) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommCount
    _check_or_init_nccl()
    if __ncclCommCount == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommCount is not found")
    return (<ncclResult_t (*)(const ncclComm_t, int*) noexcept nogil>__ncclCommCount)(
        comm, count)


cdef ncclResult_t _ncclCommCuDevice(const ncclComm_t comm, int* device) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommCuDevice
    _check_or_init_nccl()
    if __ncclCommCuDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommCuDevice is not found")
    return (<ncclResult_t (*)(const ncclComm_t, int*) noexcept nogil>__ncclCommCuDevice)(
        comm, device)


cdef ncclResult_t _ncclCommUserRank(const ncclComm_t comm, int* rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommUserRank
    _check_or_init_nccl()
    if __ncclCommUserRank == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommUserRank is not found")
    return (<ncclResult_t (*)(const ncclComm_t, int*) noexcept nogil>__ncclCommUserRank)(
        comm, rank)


cdef ncclResult_t _ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommRegister
    _check_or_init_nccl()
    if __ncclCommRegister == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommRegister is not found")
    return (<ncclResult_t (*)(const ncclComm_t, void*, size_t, void**) noexcept nogil>__ncclCommRegister)(
        comm, buff, size, handle)


cdef ncclResult_t _ncclCommDeregister(const ncclComm_t comm, void* handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommDeregister
    _check_or_init_nccl()
    if __ncclCommDeregister == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommDeregister is not found")
    return (<ncclResult_t (*)(const ncclComm_t, void*) noexcept nogil>__ncclCommDeregister)(
        comm, handle)


cdef ncclResult_t _ncclCommSuspend(ncclComm_t comm, int flags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommSuspend
    _check_or_init_nccl()
    if __ncclCommSuspend == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommSuspend is not found")
    return (<ncclResult_t (*)(ncclComm_t, int) noexcept nogil>__ncclCommSuspend)(
        comm, flags)


cdef ncclResult_t _ncclCommResume(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommResume
    _check_or_init_nccl()
    if __ncclCommResume == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommResume is not found")
    return (<ncclResult_t (*)(ncclComm_t) noexcept nogil>__ncclCommResume)(
        comm)


cdef ncclResult_t _ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommMemStats
    _check_or_init_nccl()
    if __ncclCommMemStats == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommMemStats is not found")
    return (<ncclResult_t (*)(ncclComm_t, ncclCommMemStat_t, uint64_t*) noexcept nogil>__ncclCommMemStats)(
        comm, stat, value)


cdef ncclResult_t _ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommWindowRegister
    _check_or_init_nccl()
    if __ncclCommWindowRegister == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommWindowRegister is not found")
    return (<ncclResult_t (*)(ncclComm_t, void*, size_t, ncclWindow_t*, int) noexcept nogil>__ncclCommWindowRegister)(
        comm, buff, size, win, winFlags)


cdef ncclResult_t _ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommWindowDeregister
    _check_or_init_nccl()
    if __ncclCommWindowDeregister == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommWindowDeregister is not found")
    return (<ncclResult_t (*)(ncclComm_t, ncclWindow_t) noexcept nogil>__ncclCommWindowDeregister)(
        comm, win)


cdef ncclResult_t _ncclWinGetUserPtr(ncclComm_t comm, ncclWindow_t win, void** outUserPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclWinGetUserPtr
    _check_or_init_nccl()
    if __ncclWinGetUserPtr == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclWinGetUserPtr is not found")
    return (<ncclResult_t (*)(ncclComm_t, ncclWindow_t, void**) noexcept nogil>__ncclWinGetUserPtr)(
        comm, win, outUserPtr)


cdef ncclResult_t _ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclRedOpCreatePreMulSum
    _check_or_init_nccl()
    if __ncclRedOpCreatePreMulSum == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclRedOpCreatePreMulSum is not found")
    return (<ncclResult_t (*)(ncclRedOp_t*, void*, ncclDataType_t, ncclScalarResidence_t, ncclComm_t) noexcept nogil>__ncclRedOpCreatePreMulSum)(
        op, scalar, datatype, residence, comm)


cdef ncclResult_t _ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclRedOpDestroy
    _check_or_init_nccl()
    if __ncclRedOpDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclRedOpDestroy is not found")
    return (<ncclResult_t (*)(ncclRedOp_t, ncclComm_t) noexcept nogil>__ncclRedOpDestroy)(
        op, comm)


cdef ncclResult_t _ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclReduce
    _check_or_init_nccl()
    if __ncclReduce == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclReduce is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t) noexcept nogil>__ncclReduce)(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream)


cdef ncclResult_t _ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclBcast
    _check_or_init_nccl()
    if __ncclBcast == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclBcast is not found")
    return (<ncclResult_t (*)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) noexcept nogil>__ncclBcast)(
        buff, count, datatype, root, comm, stream)


cdef ncclResult_t _ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclBroadcast
    _check_or_init_nccl()
    if __ncclBroadcast == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclBroadcast is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) noexcept nogil>__ncclBroadcast)(
        sendbuff, recvbuff, count, datatype, root, comm, stream)


cdef ncclResult_t _ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclAllReduce
    _check_or_init_nccl()
    if __ncclAllReduce == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclAllReduce is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) noexcept nogil>__ncclAllReduce)(
        sendbuff, recvbuff, count, datatype, op, comm, stream)


cdef ncclResult_t _ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclReduceScatter
    _check_or_init_nccl()
    if __ncclReduceScatter == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclReduceScatter is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) noexcept nogil>__ncclReduceScatter)(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream)


cdef ncclResult_t _ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclAllGather
    _check_or_init_nccl()
    if __ncclAllGather == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclAllGather is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) noexcept nogil>__ncclAllGather)(
        sendbuff, recvbuff, sendcount, datatype, comm, stream)


cdef ncclResult_t _ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclAlltoAll
    _check_or_init_nccl()
    if __ncclAlltoAll == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclAlltoAll is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) noexcept nogil>__ncclAlltoAll)(
        sendbuff, recvbuff, count, datatype, comm, stream)


cdef ncclResult_t _ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGather
    _check_or_init_nccl()
    if __ncclGather == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGather is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) noexcept nogil>__ncclGather)(
        sendbuff, recvbuff, count, datatype, root, comm, stream)


cdef ncclResult_t _ncclScatter(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclScatter
    _check_or_init_nccl()
    if __ncclScatter == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclScatter is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) noexcept nogil>__ncclScatter)(
        sendbuff, recvbuff, count, datatype, root, comm, stream)


cdef ncclResult_t _ncclAllReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclAllReduceConfig
    _check_or_init_nccl()
    if __ncclAllReduceConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclAllReduceConfig is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t, const ncclCollConfig_t*) noexcept nogil>__ncclAllReduceConfig)(
        sendbuff, recvbuff, count, datatype, op, comm, stream, config)


cdef ncclResult_t _ncclBroadcastConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclBroadcastConfig
    _check_or_init_nccl()
    if __ncclBroadcastConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclBroadcastConfig is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t, const ncclCollConfig_t*) noexcept nogil>__ncclBroadcastConfig)(
        sendbuff, recvbuff, count, datatype, root, comm, stream, config)


cdef ncclResult_t _ncclReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclReduceConfig
    _check_or_init_nccl()
    if __ncclReduceConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclReduceConfig is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t, const ncclCollConfig_t*) noexcept nogil>__ncclReduceConfig)(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream, config)


cdef ncclResult_t _ncclAllGatherConfig(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclAllGatherConfig
    _check_or_init_nccl()
    if __ncclAllGatherConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclAllGatherConfig is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t, const ncclCollConfig_t*) noexcept nogil>__ncclAllGatherConfig)(
        sendbuff, recvbuff, sendcount, datatype, comm, stream, config)


cdef ncclResult_t _ncclReduceScatterConfig(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclReduceScatterConfig
    _check_or_init_nccl()
    if __ncclReduceScatterConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclReduceScatterConfig is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t, const ncclCollConfig_t*) noexcept nogil>__ncclReduceScatterConfig)(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream, config)


cdef ncclResult_t _ncclAlltoAllConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclAlltoAllConfig
    _check_or_init_nccl()
    if __ncclAlltoAllConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclAlltoAllConfig is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t, const ncclCollConfig_t*) noexcept nogil>__ncclAlltoAllConfig)(
        sendbuff, recvbuff, count, datatype, comm, stream, config)


cdef ncclResult_t _ncclGatherConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGatherConfig
    _check_or_init_nccl()
    if __ncclGatherConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGatherConfig is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t, const ncclCollConfig_t*) noexcept nogil>__ncclGatherConfig)(
        sendbuff, recvbuff, count, datatype, root, comm, stream, config)


cdef ncclResult_t _ncclScatterConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclScatterConfig
    _check_or_init_nccl()
    if __ncclScatterConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclScatterConfig is not found")
    return (<ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t, const ncclCollConfig_t*) noexcept nogil>__ncclScatterConfig)(
        sendbuff, recvbuff, count, datatype, root, comm, stream, config)


cdef ncclResult_t _ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclSend
    _check_or_init_nccl()
    if __ncclSend == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclSend is not found")
    return (<ncclResult_t (*)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) noexcept nogil>__ncclSend)(
        sendbuff, count, datatype, peer, comm, stream)


cdef ncclResult_t _ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclRecv
    _check_or_init_nccl()
    if __ncclRecv == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclRecv is not found")
    return (<ncclResult_t (*)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) noexcept nogil>__ncclRecv)(
        recvbuff, count, datatype, peer, comm, stream)


cdef ncclResult_t _ncclPutSignal(const void* localbuff, size_t count, ncclDataType_t datatype, int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclPutSignal
    _check_or_init_nccl()
    if __ncclPutSignal == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclPutSignal is not found")
    return (<ncclResult_t (*)(const void*, size_t, ncclDataType_t, int, ncclWindow_t, size_t, int, int, unsigned int, ncclComm_t, cudaStream_t) noexcept nogil>__ncclPutSignal)(
        localbuff, count, datatype, peer, peerWin, peerWinOffset, sigIdx, ctx, flags, comm, stream)


cdef ncclResult_t _ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclSignal
    _check_or_init_nccl()
    if __ncclSignal == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclSignal is not found")
    return (<ncclResult_t (*)(int, int, int, unsigned int, ncclComm_t, cudaStream_t) noexcept nogil>__ncclSignal)(
        peer, sigIdx, ctx, flags, comm, stream)


cdef ncclResult_t _ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclWaitSignal
    _check_or_init_nccl()
    if __ncclWaitSignal == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclWaitSignal is not found")
    return (<ncclResult_t (*)(int, ncclWaitSignalDesc_t*, ncclComm_t, cudaStream_t) noexcept nogil>__ncclWaitSignal)(
        nDesc, signalDescs, comm, stream)


cdef ncclResult_t _ncclGroupStart() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGroupStart
    _check_or_init_nccl()
    if __ncclGroupStart == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGroupStart is not found")
    return (<ncclResult_t (*)() noexcept nogil>__ncclGroupStart)(
        )


cdef ncclResult_t _ncclGroupEnd() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGroupEnd
    _check_or_init_nccl()
    if __ncclGroupEnd == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGroupEnd is not found")
    return (<ncclResult_t (*)() noexcept nogil>__ncclGroupEnd)(
        )


cdef ncclResult_t _ncclGroupSimulateEnd(ncclSimInfo_t* simInfo) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGroupSimulateEnd
    _check_or_init_nccl()
    if __ncclGroupSimulateEnd == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGroupSimulateEnd is not found")
    return (<ncclResult_t (*)(ncclSimInfo_t*) noexcept nogil>__ncclGroupSimulateEnd)(
        simInfo)


cdef ncclResult_t _ncclParamBind(ncclParamHandle_t* out, const char* key) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamBind
    _check_or_init_nccl()
    if __ncclParamBind == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamBind is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t*, const char*) noexcept nogil>__ncclParamBind)(
        out, key)


cdef ncclResult_t _ncclParamGetI8(ncclParamHandle_t h, int8_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetI8
    _check_or_init_nccl()
    if __ncclParamGetI8 == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetI8 is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t, int8_t*) noexcept nogil>__ncclParamGetI8)(
        h, out)


cdef ncclResult_t _ncclParamGetI16(ncclParamHandle_t h, int16_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetI16
    _check_or_init_nccl()
    if __ncclParamGetI16 == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetI16 is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t, int16_t*) noexcept nogil>__ncclParamGetI16)(
        h, out)


cdef ncclResult_t _ncclParamGetI32(ncclParamHandle_t h, int32_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetI32
    _check_or_init_nccl()
    if __ncclParamGetI32 == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetI32 is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t, int32_t*) noexcept nogil>__ncclParamGetI32)(
        h, out)


cdef ncclResult_t _ncclParamGetI64(ncclParamHandle_t h, int64_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetI64
    _check_or_init_nccl()
    if __ncclParamGetI64 == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetI64 is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t, int64_t*) noexcept nogil>__ncclParamGetI64)(
        h, out)


cdef ncclResult_t _ncclParamGetU8(ncclParamHandle_t h, uint8_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetU8
    _check_or_init_nccl()
    if __ncclParamGetU8 == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetU8 is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t, uint8_t*) noexcept nogil>__ncclParamGetU8)(
        h, out)


cdef ncclResult_t _ncclParamGetU16(ncclParamHandle_t h, uint16_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetU16
    _check_or_init_nccl()
    if __ncclParamGetU16 == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetU16 is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t, uint16_t*) noexcept nogil>__ncclParamGetU16)(
        h, out)


cdef ncclResult_t _ncclParamGetU32(ncclParamHandle_t h, uint32_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetU32
    _check_or_init_nccl()
    if __ncclParamGetU32 == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetU32 is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t, uint32_t*) noexcept nogil>__ncclParamGetU32)(
        h, out)


cdef ncclResult_t _ncclParamGetU64(ncclParamHandle_t h, uint64_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetU64
    _check_or_init_nccl()
    if __ncclParamGetU64 == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetU64 is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t, uint64_t*) noexcept nogil>__ncclParamGetU64)(
        h, out)


cdef ncclResult_t _ncclParamGetStr(ncclParamHandle_t h, const char** out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetStr
    _check_or_init_nccl()
    if __ncclParamGetStr == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetStr is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t, const char**) noexcept nogil>__ncclParamGetStr)(
        h, out)


cdef ncclResult_t _ncclParamGet(ncclParamHandle_t h, void* out, int maxLen, int* len) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGet
    _check_or_init_nccl()
    if __ncclParamGet == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGet is not found")
    return (<ncclResult_t (*)(ncclParamHandle_t, void*, int, int*) noexcept nogil>__ncclParamGet)(
        h, out, maxLen, len)


cdef ncclResult_t _ncclParamGetParameter(const char* key, const char** value, int* valueLen) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetParameter
    _check_or_init_nccl()
    if __ncclParamGetParameter == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetParameter is not found")
    return (<ncclResult_t (*)(const char*, const char**, int*) noexcept nogil>__ncclParamGetParameter)(
        key, value, valueLen)


cdef ncclResult_t _ncclParamGetAllParameterKeys(const char*** table, int* tableLen) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclParamGetAllParameterKeys
    _check_or_init_nccl()
    if __ncclParamGetAllParameterKeys == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamGetAllParameterKeys is not found")
    return (<ncclResult_t (*)(const char***, int*) noexcept nogil>__ncclParamGetAllParameterKeys)(
        table, tableLen)


@_cyb_cython.show_performance_hints(False)
cdef void _ncclParamDumpAll() except* nogil:
    global __ncclParamDumpAll
    _check_or_init_nccl()
    if __ncclParamDumpAll == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclParamDumpAll is not found")
    (<void (*)() noexcept nogil>__ncclParamDumpAll)(
        )


cdef ncclResult_t _ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommQueryProperties
    _check_or_init_nccl()
    if __ncclCommQueryProperties == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommQueryProperties is not found")
    return (<ncclResult_t (*)(ncclComm_t, ncclCommProperties_t*) noexcept nogil>__ncclCommQueryProperties)(
        comm, props)


cdef ncclResult_t _ncclDevCommCreate(ncclComm_t comm, const ncclDevCommRequirements_t* reqs, ncclDevComm_t* outDevComm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclDevCommCreate
    _check_or_init_nccl()
    if __ncclDevCommCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclDevCommCreate is not found")
    return (<ncclResult_t (*)(ncclComm_t, const ncclDevCommRequirements_t*, ncclDevComm_t*) noexcept nogil>__ncclDevCommCreate)(
        comm, reqs, outDevComm)


cdef ncclResult_t _ncclDevCommDestroy(ncclComm_t comm, const ncclDevComm_t* devComm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclDevCommDestroy
    _check_or_init_nccl()
    if __ncclDevCommDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclDevCommDestroy is not found")
    return (<ncclResult_t (*)(ncclComm_t, const ncclDevComm_t*) noexcept nogil>__ncclDevCommDestroy)(
        comm, devComm)


cdef ncclResult_t _ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetLsaMultimemDevicePointer
    _check_or_init_nccl()
    if __ncclGetLsaMultimemDevicePointer == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetLsaMultimemDevicePointer is not found")
    return (<ncclResult_t (*)(ncclWindow_t, size_t, void**) noexcept nogil>__ncclGetLsaMultimemDevicePointer)(
        window, offset, outPtr)


cdef ncclResult_t _ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetLsaDevicePointer
    _check_or_init_nccl()
    if __ncclGetLsaDevicePointer == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetLsaDevicePointer is not found")
    return (<ncclResult_t (*)(ncclWindow_t, size_t, int, void**) noexcept nogil>__ncclGetLsaDevicePointer)(
        window, offset, lsaRank, outPtr)


cdef ncclResult_t _ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset, ncclMultimemHandle_t multimem, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetMultimemDevicePointer
    _check_or_init_nccl()
    if __ncclGetMultimemDevicePointer == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetMultimemDevicePointer is not found")
    return (<ncclResult_t (*)(ncclWindow_t, size_t, ncclMultimemHandle_t, void**) noexcept nogil>__ncclGetMultimemDevicePointer)(
        window, offset, multimem, outPtr)


cdef ncclResult_t _ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetPeerDevicePointer
    _check_or_init_nccl()
    if __ncclGetPeerDevicePointer == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetPeerDevicePointer is not found")
    return (<ncclResult_t (*)(ncclWindow_t, size_t, int, void**) noexcept nogil>__ncclGetPeerDevicePointer)(
        window, offset, peer, outPtr)


cdef ncclResult_t _ncclGetMultimemDeviceLeInfo(ncclWindow_t window, size_t offset, ncclCftLeId* leId, size_t* leOffset) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetMultimemDeviceLeInfo
    _check_or_init_nccl()
    if __ncclGetMultimemDeviceLeInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetMultimemDeviceLeInfo is not found")
    return (<ncclResult_t (*)(ncclWindow_t, size_t, ncclCftLeId*, size_t*) noexcept nogil>__ncclGetMultimemDeviceLeInfo)(
        window, offset, leId, leOffset)


cdef ncclResult_t _ncclGetCftDeviceLeInfo(ncclWindow_t window, size_t offset, int peerCft, ncclTeam_t cftTeam, ncclCftLeId* leId, size_t* leOffset) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetCftDeviceLeInfo
    _check_or_init_nccl()
    if __ncclGetCftDeviceLeInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetCftDeviceLeInfo is not found")
    return (<ncclResult_t (*)(ncclWindow_t, size_t, int, ncclTeam_t, ncclCftLeId*, size_t*) noexcept nogil>__ncclGetCftDeviceLeInfo)(
        window, offset, peerCft, cftTeam, leId, leOffset)


cdef ncclResult_t _ncclGetPeerDeviceLeInfo(ncclWindow_t window, size_t offset, int peerWorld, ncclCftLeId* leId, size_t* leOffset) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetPeerDeviceLeInfo
    _check_or_init_nccl()
    if __ncclGetPeerDeviceLeInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetPeerDeviceLeInfo is not found")
    return (<ncclResult_t (*)(ncclWindow_t, size_t, int, ncclCftLeId*, size_t*) noexcept nogil>__ncclGetPeerDeviceLeInfo)(
        window, offset, peerWorld, leId, leOffset)


cdef ncclTeam_t _ncclTeamWorld(ncclComm_t comm) except* nogil:
    global __ncclTeamWorld
    _check_or_init_nccl()
    if __ncclTeamWorld == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclTeamWorld is not found")
    return (<ncclTeam_t (*)(ncclComm_t) noexcept nogil>__ncclTeamWorld)(
        comm)


cdef ncclTeam_t _ncclTeamLsa(ncclComm_t comm) except* nogil:
    global __ncclTeamLsa
    _check_or_init_nccl()
    if __ncclTeamLsa == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclTeamLsa is not found")
    return (<ncclTeam_t (*)(ncclComm_t) noexcept nogil>__ncclTeamLsa)(
        comm)


cdef ncclTeam_t _ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode) except* nogil:
    global __ncclTeamCft
    _check_or_init_nccl()
    if __ncclTeamCft == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclTeamCft is not found")
    return (<ncclTeam_t (*)(ncclComm_t, ncclCftTeamMode_t) noexcept nogil>__ncclTeamCft)(
        comm, mode)


cdef ncclTeam_t _ncclTeamCftMultimem(ncclComm_t comm) except* nogil:
    global __ncclTeamCftMultimem
    _check_or_init_nccl()
    if __ncclTeamCftMultimem == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclTeamCftMultimem is not found")
    return (<ncclTeam_t (*)(ncclComm_t) noexcept nogil>__ncclTeamCftMultimem)(
        comm)


cdef ncclTeam_t _ncclTeamRail(ncclComm_t comm) except* nogil:
    global __ncclTeamRail
    _check_or_init_nccl()
    if __ncclTeamRail == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclTeamRail is not found")
    return (<ncclTeam_t (*)(ncclComm_t) noexcept nogil>__ncclTeamRail)(
        comm)


cdef int _ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) except?-42 nogil:
    global __ncclTeamRankToWorld
    _check_or_init_nccl()
    if __ncclTeamRankToWorld == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclTeamRankToWorld is not found")
    return (<int (*)(ncclComm_t, ncclTeam_t, int) noexcept nogil>__ncclTeamRankToWorld)(
        comm, team, rank)


cdef int _ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank) except?-42 nogil:
    global __ncclTeamRankToLsa
    _check_or_init_nccl()
    if __ncclTeamRankToLsa == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclTeamRankToLsa is not found")
    return (<int (*)(ncclComm_t, ncclTeam_t, int) noexcept nogil>__ncclTeamRankToLsa)(
        comm, team, rank)


cdef ncclResult_t _ncclLsaBarrierCreateRequirement(ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclLsaBarrierCreateRequirement
    _check_or_init_nccl()
    if __ncclLsaBarrierCreateRequirement == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclLsaBarrierCreateRequirement is not found")
    return (<ncclResult_t (*)(ncclTeam_t, int, ncclLsaBarrierHandle_t*, ncclDevResourceRequirements_t*) noexcept nogil>__ncclLsaBarrierCreateRequirement)(
        team, nBarriers, outHandle, outReq)


cdef ncclResult_t _ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers, ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGinBarrierCreateRequirement
    _check_or_init_nccl()
    if __ncclGinBarrierCreateRequirement == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGinBarrierCreateRequirement is not found")
    return (<ncclResult_t (*)(ncclComm_t, ncclTeam_t, int, ncclGinBarrierHandle_t*, ncclDevResourceRequirements_t*) noexcept nogil>__ncclGinBarrierCreateRequirement)(
        comm, team, nBarriers, outHandle, outReq)


cdef ncclResult_t _ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclLLA2ACreateRequirement
    _check_or_init_nccl()
    if __ncclLLA2ACreateRequirement == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclLLA2ACreateRequirement is not found")
    return (<ncclResult_t (*)(int, int, ncclLLA2AHandle_t*, ncclDevResourceRequirements_t*) noexcept nogil>__ncclLLA2ACreateRequirement)(
        nBlocks, nSlots, outHandle, outReq)


cdef int _ncclLLA2ACalcSlots(int maxElts, int maxEltSize) except?-42 nogil:
    global __ncclLLA2ACalcSlots
    _check_or_init_nccl()
    if __ncclLLA2ACalcSlots == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclLLA2ACalcSlots is not found")
    return (<int (*)(int, int) noexcept nogil>__ncclLLA2ACalcSlots)(
        maxElts, maxEltSize)
