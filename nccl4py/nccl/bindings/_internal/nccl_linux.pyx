# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.28.0. Do not modify it directly.

from libc.stdint cimport intptr_t

import threading

from .utils import FunctionNotFoundError, NotSupportedError


###############################################################################
# Extern
###############################################################################

# You must 'from .utils import NotSupportedError' before using this template

cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()
    void* dlsym(void*, const char*)
    int dlclose(void*)

    enum:
        RTLD_LAZY
        RTLD_NOW
        RTLD_GLOBAL
        RTLD_LOCAL

    const void* RTLD_DEFAULT 'RTLD_DEFAULT'

cdef int get_cuda_version():
    cdef void* handle = NULL
    cdef int err, driver_ver = 0

    # Load driver to check version
    handle = dlopen('libcuda.so.1', RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        err_msg = dlerror()
        raise NotSupportedError(f'CUDA driver is not found ({err_msg.decode()})')
    cuDriverGetVersion = dlsym(handle, "cuDriverGetVersion")
    if cuDriverGetVersion == NULL:
        raise RuntimeError('Did not find cuDriverGetVersion symbol in libcuda.so.1')
    err = (<int (*)(int*) noexcept nogil>cuDriverGetVersion)(&driver_ver)
    if err != 0:
        raise RuntimeError(f'cuDriverGetVersion returned error code {err}')

    return driver_ver



###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_nccl_init = False

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
cdef void* __ncclCommSplit = NULL
cdef void* __ncclCommShrink = NULL
cdef void* __ncclCommInitRankScalable = NULL
cdef void* __ncclGetErrorString = NULL
cdef void* __ncclGetLastError = NULL
cdef void* __ncclCommGetAsyncError = NULL
cdef void* __ncclCommCount = NULL
cdef void* __ncclCommCuDevice = NULL
cdef void* __ncclCommUserRank = NULL
cdef void* __ncclCommRegister = NULL
cdef void* __ncclCommDeregister = NULL
cdef void* __ncclCommWindowRegister = NULL
cdef void* __ncclCommWindowDeregister = NULL
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
cdef void* __ncclSend = NULL
cdef void* __ncclRecv = NULL
cdef void* __ncclGroupStart = NULL
cdef void* __ncclGroupEnd = NULL
cdef void* __ncclGroupSimulateEnd = NULL


cdef void* load_library() except* nogil:
    cdef void* handle
    handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise RuntimeError(f'Failed to dlopen libnccl ({err_msg.decode()})')
    return handle


cdef int _check_or_init_nccl() except -1 nogil:
    global __py_nccl_init
    if __py_nccl_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_nccl_init:
            return 0

        # Load function
        global __ncclMemAlloc
        __ncclMemAlloc = dlsym(RTLD_DEFAULT, 'ncclMemAlloc')
        if __ncclMemAlloc == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclMemAlloc = dlsym(handle, 'ncclMemAlloc')

        global __ncclMemFree
        __ncclMemFree = dlsym(RTLD_DEFAULT, 'ncclMemFree')
        if __ncclMemFree == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclMemFree = dlsym(handle, 'ncclMemFree')

        global __ncclGetVersion
        __ncclGetVersion = dlsym(RTLD_DEFAULT, 'ncclGetVersion')
        if __ncclGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetVersion = dlsym(handle, 'ncclGetVersion')

        global __ncclGetUniqueId
        __ncclGetUniqueId = dlsym(RTLD_DEFAULT, 'ncclGetUniqueId')
        if __ncclGetUniqueId == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetUniqueId = dlsym(handle, 'ncclGetUniqueId')

        global __ncclCommInitRankConfig
        __ncclCommInitRankConfig = dlsym(RTLD_DEFAULT, 'ncclCommInitRankConfig')
        if __ncclCommInitRankConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommInitRankConfig = dlsym(handle, 'ncclCommInitRankConfig')

        global __ncclCommInitRank
        __ncclCommInitRank = dlsym(RTLD_DEFAULT, 'ncclCommInitRank')
        if __ncclCommInitRank == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommInitRank = dlsym(handle, 'ncclCommInitRank')

        global __ncclCommInitAll
        __ncclCommInitAll = dlsym(RTLD_DEFAULT, 'ncclCommInitAll')
        if __ncclCommInitAll == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommInitAll = dlsym(handle, 'ncclCommInitAll')

        global __ncclCommFinalize
        __ncclCommFinalize = dlsym(RTLD_DEFAULT, 'ncclCommFinalize')
        if __ncclCommFinalize == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommFinalize = dlsym(handle, 'ncclCommFinalize')

        global __ncclCommDestroy
        __ncclCommDestroy = dlsym(RTLD_DEFAULT, 'ncclCommDestroy')
        if __ncclCommDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommDestroy = dlsym(handle, 'ncclCommDestroy')

        global __ncclCommAbort
        __ncclCommAbort = dlsym(RTLD_DEFAULT, 'ncclCommAbort')
        if __ncclCommAbort == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommAbort = dlsym(handle, 'ncclCommAbort')

        global __ncclCommSplit
        __ncclCommSplit = dlsym(RTLD_DEFAULT, 'ncclCommSplit')
        if __ncclCommSplit == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommSplit = dlsym(handle, 'ncclCommSplit')

        global __ncclCommShrink
        __ncclCommShrink = dlsym(RTLD_DEFAULT, 'ncclCommShrink')
        if __ncclCommShrink == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommShrink = dlsym(handle, 'ncclCommShrink')

        global __ncclCommInitRankScalable
        __ncclCommInitRankScalable = dlsym(RTLD_DEFAULT, 'ncclCommInitRankScalable')
        if __ncclCommInitRankScalable == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommInitRankScalable = dlsym(handle, 'ncclCommInitRankScalable')

        global __ncclGetErrorString
        __ncclGetErrorString = dlsym(RTLD_DEFAULT, 'ncclGetErrorString')
        if __ncclGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetErrorString = dlsym(handle, 'ncclGetErrorString')

        global __ncclGetLastError
        __ncclGetLastError = dlsym(RTLD_DEFAULT, 'ncclGetLastError')
        if __ncclGetLastError == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetLastError = dlsym(handle, 'ncclGetLastError')

        global __ncclCommGetAsyncError
        __ncclCommGetAsyncError = dlsym(RTLD_DEFAULT, 'ncclCommGetAsyncError')
        if __ncclCommGetAsyncError == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommGetAsyncError = dlsym(handle, 'ncclCommGetAsyncError')

        global __ncclCommCount
        __ncclCommCount = dlsym(RTLD_DEFAULT, 'ncclCommCount')
        if __ncclCommCount == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommCount = dlsym(handle, 'ncclCommCount')

        global __ncclCommCuDevice
        __ncclCommCuDevice = dlsym(RTLD_DEFAULT, 'ncclCommCuDevice')
        if __ncclCommCuDevice == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommCuDevice = dlsym(handle, 'ncclCommCuDevice')

        global __ncclCommUserRank
        __ncclCommUserRank = dlsym(RTLD_DEFAULT, 'ncclCommUserRank')
        if __ncclCommUserRank == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommUserRank = dlsym(handle, 'ncclCommUserRank')

        global __ncclCommRegister
        __ncclCommRegister = dlsym(RTLD_DEFAULT, 'ncclCommRegister')
        if __ncclCommRegister == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommRegister = dlsym(handle, 'ncclCommRegister')

        global __ncclCommDeregister
        __ncclCommDeregister = dlsym(RTLD_DEFAULT, 'ncclCommDeregister')
        if __ncclCommDeregister == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommDeregister = dlsym(handle, 'ncclCommDeregister')

        global __ncclCommWindowRegister
        __ncclCommWindowRegister = dlsym(RTLD_DEFAULT, 'ncclCommWindowRegister')
        if __ncclCommWindowRegister == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommWindowRegister = dlsym(handle, 'ncclCommWindowRegister')

        global __ncclCommWindowDeregister
        __ncclCommWindowDeregister = dlsym(RTLD_DEFAULT, 'ncclCommWindowDeregister')
        if __ncclCommWindowDeregister == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommWindowDeregister = dlsym(handle, 'ncclCommWindowDeregister')

        global __ncclRedOpCreatePreMulSum
        __ncclRedOpCreatePreMulSum = dlsym(RTLD_DEFAULT, 'ncclRedOpCreatePreMulSum')
        if __ncclRedOpCreatePreMulSum == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclRedOpCreatePreMulSum = dlsym(handle, 'ncclRedOpCreatePreMulSum')

        global __ncclRedOpDestroy
        __ncclRedOpDestroy = dlsym(RTLD_DEFAULT, 'ncclRedOpDestroy')
        if __ncclRedOpDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclRedOpDestroy = dlsym(handle, 'ncclRedOpDestroy')

        global __ncclReduce
        __ncclReduce = dlsym(RTLD_DEFAULT, 'ncclReduce')
        if __ncclReduce == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclReduce = dlsym(handle, 'ncclReduce')

        global __ncclBcast
        __ncclBcast = dlsym(RTLD_DEFAULT, 'ncclBcast')
        if __ncclBcast == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclBcast = dlsym(handle, 'ncclBcast')

        global __ncclBroadcast
        __ncclBroadcast = dlsym(RTLD_DEFAULT, 'ncclBroadcast')
        if __ncclBroadcast == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclBroadcast = dlsym(handle, 'ncclBroadcast')

        global __ncclAllReduce
        __ncclAllReduce = dlsym(RTLD_DEFAULT, 'ncclAllReduce')
        if __ncclAllReduce == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclAllReduce = dlsym(handle, 'ncclAllReduce')

        global __ncclReduceScatter
        __ncclReduceScatter = dlsym(RTLD_DEFAULT, 'ncclReduceScatter')
        if __ncclReduceScatter == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclReduceScatter = dlsym(handle, 'ncclReduceScatter')

        global __ncclAllGather
        __ncclAllGather = dlsym(RTLD_DEFAULT, 'ncclAllGather')
        if __ncclAllGather == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclAllGather = dlsym(handle, 'ncclAllGather')

        global __ncclAlltoAll
        __ncclAlltoAll = dlsym(RTLD_DEFAULT, 'ncclAlltoAll')
        if __ncclAlltoAll == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclAlltoAll = dlsym(handle, 'ncclAlltoAll')

        global __ncclGather
        __ncclGather = dlsym(RTLD_DEFAULT, 'ncclGather')
        if __ncclGather == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGather = dlsym(handle, 'ncclGather')

        global __ncclScatter
        __ncclScatter = dlsym(RTLD_DEFAULT, 'ncclScatter')
        if __ncclScatter == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclScatter = dlsym(handle, 'ncclScatter')

        global __ncclSend
        __ncclSend = dlsym(RTLD_DEFAULT, 'ncclSend')
        if __ncclSend == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclSend = dlsym(handle, 'ncclSend')

        global __ncclRecv
        __ncclRecv = dlsym(RTLD_DEFAULT, 'ncclRecv')
        if __ncclRecv == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclRecv = dlsym(handle, 'ncclRecv')

        global __ncclGroupStart
        __ncclGroupStart = dlsym(RTLD_DEFAULT, 'ncclGroupStart')
        if __ncclGroupStart == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGroupStart = dlsym(handle, 'ncclGroupStart')

        global __ncclGroupEnd
        __ncclGroupEnd = dlsym(RTLD_DEFAULT, 'ncclGroupEnd')
        if __ncclGroupEnd == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGroupEnd = dlsym(handle, 'ncclGroupEnd')

        global __ncclGroupSimulateEnd
        __ncclGroupSimulateEnd = dlsym(RTLD_DEFAULT, 'ncclGroupSimulateEnd')
        if __ncclGroupSimulateEnd == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGroupSimulateEnd = dlsym(handle, 'ncclGroupSimulateEnd')
        __py_nccl_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

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

    global __ncclCommSplit
    data["__ncclCommSplit"] = <intptr_t>__ncclCommSplit

    global __ncclCommShrink
    data["__ncclCommShrink"] = <intptr_t>__ncclCommShrink

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

    global __ncclCommWindowRegister
    data["__ncclCommWindowRegister"] = <intptr_t>__ncclCommWindowRegister

    global __ncclCommWindowDeregister
    data["__ncclCommWindowDeregister"] = <intptr_t>__ncclCommWindowDeregister

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

    global __ncclSend
    data["__ncclSend"] = <intptr_t>__ncclSend

    global __ncclRecv
    data["__ncclRecv"] = <intptr_t>__ncclRecv

    global __ncclGroupStart
    data["__ncclGroupStart"] = <intptr_t>__ncclGroupStart

    global __ncclGroupEnd
    data["__ncclGroupEnd"] = <intptr_t>__ncclGroupEnd

    global __ncclGroupSimulateEnd
    data["__ncclGroupSimulateEnd"] = <intptr_t>__ncclGroupSimulateEnd

    func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]


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
