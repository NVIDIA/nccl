# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.0.1. Do not modify it directly.

from libc.stdint cimport intptr_t, uint64_t, uintptr_t

import os
import threading

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib


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


cdef extern from "<dlfcn.h>" nogil:
    ctypedef struct Dl_info:
        const char* dli_fname
        void* dli_fbase
        const char* dli_sname
        void* dli_saddr
    int dladdr(const void*, Dl_info*)


###############################################################################
# Library resolution (mirrors cuda.pathfinder.load_nvidia_dynamic_lib precedence,
# adapted for libnccl_ep.so which is not registered as an NVIDIA pip wheel.)
###############################################################################

# Resolved at first import via _resolve_library_path() below. Path lookup runs
# once, then dlopen handle is cached in the lowpp nccl_ep init guard.
_PACKAGE_LIB_RELPATH = os.path.join("ep", "lib", "libnccl_ep.so")


def _resolve_library_path() -> str:
    # 1. nccl4py package path (replaces cuda.pathfinder's NVIDIA-pip-wheel step).
    #    libnccl_ep.so is at nccl/ep/lib/; this file lives in
    #    nccl/bindings/_internal/, so go up two dirs to reach nccl/.
    pkg_lib = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", _PACKAGE_LIB_RELPATH
    ))
    if os.path.exists(pkg_lib):
        return pkg_lib

    # 2. CONDA_PREFIX/lib[64]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for sub in ("lib", "lib64"):
            candidate = os.path.join(conda_prefix, sub, "libnccl_ep.so")
            if os.path.exists(candidate):
                return candidate

    # 3. CUDA_HOME / CUDA_PATH lib[64]
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_var)
        if root:
            for sub in ("lib", "lib64"):
                candidate = os.path.join(root, sub, "libnccl_ep.so")
                if os.path.exists(candidate):
                    return candidate

    # 4. SONAME fallback — let dlopen perform its own search across
    # LD_LIBRARY_PATH, /etc/ld.so.cache, and /lib, /usr/lib, /lib64,
    # /usr/lib64. If it fails the caller surfaces a clear error.
    return "libnccl_ep.so"


###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_nccl_ep_init = False

cdef void* __ncclEpGetVersion = NULL
cdef void* __ncclEpTensorAlloc = NULL
cdef void* __ncclEpTensorDestroy = NULL
cdef void* __ncclEpCreateGroup = NULL
cdef void* __ncclEpGroupDestroy = NULL
cdef void* __ncclEpCreateHandle = NULL
cdef void* __ncclEpHandleDestroy = NULL
cdef void* __ncclEpHandleMemSize = NULL
cdef void* __ncclEpInitHandle = NULL
cdef void* __ncclEpUpdateHandle = NULL
cdef void* __ncclEpDispatch = NULL
cdef void* __ncclEpCombine = NULL
cdef void* __ncclEpComplete = NULL


cdef void* load_library() except* with gil:
    # libnccl_ep.so has NEEDED libnccl.so.2. Pre-load it with RTLD_GLOBAL so the
    # SONAME is already mapped when libnccl_ep.so's NEEDED is resolved,
    # without depending on filesystem search.
    load_nvidia_dynamic_lib("nccl")

    cdef bytes path_bytes = _resolve_library_path().encode()
    cdef void* handle = dlopen(path_bytes, RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        err_msg = dlerror()
        raise RuntimeError(
            f'Failed to dlopen libnccl_ep ({err_msg.decode()}); '
            f'tried path {path_bytes.decode()!r}'
        )
    return handle


cdef int _check_or_init_nccl_ep() except -1 nogil:
    global __py_nccl_ep_init
    if __py_nccl_ep_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_nccl_ep_init:
            return 0

        # Load function
        global __ncclEpGetVersion
        __ncclEpGetVersion = dlsym(RTLD_DEFAULT, 'ncclEpGetVersion')
        if __ncclEpGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpGetVersion = dlsym(handle, 'ncclEpGetVersion')

        global __ncclEpTensorAlloc
        __ncclEpTensorAlloc = dlsym(RTLD_DEFAULT, 'ncclEpTensorAlloc')
        if __ncclEpTensorAlloc == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpTensorAlloc = dlsym(handle, 'ncclEpTensorAlloc')

        global __ncclEpTensorDestroy
        __ncclEpTensorDestroy = dlsym(RTLD_DEFAULT, 'ncclEpTensorDestroy')
        if __ncclEpTensorDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpTensorDestroy = dlsym(handle, 'ncclEpTensorDestroy')

        global __ncclEpCreateGroup
        __ncclEpCreateGroup = dlsym(RTLD_DEFAULT, 'ncclEpCreateGroup')
        if __ncclEpCreateGroup == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpCreateGroup = dlsym(handle, 'ncclEpCreateGroup')

        global __ncclEpGroupDestroy
        __ncclEpGroupDestroy = dlsym(RTLD_DEFAULT, 'ncclEpGroupDestroy')
        if __ncclEpGroupDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpGroupDestroy = dlsym(handle, 'ncclEpGroupDestroy')

        global __ncclEpCreateHandle
        __ncclEpCreateHandle = dlsym(RTLD_DEFAULT, 'ncclEpCreateHandle')
        if __ncclEpCreateHandle == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpCreateHandle = dlsym(handle, 'ncclEpCreateHandle')

        global __ncclEpHandleDestroy
        __ncclEpHandleDestroy = dlsym(RTLD_DEFAULT, 'ncclEpHandleDestroy')
        if __ncclEpHandleDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpHandleDestroy = dlsym(handle, 'ncclEpHandleDestroy')

        global __ncclEpHandleMemSize
        __ncclEpHandleMemSize = dlsym(RTLD_DEFAULT, 'ncclEpHandleMemSize')
        if __ncclEpHandleMemSize == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpHandleMemSize = dlsym(handle, 'ncclEpHandleMemSize')

        global __ncclEpInitHandle
        __ncclEpInitHandle = dlsym(RTLD_DEFAULT, 'ncclEpInitHandle')
        if __ncclEpInitHandle == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpInitHandle = dlsym(handle, 'ncclEpInitHandle')

        global __ncclEpUpdateHandle
        __ncclEpUpdateHandle = dlsym(RTLD_DEFAULT, 'ncclEpUpdateHandle')
        if __ncclEpUpdateHandle == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpUpdateHandle = dlsym(handle, 'ncclEpUpdateHandle')

        global __ncclEpDispatch
        __ncclEpDispatch = dlsym(RTLD_DEFAULT, 'ncclEpDispatch')
        if __ncclEpDispatch == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpDispatch = dlsym(handle, 'ncclEpDispatch')

        global __ncclEpCombine
        __ncclEpCombine = dlsym(RTLD_DEFAULT, 'ncclEpCombine')
        if __ncclEpCombine == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpCombine = dlsym(handle, 'ncclEpCombine')

        global __ncclEpComplete
        __ncclEpComplete = dlsym(RTLD_DEFAULT, 'ncclEpComplete')
        if __ncclEpComplete == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclEpComplete = dlsym(handle, 'ncclEpComplete')
        __py_nccl_ep_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nccl_ep()
    cdef dict data = {}

    global __ncclEpGetVersion
    data["__ncclEpGetVersion"] = <intptr_t>__ncclEpGetVersion

    global __ncclEpTensorAlloc
    data["__ncclEpTensorAlloc"] = <intptr_t>__ncclEpTensorAlloc

    global __ncclEpTensorDestroy
    data["__ncclEpTensorDestroy"] = <intptr_t>__ncclEpTensorDestroy

    global __ncclEpCreateGroup
    data["__ncclEpCreateGroup"] = <intptr_t>__ncclEpCreateGroup

    global __ncclEpGroupDestroy
    data["__ncclEpGroupDestroy"] = <intptr_t>__ncclEpGroupDestroy

    global __ncclEpCreateHandle
    data["__ncclEpCreateHandle"] = <intptr_t>__ncclEpCreateHandle

    global __ncclEpHandleDestroy
    data["__ncclEpHandleDestroy"] = <intptr_t>__ncclEpHandleDestroy

    global __ncclEpHandleMemSize
    data["__ncclEpHandleMemSize"] = <intptr_t>__ncclEpHandleMemSize

    global __ncclEpInitHandle
    data["__ncclEpInitHandle"] = <intptr_t>__ncclEpInitHandle

    global __ncclEpUpdateHandle
    data["__ncclEpUpdateHandle"] = <intptr_t>__ncclEpUpdateHandle

    global __ncclEpDispatch
    data["__ncclEpDispatch"] = <intptr_t>__ncclEpDispatch

    global __ncclEpCombine
    data["__ncclEpCombine"] = <intptr_t>__ncclEpCombine

    global __ncclEpComplete
    data["__ncclEpComplete"] = <intptr_t>__ncclEpComplete

    func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]


cdef object __nccl_ep_loaded_so_path = None


cpdef object _inspect_loaded_library_path():
    # Path of the .so backing the loaded symbols, via dladdr() on a
    # resolved entry point. None if it cannot be determined.
    global __nccl_ep_loaded_so_path
    if __nccl_ep_loaded_so_path is not None:
        return __nccl_ep_loaded_so_path

    cdef dict ptrs = _inspect_function_pointers()
    # Any resolved symbol maps to the same .so; anchor on a stable one.
    cdef intptr_t addr = ptrs.get("__ncclEpGetVersion", 0)
    if addr == 0:
        for value in ptrs.values():
            if value:
                addr = value
                break

    cdef Dl_info info
    if addr == 0:
        return None
    if dladdr(<void*>addr, &info) == 0 or info.dli_fname == NULL:
        return None
    __nccl_ep_loaded_so_path = info.dli_fname.decode()
    return __nccl_ep_loaded_so_path


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t _ncclEpGetVersion(int* version) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpGetVersion
    _check_or_init_nccl_ep()
    if __ncclEpGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpGetVersion is not found")
    return (<ncclResult_t (*)(int*) noexcept nogil>__ncclEpGetVersion)(
        version)


cdef ncclResult_t _ncclEpTensorAlloc(ncclEpTensor_t** tensor, unsigned int ndim, ncclDataType_t datatype, const size_t* sizes, const ncclEpTensorAllocConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpTensorAlloc
    _check_or_init_nccl_ep()
    if __ncclEpTensorAlloc == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpTensorAlloc is not found")
    return (<ncclResult_t (*)(ncclEpTensor_t**, unsigned int, ncclDataType_t, const size_t*, const ncclEpTensorAllocConfig_t*) noexcept nogil>__ncclEpTensorAlloc)(
        tensor, ndim, datatype, sizes, config)


cdef ncclResult_t _ncclEpTensorDestroy(ncclEpTensor_t* tensor) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpTensorDestroy
    _check_or_init_nccl_ep()
    if __ncclEpTensorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpTensorDestroy is not found")
    return (<ncclResult_t (*)(ncclEpTensor_t*) noexcept nogil>__ncclEpTensorDestroy)(
        tensor)


cdef ncclResult_t _ncclEpCreateGroup(ncclEpGroup_t* ep_group, ncclComm_t comm, const ncclEpGroupConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpCreateGroup
    _check_or_init_nccl_ep()
    if __ncclEpCreateGroup == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpCreateGroup is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t*, ncclComm_t, const ncclEpGroupConfig_t*) noexcept nogil>__ncclEpCreateGroup)(
        ep_group, comm, config)


cdef ncclResult_t _ncclEpGroupDestroy(ncclEpGroup_t ep_group) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpGroupDestroy
    _check_or_init_nccl_ep()
    if __ncclEpGroupDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpGroupDestroy is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t) noexcept nogil>__ncclEpGroupDestroy)(
        ep_group)


cdef ncclResult_t _ncclEpCreateHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpTensor_t* topk_idx, const ncclEpLayoutInfo_t* layout_info, const ncclEpHandleConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpCreateHandle
    _check_or_init_nccl_ep()
    if __ncclEpCreateHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpCreateHandle is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t*, ncclEpGroup_t, ncclEpLayout_t, const ncclEpTensor_t*, const ncclEpLayoutInfo_t*, const ncclEpHandleConfig_t*, cudaStream_t) noexcept nogil>__ncclEpCreateHandle)(
        handle, ep_group, layout, topk_idx, layout_info, config, stream)


cdef ncclResult_t _ncclEpHandleDestroy(ncclEpHandle_t handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpHandleDestroy
    _check_or_init_nccl_ep()
    if __ncclEpHandleDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpHandleDestroy is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t) noexcept nogil>__ncclEpHandleDestroy)(
        handle)


cdef ncclResult_t _ncclEpHandleMemSize(ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpHandleConfig_t* config, size_t* size_out, int num_topk) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpHandleMemSize
    _check_or_init_nccl_ep()
    if __ncclEpHandleMemSize == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpHandleMemSize is not found")
    return (<ncclResult_t (*)(ncclEpGroup_t, ncclEpLayout_t, const ncclEpHandleConfig_t*, size_t*, int) noexcept nogil>__ncclEpHandleMemSize)(
        ep_group, layout, config, size_out, num_topk)


cdef ncclResult_t _ncclEpInitHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpHandleConfig_t* config, int num_topk, const ncclEpTensor_t* handle_mem) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpInitHandle
    _check_or_init_nccl_ep()
    if __ncclEpInitHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpInitHandle is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t*, ncclEpGroup_t, ncclEpLayout_t, const ncclEpHandleConfig_t*, int, const ncclEpTensor_t*) noexcept nogil>__ncclEpInitHandle)(
        handle, ep_group, layout, config, num_topk, handle_mem)


cdef ncclResult_t _ncclEpUpdateHandle(ncclEpHandle_t handle, const ncclEpTensor_t* topk_idx, const ncclEpLayoutInfo_t* layout_info, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpUpdateHandle
    _check_or_init_nccl_ep()
    if __ncclEpUpdateHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpUpdateHandle is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpTensor_t*, const ncclEpLayoutInfo_t*, cudaStream_t) noexcept nogil>__ncclEpUpdateHandle)(
        handle, topk_idx, layout_info, stream)


cdef ncclResult_t _ncclEpDispatch(ncclEpHandle_t handle, const ncclEpDispatchInputs_t* inputs, const ncclEpDispatchOutputs_t* outputs, const ncclEpLayoutInfo_t* layout_info, const ncclEpDispatchConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpDispatch
    _check_or_init_nccl_ep()
    if __ncclEpDispatch == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpDispatch is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpDispatchInputs_t*, const ncclEpDispatchOutputs_t*, const ncclEpLayoutInfo_t*, const ncclEpDispatchConfig_t*, cudaStream_t) noexcept nogil>__ncclEpDispatch)(
        handle, inputs, outputs, layout_info, config, stream)


cdef ncclResult_t _ncclEpCombine(ncclEpHandle_t handle, const ncclEpCombineInputs_t* inputs, const ncclEpCombineOutputs_t* outputs, const ncclEpCombineConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpCombine
    _check_or_init_nccl_ep()
    if __ncclEpCombine == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpCombine is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpCombineInputs_t*, const ncclEpCombineOutputs_t*, const ncclEpCombineConfig_t*, cudaStream_t) noexcept nogil>__ncclEpCombine)(
        handle, inputs, outputs, config, stream)


cdef ncclResult_t _ncclEpComplete(ncclEpHandle_t handle, const ncclEpCompleteConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclEpComplete
    _check_or_init_nccl_ep()
    if __ncclEpComplete == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclEpComplete is not found")
    return (<ncclResult_t (*)(ncclEpHandle_t, const ncclEpCompleteConfig_t*, cudaStream_t) noexcept nogil>__ncclEpComplete)(
        handle, config, stream)
