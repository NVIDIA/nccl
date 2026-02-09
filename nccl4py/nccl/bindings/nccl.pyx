# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.28.0. Do not modify it directly.

cimport cython  # NOQA
from cpython cimport buffer as _buffer
from cpython.memoryview cimport PyMemoryView_FromMemory
from libcpp.vector cimport vector

from ._internal.utils cimport (nested_resource, nullable_unique_ptr, get_buffer_pointer,
                              get_resource_ptr, get_nested_resource_ptr)

from enum import IntEnum as _IntEnum

import numpy as _numpy


###############################################################################
# POD
###############################################################################

unique_id_dtype = _numpy.dtype([
    ("internal", _numpy.int8, (128,)),
    ], align=True)


cdef class UniqueId:
    """Empty-initialize an instance of `ncclUniqueId`.


    .. seealso:: `ncclUniqueId`
    """
    cdef:
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=unique_id_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(ncclUniqueId), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(ncclUniqueId)}"

    def __repr__(self):
        return f"<{__name__}.UniqueId object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        return self._data.ctypes.data

    def __eq__(self, other):
        if not isinstance(other, UniqueId):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an UniqueId instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `unique_id_dtype` holding the data.
        """
        cdef UniqueId obj = UniqueId.__new__(UniqueId)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != unique_id_dtype:
            raise ValueError("data array must be of dtype unique_id_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False):
        """Create an UniqueId instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef UniqueId obj = UniqueId.__new__(UniqueId)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(ncclUniqueId), flag)
        data = _numpy.ndarray((1,), buffer=buf,
                              dtype=unique_id_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


config_dtype = _numpy.dtype([
    ("size_", _numpy.uint64, ),
    ("magic", _numpy.uint32, ),
    ("version", _numpy.uint32, ),
    ("blocking", _numpy.int32, ),
    ("cga_cluster_size", _numpy.int32, ),
    ("min_ctas", _numpy.int32, ),
    ("max_ctas", _numpy.int32, ),
    ("net_name", _numpy.intp, ),
    ("split_share", _numpy.int32, ),
    ("traffic_class", _numpy.int32, ),
    ("comm_name", _numpy.intp, ),
    ("collnet_enable", _numpy.int32, ),
    ("cta_policy", _numpy.int32, ),
    ("shrink_share", _numpy.int32, ),
    ("nvls_ctas", _numpy.int32, ),
    ("n_channels_per_net_peer", _numpy.int32, ),
    ("nvlink_centric_sched", _numpy.int32, ),
    ], align=True)


cdef class Config:
    """Empty-initialize an instance of `ncclConfig_t`.


    .. seealso:: `ncclConfig_t`
    """
    cdef:
        readonly object _data
        dict _holder

    def __init__(self):
        arr = _numpy.empty(1, dtype=config_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(ncclConfig_t), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(ncclConfig_t)}"
        self._holder = {}

    def __repr__(self):
        return f"<{__name__}.Config object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        return self._data.ctypes.data

    def __eq__(self, other):
        if not isinstance(other, Config):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def size_(self):
        """int: """
        return int(self._data.size_[0])

    @size_.setter
    def size_(self, val):
        self._data.size_ = val

    @property
    def magic(self):
        """int: """
        return int(self._data.magic[0])

    @magic.setter
    def magic(self, val):
        self._data.magic = val

    @property
    def version(self):
        """int: """
        return int(self._data.version[0])

    @version.setter
    def version(self, val):
        self._data.version = val

    @property
    def blocking(self):
        """int: """
        return int(self._data.blocking[0])

    @blocking.setter
    def blocking(self, val):
        self._data.blocking = val

    @property
    def cga_cluster_size(self):
        """int: """
        return int(self._data.cga_cluster_size[0])

    @cga_cluster_size.setter
    def cga_cluster_size(self, val):
        self._data.cga_cluster_size = val

    @property
    def min_ctas(self):
        """int: """
        return int(self._data.min_ctas[0])

    @min_ctas.setter
    def min_ctas(self, val):
        self._data.min_ctas = val

    @property
    def max_ctas(self):
        """int: """
        return int(self._data.max_ctas[0])

    @max_ctas.setter
    def max_ctas(self, val):
        self._data.max_ctas = val

    @property
    def net_name(self):
        """str: """
        cdef char* ptr
        cdef bytes buf
        ptr = <char*><intptr_t>(int(self._data.net_name[0]))
        if ptr:
           buf = ptr
           return buf.decode()
        return ""

    @net_name.setter
    def net_name(self, val):
        cdef char* ptr
        cdef bytes buf
        buf = val.encode()
        ptr = buf
        self._holder["net_name"] = buf
        self._data.net_name = <intptr_t>ptr
        return

    @property
    def split_share(self):
        """int: """
        return int(self._data.split_share[0])

    @split_share.setter
    def split_share(self, val):
        self._data.split_share = val

    @property
    def traffic_class(self):
        """int: """
        return int(self._data.traffic_class[0])

    @traffic_class.setter
    def traffic_class(self, val):
        self._data.traffic_class = val

    @property
    def comm_name(self):
        """str: """
        cdef char* ptr
        cdef bytes buf
        ptr = <char*><intptr_t>(int(self._data.comm_name[0]))
        if ptr:
           buf = ptr
           return buf.decode()
        return ""

    @comm_name.setter
    def comm_name(self, val):
        cdef char* ptr
        cdef bytes buf
        buf = val.encode()
        ptr = buf
        self._holder["comm_name"] = buf
        self._data.comm_name = <intptr_t>ptr
        return

    @property
    def collnet_enable(self):
        """int: """
        return int(self._data.collnet_enable[0])

    @collnet_enable.setter
    def collnet_enable(self, val):
        self._data.collnet_enable = val

    @property
    def cta_policy(self):
        """int: """
        return int(self._data.cta_policy[0])

    @cta_policy.setter
    def cta_policy(self, val):
        self._data.cta_policy = val

    @property
    def shrink_share(self):
        """int: """
        return int(self._data.shrink_share[0])

    @shrink_share.setter
    def shrink_share(self, val):
        self._data.shrink_share = val

    @property
    def nvls_ctas(self):
        """int: """
        return int(self._data.nvls_ctas[0])

    @nvls_ctas.setter
    def nvls_ctas(self, val):
        self._data.nvls_ctas = val

    @property
    def n_channels_per_net_peer(self):
        """int: """
        return int(self._data.n_channels_per_net_peer[0])

    @n_channels_per_net_peer.setter
    def n_channels_per_net_peer(self, val):
        self._data.n_channels_per_net_peer = val

    @property
    def nvlink_centric_sched(self):
        """int: """
        return int(self._data.nvlink_centric_sched[0])

    @nvlink_centric_sched.setter
    def nvlink_centric_sched(self, val):
        self._data.nvlink_centric_sched = val

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an Config instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `config_dtype` holding the data.
        """
        cdef Config obj = Config.__new__(Config)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != config_dtype:
            raise ValueError("data array must be of dtype config_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False):
        """Create an Config instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Config obj = Config.__new__(Config)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(ncclConfig_t), flag)
        data = _numpy.ndarray((1,), buffer=buf,
                              dtype=config_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


sim_info_dtype = _numpy.dtype([
    ("size_", _numpy.uint64, ),
    ("magic", _numpy.uint32, ),
    ("version", _numpy.uint32, ),
    ("estimated_time", _numpy.float32, ),
    ], align=True)


cdef class SimInfo:
    """Empty-initialize an instance of `ncclSimInfo_t`.


    .. seealso:: `ncclSimInfo_t`
    """
    cdef:
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=sim_info_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(ncclSimInfo_t), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(ncclSimInfo_t)}"

    def __repr__(self):
        return f"<{__name__}.SimInfo object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        return self._data.ctypes.data

    def __eq__(self, other):
        if not isinstance(other, SimInfo):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def size_(self):
        """int: """
        return int(self._data.size_[0])

    @size_.setter
    def size_(self, val):
        self._data.size_ = val

    @property
    def magic(self):
        """int: """
        return int(self._data.magic[0])

    @magic.setter
    def magic(self, val):
        self._data.magic = val

    @property
    def version(self):
        """int: """
        return int(self._data.version[0])

    @version.setter
    def version(self, val):
        self._data.version = val

    @property
    def estimated_time(self):
        """float: """
        return float(self._data.estimated_time[0])

    @estimated_time.setter
    def estimated_time(self, val):
        self._data.estimated_time = val

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an SimInfo instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `sim_info_dtype` holding the data.
        """
        cdef SimInfo obj = SimInfo.__new__(SimInfo)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != sim_info_dtype:
            raise ValueError("data array must be of dtype sim_info_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False):
        """Create an SimInfo instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef SimInfo obj = SimInfo.__new__(SimInfo)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(ncclSimInfo_t), flag)
        data = _numpy.ndarray((1,), buffer=buf,
                              dtype=sim_info_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj



###############################################################################
# Enum
###############################################################################

class Result(_IntEnum):
    """See `ncclResult_t`."""
    Success = ncclSuccess
    UnhandledCudaError = ncclUnhandledCudaError
    SystemError = ncclSystemError
    InternalError = ncclInternalError
    InvalidArgument = ncclInvalidArgument
    InvalidUsage = ncclInvalidUsage
    RemoteError = ncclRemoteError
    InProgress = ncclInProgress
    NumResults = ncclNumResults

class RedOp_dummy(_IntEnum):
    """See `ncclRedOp_dummy_t`."""
    NumOps_dummy = ncclNumOps_dummy

class RedOp(_IntEnum):
    """See `ncclRedOp_t`."""
    Sum = ncclSum
    Prod = ncclProd
    Max = ncclMax
    Min = ncclMin
    Avg = ncclAvg
    NumOps = ncclNumOps
    MaxRedOp = ncclMaxRedOp

class DataType(_IntEnum):
    """See `ncclDataType_t`."""
    Int8 = ncclInt8
    Char = ncclChar
    Uint8 = ncclUint8
    Int32 = ncclInt32
    Int = ncclInt
    Uint32 = ncclUint32
    Int64 = ncclInt64
    Uint64 = ncclUint64
    Float16 = ncclFloat16
    Half = ncclHalf
    Float32 = ncclFloat32
    Float = ncclFloat
    Float64 = ncclFloat64
    Double = ncclDouble
    Bfloat16 = ncclBfloat16
    Float8e4m3 = ncclFloat8e4m3
    Float8e5m2 = ncclFloat8e5m2
    NumTypes = ncclNumTypes

class ScalarResidence(_IntEnum):
    """See `ncclScalarResidence_t`."""
    Device = ncclScalarDevice
    HostImmediate = ncclScalarHostImmediate


###############################################################################
# Error handling
###############################################################################

class NCCLError(Exception):

    def __init__(self, status):
        self.status = status
        s = Result(status)
        cdef str err = f"{s.name} ({s.value}): {get_error_string(status)}"
        super(NCCLError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != Result.Success and status != Result.InProgress:
        raise NCCLError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t mem_alloc(size_t size) except? 0:
    cdef void* ptr
    with nogil:
        __status__ = ncclMemAlloc(&ptr, size)
    check_status(__status__)
    return <intptr_t>ptr


cpdef mem_free(intptr_t ptr):
    with nogil:
        __status__ = ncclMemFree(<void*>ptr)
    check_status(__status__)


cpdef int get_version() except? -1:
    cdef int version
    with nogil:
        __status__ = ncclGetVersion(&version)
    check_status(__status__)
    return version


cpdef get_unique_id(intptr_t unique_id):
    with nogil:
        __status__ = ncclGetUniqueId(<ncclUniqueId*>unique_id)
    check_status(__status__)


cpdef intptr_t comm_init_rank_config(int nranks, comm_id, int rank, intptr_t config) except? 0:
    cdef void* _comm_id_ = get_buffer_pointer(comm_id, -1, readonly=False)
    cdef Comm comm
    with nogil:
        __status__ = ncclCommInitRankConfig(&comm, nranks, (<ncclUniqueId*>(_comm_id_))[0], rank, <ncclConfig_t*>config)
    check_status(__status__)
    return <intptr_t>comm


cpdef intptr_t comm_init_rank(int nranks, comm_id, int rank) except? 0:
    cdef void* _comm_id_ = get_buffer_pointer(comm_id, -1, readonly=False)
    cdef Comm comm
    with nogil:
        __status__ = ncclCommInitRank(&comm, nranks, (<ncclUniqueId*>(_comm_id_))[0], rank)
    check_status(__status__)
    return <intptr_t>comm


cpdef comm_init_all(intptr_t comm, int ndev, devlist):
    cdef nullable_unique_ptr[ vector[int] ] _devlist_
    get_resource_ptr[int](_devlist_, devlist, <int*>NULL)
    with nogil:
        __status__ = ncclCommInitAll(<Comm*>comm, ndev, <const int*>(_devlist_.data()))
    check_status(__status__)


cpdef comm_finalize(intptr_t comm):
    with nogil:
        __status__ = ncclCommFinalize(<Comm>comm)
    check_status(__status__)


cpdef comm_destroy(intptr_t comm):
    with nogil:
        __status__ = ncclCommDestroy(<Comm>comm)
    check_status(__status__)


cpdef comm_abort(intptr_t comm):
    with nogil:
        __status__ = ncclCommAbort(<Comm>comm)
    check_status(__status__)


cpdef intptr_t comm_split(intptr_t comm, int color, int key, intptr_t config) except? 0:
    cdef Comm newcomm
    with nogil:
        __status__ = ncclCommSplit(<Comm>comm, color, key, &newcomm, <ncclConfig_t*>config)
    check_status(__status__)
    return <intptr_t>newcomm


cpdef intptr_t comm_shrink(intptr_t comm, exclude_ranks_list, int exclude_ranks_count, intptr_t config, int shrink_flags) except? 0:
    cdef nullable_unique_ptr[ vector[int] ] _exclude_ranks_list_
    get_resource_ptr[int](_exclude_ranks_list_, exclude_ranks_list, <int*>NULL)
    cdef Comm newcomm
    with nogil:
        __status__ = ncclCommShrink(<Comm>comm, <int*>(_exclude_ranks_list_.data()), exclude_ranks_count, &newcomm, <ncclConfig_t*>config, shrink_flags)
    check_status(__status__)
    return <intptr_t>newcomm


cpdef intptr_t comm_init_rank_scalable(int nranks, int myrank, int n_id, comm_ids, intptr_t config) except? 0:
    cdef nested_resource[ ncclUniqueId ] _comm_ids_
    get_nested_resource_ptr[ncclUniqueId](_comm_ids_, comm_ids, <ncclUniqueId*>NULL)
    cdef Comm newcomm
    with nogil:
        __status__ = ncclCommInitRankScalable(&newcomm, nranks, myrank, n_id, <ncclUniqueId*>(_comm_ids_.ptrs.data()), <ncclConfig_t*>config)
    check_status(__status__)
    return <intptr_t>newcomm


cpdef str get_error_string(int result):
    cdef bytes _output_
    _output_ = ncclGetErrorString(<_Result>result)
    return _output_.decode()


cpdef str get_last_error(intptr_t comm):
    cdef bytes _output_
    _output_ = ncclGetLastError(<Comm>comm)
    return _output_.decode()


cpdef int comm_get_async_error(intptr_t comm) except? -1:
    cdef _Result async_error
    with nogil:
        __status__ = ncclCommGetAsyncError(<Comm>comm, &async_error)
    check_status(__status__)
    return <int>async_error


cpdef int comm_count(intptr_t comm) except? -1:
    cdef int count
    with nogil:
        __status__ = ncclCommCount(<const Comm>comm, &count)
    check_status(__status__)
    return count


cpdef int comm_cu_device(intptr_t comm) except? -1:
    cdef int device
    with nogil:
        __status__ = ncclCommCuDevice(<const Comm>comm, &device)
    check_status(__status__)
    return device


cpdef int comm_user_rank(intptr_t comm) except? -1:
    cdef int rank
    with nogil:
        __status__ = ncclCommUserRank(<const Comm>comm, &rank)
    check_status(__status__)
    return rank


cpdef intptr_t comm_register(intptr_t comm, intptr_t buff, size_t size) except? 0:
    cdef void* handle
    with nogil:
        __status__ = ncclCommRegister(<const Comm>comm, <void*>buff, size, &handle)
    check_status(__status__)
    return <intptr_t>handle


cpdef comm_deregister(intptr_t comm, intptr_t handle):
    with nogil:
        __status__ = ncclCommDeregister(<const Comm>comm, <void*>handle)
    check_status(__status__)


cpdef intptr_t comm_window_register(intptr_t comm, intptr_t buff, size_t size, int win_flags) except? 0:
    cdef Window win
    with nogil:
        __status__ = ncclCommWindowRegister(<Comm>comm, <void*>buff, size, &win, win_flags)
    check_status(__status__)
    return <intptr_t>win


cpdef comm_window_deregister(intptr_t comm, intptr_t win):
    with nogil:
        __status__ = ncclCommWindowDeregister(<Comm>comm, <Window>win)
    check_status(__status__)


cpdef int red_op_create_pre_mul_sum(intptr_t scalar, int datatype, int residence, intptr_t comm) except? -1:
    cdef _RedOp op
    with nogil:
        __status__ = ncclRedOpCreatePreMulSum(&op, <void*>scalar, <_DataType>datatype, <_ScalarResidence>residence, <Comm>comm)
    check_status(__status__)
    return <int>op


cpdef red_op_destroy(int op, intptr_t comm):
    with nogil:
        __status__ = ncclRedOpDestroy(<_RedOp>op, <Comm>comm)
    check_status(__status__)


cpdef reduce(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, int root, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclReduce(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, <_RedOp>op, root, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef bcast(intptr_t buff, size_t count, int datatype, int root, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclBcast(<void*>buff, count, <_DataType>datatype, root, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef broadcast(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclBroadcast(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, root, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef all_reduce(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclAllReduce(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, <_RedOp>op, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef reduce_scatter(intptr_t sendbuff, intptr_t recvbuff, size_t recvcount, int datatype, int op, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclReduceScatter(<const void*>sendbuff, <void*>recvbuff, recvcount, <_DataType>datatype, <_RedOp>op, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef all_gather(intptr_t sendbuff, intptr_t recvbuff, size_t sendcount, int datatype, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclAllGather(<const void*>sendbuff, <void*>recvbuff, sendcount, <_DataType>datatype, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef allto_all(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclAlltoAll(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef gather(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclGather(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, root, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef scatter(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclScatter(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, root, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef send(intptr_t sendbuff, size_t count, int datatype, int peer, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclSend(<const void*>sendbuff, count, <_DataType>datatype, peer, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef recv(intptr_t recvbuff, size_t count, int datatype, int peer, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclRecv(<void*>recvbuff, count, <_DataType>datatype, peer, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef group_start():
    with nogil:
        __status__ = ncclGroupStart()
    check_status(__status__)


cpdef group_end():
    with nogil:
        __status__ = ncclGroupEnd()
    check_status(__status__)


cpdef group_simulate_end(intptr_t sim_info):
    with nogil:
        __status__ = ncclGroupSimulateEnd(<ncclSimInfo_t*>sim_info)
    check_status(__status__)
