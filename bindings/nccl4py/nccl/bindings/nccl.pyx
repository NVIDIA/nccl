# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.30.0. Do not modify it directly.

cimport cython  # NOQA
from libcpp.vector cimport vector

from ._internal.utils cimport (nested_resource, nullable_unique_ptr, get_buffer_pointer,
                              get_resource_ptr, get_nested_resource_ptr)

from enum import IntEnum as _IntEnum


from libc.stdlib cimport calloc, free, malloc
from cython cimport view
cimport cpython.buffer
cimport cpython.memoryview
cimport cpython
from libc.string cimport memcmp, memcpy
import numpy as _numpy
import pickle


cdef __from_data(data, dtype_name, expected_dtype, lowpp_type):
    # _numpy.recarray is a subclass of _numpy.ndarray, so implicitly handled here.
    if isinstance(data, lowpp_type):
        return data
    if not isinstance(data, _numpy.ndarray):
        raise TypeError("data argument must be a NumPy ndarray")
    if data.size != 1:
        raise ValueError("data array must have a size of 1")
    if data.dtype != expected_dtype:
        raise ValueError(f"data array must be of dtype {dtype_name}")
    return lowpp_type.from_ptr(data.ctypes.data, not data.flags.writeable, data)


cdef __from_buffer(buffer, size, lowpp_type):
    cdef Py_buffer view
    if cpython.PyObject_GetBuffer(buffer, &view, cpython.PyBUF_SIMPLE) != 0:
        raise TypeError("buffer argument does not support the buffer protocol")
    try:
        if view.itemsize != 1:
            raise ValueError("buffer itemsize must be 1 byte")
        if view.len != size:
            raise ValueError(f"buffer length must be {size} bytes")
        return lowpp_type.from_ptr(<intptr_t><void *>view.buf, not view.readonly, buffer)
    finally:
        cpython.PyBuffer_Release(&view)


cdef __getbuffer(object self, cpython.Py_buffer *buffer, void *ptr, int size, bint readonly):
    buffer.buf = <char *>ptr
    buffer.format = 'b'
    buffer.internal = NULL
    buffer.itemsize = 1
    buffer.len = size
    buffer.ndim = 1
    buffer.obj = self
    buffer.readonly = readonly
    buffer.shape = &buffer.len
    buffer.strides = &buffer.itemsize
    buffer.suboffsets = NULL


###############################################################################
# POD
###############################################################################

cdef _get_unique_id_dtype_offsets():
    cdef ncclUniqueId pod = ncclUniqueId()
    return _numpy.dtype({
        'names': ['internal'],
        'formats': [(_numpy.int8, 128)],
        'offsets': [
            (<intptr_t>&(pod.internal)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclUniqueId),
    })

unique_id_dtype = _get_unique_id_dtype_offsets()

cdef class UniqueId:
    """Empty-initialize an instance of `ncclUniqueId`.


    .. seealso:: `ncclUniqueId`
    """
    cdef:
        ncclUniqueId *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclUniqueId *>calloc(1, sizeof(ncclUniqueId))
        if self._ptr == NULL:
            raise MemoryError("Error allocating UniqueId")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclUniqueId *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.UniqueId object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef UniqueId other_
        if not isinstance(other, UniqueId):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclUniqueId)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclUniqueId), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclUniqueId *>malloc(sizeof(ncclUniqueId))
            if self._ptr == NULL:
                raise MemoryError("Error allocating UniqueId")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclUniqueId))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    def __getstate__(self):
        return cpython.PyBytes_FromStringAndSize(<char *><void *>self._ptr, sizeof(ncclUniqueId))

    def __setstate__(self, state):
        if not isinstance(state, bytes):
            raise TypeError(f"Invalid state type for UniqueId, expected bytes, got {type(state).__name__}")
        if len(state) != sizeof(ncclUniqueId):
            raise ValueError(f"Invalid state length for UniqueId, expected sizeof(ncclUniqueId), got {len(state)}")
        cdef char *state_ptr = cpython.PyBytes_AsString(state)
        self._ptr = <ncclUniqueId *>malloc(sizeof(ncclUniqueId))
        memcpy(<void *>self._ptr, <void *>state_ptr, sizeof(ncclUniqueId))

    @staticmethod
    def from_buffer(buffer):
        """Create an UniqueId instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclUniqueId), UniqueId)

    @staticmethod
    def from_data(data):
        """Create an UniqueId instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `unique_id_dtype` holding the data.
        """
        return __from_data(data, "unique_id_dtype", unique_id_dtype, UniqueId)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an UniqueId instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef UniqueId obj = UniqueId.__new__(UniqueId)
        if owner is None:
            obj._ptr = <ncclUniqueId *>malloc(sizeof(ncclUniqueId))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating UniqueId")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclUniqueId))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclUniqueId *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_config_dtype_offsets():
    cdef ncclConfig_t pod = ncclConfig_t()
    return _numpy.dtype({
        'names': ['size_', 'magic', 'version', 'blocking', 'cga_cluster_size', 'min_ctas', 'max_ctas', 'net_name', 'split_share', 'traffic_class', 'comm_name', 'collnet_enable', 'cta_policy', 'shrink_share', 'nvls_ctas', 'n_channels_per_net_peer', 'nvlink_centric_sched', 'graph_usage_mode', 'num_rma_ctx', 'max_p2p_peers'],
        'formats': [_numpy.uint64, _numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.intp, _numpy.int32, _numpy.int32, _numpy.intp, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.blocking)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cgaClusterSize)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.minCTAs)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.maxCTAs)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.netName)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.splitShare)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.trafficClass)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.commName)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.collnetEnable)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.CTAPolicy)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.shrinkShare)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nvlsCTAs)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nChannelsPerNetPeer)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nvlinkCentricSched)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.graphUsageMode)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.numRmaCtx)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.maxP2pPeers)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclConfig_t),
    })

config_dtype = _get_config_dtype_offsets()

cdef class Config:
    """Empty-initialize an instance of `ncclConfig_t`.


    .. seealso:: `ncclConfig_t`
    """
    cdef:
        ncclConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly
        dict _refs

    def __init__(self):
        self._ptr = <ncclConfig_t *>calloc(1, sizeof(ncclConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Config")
        self._owner = None
        self._owned = True
        self._readonly = False
        self._refs = {}

    def __dealloc__(self):
        cdef ncclConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.Config object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef Config other_
        if not isinstance(other, Config):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclConfig_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclConfig_t *>malloc(sizeof(ncclConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating Config")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].magic = val

    @property
    def version(self):
        """int: """
        return self._ptr[0].version

    @version.setter
    def version(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].version = val

    @property
    def blocking(self):
        """int: """
        return self._ptr[0].blocking

    @blocking.setter
    def blocking(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].blocking = val

    @property
    def cga_cluster_size(self):
        """int: """
        return self._ptr[0].cgaClusterSize

    @cga_cluster_size.setter
    def cga_cluster_size(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].cgaClusterSize = val

    @property
    def min_ctas(self):
        """int: """
        return self._ptr[0].minCTAs

    @min_ctas.setter
    def min_ctas(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].minCTAs = val

    @property
    def max_ctas(self):
        """int: """
        return self._ptr[0].maxCTAs

    @max_ctas.setter
    def max_ctas(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].maxCTAs = val

    @property
    def net_name(self):
        """str: """
        cdef char* ptr = <char*>self._ptr[0].netName
        if ptr:
            return cpython.PyUnicode_FromString(ptr)
        return ""

    @net_name.setter
    def net_name(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        cdef bytes buf = val.encode()
        cdef char *ptr = buf
        self._refs["net_name"] = buf
        self._ptr.netName = <char *><intptr_t>ptr

    @property
    def split_share(self):
        """int: """
        return self._ptr[0].splitShare

    @split_share.setter
    def split_share(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].splitShare = val

    @property
    def traffic_class(self):
        """int: """
        return self._ptr[0].trafficClass

    @traffic_class.setter
    def traffic_class(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].trafficClass = val

    @property
    def comm_name(self):
        """str: """
        cdef char* ptr = <char*>self._ptr[0].commName
        if ptr:
            return cpython.PyUnicode_FromString(ptr)
        return ""

    @comm_name.setter
    def comm_name(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        cdef bytes buf = val.encode()
        cdef char *ptr = buf
        self._refs["comm_name"] = buf
        self._ptr.commName = <char *><intptr_t>ptr

    @property
    def collnet_enable(self):
        """int: """
        return self._ptr[0].collnetEnable

    @collnet_enable.setter
    def collnet_enable(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].collnetEnable = val

    @property
    def cta_policy(self):
        """int: """
        return self._ptr[0].CTAPolicy

    @cta_policy.setter
    def cta_policy(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].CTAPolicy = val

    @property
    def shrink_share(self):
        """int: """
        return self._ptr[0].shrinkShare

    @shrink_share.setter
    def shrink_share(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].shrinkShare = val

    @property
    def nvls_ctas(self):
        """int: """
        return self._ptr[0].nvlsCTAs

    @nvls_ctas.setter
    def nvls_ctas(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].nvlsCTAs = val

    @property
    def n_channels_per_net_peer(self):
        """int: """
        return self._ptr[0].nChannelsPerNetPeer

    @n_channels_per_net_peer.setter
    def n_channels_per_net_peer(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].nChannelsPerNetPeer = val

    @property
    def nvlink_centric_sched(self):
        """int: """
        return self._ptr[0].nvlinkCentricSched

    @nvlink_centric_sched.setter
    def nvlink_centric_sched(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].nvlinkCentricSched = val

    @property
    def graph_usage_mode(self):
        """int: """
        return self._ptr[0].graphUsageMode

    @graph_usage_mode.setter
    def graph_usage_mode(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].graphUsageMode = val

    @property
    def num_rma_ctx(self):
        """int: """
        return self._ptr[0].numRmaCtx

    @num_rma_ctx.setter
    def num_rma_ctx(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].numRmaCtx = val

    @property
    def max_p2p_peers(self):
        """int: """
        return self._ptr[0].maxP2pPeers

    @max_p2p_peers.setter
    def max_p2p_peers(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].maxP2pPeers = val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for Config")

    @staticmethod
    def from_buffer(buffer):
        """Create an Config instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclConfig_t), Config)

    @staticmethod
    def from_data(data):
        """Create an Config instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `config_dtype` holding the data.
        """
        return __from_data(data, "config_dtype", config_dtype, Config)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an Config instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Config obj = Config.__new__(Config)
        if owner is None:
            obj._ptr = <ncclConfig_t *>malloc(sizeof(ncclConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Config")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        obj._refs = {}
        return obj


cdef _get_sim_info_dtype_offsets():
    cdef ncclSimInfo_t pod = ncclSimInfo_t()
    return _numpy.dtype({
        'names': ['size_', 'magic', 'version', 'estimated_time'],
        'formats': [_numpy.uint64, _numpy.uint32, _numpy.uint32, _numpy.float32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.estimatedTime)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclSimInfo_t),
    })

sim_info_dtype = _get_sim_info_dtype_offsets()

cdef class SimInfo:
    """Empty-initialize an instance of `ncclSimInfo_t`.


    .. seealso:: `ncclSimInfo_t`
    """
    cdef:
        ncclSimInfo_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclSimInfo_t *>calloc(1, sizeof(ncclSimInfo_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating SimInfo")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclSimInfo_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.SimInfo object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef SimInfo other_
        if not isinstance(other, SimInfo):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclSimInfo_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclSimInfo_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclSimInfo_t *>malloc(sizeof(ncclSimInfo_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating SimInfo")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclSimInfo_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This SimInfo instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This SimInfo instance is read-only")
        self._ptr[0].magic = val

    @property
    def version(self):
        """int: """
        return self._ptr[0].version

    @version.setter
    def version(self, val):
        if self._readonly:
            raise ValueError("This SimInfo instance is read-only")
        self._ptr[0].version = val

    @property
    def estimated_time(self):
        """float: """
        return self._ptr[0].estimatedTime

    @estimated_time.setter
    def estimated_time(self, val):
        if self._readonly:
            raise ValueError("This SimInfo instance is read-only")
        self._ptr[0].estimatedTime = val

    def __getstate__(self):
        return cpython.PyBytes_FromStringAndSize(<char *><void *>self._ptr, sizeof(ncclSimInfo_t))

    def __setstate__(self, state):
        if not isinstance(state, bytes):
            raise TypeError(f"Invalid state type for SimInfo, expected bytes, got {type(state).__name__}")
        if len(state) != sizeof(ncclSimInfo_t):
            raise ValueError(f"Invalid state length for SimInfo, expected sizeof(ncclSimInfo_t), got {len(state)}")
        cdef char *state_ptr = cpython.PyBytes_AsString(state)
        self._ptr = <ncclSimInfo_t *>malloc(sizeof(ncclSimInfo_t))
        memcpy(<void *>self._ptr, <void *>state_ptr, sizeof(ncclSimInfo_t))

    @staticmethod
    def from_buffer(buffer):
        """Create an SimInfo instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclSimInfo_t), SimInfo)

    @staticmethod
    def from_data(data):
        """Create an SimInfo instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `sim_info_dtype` holding the data.
        """
        return __from_data(data, "sim_info_dtype", sim_info_dtype, SimInfo)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an SimInfo instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef SimInfo obj = SimInfo.__new__(SimInfo)
        if owner is None:
            obj._ptr = <ncclSimInfo_t *>malloc(sizeof(ncclSimInfo_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating SimInfo")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclSimInfo_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclSimInfo_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_wait_signal_desc_dtype_offsets():
    cdef ncclWaitSignalDesc_t pod = ncclWaitSignalDesc_t()
    return _numpy.dtype({
        'names': ['op_cnt', 'peer', 'sig_idx', 'ctx'],
        'formats': [_numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.opCnt)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.peer)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.sigIdx)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ctx)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclWaitSignalDesc_t),
    })

wait_signal_desc_dtype = _get_wait_signal_desc_dtype_offsets()

cdef class WaitSignalDesc:
    """Empty-initialize an instance of `ncclWaitSignalDesc_t`.


    .. seealso:: `ncclWaitSignalDesc_t`
    """
    cdef:
        ncclWaitSignalDesc_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclWaitSignalDesc_t *>calloc(1, sizeof(ncclWaitSignalDesc_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating WaitSignalDesc")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclWaitSignalDesc_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.WaitSignalDesc object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef WaitSignalDesc other_
        if not isinstance(other, WaitSignalDesc):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclWaitSignalDesc_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclWaitSignalDesc_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclWaitSignalDesc_t *>malloc(sizeof(ncclWaitSignalDesc_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating WaitSignalDesc")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclWaitSignalDesc_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def op_cnt(self):
        """int: """
        return self._ptr[0].opCnt

    @op_cnt.setter
    def op_cnt(self, val):
        if self._readonly:
            raise ValueError("This WaitSignalDesc instance is read-only")
        self._ptr[0].opCnt = val

    @property
    def peer(self):
        """int: """
        return self._ptr[0].peer

    @peer.setter
    def peer(self, val):
        if self._readonly:
            raise ValueError("This WaitSignalDesc instance is read-only")
        self._ptr[0].peer = val

    @property
    def sig_idx(self):
        """int: """
        return self._ptr[0].sigIdx

    @sig_idx.setter
    def sig_idx(self, val):
        if self._readonly:
            raise ValueError("This WaitSignalDesc instance is read-only")
        self._ptr[0].sigIdx = val

    @property
    def ctx(self):
        """int: """
        return self._ptr[0].ctx

    @ctx.setter
    def ctx(self, val):
        if self._readonly:
            raise ValueError("This WaitSignalDesc instance is read-only")
        self._ptr[0].ctx = val

    def __getstate__(self):
        return cpython.PyBytes_FromStringAndSize(<char *><void *>self._ptr, sizeof(ncclWaitSignalDesc_t))

    def __setstate__(self, state):
        if not isinstance(state, bytes):
            raise TypeError(f"Invalid state type for WaitSignalDesc, expected bytes, got {type(state).__name__}")
        if len(state) != sizeof(ncclWaitSignalDesc_t):
            raise ValueError(f"Invalid state length for WaitSignalDesc, expected sizeof(ncclWaitSignalDesc_t), got {len(state)}")
        cdef char *state_ptr = cpython.PyBytes_AsString(state)
        self._ptr = <ncclWaitSignalDesc_t *>malloc(sizeof(ncclWaitSignalDesc_t))
        memcpy(<void *>self._ptr, <void *>state_ptr, sizeof(ncclWaitSignalDesc_t))

    @staticmethod
    def from_buffer(buffer):
        """Create an WaitSignalDesc instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclWaitSignalDesc_t), WaitSignalDesc)

    @staticmethod
    def from_data(data):
        """Create an WaitSignalDesc instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `wait_signal_desc_dtype` holding the data.
        """
        return __from_data(data, "wait_signal_desc_dtype", wait_signal_desc_dtype, WaitSignalDesc)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an WaitSignalDesc instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef WaitSignalDesc obj = WaitSignalDesc.__new__(WaitSignalDesc)
        if owner is None:
            obj._ptr = <ncclWaitSignalDesc_t *>malloc(sizeof(ncclWaitSignalDesc_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating WaitSignalDesc")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclWaitSignalDesc_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclWaitSignalDesc_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_comm_properties_dtype_offsets():
    cdef ncclCommProperties_t pod = ncclCommProperties_t()
    return _numpy.dtype({
        'names': ['size_', 'magic', 'version', 'rank', 'n_ranks', 'cuda_dev', 'nvml_dev', 'device_api_support', 'multimem_support', 'gin_type', 'n_lsa_teams', 'host_rma_support', 'railed_gin_type'],
        'formats': [_numpy.uint64, _numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.uint8, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.rank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nRanks)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cudaDev)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nvmlDev)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.deviceApiSupport)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.multimemSupport)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginType)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nLsaTeams)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.hostRmaSupport)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.railedGinType)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclCommProperties_t),
    })

comm_properties_dtype = _get_comm_properties_dtype_offsets()

cdef class CommProperties:
    """Empty-initialize an instance of `ncclCommProperties_t`.


    .. seealso:: `ncclCommProperties_t`
    """
    cdef:
        ncclCommProperties_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclCommProperties_t *>calloc(1, sizeof(ncclCommProperties_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating CommProperties")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclCommProperties_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.CommProperties object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef CommProperties other_
        if not isinstance(other, CommProperties):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclCommProperties_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclCommProperties_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclCommProperties_t *>malloc(sizeof(ncclCommProperties_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating CommProperties")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclCommProperties_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].magic = val

    @property
    def version(self):
        """int: """
        return self._ptr[0].version

    @version.setter
    def version(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].version = val

    @property
    def rank(self):
        """int: """
        return self._ptr[0].rank

    @rank.setter
    def rank(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].rank = val

    @property
    def n_ranks(self):
        """int: """
        return self._ptr[0].nRanks

    @n_ranks.setter
    def n_ranks(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].nRanks = val

    @property
    def cuda_dev(self):
        """int: """
        return self._ptr[0].cudaDev

    @cuda_dev.setter
    def cuda_dev(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].cudaDev = val

    @property
    def nvml_dev(self):
        """int: """
        return self._ptr[0].nvmlDev

    @nvml_dev.setter
    def nvml_dev(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].nvmlDev = val

    @property
    def device_api_support(self):
        """int: """
        return self._ptr[0].deviceApiSupport

    @device_api_support.setter
    def device_api_support(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].deviceApiSupport = val

    @property
    def multimem_support(self):
        """int: """
        return self._ptr[0].multimemSupport

    @multimem_support.setter
    def multimem_support(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].multimemSupport = val

    @property
    def gin_type(self):
        """int: """
        return <int>(self._ptr[0].ginType)

    @gin_type.setter
    def gin_type(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].ginType = <ncclGinType_t><int>val

    @property
    def n_lsa_teams(self):
        """int: """
        return self._ptr[0].nLsaTeams

    @n_lsa_teams.setter
    def n_lsa_teams(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].nLsaTeams = val

    @property
    def host_rma_support(self):
        """int: """
        return self._ptr[0].hostRmaSupport

    @host_rma_support.setter
    def host_rma_support(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].hostRmaSupport = val

    @property
    def railed_gin_type(self):
        """int: """
        return <int>(self._ptr[0].railedGinType)

    @railed_gin_type.setter
    def railed_gin_type(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].railedGinType = <ncclGinType_t><int>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for CommProperties")

    @staticmethod
    def from_buffer(buffer):
        """Create an CommProperties instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclCommProperties_t), CommProperties)

    @staticmethod
    def from_data(data):
        """Create an CommProperties instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `comm_properties_dtype` holding the data.
        """
        return __from_data(data, "comm_properties_dtype", comm_properties_dtype, CommProperties)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an CommProperties instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef CommProperties obj = CommProperties.__new__(CommProperties)
        if owner is None:
            obj._ptr = <ncclCommProperties_t *>malloc(sizeof(ncclCommProperties_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating CommProperties")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclCommProperties_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclCommProperties_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_dev_resource_requirements_dtype_offsets():
    cdef ncclDevResourceRequirements_t pod = ncclDevResourceRequirements_t()
    return _numpy.dtype({
        'names': ['next', 'buffer_size', 'buffer_align', 'out_buffer_handle', 'gin_signal_count', 'gin_counter_count', 'out_gin_signal_start', 'out_gin_counter_start'],
        'formats': [_numpy.intp, _numpy.uint64, _numpy.uint64, _numpy.intp, _numpy.int32, _numpy.int32, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.next)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.bufferSize)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.bufferAlign)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.outBufferHandle)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginSignalCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginCounterCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.outGinSignalStart)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.outGinCounterStart)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclDevResourceRequirements_t),
    })

dev_resource_requirements_dtype = _get_dev_resource_requirements_dtype_offsets()

cdef class DevResourceRequirements:
    """Empty-initialize an instance of `ncclDevResourceRequirements_t`.


    .. seealso:: `ncclDevResourceRequirements_t`
    """
    cdef:
        ncclDevResourceRequirements_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclDevResourceRequirements_t *>calloc(1, sizeof(ncclDevResourceRequirements_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating DevResourceRequirements")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclDevResourceRequirements_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.DevResourceRequirements object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef DevResourceRequirements other_
        if not isinstance(other, DevResourceRequirements):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclDevResourceRequirements_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclDevResourceRequirements_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclDevResourceRequirements_t *>malloc(sizeof(ncclDevResourceRequirements_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DevResourceRequirements")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclDevResourceRequirements_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def next(self):
        """int: """
        return <intptr_t>(self._ptr[0].next)

    @next.setter
    def next(self, val):
        if self._readonly:
            raise ValueError("This DevResourceRequirements instance is read-only")
        self._ptr[0].next = <void *><intptr_t>val

    @property
    def buffer_size(self):
        """int: """
        return self._ptr[0].bufferSize

    @buffer_size.setter
    def buffer_size(self, val):
        if self._readonly:
            raise ValueError("This DevResourceRequirements instance is read-only")
        self._ptr[0].bufferSize = val

    @property
    def buffer_align(self):
        """int: """
        return self._ptr[0].bufferAlign

    @buffer_align.setter
    def buffer_align(self, val):
        if self._readonly:
            raise ValueError("This DevResourceRequirements instance is read-only")
        self._ptr[0].bufferAlign = val

    @property
    def out_buffer_handle(self):
        """int: """
        return <intptr_t>(self._ptr[0].outBufferHandle)

    @out_buffer_handle.setter
    def out_buffer_handle(self, val):
        if self._readonly:
            raise ValueError("This DevResourceRequirements instance is read-only")
        self._ptr[0].outBufferHandle = <ncclDevResourceHandle_t*><intptr_t>val

    @property
    def gin_signal_count(self):
        """int: """
        return self._ptr[0].ginSignalCount

    @gin_signal_count.setter
    def gin_signal_count(self, val):
        if self._readonly:
            raise ValueError("This DevResourceRequirements instance is read-only")
        self._ptr[0].ginSignalCount = val

    @property
    def gin_counter_count(self):
        """int: """
        return self._ptr[0].ginCounterCount

    @gin_counter_count.setter
    def gin_counter_count(self, val):
        if self._readonly:
            raise ValueError("This DevResourceRequirements instance is read-only")
        self._ptr[0].ginCounterCount = val

    @property
    def out_gin_signal_start(self):
        """int: """
        return <intptr_t>(self._ptr[0].outGinSignalStart)

    @out_gin_signal_start.setter
    def out_gin_signal_start(self, val):
        if self._readonly:
            raise ValueError("This DevResourceRequirements instance is read-only")
        self._ptr[0].outGinSignalStart = <ncclGinSignal_t*><intptr_t>val

    @property
    def out_gin_counter_start(self):
        """int: """
        return <intptr_t>(self._ptr[0].outGinCounterStart)

    @out_gin_counter_start.setter
    def out_gin_counter_start(self, val):
        if self._readonly:
            raise ValueError("This DevResourceRequirements instance is read-only")
        self._ptr[0].outGinCounterStart = <ncclGinCounter_t*><intptr_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for DevResourceRequirements")

    @staticmethod
    def from_buffer(buffer):
        """Create an DevResourceRequirements instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclDevResourceRequirements_t), DevResourceRequirements)

    @staticmethod
    def from_data(data):
        """Create an DevResourceRequirements instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dev_resource_requirements_dtype` holding the data.
        """
        return __from_data(data, "dev_resource_requirements_dtype", dev_resource_requirements_dtype, DevResourceRequirements)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an DevResourceRequirements instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef DevResourceRequirements obj = DevResourceRequirements.__new__(DevResourceRequirements)
        if owner is None:
            obj._ptr = <ncclDevResourceRequirements_t *>malloc(sizeof(ncclDevResourceRequirements_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DevResourceRequirements")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclDevResourceRequirements_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclDevResourceRequirements_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_team_dtype_offsets():
    cdef ncclTeam_t pod = ncclTeam_t()
    return _numpy.dtype({
        'names': ['n_ranks', 'rank', 'stride'],
        'formats': [_numpy.int32, _numpy.int32, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.nRanks)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.rank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.stride)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclTeam_t),
    })

team_dtype = _get_team_dtype_offsets()

cdef class Team:
    """Empty-initialize an instance of `ncclTeam_t`.


    .. seealso:: `ncclTeam_t`
    """
    cdef:
        ncclTeam_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclTeam_t *>calloc(1, sizeof(ncclTeam_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Team")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclTeam_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.Team object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef Team other_
        if not isinstance(other, Team):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclTeam_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclTeam_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclTeam_t *>malloc(sizeof(ncclTeam_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating Team")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclTeam_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def n_ranks(self):
        """int: """
        return self._ptr[0].nRanks

    @n_ranks.setter
    def n_ranks(self, val):
        if self._readonly:
            raise ValueError("This Team instance is read-only")
        self._ptr[0].nRanks = val

    @property
    def rank(self):
        """int: """
        return self._ptr[0].rank

    @rank.setter
    def rank(self, val):
        if self._readonly:
            raise ValueError("This Team instance is read-only")
        self._ptr[0].rank = val

    @property
    def stride(self):
        """int: """
        return self._ptr[0].stride

    @stride.setter
    def stride(self, val):
        if self._readonly:
            raise ValueError("This Team instance is read-only")
        self._ptr[0].stride = val

    def __getstate__(self):
        return cpython.PyBytes_FromStringAndSize(<char *><void *>self._ptr, sizeof(ncclTeam_t))

    def __setstate__(self, state):
        if not isinstance(state, bytes):
            raise TypeError(f"Invalid state type for Team, expected bytes, got {type(state).__name__}")
        if len(state) != sizeof(ncclTeam_t):
            raise ValueError(f"Invalid state length for Team, expected sizeof(ncclTeam_t), got {len(state)}")
        cdef char *state_ptr = cpython.PyBytes_AsString(state)
        self._ptr = <ncclTeam_t *>malloc(sizeof(ncclTeam_t))
        memcpy(<void *>self._ptr, <void *>state_ptr, sizeof(ncclTeam_t))

    @staticmethod
    def from_buffer(buffer):
        """Create an Team instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclTeam_t), Team)

    @staticmethod
    def from_data(data):
        """Create an Team instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `team_dtype` holding the data.
        """
        return __from_data(data, "team_dtype", team_dtype, Team)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an Team instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Team obj = Team.__new__(Team)
        if owner is None:
            obj._ptr = <ncclTeam_t *>malloc(sizeof(ncclTeam_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Team")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclTeam_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclTeam_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_multimem_handle_dtype_offsets():
    cdef ncclMultimemHandle_t pod = ncclMultimemHandle_t()
    return _numpy.dtype({
        'names': ['mc_base_ptr'],
        'formats': [_numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.mcBasePtr)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclMultimemHandle_t),
    })

multimem_handle_dtype = _get_multimem_handle_dtype_offsets()

cdef class MultimemHandle:
    """Empty-initialize an instance of `ncclMultimemHandle_t`.


    .. seealso:: `ncclMultimemHandle_t`
    """
    cdef:
        ncclMultimemHandle_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclMultimemHandle_t *>calloc(1, sizeof(ncclMultimemHandle_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating MultimemHandle")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclMultimemHandle_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.MultimemHandle object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef MultimemHandle other_
        if not isinstance(other, MultimemHandle):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclMultimemHandle_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclMultimemHandle_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclMultimemHandle_t *>malloc(sizeof(ncclMultimemHandle_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating MultimemHandle")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclMultimemHandle_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def mc_base_ptr(self):
        """int: """
        return <intptr_t>(self._ptr[0].mcBasePtr)

    @mc_base_ptr.setter
    def mc_base_ptr(self, val):
        if self._readonly:
            raise ValueError("This MultimemHandle instance is read-only")
        self._ptr[0].mcBasePtr = <void *><intptr_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for MultimemHandle")

    @staticmethod
    def from_buffer(buffer):
        """Create an MultimemHandle instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclMultimemHandle_t), MultimemHandle)

    @staticmethod
    def from_data(data):
        """Create an MultimemHandle instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `multimem_handle_dtype` holding the data.
        """
        return __from_data(data, "multimem_handle_dtype", multimem_handle_dtype, MultimemHandle)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an MultimemHandle instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef MultimemHandle obj = MultimemHandle.__new__(MultimemHandle)
        if owner is None:
            obj._ptr = <ncclMultimemHandle_t *>malloc(sizeof(ncclMultimemHandle_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating MultimemHandle")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclMultimemHandle_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclMultimemHandle_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_window_vidmem_dtype_offsets():
    cdef ncclWindow_vidmem_t pod = ncclWindow_vidmem_t()
    return _numpy.dtype({
        'names': ['win_host', 'lsa_flat_base', 'lsa_rank', 'world_rank', 'stride4g', 'mc_offset4k', 'gin_offset4k', 'gin_wins'],
        'formats': [_numpy.intp, _numpy.intp, _numpy.int32, _numpy.int32, _numpy.uint32, _numpy.uint32, _numpy.uint32, (_numpy.intp, 4)],
        'offsets': [
            (<intptr_t>&(pod.winHost)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaFlatBase)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaRank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.worldRank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.stride4G)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.mcOffset4K)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginOffset4K)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginWins)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclWindow_vidmem_t),
    })

window_vidmem_dtype = _get_window_vidmem_dtype_offsets()

cdef class Window_vidmem:
    """Empty-initialize an instance of `ncclWindow_vidmem_t`.


    .. seealso:: `ncclWindow_vidmem_t`
    """
    cdef:
        ncclWindow_vidmem_t *_ptr
        object _owner
        bint _owned
        bint _readonly
        dict _refs

    def __init__(self):
        self._ptr = <ncclWindow_vidmem_t *>calloc(1, sizeof(ncclWindow_vidmem_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Window_vidmem")
        self._owner = None
        self._owned = True
        self._readonly = False
        self._refs = {}

    def __dealloc__(self):
        cdef ncclWindow_vidmem_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.Window_vidmem object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef Window_vidmem other_
        if not isinstance(other, Window_vidmem):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclWindow_vidmem_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclWindow_vidmem_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclWindow_vidmem_t *>malloc(sizeof(ncclWindow_vidmem_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating Window_vidmem")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclWindow_vidmem_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def win_host(self):
        """int: """
        return <intptr_t>(self._ptr[0].winHost)

    @win_host.setter
    def win_host(self, val):
        if self._readonly:
            raise ValueError("This Window_vidmem instance is read-only")
        self._ptr[0].winHost = <void *><intptr_t>val

    @property
    def lsa_flat_base(self):
        """str: """
        cdef char* ptr = <char*>self._ptr[0].lsaFlatBase
        if ptr:
            return cpython.PyUnicode_FromString(ptr)
        return ""

    @lsa_flat_base.setter
    def lsa_flat_base(self, val):
        if self._readonly:
            raise ValueError("This Window_vidmem instance is read-only")
        cdef bytes buf = val.encode()
        cdef char *ptr = buf
        self._refs["lsa_flat_base"] = buf
        self._ptr.lsaFlatBase = <char *><intptr_t>ptr

    @property
    def lsa_rank(self):
        """int: """
        return self._ptr[0].lsaRank

    @lsa_rank.setter
    def lsa_rank(self, val):
        if self._readonly:
            raise ValueError("This Window_vidmem instance is read-only")
        self._ptr[0].lsaRank = val

    @property
    def world_rank(self):
        """int: """
        return self._ptr[0].worldRank

    @world_rank.setter
    def world_rank(self, val):
        if self._readonly:
            raise ValueError("This Window_vidmem instance is read-only")
        self._ptr[0].worldRank = val

    @property
    def stride4g(self):
        """int: """
        return self._ptr[0].stride4G

    @stride4g.setter
    def stride4g(self, val):
        if self._readonly:
            raise ValueError("This Window_vidmem instance is read-only")
        self._ptr[0].stride4G = val

    @property
    def mc_offset4k(self):
        """int: """
        return self._ptr[0].mcOffset4K

    @mc_offset4k.setter
    def mc_offset4k(self, val):
        if self._readonly:
            raise ValueError("This Window_vidmem instance is read-only")
        self._ptr[0].mcOffset4K = val

    @property
    def gin_offset4k(self):
        """int: """
        return self._ptr[0].ginOffset4K

    @gin_offset4k.setter
    def gin_offset4k(self, val):
        if self._readonly:
            raise ValueError("This Window_vidmem instance is read-only")
        self._ptr[0].ginOffset4K = val

    @property
    def gin_wins(self):
        """~_numpy.intp: (array of length 4)."""
        cdef view.array arr = view.array(shape=(4,), itemsize=sizeof(intptr_t), format="q", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].ginWins))
        return _numpy.asarray(arr)

    @gin_wins.setter
    def gin_wins(self, val):
        if self._readonly:
            raise ValueError("This Window_vidmem instance is read-only")
        if len(val) != 4:
            raise ValueError(f"Expected length { 4 } for field gin_wins, got {len(val)}")
        cdef view.array arr = view.array(shape=(4,), itemsize=sizeof(intptr_t), format="q", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.intp)
        memcpy(<void *>(&(self._ptr[0].ginWins)), <void *>(arr.data), sizeof(intptr_t) * len(val))

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for Window_vidmem")

    @staticmethod
    def from_buffer(buffer):
        """Create an Window_vidmem instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclWindow_vidmem_t), Window_vidmem)

    @staticmethod
    def from_data(data):
        """Create an Window_vidmem instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `window_vidmem_dtype` holding the data.
        """
        return __from_data(data, "window_vidmem_dtype", window_vidmem_dtype, Window_vidmem)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an Window_vidmem instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Window_vidmem obj = Window_vidmem.__new__(Window_vidmem)
        if owner is None:
            obj._ptr = <ncclWindow_vidmem_t *>malloc(sizeof(ncclWindow_vidmem_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Window_vidmem")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclWindow_vidmem_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclWindow_vidmem_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        obj._refs = {}
        return obj


cdef _get_lsa_barrier_handle_dtype_offsets():
    cdef ncclLsaBarrierHandle_t pod = ncclLsaBarrierHandle_t()
    return _numpy.dtype({
        'names': ['buf_handle', 'n_barriers'],
        'formats': [_numpy.uint32, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.bufHandle)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nBarriers)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclLsaBarrierHandle_t),
    })

lsa_barrier_handle_dtype = _get_lsa_barrier_handle_dtype_offsets()

cdef class LsaBarrierHandle:
    """Empty-initialize an instance of `ncclLsaBarrierHandle_t`.


    .. seealso:: `ncclLsaBarrierHandle_t`
    """
    cdef:
        ncclLsaBarrierHandle_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclLsaBarrierHandle_t *>calloc(1, sizeof(ncclLsaBarrierHandle_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating LsaBarrierHandle")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclLsaBarrierHandle_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.LsaBarrierHandle object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef LsaBarrierHandle other_
        if not isinstance(other, LsaBarrierHandle):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclLsaBarrierHandle_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclLsaBarrierHandle_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclLsaBarrierHandle_t *>malloc(sizeof(ncclLsaBarrierHandle_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating LsaBarrierHandle")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclLsaBarrierHandle_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def buf_handle(self):
        """int: """
        return <uint32_t>(self._ptr[0].bufHandle)

    @buf_handle.setter
    def buf_handle(self, val):
        if self._readonly:
            raise ValueError("This LsaBarrierHandle instance is read-only")
        self._ptr[0].bufHandle = <ncclDevResourceHandle_t><uint32_t>val

    @property
    def n_barriers(self):
        """int: """
        return self._ptr[0].nBarriers

    @n_barriers.setter
    def n_barriers(self, val):
        if self._readonly:
            raise ValueError("This LsaBarrierHandle instance is read-only")
        self._ptr[0].nBarriers = val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for LsaBarrierHandle")

    @staticmethod
    def from_buffer(buffer):
        """Create an LsaBarrierHandle instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclLsaBarrierHandle_t), LsaBarrierHandle)

    @staticmethod
    def from_data(data):
        """Create an LsaBarrierHandle instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `lsa_barrier_handle_dtype` holding the data.
        """
        return __from_data(data, "lsa_barrier_handle_dtype", lsa_barrier_handle_dtype, LsaBarrierHandle)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an LsaBarrierHandle instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef LsaBarrierHandle obj = LsaBarrierHandle.__new__(LsaBarrierHandle)
        if owner is None:
            obj._ptr = <ncclLsaBarrierHandle_t *>malloc(sizeof(ncclLsaBarrierHandle_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating LsaBarrierHandle")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclLsaBarrierHandle_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclLsaBarrierHandle_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_gin_barrier_handle_dtype_offsets():
    cdef ncclGinBarrierHandle_t pod = ncclGinBarrierHandle_t()
    return _numpy.dtype({
        'names': ['signal0', 'unused'],
        'formats': [_numpy.uint32, _numpy.uint32],
        'offsets': [
            (<intptr_t>&(pod.signal0)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.unused)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclGinBarrierHandle_t),
    })

gin_barrier_handle_dtype = _get_gin_barrier_handle_dtype_offsets()

cdef class GinBarrierHandle:
    """Empty-initialize an instance of `ncclGinBarrierHandle_t`.


    .. seealso:: `ncclGinBarrierHandle_t`
    """
    cdef:
        ncclGinBarrierHandle_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclGinBarrierHandle_t *>calloc(1, sizeof(ncclGinBarrierHandle_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating GinBarrierHandle")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclGinBarrierHandle_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.GinBarrierHandle object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef GinBarrierHandle other_
        if not isinstance(other, GinBarrierHandle):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclGinBarrierHandle_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclGinBarrierHandle_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclGinBarrierHandle_t *>malloc(sizeof(ncclGinBarrierHandle_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating GinBarrierHandle")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclGinBarrierHandle_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def signal0(self):
        """int: """
        return <uint32_t>(self._ptr[0].signal0)

    @signal0.setter
    def signal0(self, val):
        if self._readonly:
            raise ValueError("This GinBarrierHandle instance is read-only")
        self._ptr[0].signal0 = <ncclGinSignal_t><uint32_t>val

    @property
    def unused(self):
        """int: """
        return <uint32_t>(self._ptr[0].unused)

    @unused.setter
    def unused(self, val):
        if self._readonly:
            raise ValueError("This GinBarrierHandle instance is read-only")
        self._ptr[0].unused = <ncclDevResourceHandle_t><uint32_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for GinBarrierHandle")

    @staticmethod
    def from_buffer(buffer):
        """Create an GinBarrierHandle instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclGinBarrierHandle_t), GinBarrierHandle)

    @staticmethod
    def from_data(data):
        """Create an GinBarrierHandle instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `gin_barrier_handle_dtype` holding the data.
        """
        return __from_data(data, "gin_barrier_handle_dtype", gin_barrier_handle_dtype, GinBarrierHandle)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an GinBarrierHandle instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef GinBarrierHandle obj = GinBarrierHandle.__new__(GinBarrierHandle)
        if owner is None:
            obj._ptr = <ncclGinBarrierHandle_t *>malloc(sizeof(ncclGinBarrierHandle_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating GinBarrierHandle")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclGinBarrierHandle_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclGinBarrierHandle_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_team_requirements_dtype_offsets():
    cdef ncclTeamRequirements_t pod = ncclTeamRequirements_t()
    return _numpy.dtype({
        'names': ['next', 'team', 'multimem', 'out_multimem_handle'],
        'formats': [_numpy.intp, team_dtype, _numpy.uint8, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.next)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.team)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.multimem)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.outMultimemHandle)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclTeamRequirements_t),
    })

team_requirements_dtype = _get_team_requirements_dtype_offsets()

cdef class TeamRequirements:
    """Empty-initialize an instance of `ncclTeamRequirements_t`.


    .. seealso:: `ncclTeamRequirements_t`
    """
    cdef:
        ncclTeamRequirements_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclTeamRequirements_t *>calloc(1, sizeof(ncclTeamRequirements_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating TeamRequirements")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclTeamRequirements_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.TeamRequirements object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef TeamRequirements other_
        if not isinstance(other, TeamRequirements):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclTeamRequirements_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclTeamRequirements_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclTeamRequirements_t *>malloc(sizeof(ncclTeamRequirements_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating TeamRequirements")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclTeamRequirements_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def team(self):
        """Team: """
        return Team.from_ptr(<intptr_t>&(self._ptr[0].team), self._readonly, self)

    @team.setter
    def team(self, val):
        if self._readonly:
            raise ValueError("This TeamRequirements instance is read-only")
        cdef Team val_ = val
        memcpy(<void *>&(self._ptr[0].team), <void *>(val_._get_ptr()), sizeof(ncclTeam_t) * 1)

    @property
    def next(self):
        """int: """
        return <intptr_t>(self._ptr[0].next)

    @next.setter
    def next(self, val):
        if self._readonly:
            raise ValueError("This TeamRequirements instance is read-only")
        self._ptr[0].next = <void *><intptr_t>val

    @property
    def multimem(self):
        """int: """
        return self._ptr[0].multimem

    @multimem.setter
    def multimem(self, val):
        if self._readonly:
            raise ValueError("This TeamRequirements instance is read-only")
        self._ptr[0].multimem = val

    @property
    def out_multimem_handle(self):
        """int: """
        return <intptr_t>(self._ptr[0].outMultimemHandle)

    @out_multimem_handle.setter
    def out_multimem_handle(self, val):
        if self._readonly:
            raise ValueError("This TeamRequirements instance is read-only")
        self._ptr[0].outMultimemHandle = <ncclMultimemHandle_t*><intptr_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for TeamRequirements")

    @staticmethod
    def from_buffer(buffer):
        """Create an TeamRequirements instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclTeamRequirements_t), TeamRequirements)

    @staticmethod
    def from_data(data):
        """Create an TeamRequirements instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `team_requirements_dtype` holding the data.
        """
        return __from_data(data, "team_requirements_dtype", team_requirements_dtype, TeamRequirements)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an TeamRequirements instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef TeamRequirements obj = TeamRequirements.__new__(TeamRequirements)
        if owner is None:
            obj._ptr = <ncclTeamRequirements_t *>malloc(sizeof(ncclTeamRequirements_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating TeamRequirements")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclTeamRequirements_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclTeamRequirements_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_dev_comm_dtype_offsets():
    cdef ncclDevComm_t pod = ncclDevComm_t()
    return _numpy.dtype({
        'names': ['rank', 'n_ranks', 'n_ranks_rcp32', 'lsa_rank', 'lsa_size', 'lsa_size_rcp32', 'window_table', 'resource_window', 'resource_window_inlined', 'lsa_multimem', 'lsa_barrier', 'rail_gin_barrier', 'gin_connection_count', 'gin_net_device_types', 'gin_handles', 'gin_signal_base', 'gin_signal_count', 'gin_counter_base', 'gin_counter_count', 'gin_signal_shadows', 'gin_context_count', 'gin_context_base', 'gin_is_railed', 'abort_flag', 'hybrid_lsa_barrier', 'hybrid_rail_gin_barrier', 'world_gin_barrier'],
        'formats': [_numpy.int32, _numpy.int32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.uint32, _numpy.intp, _numpy.intp, window_vidmem_dtype, multimem_handle_dtype, lsa_barrier_handle_dtype, gin_barrier_handle_dtype, _numpy.uint8, (_numpy.uint8, 4), (_numpy.int64, 4), _numpy.uint32, _numpy.int32, _numpy.uint32, _numpy.int32, _numpy.intp, _numpy.uint32, _numpy.uint32, _numpy.uint8, _numpy.intp, lsa_barrier_handle_dtype, gin_barrier_handle_dtype, gin_barrier_handle_dtype],
        'offsets': [
            (<intptr_t>&(pod.rank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nRanks)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nRanks_rcp32)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaRank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaSize)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaSize_rcp32)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.windowTable)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.resourceWindow)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.resourceWindow_inlined)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaMultimem)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.railGinBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginConnectionCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginNetDeviceTypes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginHandles)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginSignalBase)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginSignalCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginCounterBase)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginCounterCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginSignalShadows)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginContextCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginContextBase)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginIsRailed)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.abortFlag)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.hybridLsaBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.hybridRailGinBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.worldGinBarrier)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclDevComm_t),
    })

dev_comm_dtype = _get_dev_comm_dtype_offsets()

cdef class DevComm:
    """Empty-initialize an instance of `ncclDevComm_t`.


    .. seealso:: `ncclDevComm_t`
    """
    cdef:
        ncclDevComm_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclDevComm_t *>calloc(1, sizeof(ncclDevComm_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating DevComm")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclDevComm_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.DevComm object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef DevComm other_
        if not isinstance(other, DevComm):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclDevComm_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclDevComm_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclDevComm_t *>malloc(sizeof(ncclDevComm_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DevComm")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclDevComm_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def resource_window_inlined(self):
        """Window_vidmem: """
        return Window_vidmem.from_ptr(<intptr_t>&(self._ptr[0].resourceWindow_inlined), self._readonly, self)

    @resource_window_inlined.setter
    def resource_window_inlined(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef Window_vidmem val_ = val
        memcpy(<void *>&(self._ptr[0].resourceWindow_inlined), <void *>(val_._get_ptr()), sizeof(ncclWindow_vidmem_t) * 1)

    @property
    def lsa_multimem(self):
        """MultimemHandle: """
        return MultimemHandle.from_ptr(<intptr_t>&(self._ptr[0].lsaMultimem), self._readonly, self)

    @lsa_multimem.setter
    def lsa_multimem(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef MultimemHandle val_ = val
        memcpy(<void *>&(self._ptr[0].lsaMultimem), <void *>(val_._get_ptr()), sizeof(ncclMultimemHandle_t) * 1)

    @property
    def lsa_barrier(self):
        """LsaBarrierHandle: """
        return LsaBarrierHandle.from_ptr(<intptr_t>&(self._ptr[0].lsaBarrier), self._readonly, self)

    @lsa_barrier.setter
    def lsa_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef LsaBarrierHandle val_ = val
        memcpy(<void *>&(self._ptr[0].lsaBarrier), <void *>(val_._get_ptr()), sizeof(ncclLsaBarrierHandle_t) * 1)

    @property
    def rail_gin_barrier(self):
        """GinBarrierHandle: """
        return GinBarrierHandle.from_ptr(<intptr_t>&(self._ptr[0].railGinBarrier), self._readonly, self)

    @rail_gin_barrier.setter
    def rail_gin_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef GinBarrierHandle val_ = val
        memcpy(<void *>&(self._ptr[0].railGinBarrier), <void *>(val_._get_ptr()), sizeof(ncclGinBarrierHandle_t) * 1)

    @property
    def hybrid_lsa_barrier(self):
        """LsaBarrierHandle: """
        return LsaBarrierHandle.from_ptr(<intptr_t>&(self._ptr[0].hybridLsaBarrier), self._readonly, self)

    @hybrid_lsa_barrier.setter
    def hybrid_lsa_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef LsaBarrierHandle val_ = val
        memcpy(<void *>&(self._ptr[0].hybridLsaBarrier), <void *>(val_._get_ptr()), sizeof(ncclLsaBarrierHandle_t) * 1)

    @property
    def hybrid_rail_gin_barrier(self):
        """GinBarrierHandle: """
        return GinBarrierHandle.from_ptr(<intptr_t>&(self._ptr[0].hybridRailGinBarrier), self._readonly, self)

    @hybrid_rail_gin_barrier.setter
    def hybrid_rail_gin_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef GinBarrierHandle val_ = val
        memcpy(<void *>&(self._ptr[0].hybridRailGinBarrier), <void *>(val_._get_ptr()), sizeof(ncclGinBarrierHandle_t) * 1)

    @property
    def world_gin_barrier(self):
        """GinBarrierHandle: """
        return GinBarrierHandle.from_ptr(<intptr_t>&(self._ptr[0].worldGinBarrier), self._readonly, self)

    @world_gin_barrier.setter
    def world_gin_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef GinBarrierHandle val_ = val
        memcpy(<void *>&(self._ptr[0].worldGinBarrier), <void *>(val_._get_ptr()), sizeof(ncclGinBarrierHandle_t) * 1)

    @property
    def rank(self):
        """int: """
        return self._ptr[0].rank

    @rank.setter
    def rank(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].rank = val

    @property
    def n_ranks(self):
        """int: """
        return self._ptr[0].nRanks

    @n_ranks.setter
    def n_ranks(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].nRanks = val

    @property
    def n_ranks_rcp32(self):
        """int: """
        return self._ptr[0].nRanks_rcp32

    @n_ranks_rcp32.setter
    def n_ranks_rcp32(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].nRanks_rcp32 = val

    @property
    def lsa_rank(self):
        """int: """
        return self._ptr[0].lsaRank

    @lsa_rank.setter
    def lsa_rank(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].lsaRank = val

    @property
    def lsa_size(self):
        """int: """
        return self._ptr[0].lsaSize

    @lsa_size.setter
    def lsa_size(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].lsaSize = val

    @property
    def lsa_size_rcp32(self):
        """int: """
        return self._ptr[0].lsaSize_rcp32

    @lsa_size_rcp32.setter
    def lsa_size_rcp32(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].lsaSize_rcp32 = val

    @property
    def window_table(self):
        """int: """
        return <intptr_t>(self._ptr[0].windowTable)

    @window_table.setter
    def window_table(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].windowTable = <ncclDevCommWindowTable_t><intptr_t>val

    @property
    def resource_window(self):
        """int: """
        return <intptr_t>(self._ptr[0].resourceWindow)

    @resource_window.setter
    def resource_window(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].resourceWindow = <ncclWindow_t><intptr_t>val

    @property
    def gin_connection_count(self):
        """int: """
        return self._ptr[0].ginConnectionCount

    @gin_connection_count.setter
    def gin_connection_count(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginConnectionCount = val

    @property
    def gin_net_device_types(self):
        """~_numpy.uint8: (array of length 4)."""
        cdef view.array arr = view.array(shape=(4,), itemsize=sizeof(uint8_t), format="B", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].ginNetDeviceTypes))
        return _numpy.asarray(arr)

    @gin_net_device_types.setter
    def gin_net_device_types(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        if len(val) != 4:
            raise ValueError(f"Expected length { 4 } for field gin_net_device_types, got {len(val)}")
        cdef view.array arr = view.array(shape=(4,), itemsize=sizeof(uint8_t), format="B", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.uint8)
        memcpy(<void *>(&(self._ptr[0].ginNetDeviceTypes)), <void *>(arr.data), sizeof(uint8_t) * len(val))

    @property
    def gin_handles(self):
        """~_numpy.int64: (array of length 4)."""
        cdef view.array arr = view.array(shape=(4,), itemsize=sizeof(intptr_t), format="q", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].ginHandles))
        return _numpy.asarray(arr)

    @gin_handles.setter
    def gin_handles(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        if len(val) != 4:
            raise ValueError(f"Expected length { 4 } for field gin_handles, got {len(val)}")
        cdef view.array arr = view.array(shape=(4,), itemsize=sizeof(intptr_t), format="q", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.intp)
        memcpy(<void *>(&(self._ptr[0].ginHandles)), <void *>(arr.data), sizeof(intptr_t) * len(val))

    @property
    def gin_signal_base(self):
        """int: """
        return self._ptr[0].ginSignalBase

    @gin_signal_base.setter
    def gin_signal_base(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginSignalBase = val

    @property
    def gin_signal_count(self):
        """int: """
        return self._ptr[0].ginSignalCount

    @gin_signal_count.setter
    def gin_signal_count(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginSignalCount = val

    @property
    def gin_counter_base(self):
        """int: """
        return self._ptr[0].ginCounterBase

    @gin_counter_base.setter
    def gin_counter_base(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginCounterBase = val

    @property
    def gin_counter_count(self):
        """int: """
        return self._ptr[0].ginCounterCount

    @gin_counter_count.setter
    def gin_counter_count(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginCounterCount = val

    @property
    def gin_signal_shadows(self):
        """int: """
        return <intptr_t>(self._ptr[0].ginSignalShadows)

    @gin_signal_shadows.setter
    def gin_signal_shadows(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginSignalShadows = <uint64_t*><intptr_t>val

    @property
    def gin_context_count(self):
        """int: """
        return self._ptr[0].ginContextCount

    @gin_context_count.setter
    def gin_context_count(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginContextCount = val

    @property
    def gin_context_base(self):
        """int: """
        return self._ptr[0].ginContextBase

    @gin_context_base.setter
    def gin_context_base(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginContextBase = val

    @property
    def gin_is_railed(self):
        """int: """
        return self._ptr[0].ginIsRailed

    @gin_is_railed.setter
    def gin_is_railed(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginIsRailed = val

    @property
    def abort_flag(self):
        """int: """
        return <intptr_t>(self._ptr[0].abortFlag)

    @abort_flag.setter
    def abort_flag(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].abortFlag = <uint32_t*><intptr_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for DevComm")

    @staticmethod
    def from_buffer(buffer):
        """Create an DevComm instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclDevComm_t), DevComm)

    @staticmethod
    def from_data(data):
        """Create an DevComm instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dev_comm_dtype` holding the data.
        """
        return __from_data(data, "dev_comm_dtype", dev_comm_dtype, DevComm)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an DevComm instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef DevComm obj = DevComm.__new__(DevComm)
        if owner is None:
            obj._ptr = <ncclDevComm_t *>malloc(sizeof(ncclDevComm_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DevComm")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclDevComm_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclDevComm_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_dev_comm_requirements_dtype_offsets():
    cdef ncclDevCommRequirements_t pod = ncclDevCommRequirements_t()
    return _numpy.dtype({
        'names': ['size_', 'magic', 'version', 'resource_requirements_list', 'team_requirements_list', 'lsa_multimem', 'barrier_count', 'lsa_barrier_count', 'rail_gin_barrier_count', 'lsa_ll_a2a_block_count', 'lsa_ll_a2a_slot_count', 'gin_force_enable', 'gin_context_count', 'gin_signal_count', 'gin_counter_count', 'gin_connection_type', 'gin_exclusive_contexts', 'gin_queue_depth', 'world_gin_barrier_count'],
        'formats': [_numpy.uint64, _numpy.uint32, _numpy.uint32, _numpy.intp, _numpy.intp, _numpy.uint8, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.int32, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.resourceRequirementsList)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.teamRequirementsList)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaMultimem)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.barrierCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaBarrierCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.railGinBarrierCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaLLA2ABlockCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaLLA2ASlotCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginForceEnable)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginContextCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginSignalCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginCounterCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginConnectionType)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginExclusiveContexts)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginQueueDepth)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.worldGinBarrierCount)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclDevCommRequirements_t),
    })

dev_comm_requirements_dtype = _get_dev_comm_requirements_dtype_offsets()

cdef class DevCommRequirements:
    """Empty-initialize an instance of `ncclDevCommRequirements_t`.


    .. seealso:: `ncclDevCommRequirements_t`
    """
    cdef:
        ncclDevCommRequirements_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclDevCommRequirements_t *>calloc(1, sizeof(ncclDevCommRequirements_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating DevCommRequirements")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclDevCommRequirements_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.DevCommRequirements object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef DevCommRequirements other_
        if not isinstance(other, DevCommRequirements):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclDevCommRequirements_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclDevCommRequirements_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclDevCommRequirements_t *>malloc(sizeof(ncclDevCommRequirements_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DevCommRequirements")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclDevCommRequirements_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].magic = val

    @property
    def version(self):
        """int: """
        return self._ptr[0].version

    @version.setter
    def version(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].version = val

    @property
    def resource_requirements_list(self):
        """int: """
        return <intptr_t>(self._ptr[0].resourceRequirementsList)

    @resource_requirements_list.setter
    def resource_requirements_list(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].resourceRequirementsList = <ncclDevResourceRequirements_t*><intptr_t>val

    @property
    def team_requirements_list(self):
        """int: """
        return <intptr_t>(self._ptr[0].teamRequirementsList)

    @team_requirements_list.setter
    def team_requirements_list(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].teamRequirementsList = <ncclTeamRequirements_t*><intptr_t>val

    @property
    def lsa_multimem(self):
        """int: """
        return self._ptr[0].lsaMultimem

    @lsa_multimem.setter
    def lsa_multimem(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].lsaMultimem = val

    @property
    def barrier_count(self):
        """int: """
        return self._ptr[0].barrierCount

    @barrier_count.setter
    def barrier_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].barrierCount = val

    @property
    def lsa_barrier_count(self):
        """int: """
        return self._ptr[0].lsaBarrierCount

    @lsa_barrier_count.setter
    def lsa_barrier_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].lsaBarrierCount = val

    @property
    def rail_gin_barrier_count(self):
        """int: """
        return self._ptr[0].railGinBarrierCount

    @rail_gin_barrier_count.setter
    def rail_gin_barrier_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].railGinBarrierCount = val

    @property
    def lsa_ll_a2a_block_count(self):
        """int: """
        return self._ptr[0].lsaLLA2ABlockCount

    @lsa_ll_a2a_block_count.setter
    def lsa_ll_a2a_block_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].lsaLLA2ABlockCount = val

    @property
    def lsa_ll_a2a_slot_count(self):
        """int: """
        return self._ptr[0].lsaLLA2ASlotCount

    @lsa_ll_a2a_slot_count.setter
    def lsa_ll_a2a_slot_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].lsaLLA2ASlotCount = val

    @property
    def gin_force_enable(self):
        """int: """
        return self._ptr[0].ginForceEnable

    @gin_force_enable.setter
    def gin_force_enable(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginForceEnable = val

    @property
    def gin_context_count(self):
        """int: """
        return self._ptr[0].ginContextCount

    @gin_context_count.setter
    def gin_context_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginContextCount = val

    @property
    def gin_signal_count(self):
        """int: """
        return self._ptr[0].ginSignalCount

    @gin_signal_count.setter
    def gin_signal_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginSignalCount = val

    @property
    def gin_counter_count(self):
        """int: """
        return self._ptr[0].ginCounterCount

    @gin_counter_count.setter
    def gin_counter_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginCounterCount = val

    @property
    def gin_connection_type(self):
        """int: """
        return <int>(self._ptr[0].ginConnectionType)

    @gin_connection_type.setter
    def gin_connection_type(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginConnectionType = <ncclGinConnectionType_t><int>val

    @property
    def gin_exclusive_contexts(self):
        """int: """
        return self._ptr[0].ginExclusiveContexts

    @gin_exclusive_contexts.setter
    def gin_exclusive_contexts(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginExclusiveContexts = val

    @property
    def gin_queue_depth(self):
        """int: """
        return self._ptr[0].ginQueueDepth

    @gin_queue_depth.setter
    def gin_queue_depth(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginQueueDepth = val

    @property
    def world_gin_barrier_count(self):
        """int: """
        return self._ptr[0].worldGinBarrierCount

    @world_gin_barrier_count.setter
    def world_gin_barrier_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].worldGinBarrierCount = val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for DevCommRequirements")

    @staticmethod
    def from_buffer(buffer):
        """Create an DevCommRequirements instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclDevCommRequirements_t), DevCommRequirements)

    @staticmethod
    def from_data(data):
        """Create an DevCommRequirements instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dev_comm_requirements_dtype` holding the data.
        """
        return __from_data(data, "dev_comm_requirements_dtype", dev_comm_requirements_dtype, DevCommRequirements)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an DevCommRequirements instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef DevCommRequirements obj = DevCommRequirements.__new__(DevCommRequirements)
        if owner is None:
            obj._ptr = <ncclDevCommRequirements_t *>malloc(sizeof(ncclDevCommRequirements_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DevCommRequirements")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclDevCommRequirements_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclDevCommRequirements_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj



###############################################################################
# Enum
###############################################################################

class Result(_IntEnum):
    """
    See `ncclResult_t`.
    """
    Success = ncclSuccess
    UnhandledCudaError = ncclUnhandledCudaError
    SystemError = ncclSystemError
    InternalError = ncclInternalError
    InvalidArgument = ncclInvalidArgument
    InvalidUsage = ncclInvalidUsage
    RemoteError = ncclRemoteError
    InProgress = ncclInProgress
    NumResults = ncclNumResults

class CommMemStat(_IntEnum):
    """
    See `ncclCommMemStat_t`.
    """
    GpuMemSuspend = ncclStatGpuMemSuspend
    GpuMemSuspended = ncclStatGpuMemSuspended
    GpuMemPersist = ncclStatGpuMemPersist
    GpuMemTotal = ncclStatGpuMemTotal

class RedOp_dummy(_IntEnum):
    """
    See `ncclRedOp_dummy_t`.
    """
    NumOps_dummy = ncclNumOps_dummy

class RedOp(_IntEnum):
    """
    See `ncclRedOp_t`.
    """
    Sum = ncclSum
    Prod = ncclProd
    Max = ncclMax
    Min = ncclMin
    Avg = ncclAvg
    NumOps = ncclNumOps
    MaxRedOp = ncclMaxRedOp

class DataType(_IntEnum):
    """
    See `ncclDataType_t`.
    """
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
    """
    See `ncclScalarResidence_t`.
    """
    Device = ncclScalarDevice
    HostImmediate = ncclScalarHostImmediate

class GinType(_IntEnum):
    """
    See `ncclGinType_t`.
    """
    NONE = NCCL_GIN_TYPE_NONE
    PROXY = NCCL_GIN_TYPE_PROXY
    GDAKI = NCCL_GIN_TYPE_GDAKI

class GinConnectionType(_IntEnum):
    """
    See `ncclGinConnectionType_t`.
    """
    NONE = NCCL_GIN_CONNECTION_NONE
    FULL = NCCL_GIN_CONNECTION_FULL
    RAIL = NCCL_GIN_CONNECTION_RAIL


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


cpdef object comm_init_all(int ndev, devlist):
    cdef nullable_unique_ptr[ vector[int] ] _devlist_
    get_resource_ptr[int](_devlist_, devlist, <int*>NULL)
    if ndev == 0:
        return view.array(shape=(1,), itemsize=sizeof(intptr_t), format="q", mode="c")[:0]
    cdef view.array comm = view.array(shape=(ndev,), itemsize=sizeof(intptr_t), format="q", mode="c")
    cdef intptr_t *comm_ptr = <intptr_t *>(comm.data)
    with nogil:
        __status__ = ncclCommInitAll(<ncclComm_t*>comm_ptr, ndev, <const int*>(_devlist_.data()))
    check_status(__status__)
    return comm


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


cpdef comm_revoke(intptr_t comm, int revoke_flags):
    with nogil:
        __status__ = ncclCommRevoke(<Comm>comm, revoke_flags)
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


cpdef comm_get_unique_id(intptr_t comm, intptr_t unique_id):
    with nogil:
        __status__ = ncclCommGetUniqueId(<Comm>comm, <ncclUniqueId*>unique_id)
    check_status(__status__)


cpdef intptr_t comm_grow(intptr_t comm, int n_ranks, intptr_t unique_id, int rank, intptr_t config) except? 0:
    cdef Comm newcomm
    with nogil:
        __status__ = ncclCommGrow(<Comm>comm, n_ranks, <const ncclUniqueId*>unique_id, rank, &newcomm, <ncclConfig_t*>config)
    check_status(__status__)
    return <intptr_t>newcomm


cpdef intptr_t comm_init_rank_scalable(int nranks, int myrank, int n_id, comm_ids, intptr_t config) except? 0:
    cdef void* _comm_ids_ = get_buffer_pointer(comm_ids, -1, readonly=False)
    cdef Comm newcomm
    with nogil:
        __status__ = ncclCommInitRankScalable(&newcomm, nranks, myrank, n_id, <ncclUniqueId*>_comm_ids_, <ncclConfig_t*>config)
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


cpdef comm_suspend(intptr_t comm, int flags):
    with nogil:
        __status__ = ncclCommSuspend(<Comm>comm, flags)
    check_status(__status__)


cpdef comm_resume(intptr_t comm):
    with nogil:
        __status__ = ncclCommResume(<Comm>comm)
    check_status(__status__)


cpdef uint64_t comm_mem_stats(intptr_t comm, int stat) except? -1:
    cdef uint64_t value
    with nogil:
        __status__ = ncclCommMemStats(<Comm>comm, <_CommMemStat>stat, &value)
    check_status(__status__)
    return value


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


cpdef intptr_t win_get_user_ptr(intptr_t comm, intptr_t win) except? 0:
    cdef void* out_user_ptr
    with nogil:
        __status__ = ncclWinGetUserPtr(<Comm>comm, <Window>win, &out_user_ptr)
    check_status(__status__)
    return <intptr_t>out_user_ptr


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


cpdef put_signal(intptr_t localbuff, size_t count, int datatype, int peer, intptr_t peer_win, size_t peer_win_offset, int sig_idx, int ctx, unsigned int flags, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclPutSignal(<const void*>localbuff, count, <_DataType>datatype, peer, <Window>peer_win, peer_win_offset, sig_idx, ctx, flags, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef signal(int peer, int sig_idx, int ctx, unsigned int flags, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclSignal(peer, sig_idx, ctx, flags, <Comm>comm, <Stream>stream)
    check_status(__status__)


cpdef wait_signal(int n_desc, intptr_t signal_descs, intptr_t comm, intptr_t stream):
    with nogil:
        __status__ = ncclWaitSignal(n_desc, <ncclWaitSignalDesc_t*>signal_descs, <Comm>comm, <Stream>stream)
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


cpdef comm_query_properties(intptr_t comm, intptr_t props):
    with nogil:
        __status__ = ncclCommQueryProperties(<Comm>comm, <ncclCommProperties_t*>props)
    check_status(__status__)


cpdef dev_comm_create(intptr_t comm, intptr_t reqs, intptr_t out_dev_comm):
    with nogil:
        __status__ = ncclDevCommCreate(<Comm>comm, <const ncclDevCommRequirements_t*>reqs, <ncclDevComm_t*>out_dev_comm)
    check_status(__status__)


cpdef dev_comm_destroy(intptr_t comm, intptr_t dev_comm):
    with nogil:
        __status__ = ncclDevCommDestroy(<Comm>comm, <const ncclDevComm_t*>dev_comm)
    check_status(__status__)


cpdef intptr_t get_lsa_multimem_device_pointer(intptr_t window, size_t offset) except? 0:
    cdef void* out_ptr
    with nogil:
        __status__ = ncclGetLsaMultimemDevicePointer(<Window>window, offset, &out_ptr)
    check_status(__status__)
    return <intptr_t>out_ptr


cpdef intptr_t get_lsa_device_pointer(intptr_t window, size_t offset, int lsa_rank) except? 0:
    cdef void* out_ptr
    with nogil:
        __status__ = ncclGetLsaDevicePointer(<Window>window, offset, lsa_rank, &out_ptr)
    check_status(__status__)
    return <intptr_t>out_ptr


cpdef intptr_t get_peer_device_pointer(intptr_t window, size_t offset, int peer) except? 0:
    cdef void* out_ptr
    with nogil:
        __status__ = ncclGetPeerDevicePointer(<Window>window, offset, peer, &out_ptr)
    check_status(__status__)
    return <intptr_t>out_ptr
