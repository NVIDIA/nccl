# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.28.0. Do not modify it directly.

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

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclUniqueId *>malloc(sizeof(ncclUniqueId))
            if self._ptr == NULL:
                raise MemoryError("Error allocating UniqueId")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclUniqueId))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

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
        'names': ['size_', 'magic', 'version', 'blocking', 'cga_cluster_size', 'min_ctas', 'max_ctas', 'net_name', 'split_share', 'traffic_class', 'comm_name', 'collnet_enable', 'cta_policy', 'shrink_share', 'nvls_ctas', 'n_channels_per_net_peer', 'nvlink_centric_sched', 'graph_usage_mode', 'num_rma_ctx'],
        'formats': [_numpy.uint64, _numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.intp, _numpy.int32, _numpy.int32, _numpy.intp, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32],
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

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
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

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
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

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
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

class CommMemStat(_IntEnum):
    """See `ncclCommMemStat_t`."""
    StatGpuMemSuspend = ncclStatGpuMemSuspend
    StatGpuMemSuspended = ncclStatGpuMemSuspended
    StatGpuMemPersist = ncclStatGpuMemPersist
    StatGpuMemTotal = ncclStatGpuMemTotal

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
