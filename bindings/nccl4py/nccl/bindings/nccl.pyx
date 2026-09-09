# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.31.2. Do not modify it directly.


# <<<< PREAMBLE CONTENT >>>>

cimport cpython as _cyb_cpython
cimport cpython.buffer as _cyb_cpython_buffer
from cython cimport view as _cyb_view
from libc.stdint cimport (
    intptr_t,
    uint32_t,
    uint64_t,
    uint8_t,
)
from libc.stdlib cimport (
    calloc as _cyb_calloc,
    free as _cyb_free,
    malloc as _cyb_malloc,
)
from libc.string cimport (
    memcmp as _cyb_memcmp,
    memcpy as _cyb_memcpy,
)

from enum import IntEnum as _cyb_IntEnum

import numpy as _numpy

cdef _cyb___getbuffer(object self, _cyb_cpython.Py_buffer *buffer, void *ptr, int size, bint readonly):
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

cdef _cyb_from_buffer(buffer, size, lowpp_type):
    cdef _cyb_cpython.Py_buffer view
    if _cyb_cpython.PyObject_GetBuffer(buffer, &view, _cyb_cpython_buffer.PyBUF_SIMPLE) != 0:
        raise TypeError("buffer argument does not support the buffer protocol")
    try:
        if view.itemsize != 1:
            raise ValueError("buffer itemsize must be 1 byte")
        if view.len != size:
            raise ValueError(f"buffer length must be {size} bytes")
        return lowpp_type.from_ptr(<intptr_t><void *>view.buf, not view.readonly, buffer)
    finally:
        _cyb_cpython.PyBuffer_Release(&view)

cdef _cyb_from_data(data, dtype_name, expected_dtype, lowpp_type):
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

cdef intptr_t _cyb_get_buffer_pointer(buf, Py_ssize_t size, readonly=True) except?-1:
    cdef intptr_t ptr
    cdef int flags = _cyb_cpython.PyBUF_ANY_CONTIGUOUS
    if not readonly:
        flags |= _cyb_cpython.PyBUF_WRITABLE
    cdef int status = -1
    cdef _cyb_cpython.Py_buffer view
    if isinstance(buf, int):
        ptr = <intptr_t>buf
    else:
        try:
            status = _cyb_cpython.PyObject_GetBuffer(buf, &view, flags)
            if size != -1:
                assert view.len == size
            assert view.ndim == 1
        except Exception as e:
            adj = "writable " if not readonly else ""
            raise ValueError(
                "buf must be either a Python int representing the pointer "
                f"address to a valid buffer, or a 1D contiguous {adj}"
                f"buffer, of size {size}"
            ) from e
        else:
            ptr = <intptr_t>view.buf
        finally:
            if status == 0:
                _cyb_cpython.PyBuffer_Release(&view)
    return ptr


# <<<< END OF PREAMBLE CONTENT >>>>

cimport cython  # NOQA
from libcpp.vector cimport vector

from ._internal.utils cimport (nested_resource, nullable_unique_ptr,
                              get_resource_ptr, get_nested_resource_ptr)

_version_span = "with version 2.31.2"
__version__ = _version_span.split()[-1]

# NCCL_VERSION(X,Y,Z) = X*10000 + Y*100 + Z (NCCL >= 2.9).
_version_parts = __version__.split(".")
__version_code__ = (
    int(_version_parts[0]) * 10000 + int(_version_parts[1]) * 100 + int(_version_parts[2])
)


###############################################################################
# POD
###############################################################################

cdef _get_unique_id_dtype_offsets():
    cdef ncclUniqueId pod
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
        self._ptr = <ncclUniqueId *>_cyb_calloc(1, sizeof(ncclUniqueId))
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
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclUniqueId)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclUniqueId), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclUniqueId *>_cyb_malloc(sizeof(ncclUniqueId))
            if self._ptr == NULL:
                raise MemoryError("Error allocating UniqueId")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclUniqueId))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @staticmethod
    def from_buffer(buffer):
        """Create an UniqueId instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclUniqueId), UniqueId)

    @staticmethod
    def from_data(data):
        """Create an UniqueId instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `unique_id_dtype` holding the data.
        """
        return _cyb_from_data(data, "unique_id_dtype", unique_id_dtype, UniqueId)

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
            obj._ptr = <ncclUniqueId *>_cyb_malloc(sizeof(ncclUniqueId))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating UniqueId")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclUniqueId))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclUniqueId *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_config_dtype_offsets():
    cdef ncclConfig_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'version', 'blocking', 'cga_cluster_size', 'min_ctas', 'max_ctas', 'net_name', 'split_share', 'traffic_class', 'comm_name', 'collnet_enable', 'cta_policy', 'shrink_share', 'nvls_ctas', 'n_channels_per_net_peer', 'nvlink_centric_sched', 'graph_usage_mode', 'num_rma_ctx', 'max_p2p_peers', 'graph_stream_ordering', 'launch_order_implicit', 'num_rma_sig', 'rma_eager_init', 'host_cft_mode'],
        'formats': [_numpy.uint64, _numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.intp, _numpy.int32, _numpy.int32, _numpy.intp, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32],
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
            (<intptr_t>&(pod.graphStreamOrdering)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.launchOrderImplicit)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.numRmaSig)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.rmaEagerInit)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.hostCftMode)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclConfig_t),
    })

config_dtype = _get_config_dtype_offsets()

cdef class Config:
    """Initialize an instance of `ncclConfig_t` using configured defaults.


    .. seealso:: `ncclConfig_t`
    """
    cdef:
        ncclConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly
        dict _refs

    def __init__(self):
        self._ptr = <ncclConfig_t *>_cyb_calloc(1, sizeof(ncclConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Config")
        self._owner = None
        self._owned = True
        self._readonly = False
        self._refs = {}

        self._ptr[0].size = sizeof(ncclConfig_t)
        self._ptr[0].magic = 0xcafebeef
        self._ptr[0].version = __version_code__
        self._ptr[0].blocking = -2147483648
        self._ptr[0].cgaClusterSize = -2147483648
        self._ptr[0].minCTAs = -2147483648
        self._ptr[0].maxCTAs = -2147483648
        self._ptr[0].netName = NULL
        self._ptr[0].splitShare = -2147483648
        self._ptr[0].trafficClass = -2147483648
        self._ptr[0].commName = NULL
        self._ptr[0].collnetEnable = -2147483648
        self._ptr[0].CTAPolicy = -2147483648
        self._ptr[0].shrinkShare = -2147483648
        self._ptr[0].nvlsCTAs = -2147483648
        self._ptr[0].nChannelsPerNetPeer = -2147483648
        self._ptr[0].nvlinkCentricSched = -2147483648
        self._ptr[0].graphUsageMode = -2147483648
        self._ptr[0].numRmaCtx = -2147483648
        self._ptr[0].maxP2pPeers = -2147483648
        self._ptr[0].graphStreamOrdering = -2147483648
        self._ptr[0].launchOrderImplicit = -2147483648
        self._ptr[0].numRmaSig = -2147483648
        self._ptr[0].rmaEagerInit = -2147483648
        self._ptr[0].hostCftMode = -2147483648

    def __dealloc__(self):
        cdef ncclConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclConfig_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclConfig_t *>_cyb_malloc(sizeof(ncclConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating Config")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclConfig_t))
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
            return _cyb_cpython.PyUnicode_FromString(ptr)
        return ""

    @net_name.setter
    def net_name(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        cdef bytes buf = val.encode()
        cdef char *ptr = buf
        self._refs["net_name"] = buf
        self._ptr[0].netName = <char *><intptr_t>ptr

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
            return _cyb_cpython.PyUnicode_FromString(ptr)
        return ""

    @comm_name.setter
    def comm_name(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        cdef bytes buf = val.encode()
        cdef char *ptr = buf
        self._refs["comm_name"] = buf
        self._ptr[0].commName = <char *><intptr_t>ptr

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

    @property
    def graph_stream_ordering(self):
        """int: """
        return self._ptr[0].graphStreamOrdering

    @graph_stream_ordering.setter
    def graph_stream_ordering(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].graphStreamOrdering = val

    @property
    def launch_order_implicit(self):
        """int: """
        return self._ptr[0].launchOrderImplicit

    @launch_order_implicit.setter
    def launch_order_implicit(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].launchOrderImplicit = val

    @property
    def num_rma_sig(self):
        """int: """
        return self._ptr[0].numRmaSig

    @num_rma_sig.setter
    def num_rma_sig(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].numRmaSig = val

    @property
    def rma_eager_init(self):
        """int: """
        return self._ptr[0].rmaEagerInit

    @rma_eager_init.setter
    def rma_eager_init(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].rmaEagerInit = val

    @property
    def host_cft_mode(self):
        """int: """
        return self._ptr[0].hostCftMode

    @host_cft_mode.setter
    def host_cft_mode(self, val):
        if self._readonly:
            raise ValueError("This Config instance is read-only")
        self._ptr[0].hostCftMode = val

    @staticmethod
    def from_buffer(buffer):
        """Create an Config instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclConfig_t), Config)

    @staticmethod
    def from_data(data):
        """Create an Config instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `config_dtype` holding the data.
        """
        return _cyb_from_data(data, "config_dtype", config_dtype, Config)

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
            obj._ptr = <ncclConfig_t *>_cyb_malloc(sizeof(ncclConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Config")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        obj._refs = {}
        return obj


cdef _get__py_anon_pod0_dtype_offsets():
    cdef nccl_bindings_nccl__anon_pod0 pod
    return _numpy.dtype({
        'names': ['vendor_id', 'option_id'],
        'formats': [_numpy.int32, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.vendorId)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.optionId)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(nccl_bindings_nccl__anon_pod0),
    })

_py_anon_pod0_dtype = _get__py_anon_pod0_dtype_offsets()

cdef class _py_anon_pod0:
    """Empty-initialize an instance of `nccl_bindings_nccl__anon_pod0`.


    .. seealso:: `nccl_bindings_nccl__anon_pod0`
    """
    cdef:
        nccl_bindings_nccl__anon_pod0 *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <nccl_bindings_nccl__anon_pod0 *>_cyb_calloc(1, sizeof(nccl_bindings_nccl__anon_pod0))
        if self._ptr == NULL:
            raise MemoryError("Error allocating _py_anon_pod0")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef nccl_bindings_nccl__anon_pod0 *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}._py_anon_pod0 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef _py_anon_pod0 other_
        if not isinstance(other, _py_anon_pod0):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(nccl_bindings_nccl__anon_pod0)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(nccl_bindings_nccl__anon_pod0), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <nccl_bindings_nccl__anon_pod0 *>_cyb_malloc(sizeof(nccl_bindings_nccl__anon_pod0))
            if self._ptr == NULL:
                raise MemoryError("Error allocating _py_anon_pod0")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(nccl_bindings_nccl__anon_pod0))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def vendor_id(self):
        """int: """
        return self._ptr[0].vendorId

    @vendor_id.setter
    def vendor_id(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod0 instance is read-only")
        self._ptr[0].vendorId = val

    @property
    def option_id(self):
        """int: """
        return self._ptr[0].optionId

    @option_id.setter
    def option_id(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod0 instance is read-only")
        self._ptr[0].optionId = val

    @staticmethod
    def from_buffer(buffer):
        """Create an _py_anon_pod0 instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(nccl_bindings_nccl__anon_pod0), _py_anon_pod0)

    @staticmethod
    def from_data(data):
        """Create an _py_anon_pod0 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `_py_anon_pod0_dtype` holding the data.
        """
        return _cyb_from_data(data, "_py_anon_pod0_dtype", _py_anon_pod0_dtype, _py_anon_pod0)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an _py_anon_pod0 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef _py_anon_pod0 obj = _py_anon_pod0.__new__(_py_anon_pod0)
        if owner is None:
            obj._ptr = <nccl_bindings_nccl__anon_pod0 *>_cyb_malloc(sizeof(nccl_bindings_nccl__anon_pod0))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating _py_anon_pod0")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(nccl_bindings_nccl__anon_pod0))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <nccl_bindings_nccl__anon_pod0 *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get__py_anon_pod1_dtype_offsets():
    cdef nccl_bindings_nccl__anon_pod1 pod
    return _numpy.dtype({
        'names': ['i', 's', 'raw'],
        'formats': [_numpy.int32, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.i)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.s)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.raw)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(nccl_bindings_nccl__anon_pod1),
    })

_py_anon_pod1_dtype = _get__py_anon_pod1_dtype_offsets()

cdef class _py_anon_pod1:
    """Empty-initialize an instance of `nccl_bindings_nccl__anon_pod1`.


    .. seealso:: `nccl_bindings_nccl__anon_pod1`
    """
    cdef:
        nccl_bindings_nccl__anon_pod1 *_ptr
        object _owner
        bint _owned
        bint _readonly
        dict _refs

    def __init__(self):
        self._ptr = <nccl_bindings_nccl__anon_pod1 *>_cyb_calloc(1, sizeof(nccl_bindings_nccl__anon_pod1))
        if self._ptr == NULL:
            raise MemoryError("Error allocating _py_anon_pod1")
        self._owner = None
        self._owned = True
        self._readonly = False
        self._refs = {}

    def __dealloc__(self):
        cdef nccl_bindings_nccl__anon_pod1 *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}._py_anon_pod1 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef _py_anon_pod1 other_
        if not isinstance(other, _py_anon_pod1):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(nccl_bindings_nccl__anon_pod1)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(nccl_bindings_nccl__anon_pod1), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <nccl_bindings_nccl__anon_pod1 *>_cyb_malloc(sizeof(nccl_bindings_nccl__anon_pod1))
            if self._ptr == NULL:
                raise MemoryError("Error allocating _py_anon_pod1")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(nccl_bindings_nccl__anon_pod1))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def i(self):
        """int: """
        return self._ptr[0].i

    @i.setter
    def i(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod1 instance is read-only")
        self._ptr[0].i = val

    @property
    def s(self):
        """str: """
        cdef char* ptr = <char*>self._ptr[0].s
        if ptr:
            return _cyb_cpython.PyUnicode_FromString(ptr)
        return ""

    @s.setter
    def s(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod1 instance is read-only")
        cdef bytes buf = val.encode()
        cdef char *ptr = buf
        self._refs["s"] = buf
        self._ptr[0].s = <char *><intptr_t>ptr

    @property
    def raw(self):
        """int: """
        return <intptr_t>(self._ptr[0].raw)

    @raw.setter
    def raw(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod1 instance is read-only")
        self._ptr[0].raw = <void *><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an _py_anon_pod1 instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(nccl_bindings_nccl__anon_pod1), _py_anon_pod1)

    @staticmethod
    def from_data(data):
        """Create an _py_anon_pod1 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `_py_anon_pod1_dtype` holding the data.
        """
        return _cyb_from_data(data, "_py_anon_pod1_dtype", _py_anon_pod1_dtype, _py_anon_pod1)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an _py_anon_pod1 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef _py_anon_pod1 obj = _py_anon_pod1.__new__(_py_anon_pod1)
        if owner is None:
            obj._ptr = <nccl_bindings_nccl__anon_pod1 *>_cyb_malloc(sizeof(nccl_bindings_nccl__anon_pod1))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating _py_anon_pod1")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(nccl_bindings_nccl__anon_pod1))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <nccl_bindings_nccl__anon_pod1 *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        obj._refs = {}
        return obj


cdef _get_sim_info_dtype_offsets():
    cdef ncclSimInfo_t pod
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
    """Initialize an instance of `ncclSimInfo_t` using configured defaults.


    .. seealso:: `ncclSimInfo_t`
    """
    cdef:
        ncclSimInfo_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclSimInfo_t *>_cyb_calloc(1, sizeof(ncclSimInfo_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating SimInfo")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclSimInfo_t)
        self._ptr[0].magic = 0x74685283
        self._ptr[0].version = __version_code__
        self._ptr[0].estimatedTime = -1.0

    def __dealloc__(self):
        cdef ncclSimInfo_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclSimInfo_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclSimInfo_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclSimInfo_t *>_cyb_malloc(sizeof(ncclSimInfo_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating SimInfo")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclSimInfo_t))
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
    def from_buffer(buffer):
        """Create an SimInfo instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclSimInfo_t), SimInfo)

    @staticmethod
    def from_data(data):
        """Create an SimInfo instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `sim_info_dtype` holding the data.
        """
        return _cyb_from_data(data, "sim_info_dtype", sim_info_dtype, SimInfo)

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
            obj._ptr = <ncclSimInfo_t *>_cyb_malloc(sizeof(ncclSimInfo_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating SimInfo")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclSimInfo_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclSimInfo_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_wait_signal_desc_dtype_offsets():
    cdef ncclWaitSignalDesc_t pod
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
        self._ptr = <ncclWaitSignalDesc_t *>_cyb_calloc(1, sizeof(ncclWaitSignalDesc_t))
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
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclWaitSignalDesc_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclWaitSignalDesc_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclWaitSignalDesc_t *>_cyb_malloc(sizeof(ncclWaitSignalDesc_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating WaitSignalDesc")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclWaitSignalDesc_t))
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
    def from_buffer(buffer):
        """Create an WaitSignalDesc instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclWaitSignalDesc_t), WaitSignalDesc)

    @staticmethod
    def from_data(data):
        """Create an WaitSignalDesc instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `wait_signal_desc_dtype` holding the data.
        """
        return _cyb_from_data(data, "wait_signal_desc_dtype", wait_signal_desc_dtype, WaitSignalDesc)

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
            obj._ptr = <ncclWaitSignalDesc_t *>_cyb_malloc(sizeof(ncclWaitSignalDesc_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating WaitSignalDesc")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclWaitSignalDesc_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclWaitSignalDesc_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_comm_properties_dtype_offsets():
    cdef ncclCommProperties_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'version', 'rank', 'n_ranks', 'cuda_dev', 'nvml_dev', 'device_api_support', 'multimem_support', 'gin_type', 'n_lsa_teams', 'host_rma_support', 'railed_gin_type', 'comm_hash', 'gin_min_stride', 'gin_connection_type', 'gin_support', 'dev_comm_runtime_version_size'],
        'formats': [_numpy.uint64, _numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.uint8, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.int32, _numpy.uint64, _numpy.int32, _numpy.int32, (_numpy.uint8, 64), _numpy.uint64],
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
            (<intptr_t>&(pod.commHash)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginMinStride)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginConnectionType)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginSupport)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.devCommRuntimeVersionSize)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclCommProperties_t),
    })

comm_properties_dtype = _get_comm_properties_dtype_offsets()

cdef class CommProperties:
    """Initialize an instance of `ncclCommProperties_t` using configured defaults.


    .. seealso:: `ncclCommProperties_t`
    """
    cdef:
        ncclCommProperties_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclCommProperties_t *>_cyb_calloc(1, sizeof(ncclCommProperties_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating CommProperties")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclCommProperties_t)
        self._ptr[0].magic = 0xcafebeef
        self._ptr[0].version = __version_code__

    def __dealloc__(self):
        cdef ncclCommProperties_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclCommProperties_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclCommProperties_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclCommProperties_t *>_cyb_malloc(sizeof(ncclCommProperties_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating CommProperties")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclCommProperties_t))
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

    @property
    def comm_hash(self):
        """int: """
        return self._ptr[0].commHash

    @comm_hash.setter
    def comm_hash(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].commHash = val

    @property
    def gin_min_stride(self):
        """int: """
        return self._ptr[0].ginMinStride

    @gin_min_stride.setter
    def gin_min_stride(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].ginMinStride = val

    @property
    def gin_connection_type(self):
        """int: """
        return <int>(self._ptr[0].ginConnectionType)

    @gin_connection_type.setter
    def gin_connection_type(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].ginConnectionType = <ncclGinConnectionType_t><int>val

    @property
    def gin_support(self):
        """~_numpy.uint8: (array of length 64)."""
        cdef _cyb_view.array arr = _cyb_view.array(shape=(64,), itemsize=sizeof(uint8_t), format="B", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].ginSupport))
        return _numpy.asarray(arr)

    @gin_support.setter
    def gin_support(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        if len(val) != 64:
            raise ValueError(f"Expected length { 64 } for field gin_support, got {len(val)}")
        cdef _cyb_view.array arr = _cyb_view.array(shape=(64,), itemsize=sizeof(uint8_t), format="B", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.uint8)
        _cyb_memcpy(<void *>(&(self._ptr[0].ginSupport)), <void *>(arr.data), sizeof(uint8_t) * len(val))

    @property
    def dev_comm_runtime_version_size(self):
        """int: """
        return self._ptr[0].devCommRuntimeVersionSize

    @dev_comm_runtime_version_size.setter
    def dev_comm_runtime_version_size(self, val):
        if self._readonly:
            raise ValueError("This CommProperties instance is read-only")
        self._ptr[0].devCommRuntimeVersionSize = val

    @staticmethod
    def from_buffer(buffer):
        """Create an CommProperties instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclCommProperties_t), CommProperties)

    @staticmethod
    def from_data(data):
        """Create an CommProperties instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `comm_properties_dtype` holding the data.
        """
        return _cyb_from_data(data, "comm_properties_dtype", comm_properties_dtype, CommProperties)

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
            obj._ptr = <ncclCommProperties_t *>_cyb_malloc(sizeof(ncclCommProperties_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating CommProperties")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclCommProperties_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclCommProperties_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_dev_resource_requirements_dtype_offsets():
    cdef ncclDevResourceRequirements_t pod
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
        self._ptr = <ncclDevResourceRequirements_t *>_cyb_calloc(1, sizeof(ncclDevResourceRequirements_t))
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
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclDevResourceRequirements_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclDevResourceRequirements_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclDevResourceRequirements_t *>_cyb_malloc(sizeof(ncclDevResourceRequirements_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DevResourceRequirements")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclDevResourceRequirements_t))
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

    @staticmethod
    def from_buffer(buffer):
        """Create an DevResourceRequirements instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclDevResourceRequirements_t), DevResourceRequirements)

    @staticmethod
    def from_data(data):
        """Create an DevResourceRequirements instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dev_resource_requirements_dtype` holding the data.
        """
        return _cyb_from_data(data, "dev_resource_requirements_dtype", dev_resource_requirements_dtype, DevResourceRequirements)

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
            obj._ptr = <ncclDevResourceRequirements_t *>_cyb_malloc(sizeof(ncclDevResourceRequirements_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DevResourceRequirements")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclDevResourceRequirements_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclDevResourceRequirements_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_team_dtype_offsets():
    cdef ncclTeam_t pod
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
        self._ptr = <ncclTeam_t *>_cyb_calloc(1, sizeof(ncclTeam_t))
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
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclTeam_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclTeam_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclTeam_t *>_cyb_malloc(sizeof(ncclTeam_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating Team")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclTeam_t))
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

    @staticmethod
    def from_buffer(buffer):
        """Create an Team instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclTeam_t), Team)

    @staticmethod
    def from_data(data):
        """Create an Team instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `team_dtype` holding the data.
        """
        return _cyb_from_data(data, "team_dtype", team_dtype, Team)

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
            obj._ptr = <ncclTeam_t *>_cyb_malloc(sizeof(ncclTeam_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Team")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclTeam_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclTeam_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_multimem_handle_dtype_offsets():
    cdef ncclMultimemHandle_t pod
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
        self._ptr = <ncclMultimemHandle_t *>_cyb_calloc(1, sizeof(ncclMultimemHandle_t))
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
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclMultimemHandle_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclMultimemHandle_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclMultimemHandle_t *>_cyb_malloc(sizeof(ncclMultimemHandle_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating MultimemHandle")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclMultimemHandle_t))
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

    @staticmethod
    def from_buffer(buffer):
        """Create an MultimemHandle instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclMultimemHandle_t), MultimemHandle)

    @staticmethod
    def from_data(data):
        """Create an MultimemHandle instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `multimem_handle_dtype` holding the data.
        """
        return _cyb_from_data(data, "multimem_handle_dtype", multimem_handle_dtype, MultimemHandle)

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
            obj._ptr = <ncclMultimemHandle_t *>_cyb_malloc(sizeof(ncclMultimemHandle_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating MultimemHandle")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclMultimemHandle_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclMultimemHandle_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_resource_window_vidmem_dtype_offsets():
    cdef ncclResourceWindow_vidmem_t pod
    return _numpy.dtype({
        'names': ['lsa_flat_base', 'stride4g', 'mc_offset4k'],
        'formats': [_numpy.intp, _numpy.uint32, _numpy.uint32],
        'offsets': [
            (<intptr_t>&(pod.lsaFlatBase)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.stride4G)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.mcOffset4K)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclResourceWindow_vidmem_t),
    })

resource_window_vidmem_dtype = _get_resource_window_vidmem_dtype_offsets()

cdef class ResourceWindowVidmem:
    """Empty-initialize an instance of `ncclResourceWindow_vidmem_t`.


    .. seealso:: `ncclResourceWindow_vidmem_t`
    """
    cdef:
        ncclResourceWindow_vidmem_t *_ptr
        object _owner
        bint _owned
        bint _readonly
        dict _refs

    def __init__(self):
        self._ptr = <ncclResourceWindow_vidmem_t *>_cyb_calloc(1, sizeof(ncclResourceWindow_vidmem_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating ResourceWindowVidmem")
        self._owner = None
        self._owned = True
        self._readonly = False
        self._refs = {}

    def __dealloc__(self):
        cdef ncclResourceWindow_vidmem_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.ResourceWindowVidmem object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef ResourceWindowVidmem other_
        if not isinstance(other, ResourceWindowVidmem):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclResourceWindow_vidmem_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclResourceWindow_vidmem_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclResourceWindow_vidmem_t *>_cyb_malloc(sizeof(ncclResourceWindow_vidmem_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating ResourceWindowVidmem")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclResourceWindow_vidmem_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def lsa_flat_base(self):
        """str: """
        cdef char* ptr = <char*>self._ptr[0].lsaFlatBase
        if ptr:
            return _cyb_cpython.PyUnicode_FromString(ptr)
        return ""

    @lsa_flat_base.setter
    def lsa_flat_base(self, val):
        if self._readonly:
            raise ValueError("This ResourceWindowVidmem instance is read-only")
        cdef bytes buf = val.encode()
        cdef char *ptr = buf
        self._refs["lsa_flat_base"] = buf
        self._ptr[0].lsaFlatBase = <char *><intptr_t>ptr

    @property
    def stride4g(self):
        """int: """
        return self._ptr[0].stride4G

    @stride4g.setter
    def stride4g(self, val):
        if self._readonly:
            raise ValueError("This ResourceWindowVidmem instance is read-only")
        self._ptr[0].stride4G = val

    @property
    def mc_offset4k(self):
        """int: """
        return self._ptr[0].mcOffset4K

    @mc_offset4k.setter
    def mc_offset4k(self, val):
        if self._readonly:
            raise ValueError("This ResourceWindowVidmem instance is read-only")
        self._ptr[0].mcOffset4K = val

    @staticmethod
    def from_buffer(buffer):
        """Create an ResourceWindowVidmem instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclResourceWindow_vidmem_t), ResourceWindowVidmem)

    @staticmethod
    def from_data(data):
        """Create an ResourceWindowVidmem instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `resource_window_vidmem_dtype` holding the data.
        """
        return _cyb_from_data(data, "resource_window_vidmem_dtype", resource_window_vidmem_dtype, ResourceWindowVidmem)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an ResourceWindowVidmem instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef ResourceWindowVidmem obj = ResourceWindowVidmem.__new__(ResourceWindowVidmem)
        if owner is None:
            obj._ptr = <ncclResourceWindow_vidmem_t *>_cyb_malloc(sizeof(ncclResourceWindow_vidmem_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating ResourceWindowVidmem")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclResourceWindow_vidmem_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclResourceWindow_vidmem_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        obj._refs = {}
        return obj


cdef _get_gin_barrier_handle_dtype_offsets():
    cdef ncclGinBarrierHandle_t pod
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
        self._ptr = <ncclGinBarrierHandle_t *>_cyb_calloc(1, sizeof(ncclGinBarrierHandle_t))
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
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclGinBarrierHandle_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclGinBarrierHandle_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclGinBarrierHandle_t *>_cyb_malloc(sizeof(ncclGinBarrierHandle_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating GinBarrierHandle")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclGinBarrierHandle_t))
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

    @staticmethod
    def from_buffer(buffer):
        """Create an GinBarrierHandle instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclGinBarrierHandle_t), GinBarrierHandle)

    @staticmethod
    def from_data(data):
        """Create an GinBarrierHandle instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `gin_barrier_handle_dtype` holding the data.
        """
        return _cyb_from_data(data, "gin_barrier_handle_dtype", gin_barrier_handle_dtype, GinBarrierHandle)

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
            obj._ptr = <ncclGinBarrierHandle_t *>_cyb_malloc(sizeof(ncclGinBarrierHandle_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating GinBarrierHandle")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclGinBarrierHandle_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclGinBarrierHandle_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_lsa_barrier_handle_dtype_offsets():
    cdef ncclLsaBarrierHandle_t pod
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
        self._ptr = <ncclLsaBarrierHandle_t *>_cyb_calloc(1, sizeof(ncclLsaBarrierHandle_t))
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
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclLsaBarrierHandle_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclLsaBarrierHandle_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclLsaBarrierHandle_t *>_cyb_malloc(sizeof(ncclLsaBarrierHandle_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating LsaBarrierHandle")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclLsaBarrierHandle_t))
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

    @staticmethod
    def from_buffer(buffer):
        """Create an LsaBarrierHandle instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclLsaBarrierHandle_t), LsaBarrierHandle)

    @staticmethod
    def from_data(data):
        """Create an LsaBarrierHandle instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `lsa_barrier_handle_dtype` holding the data.
        """
        return _cyb_from_data(data, "lsa_barrier_handle_dtype", lsa_barrier_handle_dtype, LsaBarrierHandle)

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
            obj._ptr = <ncclLsaBarrierHandle_t *>_cyb_malloc(sizeof(ncclLsaBarrierHandle_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating LsaBarrierHandle")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclLsaBarrierHandle_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclLsaBarrierHandle_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_cft_barrier_handle_dtype_offsets():
    cdef ncclCftBarrierHandle_t pod
    return _numpy.dtype({
        'names': ['buf_handle', 'n_barriers'],
        'formats': [_numpy.uint32, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.bufHandle)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nBarriers)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclCftBarrierHandle_t),
    })

cft_barrier_handle_dtype = _get_cft_barrier_handle_dtype_offsets()

cdef class CftBarrierHandle:
    """Empty-initialize an instance of `ncclCftBarrierHandle_t`.


    .. seealso:: `ncclCftBarrierHandle_t`
    """
    cdef:
        ncclCftBarrierHandle_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclCftBarrierHandle_t *>_cyb_calloc(1, sizeof(ncclCftBarrierHandle_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating CftBarrierHandle")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclCftBarrierHandle_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.CftBarrierHandle object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef CftBarrierHandle other_
        if not isinstance(other, CftBarrierHandle):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclCftBarrierHandle_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclCftBarrierHandle_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclCftBarrierHandle_t *>_cyb_malloc(sizeof(ncclCftBarrierHandle_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating CftBarrierHandle")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclCftBarrierHandle_t))
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
            raise ValueError("This CftBarrierHandle instance is read-only")
        self._ptr[0].bufHandle = <ncclDevResourceHandle_t><uint32_t>val

    @property
    def n_barriers(self):
        """int: """
        return self._ptr[0].nBarriers

    @n_barriers.setter
    def n_barriers(self, val):
        if self._readonly:
            raise ValueError("This CftBarrierHandle instance is read-only")
        self._ptr[0].nBarriers = val

    @staticmethod
    def from_buffer(buffer):
        """Create an CftBarrierHandle instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclCftBarrierHandle_t), CftBarrierHandle)

    @staticmethod
    def from_data(data):
        """Create an CftBarrierHandle instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `cft_barrier_handle_dtype` holding the data.
        """
        return _cyb_from_data(data, "cft_barrier_handle_dtype", cft_barrier_handle_dtype, CftBarrierHandle)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an CftBarrierHandle instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef CftBarrierHandle obj = CftBarrierHandle.__new__(CftBarrierHandle)
        if owner is None:
            obj._ptr = <ncclCftBarrierHandle_t *>_cyb_malloc(sizeof(ncclCftBarrierHandle_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating CftBarrierHandle")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclCftBarrierHandle_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclCftBarrierHandle_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_ll_a2a_handle_dtype_offsets():
    cdef ncclLLA2AHandle_t pod
    return _numpy.dtype({
        'names': ['buf_handle', 'n_slots'],
        'formats': [_numpy.uint32, _numpy.uint32],
        'offsets': [
            (<intptr_t>&(pod.bufHandle)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nSlots)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclLLA2AHandle_t),
    })

ll_a2a_handle_dtype = _get_ll_a2a_handle_dtype_offsets()

cdef class LLA2AHandle:
    """Empty-initialize an instance of `ncclLLA2AHandle_t`.


    .. seealso:: `ncclLLA2AHandle_t`
    """
    cdef:
        ncclLLA2AHandle_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclLLA2AHandle_t *>_cyb_calloc(1, sizeof(ncclLLA2AHandle_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating LLA2AHandle")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclLLA2AHandle_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.LLA2AHandle object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef LLA2AHandle other_
        if not isinstance(other, LLA2AHandle):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclLLA2AHandle_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclLLA2AHandle_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclLLA2AHandle_t *>_cyb_malloc(sizeof(ncclLLA2AHandle_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating LLA2AHandle")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclLLA2AHandle_t))
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
            raise ValueError("This LLA2AHandle instance is read-only")
        self._ptr[0].bufHandle = <ncclDevResourceHandle_t><uint32_t>val

    @property
    def n_slots(self):
        """int: """
        return self._ptr[0].nSlots

    @n_slots.setter
    def n_slots(self, val):
        if self._readonly:
            raise ValueError("This LLA2AHandle instance is read-only")
        self._ptr[0].nSlots = val

    @staticmethod
    def from_buffer(buffer):
        """Create an LLA2AHandle instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclLLA2AHandle_t), LLA2AHandle)

    @staticmethod
    def from_data(data):
        """Create an LLA2AHandle instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ll_a2a_handle_dtype` holding the data.
        """
        return _cyb_from_data(data, "ll_a2a_handle_dtype", ll_a2a_handle_dtype, LLA2AHandle)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an LLA2AHandle instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef LLA2AHandle obj = LLA2AHandle.__new__(LLA2AHandle)
        if owner is None:
            obj._ptr = <ncclLLA2AHandle_t *>_cyb_malloc(sizeof(ncclLLA2AHandle_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating LLA2AHandle")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclLLA2AHandle_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclLLA2AHandle_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_config_ext_dtype_offsets():
    cdef ncclConfigExt_t pod
    return _numpy.dtype({
        'names': ['next', 'key', 'val'],
        'formats': [_numpy.intp, _py_anon_pod0_dtype, _py_anon_pod1_dtype],
        'offsets': [
            (<intptr_t>&(pod.next)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.key)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.val)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclConfigExt_t),
    })

config_ext_dtype = _get_config_ext_dtype_offsets()

cdef class ConfigExt:
    """Empty-initialize an instance of `ncclConfigExt_t`.


    .. seealso:: `ncclConfigExt_t`
    """
    cdef:
        ncclConfigExt_t *_ptr
        object _owner
        bint _owned
        bint _readonly
        dict _refs

    def __init__(self):
        self._ptr = <ncclConfigExt_t *>_cyb_calloc(1, sizeof(ncclConfigExt_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating ConfigExt")
        self._owner = None
        self._owned = True
        self._readonly = False
        self._refs = {}

    def __dealloc__(self):
        cdef ncclConfigExt_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.ConfigExt object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef ConfigExt other_
        if not isinstance(other, ConfigExt):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclConfigExt_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclConfigExt_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclConfigExt_t *>_cyb_malloc(sizeof(ncclConfigExt_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating ConfigExt")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclConfigExt_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def next(self):
        """ConfigExt: """
        if self._ptr[0].next == NULL:
            return None
        ref = self._refs.get("next")
        if (
            ref is not None
            and (<ConfigExt>ref)._get_ptr() == <intptr_t>(self._ptr[0].next)
        ):
            return ref
        return ConfigExt.from_ptr(
            <intptr_t>(self._ptr[0].next),
            readonly=self._readonly,
            owner=self,
        )

    @next.setter
    def next(self, val):
        if self._readonly:
            raise ValueError("This ConfigExt instance is read-only")
        if val is None:
            self._ptr[0].next = NULL
            self._refs.pop("next", None)
            return
        if not isinstance(val, ConfigExt):
            raise TypeError("next must be ConfigExt or None")
        self._ptr[0].next = <ncclConfigExt*><intptr_t>(<ConfigExt>val)._get_ptr()
        self._refs["next"] = val

    @property
    def key(self):
        """_py_anon_pod0: """
        return _py_anon_pod0.from_ptr(
            <intptr_t>&(self._ptr[0].key),
            readonly=self._readonly,
            owner=self,
        )

    @key.setter
    def key(self, val):
        if self._readonly:
            raise ValueError("This ConfigExt instance is read-only")
        cdef _py_anon_pod0 val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].key), <void *>(val_._get_ptr()), sizeof(nccl_bindings_nccl__anon_pod0) * 1)

    @property
    def val(self):
        """_py_anon_pod1: """
        return _py_anon_pod1.from_ptr(
            <intptr_t>&(self._ptr[0].val),
            readonly=self._readonly,
            owner=self,
        )

    @val.setter
    def val(self, val):
        if self._readonly:
            raise ValueError("This ConfigExt instance is read-only")
        cdef _py_anon_pod1 val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].val), <void *>(val_._get_ptr()), sizeof(nccl_bindings_nccl__anon_pod1) * 1)

    @staticmethod
    def from_buffer(buffer):
        """Create an ConfigExt instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclConfigExt_t), ConfigExt)

    @staticmethod
    def from_data(data):
        """Create an ConfigExt instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `config_ext_dtype` holding the data.
        """
        return _cyb_from_data(data, "config_ext_dtype", config_ext_dtype, ConfigExt)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an ConfigExt instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef ConfigExt obj = ConfigExt.__new__(ConfigExt)
        if owner is None:
            obj._ptr = <ncclConfigExt_t *>_cyb_malloc(sizeof(ncclConfigExt_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating ConfigExt")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclConfigExt_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclConfigExt_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        obj._refs = {}
        return obj


cdef _get_team_requirements_dtype_offsets():
    cdef ncclTeamRequirements_t pod
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
        self._ptr = <ncclTeamRequirements_t *>_cyb_calloc(1, sizeof(ncclTeamRequirements_t))
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
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclTeamRequirements_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclTeamRequirements_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclTeamRequirements_t *>_cyb_malloc(sizeof(ncclTeamRequirements_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating TeamRequirements")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclTeamRequirements_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def team(self):
        """Team: """
        return Team.from_ptr(
            <intptr_t>&(self._ptr[0].team),
            readonly=self._readonly,
            owner=self,
        )

    @team.setter
    def team(self, val):
        if self._readonly:
            raise ValueError("This TeamRequirements instance is read-only")
        cdef Team val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].team), <void *>(val_._get_ptr()), sizeof(ncclTeam_t) * 1)

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

    @staticmethod
    def from_buffer(buffer):
        """Create an TeamRequirements instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclTeamRequirements_t), TeamRequirements)

    @staticmethod
    def from_data(data):
        """Create an TeamRequirements instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `team_requirements_dtype` holding the data.
        """
        return _cyb_from_data(data, "team_requirements_dtype", team_requirements_dtype, TeamRequirements)

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
            obj._ptr = <ncclTeamRequirements_t *>_cyb_malloc(sizeof(ncclTeamRequirements_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating TeamRequirements")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclTeamRequirements_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclTeamRequirements_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_dev_comm_dtype_offsets():
    cdef ncclDevComm_t pod
    return _numpy.dtype({
        'names': ['magic', 'version', 'rank', 'n_ranks', 'n_ranks_rcp32', 'lsa_rank', 'lsa_size', 'lsa_size_rcp32', 'window_table', 'resource_window', 'resource_window_inlined', 'hybrid_dense_gin_barrier', 'lsa_multimem', 'lsa_barrier', 'rail_gin_barrier', 'gin_connection_count', 'backend_index', 'gin_net_device_types', 'gin_handles', 'gin_signal_count', 'gin_counter_count', 'gin_signal_shadows', 'gin_context_count', 'gin_connection_stride', 'gin_context_stride', 'gin_strong_legacy_signals', 'abort_flag', 'hybrid_lsa_barrier', 'hybrid_rail_gin_barrier', 'world_gin_barrier', 'gin_connection_stride_rcp32', 'cft_rank', 'cft_size', 'cft_multimem_rank', 'cft_multimem_size', 'cft_multimem_size_rcp32', 'uc_le_id', 'mc_le_id', 'cft_barrier', 'cft_multimem_barrier'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.uint32, _numpy.intp, _numpy.intp, resource_window_vidmem_dtype, gin_barrier_handle_dtype, multimem_handle_dtype, lsa_barrier_handle_dtype, gin_barrier_handle_dtype, _numpy.uint8, _numpy.uint8, (_numpy.uint8, 4), (_numpy.int64, 4), _numpy.int32, _numpy.int32, _numpy.intp, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.intp, lsa_barrier_handle_dtype, gin_barrier_handle_dtype, gin_barrier_handle_dtype, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.uint32, _numpy.uint32, _numpy.uint32, cft_barrier_handle_dtype, cft_barrier_handle_dtype],
        'offsets': [
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.rank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nRanks)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nRanks_rcp32)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaRank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaSize)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaSize_rcp32)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.windowTable)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.resourceWindow)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.resourceWindow_inlined)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.hybridDenseGinBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaMultimem)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.lsaBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.railGinBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginConnectionCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.backendIndex)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginNetDeviceTypes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginHandles)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginSignalCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginCounterCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginSignalShadows)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginContextCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginConnectionStride)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginContextStride)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginStrongLegacySignals)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.abortFlag)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.hybridLsaBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.hybridRailGinBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.worldGinBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginConnectionStride_rcp32)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cftRank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cftSize)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cftMultimemRank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cftMultimemSize)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cftMultimemSize_rcp32)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ucLeId)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.mcLeId)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cftBarrier)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cftMultimemBarrier)) - (<intptr_t>&pod),
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
        self._ptr = <ncclDevComm_t *>_cyb_calloc(1, sizeof(ncclDevComm_t))
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
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclDevComm_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclDevComm_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclDevComm_t *>_cyb_malloc(sizeof(ncclDevComm_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DevComm")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclDevComm_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def resource_window_inlined(self):
        """ResourceWindowVidmem: """
        return ResourceWindowVidmem.from_ptr(
            <intptr_t>&(self._ptr[0].resourceWindow_inlined),
            readonly=self._readonly,
            owner=self,
        )

    @resource_window_inlined.setter
    def resource_window_inlined(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef ResourceWindowVidmem val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].resourceWindow_inlined), <void *>(val_._get_ptr()), sizeof(ncclResourceWindow_vidmem_t) * 1)

    @property
    def hybrid_dense_gin_barrier(self):
        """GinBarrierHandle: """
        return GinBarrierHandle.from_ptr(
            <intptr_t>&(self._ptr[0].hybridDenseGinBarrier),
            readonly=self._readonly,
            owner=self,
        )

    @hybrid_dense_gin_barrier.setter
    def hybrid_dense_gin_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef GinBarrierHandle val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].hybridDenseGinBarrier), <void *>(val_._get_ptr()), sizeof(ncclGinBarrierHandle_t) * 1)

    @property
    def lsa_multimem(self):
        """MultimemHandle: """
        return MultimemHandle.from_ptr(
            <intptr_t>&(self._ptr[0].lsaMultimem),
            readonly=self._readonly,
            owner=self,
        )

    @lsa_multimem.setter
    def lsa_multimem(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef MultimemHandle val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].lsaMultimem), <void *>(val_._get_ptr()), sizeof(ncclMultimemHandle_t) * 1)

    @property
    def lsa_barrier(self):
        """LsaBarrierHandle: """
        return LsaBarrierHandle.from_ptr(
            <intptr_t>&(self._ptr[0].lsaBarrier),
            readonly=self._readonly,
            owner=self,
        )

    @lsa_barrier.setter
    def lsa_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef LsaBarrierHandle val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].lsaBarrier), <void *>(val_._get_ptr()), sizeof(ncclLsaBarrierHandle_t) * 1)

    @property
    def rail_gin_barrier(self):
        """GinBarrierHandle: """
        return GinBarrierHandle.from_ptr(
            <intptr_t>&(self._ptr[0].railGinBarrier),
            readonly=self._readonly,
            owner=self,
        )

    @rail_gin_barrier.setter
    def rail_gin_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef GinBarrierHandle val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].railGinBarrier), <void *>(val_._get_ptr()), sizeof(ncclGinBarrierHandle_t) * 1)

    @property
    def hybrid_lsa_barrier(self):
        """LsaBarrierHandle: """
        return LsaBarrierHandle.from_ptr(
            <intptr_t>&(self._ptr[0].hybridLsaBarrier),
            readonly=self._readonly,
            owner=self,
        )

    @hybrid_lsa_barrier.setter
    def hybrid_lsa_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef LsaBarrierHandle val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].hybridLsaBarrier), <void *>(val_._get_ptr()), sizeof(ncclLsaBarrierHandle_t) * 1)

    @property
    def hybrid_rail_gin_barrier(self):
        """GinBarrierHandle: """
        return GinBarrierHandle.from_ptr(
            <intptr_t>&(self._ptr[0].hybridRailGinBarrier),
            readonly=self._readonly,
            owner=self,
        )

    @hybrid_rail_gin_barrier.setter
    def hybrid_rail_gin_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef GinBarrierHandle val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].hybridRailGinBarrier), <void *>(val_._get_ptr()), sizeof(ncclGinBarrierHandle_t) * 1)

    @property
    def world_gin_barrier(self):
        """GinBarrierHandle: """
        return GinBarrierHandle.from_ptr(
            <intptr_t>&(self._ptr[0].worldGinBarrier),
            readonly=self._readonly,
            owner=self,
        )

    @world_gin_barrier.setter
    def world_gin_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef GinBarrierHandle val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].worldGinBarrier), <void *>(val_._get_ptr()), sizeof(ncclGinBarrierHandle_t) * 1)

    @property
    def cft_barrier(self):
        """CftBarrierHandle: """
        return CftBarrierHandle.from_ptr(
            <intptr_t>&(self._ptr[0].cftBarrier),
            readonly=self._readonly,
            owner=self,
        )

    @cft_barrier.setter
    def cft_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef CftBarrierHandle val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].cftBarrier), <void *>(val_._get_ptr()), sizeof(ncclCftBarrierHandle_t) * 1)

    @property
    def cft_multimem_barrier(self):
        """CftBarrierHandle: """
        return CftBarrierHandle.from_ptr(
            <intptr_t>&(self._ptr[0].cftMultimemBarrier),
            readonly=self._readonly,
            owner=self,
        )

    @cft_multimem_barrier.setter
    def cft_multimem_barrier(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        cdef CftBarrierHandle val_ = val
        _cyb_memcpy(<void *>&(self._ptr[0].cftMultimemBarrier), <void *>(val_._get_ptr()), sizeof(ncclCftBarrierHandle_t) * 1)

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].magic = val

    @property
    def version(self):
        """int: """
        return self._ptr[0].version

    @version.setter
    def version(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].version = val

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
    def backend_index(self):
        """int: """
        return self._ptr[0].backendIndex

    @backend_index.setter
    def backend_index(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].backendIndex = val

    @property
    def gin_net_device_types(self):
        """~_numpy.uint8: (array of length 4)."""
        cdef _cyb_view.array arr = _cyb_view.array(shape=(4,), itemsize=sizeof(uint8_t), format="B", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].ginNetDeviceTypes))
        return _numpy.asarray(arr)

    @gin_net_device_types.setter
    def gin_net_device_types(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        if len(val) != 4:
            raise ValueError(f"Expected length { 4 } for field gin_net_device_types, got {len(val)}")
        cdef _cyb_view.array arr = _cyb_view.array(shape=(4,), itemsize=sizeof(uint8_t), format="B", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.uint8)
        _cyb_memcpy(<void *>(&(self._ptr[0].ginNetDeviceTypes)), <void *>(arr.data), sizeof(uint8_t) * len(val))

    @property
    def gin_handles(self):
        """~_numpy.int64: (array of length 4)."""
        cdef _cyb_view.array arr = _cyb_view.array(shape=(4,), itemsize=sizeof(intptr_t), format="q", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].ginHandles))
        return _numpy.asarray(arr)

    @gin_handles.setter
    def gin_handles(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        if len(val) != 4:
            raise ValueError(f"Expected length { 4 } for field gin_handles, got {len(val)}")
        cdef _cyb_view.array arr = _cyb_view.array(shape=(4,), itemsize=sizeof(intptr_t), format="q", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.intp)
        _cyb_memcpy(<void *>(&(self._ptr[0].ginHandles)), <void *>(arr.data), sizeof(intptr_t) * len(val))

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
    def gin_connection_stride(self):
        """int: """
        return self._ptr[0].ginConnectionStride

    @gin_connection_stride.setter
    def gin_connection_stride(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginConnectionStride = val

    @property
    def gin_context_stride(self):
        """int: """
        return self._ptr[0].ginContextStride

    @gin_context_stride.setter
    def gin_context_stride(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginContextStride = val

    @property
    def gin_strong_legacy_signals(self):
        """int: """
        return self._ptr[0].ginStrongLegacySignals

    @gin_strong_legacy_signals.setter
    def gin_strong_legacy_signals(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginStrongLegacySignals = val

    @property
    def abort_flag(self):
        """int: """
        return <intptr_t>(self._ptr[0].abortFlag)

    @abort_flag.setter
    def abort_flag(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].abortFlag = <uint32_t*><intptr_t>val

    @property
    def gin_connection_stride_rcp32(self):
        """int: """
        return self._ptr[0].ginConnectionStride_rcp32

    @gin_connection_stride_rcp32.setter
    def gin_connection_stride_rcp32(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ginConnectionStride_rcp32 = val

    @property
    def cft_rank(self):
        """int: """
        return self._ptr[0].cftRank

    @cft_rank.setter
    def cft_rank(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].cftRank = val

    @property
    def cft_size(self):
        """int: """
        return self._ptr[0].cftSize

    @cft_size.setter
    def cft_size(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].cftSize = val

    @property
    def cft_multimem_rank(self):
        """int: """
        return self._ptr[0].cftMultimemRank

    @cft_multimem_rank.setter
    def cft_multimem_rank(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].cftMultimemRank = val

    @property
    def cft_multimem_size(self):
        """int: """
        return self._ptr[0].cftMultimemSize

    @cft_multimem_size.setter
    def cft_multimem_size(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].cftMultimemSize = val

    @property
    def cft_multimem_size_rcp32(self):
        """int: """
        return self._ptr[0].cftMultimemSize_rcp32

    @cft_multimem_size_rcp32.setter
    def cft_multimem_size_rcp32(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].cftMultimemSize_rcp32 = val

    @property
    def uc_le_id(self):
        """int: """
        return <uint32_t>(self._ptr[0].ucLeId)

    @uc_le_id.setter
    def uc_le_id(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].ucLeId = <ncclCftLeId><uint32_t>val

    @property
    def mc_le_id(self):
        """int: """
        return <uint32_t>(self._ptr[0].mcLeId)

    @mc_le_id.setter
    def mc_le_id(self, val):
        if self._readonly:
            raise ValueError("This DevComm instance is read-only")
        self._ptr[0].mcLeId = <ncclCftLeId><uint32_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an DevComm instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclDevComm_t), DevComm)

    @staticmethod
    def from_data(data):
        """Create an DevComm instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dev_comm_dtype` holding the data.
        """
        return _cyb_from_data(data, "dev_comm_dtype", dev_comm_dtype, DevComm)

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
            obj._ptr = <ncclDevComm_t *>_cyb_malloc(sizeof(ncclDevComm_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DevComm")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclDevComm_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclDevComm_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_coll_config_dtype_offsets():
    cdef ncclCollConfig_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'version', 'ext', 'min_ctas', 'max_ctas', 'nvls_ctas', 'cga_cluster_size', 'alg_selection', 'force_alg_selection', 'cta_policy', 'user_profiler_tag'],
        'formats': [_numpy.uint64, _numpy.uint32, _numpy.uint32, _numpy.intp, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.intp, _numpy.int32, _numpy.int32, _numpy.uint64],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.magic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ext)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.minCTAs)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.maxCTAs)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.nvlsCTAs)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cgaClusterSize)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.algSelection)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.forceAlgSelection)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.CTAPolicy)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.userProfilerTag)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclCollConfig_t),
    })

coll_config_dtype = _get_coll_config_dtype_offsets()

cdef class CollConfig:
    """Initialize an instance of `ncclCollConfig_t` using configured defaults.


    .. seealso:: `ncclCollConfig_t`
    """
    cdef:
        ncclCollConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly
        dict _refs

    def __init__(self):
        self._ptr = <ncclCollConfig_t *>_cyb_calloc(1, sizeof(ncclCollConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating CollConfig")
        self._owner = None
        self._owned = True
        self._readonly = False
        self._refs = {}

        self._ptr[0].size = sizeof(ncclCollConfig_t)
        self._ptr[0].magic = 0xcafebeef
        self._ptr[0].version = __version_code__
        self._ptr[0].ext = NULL
        self._ptr[0].minCTAs = -2147483648
        self._ptr[0].maxCTAs = -2147483648
        self._ptr[0].nvlsCTAs = -2147483648
        self._ptr[0].cgaClusterSize = -2147483648
        self._ptr[0].algSelection = NULL
        self._ptr[0].forceAlgSelection = 1
        self._ptr[0].CTAPolicy = -2147483648
        self._ptr[0].userProfilerTag = 0

    def __dealloc__(self):
        cdef ncclCollConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

    def __repr__(self):
        return f"<{__name__}.CollConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef CollConfig other_
        if not isinstance(other, CollConfig):
            return False
        other_ = other
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclCollConfig_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclCollConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclCollConfig_t *>_cyb_malloc(sizeof(ncclCollConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating CollConfig")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclCollConfig_t))
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
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def magic(self):
        """int: """
        return self._ptr[0].magic

    @magic.setter
    def magic(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].magic = val

    @property
    def version(self):
        """int: """
        return self._ptr[0].version

    @version.setter
    def version(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].version = val

    @property
    def ext(self):
        """int: """
        return <intptr_t>(self._ptr[0].ext)

    @ext.setter
    def ext(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].ext = <ncclConfigExt_t*><intptr_t>val

    @property
    def min_ctas(self):
        """int: """
        return self._ptr[0].minCTAs

    @min_ctas.setter
    def min_ctas(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].minCTAs = val

    @property
    def max_ctas(self):
        """int: """
        return self._ptr[0].maxCTAs

    @max_ctas.setter
    def max_ctas(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].maxCTAs = val

    @property
    def nvls_ctas(self):
        """int: """
        return self._ptr[0].nvlsCTAs

    @nvls_ctas.setter
    def nvls_ctas(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].nvlsCTAs = val

    @property
    def cga_cluster_size(self):
        """int: """
        return self._ptr[0].cgaClusterSize

    @cga_cluster_size.setter
    def cga_cluster_size(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].cgaClusterSize = val

    @property
    def alg_selection(self):
        """str: """
        cdef char* ptr = <char*>self._ptr[0].algSelection
        if ptr:
            return _cyb_cpython.PyUnicode_FromString(ptr)
        return ""

    @alg_selection.setter
    def alg_selection(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        cdef bytes buf = val.encode()
        cdef char *ptr = buf
        self._refs["alg_selection"] = buf
        self._ptr[0].algSelection = <char *><intptr_t>ptr

    @property
    def force_alg_selection(self):
        """int: """
        return self._ptr[0].forceAlgSelection

    @force_alg_selection.setter
    def force_alg_selection(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].forceAlgSelection = val

    @property
    def cta_policy(self):
        """int: """
        return self._ptr[0].CTAPolicy

    @cta_policy.setter
    def cta_policy(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].CTAPolicy = val

    @property
    def user_profiler_tag(self):
        """int: """
        return self._ptr[0].userProfilerTag

    @user_profiler_tag.setter
    def user_profiler_tag(self, val):
        if self._readonly:
            raise ValueError("This CollConfig instance is read-only")
        self._ptr[0].userProfilerTag = val

    @staticmethod
    def from_buffer(buffer):
        """Create an CollConfig instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclCollConfig_t), CollConfig)

    @staticmethod
    def from_data(data):
        """Create an CollConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `coll_config_dtype` holding the data.
        """
        return _cyb_from_data(data, "coll_config_dtype", coll_config_dtype, CollConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an CollConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef CollConfig obj = CollConfig.__new__(CollConfig)
        if owner is None:
            obj._ptr = <ncclCollConfig_t *>_cyb_malloc(sizeof(ncclCollConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating CollConfig")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclCollConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclCollConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        obj._refs = {}
        return obj


cdef _get_dev_comm_requirements_dtype_offsets():
    cdef ncclDevCommRequirements_t pod
    return _numpy.dtype({
        'names': ['size_', 'magic', 'version', 'resource_requirements_list', 'team_requirements_list', 'lsa_multimem', 'barrier_count', 'lsa_barrier_count', 'rail_gin_barrier_count', 'lsa_ll_a2a_block_count', 'lsa_ll_a2a_slot_count', 'gin_force_enable', 'gin_context_count', 'gin_signal_count', 'gin_counter_count', 'gin_connection_type', 'gin_exclusive_contexts', 'gin_queue_depth', 'gin_traffic_class', 'world_gin_barrier_count', 'gin_strong_signals_required', 'gin_va_signals_required', 'gin_custom_stride', 'gin_type', 'use_runtime_version', 'cft_caps', 'cft_barrier_count'],
        'formats': [_numpy.uint64, _numpy.uint32, _numpy.uint32, _numpy.intp, _numpy.intp, _numpy.uint8, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.int32, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.uint8, _numpy.int32, _numpy.int32, _numpy.uint8, _numpy.int32, _numpy.int32],
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
            (<intptr_t>&(pod.ginTrafficClass)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.worldGinBarrierCount)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginStrongSignalsRequired)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginVaSignalsRequired)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginCustomStride)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ginType)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.useRuntimeVersion)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cftCaps)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cftBarrierCount)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclDevCommRequirements_t),
    })

dev_comm_requirements_dtype = _get_dev_comm_requirements_dtype_offsets()

cdef class DevCommRequirements:
    """Initialize an instance of `ncclDevCommRequirements_t` using configured defaults.


    .. seealso:: `ncclDevCommRequirements_t`
    """
    cdef:
        ncclDevCommRequirements_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclDevCommRequirements_t *>_cyb_calloc(1, sizeof(ncclDevCommRequirements_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating DevCommRequirements")
        self._owner = None
        self._owned = True
        self._readonly = False

        self._ptr[0].size = sizeof(ncclDevCommRequirements_t)
        self._ptr[0].magic = 0xcafebeef
        self._ptr[0].version = __version_code__
        self._ptr[0].resourceRequirementsList = NULL
        self._ptr[0].teamRequirementsList = NULL
        self._ptr[0].lsaMultimem = 0
        self._ptr[0].barrierCount = 0
        self._ptr[0].lsaBarrierCount = 0
        self._ptr[0].railGinBarrierCount = 0
        self._ptr[0].lsaLLA2ABlockCount = 0
        self._ptr[0].lsaLLA2ASlotCount = 0
        self._ptr[0].ginForceEnable = 0
        self._ptr[0].ginContextCount = 4
        self._ptr[0].ginSignalCount = 0
        self._ptr[0].ginCounterCount = 0
        self._ptr[0].ginConnectionType = NCCL_GIN_CONNECTION_NONE
        self._ptr[0].ginExclusiveContexts = 0
        self._ptr[0].ginQueueDepth = 0
        self._ptr[0].ginTrafficClass = -2147483648
        self._ptr[0].worldGinBarrierCount = 0
        self._ptr[0].ginStrongSignalsRequired = 1
        self._ptr[0].ginVaSignalsRequired = 1
        self._ptr[0].ginCustomStride = 1
        self._ptr[0].ginType = NCCL_GIN_TYPE_NONE
        self._ptr[0].useRuntimeVersion = 0
        self._ptr[0].cftCaps = 0
        self._ptr[0].cftBarrierCount = 0

    def __dealloc__(self):
        cdef ncclDevCommRequirements_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            _cyb_free(ptr)

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
        return (_cyb_memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclDevCommRequirements_t)) == 0)

    def __getbuffer__(self, _cyb_cpython.Py_buffer *buffer, int flags):
        _cyb___getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclDevCommRequirements_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <ncclDevCommRequirements_t *>_cyb_malloc(sizeof(ncclDevCommRequirements_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DevCommRequirements")
            _cyb_memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclDevCommRequirements_t))
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
    def gin_traffic_class(self):
        """int: """
        return self._ptr[0].ginTrafficClass

    @gin_traffic_class.setter
    def gin_traffic_class(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginTrafficClass = val

    @property
    def world_gin_barrier_count(self):
        """int: """
        return self._ptr[0].worldGinBarrierCount

    @world_gin_barrier_count.setter
    def world_gin_barrier_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].worldGinBarrierCount = val

    @property
    def gin_strong_signals_required(self):
        """int: """
        return self._ptr[0].ginStrongSignalsRequired

    @gin_strong_signals_required.setter
    def gin_strong_signals_required(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginStrongSignalsRequired = val

    @property
    def gin_va_signals_required(self):
        """int: """
        return self._ptr[0].ginVaSignalsRequired

    @gin_va_signals_required.setter
    def gin_va_signals_required(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginVaSignalsRequired = val

    @property
    def gin_custom_stride(self):
        """int: """
        return self._ptr[0].ginCustomStride

    @gin_custom_stride.setter
    def gin_custom_stride(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginCustomStride = val

    @property
    def gin_type(self):
        """int: """
        return <int>(self._ptr[0].ginType)

    @gin_type.setter
    def gin_type(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].ginType = <ncclGinType_t><int>val

    @property
    def use_runtime_version(self):
        """int: """
        return self._ptr[0].useRuntimeVersion

    @use_runtime_version.setter
    def use_runtime_version(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].useRuntimeVersion = val

    @property
    def cft_caps(self):
        """int: """
        return self._ptr[0].cftCaps

    @cft_caps.setter
    def cft_caps(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].cftCaps = val

    @property
    def cft_barrier_count(self):
        """int: """
        return self._ptr[0].cftBarrierCount

    @cft_barrier_count.setter
    def cft_barrier_count(self, val):
        if self._readonly:
            raise ValueError("This DevCommRequirements instance is read-only")
        self._ptr[0].cftBarrierCount = val

    @staticmethod
    def from_buffer(buffer):
        """Create an DevCommRequirements instance with the memory from the given buffer."""
        return _cyb_from_buffer(buffer, sizeof(ncclDevCommRequirements_t), DevCommRequirements)

    @staticmethod
    def from_data(data):
        """Create an DevCommRequirements instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dev_comm_requirements_dtype` holding the data.
        """
        return _cyb_from_data(data, "dev_comm_requirements_dtype", dev_comm_requirements_dtype, DevCommRequirements)

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
            obj._ptr = <ncclDevCommRequirements_t *>_cyb_malloc(sizeof(ncclDevCommRequirements_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DevCommRequirements")
            _cyb_memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclDevCommRequirements_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclDevCommRequirements_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef class Comm:
    cdef ncclComm_t _handle
    cdef ncclComm_t* _handle_ptr
    cdef object _owner

    def __cinit__(self, *args, **kwargs):
        self._handle_ptr = &self._handle

    def __init__(self, intptr_t handle=0):
        self._handle = <ncclComm_t>handle
        self._handle_ptr = &self._handle
        self._owner = None

    @property
    def handle(self):
        """Get the current handle value as a Python :class:`int`."""
        return <intptr_t>self._handle_ptr[0]

    @property
    def handle_ptr(self):
        """Get the handle slot address as a Python :class:`int`."""
        return <intptr_t>self._handle_ptr

    cdef ncclComm_t _get_handle(self):
        return self._handle_ptr[0]

    cdef ncclComm_t* _get_handle_ptr(self):
        return self._handle_ptr

    @staticmethod
    cdef Comm _from_handle_ptr(ncclComm_t* handle_ptr, object owner):
        cdef Comm obj = Comm.__new__(Comm)
        obj._handle_ptr = handle_ptr
        obj._owner = owner
        return obj

    def __int__(self):
        return <intptr_t>self._handle_ptr[0]

    def __index__(self):
        return <intptr_t>self._handle_ptr[0]

    def __bool__(self):
        return self._handle_ptr[0] != NULL

    def __repr__(self):
        return f"<{__name__}.Comm handle={self.handle:#x} object at {hex(id(self))}>"


cdef class Window:
    cdef ncclWindow_t _handle
    cdef ncclWindow_t* _handle_ptr
    cdef object _owner

    def __cinit__(self, *args, **kwargs):
        self._handle_ptr = &self._handle

    def __init__(self, intptr_t handle=0):
        self._handle = <ncclWindow_t>handle
        self._handle_ptr = &self._handle
        self._owner = None

    @property
    def handle(self):
        """Get the current handle value as a Python :class:`int`."""
        return <intptr_t>self._handle_ptr[0]

    @property
    def handle_ptr(self):
        """Get the handle slot address as a Python :class:`int`."""
        return <intptr_t>self._handle_ptr

    cdef ncclWindow_t _get_handle(self):
        return self._handle_ptr[0]

    cdef ncclWindow_t* _get_handle_ptr(self):
        return self._handle_ptr

    @staticmethod
    cdef Window _from_handle_ptr(ncclWindow_t* handle_ptr, object owner):
        cdef Window obj = Window.__new__(Window)
        obj._handle_ptr = handle_ptr
        obj._owner = owner
        return obj

    def __int__(self):
        return <intptr_t>self._handle_ptr[0]

    def __index__(self):
        return <intptr_t>self._handle_ptr[0]

    def __bool__(self):
        return self._handle_ptr[0] != NULL

    def __repr__(self):
        return f"<{__name__}.Window handle={self.handle:#x} object at {hex(id(self))}>"


###############################################################################
# Enum
###############################################################################

class Result(_cyb_IntEnum):
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
    Timeout = ncclTimeout
    NumResults = ncclNumResults

class HostCftMode(_cyb_IntEnum):
    """
    See `ncclHostCftMode_t`.
    """
    Default = ncclHostCftDefault
    Enable = ncclHostCftEnable
    Disable = ncclHostCftDisable
    Fallback = ncclHostCftFallback

class CommMemStat(_cyb_IntEnum):
    """
    See `ncclCommMemStat_t`.
    """
    GpuMemSuspend = ncclStatGpuMemSuspend
    GpuMemSuspended = ncclStatGpuMemSuspended
    GpuMemPersist = ncclStatGpuMemPersist
    GpuMemTotal = ncclStatGpuMemTotal

class RedOpDummy(_cyb_IntEnum):
    """
    See `ncclRedOp_dummy_t`.
    """
    NumOps_dummy = ncclNumOps_dummy

class RedOp(_cyb_IntEnum):
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

class DataType(_cyb_IntEnum):
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

class ScalarResidence(_cyb_IntEnum):
    """
    See `ncclScalarResidence_t`.
    """
    Device = ncclScalarDevice
    HostImmediate = ncclScalarHostImmediate

class GinType(_cyb_IntEnum):
    """
    See `ncclGinType_t`.
    """
    NONE = NCCL_GIN_TYPE_NONE
    PROXY = NCCL_GIN_TYPE_PROXY
    GDAKI = NCCL_GIN_TYPE_GDAKI
    GPI = NCCL_GIN_TYPE_GPI
    EFA_GDA = NCCL_GIN_TYPE_EFA_GDA
    GIN_MAX_TYPES = NCCL_GIN_MAX_TYPES

class GinConnectionType(_cyb_IntEnum):
    """
    See `ncclGinConnectionType_t`.
    """
    NONE = NCCL_GIN_CONNECTION_NONE
    FULL = NCCL_GIN_CONNECTION_FULL
    RAIL = NCCL_GIN_CONNECTION_RAIL
    CUSTOM_STRIDE = NCCL_GIN_CONNECTION_CUSTOM_STRIDE

class CftTeamMode(_cyb_IntEnum):
    """
    See `ncclCftTeamMode_t`.
    """
    FLAT = NCCL_CFT_TEAM_FLAT
    HIER_MULTIMEM = NCCL_CFT_TEAM_HIER_MULTIMEM
    HIER_LSA = NCCL_CFT_TEAM_HIER_LSA


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
    if status != ncclSuccess and status != ncclInProgress:
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


cpdef object get_unique_id():
    cdef UniqueId unique_id_py = UniqueId()
    cdef ncclUniqueId *unique_id = <ncclUniqueId *><intptr_t>(unique_id_py._get_ptr())
    with nogil:
        __status__ = ncclGetUniqueId(unique_id)
    check_status(__status__)
    return unique_id_py


cpdef object comm_init_rank_config(int nranks, comm_id, int rank, intptr_t config):
    cdef void* _comm_id_ = <void *>_cyb_get_buffer_pointer(comm_id, -1, readonly=False)
    cdef Comm comm_py = Comm()
    cdef ncclComm_t *comm = comm_py._get_handle_ptr()
    with nogil:
        __status__ = ncclCommInitRankConfig(comm, nranks, (<ncclUniqueId*>(_comm_id_))[0], rank, <ncclConfig_t*>config)
    check_status(__status__)
    return comm_py


cpdef object comm_init_rank(int nranks, comm_id, int rank):
    cdef void* _comm_id_ = <void *>_cyb_get_buffer_pointer(comm_id, -1, readonly=False)
    cdef Comm comm_py = Comm()
    cdef ncclComm_t *comm = comm_py._get_handle_ptr()
    with nogil:
        __status__ = ncclCommInitRank(comm, nranks, (<ncclUniqueId*>(_comm_id_))[0], rank)
    check_status(__status__)
    return comm_py


cpdef object comm_init_all(int ndev, devlist):
    cdef nullable_unique_ptr[ vector[int] ] _devlist_
    get_resource_ptr[int](_devlist_, devlist, <int*>NULL)
    if ndev == 0:
        return []
    cdef _cyb_view.array comm = _cyb_view.array(shape=(ndev,), itemsize=sizeof(intptr_t), format="q", mode="c")
    cdef intptr_t *comm_ptr = <intptr_t *>(comm.data)
    cdef list comm_wrappers = []
    cdef Py_ssize_t comm_index
    for comm_index in range(ndev):
        comm_wrappers.append(
            Comm._from_handle_ptr(
                <ncclComm_t*>(comm_ptr + comm_index),
                comm,
            )
        )
    with nogil:
        __status__ = ncclCommInitAll(<ncclComm_t*>comm_ptr, ndev, <const int*>(_devlist_.data()))
    check_status(__status__)
    return comm_wrappers


cpdef comm_finalize(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclCommFinalize(_comm_handle)
    check_status(__status__)


cpdef comm_destroy(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclCommDestroy(_comm_handle)
    check_status(__status__)


cpdef comm_abort(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclCommAbort(_comm_handle)
    check_status(__status__)


cpdef comm_revoke(object comm, int revoke_flags):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclCommRevoke(_comm_handle, revoke_flags)
    check_status(__status__)


cpdef object comm_split(object comm, int color, int key, intptr_t config):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef Comm newcomm_py = Comm()
    cdef ncclComm_t *newcomm = newcomm_py._get_handle_ptr()
    with nogil:
        __status__ = ncclCommSplit(_comm_handle, color, key, newcomm, <ncclConfig_t*>config)
    check_status(__status__)
    return newcomm_py


cpdef object comm_shrink(object comm, exclude_ranks_list, int exclude_ranks_count, intptr_t config, int shrink_flags):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef nullable_unique_ptr[ vector[int] ] _exclude_ranks_list_
    get_resource_ptr[int](_exclude_ranks_list_, exclude_ranks_list, <int*>NULL)
    cdef Comm newcomm_py = Comm()
    cdef ncclComm_t *newcomm = newcomm_py._get_handle_ptr()
    with nogil:
        __status__ = ncclCommShrink(_comm_handle, <int*>(_exclude_ranks_list_.data()), exclude_ranks_count, newcomm, <ncclConfig_t*>config, shrink_flags)
    check_status(__status__)
    return newcomm_py


cpdef object comm_get_unique_id(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef UniqueId unique_id_py = UniqueId()
    cdef ncclUniqueId *unique_id = <ncclUniqueId *><intptr_t>(unique_id_py._get_ptr())
    with nogil:
        __status__ = ncclCommGetUniqueId(_comm_handle, unique_id)
    check_status(__status__)
    return unique_id_py


cpdef object comm_grow(object comm, int n_ranks, intptr_t unique_id, int rank, intptr_t config):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef Comm newcomm_py = Comm()
    cdef ncclComm_t *newcomm = newcomm_py._get_handle_ptr()
    with nogil:
        __status__ = ncclCommGrow(_comm_handle, n_ranks, <const ncclUniqueId*>unique_id, rank, newcomm, <ncclConfig_t*>config)
    check_status(__status__)
    return newcomm_py


cpdef object comm_init_rank_scalable(int nranks, int myrank, int n_id, comm_ids, intptr_t config):
    cdef void* _comm_ids_ = <void *>_cyb_get_buffer_pointer(comm_ids, -1, readonly=False)
    cdef Comm newcomm_py = Comm()
    cdef ncclComm_t *newcomm = newcomm_py._get_handle_ptr()
    with nogil:
        __status__ = ncclCommInitRankScalable(newcomm, nranks, myrank, n_id, <ncclUniqueId*>_comm_ids_, <ncclConfig_t*>config)
    check_status(__status__)
    return newcomm_py


cpdef str get_error_string(int result):
    cdef const char *_output_cstr_
    cdef bytes _output_
    with nogil:
        _output_cstr_ = ncclGetErrorString(<_Result>result)
    _output_ = _output_cstr_
    return _output_.decode()


cpdef str get_last_error(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef const char *_output_cstr_
    cdef bytes _output_
    with nogil:
        _output_cstr_ = ncclGetLastError(_comm_handle)
    _output_ = _output_cstr_
    return _output_.decode()


cpdef int comm_get_async_error(object comm) except? -1:
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef _Result async_error
    with nogil:
        __status__ = ncclCommGetAsyncError(_comm_handle, &async_error)
    check_status(__status__)
    return <int>async_error


cpdef int comm_count(object comm) except? -1:
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef int count
    with nogil:
        __status__ = ncclCommCount(_comm_handle, &count)
    check_status(__status__)
    return count


cpdef int comm_cu_device(object comm) except? -1:
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef int device
    with nogil:
        __status__ = ncclCommCuDevice(_comm_handle, &device)
    check_status(__status__)
    return device


cpdef int comm_user_rank(object comm) except? -1:
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef int rank
    with nogil:
        __status__ = ncclCommUserRank(_comm_handle, &rank)
    check_status(__status__)
    return rank


cpdef intptr_t comm_register(object comm, intptr_t buff, size_t size) except? 0:
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef void* handle
    with nogil:
        __status__ = ncclCommRegister(_comm_handle, <void*>buff, size, &handle)
    check_status(__status__)
    return <intptr_t>handle


cpdef comm_deregister(object comm, intptr_t handle):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclCommDeregister(_comm_handle, <void*>handle)
    check_status(__status__)


cpdef comm_suspend(object comm, int flags):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclCommSuspend(_comm_handle, flags)
    check_status(__status__)


cpdef comm_resume(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclCommResume(_comm_handle)
    check_status(__status__)


cpdef uint64_t comm_mem_stats(object comm, int stat) except? -1:
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef uint64_t value
    with nogil:
        __status__ = ncclCommMemStats(_comm_handle, <_CommMemStat>stat, &value)
    check_status(__status__)
    return value


cpdef object comm_window_register(object comm, intptr_t buff, size_t size, int win_flags):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef Window win_py = Window()
    cdef ncclWindow_t *win = win_py._get_handle_ptr()
    with nogil:
        __status__ = ncclCommWindowRegister(_comm_handle, <void*>buff, size, win, win_flags)
    check_status(__status__)
    return win_py


cpdef comm_window_deregister(object comm, object win):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef ncclWindow_t _win_handle = (<Window?>win)._get_handle()
    with nogil:
        __status__ = ncclCommWindowDeregister(_comm_handle, _win_handle)
    check_status(__status__)


cpdef intptr_t win_get_user_ptr(object comm, object win) except? 0:
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef ncclWindow_t _win_handle = (<Window?>win)._get_handle()
    cdef void* out_user_ptr
    with nogil:
        __status__ = ncclWinGetUserPtr(_comm_handle, _win_handle, &out_user_ptr)
    check_status(__status__)
    return <intptr_t>out_user_ptr


cpdef int red_op_create_pre_mul_sum(intptr_t scalar, int datatype, int residence, object comm) except? -1:
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef _RedOp op
    with nogil:
        __status__ = ncclRedOpCreatePreMulSum(&op, <void*>scalar, <_DataType>datatype, <_ScalarResidence>residence, _comm_handle)
    check_status(__status__)
    return <int>op


cpdef red_op_destroy(int op, object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclRedOpDestroy(<_RedOp>op, _comm_handle)
    check_status(__status__)


cpdef reduce(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, int root, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclReduce(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, <_RedOp>op, root, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef bcast(intptr_t buff, size_t count, int datatype, int root, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclBcast(<void*>buff, count, <_DataType>datatype, root, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef broadcast(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclBroadcast(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, root, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef all_reduce(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclAllReduce(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, <_RedOp>op, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef reduce_scatter(intptr_t sendbuff, intptr_t recvbuff, size_t recvcount, int datatype, int op, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclReduceScatter(<const void*>sendbuff, <void*>recvbuff, recvcount, <_DataType>datatype, <_RedOp>op, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef all_gather(intptr_t sendbuff, intptr_t recvbuff, size_t sendcount, int datatype, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclAllGather(<const void*>sendbuff, <void*>recvbuff, sendcount, <_DataType>datatype, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef allto_all(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclAlltoAll(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef gather(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclGather(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, root, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef scatter(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclScatter(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, root, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef all_reduce_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, object comm, intptr_t stream, intptr_t config):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclAllReduceConfig(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, <_RedOp>op, _comm_handle, <Stream>stream, <const ncclCollConfig_t*>config)
    check_status(__status__)


cpdef broadcast_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream, intptr_t config):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclBroadcastConfig(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, root, _comm_handle, <Stream>stream, <const ncclCollConfig_t*>config)
    check_status(__status__)


cpdef reduce_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, int root, object comm, intptr_t stream, intptr_t config):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclReduceConfig(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, <_RedOp>op, root, _comm_handle, <Stream>stream, <const ncclCollConfig_t*>config)
    check_status(__status__)


cpdef all_gather_config(intptr_t sendbuff, intptr_t recvbuff, size_t sendcount, int datatype, object comm, intptr_t stream, intptr_t config):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclAllGatherConfig(<const void*>sendbuff, <void*>recvbuff, sendcount, <_DataType>datatype, _comm_handle, <Stream>stream, <const ncclCollConfig_t*>config)
    check_status(__status__)


cpdef reduce_scatter_config(intptr_t sendbuff, intptr_t recvbuff, size_t recvcount, int datatype, int op, object comm, intptr_t stream, intptr_t config):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclReduceScatterConfig(<const void*>sendbuff, <void*>recvbuff, recvcount, <_DataType>datatype, <_RedOp>op, _comm_handle, <Stream>stream, <const ncclCollConfig_t*>config)
    check_status(__status__)


cpdef allto_all_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, object comm, intptr_t stream, intptr_t config):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclAlltoAllConfig(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, _comm_handle, <Stream>stream, <const ncclCollConfig_t*>config)
    check_status(__status__)


cpdef gather_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream, intptr_t config):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclGatherConfig(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, root, _comm_handle, <Stream>stream, <const ncclCollConfig_t*>config)
    check_status(__status__)


cpdef scatter_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream, intptr_t config):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclScatterConfig(<const void*>sendbuff, <void*>recvbuff, count, <_DataType>datatype, root, _comm_handle, <Stream>stream, <const ncclCollConfig_t*>config)
    check_status(__status__)


cpdef send(intptr_t sendbuff, size_t count, int datatype, int peer, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclSend(<const void*>sendbuff, count, <_DataType>datatype, peer, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef recv(intptr_t recvbuff, size_t count, int datatype, int peer, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclRecv(<void*>recvbuff, count, <_DataType>datatype, peer, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef put_signal(intptr_t localbuff, size_t count, int datatype, int peer, object peer_win, size_t peer_win_offset, int sig_idx, int ctx, unsigned int flags, object comm, intptr_t stream):
    cdef ncclWindow_t _peer_win_handle = (<Window?>peer_win)._get_handle()
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclPutSignal(<const void*>localbuff, count, <_DataType>datatype, peer, _peer_win_handle, peer_win_offset, sig_idx, ctx, flags, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef signal(int peer, int sig_idx, int ctx, unsigned int flags, object comm, intptr_t stream):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclSignal(peer, sig_idx, ctx, flags, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef wait_signal(int n_desc, signal_descs, object comm, intptr_t stream):
    cdef void* _signal_descs_ = <void *>_cyb_get_buffer_pointer(signal_descs, -1, readonly=False)
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclWaitSignal(n_desc, <ncclWaitSignalDesc_t*>_signal_descs_, _comm_handle, <Stream>stream)
    check_status(__status__)


cpdef group_start():
    with nogil:
        __status__ = ncclGroupStart()
    check_status(__status__)


cpdef group_end():
    with nogil:
        __status__ = ncclGroupEnd()
    check_status(__status__)


cpdef object group_simulate_end():
    cdef SimInfo sim_info_py = SimInfo()
    cdef ncclSimInfo_t *sim_info = <ncclSimInfo_t *><intptr_t>(sim_info_py._get_ptr())
    with nogil:
        __status__ = ncclGroupSimulateEnd(sim_info)
    check_status(__status__)
    return sim_info_py


cpdef object comm_query_properties(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef CommProperties props_py = CommProperties()
    cdef ncclCommProperties_t *props = <ncclCommProperties_t *><intptr_t>(props_py._get_ptr())
    with nogil:
        __status__ = ncclCommQueryProperties(_comm_handle, props)
    check_status(__status__)
    return props_py


cpdef object dev_comm_create(object comm, intptr_t reqs):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef DevComm out_dev_comm_py = DevComm()
    cdef ncclDevComm_t *out_dev_comm = <ncclDevComm_t *><intptr_t>(out_dev_comm_py._get_ptr())
    with nogil:
        __status__ = ncclDevCommCreate(_comm_handle, <const ncclDevCommRequirements_t*>reqs, out_dev_comm)
    check_status(__status__)
    return out_dev_comm_py


cpdef dev_comm_destroy(object comm, intptr_t dev_comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    with nogil:
        __status__ = ncclDevCommDestroy(_comm_handle, <const ncclDevComm_t*>dev_comm)
    check_status(__status__)


cpdef intptr_t get_lsa_multimem_device_pointer(object window, size_t offset) except? 0:
    cdef ncclWindow_t _window_handle = (<Window?>window)._get_handle()
    cdef void* out_ptr
    with nogil:
        __status__ = ncclGetLsaMultimemDevicePointer(_window_handle, offset, &out_ptr)
    check_status(__status__)
    return <intptr_t>out_ptr


cpdef intptr_t get_lsa_device_pointer(object window, size_t offset, int lsa_rank) except? 0:
    cdef ncclWindow_t _window_handle = (<Window?>window)._get_handle()
    cdef void* out_ptr
    with nogil:
        __status__ = ncclGetLsaDevicePointer(_window_handle, offset, lsa_rank, &out_ptr)
    check_status(__status__)
    return <intptr_t>out_ptr


cpdef intptr_t get_multimem_device_pointer(object window, size_t offset, multimem) except? 0:
    cdef ncclWindow_t _window_handle = (<Window?>window)._get_handle()
    cdef intptr_t _multimem_ = multimem
    cdef void* out_ptr
    with nogil:
        __status__ = ncclGetMultimemDevicePointer(_window_handle, offset, (<ncclMultimemHandle_t*>(_multimem_))[0], &out_ptr)
    check_status(__status__)
    return <intptr_t>out_ptr


cpdef intptr_t get_peer_device_pointer(object window, size_t offset, int peer) except? 0:
    cdef ncclWindow_t _window_handle = (<Window?>window)._get_handle()
    cdef void* out_ptr
    with nogil:
        __status__ = ncclGetPeerDevicePointer(_window_handle, offset, peer, &out_ptr)
    check_status(__status__)
    return <intptr_t>out_ptr


cpdef tuple get_multimem_device_le_info(object window, size_t offset):
    cdef ncclWindow_t _window_handle = (<Window?>window)._get_handle()
    cdef ncclCftLeId le_id
    cdef size_t le_offset
    with nogil:
        __status__ = ncclGetMultimemDeviceLeInfo(_window_handle, offset, &le_id, &le_offset)
    check_status(__status__)
    return (<uint32_t>le_id, le_offset)


cpdef tuple get_cft_device_le_info(object window, size_t offset, int peer_cft, cft_team):
    cdef ncclWindow_t _window_handle = (<Window?>window)._get_handle()
    cdef intptr_t _cft_team_ = cft_team
    cdef ncclCftLeId le_id
    cdef size_t le_offset
    with nogil:
        __status__ = ncclGetCftDeviceLeInfo(_window_handle, offset, peer_cft, (<ncclTeam_t*>(_cft_team_))[0], &le_id, &le_offset)
    check_status(__status__)
    return (<uint32_t>le_id, le_offset)


cpdef tuple get_peer_device_le_info(object window, size_t offset, int peer_world):
    cdef ncclWindow_t _window_handle = (<Window?>window)._get_handle()
    cdef ncclCftLeId le_id
    cdef size_t le_offset
    with nogil:
        __status__ = ncclGetPeerDeviceLeInfo(_window_handle, offset, peer_world, &le_id, &le_offset)
    check_status(__status__)
    return (<uint32_t>le_id, le_offset)


cpdef lsa_barrier_create_requirement(team, int n_barriers, intptr_t out_handle, intptr_t out_req):
    cdef intptr_t _team_ = team
    with nogil:
        __status__ = ncclLsaBarrierCreateRequirement((<ncclTeam_t*>(_team_))[0], n_barriers, <ncclLsaBarrierHandle_t*>out_handle, <ncclDevResourceRequirements_t*>out_req)
    check_status(__status__)


cpdef gin_barrier_create_requirement(object comm, team, int n_barriers, intptr_t out_handle, intptr_t out_req):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef intptr_t _team_ = team
    with nogil:
        __status__ = ncclGinBarrierCreateRequirement(_comm_handle, (<ncclTeam_t*>(_team_))[0], n_barriers, <ncclGinBarrierHandle_t*>out_handle, <ncclDevResourceRequirements_t*>out_req)
    check_status(__status__)


cpdef ll_a2a_create_requirement(int n_blocks, int n_slots, intptr_t out_handle, intptr_t out_req):
    with nogil:
        __status__ = ncclLLA2ACreateRequirement(n_blocks, n_slots, <ncclLLA2AHandle_t*>out_handle, <ncclDevResourceRequirements_t*>out_req)
    check_status(__status__)



# Hand-written: cybind cannot emit by-value struct returns (ncclTeam_t).

cpdef object team_world(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef Team team_py = Team()
    cdef ncclTeam_t *team = <ncclTeam_t *><intptr_t>(team_py._get_ptr())
    with nogil:
        team[0] = ncclTeamWorld(_comm_handle)
    return team_py


cpdef object team_lsa(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef Team team_py = Team()
    cdef ncclTeam_t *team = <ncclTeam_t *><intptr_t>(team_py._get_ptr())
    with nogil:
        team[0] = ncclTeamLsa(_comm_handle)
    return team_py


cpdef object team_rail(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef Team team_py = Team()
    cdef ncclTeam_t *team = <ncclTeam_t *><intptr_t>(team_py._get_ptr())
    with nogil:
        team[0] = ncclTeamRail(_comm_handle)
    return team_py


cpdef object team_cft(object comm, int mode):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef Team team_py = Team()
    cdef ncclTeam_t *team = <ncclTeam_t *><intptr_t>(team_py._get_ptr())
    with nogil:
        team[0] = ncclTeamCft(_comm_handle, <ncclCftTeamMode_t>mode)
    return team_py


cpdef object team_cft_multimem(object comm):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef Team team_py = Team()
    cdef ncclTeam_t *team = <ncclTeam_t *><intptr_t>(team_py._get_ptr())
    with nogil:
        team[0] = ncclTeamCftMultimem(_comm_handle)
    return team_py


# Hand-written: the team rank mappers return int (not ncclResult_t), so there is
# no status to check and the generated wrapper would pass a rank to
# check_status(). The by-value ncclTeam_t is not the obstacle.

cpdef int team_rank_to_world(object comm, intptr_t team, int rank):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef int result
    with nogil:
        result = ncclTeamRankToWorld(_comm_handle, (<ncclTeam_t*>team)[0], rank)
    return result


cpdef int team_rank_to_lsa(object comm, intptr_t team, int rank):
    cdef ncclComm_t _comm_handle = (<Comm?>comm)._get_handle()
    cdef int result
    with nogil:
        result = ncclTeamRankToLsa(_comm_handle, (<ncclTeam_t*>team)[0], rank)
    return result


# Hand-written: ncclLLA2ACalcSlots returns int (not ncclResult_t).
cpdef int ll_a2a_calc_slots(int max_elts, int max_elt_size):
    cdef int result
    with nogil:
        result = ncclLLA2ACalcSlots(max_elts, max_elt_size)
    return result


# Hand-written: Param API (SKIP_LOWPP in nccl.cybind.yaml).
# param_get_parameter raises KeyError on unknown key (not NCCLError).
# ncclParamDumpAll returns void — auto-gen wraps only ncclResult_t.

cpdef str param_get_parameter(str key):
    cdef const char* value
    cdef int value_len
    cdef bytes key_bytes = key.encode()
    cdef const char* key_ptr = key_bytes
    with nogil:
        __status__ = ncclParamGetParameter(key_ptr, &value, &value_len)
    if __status__ == Result.InvalidArgument:
        raise KeyError(key)
    check_status(__status__)
    return value[:value_len].decode()


cpdef list param_get_all_keys():
    cdef const char** table
    cdef int table_len
    with nogil:
        __status__ = ncclParamGetAllParameterKeys(&table, &table_len)
    check_status(__status__)
    return [table[i].decode() for i in range(table_len)]


cpdef param_dump_all():
    with nogil:
        ncclParamDumpAll()


# Hand-written: not an NCCL entry point; reports the path of the loaded DSO.
cpdef object get_library_path():
    from ._internal.nccl import _inspect_loaded_library_path
    return _inspect_loaded_library_path()
del _cyb_IntEnum
