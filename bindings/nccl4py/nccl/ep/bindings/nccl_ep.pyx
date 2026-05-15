# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 1.0.0. Do not modify it directly.

cimport cython  # NOQA
from libc.stdint cimport uint64_t
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

cdef _get_ep_alloc_config_dtype_offsets():
    cdef ncclEpAllocConfig_t pod = ncclEpAllocConfig_t()
    return _numpy.dtype({
        'names': ['alloc_fn', 'free_fn', 'context'],
        'formats': [_numpy.dtype(('V', sizeof(ncclEpAllocFn_t))), _numpy.dtype(('V', sizeof(ncclEpFreeFn_t))), _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.alloc_fn)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.free_fn)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.context)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpAllocConfig_t),
    })

ep_alloc_config_dtype = _get_ep_alloc_config_dtype_offsets()

cdef class EpAllocConfig:
    """Empty-initialize an instance of `ncclEpAllocConfig_t`.


    .. seealso:: `ncclEpAllocConfig_t`
    """
    cdef:
        ncclEpAllocConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpAllocConfig_t *>calloc(1, sizeof(ncclEpAllocConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating EpAllocConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpAllocConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.EpAllocConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef EpAllocConfig other_
        if not isinstance(other, EpAllocConfig):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpAllocConfig_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpAllocConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclEpAllocConfig_t *>malloc(sizeof(ncclEpAllocConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating EpAllocConfig")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpAllocConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def alloc_fn(self):
        return <intptr_t>(self._ptr[0].alloc_fn)

    @alloc_fn.setter
    def alloc_fn(self, val):
        if self._readonly:
            raise ValueError("This EpAllocConfig instance is read-only")
        self._ptr[0].alloc_fn = <ncclEpAllocFn_t><intptr_t>val

    @property
    def free_fn(self):
        return <intptr_t>(self._ptr[0].free_fn)

    @free_fn.setter
    def free_fn(self, val):
        if self._readonly:
            raise ValueError("This EpAllocConfig instance is read-only")
        self._ptr[0].free_fn = <ncclEpFreeFn_t><intptr_t>val

    @property
    def context(self):
        """int: """
        return <intptr_t>(self._ptr[0].context)

    @context.setter
    def context(self, val):
        if self._readonly:
            raise ValueError("This EpAllocConfig instance is read-only")
        self._ptr[0].context = <void *><intptr_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for EpAllocConfig")

    @staticmethod
    def from_buffer(buffer):
        """Create an EpAllocConfig instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclEpAllocConfig_t), EpAllocConfig)

    @staticmethod
    def from_data(data):
        """Create an EpAllocConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ep_alloc_config_dtype` holding the data.
        """
        return __from_data(data, "ep_alloc_config_dtype", ep_alloc_config_dtype, EpAllocConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an EpAllocConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef EpAllocConfig obj = EpAllocConfig.__new__(EpAllocConfig)
        if owner is None:
            obj._ptr = <ncclEpAllocConfig_t *>malloc(sizeof(ncclEpAllocConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating EpAllocConfig")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpAllocConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpAllocConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_ep_layout_info_dtype_offsets():
    cdef ncclEpLayoutInfo_t pod = ncclEpLayoutInfo_t()
    return _numpy.dtype({
        'names': ['size_', 'expert_counters', 'src_rank_counters', 'expert_offsets', 'recv_total_counter'],
        'formats': [_numpy.uint32, _numpy.intp, _numpy.intp, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.expert_counters)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.src_rank_counters)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.expert_offsets)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.recv_total_counter)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpLayoutInfo_t),
    })

ep_layout_info_dtype = _get_ep_layout_info_dtype_offsets()

cdef class EpLayoutInfo:
    """Empty-initialize an instance of `ncclEpLayoutInfo_t`.


    .. seealso:: `ncclEpLayoutInfo_t`
    """
    cdef:
        ncclEpLayoutInfo_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpLayoutInfo_t *>calloc(1, sizeof(ncclEpLayoutInfo_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating EpLayoutInfo")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpLayoutInfo_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.EpLayoutInfo object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef EpLayoutInfo other_
        if not isinstance(other, EpLayoutInfo):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpLayoutInfo_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpLayoutInfo_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclEpLayoutInfo_t *>malloc(sizeof(ncclEpLayoutInfo_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating EpLayoutInfo")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpLayoutInfo_t))
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
            raise ValueError("This EpLayoutInfo instance is read-only")
        self._ptr[0].size = val

    @property
    def expert_counters(self):
        """int: """
        return <intptr_t>(self._ptr[0].expert_counters)

    @expert_counters.setter
    def expert_counters(self, val):
        if self._readonly:
            raise ValueError("This EpLayoutInfo instance is read-only")
        self._ptr[0].expert_counters = <ncclNDTensor_t><intptr_t>val

    @property
    def src_rank_counters(self):
        """int: """
        return <intptr_t>(self._ptr[0].src_rank_counters)

    @src_rank_counters.setter
    def src_rank_counters(self, val):
        if self._readonly:
            raise ValueError("This EpLayoutInfo instance is read-only")
        self._ptr[0].src_rank_counters = <ncclNDTensor_t><intptr_t>val

    @property
    def expert_offsets(self):
        """int: """
        return <intptr_t>(self._ptr[0].expert_offsets)

    @expert_offsets.setter
    def expert_offsets(self, val):
        if self._readonly:
            raise ValueError("This EpLayoutInfo instance is read-only")
        self._ptr[0].expert_offsets = <ncclNDTensor_t><intptr_t>val

    @property
    def recv_total_counter(self):
        """int: """
        return <intptr_t>(self._ptr[0].recv_total_counter)

    @recv_total_counter.setter
    def recv_total_counter(self, val):
        if self._readonly:
            raise ValueError("This EpLayoutInfo instance is read-only")
        self._ptr[0].recv_total_counter = <ncclNDTensor_t><intptr_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for EpLayoutInfo")

    @staticmethod
    def from_buffer(buffer):
        """Create an EpLayoutInfo instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclEpLayoutInfo_t), EpLayoutInfo)

    @staticmethod
    def from_data(data):
        """Create an EpLayoutInfo instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ep_layout_info_dtype` holding the data.
        """
        return __from_data(data, "ep_layout_info_dtype", ep_layout_info_dtype, EpLayoutInfo)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an EpLayoutInfo instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef EpLayoutInfo obj = EpLayoutInfo.__new__(EpLayoutInfo)
        if owner is None:
            obj._ptr = <ncclEpLayoutInfo_t *>malloc(sizeof(ncclEpLayoutInfo_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating EpLayoutInfo")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpLayoutInfo_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpLayoutInfo_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_ep_dispatch_inputs_dtype_offsets():
    cdef ncclEpDispatchInputs_t pod = ncclEpDispatchInputs_t()
    return _numpy.dtype({
        'names': ['size_', 'tokens', 'topk_weights', 'scales'],
        'formats': [_numpy.uint32, _numpy.intp, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.tokens)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_weights)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.scales)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpDispatchInputs_t),
    })

ep_dispatch_inputs_dtype = _get_ep_dispatch_inputs_dtype_offsets()

cdef class EpDispatchInputs:
    """Empty-initialize an instance of `ncclEpDispatchInputs_t`.


    .. seealso:: `ncclEpDispatchInputs_t`
    """
    cdef:
        ncclEpDispatchInputs_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpDispatchInputs_t *>calloc(1, sizeof(ncclEpDispatchInputs_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating EpDispatchInputs")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpDispatchInputs_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.EpDispatchInputs object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef EpDispatchInputs other_
        if not isinstance(other, EpDispatchInputs):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpDispatchInputs_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpDispatchInputs_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclEpDispatchInputs_t *>malloc(sizeof(ncclEpDispatchInputs_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating EpDispatchInputs")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpDispatchInputs_t))
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
            raise ValueError("This EpDispatchInputs instance is read-only")
        self._ptr[0].size = val

    @property
    def tokens(self):
        """int: """
        return <intptr_t>(self._ptr[0].tokens)

    @tokens.setter
    def tokens(self, val):
        if self._readonly:
            raise ValueError("This EpDispatchInputs instance is read-only")
        self._ptr[0].tokens = <ncclNDTensor_t><intptr_t>val

    @property
    def topk_weights(self):
        """int: """
        return <intptr_t>(self._ptr[0].topk_weights)

    @topk_weights.setter
    def topk_weights(self, val):
        if self._readonly:
            raise ValueError("This EpDispatchInputs instance is read-only")
        self._ptr[0].topk_weights = <ncclNDTensor_t><intptr_t>val

    @property
    def scales(self):
        """int: """
        return <intptr_t>(self._ptr[0].scales)

    @scales.setter
    def scales(self, val):
        if self._readonly:
            raise ValueError("This EpDispatchInputs instance is read-only")
        self._ptr[0].scales = <ncclNDTensor_t><intptr_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for EpDispatchInputs")

    @staticmethod
    def from_buffer(buffer):
        """Create an EpDispatchInputs instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclEpDispatchInputs_t), EpDispatchInputs)

    @staticmethod
    def from_data(data):
        """Create an EpDispatchInputs instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ep_dispatch_inputs_dtype` holding the data.
        """
        return __from_data(data, "ep_dispatch_inputs_dtype", ep_dispatch_inputs_dtype, EpDispatchInputs)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an EpDispatchInputs instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef EpDispatchInputs obj = EpDispatchInputs.__new__(EpDispatchInputs)
        if owner is None:
            obj._ptr = <ncclEpDispatchInputs_t *>malloc(sizeof(ncclEpDispatchInputs_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating EpDispatchInputs")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpDispatchInputs_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpDispatchInputs_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_ep_dispatch_outputs_dtype_offsets():
    cdef ncclEpDispatchOutputs_t pod = ncclEpDispatchOutputs_t()
    return _numpy.dtype({
        'names': ['size_', 'tokens', 'topk_weights', 'scales', 'topk_idx'],
        'formats': [_numpy.uint32, _numpy.intp, _numpy.intp, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.tokens)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_weights)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.scales)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_idx)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpDispatchOutputs_t),
    })

ep_dispatch_outputs_dtype = _get_ep_dispatch_outputs_dtype_offsets()

cdef class EpDispatchOutputs:
    """Empty-initialize an instance of `ncclEpDispatchOutputs_t`.


    .. seealso:: `ncclEpDispatchOutputs_t`
    """
    cdef:
        ncclEpDispatchOutputs_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpDispatchOutputs_t *>calloc(1, sizeof(ncclEpDispatchOutputs_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating EpDispatchOutputs")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpDispatchOutputs_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.EpDispatchOutputs object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef EpDispatchOutputs other_
        if not isinstance(other, EpDispatchOutputs):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpDispatchOutputs_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpDispatchOutputs_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclEpDispatchOutputs_t *>malloc(sizeof(ncclEpDispatchOutputs_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating EpDispatchOutputs")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpDispatchOutputs_t))
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
            raise ValueError("This EpDispatchOutputs instance is read-only")
        self._ptr[0].size = val

    @property
    def tokens(self):
        """int: """
        return <intptr_t>(self._ptr[0].tokens)

    @tokens.setter
    def tokens(self, val):
        if self._readonly:
            raise ValueError("This EpDispatchOutputs instance is read-only")
        self._ptr[0].tokens = <ncclNDTensor_t><intptr_t>val

    @property
    def topk_weights(self):
        """int: """
        return <intptr_t>(self._ptr[0].topk_weights)

    @topk_weights.setter
    def topk_weights(self, val):
        if self._readonly:
            raise ValueError("This EpDispatchOutputs instance is read-only")
        self._ptr[0].topk_weights = <ncclNDTensor_t><intptr_t>val

    @property
    def scales(self):
        """int: """
        return <intptr_t>(self._ptr[0].scales)

    @scales.setter
    def scales(self, val):
        if self._readonly:
            raise ValueError("This EpDispatchOutputs instance is read-only")
        self._ptr[0].scales = <ncclNDTensor_t><intptr_t>val

    @property
    def topk_idx(self):
        """int: """
        return <intptr_t>(self._ptr[0].topk_idx)

    @topk_idx.setter
    def topk_idx(self, val):
        if self._readonly:
            raise ValueError("This EpDispatchOutputs instance is read-only")
        self._ptr[0].topk_idx = <ncclNDTensor_t><intptr_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for EpDispatchOutputs")

    @staticmethod
    def from_buffer(buffer):
        """Create an EpDispatchOutputs instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclEpDispatchOutputs_t), EpDispatchOutputs)

    @staticmethod
    def from_data(data):
        """Create an EpDispatchOutputs instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ep_dispatch_outputs_dtype` holding the data.
        """
        return __from_data(data, "ep_dispatch_outputs_dtype", ep_dispatch_outputs_dtype, EpDispatchOutputs)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an EpDispatchOutputs instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef EpDispatchOutputs obj = EpDispatchOutputs.__new__(EpDispatchOutputs)
        if owner is None:
            obj._ptr = <ncclEpDispatchOutputs_t *>malloc(sizeof(ncclEpDispatchOutputs_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating EpDispatchOutputs")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpDispatchOutputs_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpDispatchOutputs_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_ep_combine_inputs_dtype_offsets():
    cdef ncclEpCombineInputs_t pod = ncclEpCombineInputs_t()
    return _numpy.dtype({
        'names': ['size_', 'tokens', 'topk_weights'],
        'formats': [_numpy.uint32, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.tokens)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_weights)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpCombineInputs_t),
    })

ep_combine_inputs_dtype = _get_ep_combine_inputs_dtype_offsets()

cdef class EpCombineInputs:
    """Empty-initialize an instance of `ncclEpCombineInputs_t`.


    .. seealso:: `ncclEpCombineInputs_t`
    """
    cdef:
        ncclEpCombineInputs_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpCombineInputs_t *>calloc(1, sizeof(ncclEpCombineInputs_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating EpCombineInputs")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpCombineInputs_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.EpCombineInputs object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef EpCombineInputs other_
        if not isinstance(other, EpCombineInputs):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpCombineInputs_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpCombineInputs_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclEpCombineInputs_t *>malloc(sizeof(ncclEpCombineInputs_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating EpCombineInputs")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpCombineInputs_t))
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
            raise ValueError("This EpCombineInputs instance is read-only")
        self._ptr[0].size = val

    @property
    def tokens(self):
        """int: """
        return <intptr_t>(self._ptr[0].tokens)

    @tokens.setter
    def tokens(self, val):
        if self._readonly:
            raise ValueError("This EpCombineInputs instance is read-only")
        self._ptr[0].tokens = <ncclNDTensor_t><intptr_t>val

    @property
    def topk_weights(self):
        """int: """
        return <intptr_t>(self._ptr[0].topk_weights)

    @topk_weights.setter
    def topk_weights(self, val):
        if self._readonly:
            raise ValueError("This EpCombineInputs instance is read-only")
        self._ptr[0].topk_weights = <ncclNDTensor_t><intptr_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for EpCombineInputs")

    @staticmethod
    def from_buffer(buffer):
        """Create an EpCombineInputs instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclEpCombineInputs_t), EpCombineInputs)

    @staticmethod
    def from_data(data):
        """Create an EpCombineInputs instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ep_combine_inputs_dtype` holding the data.
        """
        return __from_data(data, "ep_combine_inputs_dtype", ep_combine_inputs_dtype, EpCombineInputs)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an EpCombineInputs instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef EpCombineInputs obj = EpCombineInputs.__new__(EpCombineInputs)
        if owner is None:
            obj._ptr = <ncclEpCombineInputs_t *>malloc(sizeof(ncclEpCombineInputs_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating EpCombineInputs")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpCombineInputs_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpCombineInputs_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_ep_combine_outputs_dtype_offsets():
    cdef ncclEpCombineOutputs_t pod = ncclEpCombineOutputs_t()
    return _numpy.dtype({
        'names': ['size_', 'tokens', 'topk_weights'],
        'formats': [_numpy.uint32, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.tokens)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.topk_weights)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpCombineOutputs_t),
    })

ep_combine_outputs_dtype = _get_ep_combine_outputs_dtype_offsets()

cdef class EpCombineOutputs:
    """Empty-initialize an instance of `ncclEpCombineOutputs_t`.


    .. seealso:: `ncclEpCombineOutputs_t`
    """
    cdef:
        ncclEpCombineOutputs_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpCombineOutputs_t *>calloc(1, sizeof(ncclEpCombineOutputs_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating EpCombineOutputs")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpCombineOutputs_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.EpCombineOutputs object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef EpCombineOutputs other_
        if not isinstance(other, EpCombineOutputs):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpCombineOutputs_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpCombineOutputs_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclEpCombineOutputs_t *>malloc(sizeof(ncclEpCombineOutputs_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating EpCombineOutputs")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpCombineOutputs_t))
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
            raise ValueError("This EpCombineOutputs instance is read-only")
        self._ptr[0].size = val

    @property
    def tokens(self):
        """int: """
        return <intptr_t>(self._ptr[0].tokens)

    @tokens.setter
    def tokens(self, val):
        if self._readonly:
            raise ValueError("This EpCombineOutputs instance is read-only")
        self._ptr[0].tokens = <ncclNDTensor_t><intptr_t>val

    @property
    def topk_weights(self):
        """int: """
        return <intptr_t>(self._ptr[0].topk_weights)

    @topk_weights.setter
    def topk_weights(self, val):
        if self._readonly:
            raise ValueError("This EpCombineOutputs instance is read-only")
        self._ptr[0].topk_weights = <ncclNDTensor_t><intptr_t>val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for EpCombineOutputs")

    @staticmethod
    def from_buffer(buffer):
        """Create an EpCombineOutputs instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclEpCombineOutputs_t), EpCombineOutputs)

    @staticmethod
    def from_data(data):
        """Create an EpCombineOutputs instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ep_combine_outputs_dtype` holding the data.
        """
        return __from_data(data, "ep_combine_outputs_dtype", ep_combine_outputs_dtype, EpCombineOutputs)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an EpCombineOutputs instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef EpCombineOutputs obj = EpCombineOutputs.__new__(EpCombineOutputs)
        if owner is None:
            obj._ptr = <ncclEpCombineOutputs_t *>malloc(sizeof(ncclEpCombineOutputs_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating EpCombineOutputs")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpCombineOutputs_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpCombineOutputs_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_ep_handle_config_dtype_offsets():
    cdef ncclEpHandleConfig_t pod = ncclEpHandleConfig_t()
    return _numpy.dtype({
        'names': ['size_', 'use_fp8', 'dispatch_output_per_expert_alignment'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.uint64],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.use_fp8)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.dispatch_output_per_expert_alignment)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpHandleConfig_t),
    })

ep_handle_config_dtype = _get_ep_handle_config_dtype_offsets()

cdef class EpHandleConfig:
    """Empty-initialize an instance of `ncclEpHandleConfig_t`.


    .. seealso:: `ncclEpHandleConfig_t`
    """
    cdef:
        ncclEpHandleConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpHandleConfig_t *>calloc(1, sizeof(ncclEpHandleConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating EpHandleConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpHandleConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.EpHandleConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef EpHandleConfig other_
        if not isinstance(other, EpHandleConfig):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpHandleConfig_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpHandleConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclEpHandleConfig_t *>malloc(sizeof(ncclEpHandleConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating EpHandleConfig")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpHandleConfig_t))
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
            raise ValueError("This EpHandleConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def use_fp8(self):
        """int: """
        return self._ptr[0].use_fp8

    @use_fp8.setter
    def use_fp8(self, val):
        if self._readonly:
            raise ValueError("This EpHandleConfig instance is read-only")
        self._ptr[0].use_fp8 = val

    @property
    def dispatch_output_per_expert_alignment(self):
        """int: """
        return self._ptr[0].dispatch_output_per_expert_alignment

    @dispatch_output_per_expert_alignment.setter
    def dispatch_output_per_expert_alignment(self, val):
        if self._readonly:
            raise ValueError("This EpHandleConfig instance is read-only")
        self._ptr[0].dispatch_output_per_expert_alignment = val

    def __getstate__(self):
        return cpython.PyBytes_FromStringAndSize(<char *><void *>self._ptr, sizeof(ncclEpHandleConfig_t))

    def __setstate__(self, state):
        if not isinstance(state, bytes):
            raise TypeError(f"Invalid state type for EpHandleConfig, expected bytes, got {type(state).__name__}")
        if len(state) != sizeof(ncclEpHandleConfig_t):
            raise ValueError(f"Invalid state length for EpHandleConfig, expected sizeof(ncclEpHandleConfig_t), got {len(state)}")
        cdef char *state_ptr = cpython.PyBytes_AsString(state)
        self._ptr = <ncclEpHandleConfig_t *>malloc(sizeof(ncclEpHandleConfig_t))
        memcpy(<void *>self._ptr, <void *>state_ptr, sizeof(ncclEpHandleConfig_t))

    @staticmethod
    def from_buffer(buffer):
        """Create an EpHandleConfig instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclEpHandleConfig_t), EpHandleConfig)

    @staticmethod
    def from_data(data):
        """Create an EpHandleConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ep_handle_config_dtype` holding the data.
        """
        return __from_data(data, "ep_handle_config_dtype", ep_handle_config_dtype, EpHandleConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an EpHandleConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef EpHandleConfig obj = EpHandleConfig.__new__(EpHandleConfig)
        if owner is None:
            obj._ptr = <ncclEpHandleConfig_t *>malloc(sizeof(ncclEpHandleConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating EpHandleConfig")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpHandleConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpHandleConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_ep_dispatch_config_dtype_offsets():
    cdef ncclEpDispatchConfig_t pod = ncclEpDispatchConfig_t()
    return _numpy.dtype({
        'names': ['size_', 'send_only', 'round_scales'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.uint32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.send_only)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.round_scales)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpDispatchConfig_t),
    })

ep_dispatch_config_dtype = _get_ep_dispatch_config_dtype_offsets()

cdef class EpDispatchConfig:
    """Empty-initialize an instance of `ncclEpDispatchConfig_t`.


    .. seealso:: `ncclEpDispatchConfig_t`
    """
    cdef:
        ncclEpDispatchConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpDispatchConfig_t *>calloc(1, sizeof(ncclEpDispatchConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating EpDispatchConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpDispatchConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.EpDispatchConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef EpDispatchConfig other_
        if not isinstance(other, EpDispatchConfig):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpDispatchConfig_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpDispatchConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclEpDispatchConfig_t *>malloc(sizeof(ncclEpDispatchConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating EpDispatchConfig")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpDispatchConfig_t))
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
            raise ValueError("This EpDispatchConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def send_only(self):
        """int: """
        return self._ptr[0].send_only

    @send_only.setter
    def send_only(self, val):
        if self._readonly:
            raise ValueError("This EpDispatchConfig instance is read-only")
        self._ptr[0].send_only = val

    @property
    def round_scales(self):
        """int: """
        return self._ptr[0].round_scales

    @round_scales.setter
    def round_scales(self, val):
        if self._readonly:
            raise ValueError("This EpDispatchConfig instance is read-only")
        self._ptr[0].round_scales = val

    def __getstate__(self):
        return cpython.PyBytes_FromStringAndSize(<char *><void *>self._ptr, sizeof(ncclEpDispatchConfig_t))

    def __setstate__(self, state):
        if not isinstance(state, bytes):
            raise TypeError(f"Invalid state type for EpDispatchConfig, expected bytes, got {type(state).__name__}")
        if len(state) != sizeof(ncclEpDispatchConfig_t):
            raise ValueError(f"Invalid state length for EpDispatchConfig, expected sizeof(ncclEpDispatchConfig_t), got {len(state)}")
        cdef char *state_ptr = cpython.PyBytes_AsString(state)
        self._ptr = <ncclEpDispatchConfig_t *>malloc(sizeof(ncclEpDispatchConfig_t))
        memcpy(<void *>self._ptr, <void *>state_ptr, sizeof(ncclEpDispatchConfig_t))

    @staticmethod
    def from_buffer(buffer):
        """Create an EpDispatchConfig instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclEpDispatchConfig_t), EpDispatchConfig)

    @staticmethod
    def from_data(data):
        """Create an EpDispatchConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ep_dispatch_config_dtype` holding the data.
        """
        return __from_data(data, "ep_dispatch_config_dtype", ep_dispatch_config_dtype, EpDispatchConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an EpDispatchConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef EpDispatchConfig obj = EpDispatchConfig.__new__(EpDispatchConfig)
        if owner is None:
            obj._ptr = <ncclEpDispatchConfig_t *>malloc(sizeof(ncclEpDispatchConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating EpDispatchConfig")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpDispatchConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpDispatchConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_ep_combine_config_dtype_offsets():
    cdef ncclEpCombineConfig_t pod = ncclEpCombineConfig_t()
    return _numpy.dtype({
        'names': ['size_', 'send_only'],
        'formats': [_numpy.uint32, _numpy.uint32],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.send_only)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpCombineConfig_t),
    })

ep_combine_config_dtype = _get_ep_combine_config_dtype_offsets()

cdef class EpCombineConfig:
    """Empty-initialize an instance of `ncclEpCombineConfig_t`.


    .. seealso:: `ncclEpCombineConfig_t`
    """
    cdef:
        ncclEpCombineConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpCombineConfig_t *>calloc(1, sizeof(ncclEpCombineConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating EpCombineConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpCombineConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.EpCombineConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef EpCombineConfig other_
        if not isinstance(other, EpCombineConfig):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpCombineConfig_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpCombineConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclEpCombineConfig_t *>malloc(sizeof(ncclEpCombineConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating EpCombineConfig")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpCombineConfig_t))
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
            raise ValueError("This EpCombineConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def send_only(self):
        """int: """
        return self._ptr[0].send_only

    @send_only.setter
    def send_only(self, val):
        if self._readonly:
            raise ValueError("This EpCombineConfig instance is read-only")
        self._ptr[0].send_only = val

    def __getstate__(self):
        return cpython.PyBytes_FromStringAndSize(<char *><void *>self._ptr, sizeof(ncclEpCombineConfig_t))

    def __setstate__(self, state):
        if not isinstance(state, bytes):
            raise TypeError(f"Invalid state type for EpCombineConfig, expected bytes, got {type(state).__name__}")
        if len(state) != sizeof(ncclEpCombineConfig_t):
            raise ValueError(f"Invalid state length for EpCombineConfig, expected sizeof(ncclEpCombineConfig_t), got {len(state)}")
        cdef char *state_ptr = cpython.PyBytes_AsString(state)
        self._ptr = <ncclEpCombineConfig_t *>malloc(sizeof(ncclEpCombineConfig_t))
        memcpy(<void *>self._ptr, <void *>state_ptr, sizeof(ncclEpCombineConfig_t))

    @staticmethod
    def from_buffer(buffer):
        """Create an EpCombineConfig instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclEpCombineConfig_t), EpCombineConfig)

    @staticmethod
    def from_data(data):
        """Create an EpCombineConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ep_combine_config_dtype` holding the data.
        """
        return __from_data(data, "ep_combine_config_dtype", ep_combine_config_dtype, EpCombineConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an EpCombineConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef EpCombineConfig obj = EpCombineConfig.__new__(EpCombineConfig)
        if owner is None:
            obj._ptr = <ncclEpCombineConfig_t *>malloc(sizeof(ncclEpCombineConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating EpCombineConfig")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpCombineConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpCombineConfig_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_ep_group_config_dtype_offsets():
    cdef ncclEpGroupConfig_t pod = ncclEpGroupConfig_t()
    return _numpy.dtype({
        'names': ['size_', 'version', 'algorithm', 'layout', 'num_experts', 'max_send_tokens_per_rank', 'max_token_bytes', 'rdma_buffer_size', 'num_qp_per_rank', 'num_channels', 'max_recv_token_slots_per_rank', 'max_num_sms', 'alloc', 'enable_mask', 'timeout_ns'],
        'formats': [_numpy.uint32, _numpy.uint32, _numpy.int32, _numpy.int32, _numpy.uint32, _numpy.uint32, _numpy.uint32, _numpy.dtype(('V', sizeof(unsigned long int))), _numpy.uint32, _numpy.uint32, _numpy.uint32, _numpy.uint32, ep_alloc_config_dtype, _numpy.uint32, _numpy.uint64],
        'offsets': [
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.version)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.algorithm)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.layout)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_experts)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.max_send_tokens_per_rank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.max_token_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.rdma_buffer_size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_qp_per_rank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_channels)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.max_recv_token_slots_per_rank)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.max_num_sms)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.alloc)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.enable_mask)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.timeout_ns)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(ncclEpGroupConfig_t),
    })

ep_group_config_dtype = _get_ep_group_config_dtype_offsets()

cdef class EpGroupConfig:
    """Empty-initialize an instance of `ncclEpGroupConfig_t`.


    .. seealso:: `ncclEpGroupConfig_t`
    """
    cdef:
        ncclEpGroupConfig_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <ncclEpGroupConfig_t *>calloc(1, sizeof(ncclEpGroupConfig_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating EpGroupConfig")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef ncclEpGroupConfig_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.EpGroupConfig object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef EpGroupConfig other_
        if not isinstance(other, EpGroupConfig):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(ncclEpGroupConfig_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(ncclEpGroupConfig_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            if self._ptr != NULL and self._owned:
                free(self._ptr)
            self._ptr = <ncclEpGroupConfig_t *>malloc(sizeof(ncclEpGroupConfig_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating EpGroupConfig")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(ncclEpGroupConfig_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def alloc(self):
        """EpAllocConfig: """
        return EpAllocConfig.from_ptr(<intptr_t>&(self._ptr[0].alloc), self._readonly, self)

    @alloc.setter
    def alloc(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        cdef EpAllocConfig val_ = val
        memcpy(<void *>&(self._ptr[0].alloc), <void *>(val_._get_ptr()), sizeof(ncclEpAllocConfig_t) * 1)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].size = val

    @property
    def version(self):
        """int: """
        return self._ptr[0].version

    @version.setter
    def version(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].version = val

    @property
    def algorithm(self):
        """int: """
        return <int>(self._ptr[0].algorithm)

    @algorithm.setter
    def algorithm(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].algorithm = <ncclEpAlgorithm_t><int>val

    @property
    def layout(self):
        """int: """
        return <int>(self._ptr[0].layout)

    @layout.setter
    def layout(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].layout = <ncclEpLayout_t><int>val

    @property
    def num_experts(self):
        """int: """
        return self._ptr[0].num_experts

    @num_experts.setter
    def num_experts(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].num_experts = val

    @property
    def max_send_tokens_per_rank(self):
        """int: """
        return self._ptr[0].max_send_tokens_per_rank

    @max_send_tokens_per_rank.setter
    def max_send_tokens_per_rank(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].max_send_tokens_per_rank = val

    @property
    def max_token_bytes(self):
        """int: """
        return self._ptr[0].max_token_bytes

    @max_token_bytes.setter
    def max_token_bytes(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].max_token_bytes = val

    @property
    def rdma_buffer_size(self):
        """: """
        return self._ptr[0].rdma_buffer_size

    @rdma_buffer_size.setter
    def rdma_buffer_size(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].rdma_buffer_size = val

    @property
    def num_qp_per_rank(self):
        """int: """
        return self._ptr[0].num_qp_per_rank

    @num_qp_per_rank.setter
    def num_qp_per_rank(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].num_qp_per_rank = val

    @property
    def num_channels(self):
        """int: """
        return self._ptr[0].num_channels

    @num_channels.setter
    def num_channels(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].num_channels = val

    @property
    def max_recv_token_slots_per_rank(self):
        """int: """
        return self._ptr[0].max_recv_token_slots_per_rank

    @max_recv_token_slots_per_rank.setter
    def max_recv_token_slots_per_rank(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].max_recv_token_slots_per_rank = val

    @property
    def max_num_sms(self):
        """int: """
        return self._ptr[0].max_num_sms

    @max_num_sms.setter
    def max_num_sms(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].max_num_sms = val

    @property
    def enable_mask(self):
        """int: """
        return self._ptr[0].enable_mask

    @enable_mask.setter
    def enable_mask(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].enable_mask = val

    @property
    def timeout_ns(self):
        """int: """
        return self._ptr[0].timeout_ns

    @timeout_ns.setter
    def timeout_ns(self, val):
        if self._readonly:
            raise ValueError("This EpGroupConfig instance is read-only")
        self._ptr[0].timeout_ns = val

    def __getstate__(self):
        raise pickle.PicklingError("Pickle not supported for EpGroupConfig")

    @staticmethod
    def from_buffer(buffer):
        """Create an EpGroupConfig instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(ncclEpGroupConfig_t), EpGroupConfig)

    @staticmethod
    def from_data(data):
        """Create an EpGroupConfig instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `ep_group_config_dtype` holding the data.
        """
        return __from_data(data, "ep_group_config_dtype", ep_group_config_dtype, EpGroupConfig)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an EpGroupConfig instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef EpGroupConfig obj = EpGroupConfig.__new__(EpGroupConfig)
        if owner is None:
            obj._ptr = <ncclEpGroupConfig_t *>malloc(sizeof(ncclEpGroupConfig_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating EpGroupConfig")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(ncclEpGroupConfig_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <ncclEpGroupConfig_t *>ptr
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
    Timeout = ncclTimeout
    NumResults = ncclNumResults

class CommMemStat(_IntEnum):
    """
    See `ncclCommMemStat_t`.
    """
    StatGpuMemSuspend = ncclStatGpuMemSuspend
    StatGpuMemSuspended = ncclStatGpuMemSuspended
    StatGpuMemPersist = ncclStatGpuMemPersist
    StatGpuMemTotal = ncclStatGpuMemTotal

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
    ScalarDevice = ncclScalarDevice
    ScalarHostImmediate = ncclScalarHostImmediate

class EpAlgorithm(_IntEnum):
    """
    See `ncclEpAlgorithm_t`.
    """
    LOW_LATENCY = NCCL_EP_ALGO_LOW_LATENCY
    HIGH_THROUGHPUT = NCCL_EP_ALGO_HIGH_THROUGHPUT

class EpLayout(_IntEnum):
    """
    See `ncclEpLayout_t`.
    """
    AUTO = NCCL_EP_LAYOUT_AUTO
    EXPERT_MAJOR = NCCL_EP_LAYOUT_EXPERT_MAJOR
    RANK_MAJOR = NCCL_EP_LAYOUT_RANK_MAJOR
    FLAT = NCCL_EP_LAYOUT_FLAT


###############################################################################
# Error handling
###############################################################################

class NCCLEpError(Exception):

    def __init__(self, status):
        self.status = status
        cdef str err = f"NCCL EP error code {status}"
        super(NCCLEpError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise NCCLEpError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t ep_create_group(intptr_t comm, intptr_t config) except? 0:
    cdef EpGroup ep_group
    with nogil:
        __status__ = ncclEpCreateGroup(&ep_group, <Comm>comm, <const ncclEpGroupConfig_t*>config)
    check_status(__status__)
    return <intptr_t>ep_group


cpdef ep_group_destroy(intptr_t ep_group):
    with nogil:
        __status__ = ncclEpGroupDestroy(<EpGroup>ep_group)
    check_status(__status__)


cpdef intptr_t ep_tensor_create(unsigned int ndim, int datatype, intptr_t data, intptr_t sizes) except? 0:
    cdef NDTensor tensor
    with nogil:
        __status__ = ncclEpTensorCreate(&tensor, ndim, <_DataType>datatype, <void*>data, <const size_t*>sizes)
    check_status(__status__)
    return <intptr_t>tensor


cpdef intptr_t ep_tensor_create_from_window(unsigned int ndim, int datatype, intptr_t win, uint64_t win_offset, intptr_t sizes) except? 0:
    cdef NDTensor tensor
    with nogil:
        __status__ = ncclEpTensorCreateFromWindow(&tensor, ndim, <_DataType>datatype, <Window>win, win_offset, <const size_t*>sizes)
    check_status(__status__)
    return <intptr_t>tensor


cpdef ep_tensor_destroy(intptr_t tensor):
    with nogil:
        __status__ = ncclEpTensorDestroy(<NDTensor>tensor)
    check_status(__status__)


cpdef intptr_t ep_create_handle(intptr_t ep_group, intptr_t topk_idx, intptr_t layout_info, intptr_t config, intptr_t stream) except? 0:
    cdef EpHandle handle
    with nogil:
        __status__ = ncclEpCreateHandle(&handle, <EpGroup>ep_group, <NDTensor>topk_idx, <const ncclEpLayoutInfo_t*>layout_info, <const ncclEpHandleConfig_t*>config, <Stream>stream)
    check_status(__status__)
    return <intptr_t>handle


cpdef ep_handle_destroy(intptr_t handle):
    with nogil:
        __status__ = ncclEpHandleDestroy(<EpHandle>handle)
    check_status(__status__)


cpdef size_t ep_handle_mem_size(intptr_t ep_group, intptr_t config, int num_topk) except? -1:
    cdef size_t size_out
    with nogil:
        __status__ = ncclEpHandleMemSize(<EpGroup>ep_group, <const ncclEpHandleConfig_t*>config, &size_out, num_topk)
    check_status(__status__)
    return size_out


cpdef intptr_t ep_init_handle(intptr_t ep_group, intptr_t config, int num_topk, intptr_t handle_mem) except? 0:
    cdef EpHandle handle
    with nogil:
        __status__ = ncclEpInitHandle(&handle, <EpGroup>ep_group, <const ncclEpHandleConfig_t*>config, num_topk, <NDTensor>handle_mem)
    check_status(__status__)
    return <intptr_t>handle


cpdef ep_update_handle(intptr_t handle, intptr_t topk_idx, intptr_t layout_info, intptr_t stream):
    with nogil:
        __status__ = ncclEpUpdateHandle(<EpHandle>handle, <NDTensor>topk_idx, <const ncclEpLayoutInfo_t*>layout_info, <Stream>stream)
    check_status(__status__)


cpdef ep_dispatch(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t layout_info, intptr_t config, intptr_t stream):
    with nogil:
        __status__ = ncclEpDispatch(<EpHandle>handle, <const ncclEpDispatchInputs_t*>inputs, <const ncclEpDispatchOutputs_t*>outputs, <const ncclEpLayoutInfo_t*>layout_info, <const ncclEpDispatchConfig_t*>config, <Stream>stream)
    check_status(__status__)


cpdef ep_combine(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t config, intptr_t stream):
    with nogil:
        __status__ = ncclEpCombine(<EpHandle>handle, <const ncclEpCombineInputs_t*>inputs, <const ncclEpCombineOutputs_t*>outputs, <const ncclEpCombineConfig_t*>config, <Stream>stream)
    check_status(__status__)


cpdef ep_complete(intptr_t handle, intptr_t config, intptr_t stream):
    with nogil:
        __status__ = ncclEpComplete(<EpHandle>handle, <const ncclEpCompleteConfig_t*>config, <Stream>stream)
    check_status(__status__)


cpdef intptr_t ep_tensor_get_data(intptr_t tensor) except? 0:
    cdef void* data
    with nogil:
        __status__ = ncclEpTensorGetData(<NDTensor>tensor, &data)
    check_status(__status__)
    return <intptr_t>data


cpdef ep_tensor_get_sizes(intptr_t tensor, intptr_t sizes, intptr_t ndim):
    with nogil:
        __status__ = ncclEpTensorGetSizes(<NDTensor>tensor, <const size_t**>sizes, <unsigned int*>ndim)
    check_status(__status__)
