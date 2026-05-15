# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Communicator-owned resource management.

This module provides resource classes for NCCL communicator-owned objects including
registered buffers for zero-copy communication, registered windows for RMA operations,
and custom reduction operators. All resources are automatically cleaned up when the
owning communicator is destroyed or aborted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nccl import bindings as _nccl_bindings

from nccl.core.constants import WindowFlag
from nccl.core.typing import NcclDataType, NcclInvalid

__all__ = [
    "RegisteredBufferHandle",
    "RegisteredWindowHandle",
    "CustomRedOp",
    "DevCommResource",
]


class CommResource(ABC):
    """Abstract base class for NCCL communicator-owned resources.

    Resources are tied to a specific communicator. They can be released
    explicitly via :py:meth:`close`, and are released automatically when the
    owning communicator is destroyed or aborted.
    """

    def __init__(self, comm_ptr: int):
        """Initializes the resource with a communicator pointer.

        Args:
            comm_ptr: Raw NCCL communicator pointer.

        Raises:
            NcclInvalid: If comm_ptr is 0 (invalid communicator).
        """
        if comm_ptr == 0:
            raise NcclInvalid(
                "Invalid communicator: cannot create resource with communicator ptr=0"
            )
        self._comm_ptr = comm_ptr
        self._closed = False

    @abstractmethod
    def _allocate(self) -> None:
        """Allocates the underlying NCCL resource.

        Called during initialization; must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _deallocate(self) -> None:
        """Deallocates the underlying NCCL resource.

        Called during :py:meth:`close`; must be implemented by subclasses.
        """
        pass

    def close(self) -> None:
        """Explicitly deallocates the resource.

        Idempotent: safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        self._deallocate()

    def _check_valid(self) -> None:
        """Raises if the resource has been closed.

        Raises:
            RuntimeError: If the resource has been closed.
        """
        if self._closed:
            raise RuntimeError(f"{self.__class__.__name__} has been closed and is no longer valid")

    @property
    def is_valid(self) -> bool:
        """Whether the resource is still valid (not closed)."""
        return not self._closed


class RegisteredBufferHandle(CommResource):
    """NCCL registered buffer handle for zero-copy optimized communication.

    Registers a user buffer with the communicator to enable performance
    optimizations in NCCL operations. Created by
    :py:meth:`Communicator.register_buffer`. The registration handle can be
    released explicitly via :py:meth:`close`, or automatically when the
    owning communicator is destroyed or aborted.
    """

    def __init__(self, comm_ptr: int, buffer_ptr: int, size: int):
        """Creates and registers a buffer with NCCL.

        Args:
            comm_ptr: NCCL communicator raw pointer.
            buffer_ptr: Device pointer to the buffer.
            size: Size of the buffer in bytes.

        Raises:
            NcclInvalid: If comm_ptr is 0 (invalid communicator).
        """
        self._buffer_ptr = buffer_ptr
        self._size = size
        self._handle: int | None = None
        super().__init__(comm_ptr)
        self._allocate()

    def _allocate(self) -> None:
        """Registers the buffer with NCCL for zero-copy communication."""
        self._handle = _nccl_bindings.comm_register(self._comm_ptr, self._buffer_ptr, self._size)

    def _deallocate(self) -> None:
        """Deregisters the buffer from NCCL."""
        if self._handle is not None:
            _nccl_bindings.comm_deregister(self._comm_ptr, self._handle)
            self._handle = None

    @property
    def handle(self) -> int:
        """Registration handle for NCCL operations.

        Raises:
            RuntimeError: If the buffer has been deregistered or the handle
                is invalid.
        """
        self._check_valid()
        if self._handle is None:
            raise RuntimeError("Buffer registration handle is invalid")
        return self._handle

    @property
    def size(self) -> int:
        """Size of the registered buffer in bytes."""
        return self._size

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<RegisteredBufferHandle: closed>"
        return f"<RegisteredBufferHandle: size={self._size}, handle={self._handle:#x}>"


class RegisteredWindowHandle(CommResource):
    """NCCL registered window handle for Remote Memory Access (RMA) operations.

    Registers a memory window with the communicator for one-sided communication
    patterns. Created by :py:meth:`Communicator.register_window`. Registration
    is collective: all ranks must call
    :py:meth:`Communicator.register_window` with equal buffer sizes by
    default. Deregistration is local. The window handle can be released
    explicitly via :py:meth:`close`, or automatically when the owning
    communicator is destroyed or aborted.
    """

    def __init__(self, comm_ptr: int, buffer_ptr: int, size: int, flags: WindowFlag | None = None):
        """Creates and registers a memory window with NCCL.

        Args:
            comm_ptr: NCCL communicator raw pointer.
            buffer_ptr: Device pointer to the buffer.
            size: Size of the window in bytes.
            flags: Window registration flags. Defaults to ``None``
                (:py:attr:`~nccl.core.WindowFlag.DEFAULT`).

        Raises:
            NcclInvalid: If comm_ptr is 0 (invalid communicator).
        """
        self._buffer_ptr = buffer_ptr
        self._size = size
        self._flags = flags if flags is not None else WindowFlag.DEFAULT
        self._handle: int | None = None
        super().__init__(comm_ptr)
        self._allocate()

    def _allocate(self) -> None:
        """Collectively registers the window with NCCL."""
        self._handle = _nccl_bindings.comm_window_register(
            self._comm_ptr, self._buffer_ptr, self._size, self._flags.value
        )

    def _deallocate(self) -> None:
        """Deregisters the window from NCCL.

        Deregistration is local to the rank. The caller must ensure the
        buffer is not being accessed by any NCCL operation.
        """
        if self._handle is not None:
            _nccl_bindings.comm_window_deregister(self._comm_ptr, self._handle)
            self._handle = None

    @property
    def handle(self) -> int:
        """Window handle for NCCL operations.

        Raises:
            RuntimeError: If the window has been deregistered or the handle
                is invalid.
        """
        self._check_valid()
        if self._handle is None:
            raise RuntimeError("Window registration handle is invalid")
        return self._handle

    @property
    def size(self) -> int:
        """Size of the registered window in bytes."""
        return self._size

    @property
    def user_ptr(self) -> int:
        """Original user buffer pointer registered with this window.

        Raises:
            RuntimeError: If the window has been deregistered.
        """
        self._check_valid()
        return self._buffer_ptr

    def get_lsa_multimem_device_pointer(self, offset: int = 0) -> int | None:
        """Returns the LSA multicast device pointer for this window.

        Returns a device pointer suitable for multicast operations over the
        LSA (Load/Store Accessible) team. The pointer is valid as long as
        the window and communicator remain alive.

        Args:
            offset: Byte offset within the window buffer. Defaults to 0.

        Returns:
            Device pointer as int, or ``None`` if multimem is not supported.

        Raises:
            RuntimeError: If the window has been closed.
        """
        self._check_valid()
        ptr = _nccl_bindings.get_lsa_multimem_device_pointer(self._handle, offset)
        return ptr if ptr != 0 else None

    def get_lsa_device_pointer(self, lsa_rank: int, offset: int = 0) -> int:
        """Returns the LSA device pointer for a peer within the LSA team.

        Returns a device pointer to the peer's window buffer addressable
        from the local GPU via LSA (Load/Store Accessible) mapping.

        Args:
            lsa_rank: Rank within the LSA team (0 to lsa_size - 1).
            offset: Byte offset within the window buffer. Defaults to 0.

        Returns:
            Device pointer as int.

        Raises:
            RuntimeError: If the window has been closed.
        """
        self._check_valid()
        return _nccl_bindings.get_lsa_device_pointer(self._handle, offset, lsa_rank)

    def get_peer_device_pointer(self, peer: int, offset: int = 0) -> int | None:
        """Returns a device pointer to a peer's window buffer by world rank.

        If the peer is not reachable via LSA, returns ``None``.

        Args:
            peer: World rank of the peer (0 to nranks - 1).
            offset: Byte offset within the window buffer. Defaults to 0.

        Returns:
            Device pointer as int, or ``None`` if the peer is not reachable
            via LSA.

        Raises:
            RuntimeError: If the window has been closed.
        """
        self._check_valid()
        ptr = _nccl_bindings.get_peer_device_pointer(self._handle, offset, peer)
        return ptr if ptr != 0 else None

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<RegisteredWindowHandle: closed>"
        return f"<RegisteredWindowHandle: size={self._size}, handle={self._handle:#x}, flags={self._flags}>"


class CustomRedOp(CommResource):
    """NCCL user-defined custom reduction operator.

    Created by :py:meth:`Communicator.create_pre_mul_sum`. The PreMulSum
    operator performs ``output = scalar * sum(inputs)``, useful for averaging
    or weighted reductions. The operator can be released explicitly via
    :py:meth:`close`, or automatically when the owning communicator is
    destroyed or aborted.
    """

    def __init__(
        self,
        comm_ptr: int,
        scalar_ptr: int,
        datatype: NcclDataType,
        residence: _nccl_bindings.ScalarResidence,
    ):
        """Creates a custom reduction operator.

        Args:
            comm_ptr: NCCL communicator raw pointer.
            scalar_ptr: Pointer to the scalar value (host or device memory).
            datatype: NCCL data type of the scalar and reduction.
            residence: Indicates scalar memory location (HostImmediate or
                Device).

        Raises:
            NcclInvalid: If comm_ptr is 0 (invalid communicator).
        """
        self._scalar_ptr = scalar_ptr
        self._datatype = datatype
        self._residence = residence
        self._op: int | None = None

        super().__init__(comm_ptr)
        self._allocate()

    def _allocate(self) -> None:
        """Creates the custom reduction operator in NCCL."""
        self._op = _nccl_bindings.red_op_create_pre_mul_sum(
            self._scalar_ptr, int(self._datatype), int(self._residence), self._comm_ptr
        )

    def _deallocate(self) -> None:
        """Destroys the custom reduction operator in NCCL."""
        if self._op is not None:
            _nccl_bindings.red_op_destroy(self._op, self._comm_ptr)
            self._op = None

    @property
    def op(self) -> int:
        """Operator handle for use in reduction operations.

        Raises:
            RuntimeError: If the operator has been destroyed or is invalid.
        """
        self._check_valid()
        if self._op is None:
            raise RuntimeError("RedOp is invalid")
        return self._op

    def __int__(self) -> int:
        """Returns the operator handle for direct use in collective operations.

        Raises:
            RuntimeError: If the operator has been destroyed or is invalid.
        """
        return self.op

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<CustomRedOp: closed>"
        return f"<CustomRedOp: type=PreMulSum, dtype={self._datatype}, residence={self._residence.name}, op={self._op}>"


class DevCommResource(CommResource):
    """NCCL device communicator resource for device-side operations.

    Wraps ``ncclDevComm_t`` and manages its lifecycle. Created by
    :py:meth:`Communicator.create_dev_comm`. The device communicator is
    automatically destroyed when the parent communicator is destroyed or
    aborted.
    """

    def __init__(self, comm_ptr: int, requirements_ptr: int):
        """Creates a device communicator from an existing host communicator.

        Args:
            comm_ptr: NCCL communicator raw pointer.
            requirements_ptr: Pointer to a ncclDevCommRequirements_t
                structure.

        Raises:
            NcclInvalid: If comm_ptr is 0 (invalid communicator).
        """
        self._requirements_ptr = requirements_ptr
        self._dev_comm: _nccl_bindings.DevComm | None = None
        super().__init__(comm_ptr)
        self._allocate()

    def _allocate(self) -> None:
        """Creates the device communicator via ncclDevCommCreate."""
        # Allocate DevComm struct first
        self._dev_comm = _nccl_bindings.DevComm()
        # Pass pointer to dev_comm_create to initialize it
        _nccl_bindings.dev_comm_create(self._comm_ptr, self._requirements_ptr, self._dev_comm.ptr)

    def _deallocate(self) -> None:
        """Destroys the device communicator via ncclDevCommDestroy."""
        if self._dev_comm is not None:
            _nccl_bindings.dev_comm_destroy(self._comm_ptr, self._dev_comm.ptr)
            self._dev_comm = None

    @property
    def dev_comm(self) -> _nccl_bindings.DevComm:
        """DevComm object wrapping :c:type:`ncclDevComm_t <ncclDevComm>`.

        Raises:
            RuntimeError: If the device communicator has been destroyed.
        """
        self._check_valid()
        if self._dev_comm is None:
            raise RuntimeError("DevComm is invalid")
        return self._dev_comm

    @property
    def ptr(self) -> int:
        """Raw pointer to the underlying :c:type:`ncclDevComm_t <ncclDevComm>` structure.

        Raises:
            RuntimeError: If the device communicator has been destroyed.
        """
        return self.dev_comm.ptr

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<DevCommResource: closed>"
        return f"<DevCommResource: ptr={self.ptr:#x}>"
