# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See LICENSE.txt for license information

"""
Communicator-owned resource management.

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
]


class CommResource(ABC):
    """
    Abstract base class for NCCL communicator-owned resources.

    Resources are tied to a specific communicator and must be explicitly
    deallocated before the communicator is destroyed.
    """

    def __init__(self, comm_ptr: int):
        """
        Initializes resource with communicator pointer.

        Args:
            - comm_ptr (int): Raw NCCL communicator pointer.

        Raises:
            - ``NcclInvalid``: If comm_ptr is 0 (invalid communicator).
        """
        if comm_ptr == 0:
            raise NcclInvalid(
                "Invalid communicator: cannot create resource with communicator ptr=0"
            )
        self._comm_ptr = comm_ptr
        self._closed = False

    @abstractmethod
    def _allocate(self) -> None:
        """
        Allocates the underlying NCCL resource.

        Notes:
            This method is called during initialization and must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _deallocate(self) -> None:
        """
        Deallocates the underlying NCCL resource.

        Notes:
            This method is called during close() and must be implemented by subclasses.
        """
        pass

    def close(self) -> None:
        """
        Explicitly deallocates the resource.

        Notes:
            This method is idempotent and safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        self._deallocate()

    def _check_valid(self) -> None:
        """
        Checks if resource is valid and raises if closed.

        Raises:
            - ``RuntimeError``: If resource has been closed.
        """
        if self._closed:
            raise RuntimeError(f"{self.__class__.__name__} has been closed and is no longer valid")

    @property
    def is_valid(self) -> bool:
        """
        Checks if resource is still valid (not closed).

        Returns:
            ``bool``: True if valid, False if closed.
        """
        return not self._closed


class RegisteredBufferHandle(CommResource):
    """
    NCCL registered buffer handle for zero-copy optimized communication.

    Registers a user buffer with the communicator to enable performance optimizations
    in NCCL operations. The registration handle can be closed manually or is automatically
    closed when the communicator is destroyed or aborted.

    Attributes:
        handle (int): Registration handle for NCCL operations.
        size (int): Size of registered buffer in bytes.
        is_valid (bool): Whether the registration is still valid.

    See Also:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommregister
    """

    def __init__(self, comm_ptr: int, buffer_ptr: int, size: int):
        """
        Creates and registers a buffer with NCCL.

        Args:
            - comm_ptr (int): NCCL communicator raw pointer.
            - buffer_ptr (int): Device pointer to the buffer.
            - size (int): Size of buffer in bytes.

        Raises:
            - ``NcclInvalid``: If comm_ptr is 0 (invalid communicator).
        """
        self._buffer_ptr = buffer_ptr
        self._size = size
        self._handle: int | None = None
        super().__init__(comm_ptr)
        self._allocate()

    def _allocate(self) -> None:
        """
        Registers buffer with NCCL for zero-copy communication.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommregister
        """
        self._handle = _nccl_bindings.comm_register(self._comm_ptr, self._buffer_ptr, self._size)

    def _deallocate(self) -> None:
        """
        Deregisters buffer from NCCL.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommderegister
        """
        if self._handle is not None:
            _nccl_bindings.comm_deregister(self._comm_ptr, self._handle)
            self._handle = None

    @property
    def handle(self) -> int:
        """
        Registration handle for NCCL operations.

        Returns:
            ``int``: The registration handle.

        Raises:
            - ``RuntimeError``: If buffer has been deregistered or handle is invalid.
        """
        self._check_valid()
        if self._handle is None:
            raise RuntimeError("Buffer registration handle is invalid")
        return self._handle

    @property
    def size(self) -> int:
        """
        Size of the registered buffer in bytes.

        Returns:
            ``int``: Buffer size in bytes.
        """
        return self._size

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<RegisteredBufferHandle: closed>"
        return f"<RegisteredBufferHandle: size={self._size}, handle={self._handle:#x}>"


class RegisteredWindowHandle(CommResource):
    """
    NCCL registered window handle for Remote Memory Access (RMA) operations.

    Registers a memory window with the communicator for one-sided communication patterns.
    The window handle can be closed manually or is automatically closed when the
    communicator is destroyed or aborted.

    Attributes:
        handle (int): Window handle for NCCL operations.
        size (int): Size of registered window in bytes.
        is_valid (bool): Whether the window registration is still valid.

    Notes:
        This is a collective operation. All ranks must call register_window() with equal buffer sizes.

    See Also:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommwindowregister
    """

    def __init__(self, comm_ptr: int, buffer_ptr: int, size: int, flags: WindowFlag | None = None):
        """
        Creates and registers a memory window with NCCL.

        Args:
            - comm_ptr (int): NCCL communicator raw pointer.
            - buffer_ptr (int): Device pointer to the buffer.
            - size (int): Size of window in bytes.
            - flags (WindowFlag, optional): Window registration flags to control behavior. Defaults to None.

        Raises:
            - ``NcclInvalid``: If comm_ptr is 0 (invalid communicator).
        """
        self._buffer_ptr = buffer_ptr
        self._size = size
        self._flags = flags if flags is not None else WindowFlag.Default
        self._handle: int | None = None
        super().__init__(comm_ptr)
        self._allocate()

    def _allocate(self) -> None:
        """
        Collectively registers window with NCCL.

        Notes:
            This is a collective operation called during initialization.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommwindowregister
        """
        self._handle = _nccl_bindings.comm_window_register(
            self._comm_ptr, self._buffer_ptr, self._size, self._flags.value
        )

    def _deallocate(self) -> None:
        """
        Deregisters window from NCCL.

        Notes:
            Deregistration is local to the rank. Caller must ensure the buffer is not being
            accessed by any NCCL operation.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommwindowderegister
        """
        if self._handle is not None:
            _nccl_bindings.comm_window_deregister(self._comm_ptr, self._handle)
            self._handle = None

    @property
    def handle(self) -> int:
        """
        Window handle for NCCL operations.

        Returns:
            ``int``: The window handle.

        Raises:
            - ``RuntimeError``: If window has been deregistered or handle is invalid.
        """
        self._check_valid()
        if self._handle is None:
            raise RuntimeError("Window registration handle is invalid")
        return self._handle

    @property
    def size(self) -> int:
        """
        Size of the registered window in bytes.

        Returns:
            ``int``: Window size in bytes.
        """
        return self._size

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<RegisteredWindowHandle: closed>"
        return f"<RegisteredWindowHandle: size={self._size}, handle={self._handle:#x}, flags={self._flags}>"


class CustomRedOp(CommResource):
    """
    NCCL user-defined custom reduction operator.

    Creates a custom reduction operator (e.g., PreMulSum) for collective operations.
    The PreMulSum operator performs: output = scalar * sum(inputs), useful for averaging
    or weighted reductions. The operator can be closed manually or is automatically
    destroyed when the communicator is destroyed or aborted.

    Attributes:
        op (int): Operator handle for use in reduction operations.

    See Also:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#ncclredopcreatepremulsum
    """

    def __init__(
        self,
        comm_ptr: int,
        scalar_ptr: int,
        datatype: NcclDataType,
        residence: _nccl_bindings.ScalarResidence,
    ):
        """
        Creates a custom reduction operator.

        Args:
            - comm_ptr (int): NCCL communicator raw pointer.
            - scalar_ptr (int): Pointer to scalar value (host or device memory).
            - datatype (NcclDataType): NCCL data type of the scalar and reduction.
            - residence (ScalarResidence): Enum indicating scalar memory location (HostImmediate or Device).

        Raises:
            - ``NcclInvalid``: If comm_ptr is 0 (invalid communicator).
        """
        self._scalar_ptr = scalar_ptr
        self._datatype = datatype
        self._residence = residence
        self._op: int | None = None

        super().__init__(comm_ptr)
        self._allocate()

    def _allocate(self) -> None:
        """
        Creates the custom reduction operator in NCCL.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#ncclredopcreatepremulsum
        """
        self._op = _nccl_bindings.red_op_create_pre_mul_sum(
            self._scalar_ptr, int(self._datatype), int(self._residence), self._comm_ptr
        )

    def _deallocate(self) -> None:
        """
        Destroys the custom reduction operator in NCCL.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#ncclredopdestroy
        """
        if self._op is not None:
            _nccl_bindings.red_op_destroy(self._op, self._comm_ptr)
            self._op = None

    @property
    def op(self) -> int:
        """
        Operator handle for use in reduction operations.

        Returns:
            ``int``: The custom reduction operator handle.

        Raises:
            - ``RuntimeError``: If operator has been destroyed or is invalid.
        """
        self._check_valid()
        if self._op is None:
            raise RuntimeError("RedOp is invalid")
        return self._op

    def __int__(self) -> int:
        """
        Returns operator handle as int for direct use in collective operations.

        Returns:
            ``int``: The custom reduction operator handle.

        Raises:
            - ``RuntimeError``: If operator has been destroyed or is invalid.
        """
        self._check_valid()
        if self._op is None:
            raise RuntimeError("RedOp is invalid")
        return self._op

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<CustomRedOp: closed>"
        return f"<CustomRedOp: type=PreMulSum, dtype={self._datatype}, residence={self._residence.name}, op={self._op}>"
