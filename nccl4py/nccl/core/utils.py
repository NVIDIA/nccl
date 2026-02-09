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
Utility functions and classes for NCCL operations.

This module provides version information, unique identifiers for communicator
initialization, and error string utilities for NCCL operations.
"""

from __future__ import annotations

import numpy as _np
from packaging.version import Version as _Version

from nccl._version import __version__
from nccl import bindings as _nccl_bindings

__all__ = ["Version", "get_version", "UniqueId", "get_unique_id", "get_error_string"]


_version_cache = None


class Version:
    """
    Version information for NCCL4Py and NCCL library.

    Attributes:
        nccl_version (Version): NCCL library version.
        nccl4py_version (Version): NCCL4Py package version.
    """

    def __init__(self, nccl_version: int) -> None:
        """
        Initializes Version object from NCCL version integer.

        Args:
            - nccl_version (int): NCCL version as an integer.
        """
        v = nccl_version
        if v >= 10000:
            major = v // 10000
            minor = (v % 10000) // 100
            patch = v % 100
        else:
            major = v // 1000
            minor = (v % 1000) // 100
            patch = v % 100

        self.nccl_version = _Version(f"{major}.{minor}.{patch}")
        self.nccl4py_version = _Version(__version__)

    def __repr__(self) -> str:
        return f"""
Versions:
    NCCL4Py version: {self.nccl4py_version}
    NCCL Library version: {self.nccl_version}
"""


def get_version() -> Version:
    """
    Gets the version information for NCCL and NCCL4Py.

    Returns:
        ``Version``: Version object containing NCCL and NCCL4Py version information.
    """
    global _version_cache
    if _version_cache is None:
        _version_cache = Version(int(_nccl_bindings.get_version()))
    return _version_cache


class UniqueId:
    """
    NCCL unique identifier for communicator initialization.

    A UniqueId is used to coordinate communicator initialization across multiple ranks.
    All ranks must use the same UniqueId to form a communicator.

    Attributes:
        ptr (int): Pointer to the internal NCCL unique ID structure.
        as_ndarray (np.ndarray): NumPy array representation of the unique ID.
        as_bytes (bytes): Bytes representation of the unique ID.
    """

    def __init__(self) -> None:
        """
        Initializes an empty UniqueId.

        Notes:
            Use ``get_unique_id()`` to generate a valid unique ID for communicator initialization.
        """
        self._internal: _nccl_bindings.UniqueId = _nccl_bindings.UniqueId()

    def __repr__(self) -> str:
        """
        Returns truncated bytes representation of UniqueId.

        Returns:
            ``str``: Hex representation showing first and last 8 bytes.
        """
        # Show first 8 and last 8 bytes in hex
        bytes_data = self.as_bytes
        if len(bytes_data) <= 32:
            hex_str = bytes_data.hex()
        else:
            hex_str = bytes_data[:8].hex() + "..." + bytes_data[-8:].hex()
        return f"<UniqueId: {hex_str}>"

    @staticmethod
    def from_bytes(b: bytes) -> UniqueId:
        """
        Creates a UniqueId from bytes.

        Args:
            - b (bytes): Bytes representation of a UniqueId.

        Returns:
            ``UniqueId``: Reconstructed UniqueId object.

        Raises:
            - ``TypeError``: If b is not a bytes-like object.
            - ``ValueError``: If b has incorrect length.
        """
        try:
            buf = b if isinstance(b, (bytes, bytearray)) else b.tobytes()
        except Exception as e:
            raise TypeError("'b' must be a bytes-like object") from e

        expected = int(_nccl_bindings.unique_id_dtype.itemsize)
        if len(buf) != expected:
            raise ValueError(f"unique id must be {expected} bytes, got {len(buf)}")

        arr = _np.frombuffer(buf, dtype=_nccl_bindings.unique_id_dtype, count=1)

        uid = UniqueId.__new__(UniqueId)
        uid._internal = _nccl_bindings.UniqueId.from_data(arr)
        return uid

    @property
    def ptr(self) -> int:
        """
        Pointer to the internal NCCL unique ID structure.

        Returns:
            ``int``: Internal structure pointer.
        """
        return self._internal.ptr

    @property
    def as_ndarray(self) -> _np.ndarray:
        """
        NumPy array representation of the unique ID.

        Returns:
            ``np.ndarray``: Array containing the unique ID data.
        """
        return self._internal._data

    @property
    def as_bytes(self) -> bytes:
        """
        Bytes representation of the unique ID.

        Returns:
            ``bytes``: Unique ID as bytes (for serialization/broadcast).
        """
        return self._internal._data.tobytes()


def get_unique_id(empty: bool = False) -> UniqueId:
    """
    Generates a new NCCL unique identifier for communicator initialization.

    Args:
        - empty (bool, optional): If True, return an empty UniqueId without calling NCCL. Defaults to False.

    Returns:
        ``UniqueId``: A new unique identifier to be shared across ranks.

    Notes:
        This should be called by one rank (typically rank 0) and the resulting
        UniqueId should be broadcast to all other ranks.
    """
    uid = UniqueId()
    if empty:
        return uid
    _nccl_bindings.get_unique_id(uid.ptr)
    return uid


def get_error_string(nccl_result: _nccl_bindings.Result | int) -> str:
    """
    Gets the error string for an NCCL result code.

    Args:
        - nccl_result (Result | int): NCCL result code.

    Returns:
        ``str``: Human-readable error message corresponding to the result code.
    """
    return _nccl_bindings.get_error_string(int(nccl_result))
