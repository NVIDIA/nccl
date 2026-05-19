# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Utility functions and classes for NCCL operations.

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
    """Version information for NCCL4Py and the NCCL library.

    For the full stack version (including ``libnccl_ep.so``), use
    :py:func:`nccl.get_version` / :py:func:`nccl.show_versions`.

    Attributes:
        nccl_version: NCCL library version.
        nccl4py_version: NCCL4Py package version.
    """

    def __init__(self, nccl_version: int) -> None:
        """Initializes a Version object from an NCCL version integer.

        Args:
            nccl_version: NCCL version as an integer (per ncclGetVersion).
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
    """Returns the version information for NCCL and NCCL4Py.

    The result is cached after the first call.

    Returns:
        :py:class:`Version` object containing NCCL and NCCL4Py version
        information.
    """
    global _version_cache
    if _version_cache is None:
        _version_cache = Version(_nccl_bindings.get_version())
    return _version_cache


class UniqueId:
    """NCCL unique identifier for communicator initialization.

    A UniqueId is used to coordinate communicator initialization across
    multiple ranks. All ranks must use the same UniqueId to form a
    communicator. Typically one rank generates the UniqueId via
    :py:func:`get_unique_id` and broadcasts it to all other ranks. Three
    serialization paths are supported:

    * **Bytes**: ``bytes(uid)`` (or :py:attr:`as_bytes`) on the producer,
      :py:meth:`from_bytes` on receivers. The bytes of unique ID can be
      transmitted through any byte-oriented channel — a TCP socket, a
      shared filesystem, etc.
    * **NumPy**: :py:attr:`as_ndarray` returns an in-place view of the
      underlying buffer, suitable for NumPy-aware buffer transports such
      as ``mpi4py.MPI.Comm.Bcast`` (uppercase ``B``).
    * **Pickle**: instances are picklable directly, so higher level
      object broadcast helpers like ``mpi4py.MPI.Comm.bcast`` (lowercase
      ``b``) work out of the box.
    """

    def __init__(self) -> None:
        """Initializes an empty UniqueId.

        Use :py:func:`get_unique_id` to generate a valid unique ID for
        communicator initialization.
        """
        self._internal: _nccl_bindings.UniqueId = _nccl_bindings.UniqueId()

    def __repr__(self) -> str:
        # Show first 8 and last 8 bytes in hex
        bytes_data = self.as_bytes
        if len(bytes_data) <= 32:
            hex_str = bytes_data.hex()
        else:
            hex_str = bytes_data[:8].hex() + "..." + bytes_data[-8:].hex()
        return f"<UniqueId: {hex_str}>"

    def __bytes__(self) -> bytes:
        return bytes(self._internal)

    @staticmethod
    def from_bytes(b: bytes | bytearray | memoryview) -> UniqueId:
        """Reconstructs a UniqueId from a bytes-like buffer.

        Args:
            b: Bytes representation of a UniqueId, typically obtained via
                the :py:attr:`as_bytes` property on the producing rank.

        Returns:
            Reconstructed :py:class:`UniqueId`.
        """
        uid = UniqueId.__new__(UniqueId)
        uid._internal = _nccl_bindings.UniqueId.from_buffer(b)
        return uid

    @property
    def ptr(self) -> int:
        """Raw pointer to the underlying NCCL unique ID structure."""
        return self._internal.ptr

    @property
    def as_ndarray(self) -> _np.ndarray:
        """NumPy array view of the unique ID data."""
        return _np.ndarray((1,), dtype=_nccl_bindings.unique_id_dtype, buffer=self._internal).view(
            _np.recarray
        )

    @property
    def as_bytes(self) -> bytes:
        """Bytes representation of the unique ID, suitable for serialization or broadcast."""
        return bytes(self)


def get_unique_id(empty: bool = False) -> UniqueId:
    """Generates a new NCCL unique identifier for communicator initialization.

    Should be called by one rank (typically rank 0); the resulting
    :py:class:`UniqueId` must then be broadcast (e.g. via MPI) to all other
    ranks.

    Args:
        empty: If True, return an empty :py:class:`UniqueId` without calling
            NCCL. Useful when the bytes will be filled in later via
            :py:meth:`UniqueId.from_bytes`. Defaults to False.

    Returns:
        A new :py:class:`UniqueId` to be shared across ranks.
    """
    uid = UniqueId()
    if empty:
        return uid
    _nccl_bindings.get_unique_id(uid.ptr)
    return uid


def get_error_string(nccl_result: _nccl_bindings.Result | int) -> str:
    """Returns a human-readable error string for an NCCL result code.

    Args:
        nccl_result: NCCL result code.

    Returns:
        Human-readable error message corresponding to the result code.
    """
    return _nccl_bindings.get_error_string(int(nccl_result))
