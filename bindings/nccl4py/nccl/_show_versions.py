# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Aggregated version info for the NCCL stack (nccl4py + libnccl + libnccl_ep).

Public entry points are re-exported from the top-level ``nccl`` package:

    import nccl
    nccl.show_versions()           # print human-readable block to stdout
    v = nccl.get_version()         # programmatic; returns a VersionInfo dataclass

For just nccl4py's own package version, use ``nccl.__version__``.
"""

from __future__ import annotations

import mmap
import re
from dataclasses import dataclass
from pathlib import Path

from packaging.version import Version as _Version

from nccl._version import __version__

__all__ = ["LibraryInfo", "VersionInfo", "get_version", "show_versions"]

# Banners embedded in libnccl.so / libnccl_ep.so (see src/init.cc VERSION_STRING
# and contrib/nccl_ep/nccl_ep.cc NCCL_EP_VERSION_STRING). Used to recover the
# CUDA toolkit major.minor each .so was built with.
_NCCL_BANNER = re.compile(rb"NCCL version [^\+\s\x00]+\+cuda(\d+)\.(\d+)")
_NCCL_EP_BANNER = re.compile(rb"NCCL EP version [^\+\s\x00]+\+cuda(\d+)\.(\d+)")


def _extract_cuda_variant(path: Path | None, pattern: re.Pattern[bytes]) -> _Version | None:
    """Recover ``+cudaA.B`` from a .so's embedded version banner via mmap."""
    if path is None:
        return None
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            m = pattern.search(mm)
            if m is not None:
                return _Version(f"{int(m.group(1))}.{int(m.group(2))}")
    except OSError:
        pass
    return None


@dataclass(frozen=True)
class LibraryInfo:
    """Version, CUDA build variant, and loaded-path info for a shared library."""

    version: _Version
    """Library release version (e.g. ``2.30.0``)."""

    cuda_variant: _Version | None
    """CUDA toolkit major.minor the library was built with (e.g. ``12.9``), or
    None if it could not be read from the library."""

    path: Path | None
    """Path of the loaded ``libnccl.so``, or None if it cannot be determined."""

    def __str__(self) -> str:
        out = str(self.version)
        if self.cuda_variant is not None:
            out += f"+cuda{self.cuda_variant}"
        if self.path is not None:
            out += f" @ {self.path}"
        return out


@dataclass(frozen=True)
class VersionInfo:
    """Aggregate version snapshot of the NCCL stack."""

    nccl4py: _Version
    """nccl4py package version."""

    nccl: LibraryInfo | None
    """Version/CUDA-variant/path of the ``libnccl.so`` nccl4py is using, or
    None when it cannot be loaded."""

    nccl_ep: LibraryInfo | None
    """Version/CUDA-variant/path of the ``libnccl_ep.so`` nccl4py is using, or
    None when it cannot be loaded."""


def get_version() -> VersionInfo:
    """Return a structured snapshot of NCCL stack versions.

    Returns:
        :py:class:`VersionInfo` with nccl4py + libnccl + libnccl_ep versions,
        CUDA build variants, and loaded ``.so`` paths.
    """
    nccl = None
    try:
        from nccl import core as _core

        path = _core.get_lib_path()
        nccl = LibraryInfo(_core.get_lib_version(), _extract_cuda_variant(path, _NCCL_BANNER), path)
    except (ImportError, RuntimeError):
        pass

    nccl_ep = None
    try:
        from nccl import ep as _ep

        path = _ep.get_lib_path()
        nccl_ep = LibraryInfo(
            _ep.get_lib_version(), _extract_cuda_variant(path, _NCCL_EP_BANNER), path
        )
    except (ImportError, RuntimeError):
        pass

    return VersionInfo(nccl4py=_Version(__version__), nccl=nccl, nccl_ep=nccl_ep)


def show_versions() -> None:
    """Print a summary of the installed NCCL stack to stdout.

    For each component, reports the release version, the CUDA toolkit it was
    built with, and (for native libraries) the path of the loaded ``.so``.
    """
    v = get_version()
    nccl = str(v.nccl) if v.nccl is not None else "not available"
    ep = str(v.nccl_ep) if v.nccl_ep is not None else "not available"
    label_width = 12
    print()
    print("NCCL versions")
    print("-------------")
    print(f"{'nccl4py':<{label_width}}: {v.nccl4py}")
    print(f"{'libnccl':<{label_width}}: {nccl}")
    print(f"{'libnccl_ep':<{label_width}}: {ep}")
