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
from nccl import bindings as _nccl_bindings

__all__ = ["LibraryInfo", "VersionInfo", "get_version", "show_versions"]

# Banners embedded in libnccl.so / libnccl_ep.so (see src/init.cc VERSION_STRING
# and contrib/nccl_ep/nccl_ep.cc NCCL_EP_VERSION_STRING). Used to recover the
# CUDA toolkit major.minor each .so was BUILT with.
_NCCL_BANNER = re.compile(rb"NCCL version [^\+\s\x00]+\+cuda(\d+)\.(\d+)")
_NCCL_EP_BANNER = re.compile(rb"NCCL EP version [^\+\s\x00]+\+cuda(\d+)\.(\d+)")

# Match a SONAME like 'libnccl.so' (possibly with a trailing .X.Y.Z suffix).
_SO_BASENAME = re.compile(r"^(?P<soname>lib[\w-]+\.so)(?:\.[\d.]+)?$")


def _decode_version(v: int) -> _Version:
    """Decode an NCCL-style packed version integer (X*10000 + Y*100 + Z, or
    legacy X*1000 + Y*100 + Z) into a packaging Version."""
    if v >= 10000:
        major = v // 10000
        minor = (v % 10000) // 100
        patch = v % 100
    else:
        major = v // 1000
        minor = (v % 1000) // 100
        patch = v % 100
    return _Version(f"{major}.{minor}.{patch}")


def _resolve_so_path(soname: str) -> Path | None:
    """Absolute path of a currently-mapped shared library, by SONAME.

    Scans /proc/self/maps for an entry whose basename, after stripping any
    trailing ``.X.Y.Z`` version suffix, equals ``soname``. So ``libnccl.so``
    matches ``libnccl.so.2.30.0`` but not ``libnccl_ep.so.0.1.0``.

    Returns None if the library is not mapped (e.g. not yet dlopened) or if
    /proc/self/maps is unreadable.
    """
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.rstrip().split(maxsplit=5)
                if len(parts) < 6:
                    continue
                p = Path(parts[5])
                m = _SO_BASENAME.match(p.name)
                if m is not None and m.group("soname") == soname:
                    return p
    except OSError:
        return None
    return None


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
    """Build- and load-time info for a single shared library."""

    version: _Version
    """Library release version (e.g. ``2.30.0``)."""

    cuda_variant: _Version | None
    """CUDA toolkit major.minor the library was built with (e.g. ``12.9``),
    or None if the CUDA version could not be determined from the shared library."""

    path: Path | None
    """Absolute path of the loaded ``.so`` file, or None if the library is
    not mapped into the process."""

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

    nccl: LibraryInfo
    """Build/load info for ``libnccl.so``."""

    nccl_ep: LibraryInfo | None
    """Build/load info for ``libnccl_ep.so``, or None on CUDA-12 hosts where
    ``libnccl_ep.so`` cannot load."""


def _nccl_library_info() -> LibraryInfo:
    version = _decode_version(_nccl_bindings.get_version())
    path = _resolve_so_path("libnccl.so")
    return LibraryInfo(
        version=version,
        cuda_variant=_extract_cuda_variant(path, _NCCL_BANNER),
        path=path,
    )


def _nccl_ep_library_info() -> LibraryInfo | None:
    # Lazy-import inside the function so plain `import nccl` doesn't trigger
    # nccl.ep's CUDA-major gate or its dlopen of libnccl_ep.so.
    try:
        from nccl.ep import bindings as _ep_bindings
    except ImportError:
        return None
    version = _decode_version(_ep_bindings.get_version())
    path = _resolve_so_path("libnccl_ep.so")
    return LibraryInfo(
        version=version,
        cuda_variant=_extract_cuda_variant(path, _NCCL_EP_BANNER),
        path=path,
    )


def get_version() -> VersionInfo:
    """Return a structured snapshot of NCCL stack versions.

    Returns:
        :py:class:`VersionInfo` carrying nccl4py + libnccl + libnccl_ep
        versions, build-time CUDA variants, and loaded ``.so`` paths.
    """
    return VersionInfo(
        nccl4py=_Version(__version__),
        nccl=_nccl_library_info(),
        nccl_ep=_nccl_ep_library_info(),
    )


def show_versions() -> None:
    """Print a summary of the installed NCCL stack to stdout.

    For each component, reports the release version, the CUDA toolkit it was
    built with, and (for native libraries) the absolute path of the loaded
    ``.so``.
    """
    v = get_version()
    label_width = 12
    print()
    print("NCCL versions")
    print("-------------")
    print(f"{'nccl4py':<{label_width}}: {v.nccl4py}")
    print(f"{'libnccl':<{label_width}}: {v.nccl}")
    ep = str(v.nccl_ep) if v.nccl_ep is not None else "not available (CUDA-12 host)"
    print(f"{'libnccl_ep':<{label_width}}: {ep}")
