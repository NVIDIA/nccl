# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import PlatformError
from Cython.Build import cythonize

PACKAGE = "nccl.bindings"
LIBNAMES = ["nccl"]


def _cuda_include_dir() -> str:
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        raise PlatformError("CUDA_HOME is not set")

    cuda_include = Path(cuda_home) / "include"
    if not cuda_include.is_dir():
        raise PlatformError(f"CUDA include directory does not exist: {cuda_include}")

    return str(cuda_include)


class BuildExt(build_ext):
    """Add CUDA headers only when extension compilation starts."""

    def build_extensions(self) -> None:
        cuda_include = _cuda_include_dir()
        for extension in self.extensions:
            extension.include_dirs.append(cuda_include)

        super().build_extensions()


def _ext(module: str, source: str) -> Extension:
    return Extension(
        module,
        sources=[source],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )


def libname_extensions(libname: str) -> list[Extension]:
    """Three per-library extensions: lowpp, cy variant, _internal loader.

    For libname="nccl":
        nccl.bindings.nccl              <- nccl/bindings/nccl.pyx
        nccl.bindings.cynccl            <- nccl/bindings/cynccl.pyx
        nccl.bindings._internal.nccl    <- nccl/bindings/_internal/nccl_linux.pyx
    """
    return [
        _ext(f"{PACKAGE}.{libname}", os.path.join(*PACKAGE.split("."), f"{libname}.pyx")),
        _ext(f"{PACKAGE}.cy{libname}", os.path.join(*PACKAGE.split("."), f"cy{libname}.pyx")),
        _ext(
            f"{PACKAGE}._internal.{libname}",
            os.path.join(*PACKAGE.split("."), "_internal", f"{libname}_linux.pyx"),
        ),
    ]


ext_modules = [
    _ext(f"{PACKAGE}._internal.utils", os.path.join(*PACKAGE.split("."), "_internal", "utils.pyx"))
]
for libname in LIBNAMES:
    ext_modules.extend(libname_extensions(libname))


compiler_directives = {
    "embedsignature": True,
    "show_performance_hints": True,
    "freethreading_compatible": True,
}


setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives=compiler_directives,
    ),
    zip_safe=False,
)
