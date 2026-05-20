import os
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup, Extension


_NCCL_EP_SO = Path(__file__).parent / "nccl" / "ep" / "lib" / "libnccl_ep.so"
if not _NCCL_EP_SO.exists():
    print(
        f"WARNING: {_NCCL_EP_SO} not found. The built nccl4py wheel will not "
        "include the NCCL EP shared library, and `import nccl.ep` will fail at "
        "runtime. Drop a CUDA 13 build of libnccl_ep.so at that path before "
        "building the wheel.",
        file=sys.stderr,
    )


# Check CUDA_HOME is set and is a valid directory
CUDA_HOME = os.environ.get("CUDA_HOME")
if not CUDA_HOME:
    raise SystemExit("Error: CUDA_HOME is not set")

cuda_path = Path(CUDA_HOME)
if not cuda_path.exists() or not cuda_path.is_dir():
    raise SystemExit(f"Error: CUDA_HOME does not exist or is not a directory: {CUDA_HOME}")
CUDA_INC = str(cuda_path / "include")

PACKAGE = "nccl.bindings"
LIBNAMES = ["nccl", "nccl_ep"]


def _ext(module: str, source: str) -> Extension:
    return Extension(
        module,
        sources=[source],
        include_dirs=[CUDA_INC],
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
    ext_modules=cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives=compiler_directives,
    ),
    zip_safe=False,
    options={"build_ext": {"inplace": False}},
)
