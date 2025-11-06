import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup, Extension


CUDA_VARIANT = "13"

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
CUDA_INC = str(Path(CUDA_HOME, "include"))
if not Path(CUDA_INC).exists():
    raise SystemExit(
        f"CUDA include directory not found: {CUDA_INC}\n"
        f"Set CUDA_HOME=/path/to/cuda"
    )

ext_modules = [
    "nccl.bindings.nccl"
]

def calculate_modules(module: str):
    module_parts = module.split(".")

    # nccl.bindings.nccl -> nccl/bindings/nccl.pyx
    lowpp_mod = module_parts.copy()
    lowpp_pyx = os.path.join(*lowpp_mod[:-1], f"{lowpp_mod[-1]}.pyx")
    lowpp_mod = ".".join(lowpp_mod)
    lowpp_ext = Extension(
        lowpp_mod,
        sources=[lowpp_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )

    # cy variant: nccl.bindings.nccl -> nccl/bindings/cynccl.pyx
    cy_mod = module_parts.copy()
    cy_mod[-1] = f"cy{cy_mod[-1]}"
    cy_mod_pyx = os.path.join(*cy_mod[:-1], f"{cy_mod[-1]}.pyx")
    cy_mod = ".".join(cy_mod)
    cy_ext = Extension(
        cy_mod,
        sources=[cy_mod_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )

    # internal variant: source is nccl_linux.pyx, but published module name is nccl.bindings._internal.nccl
    inter_mod = module_parts.copy()
    inter_mod.insert(-1, "_internal")
    inter_mod_pyx = os.path.join(*inter_mod[:-1], f"{inter_mod[-1]}_linux.pyx")
    inter_mod = ".".join(inter_mod)
    inter_ext = Extension(
        inter_mod,
        sources=[inter_mod_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )

    # internal variant: insert _internal and use utils.pyx
    inter_utils_mod = module_parts.copy()
    inter_utils_mod.insert(-1, "_internal")
    inter_utils_mod[-1] = "utils"
    inter_utils_mod_pyx = os.path.join(*inter_utils_mod[:-1], f"{inter_utils_mod[-1]}.pyx")
    inter_utils_mod = ".".join(inter_utils_mod)
    inter_utils_ext = Extension(
        inter_utils_mod,
        sources=[inter_utils_mod_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )

    return lowpp_ext, cy_ext, inter_ext, inter_utils_ext


# Note: the extension attributes are overwritten in build_extension()
ext_modules = [e for ext in ext_modules for e in calculate_modules(ext)]


compiler_directives = {"embedsignature": True, "show_performance_hints": True}


BASE_REQ = [
    "packaging",
    "numpy",
    "cuda.core==0.4.1",
]


DEPS_CU13 = BASE_REQ + [
    "nvidia-nccl-cu13",
    "cuda-python>=13.0,<14.0",
]

VARIANT_TO_REQ = {

    "13": DEPS_CU13,
}
INSTALL_REQUIRES = VARIANT_TO_REQ.get((CUDA_VARIANT or "").strip(), DEPS_CU13)


setup(
    ext_modules=cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives=compiler_directives,
    ),
    zip_safe=False,
    options={"build_ext": {"inplace": False}},
    install_requires=INSTALL_REQUIRES,
)
