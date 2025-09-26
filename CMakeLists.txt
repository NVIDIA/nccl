cmake_minimum_required(VERSION 4.0)


# === @begin: Reading version dynamically (from the file) ===
# @NVIDIA_ORIGINAL: @begin:
# @EUGO_CHANGE: We preserved this as this allows reading version dynamically and not update that manually on every upstream sync.

# Version information
# Read makefiles/version.mk file
file(READ ${CMAKE_SOURCE_DIR}/makefiles/version.mk VERSION_CONTENT)
string(REGEX REPLACE ".*NCCL_MAJOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MAJOR "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_MINOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MINOR "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_PATCH[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_PATCH "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_SUFFIX[ ]*:=[ ]*([a-zA-Z0-9]*).*" "\\1" NCCL_SUFFIX "${VERSION_CONTENT}")
string(REGEX REPLACE ".*PKG_REVISION[ ]*:=[ ]*([0-9]+).*" "\\1" PKG_REVISION "${VERSION_CONTENT}")
math(EXPR NCCL_VERSION_CODE "(${NCCL_MAJOR} * 10000) + (${NCCL_MINOR} * 100) + ${NCCL_PATCH}")

# @EUGO_CHANGE:
# 1. `NCCL_USE_CMAKE` - is only used in `@/src/device/generate.py` and `@/src/device/symmetric/generate.py`, so we shouldn't pass it as a compile definition but as an environment variable as done a few lines below!
#    1. ~~@EUGO_CHANGE: `NCCL_USE_CMAKE` -> `NCCL_USE_CMAKE=1` - this makes check work better as w/o value `-D` makes value to be defined as an empty string (yep, in C pre-processor it's distinct from undefined).~~
# 2. `NCCL_MAJOR`, `NCCL_MINOR`, `NCCL_PATCH`, and `NCCL_VERSION_CODE` are commented out here as they're set via `nccl.h` generated header and we shouldn't provide their definition twice to avoid build- and runtime ODR violation.
#
# Make version information available to C++ source files
# add_compile_definitions(
    # NCCL_USE_CMAKE=1

    # NCCL_MAJOR=${NCCL_MAJOR}
    # NCCL_MINOR=${NCCL_MINOR}
    # NCCL_PATCH=${NCCL_PATCH}
    # NCCL_VERSION_CODE=${NCCL_VERSION_CODE}
# )

# @EUGO_CHANGE:
# This is used in `@/src/device/generate.py` and ``@/src/device/symmetric/generate.py` so we preserve it.
set(ENV{NCCL_USE_CMAKE} "1")
# @NVIDIA_ORIGINAL: @end


# This is our change as their reference implementation sets this guy separately every time it's needed making it more fragile if somebody will set it differently in different use places.
set(NCCL_VERSION "${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}")
# === @end: Reading version dynamically (from the file) ===


# === @begin: General Configuration ===
# Unset CMAKE_CUDA_FLAGS to fix CMake bugs in detecting the CUDA toolchain
set(CMAKE_CUDA_FLAGS "") # @EUGO_CHANGE: Added to fix CMake bugs in detecting the CUDA toolchain

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#set(CMAKE_C_VISIBILITY_PRESET default) # Intentionally commented out
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)

# `C` language seems to not be used but we set that for the match w/ the original version (where it's used for `@/src/enhcompat.cc` which has a few functions w/ C linkage).
project(nccl
    LANGUAGES CUDA CXX C
    VERSION ${NCCL_VERSION}
)

set(CMAKE_CUDA_FLAGS
  " $ENV{CUDAFLAGS}"
)

# @EUGO_CHANGE:
# Disabled as we prefer to control warnings by using our global / shared flags.
# set(CMAKE_CXX_FLAGS
#     "${CMAKE_CXX_FLAGS} \
#     -Wall -Wvla \
#     -Wno-unused-function \
#     -Wno-sign-compare"
# )
# === @end: General Configuration ===


# === @begin: Warnings Configuration ===
set(
    EUGO_COMMON_WARNING_FLAGS
    "-Wno-deprecated-this-capture \
    -Wno-null-conversion \
    -Wno-deprecated-enum-enum-conversion \
    -Wno-deprecated-volatile"
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EUGO_COMMON_WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EUGO_COMMON_WARNING_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${EUGO_COMMON_WARNING_FLAGS}")
# === @end: Warnings Configuration ===


# === @begin: Options ===
set(NCCL_COMMON_COMPILE_DEFINITIONS
    # VERBOSE=1 # "Enable verbose output" [there is no associated C/C++ macro], disabled as this only applies to Make/CMake verbosity in original implementation
    # KEEP=1 # "Keep intermediate files" [there is no associated C/C++ macro], disabled as this only applies to original implementation and its Make/CMake execution
    # ASAN=1 # "Enable Address Sanitizer" [there is no associated C/C++ macro], not yet supported in Eugo environment
    # UBSAN=1 # "Enable Undefined Behavior Sanitizer" [there is no associated C/C++ macro], not yet supported in Eugo environment

    # WERROR=1 # "Treat warnings as errors" [there is no associated C/C++ macro]
    # We're not reimplementing this as it repeats the built-in CMake flags:
    # 1. `CMAKE_COMPILE_WARNING_AS_ERROR`
    #    1. https://cmake.org/cmake/help/latest/prop_tgt/COMPILE_WARNING_AS_ERROR.html
    #    2. https://cmake.org/cmake/help/latest/variable/CMAKE_COMPILE_WARNING_AS_ERROR.html
    # 2. `CMAKE_LINK_WARNING_AS_ERROR`
    #    1. https://cmake.org/cmake/help/latest/prop_tgt/LINK_WARNING_AS_ERROR.html
    #    2. https://cmake.org/cmake/help/latest/variable/CMAKE_LINK_WARNING_AS_ERROR.html

    PROFAPI=1 # "Enable profiling API", @Important: this is not about "NCCL's API for profiling" but about "profiling NCCL's own API".
    # If enabled, this causes every public NCCL function to be emitted as a weak symbol instead of a strong one w/ an extra strong alias symbol prefixed w/ `p`.
    # E.g., instead of having just `strong` `ncclCommInitAll`, w/ `PROFAPI` enabled we'll have both `weak` `ncclCommInitAll` symbol and `strong` alias-symbol `pncclCommInitAll` pointing to `ncclCommncclCommInitAllInit`.
    # This, in theory, should allow redirecting calls to the `NCCL_API`-marked functions to the ad-hoc stubs for profiling purposes w/o losing the access to original symbol.
    # Yet, I'm not aware of the public code relying on that + I'm not sure if this is correct - usually, you define `alias` symbol to be `weak` and not its target, but who knows what they've meant here ... It's not the first confusing design choice we've seen so far.
    # This setting is enabled in reference `NCCL` builds so we keep it too.
    #
    # References:
    # 1. @/src/include/core.h#L17-32
    # 2. https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-alias-function-attribute

    # NVTX=1 # "Enable NVTX" [there is no associated C/C++ macro]. In reference implementation. this is only used in negated logic - if this option is disabled, it sets `NVTX_DISABLE` macro, which causes vendored `native/cuda_nvtx` headers to contain no compilable symbols (some NVTX macros are used by NCCL regardless of this setting).

    # NCCL_ENABLE_NET_PROFILING=1 # "Enable network profiler"  [option: `NET_PROFILER`], disabled by default + causes building an extra plugin which isn't easy to build and maintain + all 3 shipped plugins are considered tests/examples of NCCL usage and not something NCCL really needs to operate properly.
    # NCCL_NET_MAX_PLUGINS=0 # "Maximum external network plugins" [option: `MAX_EXT_NET_PLUGINS=<int>`], this is set to `0` by default, meaning no extra plugins can be loaded, which makes sense as we're not building any of them.
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND NCCL_COMMON_COMPILE_DEFINITIONS
        # DEBUG=1 # "Enable debug build" [there is no associated C/C++ macro], this implies the following individual macros:
        DEBUG_EVENTS=1
        DEBUG_PROXY=1
        NVTX_DEBUG_PRINT=1 # This macro isn't used in `native/cuda_nccl`, but in its vendored `native/cuda_nvtx` headers.

        # TRACE=1 # "Enable tracing" [there is no associated C/C++ macro], this implies the following individual macros:
        ENABLE_TRACE=1
    )
endif()
# === @end: Options ===


# === @begin: dependencies/ ===
# === @@begin: CUDAToolkit ===
find_package(CUDAToolkit REQUIRED)
message(STATUS "[EUGO]: Find CUDAToolkit: ${CUDAToolkit_CONFIG}")
# === @@end: CUDAToolkit ===


# === @@begin: CCCL ===
# This is also part of `CUDAToolkit` (`native/cuda`) but it's not found by `FindCUDAToolkit` as it's open-source and header-only library + this may even be installed separately.
# Hence, we search it separately and use only the portion we need.
find_package(cccl CONFIG REQUIRED)
message(STATUS "[EUGO]: CCCL found: ${cccl_CONFIG}")
# === @@end: CCCL ===


# === @@begin: glibc-merged libs ===
# In reference implementation, all `native/cuda_nccl` are linked against `librt` (realtime, not runtime!), `libpthread`, and `libdl`.
# However, in modern `glibc` versions they're all merged into main `libc.so` file and dedicated binaries are simply compatability stubs.
# Hence, we don't have to lookup and link them explicitely.
# === @@end: glibc-merged libs ===


# === @@begin: Discarded dependencies ===
# option(RDMA_CORE "Enable RDMA core" OFF) # Sets `NCCL_BUILD_RDMA_CORE=1` macro and links `libibverbs` (`native/rdma_core`)
# option(MLX5DV "Enable MLX5DV" OFF) # Sets `NCCL_BUILD_MLX5DV=1` macro and links `libmlx5` (`native/rdma_core`)
# === @@end: Discarded dependencies ===


# === @@begin: Finalized dependencies list ===
# @EUGO_CHANGE:
# In original implementation, they link their binaries against all `CUDAToolkit_LIBRARIES` which is an overkill. Based on our research, they only use `cudart` functions here to manage the GPU on the host side and directly access intrinsics (like `fp8`) on the device one.
# They also rely on `CCCL` (in particular, `libcu++`/`libcudacxx`) but as it's the header only library, they just add that into include dirs list and don't explicitely search for it.
#
# In our NCCL implementation, we limit dependencies list to that is exactly used to avoid too tight dependency graph and reduce the number of unneeded libraries loaded at runtime.
#
# In theory, the best solution here is to check every built component (as we have 1 library, 1 (buildtime-only) object library, and 1 executable in this package) and determine on which CUDA parts they depend but it may be overly complicated due to how messy the headers chain in NCCL.
# As after the build we're getting only 2 binaries - `libnccl` (library) and `ncclras` (executable) - and the latter links against former, it's safe and sound to define just a single list of CUDA dependencies for all of them.
#
# @Important:
# Based on our research, NCCL doesn't directly depend on `libcuda` (`CUDA::driver_library`) so we don't add it here (or `cudadevrt`).
set(NCCL_DEPENDENCIES CCCL::libcudacxx CUDA::cudart)
# === @@end: Finalized dependencies list ===
# === @end: dependencies/ ===


# === @begin: src/ ===
add_subdirectory(src)
# === @end: src/ ===


# === @begin: Install and Configuration Targets ===
include(GNUInstallDirs)

# Install the headers into the include directory
install(FILES ${NCCL_PUBLIC_HEADERS} TYPE INCLUDE)

install(
    TARGETS nccl ncclras # `nccl_colldevice` isn't installed separately but used as a private implementation detail for `nccl`
    EXPORT NCCLConfig
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# @HELP: Implement pkg-config guy and adjust our cmake stuff in a compatible way!
install(
    EXPORT NCCLConfig
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nccl
    NAMESPACE NCCL::
)

# Configure pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nccl.pc.in
    ${CMAKE_BINARY_DIR}/lib/pkgconfig/nccl.pc
    @ONLY
)


# NOTE: This is the install target output!
# -- Installing: /usr/local/include/nccl.h
# -- Up-to-date: /usr/local/include/nccl_net.h
# -- Installing: /usr/local/lib64/libnccl.so.2.16.2.1
# -- Installing: /usr/local/lib64/libnccl.so.2
# -- Installing: /usr/local/lib64/libnccl.so
# -- Installing: /usr/local/bin/ncclras
# -- Installing: /usr/local/lib64/cmake/nccl/NCCLConfig.cmake
# -- Installing: /usr/local/lib64/cmake/nccl/NCCLConfig-release.cmake
# === @end: Install Targets ===
