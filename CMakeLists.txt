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
# 1. `NCCL_USE_CMAKE` -> `NCCL_USE_CMAKE=1` - this makes check work better as w/o value `-D` makes value to be defined as an empty string (yep, in C pre-processor it's distinct from undefined).
# 2. @HELP(check if we actually need this or not - or probably we shouldn't pass that at all here? as we put that into `nccl.h`) Added `NCCL_SUFFIX` to match other variables.
#
# Make version information available to C++ source files
add_compile_definitions(
    NCCL_USE_CMAKE=1
    NCCL_MAJOR=${NCCL_MAJOR}
    NCCL_MINOR=${NCCL_MINOR}
    NCCL_PATCH=${NCCL_PATCH}
    NCCL_VERSION_CODE=${NCCL_VERSION_CODE}
)

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

# `C` language seems to not be used but we set that for the match w/ the original version (where it's used for `@/src/enhcompat.cc` which has a few functions w/ C linkage).
project(nccl
    LANGUAGES CUDA CXX C
    VERSION ${NCCL_VERSION}
)

# @TODO+:return to 23, see other issue blah-blah (can delete as soon as we confirm works)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Reset CMAKE_CUDA_FLAGS to fix CMake bugs in detecting the CUDA toolchain
string(APPEND CMAKE_CUDA_FLAGS
  " $ENV{CUDAFLAGS}"
)

# @HELP: Probably, we can remove them now
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} \
    -Wall -Wvla \
    -Wno-unused-function \
    -Wno-sign-compare"
)
# === @end: General Configuration ===


# === @begin: Options ===
option(VERBOSE "Enable verbose output" OFF)
option(KEEP "Keep intermediate files" OFF)
option(DEBUG "Enable debug build" OFF)
# option(ASAN "Enable Address Sanitizer" OFF) -> not reimplementing as no value for Eugo right now
# option(UBSAN "Enable Undefined Behavior Sanitizer" OFF) -> not reimplementing as no value for Eugo right now
option(TRACE "Enable tracing" OFF)
# option(WERROR "Treat warnings as errors" OFF) -> not reimplementing as no value for Eugo right now
option(PROFAPI "Enable profiling API" ON)
option(NVTX "Enable NVTX" ON)
option(RDMA_CORE "Enable RDMA core" OFF)
option(NET_PROFILER "Enable network profiler" OFF)
option(MLX5DV "Enable MLX5DV" OFF)
option(MAX_EXT_NET_PLUGINS "Maximum external network plugins" 0)
# === @end: Options ===


# === @begin: dependencies/ ===
# === @@begin: CUDAToolkit ===
find_package(CUDAToolkit REQUIRED)
message(STATUS "[EUGO]: Find CUDAToolkit: ${CUDAToolkit_CONFIG}")


# @NVIDIA_ORIGINAL: @begin:
# @EUGO_CHANGE: We preserved this snippet as they rely on `CUDA_MAJOR` and `CUDA_MINOR` definitions in their code.

# CUDA version detection
string(REGEX MATCH "([0-9]+\\.[0-9]+)" CUDA_VERSION "${CUDAToolkit_VERSION}")

# Extract major and minor version numbers
string(REGEX MATCH "([0-9]+)" CUDA_MAJOR "${CUDA_VERSION}")
string(REGEX MATCH "([0-9]+)$" CUDA_MINOR "${CUDA_VERSION}")
string(REGEX REPLACE ".*\\.([0-9]+)$" "\\1" CUDA_MINOR "${CUDA_VERSION}")

# Add CUDA version definitions after find_package
add_compile_definitions(
    CUDA_MAJOR=${CUDA_MAJOR}
    CUDA_MINOR=${CUDA_MINOR}
)
# @NVIDIA_ORIGINAL: @end
# === @@end: CUDAToolkit ===

# === @@begin: CCCL ===
# This is also part of `CUDAToolkit` (`native/cuda`) but it's not found by `FindCUDAToolkit` as it's open-source and header-only library + this may even be installed separately.
# Hence, we search it separately and use only the portion we need.
find_package(cccl CONFIG REQUIRED)
message(STATUS "[EUGO]: CCCL found: ${cccl_CONFIG}")
# === @@end: CCCL ===
# === @end: dependencies/ ===


# === @begin: src/ ===
add_subdirectory(src)
# === @end: src/ ===


# === @begin: Install and Configuration Targets ===
include(GNUInstallDirs)

# @HELP: Probably do the same as we did for `python/torch`?
# Install the headers into the include directory
install(FILES ${INCEXPORTS} TYPE INCLUDE)

install(
    TARGETS nccl ncclras
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
