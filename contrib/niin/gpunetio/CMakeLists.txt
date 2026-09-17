# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(niin_gpunetio_atomics LANGUAGES CXX CUDA)

option(NIIN_GPUNETIO_BUILD_DEVICE_COMPILE_TEST
       "Compile the NIIN public atomic API against GPUNetIO device headers" ON)
option(NIIN_GPUNETIO_BUILD_SMOKE_TEST
       "Build the MPI/Slurm multi-node direct-atomic smoke test when MPI is available" ON)
option(NIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS
       "Build NIIN's experimental raw-verbs/GDRCopy SRQ AMO prototype" OFF)

set(NCCL_HOME "" CACHE PATH "NCCL build or install prefix")
set(NCCL_SOURCE_ROOT "" CACHE PATH "NCCL source checkout (needed only for its vendored GPUNetIO tree)")
set(GPUNETIO_HOME "" CACHE PATH "Unmodified GPUNetIO Open source tree")
set(GDRCOPY_INCLUDE_DIR "" CACHE PATH "Directory containing gdrapi.h (needed only for SRQ proxy atomics)")

if(NOT GPUNETIO_HOME OR NOT EXISTS "${GPUNETIO_HOME}/include")
  message(FATAL_ERROR
    "Set GPUNETIO_HOME to an unmodified GPUNetIO Open source tree. "
    "It may be a standalone tree with CMakeLists.txt or NCCL's vendored tree; "
    "NIIN builds it as a customer dependency and never patches it.")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# CMake's imported CUDA targets normally propagate this directory.  Some
# relocatable toolkit installations report the runtime library correctly but
# leave CUDAToolkit_INCLUDE_DIRS empty for C++ object targets, while GPUNetIO's
# host sources include cuda_runtime.h directly.  Derive the canonical include
# directory from nvcc only in that case.
if(NOT CUDAToolkit_INCLUDE_DIRS)
  get_filename_component(NIIN_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(NIIN_CUDA_TOOLKIT_ROOT "${NIIN_CUDA_BIN_DIR}" DIRECTORY)
  set(CUDAToolkit_INCLUDE_DIRS "${NIIN_CUDA_TOOLKIT_ROOT}/include")
endif()
if(NOT EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cuda_runtime.h")
  message(FATAL_ERROR "Could not locate cuda_runtime.h below the selected CUDA toolkit")
endif()

find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS "${NCCL_HOME}/build/include" "${NCCL_HOME}/include"
  REQUIRED)
find_path(NCCL_DEVICE_INCLUDE_DIR nccl_device.h
  HINTS "${NCCL_HOME}/build/include" "${NCCL_HOME}/include"
  REQUIRED)
find_library(NCCL_LIBRARY NAMES nccl
  HINTS "${NCCL_HOME}/build/lib" "${NCCL_HOME}/lib" "${NCCL_HOME}/lib64"
  REQUIRED)
find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h REQUIRED)
find_library(IBVERBS_LIBRARY NAMES ibverbs REQUIRED)

# NIIN directly emits GPUNetIO WQEs, while the surrounding NIIN headers also
# include NCCL's vendored GPUNetIO device ABI through nccl_device.h.  The two
# must be exactly the same provider release: mixing their identically named
# device structs/functions is an ABI bug even when both headers compile.
set(NCCL_GPUNETIO_VERSION_HEADER
    "${NCCL_DEVICE_INCLUDE_DIR}/nccl_device/gin/gdaki/doca_gpunetio/common/doca_gpunetio_verbs_def.h")
if(EXISTS "${GPUNETIO_HOME}/include/gpunetio/common/doca_gpunetio_verbs_def.h")
  set(NIIN_GPUNETIO_VERSION_HEADER
      "${GPUNETIO_HOME}/include/gpunetio/common/doca_gpunetio_verbs_def.h")
elseif(EXISTS "${GPUNETIO_HOME}/include/common/doca_gpunetio_verbs_def.h")
  set(NIIN_GPUNETIO_VERSION_HEADER
      "${GPUNETIO_HOME}/include/common/doca_gpunetio_verbs_def.h")
else()
  message(FATAL_ERROR "Could not locate GPUNetIO's public device-version header")
endif()
if(NOT EXISTS "${NCCL_GPUNETIO_VERSION_HEADER}")
  message(FATAL_ERROR
    "NCCL's generated GPUNetIO device headers are missing from ${NCCL_DEVICE_INCLUDE_DIR}; "
    "build NCCL before configuring this optional NIIN provider")
endif()

function(niin_read_gpunetio_version component header output)
  file(STRINGS "${header}" version_line
       REGEX "^#define[ \t]+DOCA_GPUNETIO_VERSION_${component}[ \t]+[0-9]+")
  string(REGEX REPLACE ".*[^0-9]([0-9]+)$" "\\1" version_value "${version_line}")
  if(NOT version_value MATCHES "^[0-9]+$")
    message(FATAL_ERROR "Could not read GPUNetIO ${component} version from ${header}")
  endif()
  set(${output} "${version_value}" PARENT_SCOPE)
endfunction()

foreach(component MAJOR MINOR PATCH)
  niin_read_gpunetio_version("${component}" "${NCCL_GPUNETIO_VERSION_HEADER}" nccl_version)
  niin_read_gpunetio_version("${component}" "${NIIN_GPUNETIO_VERSION_HEADER}" niin_version)
  if(NOT nccl_version STREQUAL niin_version)
    message(FATAL_ERROR
      "GPUNetIO ${component} version mismatch: NCCL device ABI is ${nccl_version}, "
      "but GPUNETIO_HOME supplies ${niin_version}. Use the exact provider release NCCL was built with.")
  endif()
endforeach()

get_filename_component(NIIN_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(NIIN_INCLUDE_DIR "${NIIN_ROOT}/include")

# A public GPUNetIO Open checkout is normally a CMake subproject. Current
# NCCL dev instead vendors the same provider sources directly and does not
# ship their CMakeLists.txt. Both are unmodified customer dependencies; for
# the latter NIIN owns only a tiny object-library wrapper around NCCL's own
# canonical source list. This does not expose an NCCL/GDAKI API or modify the
# vendored provider.
set(NIIN_GPUNETIO_PROVIDER_OBJECTS)
if(EXISTS "${GPUNETIO_HOME}/CMakeLists.txt")
  # GPUNetIO Open uses DOCA_HOME as its source-root input. Keep it local to
  # configuring the external subproject rather than relying on it elsewhere.
  set(DOCA_HOME "${GPUNETIO_HOME}")
  add_subdirectory("${GPUNETIO_HOME}" "${CMAKE_CURRENT_BINARY_DIR}/gpunetio-open")
  unset(DOCA_HOME)
  # Some relocatable CUDA installations leave the imported cudart target
  # without its include directory even though nvcc and the runtime library
  # were found. GPUNetIO's object target includes cuda_runtime.h directly, so
  # make the canonical toolkit include path explicit at this consumer
  # boundary. This augments the target; it does not modify GPUNetIO.
  target_include_directories(doca_gpunetio PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
  list(APPEND NIIN_GPUNETIO_PROVIDER_OBJECTS $<TARGET_OBJECTS:doca_gpunetio>)
else()
  if(NOT NCCL_SOURCE_ROOT AND EXISTS "${NCCL_HOME}/src/include/compiler.h")
    set(NCCL_SOURCE_ROOT "${NCCL_HOME}")
  endif()
  if(NOT NCCL_SOURCE_ROOT OR NOT EXISTS "${NCCL_SOURCE_ROOT}/src/include/compiler.h")
    message(FATAL_ERROR
      "GPUNETIO_HOME has no CMakeLists.txt, so it is treated as NCCL's vendored provider. "
      "Set NCCL_SOURCE_ROOT to the matching NCCL checkout (it supplies src/include/compiler.h).")
  endif()
  if(EXISTS "${IBVERBS_INCLUDE_DIR}/infiniband/mlx5dv.h")
    set(MLX5DV_INCLUDE_DIR "${IBVERBS_INCLUDE_DIR}")
  else()
    find_path(MLX5DV_INCLUDE_DIR infiniband/mlx5dv.h REQUIRED)
  endif()

  set(NIIN_GPUNETIO_OPEN_SOURCES
    doca_verbs_qp.cpp
    doca_verbs_cq.cpp
    doca_verbs_device_attr.cpp
    doca_verbs_umem.cpp
    doca_verbs_srq.cpp
    doca_verbs_uar.cpp
    doca_verbs_dev.cpp
    doca_gpunetio.cpp
    doca_gpunetio_log.cpp
    doca_gpunetio_high_level.cpp
    doca_gpunetio_cuda_wrapper.cpp
    doca_verbs_mlx5dv_wrapper.cpp
    doca_verbs_ibv_wrapper.cpp
    doca_gpunetio_gdrcopy.cpp
    doca_gpunetio_sdk_wrapper.cpp
    doca_verbs_dev_sdk_wrapper.cpp
    doca_verbs_umem_sdk_wrapper.cpp
    doca_verbs_uar_sdk_wrapper.cpp
    doca_verbs_cq_sdk_wrapper.cpp
    doca_verbs_qp_sdk_wrapper.cpp)
  list(TRANSFORM NIIN_GPUNETIO_OPEN_SOURCES PREPEND "${GPUNETIO_HOME}/src/")

  add_library(niin_gpunetio_open_obj OBJECT ${NIIN_GPUNETIO_OPEN_SOURCES})
  set_target_properties(niin_gpunetio_open_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_features(niin_gpunetio_open_obj PRIVATE cxx_std_17)
  target_compile_definitions(niin_gpunetio_open_obj PRIVATE
    DOCA_GPUNETIO_USE_CUDA_WRAPPER
    DOCA_VERBS_USE_NET_WRAPPER)
  target_include_directories(niin_gpunetio_open_obj PRIVATE
    "${GPUNETIO_HOME}/include"
    "${GPUNETIO_HOME}/src"
    "${NCCL_SOURCE_ROOT}/src/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${IBVERBS_INCLUDE_DIR}"
    "${MLX5DV_INCLUDE_DIR}")
  target_link_libraries(niin_gpunetio_open_obj PRIVATE
    CUDA::cudart
    Threads::Threads
    ${CMAKE_DL_LIBS})
  list(APPEND NIIN_GPUNETIO_PROVIDER_OBJECTS $<TARGET_OBJECTS:niin_gpunetio_open_obj>)
endif()

set(NIIN_GPUNETIO_ATOMIC_SOURCES src/atomics_host.cc)
if(NIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS)
  find_path(GDRCOPY_INCLUDE_DIR gdrapi.h
    HINTS "${GDRCOPY_INCLUDE_DIR}"
          "$ENV{GDRCOPY_HOME}/include"
          "$ENV{GDRCOPY_INCLUDE_DIR}"
    REQUIRED)
  # The proxy is deliberately a NIIN source directory rather than a GPUNetIO
  # extension. It dynamically loads libgdrapi at runtime, so no GDRCopy
  # library is linked into ordinary direct-QP applications.
  list(APPEND NIIN_GPUNETIO_ATOMIC_SOURCES proxy_atomics/src/proxy_atomics_host.cc)
endif()

add_library(niin_gpunetio_atomics STATIC
  ${NIIN_GPUNETIO_ATOMIC_SOURCES}
  ${NIIN_GPUNETIO_PROVIDER_OBJECTS})

target_compile_features(niin_gpunetio_atomics PUBLIC cxx_std_17)
target_compile_definitions(niin_gpunetio_atomics PUBLIC NIIN_GPUNETIO_ENABLE=1)
if(NIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS)
  target_compile_definitions(niin_gpunetio_atomics PUBLIC NIIN_GPUNETIO_SRQ_PROXY_ATOMICS_ENABLE=1)
endif()
target_include_directories(niin_gpunetio_atomics
  PUBLIC
    "${NIIN_INCLUDE_DIR}"
    # `nvshmem.h`/NIIN device headers include NCCL's public device ABI, so a
    # CMake consumer must see the same generated NCCL headers while compiling
    # CUDA call sites, not only while compiling this archive's host source.
    "${NCCL_INCLUDE_DIR}"
    "${NCCL_DEVICE_INCLUDE_DIR}"
    # Consumers compile the header-only device AMO dispatch, so GPUNetIO's
    # public device headers are part of this target's public build interface.
    "${GPUNETIO_HOME}/include"
  PRIVATE
    "${IBVERBS_INCLUDE_DIR}")
if(NIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS)
  target_include_directories(niin_gpunetio_atomics PRIVATE "${GDRCOPY_INCLUDE_DIR}")
endif()
target_link_libraries(niin_gpunetio_atomics
  PUBLIC
    "${NCCL_LIBRARY}"
  PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    "${IBVERBS_LIBRARY}"
    Threads::Threads
    dl)

# This is compile-only by design. It instantiates every current public NIIN
# atomic API at both 4 and 8 bytes, including the masked/extended WQEs, but
# does not require a NIC or launch a kernel during a normal build.
if(NIIN_GPUNETIO_BUILD_DEVICE_COMPILE_TEST)
  add_library(niin_gpunetio_atomic_device_compile OBJECT tests/device_compile.cu)
  target_compile_features(niin_gpunetio_atomic_device_compile PRIVATE cxx_std_17)
  target_compile_definitions(niin_gpunetio_atomic_device_compile PRIVATE NIIN_GPUNETIO_ENABLE=1)
  target_include_directories(niin_gpunetio_atomic_device_compile PRIVATE
    "${NIIN_INCLUDE_DIR}"
    "${NCCL_INCLUDE_DIR}"
    "${NCCL_DEVICE_INCLUDE_DIR}"
    "${GPUNETIO_HOME}/include")
  set_target_properties(niin_gpunetio_atomic_device_compile PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  add_custom_target(niin_gpunetio_compile_check
    DEPENDS niin_gpunetio_atomics niin_gpunetio_atomic_device_compile)
endif()

if(NIIN_GPUNETIO_BUILD_SMOKE_TEST)
  find_package(MPI QUIET COMPONENTS CXX)
  if(MPI_CXX_FOUND)
    add_executable(niin_gpunetio_atomic_smoke tests/atomic_smoke.cu)
    target_compile_features(niin_gpunetio_atomic_smoke PRIVATE cxx_std_17)
    target_link_libraries(niin_gpunetio_atomic_smoke PRIVATE
      niin_gpunetio_atomics
      MPI::MPI_CXX
      CUDA::cudart)
    set_target_properties(niin_gpunetio_atomic_smoke PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    if(NIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS)
      add_executable(niin_gpunetio_proxy_atomic_smoke proxy_atomics/tests/proxy_atomic_smoke.cu)
      target_compile_features(niin_gpunetio_proxy_atomic_smoke PRIVATE cxx_std_17)
      target_link_libraries(niin_gpunetio_proxy_atomic_smoke PRIVATE
        niin_gpunetio_atomics
        MPI::MPI_CXX
        CUDA::cudart)
      set_target_properties(niin_gpunetio_proxy_atomic_smoke PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
    endif()
  else()
    message(STATUS "MPI C++ was not found; skipping niin_gpunetio_atomic_smoke")
  endif()
endif()
