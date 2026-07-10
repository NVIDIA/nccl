# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# tests/CMakeLists.txt
#
# Picked up by the root CMakeLists.txt when -DNCCL_M2N_BUILD_TESTS=ON.
# Builds:
#   basic_api_test_mpi    — MPI bootstrap, links libnccl_m2n.so
#   basic_api_test_local  — pthread (no MPI), links libnccl_m2n.so
#   unit_tests            — gtest, links library SRCS directly so the
#                           tests exercise the same TUs (matches CI's
#                           tests/ci/Makefile linkage choice).
#
# GTest discovery, in order:
#   1) GTEST_DIR override (set -DGTEST_DIR=/path/to/googletest source) →
#      add_subdirectory($GTEST_DIR ...).
#   2) Vendored copy at tests/googletest/ → add_subdirectory().
#   3) Skip the test build with a clear message.
#
# MPI is required for basic_api_test_mpi; basic_api_test_local builds even
# when MPI is absent.

find_package(Threads REQUIRED)

# ─── GTest ────────────────────────────────────────────────────────────────────
# Mirror tests/gtest.mk: don't rely on vendored googletest's own CMakeLists
# (older copy, breaks against newer cmake's CMP0148 + missing PRIVATE include
# on gtest-all.cc). Resolve the source tree, then compile gtest-all.cc +
# gtest_main.cc into static libs ourselves — matches the Makefile path 1:1.

set(_NCCL_M2N_GTEST_DIR "")
if(GTEST_DIR AND EXISTS "${GTEST_DIR}/include/gtest/gtest.h")
  set(_NCCL_M2N_GTEST_DIR "${GTEST_DIR}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest/include/gtest/gtest.h")
  set(_NCCL_M2N_GTEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/googletest")
endif()

if(NOT _NCCL_M2N_GTEST_DIR)
  message(STATUS
    "nccl_m2n tests/: googletest not found — skipping basic_api + unit_tests. "
    "Pass -DGTEST_DIR=/path/to/googletest or vendor a copy at tests/googletest/.")
  return()
endif()

add_library(nccl_m2n_gtest STATIC
    "${_NCCL_M2N_GTEST_DIR}/src/gtest-all.cc"
)
target_include_directories(nccl_m2n_gtest
    PUBLIC  "${_NCCL_M2N_GTEST_DIR}/include"
    PRIVATE "${_NCCL_M2N_GTEST_DIR}"
)
target_compile_options(nccl_m2n_gtest PRIVATE -Wno-uninitialized)
set_target_properties(nccl_m2n_gtest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

add_library(nccl_m2n_gtest_main STATIC
    "${_NCCL_M2N_GTEST_DIR}/src/gtest-all.cc"
    "${_NCCL_M2N_GTEST_DIR}/src/gtest_main.cc"
)
target_include_directories(nccl_m2n_gtest_main
    PUBLIC  "${_NCCL_M2N_GTEST_DIR}/include"
    PRIVATE "${_NCCL_M2N_GTEST_DIR}"
)
target_compile_options(nccl_m2n_gtest_main PRIVATE -Wno-uninitialized)
set_target_properties(nccl_m2n_gtest_main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

add_library(gtest      ALIAS nccl_m2n_gtest)
add_library(gtest_main ALIAS nccl_m2n_gtest_main)

# ─── MPI (optional) ───────────────────────────────────────────────────────────
find_package(MPI)

# ─── basic_api_test_local (no MPI) ────────────────────────────────────────────
add_executable(basic_api_test_local
    basic_api_test_local.cc
    test_helpers.cu
)
target_include_directories(basic_api_test_local PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/include
)
target_include_directories(basic_api_test_local SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
    ${_NCCL_M2N_NCCL_INCLUDES}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(basic_api_test_local PRIVATE
    nccl_m2n_shared
    ${_NCCL_M2N_NCCL_TARGET}
    CUDA::cudart
    gtest gtest_main
    Threads::Threads
)

# ─── basic_api_test_mpi (only when MPI present) ───────────────────────────────
if(MPI_FOUND)
  add_executable(basic_api_test_mpi
      basic_api_test_mpi.cc
      test_helpers.cu
  )
  target_include_directories(basic_api_test_mpi PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_BINARY_DIR}/include
  )
  target_include_directories(basic_api_test_mpi SYSTEM PRIVATE
      ${CMAKE_SOURCE_DIR}/third_party
      ${_NCCL_M2N_NCCL_INCLUDES}
      ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(basic_api_test_mpi PRIVATE
      nccl_m2n_shared
      ${_NCCL_M2N_NCCL_TARGET}
      CUDA::cudart
      gtest gtest_main
      MPI::MPI_CXX
      Threads::Threads
  )
else()
  message(STATUS "nccl_m2n tests/: MPI not found — skipping basic_api_test_mpi.")
endif()

# ─── tests/ci/unit_tests ──────────────────────────────────────────────────────
# Links library SRCS directly (not the .so) so the gtests exercise the same
# implementation without a separate test-only build artifact. Matches CI's
# tests/ci/Makefile linkage and the compile_commands.json layout consumed by
# lint:tidy.
set(CI_UNIT_TEST_SRCS
    ci/unit_tests_main.cc
    ci/unit/config_test.cc
    ci/unit/loadbalance_test.cc
    ci/unit/log_test.cc
    ci/unit/mesh_test.cc
    ci/unit/negative_path_test.cc
    ci/unit/offset_check_test.cc
    ci/unit/transpose_decision_test.cc
)

add_executable(unit_tests
    ${CI_UNIT_TEST_SRCS}
    ${NCCL_M2N_SRC_FULL_PATHS}
)
target_include_directories(unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(unit_tests SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
    ${_NCCL_M2N_NCCL_INCLUDES}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(unit_tests PRIVATE NCCL_OS_LINUX)
target_compile_options(unit_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_link_libraries(unit_tests PRIVATE
    ${_NCCL_M2N_NCCL_TARGET}
    CUDA::cudart
    CUDA::cuda_driver
    gtest gtest_main
    Threads::Threads
)

set_target_properties(basic_api_test_local unit_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

if(MPI_FOUND)
  set_target_properties(basic_api_test_mpi PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
      INSTALL_RPATH "$ORIGIN/../lib"
  )
endif()

# CTest registration (`cmake --build build && ctest --test-dir build`).
enable_testing()
add_test(NAME basic_api_test_local
         COMMAND basic_api_test_local
         WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME unit_tests_host
         COMMAND unit_tests --gtest_filter='HostOnly*'
         WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
# basic_api_test_mpi + unit_tests GPU subset live behind salloc/mpirun
# in CI — not registered as plain ctest entries.
