#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.18)

project(NCCLExamples)

if(NOT "${CMAKE_CUDA_ARCHITECTURES}" AND NOT "$ENV{CUDAARCHS}")
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endif()
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

# Map Make options to their canonical CMake equivalents.
macro(nccl_examples_cmake_from_make_var CMAKE_VAR MAKE_VAR)
    if(NOT DEFINED ${CMAKE_VAR})
        if(DEFINED ENV{${CMAKE_VAR}})
            set(${CMAKE_VAR} "$ENV{${CMAKE_VAR}}")
        elseif(DEFINED ${MAKE_VAR})
            set(${CMAKE_VAR} "${${MAKE_VAR}}")
        elseif(DEFINED ENV{${MAKE_VAR}})
            set(${CMAKE_VAR} "$ENV{${MAKE_VAR}}")
        endif()
    endif()

    unset(ENV{${CMAKE_VAR}})
    unset(${MAKE_VAR})
    unset(${MAKE_VAR} CACHE)
    unset(ENV{${MAKE_VAR}})
endmacro()

nccl_examples_cmake_from_make_var(NCCL_ROOT NCCL_HOME)
nccl_examples_cmake_from_make_var(MPI_ROOT MPI_HOME)
nccl_examples_cmake_from_make_var(CUDAToolkit_ROOT CUDA_HOME)

nccl_examples_cmake_from_make_var(mpi_default MPI)
if(NOT DEFINED mpi_default)
    set(mpi_default OFF)
endif()
option(NCCLExamples_USE_MPI "Enable MPI examples" "${mpi_default}")
unset(mpi_default)

add_subdirectory(01_communicators)
add_subdirectory(02_point_to_point)
add_subdirectory(03_collectives)
add_subdirectory(04_user_buffer_registration)
add_subdirectory(05_symmetric_memory)
add_subdirectory(06_device_api)
add_subdirectory(07_kernel_fusion)
