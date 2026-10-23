#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.18)

project(rmsnorm_multimem LANGUAGES CXX)

if(NOT "${CMAKE_CUDA_ARCHITECTURES}" AND NOT "$ENV{CUDAARCHS}")
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

enable_language(CUDA)

option(NCCLExamples_USE_MPI "Use MPI for process coordination" OFF)

find_package(CUDAToolkit REQUIRED)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../cmake")
find_package(NCCL 2.30 REQUIRED)

add_executable(rmsnorm_multimem
    main.cu
    ../../../common/src/utils.cc
)
target_include_directories(rmsnorm_multimem PRIVATE
    ../../include
    ../../../common/include
)
target_link_libraries(rmsnorm_multimem PRIVATE NCCL::nccl)
if(NCCLExamples_USE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_compile_definitions(rmsnorm_multimem PRIVATE MPI_SUPPORT)
    target_link_libraries(rmsnorm_multimem PRIVATE MPI::MPI_CXX)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(rmsnorm_multimem PRIVATE Threads::Threads)
endif()
