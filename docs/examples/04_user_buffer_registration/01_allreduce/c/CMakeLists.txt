#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.18)
project(allreduce_ub LANGUAGES CXX)

option(NCCLExamples_USE_MPI "Use MPI for process coordination" OFF)

find_package(CUDAToolkit REQUIRED)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../cmake")
find_package(NCCL 2.19 REQUIRED)

add_executable(allreduce_ub
    main.cc
    ../../../common/src/utils.cc
)
target_include_directories(allreduce_ub PRIVATE
    ../../../common/include
)
target_link_libraries(allreduce_ub PRIVATE
    CUDA::cudart
    NCCL::nccl
)
if(NCCLExamples_USE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_compile_definitions(allreduce_ub PRIVATE MPI_SUPPORT)
    target_link_libraries(allreduce_ub PRIVATE MPI::MPI_CXX)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(allreduce_ub PRIVATE Threads::Threads)
endif()
