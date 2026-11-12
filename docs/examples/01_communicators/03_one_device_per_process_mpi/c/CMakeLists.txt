#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.18)
project(one_device_per_process_mpi LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../cmake")
find_package(NCCL 2.14 REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(one_device_per_process_mpi main.cc)
target_link_libraries(one_device_per_process_mpi PRIVATE
    CUDA::cudart
    MPI::MPI_CXX
    NCCL::nccl
)
